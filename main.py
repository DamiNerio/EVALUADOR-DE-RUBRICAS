import io
import os
import re
import zipfile
from pathlib import Path
from typing import Optional

import httpx
from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Carga variables del archivo .env local (en Render se leen desde Environment Variables)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Extensiones de código permitidas dentro de un .zip, repositorio de GitHub o archivo suelto
EXTENSIONES_CODIGO = {
    ".py", ".cs", ".js", ".ts", ".jsx", ".tsx",
    ".html", ".css", ".sql", ".yaml", ".yml",
    ".json", ".md", ".c", ".cpp", ".h", ".gd", ".dockerfile"
}

# Carpetas que se ignoran automáticamente al leer un .zip o repositorio
CARPETAS_IGNORADAS = {
    ".git", "__pycache__", "node_modules", "venv",
    ".venv", "dist", "build", "bin", "obj", ".vs", ".vscode", ".pytest_cache"
}

# Archivos pesados de dependencias que no aportan a la evaluación y saturan la memoria
ARCHIVOS_IGNORADOS = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "cargo.lock"
}


# ---------------------------------------------------------
# 1. ESQUEMAS PYDANTIC DE SALIDA ESTRUCTURADA
# ---------------------------------------------------------
class EvaluacionCriterio(BaseModel):
    criterio: str = Field(
        description="Nombre del criterio o requerimiento según la rúbrica del profesor"
    )
    puntos_maximos: float = Field(
        description="Puntaje máximo asignado a este criterio en la rúbrica"
    )
    puntos_estimados: float = Field(
        description="Puntaje obtenido según lo que sí cumple el entregable"
    )
    estado: str = Field(
        description="Cumple totalmente | Cumple parcialmente | No cumple"
    )
    evidencia_o_fallo: str = Field(
        description="Qué parte de los documentos o código cumple o qué elemento específico falta"
    )
    accion_correctiva: str = Field(
        description="Instrucción concreta de qué agregar o corregir para obtener el 100% de los puntos"
    )


class DictamenRubrica(BaseModel):
    calificacion_total_estimada: float = Field(
        description="Suma total de los puntos estimados"
    )
    calificacion_maxima_posible: float = Field(
        description="Suma total de los puntos máximos de la rúbrica (ej. 100)"
    )
    resumen_general: str = Field(
        description="Diagnóstico ejecutivo de la entrega en 2 o 3 oraciones"
    )
    revision_formato_y_apa: list[str] = Field(
        description="Observaciones sobre citas APA 7, bibliografía, arquitectura de carpetas o estándares de código"
    )
    desglose_criterios: list[EvaluacionCriterio] = Field(
        description="Evaluación detallada fila por fila de la rúbrica"
    )


# ---------------------------------------------------------
# 2. FUNCIONES PROCESADORAS EN MEMORIA RAM
# ---------------------------------------------------------
def leer_docx_en_memoria(archivo_bytes: bytes, nombre_archivo: str) -> str:
    """Extrae párrafos y tablas de un archivo Word (.docx) directamente en RAM."""
    try:
        doc = Document(io.BytesIO(archivo_bytes))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo abrir '{nombre_archivo}' como un documento Word (.docx) válido."
        )

    bloques = [f"=== DOCUMENTO WORD ENTREGADO: {nombre_archivo} ===\n"]

    for p in doc.paragraphs:
        if p.text.strip():
            bloques.append(p.text)

    for idx, tabla in enumerate(doc.tables, start=1):
        bloques.append(f"\n--- [Tabla {idx} de {nombre_archivo}] ---")
        for fila in tabla.rows:
            celdas = [c.text.strip() for c in fila.cells]
            bloques.append(" | ".join(celdas))

    return "\n".join(bloques)


def leer_zip_en_memoria(
    archivo_bytes: bytes,
    origen: str = "ARCHIVO .ZIP",
    max_caracteres: int = 90_000,
) -> str:
    """Descomprime un .zip en RAM, filtra carpetas innecesarias y concatena el código fuente."""
    bloques = [f"=== CÓDIGO FUENTE DEL PROYECTO ({origen}) ===\n"]
    caracteres_actuales = 0

    try:
        with zipfile.ZipFile(io.BytesIO(archivo_bytes), "r") as zf:
            rutas_validas = []
            for info in sorted(zf.infolist(), key=lambda x: x.filename):
                if info.is_dir():
                    continue
                ruta_interna = Path(info.filename)
                if any(parte in CARPETAS_IGNORADAS for parte in ruta_interna.parts):
                    continue
                rutas_validas.append(info)

            if rutas_validas:
                arbol = "\n".join(f"- {r.filename}" for r in rutas_validas[:120])
                bloques.append(f"ESTRUCTURA DE ARCHIVOS DETECTADA:\n{arbol}\n")

            for info in rutas_validas:
                ruta_interna = Path(info.filename)
                nombre_lower = ruta_interna.name.lower()

                if nombre_lower in ARCHIVOS_IGNORADOS:
                    continue

                es_dockerfile = nombre_lower in {"dockerfile", "docker-compose.yml", "docker-compose.yaml"}

                if ruta_interna.suffix.lower() not in EXTENSIONES_CODIGO and not es_dockerfile:
                    continue

                try:
                    contenido_bytes = zf.read(info.filename)
                    texto_codigo = contenido_bytes.decode("utf-8", errors="ignore")
                    fragmento = f"\n--- ARCHIVO: {info.filename} ---\n{texto_codigo}\n"

                    if caracteres_actuales + len(fragmento) > max_caracteres:
                        bloques.append("\n[Aviso: Se alcanzó el límite de lectura optimizado para memoria]")
                        break

                    bloques.append(fragmento)
                    caracteres_actuales += len(fragmento)
                except Exception:
                    continue
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail=f"El archivo comprimido ({origen}) está dañado o no es válido.")

    if len(bloques) <= 1:
        raise HTTPException(
            status_code=400,
            detail=f"No se encontraron archivos de código reconocidos dentro de {origen}."
        )

    return "".join(bloques)


async def preparar_entregable(entregable: UploadFile, indice: int) -> list:
    """Detecta el formato de cada archivo subido y devuelve los elementos para Gemini."""
    nombre = entregable.filename or f"entregable_{indice}"
    extension = Path(nombre).suffix.lower()
    archivo_bytes = await entregable.read()

    if not archivo_bytes:
        raise HTTPException(status_code=400, detail=f"El archivo '{nombre}' está vacío.")

    # Si es PDF, enviamos una etiqueta identificadora seguida del documento PDF en bytes
    if extension == ".pdf":
        return [
            f"=== ARCHIVO ENTREGABLE #{indice} DEL ESTUDIANTE (PDF): {nombre} ===",
            types.Part.from_bytes(data=archivo_bytes, mime_type="application/pdf"),
        ]

    if extension == ".docx":
        return [leer_docx_en_memoria(archivo_bytes, nombre)]

    if extension == ".zip":
        return [leer_zip_en_memoria(archivo_bytes, origen=f"ZIP #{indice}: {nombre}")]

    if extension in EXTENSIONES_CODIGO or Path(nombre).name.lower() == "dockerfile":
        texto = archivo_bytes.decode("utf-8", errors="ignore")
        return [f"=== ARCHIVO DE CÓDIGO ENTREGADO #{indice}: {nombre} ===\n{texto}"]

    raise HTTPException(
        status_code=400,
        detail=f"Formato '{extension}' en '{nombre}' no soportado. Sube archivos .pdf, .docx, .zip o código fuente."
    )


async def descargar_repo_github_en_memoria(github_url: str) -> str:
    """Descarga un repositorio de GitHub como .zip en RAM y extrae su código fuente."""
    url_limpia = github_url.strip().split("?")[0].split("#")[0]
    patron = r"github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/tree/([^/]+))?/?$"
    coincidencia = re.search(patron, url_limpia)

    if not coincidencia:
        raise HTTPException(
            status_code=400,
            detail="URL de GitHub no válida. Usa el formato: https://github.com/usuario/repositorio"
        )

    owner, repo, rama = coincidencia.groups()

    api_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
    if rama:
        api_url += f"/{rama}"

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "FastAPI-Rubric-Evaluator"
    }

    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token.strip()}"

    async with httpx.AsyncClient(follow_redirects=True, timeout=45.0) as http_client:
        respuesta = await http_client.get(api_url, headers=headers)

        if respuesta.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No se pudo acceder al repositorio '{owner}/{repo}'. "
                    "Verifica que la URL esté bien escrita y que el repositorio sea Público "
                    "(o agrega GITHUB_TOKEN en las variables de entorno si es Privado)."
                )
            )
        if respuesta.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"GitHub devolvió un error al descargar el repositorio (Código HTTP {respuesta.status_code})."
            )

        return leer_zip_en_memoria(respuesta.content, origen=f"GitHub: {owner}/{repo}")


# ---------------------------------------------------------
# 3. CONFIGURACIÓN DE FASTAPI Y MANEJO DE ERRORES JSON
# ---------------------------------------------------------
app = FastAPI(
    title="API Evaluadora de Rúbricas Universitarias (Multi-Archivo)",
    description="Evalúa múltiples reportes (.pdf, .docx), proyectos (.zip) y repositorios de GitHub contra el PDF de la rúbrica oficial.",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def manejador_excepciones_global(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Error interno en el servidor: {str(exc)}"},
    )


@app.post("/api/v1/evaluar", response_model=DictamenRubrica)
async def evaluar_entrega_endpoint(
    rubrica_pdf: UploadFile = File(..., description="PDF oficial de la rúbrica del profesor"),
    entregables: Optional[list[UploadFile]] = File(
        None,
        description="(Opcional) Sube uno o varios archivos (.pdf, .docx, .zip o scripts)"
    ),
    github_url: Optional[str] = Form(
        None,
        description="(Opcional) Pega la URL de tu repositorio de GitHub"
    ),
):
    try:
        # 1. Verificar API Key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="Falta configurar la variable de entorno GEMINI_API_KEY en Render o en tu archivo .env"
            )

        # 2. Validar que la rúbrica sea un archivo PDF
        if not (rubrica_pdf.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="El archivo en 'rubrica_pdf' debe tener extensión .pdf")

        # 3. Filtrar archivos válidos subidos en la lista
        archivos_validos = [f for f in (entregables or []) if f is not None and bool(f.filename)]
        tiene_archivos = len(archivos_validos) > 0
        tiene_url = github_url is not None and bool(github_url.strip())

        if not tiene_archivos and not tiene_url:
            raise HTTPException(
                status_code=400,
                detail="Debes subir al menos un archivo entregable o pegar una URL de GitHub."
            )

        # 4. Construir la lista de elementos que revisará Gemini
        bytes_rubrica = await rubrica_pdf.read()
        elementos_para_gemini: list = [
            "=== DOCUMENTO BASE: RÚBRICA OFICIAL DE EVALUACIÓN (PDF) ===",
            types.Part.from_bytes(data=bytes_rubrica, mime_type="application/pdf"),
        ]

        # Agregar todos los archivos subidos por el estudiante
        for idx, archivo in enumerate(archivos_validos, start=1):
            partes_archivo = await preparar_entregable(archivo, indice=idx)
            elementos_para_gemini.extend(partes_archivo)

        # Agregar el repositorio de GitHub si también se proporcionó
        if tiene_url and github_url is not None:
            contenido_repo = await descargar_repo_github_en_memoria(github_url)
            elementos_para_gemini.append(contenido_repo)

        instrucciones = (
            "Actúa como un profesor universitario estricto de Ingeniería de Software. "
            "El primer documento adjunto es la RÚBRICA OFICIAL DE EVALUACIÓN en PDF. "
            "Todos los archivos siguientes (uno o varios documentos PDF, Word, ZIP o repositorio de GitHub) "
            "forman en conjunto la ENTREGA COMPLETA DEL ESTUDIANTE para la misma actividad.\n\n"
            "Tu tarea:\n"
            "1. Identifica cada criterio y su puntaje en la rúbrica oficial.\n"
            "2. Evalúa de forma integral todos los archivos entregados (si un requerimiento está en el "
            "primer archivo y otro requerimiento está en el segundo archivo o repositorio, tómalos en cuenta juntos).\n"
            "3. Señala exactamente qué archivos o secciones cumplen y qué debe corregir el alumno "
            "para alcanzar el 100% de la calificación."
        )
        elementos_para_gemini.append(instrucciones)

        # 5. Llamada asíncrona a Gemini API con respaldo automático de modelo
        client = genai.Client(api_key=api_key.strip())
        modelo_preferido = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

        config_generacion = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DictamenRubrica,
            temperature=0.1,
        )

        try:
            response = await client.aio.models.generate_content(
                model=modelo_preferido,
                contents=elementos_para_gemini,
                config=config_generacion,
            )
        except Exception as err_modelo:
            if modelo_preferido != "gemini-2.5-flash":
                response = await client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=elementos_para_gemini,
                    config=config_generacion,
                )
            else:
                raise err_modelo

        return DictamenRubrica.model_validate_json(response.text)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error al procesar la evaluación con Gemini API: {str(e)}"
        )


@app.get("/", include_in_schema=False)
async def servir_frontend():
    ruta_html = BASE_DIR / "index.html"
    if not ruta_html.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": "No se encontró el archivo index.html en la raíz del proyecto."}
        )
    return FileResponse(ruta_html)