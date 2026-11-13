import io
import os
import re
import zipfile
from pathlib import Path
from typing import Optional
from fastapi.responses import FileResponse
import httpx
from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Carga automáticamente las variables de tu archivo .env (GEMINI_API_KEY y opcionalmente GITHUB_TOKEN)
load_dotenv()

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
        description="Qué parte del documento o código cumple o qué elemento específico falta"
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
        raise HTTPException(status_code=400, detail=f"No se pudo abrir '{nombre_archivo}' como un documento Word válido.")

    bloques = [f"=== DOCUMENTO WORD ENTREGADO: {nombre_archivo} ===\n"]

    for p in doc.paragraphs:
        if p.text.strip():
            bloques.append(p.text)

    for idx, tabla in enumerate(doc.tables, start=1):
        bloques.append(f"\n--- [Tabla {idx} del documento] ---")
        for fila in tabla.rows:
            celdas = [c.text.strip() for c in fila.cells]
            bloques.append(" | ".join(celdas))

    return "\n".join(bloques)


def leer_zip_en_memoria(archivo_bytes: bytes, origen: str = "ARCHIVO .ZIP", max_caracteres: int = 250_000) -> str:
    """Descomprime un .zip en RAM, filtra carpetas innecesarias y concatena el código fuente."""
    bloques = [f"=== CÓDIGO FUENTE DEL PROYECTO ({origen}) ===\n"]
    caracteres_actuales = 0

    try:
        with zipfile.ZipFile(io.BytesIO(archivo_bytes), "r") as zf:
            for info in sorted(zf.infolist(), key=lambda x: x.filename):
                if info.is_dir():
                    continue

                ruta_interna = Path(info.filename)

                # Ignorar carpetas como node_modules, venv, .git, bin, obj
                if any(parte in CARPETAS_IGNORADAS for parte in ruta_interna.parts):
                    continue

                nombre_lower = ruta_interna.name.lower()
                es_dockerfile = nombre_lower == "dockerfile" or nombre_lower == "docker-compose.yml"

                if ruta_interna.suffix.lower() not in EXTENSIONES_CODIGO and not es_dockerfile:
                    continue

                try:
                    contenido_bytes = zf.read(info.filename)
                    texto_codigo = contenido_bytes.decode("utf-8", errors="ignore")
                    fragmento = f"\n--- ARCHIVO: {info.filename} ---\n{texto_codigo}\n"

                    if caracteres_actuales + len(fragmento) > max_caracteres:
                        bloques.append("\n[Aviso: Se alcanzó el límite máximo de caracteres leídos del repositorio]")
                        break

                    bloques.append(fragmento)
                    caracteres_actuales += len(fragmento)
                except Exception:
                    continue
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="El archivo comprimido (.zip) está dañado o no es válido.")

    if len(bloques) == 1:
        raise HTTPException(
            status_code=400,
            detail="No se encontraron archivos de código reconocidos (.py, .cs, .js, .html, .sql, .md, etc.) dentro del .zip."
        )

    return "".join(bloques)


async def preparar_entregable(entregable: UploadFile) -> types.Part | str:
    """Detecta el formato del archivo subido (.pdf, .docx, .zip o código) y lo procesa en RAM."""
    nombre = entregable.filename or "entregable"
    extension = Path(nombre).suffix.lower()
    archivo_bytes = await entregable.read()

    if not archivo_bytes:
        raise HTTPException(status_code=400, detail=f"El archivo '{nombre}' está vacío.")

    if extension == ".pdf":
        return types.Part.from_bytes(data=archivo_bytes, mime_type="application/pdf")

    if extension == ".docx":
        return leer_docx_en_memoria(archivo_bytes, nombre)

    if extension == ".zip":
        return leer_zip_en_memoria(archivo_bytes, origen=f"ZIP: {nombre}")

    if extension in EXTENSIONES_CODIGO or Path(nombre).name.lower() == "dockerfile":
        texto = archivo_bytes.decode("utf-8", errors="ignore")
        return f"=== ARCHIVO DE CÓDIGO ENTREGADO: {nombre} ===\n{texto}"

    raise HTTPException(
        status_code=400,
        detail=f"Formato '{extension}' no soportado. Sube un archivo .pdf, .docx, .zip o código fuente."
    )


async def descargar_repo_github_en_memoria(github_url: str) -> str:
    """Descarga un repositorio de GitHub como .zip en RAM y extrae su código fuente."""
    patron = r"github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/tree/([^/]+))?/?$"
    coincidencia = re.search(patron, github_url.strip())

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

    # Si defines GITHUB_TOKEN en tu .env, también podrás evaluar repositorios privados
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as http_client:
        respuesta = await http_client.get(api_url, headers=headers)

        if respuesta.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró el repositorio '{owner}/{repo}' o es privado (agrega GITHUB_TOKEN en tu .env si es privado)."
            )
        if respuesta.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Error al descargar el repositorio desde GitHub (Código HTTP {respuesta.status_code})."
            )

        return leer_zip_en_memoria(respuesta.content, origen=f"GitHub: {owner}/{repo}")


# ---------------------------------------------------------
# 3. CONFIGURACIÓN DE FASTAPI Y ENDPOINT PRINCIPAL
# ---------------------------------------------------------
app = FastAPI(
    title="API Evaluadora de Rúbricas Universitarias",
    description="Evalúa reportes (.pdf, .docx), proyectos (.zip) o repositorios de GitHub contra el PDF de la rúbrica oficial.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/evaluar", response_model=DictamenRubrica)
async def evaluar_entrega_endpoint(
    rubrica_pdf: UploadFile = File(..., description="PDF oficial de la rúbrica del profesor"),
    entregable: Optional[UploadFile] = File(
        None,
        description="(Opcional) Sube tu reporte o código en formato .pdf, .docx, .zip o script suelto"
    ),
    github_url: Optional[str] = Form(
        None,
        description="(Opcional) Pega la URL de tu repositorio de GitHub (ej. https://github.com/usuario/repo)"
    ),
):
    # 1. Verificar API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Falta configurar la variable de entorno GEMINI_API_KEY en el archivo .env"
        )

    # 2. Validar que la rúbrica sea un archivo PDF
    if not (rubrica_pdf.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="El archivo en 'rubrica_pdf' debe tener extensión .pdf")

    # 3. Comprobar que el usuario haya enviado al menos un archivo entregable o un link de GitHub
    tiene_archivo = entregable is not None and bool(entregable.filename)
    tiene_url = github_url is not None and bool(github_url.strip())

    if not tiene_archivo and not tiene_url:
        raise HTTPException(
            status_code=400,
            detail="Debes proporcionar al menos un archivo en 'entregable' o una URL en 'github_url'."
        )

    # 4. Construir la lista de elementos que revisará Gemini
    bytes_rubrica = await rubrica_pdf.read()
    elementos_para_gemini: list = [
        types.Part.from_bytes(data=bytes_rubrica, mime_type="application/pdf")
    ]

    if tiene_archivo and entregable is not None:
        contenido_archivo = await preparar_entregable(entregable)
        elementos_para_gemini.append(contenido_archivo)

    if tiene_url and github_url is not None:
        contenido_repo = await descargar_repo_github_en_memoria(github_url)
        elementos_para_gemini.append(contenido_repo)

    instrucciones = (
        "Actúa como un profesor universitario estricto de Ingeniería de Software. "
        "El primer documento adjunto es la RÚBRICA OFICIAL DE EVALUACIÓN en PDF. "
        "Los elementos siguientes constituyen la ENTREGA DEL ESTUDIANTE (documento, código o ambos).\n\n"
        "Tu tarea:\n"
        "1. Identifica cada criterio y su puntaje en la rúbrica oficial.\n"
        "2. Evalúa rigurosamente si la entrega cumple con la arquitectura, endpoints, pruebas, "
        "reglas de negocio, profundidad teórica o formato APA 7 solicitados.\n"
        "3. Señala exactamente qué archivos o secciones cumplen y qué debe corregir el alumno "
        "para alcanzar el 100% de la calificación."
    )
    elementos_para_gemini.append(instrucciones)

    # 5. Llamada asíncrona a Gemini API
    try:
        client = genai.Client(api_key=api_key)
        response = await client.aio.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-4-hydrogen"),
            contents=elementos_para_gemini,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DictamenRubrica,
                temperature=0.1,
            ),
        )
        return DictamenRubrica.model_validate_json(response.text)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error al procesar la evaluación con Gemini API: {str(e)}"
        )

@app.get("/", include_in_schema=False)
async def servir_frontend():
    return FileResponse("index.html")