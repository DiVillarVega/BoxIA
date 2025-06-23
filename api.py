from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import psycopg2
import time

# --- CREA SIEMPRE LA TABLA DE REPORTES AL INICIAR ---
def ensure_table_exists():
    for _ in range(10):
        try:
            conn = psycopg2.connect(
                dbname="boxia_db",
                user="boxia_user",
                password="boxia",
                host="postgres"
            )
            break
        except psycopg2.OperationalError:
            time.sleep(2)
    else:
        raise Exception("No se pudo conectar a la base de datos después de varios intentos.")

    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reported_questions (
        id SERIAL PRIMARY KEY,
        question VARCHAR,
        answer VARCHAR,
        expert_answer TEXT,
        reported_date TIMESTAMP,
        checked VARCHAR(20)
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

ensure_table_exists()
# --- FIN BLOQUE AUTO TABLA ---

from conexion import conexion
from datetime import datetime
from openpyxl import Workbook
import io
import re
import pandas as pd
import os
# Inicialización de FastAPI
app = FastAPI(title="BoxIA API Local")

# Modelo para el body de la petición
class Pregunta(BaseModel):
    pregunta: str

# Inicialización de modelos y recursos (solo una vez al arrancar)

llm = Ollama(
    model="llama3.1",
    base_url="http://ollama:11434"
)

embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    embedding_function=embed_model,
    persist_directory="chroma_db_dir",
    collection_name="stanford_report_data"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})


# Prompt personalizado
prompt_template = """
Eres un asistente experto que responde solicitudes de información únicamente en base al contenido brindado por el context. Tu tarea es entregar respuestas claras, directas y justificadas, siempre fundamentadas en el context entregado.

### Reglas:
- No uses conocimientos externos al contenido entregado.
- No uses entregues informacion externa al contexto.
- No inventes datos.
- Si la pregunta no tiene relacion alguna con el contexto, responde: "Lo siento, no tengo información suficiente para responder."
- Si la pregunta es literal, responde directamente con el texto del contexto.
- Si la información solicitada no está presente o no puede deducirse lógicamente del contexto, responde: "Lo siento, no tengo información suficiente para responder."
- Puedes razonar o resumir ideas si están explícitamente respaldadas por el contenido textual.

### Ejemplos

**Solicitud literal:**
- Contexto: "El sol es una estrella que emite luz y calor."
- Solicitud: "Indica qué es el sol."
- Respuesta: "El sol es una estrella que emite luz y calor."

**Solicitud inferencial:**
- Contexto: "El sol sale cada mañana y su luz despierta a los animales del bosque."
- Solicitud: "Explica el efecto del sol sobre los animales del bosque."
- Respuesta: "El sol hace que los animales del bosque se despierten cada mañana."

**Sin información suficiente (aunque haya contexto):**
- Contexto: "La Constitución establece que todos los ciudadanos tienen derecho a voto."
- Solicitud: "¿Quién descubrió América?"
- Respuesta: "Lo siento, no tengo información suficiente para responder."

---

### Contexto:
{context}

### Solicitud:
{question}

### Respuesta:
"""







prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)



# Endpoint para preguntar
@app.post("/preguntar")
def preguntar(p: Pregunta):
    # Obtener documentos relevantes
    documentos_relacionados = retriever.get_relevant_documents(p.pregunta)

    # Unir el contenido de los documentos en un solo string
    contexto = "\n\n".join([doc.page_content for doc in documentos_relacionados])


    # Formatear el prompt con el contexto
    frase_fija = ("Responde solo si la pregunta/solicitud tiene relacion con el context proporcionado, de lo contrario responde exactamente con: 'Lo siento, no tengo información suficiente para responder.'")

    pregunta_modificada = frase_fija + p.pregunta
    pregunta_formateada = prompt.format(context=contexto, question=pregunta_modificada)


    # Ejecutar la IA directamente
    respuesta = llm.invoke(pregunta_formateada).strip()
    # Validar si la respuesta comienza con "Lo siento"
    if respuesta.lower().startswith(("lo siento", "no")):
        return {"respuesta": "Lo siento, no tengo información suficiente para responder."}

    return {"respuesta": respuesta}




# Endpoint para cargar documentos PDF
# Función de limpieza del texto
def limpiar_texto(texto: str) -> str:
    # Quitar múltiples saltos de línea
    texto = re.sub(r'\n+', '\n', texto)
    # Eliminar encabezados o pies de página típicos
    texto = re.sub(r'Página \d+|\d+ de \d+', '', texto, flags=re.IGNORECASE)
    # Unir líneas fragmentadas que no terminan en punto, signo de interrogación o exclamación
    texto = re.sub(r'(?<![.?!])\n', ' ', texto)
    # Quitar espacios redundantes
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


@app.post("/cargar-documento-pdf")
async def cargar_documento(archivo: UploadFile = File(...)):
    if not archivo.filename.endswith(".pdf"):
        return {"error": "Solo se permiten archivos PDF."}

    ruta_temporal = f"docs/{archivo.filename}"
    with open(ruta_temporal, "wb") as f:
        f.write(await archivo.read())

    loader = PyPDFLoader(ruta_temporal)
    documentos = loader.load()

    # Limpieza del texto de cada documento
    documentos_limpios = []
    for doc in documentos:
        texto_limpio = limpiar_texto(doc.page_content)
        doc.page_content = texto_limpio
        documentos_limpios.append(doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=500)
    documentos_divididos = splitter.split_documents(documentos_limpios)

    vectorstore.add_documents(documentos_divididos)
    vectorstore.persist()

    return {"mensaje": f"{archivo.filename} cargado exitosamente con limpieza aplicada."}

#  1. Endpoint para reportar pregunta
class ReportePregunta(BaseModel):
    pregunta: str
    respuesta: str

@app.post("/reportar-pregunta")
def reportar_pregunta(data: ReportePregunta):
    try:
        pregunta_normalizada = data.pregunta.strip().lower()
        cursor = conexion.cursor()

        cursor.execute("""
            SELECT id, checked 
            FROM reported_questions 
            WHERE LOWER(question) = %s
        """, (pregunta_normalizada,))
        resultado = cursor.fetchone()

        if resultado:
            id_pregunta, estado = resultado
            if estado == 'reportada':
                cursor.close()
                return {"mensaje": "Esta pregunta ya ha sido reportada y está en revisión."}
            else:
                cursor.execute("""
                    UPDATE reported_questions
                    SET answer = %s, checked = 'reportada', reported_date = %s
                    WHERE id = %s
                """, (data.respuesta, datetime.now(), id_pregunta))
                conexion.commit()
                cursor.close()
                return {"mensaje": "Pregunta actualizada y marcada nuevamente como pendiente de revisión."}
        else:
            cursor.execute("""
                INSERT INTO reported_questions (question, answer, reported_date, checked)
                VALUES (%s, %s, %s, 'reportada')
            """, (pregunta_normalizada, data.respuesta, datetime.now()))
            conexion.commit()
            cursor.close()
            return {"mensaje": "Pregunta reportada exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al guardar la pregunta")


#  2. Endpoint para listar preguntas reportadas
@app.get("/preguntas-reportadas")
def listar_preguntas_reportadas(estado: str = "reportada"):
    """
    Devuelve una lista de preguntas reportadas según el estado:
    - "reportada"
    - "revisada"
    - "eliminada"
    - "todas" → sin filtro
    """
    estado = estado.lower()
    estados_validos = {"reportada", "revisada", "eliminada", "todas"}

    if estado not in estados_validos:
        raise HTTPException(status_code=400, detail=f"Estado inválido. Debe ser uno de: {estados_validos}")

    try:
        cursor = conexion.cursor()

        
        cursor.execute("""
            SELECT id, question, answer, reported_date, checked, expert_answer
            FROM reported_questions
            WHERE checked = %s
            ORDER BY reported_date DESC
        """, (estado,))

        rows = cursor.fetchall()
        cursor.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al consultar la base de datos")

    # Mapear a lista de diccionarios
    result = [
        {
            "id": r[0],
            "pregunta": r[1],
            "respuesta": r[2],
            "fecha": r[3].isoformat(),
            "estado": r[4],
            "respuesta_experto": r[5]
        }
        for r in rows
    ]

    return result

#  3. Endpoint para marcar revisada
class MarcarRevisada(BaseModel):
    id: int

@app.post("/marcar-revisado")
def marcar_revisada(data: MarcarRevisada):
    try:
        cursor = conexion.cursor()
        cursor.execute("""
            UPDATE reported_questions
            SET checked = 'revisada'
            WHERE id = %s
        """, (data.id,))
        conexion.commit()
        cursor.close()
        return {"mensaje": f"Pregunta con ID {data.id} marcada como revisada."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al actualizar el estado de la pregunta.")


@app.get("/exportar-preguntas")
def exportar_preguntas_excel(estado: str = "reportada"):
    """
    Exporta preguntas reportadas según su estado en un archivo Excel:
    - "reportada"
    - "revisada"
    - "eliminada"
    """
    estado = estado.lower()
    estados_validos = {"reportada", "revisada", "eliminada"}

    if estado not in estados_validos:
        raise HTTPException(status_code=400, detail=f"Estado inválido. Debe ser uno de: {estados_validos}")

    try:
        cursor = conexion.cursor()
        cursor.execute("""
            SELECT id, question, expert_answer
            FROM reported_questions
            WHERE checked = %s
            ORDER BY reported_date DESC
        """, (estado,))
        rows = cursor.fetchall()
        cursor.close()

        if not rows:
            raise HTTPException(status_code=404, detail="No hay preguntas reportadas con ese estado.")

        # Crear Excel
        wb = Workbook()
        ws = wb.active
        ws.title = f"Preguntas {estado.capitalize()}"
        ws.append(["ID", "Pregunta", "Respuesta Experto"])

        for r in rows:
            ws.append([r[0], r[1], r[2] or ""])

        stream = io.BytesIO()
        wb.save(stream)
        stream.seek(0)

        return StreamingResponse(
            stream,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=preguntas_{estado}.xlsx"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al generar el archivo Excel.")


#  5. Subir respuestas del experto desde Excel
@app.post("/subir-respuestas-excel")
async def subir_respuestas_excel(file: UploadFile = File(...)):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Archivo no válido. Solo se aceptan archivos Excel.")

    try:
        content = await file.read()
        df = pd.read_excel(io.BytesIO(content))

        expected_cols = {"ID", "Pregunta", "Respuesta Experto"}
        if not expected_cols.issubset(df.columns):
            raise HTTPException(status_code=400, detail=f"El archivo debe contener las columnas: {expected_cols}")

        cursor = conexion.cursor()
        docs = []
        ids = []

        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=600)

        for _, row in df.iterrows():
            id_pregunta = str(row["ID"])
            pregunta = row["Pregunta"]
            respuesta_experto = row["Respuesta Experto"]

            if pd.notna(respuesta_experto):
                # Actualizar en la base de datos
                cursor.execute("""
                    UPDATE reported_questions
                    SET expert_answer = %s, checked = 'revisada'
                    WHERE id = %s
                """, (respuesta_experto, id_pregunta))

                texto = f"{pregunta} {respuesta_experto}"
                chunks = splitter.create_documents([texto])

                for i, chunk in enumerate(chunks):
                    for j in range(12):
                        doc = Document(
                            page_content=chunk.page_content,
                            metadata={"source": "experto", "id": f"{id_pregunta}-{i}-{j}"}
                        )
                        docs.append(doc)
                        ids.append(f"{id_pregunta}-{i}-{j}")

        # Agregar documentos al vectorstore
        if docs:
            vectorstore.add_documents(docs, ids=ids)
            vectorstore.persist()

        conexion.commit()
        cursor.close()

        return {"mensaje": "Respuestas del experto registradas, divididas y repetidas 5 veces por chunk."}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error al procesar el archivo Excel.")


#  6. Eliminar pregunta del postgreSQL
class EliminarDefinitivo(BaseModel):
    id: int

@app.post("/eliminar-pregunta")
def eliminar_definitivo(data: EliminarDefinitivo):
    try:
        cursor = conexion.cursor()
        cursor.execute("DELETE FROM reported_questions WHERE id = %s", (data.id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Pregunta no encontrada.")
        conexion.commit()
        cursor.close()
        return {"mensaje": f"Pregunta con ID {data.id} eliminada de la base de datos."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al eliminar la pregunta desde PostgreSQL.")
    
#  7. Endpoint para eliminar pregunta del vectorstore
class EliminarDeChroma(BaseModel):
    id: int

@app.post("/eliminar-de-chroma")
def eliminar_de_chroma(data: EliminarDeChroma):
    try:
        id_base = str(data.id)

        # Recuperar todos los documentos y filtrar por los que comienzan con el ID base
        all_docs = vectorstore.get()
        ids_a_eliminar = [doc_id for doc_id in all_docs['ids'] if doc_id.startswith(f"{id_base}-")]

        if not ids_a_eliminar:
            raise HTTPException(status_code=404, detail="No se encontraron chunks en Chroma con ese ID.")

        vectorstore.delete(ids=ids_a_eliminar)
        vectorstore.persist()

        # Actualizar estado en la base de datos
        cursor = conexion.cursor()
        cursor.execute("""
            UPDATE reported_questions
            SET checked = 'eliminada'
            WHERE id = %s
        """, (data.id,))
        conexion.commit()
        cursor.close()

        return {"mensaje": f"Eliminados {len(ids_a_eliminar)} chunks de Chroma para ID {id_base}."}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al eliminar de Chroma: {str(e)}")


#  8. Endpoint para reactivar pregunta eliminada de la vectorstore
class ReactivarPregunta(BaseModel):
    id: int

@app.post("/reactivar-pregunta")
def reactivar_pregunta(data: ReactivarPregunta):
    try:
        cursor = conexion.cursor()
        cursor.execute("""
            UPDATE reported_questions
            SET checked = 'reportada'
            WHERE id = %s
        """, (data.id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Pregunta no encontrada.")
        conexion.commit()
        cursor.close()
        return {"mensaje": f"Pregunta con ID {data.id} reactivada como 'reportada'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al reactivar la pregunta.")
    
#  9. Endpoint para marcar pregunta como revisada
@app.post("/marcar-revisado")
def marcar_revisada(data: MarcarRevisada):
    try:
        cursor = conexion.cursor()
        cursor.execute("""
            UPDATE reported_questions
            SET checked = 'revisada'
            WHERE id = %s
        """, (data.id,))
        conexion.commit()
        cursor.close()
        return {"mensaje": f"Pregunta con ID {data.id} marcada como revisada."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al actualizar el estado de la pregunta.")



