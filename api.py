#api.py
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
from conexion import conexion
from datetime import datetime
from openpyxl import Workbook
import io
import pandas as pd
import os
from pydantic import BaseModel

# Inicialización de FastAPI
app = FastAPI(title="BoxIA API Local")

# Modelo para el body de la petición
class Pregunta(BaseModel):
    pregunta: str

# Inicialización de modelos y recursos (solo una vez al arrancar)
llm = Ollama(model="llama3.1")  # modelo Ollama local
embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    embedding_function=embed_model,
    persist_directory="chroma_db_dir",
    collection_name="stanford_report_data"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})


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

    # Mostrar el contexto en consola (debug)
    print("=== CONTEXTO ===")
    print(contexto)

    # Formatear el prompt con el contexto
    frase_fija = "Responde solo si la pregunta tiene relacion con el context proporcionado, de lo contrario comienza tu respuesta con 'Lo siento'.\n\n"
    pregunta_modificada = p.pregunta + frase_fija
    pregunta_formateada = prompt.format(context=contexto, question=pregunta_modificada)


    # Ejecutar la IA directamente
    respuesta = llm.invoke(pregunta_formateada).strip()
    print(respuesta)
    # Validar si la respuesta comienza con "Lo siento"
    if respuesta.lower().startswith(("lo siento", "no")):
        return {"respuesta": "Lo siento, no tengo información suficiente para responder."}

    return {"respuesta": respuesta}




# Endpoint para cargar documentos PDF
@app.post("/cargar-documento-pdf")
async def cargar_documento(archivo: UploadFile = File(...)):
    if not archivo.filename.endswith(".pdf"):
        return {"error": "Solo se permiten archivos PDF."}

    ruta_temporal = f"docs/{archivo.filename}"
    with open(ruta_temporal, "wb") as f:
        f.write(await archivo.read())

    loader = PyPDFLoader(ruta_temporal)
    documentos = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    documentos_divididos = splitter.split_documents(documentos)

    vectorstore.add_documents(documentos_divididos)
    vectorstore.persist()

    return {"mensaje": f"{archivo.filename} cargado exitosamente."}

# Endpoint recibir reporte de preguntas sin respuesta
class ReportePregunta(BaseModel):
    pregunta: str
    respuesta: str

@app.post("/reportar-pregunta")
def reportar_pregunta(data: ReportePregunta):
    try:
        pregunta_normalizada = data.pregunta.strip().lower()
        cursor = conexion.cursor()

        # 1. Verifica si la pregunta ya existe (sin distinguir mayúsculas/minúsculas)
        cursor.execute("""
            SELECT id, checked 
            FROM reported_questions 
            WHERE LOWER(question) = %s
        """, (pregunta_normalizada,))
        resultado = cursor.fetchone()

        if resultado:
            id_pregunta, checked = resultado

            if not checked:
                cursor.close()
                return {"mensaje": "Esta pregunta ya ha sido reportada y está en revisión."}
            else:
                cursor.execute("""
                    UPDATE reported_questions
                    SET answer = %s, checked = FALSE, reported_date = %s
                    WHERE id = %s
                """, (data.respuesta, datetime.now(), id_pregunta))
                conexion.commit()
                cursor.close()
                return {"mensaje": "Pregunta actualizada y marcada nuevamente como pendiente de revisión."}
        else:
            cursor.execute("""
                INSERT INTO reported_questions (question, answer, reported_date)
                VALUES (%s, %s, %s)
            """, (pregunta_normalizada, data.respuesta, datetime.now()))
            conexion.commit()
            cursor.close()
            return {"mensaje": "Pregunta reportada exitosamente."}
    except Exception as e:
        print("Error al reportar pregunta:", e)
        raise HTTPException(status_code=500, detail="Error al guardar la pregunta")



# Endpoint para listar preguntas reportadas
@app.get("/preguntas-reportadas")
def listar_preguntas_reportadas(revisadas: bool = False):
    """
    Devuelve en JSON las preguntas reportadas.
    Parámetro opcional:
      - revisadas: False (por defecto) → solo checked = FALSE
                   True  → todas (checked TRUE/FALSE)
    """
    try:
        cursor = conexion.cursor()
        if revisadas:
            cursor.execute("""
                SELECT id, question, answer, reported_date, checked
                FROM reported_questions
                ORDER BY reported_date DESC
            """)
        else:
            cursor.execute("""
                SELECT id, question, answer, reported_date, checked
                FROM reported_questions
                WHERE checked = FALSE
                ORDER BY reported_date DESC
            """)
        rows = cursor.fetchall()
        cursor.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al consultar la base de datos")

    # Mapeamos a lista de dict
    result = [
        {
            "id":      r[0],
            "pregunta":r[1],
            "respuesta":r[2],
            "fecha":   r[3].isoformat(),
            "checked": r[4],
        }
        for r in rows
    ]


    return result


class MarcarRevisada(BaseModel):
    id: int

@app.post("/marcar-revisado")
def marcar_revisada(data: MarcarRevisada):
    try:
        cursor = conexion.cursor()
        cursor.execute("""
            UPDATE reported_questions
            SET checked = TRUE
            WHERE id = %s
        """, (data.id,))
        conexion.commit()
        cursor.close()
        return {"mensaje": f"Pregunta con ID {data.id} marcada como revisada."}
    except Exception as e:
        print("Error al marcar como revisada:", e)
        raise HTTPException(status_code=500, detail="Error al actualizar el estado de la pregunta.")


#Endpoint para exportar preguntas reportadas a Excel
@app.get("/exportar-preguntas")
def exportar_preguntas_excel(revisadas: bool = False):
    try:
        cursor = conexion.cursor()
        if revisadas:
            cursor.execute("""
                SELECT id, question
                FROM reported_questions
                ORDER BY reported_date DESC
            """)
        else:
            cursor.execute("""
                SELECT id, question
                FROM reported_questions
                WHERE checked = FALSE
                ORDER BY reported_date DESC
            """)
        rows = cursor.fetchall()
        cursor.close()

        if not rows:
            raise HTTPException(status_code=404, detail="No hay preguntas reportadas.")

        wb = Workbook()
        ws = wb.active
        ws.title = "Respuestas"

        # Encabezados
        ws.append(["ID", "Pregunta"])

        # Datos
        for r in rows:
            ws.append([r[0], r[1]])  # Se debe de responde en la misma celda de la pregunta

        stream = io.BytesIO()
        wb.save(stream)
        stream.seek(0)

        return StreamingResponse(
            stream,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=preguntas_respuestas.xlsx"}
        )

    except Exception as e:
        print("Error exportando preguntas:", e)
        raise HTTPException(status_code=500, detail="Error al generar el archivo Excel.")
    
@app.post("/subir-respuestas-excel")
async def subir_respuestas_excel(file: UploadFile = File(...)):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Archivo no válido. Solo se aceptan archivos Excel.")

    try:
        content = await file.read()
        df = pd.read_excel(io.BytesIO(content))

        # Validar columnas
        expected_cols = {"ID", "Pregunta"}
        if not expected_cols.issubset(df.columns):
            raise HTTPException(status_code=400, detail=f"El archivo Excel debe contener las columnas: {expected_cols}")

        cursor = conexion.cursor()

        for _, row in df.iterrows():
            id_pregunta = row["ID"]
            pregunta_completa = row["Pregunta"]  # Esta ya incluye la respuesta en la misma celda

            # 1. Marcar como revisada en PostgreSQL
            cursor.execute("""
                UPDATE reported_questions
                SET checked = TRUE
                WHERE id = %s
            """, (id_pregunta,))

            # 2. Agregar pregunta + respuesta como nuevo documento en ChromaDB (sin metadata)
            from langchain.schema import Document

            nuevo_doc = Document(
                page_content=pregunta_completa  # Solo el texto
            )
            vectorstore.add_documents([nuevo_doc])


        conexion.commit()
        cursor.close()

        # Persistir cambios en vectorstore
        vectorstore.persist()

        return {"mensaje": "Preguntas marcadas como revisadas y vectorstore actualizado."}

    except Exception as e:
        print(f"Error procesando archivo Excel: {e}")
        raise HTTPException(status_code=500, detail="Error al procesar el archivo Excel.")
