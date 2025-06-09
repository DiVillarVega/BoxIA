from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import date
from conexion import conexion
from fastapi import HTTPException
import os

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
        cursor = conexion.cursor()
        cursor.execute("""
            INSERT INTO reported_questions (question, answer, reported_date)
            VALUES (%s, %s, %s)
        """, (data.pregunta, data.respuesta, date.today()))
        conexion.commit()
        cursor.close()
        return {"mensaje": "Pregunta reportada exitosamente"}
    except Exception as e:
        print("Error al reportar pregunta:", e)
        raise HTTPException(status_code=500, detail="Error al guardar la pregunta")