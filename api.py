from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
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
                        Eres un asistente útil, claro y analítico que **solo puede responder preguntas basándose en la información contenida en los documentos proporcionados**. No tienes acceso a conocimientos generales ni a información externa, y no debes hacer suposiciones sin justificación directa en los textos.

                        ### Instrucciones clave:
                        - Tu conocimiento está **limitado únicamente** al contenido textual de los documentos cargados.
                        - **No debes utilizar conocimientos previos** que no estén expresamente en los textos.
                        - Si una pregunta no puede responderse con base en los documentos, responde con: **"No tengo suficiente información para responder."**
                        - No rellenes vacíos con suposiciones. No inventes. No des definiciones que no estén explícitas o razonablemente inferidas del contenido.
                        - **No digas nada** que no se sustente clara y directamente en los documentos.

                        ### Tu objetivo:
                        No solo repetir lo que dicen los textos, sino **comprender el contexto**, **identificar relaciones entre ideas**, y **razonar** para construir respuestas lógicas — siempre respaldadas por el contenido entregado.

                        Si la pregunta requiere interpretación o análisis, hazlo **dentro del marco de información que te entregan los documentos**.

                        Responde **siempre en español**, con un lenguaje claro, simple y directo, como si explicaras a alguien sin conocimientos técnicos.

                        ### Ejemplos

                        **Respuesta literal:**
                        - Contexto: "El sol es una estrella que emite luz y calor."
                        - Pregunta: "¿Qué es el sol?"
                        - Respuesta: "El sol es una estrella que emite luz y calor."

                        **Respuesta inferencial:**
                        - Contexto: "El sol sale cada mañana y su luz despierta a los animales del bosque."
                        - Pregunta: "¿Qué efecto tiene el sol sobre los animales del bosque?"
                        - Respuesta: "El sol hace que los animales del bosque se despierten cada mañana."

                        **Respuesta cuando falta información:**
                        - Pregunta: "¿Quién descubrió América?"
                        - Respuesta: "No tengo suficiente información para responder."

                        ---

                        **Contexto:**  
                        {context}

                        **Pregunta:**  
                        {question}

                        **Respuesta:**
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
    # Recuperar documentos relevantes manualmente
    retrieved_docs = retriever.get_relevant_documents(p.pregunta)

    # Log de chunks recuperados (imprime en consola)
    print("\n\n🔍 Chunks recuperados:")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Chunk {i+1} ---")
        print(doc.page_content)
        print("-" * 40)

    # Invocar al sistema QA normalmente
    respuesta = qa.invoke({"query": p.pregunta})

    return {"respuesta": respuesta['result']}



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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documentos_divididos = splitter.split_documents(documentos)

    vectorstore.add_documents(documentos_divididos)

    return {"mensaje": f"{archivo.filename} cargado exitosamente."}

# Endpoint para cargar documentos xlsx
@app.post("/cargar-documento-xlsx")
async def cargar_excel(archivo: UploadFile = File(...)):
    if not archivo.filename.endswith(".xlsx"):
        return {"error": "Solo se permiten archivos XLSX."}

    ruta_temporal = f"docs/{archivo.filename}"
    with open(ruta_temporal, "wb") as f:
        f.write(await archivo.read())

    loader = UnstructuredExcelLoader(ruta_temporal)
    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documentos_divididos = splitter.split_documents(documentos)

    vectorstore.add_documents(documentos_divididos)

    return {"mensaje": f"{archivo.filename} cargado exitosamente."}
