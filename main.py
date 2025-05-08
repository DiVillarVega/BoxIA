from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from upload_data import cargar_documentos, crear_vectorstore, actualizar_vectorstore_desde_docs
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import os

# Colores para el texto en la consola
AZUL = "\033[94m"
VERDE = "\033[92m"
RESET = "\033[0m"

# Función principal para iniciar el chat
def iniciar_chat():
    # Inicializa el modelo de lenguaje LLaMA 3.1
    llm = Ollama(model="llama3.1")  # Usando LLaMA 3.1 8B
    
    # Inicializa el modelo de embeddings
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Carga el vectorstore existente o lo crea si no existe
    vectorstore = Chroma(
        embedding_function=embed_model,
        persist_directory="chroma_db_dir",  # Directorio donde se guarda el vectorstore
        collection_name="stanford_report_data"  # Nombre de la colección
    )
    
    # Verifica si el vectorstore está vacío
    total_rows = len(vectorstore.get()['ids'])
    if total_rows == 0:
        print("⚠️  Vectorstore vacío. Cargando todos los documentos de 'docs'...")
        documentos = []  # Lista para almacenar documentos
        # Itera sobre los archivos en el directorio "docs"
        for archivo in os.listdir("docs"):
            if archivo.endswith((".pdf", ".txt", ".md")):  # Filtra archivos con extensiones específicas
                ruta = os.path.join("docs", archivo)  # Construye la ruta completa del archivo
                documentos.extend(cargar_documentos(ruta))  # Carga los documentos desde el archivo
        vectorstore = crear_vectorstore(documentos)  # Crea el vectorstore con los documentos

    # Actualiza el vectorstore con nuevos documentos
    actualizar_vectorstore_desde_docs(vectorstore)

    # Configura el recuperador para buscar documentos relevantes
    retriever = vectorstore.as_retriever(search_kwargs={'k': 10})  # Recupera los 4 documentos más relevantes

    # Plantilla personalizada para el prompt del modelo
    custom_prompt_template = """
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
    # Crea el prompt usando la plantilla personalizada
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    # Configura la cadena de preguntas y respuestas (QA)
    qa = RetrievalQA.from_chain_type(
        llm=llm,  # Modelo de lenguaje
        chain_type="stuff",  # Tipo de cadena
        retriever=retriever,  # Recuperador configurado
        return_source_documents=True,  # Devuelve los documentos fuente
        chain_type_kwargs={"prompt": prompt}  # Usa el prompt personalizado
    )

    # Inicia el bucle del chat
    print("¡Bienvenido al chat! Escribe 'salir' para terminar.")
    while True:
        # Solicita una pregunta al usuario
        pregunta = input(f"{AZUL}Tú:{RESET} ")
        if pregunta.lower() == 'salir':  # Verifica si el usuario quiere salir
            print("¡Hasta luego!")
            break

        # Obtiene la respuesta del modelo
        respuesta = qa.invoke({"query": pregunta})
        metadata = []  # Lista para almacenar metadatos de los documentos fuente
        # Itera sobre los documentos fuente y extrae los metadatos
        for doc in respuesta['source_documents']:
            metadata.append(('page: ' + str(doc.metadata['page']), doc.metadata['file_path']))
        
        # Muestra la respuesta y los metadatos
        print(f"{VERDE}Asistente:{RESET}", respuesta['result'], '\n', metadata)

# Punto de entrada principal del programa
if __name__ == "__main__":
    iniciar_chat()
