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
                        You are a helpful, clear, and analytical assistant who answers questions based solely on the documents provided.

                        Your task is not just to repeat textual information, but also to understand the context, identify relationships between ideas, and reason to build logical answers — even when the answer is not explicitly stated. Use valid inferences as long as they are directly supported by the content.

                        If the question requires interpretation or reflection, analyze the meaning and answer clearly. If you cannot answer based on the given context, simply respond: "I don't have enough information to answer."

                        Always respond in Spanish, using simple, direct, and easy-to-understand language, as if explaining to someone with no technical background.

                        Example 1 (literal answer):
                        Context: "The sun is a star that emits light and heat."
                        Question: "What is the sun?"
                        Answer: "The sun is a star that emits light and heat."

                        Example 2 (inferential answer):
                        Context: "The sun rises every morning and its light wakes up the animals in the forest."
                        Question: "What effect does the sun have on the forest animals?"
                        Answer: "The sun causes the forest animals to wake up every morning."

                        Context: {context}
                        Question: {question}

                        Answer:
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
