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
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})  # Recupera los 4 documentos más relevantes

    # Plantilla personalizada para el prompt del modelo
    custom_prompt_template = """
                    Eres un asistente útil y analítico que responde preguntas basadas en documentos entregados.

                    Tu tarea es razonar sobre la información contenida en los documentos, no solo repetirla. 
                    Analiza el contexto cuidadosamente, identifica relaciones entre ideas, y responde con conclusiones lógicas incluso si no están explícitas en el texto.

                    Responde siempre en español. Si no sabes la respuesta, di "No lo sé".

                    Usa un máximo de tres frases y mantén la respuesta clara y precisa.

                    Contexto: {context}
                    Pregunta: {question}

                    Respuesta:
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
