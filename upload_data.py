import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

# Función para cargar documentos desde un archivo
def cargar_documentos(ruta_archivo):
    # Verifica si el archivo existe
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"El archivo {ruta_archivo} no existe.")

    # Carga el archivo usando PyMuPDFLoader
    loader = PyMuPDFLoader(ruta_archivo)
    documentos = loader.load()

    # Agrega la ruta del archivo como metadato a cada documento
    for doc in documentos:
        doc.metadata["file_path"] = ruta_archivo

    # Divide los documentos en fragmentos más pequeños
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = text_splitter.split_documents(documentos)
    return docs

# Función para crear un vectorstore a partir de documentos
def crear_vectorstore(docs):
    # Inicializa el modelo de embeddings
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Crea un vectorstore usando los documentos y el modelo de embeddings
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="chroma_db_dir",  # Directorio donde se guardará el vectorstore
        collection_name="stanford_report_data"  # Nombre de la colección
    )
    return vectorstore

# Función para actualizar el vectorstore con nuevos documentos
def actualizar_vectorstore_desde_docs(vectorstore):
    # Obtiene los IDs existentes en el vectorstore
    ids_existentes = set(vectorstore.get()['ids'])

    nuevos_docs = []  # Lista para almacenar nuevos documentos
    nuevos_ids = []  # Lista para almacenar nuevos IDs

    # Itera sobre los archivos en el directorio "docs"
    for archivo in os.listdir("docs"):
        # Filtra archivos con extensiones específicas
        if archivo.endswith((".pdf", ".txt", ".md")):
            ruta = os.path.join("docs", archivo)  # Construye la ruta completa del archivo

            id_base = os.path.splitext(archivo)[0]  # Obtiene el nombre base del archivo (sin extensión)

            # Verifica si el ID base ya existe en los IDs del vectorstore
            if any(id_base in id for id in ids_existentes):
                continue

            print(f"📄 Nuevo documento detectado: {archivo}")
            documentos = cargar_documentos(ruta)  # Carga los documentos desde el archivo

            # Agrega la ruta del archivo como metadato y genera nuevos IDs
            for i, doc in enumerate(documentos):
                doc.metadata['file_path'] = ruta  # Agrega la ruta del archivo como metadato
                nuevos_ids.append(f"{id_base}_{i}")  # Genera un nuevo ID para cada fragmento
            nuevos_docs.extend(documentos)  # Agrega los documentos a la lista de nuevos documentos

    # Si hay nuevos documentos, los añade al vectorstore
    if nuevos_docs:
        vectorstore.add_documents(documents=nuevos_docs, ids=nuevos_ids)
        print(f"✅ Nuevos documentos añadidos al vectorstore.")
    else:
        print("📁 No hay documentos nuevos por indexar.")