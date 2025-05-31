Requeridos:
Instalar Python (en el PATCH)
-Instalar Ollama. Luego en CMD installar llama3.1:
ollama run llama3.1

WINDOWS
-Para crear el ambiente virtual:
python -m venv boxia-env
-Para usar el ambiente virtual: 
boxia-env\Scripts\activate
-Instalar dependencias:
pip install -r requirements.txt
-Para borrar la BD:
rmdir /s /q chroma_db_dir
-Arrancar proyecto:
python main.py

UBUNTU
-Para boorrar el ambiente virtual anterior:
rm -rf boxia-env
-Para crear el ambiente virtual:
python3 -m venv boxia-env
-Para usar el ambiente virtual: 
source boxia-env/bin/activate
-Para desactivar el ambiente virtual:
deactivate
-Instalar dependencias:
pip install -r requirements.txt
-Para borrar la BD:
rm -r chroma_db_dir
-Arrancar el proyecto:
python3 main.py
-Arrancar el proyecto desde la api:
uvicorn api:app --reload