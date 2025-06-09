Requeridos:
Instalar Python (en el PATCH)

python main.py

UBUNTU
-Para boorrar el ambiente virtual anterior:
rm -rf docs-env
-Para crear el ambiente virtual:
python3 -m venv docs-env
-Para usar el ambiente virtual: 
source docs-env/bin/activate
-Para desactivar el ambiente virtual:
deactivate
-Instalar dependencias:
pip install -r requirements.txt
-Arrancar la interfaz:
python3 interfaz_docs.py
