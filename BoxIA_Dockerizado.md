# 🧠 BoxIA – Proyecto Dockerizado

¡Bienvenido/a a BoxIA!\
Este proyecto incluye toda la infraestructura necesaria para tener tu propio asistente conversacional (IA), almacenamiento vectorial y base de datos de reportes.\
**No necesitas instalar Python, Postgres ni Ollama manualmente:** ¡todo corre con Docker! 🚀

---

## ⚙️ Prerrequisitos

- [Docker](https://docs.docker.com/get-docker/) instalado.
- [Docker Compose](https://docs.docker.com/compose/install/) instalado (la mayoría de las veces, ya viene con Docker).
- **Al menos 20GB de espacio libre en disco** (el modelo de IA es pesado).

---

## 📁 Estructura del proyecto

```
BoxIA/
│
├── docker-compose.yml
├── requirements.txt
├── Dockerfile
├── api.py
├── conexion.py
├── ... (otros archivos de tu backend)
├── docs/                 # Aquí se almacenan los documentos cargados
├── chroma_db_dir/        # Aquí se almacena la base de datos vectorial (se crea sola)
├── postgres_init/
│   └── init.sql          # Script para crear usuario, base y tablas en Postgres
└── README.md
```

---

## 🏁 Instalación paso a paso

1. **Clona o descomprime este proyecto en tu PC.**

2. **Abre una terminal en la carpeta raíz del proyecto.**

3. **Arranca todo el sistema** (por primera vez puede demorar por la descarga del modelo de IA):

   ```bash
   sudo docker compose up --build
   ```

   - Si no tienes que usar `sudo`, omítelo: `docker compose up --build`
   - Si tu sistema usa `docker-compose` (con guión), cambia el comando por `sudo docker-compose up --build`.

4. **¡Listo!**

   - Cuando veas en la terminal mensajes como:
     ```
     Uvicorn running on http://0.0.0.0:8000
     Listening on [::]:11434 (Ollama)
     ```
     ya está todo funcionando.

---

## 🚀 Primer arranque: ¿Qué sucede?

- **Todo se crea automáticamente**:

  - Usuario de base de datos, base de datos y tabla (`reported_questions`) usando el script `postgres_init/init.sql`
  - Modelo de IA se descarga automáticamente.
  - Volúmenes de datos se configuran para que tus datos NO se pierdan si apagas los contenedores.

- **No necesitas hacer nada manualmente** (no crear usuarios, ni base, ni tabla, ni instalar dependencias Python).

---

## 🖥️ Acceso y uso

- **Prueba la API** desde el navegador en: [http://localhost:8000/docs](http://localhost:8000/docs)

- **Tus interfaces gráficas** PyQt se conectan a la API en `http://localhost:8000`.

- **Para detener todo**:

  ```bash
  sudo docker compose down
  ```

- **Para borrar todo y reiniciar desde cero** (incluye la base de datos, ¡cuidado!):

  ```bash
  sudo docker compose down --volumes
  ```

---

## 🛠️ Consejos útiles

- Si el puerto 5432 está ocupado (Postgres local), cámbialo en `docker-compose.yml` y en los archivos de conexión.
- **La primera vez** puede tardar en descargar el modelo de IA (\~4GB).
- Los datos (BD, modelo) se mantienen entre reinicios del contenedor.
- Puedes consultar el estado de la base desde dentro del contenedor:
  ```bash
  sudo docker exec -it boxia-postgres-1 psql -U boxia_user -d boxia_db
  ```
- Si cambias el código fuente, basta con repetir `docker compose up --build`.

---

## 🚑 Problemas frecuentes y soluciones

- **Error "role ... already exists" o "database ... already exists":**
  - Es normal al recrear contenedores, el script ignora estos errores si ya existen usuario y BD.
- **Error "relation ... does not exist" (no existe la tabla):**
  - Puede ocurrir si borraste el volumen y la tabla no estaba en `init.sql`.
    - Solución: agrega el CREATE TABLE al `init.sql`, borra volúmenes y repite el arranque.
- **Permiso denegado a Docker:**
  - Antepone `sudo` a todos los comandos, o agrega tu usuario al grupo `docker`:
    ```bash
    sudo usermod -aG docker $USER
    # Luego cierra sesión y vuelve a entrar
    ```
- **No puedes acceder a la API:**
  - Asegúrate de que la terminal dice `Uvicorn running on http://0.0.0.0:8000`
  - Espera a que Ollama termine de descargar el modelo.
- **¿Se crea todo en un PC nuevo?**
  - ¡Sí! Al primer arranque, Docker y Postgres crean todo usando el script y variables configuradas.

---

## 🧠 Autores y créditos

- Desarrollado por [BoxIA Team]
- IA local: [Ollama Llama 3.1 8B]
- Base vectorial: ChromaDB
- Base de datos: PostgreSQL
- Infraestructura: Docker Compose

---

¡Disfruta tu instancia de BoxIA!\

