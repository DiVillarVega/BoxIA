FROM python:3.10-slim

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements y el resto del código
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Variables de entorno para la conexión a PostgreSQL
ENV POSTGRES_DB=boxia_db
ENV POSTGRES_USER=boxia_user
ENV POSTGRES_PASSWORD=boxia
ENV POSTGRES_HOST=postgres
ENV POSTGRES_PORT=5432

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
