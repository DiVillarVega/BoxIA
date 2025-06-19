import psycopg2
import os
conexion = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),  # <- Esto es 'postgres' en docker-compose
    port=os.getenv("POSTGRES_PORT", 5432)
)





