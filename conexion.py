import psycopg2
conexion = psycopg2.connect(
    dbname="boxia_db",
    user="boxia_user",
    password="boxia",
    host="localhost",
    port="5432"
)
