import psycopg2
conexion = psycopg2.connect(
    dbname="boxia_db",
    user="boxia_user",
    password="boxia",
    host="localhost",
    port="5432"
)

cursor = conexion.cursor()
cursor.execute("SELECT * FROM reported_questions")
resultados = cursor.fetchall()
for fila in resultados:
    print(fila)
cursor.close()
conexion.close()


