import mysql.connector
import pandas as pd

df = pd.read_csv("data_with_embeddings.csv")

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "76825917jy",
    "database": "my_database",
    "charset": "utf8mb4"
}

conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

insert_query = """
INSERT INTO store_knowledge (key_name, value_text) VALUES (%s, %s)
"""

for _, row in df.iterrows():
    key = str(row["key"])
    value = str(row["value"])
    cursor.execute(insert_query, (key, value))

conn.commit()
cursor.close()
conn.close()

print("Data successfully inserted into MySQL!")
