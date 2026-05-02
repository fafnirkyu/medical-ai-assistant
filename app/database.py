import sqlite3
import sqlite_vss
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def search_medical_data(query: str):
    query_vector = embed_model.encode(query).tobytes()  
    db = sqlite3.connect("/app/data/medical_data.db")
    db.enable_load_extension(True)
    sqlite_vss.load(db)
    cursor = db.cursor()
    cursor.execute("""
        SELECT answer, vss_distance(embedding, ?) 
        FROM medquad 
        ORDER BY vss_distance(embedding, ?) ASC LIMIT 1
    """, [query_vector, query_vector])
    
    row = cursor.fetchone()
    db.close()
    
    if row:
        return row[0], (1 - row[1]) # Return text and confidence
    return "No relevant medical data found.", 0.0