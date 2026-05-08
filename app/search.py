import os
import sqlite3
import sqlite_vec
import struct
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "app" / "data" / "medical_data.db"))

# Re-use our serialization helper
def serialize_f32(vector):
    return struct.pack(f"{len(vector)}f", *vector)

def search_database(user_query, limit=3):
    print("1. Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("2. Connecting to database...")
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)

    print(f"\nThinking about: '{user_query}'...\n")
    
    # Convert the user's text into a vector
    query_vector = model.encode(user_query).tolist()
    query_bytes = serialize_f32(query_vector)

    # Perform the Vector Search (KNN - K-Nearest Neighbors)
    # This asks sqlite-vec to find the rows with the smallest mathematical distance to our query
    cursor = db.execute("""
        SELECT 
            m.question, 
            m.answer,
            v.distance
        FROM vec_medquad v
        LEFT JOIN medquad m ON v.rowid = m.id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
    """, [query_bytes, limit])

    results = cursor.fetchall()
    
    print("--- TOP 3 MEDICAL MATCHES ---")
    for i, row in enumerate(results):
        question = row[0]
        answer = row[1][:200] + "..." # Truncate answer for readability
        distance = row[2]
        
        print(f"\nMatch {i+1} (Distance: {distance:.4f})")
        print(f"Q: {question}")
        print(f"A: {answer}")

    db.close()

if __name__ == "__main__":
    # Test it out with a sample question
    test_query = "What are the early symptoms of asthma?"
    search_database(test_query)