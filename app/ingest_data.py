import sqlite3
import sqlite_vec
import os
import struct
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "app" / "data" / "medical_data.db"))
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

def serialize_f32(vector):
    return struct.pack(f"{len(vector)}f", *vector)

def setup_database():
    print("1. Loading Embedding Model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("2. Connecting to SQLite and loading sqlite-vec...")
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.execute("DROP TABLE IF EXISTS medquad")
    db.execute("DROP TABLE IF EXISTS vec_medquad")
    
    db.execute("""
        CREATE TABLE medquad (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT
        )
    """)
    db.execute("""
        CREATE VIRTUAL TABLE vec_medquad USING vec0(
            embedding float[384]
        )
    """)

    print("3. Streaming dataset from Hugging Face...")
    ds = load_dataset("lavita/MedQuAD", split='train', streaming=True)

    print("4. Starting ingestion (Batching 100 rows at a time)...")
    text_batch = []
    vec_batch = []
    
    for i, row in enumerate(ds):
        q = row['question']
        a = row['answer']
        raw_vector = model.encode(q).tolist()
        vec_bytes = serialize_f32(raw_vector)
        row_id = i + 1 
        text_batch.append((row_id, q, a))
        vec_batch.append((row_id, vec_bytes))
        
        if len(text_batch) >= 100:
            db.executemany("INSERT INTO medquad (id, question, answer) VALUES (?, ?, ?)", text_batch)
            db.executemany("INSERT INTO vec_medquad (rowid, embedding) VALUES (?, ?)", vec_batch)
            db.commit()
            text_batch = []
            vec_batch = []
            print(f"   ...Indexed {row_id} rows")

    if text_batch:
        db.executemany("INSERT INTO medquad (id, question, answer) VALUES (?, ?, ?)", text_batch)
        db.executemany("INSERT INTO vec_medquad (rowid, embedding) VALUES (?, ?)", vec_batch)
        db.commit()

    print(f"Success! Database created at {DB_PATH}")
    db.close()

if __name__ == "__main__":
    setup_database()