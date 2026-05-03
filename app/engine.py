import os
import struct
import sqlite3
import sqlite_vec
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# Load models once when the module is imported
print("Initializing Engines...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_llm():
    repo = "unsloth/gemma-3-1b-it-GGUF"
    file = "gemma-3-1b-it-BF16.gguf"
    path = hf_hub_download(repo_id=repo, filename=file, local_dir="app/models")
    return Llama(model_path=path, n_ctx=2048, n_threads=4)

llm = get_llm()

def search_db(query):
    db = sqlite3.connect("app/data/medical_data.db")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    
    query_vector = embed_model.encode(query).tolist()
    query_bytes = struct.pack(f"{len(query_vector)}f", *query_vector)

    cursor = db.execute("""
        SELECT m.answer FROM vec_medquad v
        LEFT JOIN medquad m ON v.rowid = m.id
        WHERE v.embedding MATCH ? AND k = 1
        ORDER BY v.distance ASC
    """, [query_bytes])
    
    row = cursor.fetchone()
    db.close()
    return row[0] if row else "No data found."