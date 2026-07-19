import os
import struct
import sqlite3
import sqlite_vec
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Load models once when the module is imported
print("Initializing Engines...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "app" / "data" / "medical_data.db"))

repo = "unsloth/gemma-3-1b-it-GGUF"
file = "gemma-3-1b-it-Q4_K_M.gguf"

MODEL_DIR = BASE_DIR / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(repo_id=repo, filename=file, local_dir=str(MODEL_DIR))

def get_llm(path):
    return Llama(model_path=path, n_ctx=2048, n_threads=4)

llm = get_llm(path)

def search_db(query):
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    
    query_vector = embed_model.encode(query).tolist()
    query_bytes = struct.pack(f"{len(query_vector)}f", *query_vector)

    cursor = db.execute("""
        SELECT m.answer, v.distance FROM vec_medquad v
        LEFT JOIN medquad m ON v.rowid = m.id
        WHERE v.embedding MATCH ? AND k = 5
        ORDER BY v.distance ASC
    """, [query_bytes])

    rows = cursor.fetchall()
    db.close()

    # Some MedQuAD entries have an empty/NULL answer field. Skip those and
    # use the first nearest-neighbor match that actually has answer text.
    for answer, distance in rows:
        if answer:
            return {"text": answer, "distance": distance}
    return None