import os
import struct
import sqlite3
import sqlite_vec
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path

print("Initializing Engines...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
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

def search_db(query, k=10):
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    
    query_vector = embed_model.encode(query).tolist()
    query_bytes = struct.pack(f"{len(query_vector)}f", *query_vector)

    cursor = db.execute("""
        SELECT m.answer, v.distance FROM vec_medquad v
        LEFT JOIN medquad m ON v.rowid = m.id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance ASC
    """, [query_bytes, k])

    rows = cursor.fetchall()
    db.close()

    candidates = [(answer, distance) for answer, distance in rows if answer]
    if not candidates:
        return None

    pairs = [(query, answer) for answer, _ in candidates]
    rerank_scores = reranker.predict(pairs)

    best_idx = int(rerank_scores.argmax())
    best_answer, best_distance = candidates[best_idx]
    best_rerank_score = float(rerank_scores[best_idx])

    return {"text": best_answer, "distance": best_distance, "rerank_score": best_rerank_score}