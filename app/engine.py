import os
import struct
import sqlite3
import sqlite_vec
import threading
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path

print("Initializing Engines...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "app" / "data" / "medical_data.db"))

MODEL_REPO = os.getenv("MODEL_REPO", "unsloth/gemma-3-1b-it-GGUF")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "gemma-3-1b-it-Q4_K_M.gguf")

MODEL_DIR = BASE_DIR / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def resolve_model_path():
    configured_path = os.getenv("MODEL_PATH")
    if configured_path:
        model_path = Path(configured_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MODEL_PATH does not point to a GGUF file: {model_path}"
            )
        return str(model_path)
    return hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=str(MODEL_DIR),
    )


def get_llm(path):
    return Llama(model_path=path, n_ctx=4096, n_threads=4)


llm = get_llm(resolve_model_path())
llm_lock = threading.Lock()


def search_db(query, k=10):
    if not Path(DB_PATH).is_file():
        return None
    db = sqlite3.connect(DB_PATH)
    try:
        db.enable_load_extension(True)
        sqlite_vec.load(db)

        query_vector = embed_model.encode(query).tolist()
        query_bytes = struct.pack(f"{len(query_vector)}f", *query_vector)

        cursor = db.execute(
            """
            SELECT m.answer, v.distance FROM vec_medquad v
            LEFT JOIN medquad m ON v.rowid = m.id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance ASC
        """,
            [query_bytes, k],
        )
        rows = cursor.fetchall()
    finally:
        db.close()

    candidates = [(answer, distance) for answer, distance in rows if answer]
    if not candidates:
        return None

    pairs = [(query, answer) for answer, _ in candidates]
    rerank_scores = reranker.predict(pairs)

    best_idx = int(rerank_scores.argmax())
    best_answer, best_distance = candidates[best_idx]
    best_rerank_score = float(rerank_scores[best_idx])

    return {
        "text": best_answer,
        "distance": best_distance,
        "rerank_score": best_rerank_score,
    }


def retrieve_contexts(query, k=10, top_n=3):
    if not Path(DB_PATH).is_file():
        return []
    db = sqlite3.connect(DB_PATH)
    try:
        db.enable_load_extension(True)
        sqlite_vec.load(db)

        query_vector = embed_model.encode(query).tolist()
        query_bytes = struct.pack(f"{len(query_vector)}f", *query_vector)

        cursor = db.execute(
            """
            SELECT m.answer, v.distance FROM vec_medquad v
            LEFT JOIN medquad m ON v.rowid = m.id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance ASC
        """,
            [query_bytes, k],
        )
        rows = cursor.fetchall()
    finally:
        db.close()

    candidates = [(answer, distance) for answer, distance in rows if answer]
    if not candidates:
        return []

    pairs = [(query, answer) for answer, _ in candidates]
    rerank_scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    return [answer for (answer, _distance), _score in ranked[:top_n]]


def generate_answer(query, context):
    # llama-cpp-python uses one shared stateful model instance in this process.
    with llm_lock:
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You provide general educational medical information, not diagnosis or "
                        "personalized treatment. Answer only when the supplied source supports "
                        "the answer. If it does not, say that you cannot verify the answer from "
                        "the available source. Recommend a qualified clinician for personal "
                        "medical decisions. Source:\n" + context
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=512,
        )
    return response["choices"][0]["message"]["content"].strip()
