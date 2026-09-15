import math
from fastapi import FastAPI
from app.engine import search_db, generate_answer

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ask")
def ask_question(query: str):
    result = search_db(query)

    if result:
        raw_score = 1 / (1 + math.exp(-result["rerank_score"]))
        retrieval_score = max(0.1, min(raw_score, 0.95))
        context = result["text"]
    else:
        return {
            "answer": "I could not find a relevant source in the local database, so I cannot verify an answer. Please consult a qualified medical professional for personal advice.",
            "source": None,
            "retrieval_score": None,
        }

    answer = generate_answer(query, context)

    return {"answer": answer, "source": context, "retrieval_score": retrieval_score}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
