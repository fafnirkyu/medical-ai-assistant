import math
from fastapi import FastAPI
from engine import search_db, llm

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ask")
def ask_question(query: str):
    result = search_db(query)
    
    if result:
        raw_score = 1 / (1 + math.exp(-result['rerank_score']))
        confidence = max(0.1, min(raw_score, 0.95))
        context = result['text']
    else:
        context = "No data found."
        confidence = 0.30
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": f"You are a medical assistant. Use this information to answer the user's question concisely: {context}",
            },
            {"role": "user", "content": query},
        ],
        max_tokens=512,
    )

    answer = response['choices'][0]['message']['content'].strip()

    return {
        "answer": answer,
        "source": context[:1000],
        "confidence": confidence
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)