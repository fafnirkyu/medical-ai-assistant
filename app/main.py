from fastapi import FastAPI
from app.engine import search_db, llm

app = FastAPI()

@app.get("/ask")
def ask_question(query: str):
    result = search_db(query)
    
    if result:
        raw_score = 1.0 - result['distance']
        confidence = max(0.1, min(raw_score, 0.95)) # Keep it between 10% and 95%
        context = result['text']
    else:
        context = "No data found."
        confidence = 0.30
    prompt = f"System: You are a medical assistant. Use this info: {context}\nUser: {query}\nAssistant:"
    
    response = llm(prompt, max_tokens=1024, stop=["<|", "User:", "System:"])
    
    answer = response['choices'][0]['text'].strip()

    return {
        "answer": answer,
        "source": context[:1000],
        "confidence": confidence
    }