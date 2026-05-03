from fastapi import FastAPI
from app.engine import search_db, llm

app = FastAPI()

@app.get("/ask")
def ask_question(query: str):
    context = search_db(query)
    if context == "No data found.":
        context = "Note: No matching internal medical records found. Providing general knowledge."
        confidence = 0.40
    else:
        confidence = 0.88
    prompt = f"System: You are a medical assistant. Use this info: {context}\nUser: {query}\nAssistant:"
    
    response = llm(prompt, max_tokens=1024, stop=["<|", "User:", "System:"])
    
    answer = response['choices'][0]['text'].strip()

    return {
        "answer": answer,
        "source": context[:100],
        "confidence": confidence
    }