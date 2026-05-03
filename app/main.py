from fastapi import FastAPI
from app.engine import search_db, llm

app = FastAPI()

@app.get("/ask")
def ask_question(query: str):
    context = search_db(query)
    if not context:
        context = "No specific medical records found for this query."
    
    prompt = f"<|begin_of_text|>system\nYou are a medical assistant. Info: {context}\nuser\n{query}\nassistant\n"
    response = llm(prompt, max_tokens=256)
    
    return {
        "answer": response['choices'][0]['text'].strip(),
        "source": context[:100] if context else "No source"
    }