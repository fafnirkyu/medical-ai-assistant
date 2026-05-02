import os
from fastapi import FastAPI
from llama_cpp import Llama
from contextlib import asynccontextmanager
from database import search_medical_data

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/gemma-3-1b-it-Q4_K_M.gguf")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=2,
        flash_attn=True,
        verbose=False
    )
    print("Model primed and ready.")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/ask")
async def ask_question(query: str):
    context, confidence = search_medical_data(query)
    if confidence < 0.60:
        disclaimer = "Note: Low confidence match. Always consult a doctor.\n"
    else:
        disclaimer = ""
    prompt = f"### Instruction: Use the context to answer.\nContext: {context}\nQuestion: {query}\n### Response:"
    
    output = app.state.llm(
        prompt,
        max_tokens=256,
        stop=["###"],
        echo=False
    )
    
    response_text = output["choices"][0]["text"]
    
    return {
        "answer": f"{disclaimer}{response_text}",
        "confidence": f"{confidence:.2%}",
        "source": "MedQuAD Dataset"
    }