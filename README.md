# Medical AI RAG Assistant 🩺

A Proof-of-Concept Medical Assistant using RAG (Retrieval-Augmented Generation). 
Optimized to run on **extreme hardware constraints** (1GB RAM / 1vCPU).

## 🚀 The Architecture
This project uses a dual-container setup:
- **Backend**: FastAPI + `llama-cpp-python` (CPU Quantized LLM) + `sqlite-vec` for vector search.
- **Frontend**: Streamlit UI for an interactive chat experience.

## 🛠️ Hardware Optimization (The 1GB Challenge)
To successfully deploy on an **EC2 Free Tier (1GB RAM / 30GB HDD)**, I implemented:
- **Pre-built CPU Wheels**: Bypassed heavy C++ compilation to prevent OOM (Out of Memory) crashes during deployment.
- **Quantized LLM**: Uses 4-bit quantization to fit the model within limited memory.
- **Docker Resource Limits**: Hard-capped container memory to leave room for the OS.
- **Shared Image layers**: Optimized the Dockerfile to reduce disk footprint.

## 📦 Local Setup
1. Clone the repo.
2. Build and run via Docker Compose:
   ```bash
   docker-compose up --build
   ```

## ⚠️ Disclaimer

# THIS IS A PROOF OF CONCEPT. ALWAYS CONSULT A REAL PHYSICIAN.