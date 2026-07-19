# 🩺 MedAI-RAG: Local-First Medical RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers medical questions using a local vector-search database and a quantized LLM running entirely on CPU — no external inference API required.

## ⚠️ Medical Disclaimer

**IMPORTANT:** This project is a Proof of Concept for technical demonstration only. The AI can hallucinate or retrieve imprecise context. It is not a diagnostic tool. Always consult a certified medical professional for health advice.

## 📚 Data Source

Medical Q&A data is [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) (Medical Question Answering Dataset), streamed from Hugging Face and embedded with `sentence-transformers/all-MiniLM-L6-v2` into a local vector store (47,441 question/answer pairs indexed).

## 🎥 Demo

[Live demo link — add once deployed]
[Screen recording / GIF — add here]

## 🏗️ System Architecture

The application is a microservices-style system with a clean separation between retrieval, inference, and orchestration:

- **Retrieval Engine:** SQLite with the `sqlite-vec` extension for local vector similarity search.
- **Inference Layer:** `llama-cpp-python` running a quantized Gemma-3-1B-it model (Q4_K_M GGUF), CPU-only, using the model's native chat template for reliable, on-topic responses.
- **Backend API:** FastAPI orchestrating retrieval → prompt construction → generation.
- **Frontend:** Streamlit chat interface with confidence-scored answers and source context.

[Architecture diagram — add here]

## 🛠️ Engineering Highlights

- **Native chat-template prompting:** Uses `llama-cpp-python`'s chat completion API with the model's built-in Gemma turn template and correct stop tokens, rather than hand-built prompt strings — this measurably reduces off-topic or malformed generations compared to raw text completion.
- **Universal pathing:** Environment-agnostic configuration via `pathlib`, so the same code runs identically across local, Docker, and Kubernetes environments.
- **Persistent storage:** Kubernetes Persistent Volume Claims (PVC) ensure the 47k+ row vector database survives pod restarts.
- **Cold-start optimization:** Model initialization happens at module import time (not on first request), avoiding lazy-load latency spikes on the first query.
- **Idempotent ingestion:** The ingestion pipeline drops and rebuilds its tables together on each run, so `medquad` and `vec_medquad` row counts never drift out of sync.
- **Graceful retrieval fallback:** Retrieval checks the top-5 nearest neighbors and skips any with empty answer fields (a real, if uncommon, gap in the source dataset), rather than surfacing an error.

## 📊 Benchmarks

Measured with the included `benchmark.py` — a memory-capped Docker container (matching this project's Kubernetes resource limits) rather than the original AWS EC2 free-tier instance. See *Reproducing these numbers* below for methodology.

| Metric | Value |
| --- | --- |
| Cold start | ~17–24s (varies by run) |
| Peak RAM (unconstrained) | ~1.6 GB |
| Query latency (p50) | 2.96s |
| Query latency (p95) | 7.35s |
| Mean retrieval confidence | 58% |
| Requests benchmarked | 18 |

**Note on memory:** this project originally targeted a 1GB AWS EC2 free-tier instance. In practice, the full stack (embedding model + quantized LLM + FastAPI/Streamlit overhead) measures closer to 1.6GB under normal use — the Kubernetes manifest (`k8s/medai.yaml`) reflects this with a 1.5Gi request / 3Gi limit rather than the original 1GB target.

### Reproducing these numbers

```bash
docker compose down
docker compose up -d
python benchmark.py --url http://localhost:8000 --cold-start --container medai-backend-1
```

This measures cold start, then samples container RAM in the background while running the query latency test, then writes `benchmark_results.md` and `benchmark_results.json`.

## 🚀 Deployment

### Local Development (Docker Compose)

```bash
docker compose up --build
```

First run downloads the GGUF model (~760MB) from Hugging Face into `app/models/`, persisted across restarts via a volume mount.

Then ingest the vector database (one-time, or anytime you want to rebuild it):

```bash
docker compose exec backend python -m app.ingest_data
```

The app is then available at `http://localhost:8501` (Streamlit UI) with the API at `http://localhost:8000`.

### Kubernetes (Production)

```bash
# 1. Apply the manifest
kubectl apply -f k8s/medai.yaml

# 2. Initialize the Vector DB inside the pod
kubectl exec -it <backend-pod-name> -- python -m app.ingest_data
```

## 📈 Tech Stack

- **Language:** Python 3.10
- **AI/ML:** sentence-transformers (`all-MiniLM-L6-v2`), llama-cpp-python (Gemma-3-1B-it GGUF, Q4_K_M)
- **Database:** SQLite + sqlite-vec
- **Orchestration:** Kubernetes, Docker, Docker Compose
- **Cloud:** AWS EC2 (Free Tier) — original deployment target; see Benchmarks for current local measurements
