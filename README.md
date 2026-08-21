# 🩺 MedAI-RAG: Local-First Medical RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers medical questions using a local vector-search database, cross-encoder reranking, and a quantized LLM running entirely on CPU — no external inference API required.

## ⚠️ Medical Disclaimer

**IMPORTANT:** This project is a Proof of Concept for technical demonstration only. The AI can hallucinate or retrieve imprecise context. It is not a diagnostic tool. Always consult a certified medical professional for health advice.

## 📚 Data Source

Medical Q&A data is [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) (Medical Question Answering Dataset), streamed from Hugging Face and embedded with `sentence-transformers/all-MiniLM-L6-v2` into a local vector store (47,441 question/answer pairs indexed).

## 🎥 Demo

[Live demo link — add once deployed]
[Screen recording / GIF — add here]

## 🏗️ System Architecture

The application is a microservices-style system with a clean separation between retrieval, reranking, inference, and orchestration:

- **Retrieval Engine:** SQLite with the `sqlite-vec` extension retrieves the top-10 candidates by embedding similarity (bi-encoder search, `all-MiniLM-L6-v2`).
- **Reranking Layer:** A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores those 10 candidates by jointly evaluating each (query, answer) pair, selecting the most relevant one — catching cases where raw embedding distance ranks a topically-close-but-wrong answer above the correct one.
- **Inference Layer:** `llama-cpp-python` running a quantized Gemma-3-1B-it model (Q4_K_M GGUF), CPU-only, using the model's native chat template for reliable, on-topic responses.
- **Backend API:** FastAPI orchestrating retrieval → reranking → prompt construction → generation.
- **Frontend:** Streamlit chat interface with confidence-scored answers and source context.

[Architecture diagram — add here]

## 🛠️ Engineering Highlights

- **Two-stage retrieval (bi-encoder + cross-encoder reranking):** Broad, cheap embedding search narrows 47k+ documents to 10 candidates; a cross-encoder then reranks those 10 by actual query-answer relevance rather than trusting embedding distance alone.
- **Systematic RAG evaluation (RAGAS):** Built an offline evaluation harness scoring faithfulness, answer relevancy, context precision, and context recall against a sampled test set — moving quality assessment from manual spot-checking to a repeatable, quantified process. See *RAG Quality Evaluation* below.
- **Native chat-template prompting:** Uses `llama-cpp-python`'s chat completion API with the model's built-in Gemma turn template and correct stop tokens, rather than hand-built prompt strings — this measurably reduces off-topic or malformed generations compared to raw text completion.
- **Universal pathing:** Environment-agnostic configuration via `pathlib`, so the same code runs identically across local, Docker, and Kubernetes environments.
- **Persistent storage:** Kubernetes Persistent Volume Claims (PVC) ensure the 47k+ row vector database survives pod restarts. Locally, `HF_HOME` is redirected into the same mounted volume so the embedding model and reranker also persist across container restarts, instead of re-downloading on every cold start.
- **Cold-start optimization:** Model initialization happens at module import time (not on first request), avoiding lazy-load latency spikes on the first query.
- **Idempotent ingestion:** The ingestion pipeline drops and rebuilds its tables together on each run, so `medquad` and `vec_medquad` row counts never drift out of sync.
- **Graceful retrieval fallback:** Retrieval checks the top-10 nearest neighbors and skips any with empty answer fields (a real, if uncommon, gap in the source dataset), rather than surfacing an error.

## 🧪 RAG Quality Evaluation (RAGAS)

Manually reading outputs can't distinguish *why* an answer is wrong — whether retrieval pulled the wrong documents, or generation ignored good documents it was given. To measure this properly, I built an offline evaluation harness (`eval/`) using [RAGAS](https://github.com/explodinggradients/ragas), scoring 50 sampled questions against the live pipeline on four metrics:

- **Faithfulness** — are the answer's claims actually backed by the retrieved context?
- **Answer Relevancy** — does the answer address the question asked?
- **Context Precision** — of what was retrieved, how much was relevant?
- **Context Recall** — of what should have been retrieved, how much was?

Judge model: a locally-run `qwen2.5:7b-instruct` (via Ollama), not GPT-4 — kept consistent with the project's local-first, zero-external-API philosophy. **Caveat, stated plainly:** smaller local judges are less consistent at RAGAS's claim-decomposition step than a GPT-4-class judge; these scores are directionally useful, not a precise ground truth.

### Results (n=50)

| Metric | All 50 questions | Retrieval succeeded (28/50) |
| --- | --- | --- |
| Faithfulness | 0.560 | **0.817** |
| Answer Relevancy | 0.412 | **0.712** |
| Context Precision | 0.264 | **0.507** |
| Context Recall | 0.708 | 0.655 |

### Key finding: retrieval fails silently on 44% of queries

The split above isn't cosmetic — **22 of 50 questions (44%) returned zero retrieval candidates**, and the pipeline fell back to `"No data found."` When retrieval *does* return something, faithfulness is solid (0.817) — the generation step is reliable. **The actual bottleneck is retrieval reliability, not the LLM.**

One example makes the risk concrete: for a question where retrieval returned nothing, the model still generated a fluent, disclaimer-wrapped, medically-plausible-sounding answer anyway. RAGAS correctly scored that response's faithfulness at `0.0` — a hallucination that would have read as a perfectly reasonable answer under manual review, invisible without this evaluation step.

**Status:** root cause under active investigation — checking for a mismatch between `medquad` and `vec_medquad` row counts from ingestion, and whether the `sqlite-vec` KNN query is failing silently on specific queries. This section will be updated with the fix and a re-run baseline once resolved.

### Reproducing this evaluation

```bash
pip install ragas==0.1.21 langchain-core==0.2.43 langchain-community==0.2.19 \
    langchain-openai==0.1.25 langchain-huggingface==0.0.3 datasets

ollama pull qwen2.5:7b-instruct

python eval/build_eval_set.py --n 50
python eval/run_ragas.py --model qwen2.5:7b-instruct
```
Writes `eval/ragas_results.md`, `eval/ragas_results.json`, and a per-example breakdown to `eval/ragas_results_detailed.csv`.

## 📊 Benchmarks

Measured with the included `benchmark.py` against a local Docker deployment (Windows/Docker Desktop, WSL2 backend). See *Reproducing these numbers* below for methodology.

| Metric | v1 (embedding-only) | v2 (+ reranking) |
| --- | --- | --- |
| Cold start | ~17–24s | ~15s |
| Peak RAM (unconstrained) | ~1.6 GB | ~1.7–1.9 GB |
| Query latency (p50) | 2.96s | ~2.0–2.8s |
| Query latency (p95) | 7.35s | ~7.3–7.7s |
| Requests benchmarked | 18 | 18 |

**Honest read on these numbers:** reranking added a small, expected RAM cost (a second small model resident in memory) but did **not** meaningfully regress latency — LLM token generation dominates total response time far more than a single batched cross-encoder pass over 10 short candidates does. Cold start held steady/slightly improved, which is more likely run-to-run variance than a real effect of reranking itself.

**On confidence scores specifically:** v1 computed confidence from raw embedding distance; v2 computes it from a sigmoid of the cross-encoder's relevance score. These are two different formulas measuring different things, so the numbers are **not directly comparable** — v2's confidence score is better-calibrated (a proper 0-1 relevance signal rather than a distance proxy), but "confidence went up" between versions isn't a meaningful claim on its own.

**Note on memory:** this project originally targeted a 1GB AWS EC2 free-tier instance. In practice, the full stack (embedding model + reranker + quantized LLM + FastAPI/Streamlit overhead) measures closer to 1.7-1.9GB under normal use — the Kubernetes manifest (`k8s/medai.yaml`) reflects this with a 1.5Gi request / 3Gi limit rather than the original 1GB target.

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

First run downloads the GGUF model (~760MB), the embedding model, and the reranker from Hugging Face — all persisted across restarts via `HF_HOME` pointing into the mounted `app/models` volume.

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
- **AI/ML:** sentence-transformers (`all-MiniLM-L6-v2`), cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`), llama-cpp-python (Gemma-3-1B-it GGUF, Q4_K_M), RAGAS (evaluation)
- **Database:** SQLite + sqlite-vec
- **Orchestration:** Kubernetes, Docker, Docker Compose
- **Cloud:** AWS EC2 (Free Tier) — original deployment target; see Benchmarks for current local measurements
