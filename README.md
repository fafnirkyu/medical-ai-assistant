# 🩺 MedAI-RAG: Local-First Medical RAG Assistant

A proof-of-concept Retrieval-Augmented Generation (RAG) assistant for general medical Q&A. It searches a local vector database, reranks candidate answers, and generates a response with a quantized local LLM. No external inference API is required after the models and dataset are downloaded.

## ⚠️ Medical Disclaimer

**IMPORTANT:** This project is a proof of concept for technical demonstration only. The AI can hallucinate or retrieve imprecise context. It is not a diagnostic tool. Always consult a qualified medical professional for health advice.

## 📚 Data Source

Medical Q&A data comes from [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD). The ingestion script streams it from Hugging Face and indexes questions with `all-MiniLM-L6-v2` in SQLite + `sqlite-vec`. The previous local benchmark used a database of roughly 47,000 Q&A pairs; a fresh checkout must run ingestion before asking questions.

## 🏗️ System Architecture

The backend is one FastAPI process with distinct retrieval, reranking, and inference steps; the Streamlit frontend is a separate process:

- **Retrieval Engine:** SQLite with the `sqlite-vec` extension retrieves the top-10 candidates by embedding similarity (bi-encoder search, `all-MiniLM-L6-v2`).
- **Reranking Layer:** A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores those 10 candidates by jointly evaluating each (query, answer) pair, selecting the most relevant one.
- **Inference Layer:** `llama-cpp-python` runs Gemma-3-1B-it Q4_K_M by default on CPU. `MODEL_PATH` can select a local GGUF instead.
- **Backend API:** FastAPI orchestrates retrieval → reranking → prompt construction → generation.
- **Frontend:** Streamlit displays the answer, retrieved source answer, and a heuristic retrieval relevance score. That score is not medical-answer confidence.

## 🛠️ Engineering Highlights

- **Two-stage retrieval:** Embedding search narrows roughly 47,000 Q&A pairs to 10 candidates; a cross-encoder reranks those candidates by query-answer relevance.
- **Chat-completion prompting:** Uses `llama-cpp-python`'s chat-completion API. The prompt asks for answers grounded in the retrieved source, but this does not guarantee against hallucinations.
- **Universal pathing:** `pathlib` keeps configuration portable across local, Docker, and Kubernetes environments.
- **Persistent storage:** Docker Compose mounts separate database and model directories. The local Kubernetes example uses separate PVCs; `HF_HOME` points into the model volume.
- **Cold-start behavior:** Models initialize at module import rather than on the first request.
- **Transactional ingestion:** The ingestion script rebuilds both tables with matching row IDs in one SQLite transaction. If streaming or embedding fails, SQLite rolls back the rebuild. This is a local-data setup script, not an online production migration.
- **Retrieval fallback:** Retrieval skips candidates with empty answer fields. If none remain, the API returns an unable-to-verify response without calling the LLM.

## 🧪 Historical RAG quality evaluation (pre-change)

An earlier version of the pipeline was evaluated on 50 sampled MedQuAD questions with RAGAS and a locally run `qwen2.5:7b-instruct` judge via Ollama. These results are a historical diagnostic baseline, **not measurements of the current code** or a clinical accuracy score. The smaller local judge makes the metrics directionally useful rather than ground truth.

| Metric | All 50 questions | Retrieval succeeded (28/50) |
| --- | --- | --- |
| Faithfulness | 0.560 | 0.817 |
| Answer relevancy | 0.412 | 0.712 |
| Context precision | 0.264 | 0.507 |
| Context recall | 0.708 | 0.655 |

The [per-example results](eval/ragas_results_detailed.csv) show that 22 of 50 records (44%) used `"No data found."` as their context. In that earlier flow, generation still ran on the placeholder and could produce fluent but unsupported medical text. This exposed a retrieval and no-source handling problem; the run does not establish its root cause.

The current `/ask` route instead returns an explicit unable-to-verify response without calling the LLM when retrieval supplies no source, and an offline test covers that behavior. Transactional ingestion and the evaluation harness also changed. **The 50-question retrieval success rate and RAGAS metrics have not been re-measured on this version**, so no improvement in those scores is claimed.

To run a new evaluation against the current local API behavior, install `requirements-eval.txt`, make the local Ollama judge available, and run:

```bash
python eval/build_eval_set.py --n 50
python eval/run_ragas.py --model qwen2.5:7b-instruct
```

The newer harness records the single retrieved source actually supplied to generation, or an empty context for an abstention. It writes summary files and `eval/ragas_results_detailed.csv`; save the historical CSV elsewhere first if you want to compare both runs.

## 📊 Benchmarks

These are **historical** measurements from `benchmark.py` against a local Docker deployment (Windows/Docker Desktop, WSL2 backend), before the current safety and API changes. Re-run the benchmark on the final version before citing them as current performance.

| Metric | v1 (embedding-only) | v2 (+ reranking) |
| --- | --- | --- |
| Cold start | ~17–24s | ~15s |
| Peak RAM (unconstrained) | ~1.6 GB | ~1.7–1.9 GB |
| Query latency (p50) | 2.96s | ~2.0–2.8s |
| Query latency (p95) | 7.35s | ~7.3–7.7s |
| Requests benchmarked | 18 | 18 |

Reranking added a small RAM cost but did not meaningfully regress latency in these runs. LLM generation dominated response time. The cold-start difference may be run-to-run variance rather than a real improvement.

**Retrieval scores:** v1 used raw embedding distance; v2 transforms the cross-encoder score with a sigmoid. The numbers are not comparable across versions and have not been calibrated against medical-answer correctness. The API calls this field `retrieval_score`.

**Memory:** The project originally targeted a 1 GB AWS EC2 instance. The historical local stack measured closer to 1.7–1.9 GB. The local `k8s/medai-local.yaml` requests 1.5 GiB and limits the backend to 3 GiB. New measurements are needed after the current changes.

### Reproducing the benchmark

```bash
docker compose down
docker compose up -d
python benchmark.py --url http://localhost:8000 --cold-start --container medai-backend-1
```

The script polls startup readiness, samples container RAM, and writes `benchmark_results.md` and `benchmark_results.json`. Start it immediately after starting the container; otherwise the cold-start figure misses most of startup. This is a small six-question latency sample, not a clinical accuracy evaluation.

## 🚀 Deployment

### Local development with Docker Compose

Build and start the image once:

```bash
docker compose up --build -d
```

For subsequent starts, avoid rebuilding the image:

```bash
docker compose up --no-build -d
```

The first run downloads the default Gemma GGUF (~764 MB), embedding model, and reranker from Hugging Face. Compose persists them under `app/models`; `HF_HOME` points into that mounted directory. The image excludes local databases and model caches. To use another GGUF, set `MODEL_PATH` to its path *inside the backend container*. `MODEL_FILENAME` must be a filename in `MODEL_REPO`, not a CLI command.

On a fresh checkout, ingest the vector database once:

```bash
docker compose exec backend python -m app.ingest_data
```

This rebuilds the database. If you already have an indexed `app/data/medical_data.db`, skip ingestion unless you intend to replace it; back up the database before a planned rebuild.

The Streamlit UI is available at `http://localhost:8501`, and the API at `http://localhost:8000`. Before ingestion, `/ask` abstains because the local database is unavailable.

### Local Kubernetes deployment example

`k8s/medai-local.yaml` uses the locally built image `medai-rag:local`. Load that image into your local cluster (for example, `kind load docker-image medai-rag:local`) or change the manifest to a versioned image you published. This example is **not** a claim of a currently running production service.

```bash
kubectl apply -f k8s/medai-local.yaml
kubectl exec -it <backend-pod-name> -- python -m app.ingest_data
kubectl port-forward svc/frontend-service 8501:80
```

The root-level `medai.yaml` is the historical AWS-oriented manifest, not the current local quick-start. It remains separate so the local example does not overwrite that earlier deployment configuration.

## 📈 Tech Stack

- **Language:** Python 3.10
- **AI/ML:** sentence-transformers (`all-MiniLM-L6-v2`), cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`), llama-cpp-python (Gemma-3-1B-it GGUF, Q4_K_M)
- **Database:** SQLite + sqlite-vec
- **Deployment:** Docker, Docker Compose, local Kubernetes example
- **Cloud history:** AWS EC2 was an earlier deployment target; the measurements above are from local Docker.

## Evaluation and limitations

The optional `eval/` scripts build a sampled MedQuAD Q&A set and run RAGAS with a local Ollama judge. Install `requirements-eval.txt` and provide an Ollama model to use them; they are not part of deployment or CI. The current harness evaluates the answer against the single retrieved source supplied to the model, or records an empty context when the API abstains.

The shared local LLM serializes generation requests. No clinician review, validated answer-confidence score, source-level citation system, authentication/TLS/rate limiting, or production monitoring is implemented. See the medical disclaimer above.

Run offline API tests with:

```bash
python -m unittest discover -s tests -v
```

These tests do not download models or assess clinical accuracy.
