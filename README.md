# 🩺 MedAI-RAG: Production-Grade Medical Assistant

High-Performance RAG on Low-Performance Hardware

A Retrieval-Augmented Generation (RAG) system engineered to provide medical insights using a vector-search database. This project is a case study in System Optimization, successfully deploying a Large Language Model and a Vector DB within an extreme 1GB RAM constraint.

## 🏗️ System Architecture

The application is architected as a microservices-based system orchestrated by Kubernetes (K8s):

- Retrieval Engine: SQLite with the sqlite-vec extension for high-speed, local vector similarity searches.

- Inference Layer: llama-cpp-python running a quantized Gemma-3-1B model, optimized for CPU execution.

- Backend API: FastAPI serving as the orchestration layer between the vector store and the LLM.

- Frontend: Streamlit provides a clean, responsive medical chat interface.

## 🛠️ Engineering Highlights (The 1GB RAM Challenge)

Deploying a modern AI stack on an EC2 T2.Micro (Free Tier) required advanced DevOps strategies:

- Universal Pathing: Implemented environment-agnostic configuration using pathlib and OS environment variables, allowing seamless transitions between Local, Docker, and Kubernetes environments.

- Persistent Storage: Configured Kubernetes Persistent Volume Claims (PVC) to ensure the 100MB+ medical vector database survives pod restarts and remains consistent.

- Cold-Start Optimization: Moved model initialization to the global module scope to prevent "Lazy Loading" latency and OOM (Out of Memory) spikes during request handling.

- K8s Resource Governance: Hard-capped memory requests and limits (700Mi / 900Mi) to ensure system stability on low-memory nodes.

- Smart Ingestion: Developed an idempotent ingestion pipeline that detects existing data to prevent redundant, high-CPU processing on pod reboots.

## 🚀 Deployment

Kubernetes (Production)

The cluster setup includes a Deployment, Service (LoadBalancer), and Persistent Volume Claim.

```bash
   # 1. Apply the manifest
   kubectl apply -f k8s/medai.yaml

   # 2. Initialize the Vector DB inside the pod
   kubectl exec -it <backend-pod-name> -- python -m app.ingest_data
   ```

Local Development (Docker):

```bash
docker-compose up --build
```

## 📈 Tech Stack

- Language: Python 3.10

- AI/ML: sentence-transformers (all-MiniLM-L6-v2), llama-cpp-python (Gemma-3-1B GGUF)

- Database: SQLite + sqlite-vec

- Orchestration: Kubernetes, Docker, Docker Compose

- Cloud: AWS EC2 (Free Tier)

## ⚠️ Medical Disclaimer

**IMPORTANT:** This project is a Proof of Concept for technical demonstration only. The AI can hallucinate. Always consult a certified medical professional for health advice.
