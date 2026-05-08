FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    sentence-transformers \
    sqlite-vec \
    requests \
    streamlit \
    huggingface_hub \
    datasets

RUN pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu    

COPY . .

EXPOSE 8000
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "app/main.py"]