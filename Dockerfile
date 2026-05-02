# Stage 1: Builder
FROM python:3.10-slim as builder
RUN apt-get update && apt-get install -y build-essential gcc
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Final Image
FROM python:3.10-slim
WORKDIR /app
# Copy only the installed packages
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

# Clean up apt to save space
RUN apt-get purge -y --auto-remove build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]