"""
MedAI-RAG Benchmark Suite
Measures cold-start time, peak container RAM, and query latency (p50/p95),
then prints a clean summary and writes results to benchmark_results.json
and benchmark_results.md (a ready-to-paste README/LinkedIn table).

Usage:
    python benchmark.py --url http://localhost:8000 --container medai-backend-1 --cold-start

Requires: pip install requests
Docker must be on PATH (used for peak RAM sampling via `docker stats`).
"""

import argparse
import json
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone

import requests

DEFAULT_QUERIES = [
    "What are the symptoms of type 2 diabetes?",
    "What is the recommended dosage for ibuprofen in adults?",
    "What are common drug interactions with warfarin?",
    "What are the symptoms of a heart attack?",
    "What causes high blood pressure?",
    "How is asthma diagnosed?",
]


def measure_cold_start(health_url: str, timeout: int = 180, poll_interval: float = 1.0) -> float | None:
    """Poll /health from the moment this is called. Start it right after `docker compose up`."""
    print(f"[cold start] polling {health_url} ...")
    start = time.perf_counter()

    while time.perf_counter() - start < timeout:
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                elapsed = time.perf_counter() - start
                print(f"[cold start] ready in {elapsed:.2f}s")
                return elapsed
        except requests.exceptions.RequestException:
            pass
        time.sleep(poll_interval)

    print("[cold start] timed out waiting for /health")
    return None


def sample_docker_ram(container: str, stop_event, interval: float = 1.0) -> dict | None:
    """Poll `docker stats` for a running container until stop_event is set. Returns peak/mean MB usage."""
    samples_mb = []

    while not stop_event.is_set():
        try:
            out = subprocess.run(
                ["docker", "stats", container, "--no-stream", "--format", "{{.MemUsage}}"],
                capture_output=True, text=True, timeout=5,
            )
            usage_str = out.stdout.strip().split("/")[0].strip()
            value = float("".join(c for c in usage_str if c.isdigit() or c == "."))
            unit = "".join(c for c in usage_str if c.isalpha())
            mb = value * 1024 if unit.upper().startswith("G") else value
            samples_mb.append(mb)
        except (subprocess.SubprocessError, ValueError, IndexError):
            pass
        stop_event.wait(interval)

    if not samples_mb:
        print("[ram] no samples collected — check the container name with `docker compose ps`")
        return None

    result = {"peak_mb": max(samples_mb), "mean_mb": statistics.mean(samples_mb), "n_samples": len(samples_mb)}
    print(f"[ram] peak: {result['peak_mb']:.1f} MB | mean: {result['mean_mb']:.1f} MB (n={result['n_samples']})")
    return result


def measure_query_latency(base_url: str, queries: list[str], n_repeats: int = 3) -> dict:
    """Hits GET /ask?query=... for each query n_repeats times."""
    latencies = []
    confidences = []

    print(f"[latency] running {len(queries)} queries x {n_repeats} repeats...")
    for query in queries:
        for _ in range(n_repeats):
            start = time.perf_counter()
            try:
                r = requests.get(f"{base_url}/ask", params={"query": query}, timeout=120)
                if r.status_code == 200:
                    latencies.append((time.perf_counter() - start) * 1000)
                    data = r.json()
                    if "confidence" in data:
                        confidences.append(data["confidence"])
                else:
                    print(f"  non-200 ({r.status_code}) for: {query[:50]}")
            except requests.exceptions.RequestException as e:
                print(f"  failed: {query[:50]}... ({e})")

    if not latencies:
        raise RuntimeError("No successful requests — check the URL and that the backend is reachable")

    latencies.sort()
    result = {
        "n_requests": len(latencies),
        "mean_ms": statistics.mean(latencies),
        "p50_ms": statistics.median(latencies),
        "p95_ms": latencies[int(len(latencies) * 0.95) - 1],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }
    if confidences:
        result["mean_confidence"] = statistics.mean(confidences)

    print(f"[latency] mean: {result['mean_ms']:.0f}ms | p50: {result['p50_ms']:.0f}ms | p95: {result['p95_ms']:.0f}ms")
    return result


def print_summary(cold_start, ram, latency):
    print("\n" + "=" * 52)
    print("  MEDAI-RAG BENCHMARK SUMMARY")
    print("=" * 52)
    if cold_start is not None:
        print(f"  Cold start          {cold_start:6.2f} s")
    if ram is not None:
        print(f"  Peak RAM            {ram['peak_mb']:6.1f} MB")
        print(f"  Mean RAM            {ram['mean_mb']:6.1f} MB")
    print(f"  Query latency (p50) {latency['p50_ms']:6.0f} ms")
    print(f"  Query latency (p95) {latency['p95_ms']:6.0f} ms")
    print(f"  Mean latency         {latency['mean_ms']:6.0f} ms")
    print(f"  Requests measured    {latency['n_requests']:6d}")
    if "mean_confidence" in latency:
        print(f"  Mean confidence      {latency['mean_confidence'] * 100:5.1f} %")
    print("=" * 52 + "\n")


def write_markdown(cold_start, ram, latency, path="benchmark_results.md"):
    lines = [
        "| Metric | Value |",
        "|---|---|",
    ]
    if cold_start is not None:
        lines.append(f"| Cold start | {cold_start:.2f}s |")
    if ram is not None:
        lines.append(f"| Peak RAM | {ram['peak_mb']:.0f} MB |")
    lines.append(f"| Query latency (p50) | {latency['p50_ms']:.0f} ms |")
    lines.append(f"| Query latency (p95) | {latency['p95_ms']:.0f} ms |")
    lines.append(f"| Requests benchmarked | {latency['n_requests']} |")
    if "mean_confidence" in latency:
        lines.append(f"| Mean confidence | {latency['mean_confidence'] * 100:.0f}% |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown table written to {path}")


def write_json(cold_start, ram, latency, path="benchmark_results.json"):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cold_start_s": cold_start,
        "ram": ram,
        "latency": latency,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Raw results written to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Backend base URL, e.g. http://localhost:8000")
    parser.add_argument("--container", help="Docker container name for RAM sampling, e.g. medai-backend-1")
    parser.add_argument("--cold-start", action="store_true", help="Measure cold start (run right after startup)")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per query (default 3)")
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    args = parser.parse_args()

    cold_start_result = measure_cold_start(f"{args.url}/health") if args.cold_start else None

    ram_result = None
    ram_thread = None
    stop_event = threading.Event()
    ram_holder = {}

    if args.container:
        print(f"[ram] starting background sampling of '{args.container}' (runs alongside the latency test)...")

        def _ram_worker():
            ram_holder["result"] = sample_docker_ram(args.container, stop_event)

        ram_thread = threading.Thread(target=_ram_worker)
        ram_thread.start()

    latency_result = measure_query_latency(args.url, args.queries, n_repeats=args.repeats)

    if ram_thread is not None:
        stop_event.set()
        ram_thread.join()
        ram_result = ram_holder.get("result")

    print_summary(cold_start_result, ram_result, latency_result)
    write_markdown(cold_start_result, ram_result, latency_result)
    write_json(cold_start_result, ram_result, latency_result)