"""
Runs RAGAS 0.1.21 metrics against a pre-built eval dataset, using a local
Ollama model as the judge.

Usage:
    python eval/run_ragas.py --dataset eval/eval_dataset.json --model qwen2.5:7b-instruct
"""
import argparse
import json
from datetime import datetime, timezone

from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall


def load_dataset(path):
    with open(path) as f:
        records = json.load(f)
    return Dataset.from_dict({
        "question": [r["question"] for r in records],
        "answer": [r["answer"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })


def print_summary(scores):
    print("\n" + "=" * 52)
    print("  MEDAI-RAG RAGAS EVALUATION SUMMARY")
    print("=" * 52)
    for metric, value in scores.items():
        print(f"  {metric:<20} {value:.3f}")
    print("=" * 52 + "\n")


def write_markdown(scores, model_name, path="eval/ragas_results.md"):
    lines = [f"_Judge model: `{model_name}` (local, via Ollama)_", "", "| Metric | Score |", "|---|---|"]
    for metric, value in scores.items():
        lines.append(f"| {metric} | {value:.3f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown table written to {path}")


def write_json(scores, n, model_name, path="eval/ragas_results.json"):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_examples": n,
        "judge_model": model_name,
        "scores": scores,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Raw results written to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/eval_dataset.json")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--ollama-host", default="http://host.docker.internal:11434")
    args = parser.parse_args()

    ds = load_dataset(args.dataset)

    judge_llm = LangchainLLMWrapper(ChatOpenAI(
    model=args.model,
    base_url=f"{args.ollama_host}/v1",
    api_key="ollama",  # Ollama ignores this, but the client requires a non-empty string
    temperature=0,
     ))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    scores = {k: float(v) for k, v in result.items()}

    print_summary(scores)
    write_markdown(scores, args.model)
    write_json(scores, n=len(ds), model_name=args.model)

    df = result.to_pandas()
    df.to_csv("eval/ragas_results_detailed.csv", index=False)
    print("Per-example breakdown written to eval/ragas_results_detailed.csv")