import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "app"))
from app.engine import search_db, retrieve_contexts, generate_answer, DB_PATH


def sample_questions(n):
    db = sqlite3.connect(DB_PATH)
    rows = db.execute(
        "SELECT question, answer FROM medquad ORDER BY RANDOM() LIMIT ?", [n]
    ).fetchall()
    db.close()
    return rows


def build_dataset(n):
    samples = sample_questions(n)
    records = []

    for i, (question, ground_truth) in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {question[:60]}...")

        top1 = search_db(question)
        contexts = retrieve_contexts(question, top_n=3)
        context_for_generation = top1["text"] if top1 else "No data found."
        answer = generate_answer(question, context_for_generation)

        records.append({
            "question": question,
            "answer": answer,
            "contexts": contexts if contexts else [context_for_generation],
            "ground_truth": ground_truth,
        })

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out", default="eval/eval_dataset.json")
    args = parser.parse_args()

    data = build_dataset(args.n)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nWrote {len(data)} eval examples to {args.out}")