import argparse
import json
import os
import sqlite3

from app.engine import DB_PATH
from app.main import ask_question


def sample_questions(n):
    db = sqlite3.connect(DB_PATH)
    rows = db.execute(
        """SELECT question, answer FROM medquad
           WHERE question IS NOT NULL AND TRIM(question) != ''
             AND answer IS NOT NULL AND TRIM(answer) != ''
           ORDER BY RANDOM() LIMIT ?""",
        [n],
    ).fetchall()
    db.close()
    return rows


def build_dataset(n):
    samples = sample_questions(n)
    records = []

    for i, (question, ground_truth) in enumerate(samples):
        print(f"[{i + 1}/{len(samples)}] {question[:60]}...")

        response = ask_question(question)
        source = response["source"]

        records.append(
            {
                "question": question,
                "answer": response["answer"],
                "contexts": [source] if source else [],
                "ground_truth": ground_truth,
            }
        )

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
