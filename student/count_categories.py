#!/usr/bin/env python3
"""Parse math_baseline.json and print category counts + examples."""

import json
import sys
from collections import defaultdict

def parse(path: str):
    with open(path) as f:
        data = json.load(f)

    accuracy = data.get("accuracy")
    results = data["results"]
    total = len(results)

    counts = defaultdict(int)
    examples = defaultdict(list)

    for r in results:
        cat = r["category"]
        counts[cat] += 1
        if len(examples[cat]) < 3:          # keep up to 3 examples per category
            examples[cat].append(r)

    print(f"Total examples : {total}")
    print(f"Overall accuracy: {accuracy:.4f}  ({int(round(accuracy * total))}/{total})\n")
    print(f"{'Cat':>4}  {'Description':<40}  {'Count':>6}  {'%':>6}")
    print("-" * 62)
    descs = {1: "format=1, answer=1 (correct)",
             2: "format=1, answer=0 (wrong answer)",
             3: "format=0, answer=0 (no \\boxed{})"}
    for cat in sorted(counts):
        pct = 100.0 * counts[cat] / total
        print(f"{cat:>4}  {descs[cat]:<40}  {counts[cat]:>6}  {pct:>5.1f}%")

    for cat in sorted(examples):
        print(f"\n{'='*70}")
        print(f"Category {cat} examples")
        print('='*70)
        for i, ex in enumerate(examples[cat], 1):
            problem = ex["prompt"].split("\n\n")[-1].strip()[:200]
            response_tail = ex["response"].strip()[-300:]
            gt = ex["ground_truth"]
            print(f"\n--- Example {i} (index {ex['index']}) ---")
            print(f"Problem (truncated): {problem}")
            print(f"Response tail      : {response_tail}")
            print(f"Ground truth       : {gt}")
            print(f"format_reward={ex['format_reward']}  answer_reward={ex['answer_reward']}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/math_baseline.json"
    parse(path)