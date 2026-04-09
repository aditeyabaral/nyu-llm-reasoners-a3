"""
Show rollouts over training time by sampling from checkpoints at multiple steps.

Deliverable for Problem (grpo_train_loop): "a few example rollouts over time."

Each checkpoint is loaded, queried on the same fixed problems, then deleted to
free GPU memory before the next checkpoint is loaded.

Usage (auto-select evenly-spaced checkpoints from a run directory):
    uv run python -m student.sample_rollouts \
        --run-dir checkpoints/grpo/baselines/reinforce_with_baseline \
        --n-checkpoints 6 \
        --output-file out/rollouts_over_time.json

Usage (explicit checkpoints):
    uv run python -m student.sample_rollouts \
        --checkpoints \
            checkpoints/grpo/baselines/reinforce_with_baseline/step_10 \
            checkpoints/grpo/baselines/reinforce_with_baseline/step_100 \
            checkpoints/grpo/baselines/reinforce_with_baseline/final \
        --output-file out/rollouts_over_time.json
"""

import argparse
import json
import re
from pathlib import Path
import gc
import torch
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def _try_evaluate(expr: str):
    safe_chars = set("0123456789+-*/() .\t\n")
    if not all(c in safe_chars for c in expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def countdown_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
    gt = json.loads(ground_truth)
    target = int(gt["target"])
    allowed = sorted(gt["numbers"])

    if "<answer>" not in response or "</answer>" not in response:
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}

    answer_text = response.split("<answer>", 1)[-1].split("</answer>", 1)[0].strip()

    for line in reversed(answer_text.splitlines()):
        line = re.sub(r"^\s*Step\s*\d+\s*[:.]\s*", "", line).strip()
        if "=" not in line:
            continue
        lhs = line.rsplit("=", 1)[0].strip()
        result = _try_evaluate(lhs)
        if result is not None and abs(result - target) < 1e-6:
            nums_used = sorted(int(n) for n in re.findall(r"\b\d+\b", lhs))
            if nums_used == allowed:
                return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}

    return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}


def make_ground_truth(numbers: list[int], target: int) -> str:
    return json.dumps({"target": target, "numbers": sorted(numbers)})


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _step_number(path: Path) -> int:
    """Return sort key: step_N → N, final → infinity."""
    name = path.name
    if name == "final":
        return 10**9
    m = re.match(r"step_(\d+)$", name)
    return int(m.group(1)) if m else -1


def discover_checkpoints(run_dir: str, n: int) -> list[Path]:
    """
    Return `n` evenly-spaced step checkpoints from `run_dir` plus the `final`
    checkpoint (if it exists), sorted by step number.
    """
    run = Path(run_dir)
    step_dirs = sorted(
        [p for p in run.iterdir() if p.is_dir() and re.match(r"step_\d+$", p.name)],
        key=_step_number,
    )
    final_dir = run / "final"

    if not step_dirs:
        raise FileNotFoundError(f"No step_N subdirs found in {run_dir}")

    # Pick n-1 evenly-spaced steps (leave room for final)
    want = max(1, n - (1 if final_dir.exists() else 0))
    if want >= len(step_dirs):
        selected = list(step_dirs)
    elif want == 1:
        selected = [step_dirs[0]]
    else:
        indices = [round(i * (len(step_dirs) - 1) / (want - 1)) for i in range(want)]
        selected = [step_dirs[i] for i in sorted(set(indices))]

    if final_dir.exists():
        selected.append(final_dir)

    return selected


# ---------------------------------------------------------------------------
# Problems to evaluate
# ---------------------------------------------------------------------------

# A small fixed set so every checkpoint sees identical inputs — easier to
# compare qualitatively.  These span easy / medium / hard difficulty.
DEFAULT_PROBLEMS = [
    {"numbers": [3, 7, 12],    "target": 16},   # easy:   12 + 7 - 3 = 16
    {"numbers": [1, 5, 8, 9],  "target": 23},   # easy:   1 + 5 + 8 + 9 = 23
    {"numbers": [2, 5, 10, 25],"target": 100},  # medium: (25 - 5) * (10 / 2) = 100
    {"numbers": [2, 4, 6, 8],  "target": 48},   # medium: (8 + 4) * (6 - 2) = 48
    {"numbers": [11, 13, 17],  "target": 7},    # hard:   11 + 13 - 17 = 7
]


def build_prompt(numbers: list[int], target: int, template: str) -> str:
    problem = f"Using the numbers in the list {numbers}, create an equation that equals {target}."
    return template + "\n" + problem


# ---------------------------------------------------------------------------
# Per-checkpoint rollout
# ---------------------------------------------------------------------------

def run_checkpoint(
    ckpt_path: str,
    problems: list[dict],
    template: str,
    temperature: float,
    max_tokens: int,
    dtype: str = "bfloat16",
) -> list[dict]:
    """Load LLM, generate one response per problem, score, delete LLM."""
    prompts = [build_prompt(p["numbers"], p["target"], template) for p in problems]
    ground_truths = [make_ground_truth(p["numbers"], p["target"]) for p in problems]

    llm = LLM(model=ckpt_path, dtype=dtype)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=["</answer>"])
    outputs = llm.generate(prompts, params)
    del llm  # free GPU memory before next checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    results = []
    for prob, gt, out in zip(problems, ground_truths, outputs):
        response = out.outputs[0].text + "</answer>"
        reward = countdown_reward_fn(response, gt)
        results.append({
            "numbers": prob["numbers"],
            "target": prob["target"],
            "response": response,
            **reward,
        })
    return results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_checkpoint_results(ckpt_label: str, results: list[dict]) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  CHECKPOINT: {ckpt_label}")
    print(sep)
    for r in results:
        reward_tag = "✓" if r["reward"] == 1.0 else ("F" if r["format_reward"] == 0.0 else "✗")
        print(f"\n  [{reward_tag}] numbers={r['numbers']}, target={r['target']}  "
              f"(reward={r['reward']:.0f}, format={r['format_reward']:.0f}, "
              f"answer={r['answer_reward']:.0f})")
        # Print just the <answer> block
        resp = r["response"]
        if "<answer>" in resp:
            answer_block = resp.split("<answer>", 1)[1]
            # Indent for readability
            for line in ("<answer>" + answer_block).splitlines():
                print(f"    {line}")
        else:
            snippet = resp[:300] + ("…" if len(resp) > 300 else "")
            print(f"    {snippet}")

    mean_reward = sum(r["reward"] for r in results) / len(results)
    mean_answer = sum(r["answer_reward"] for r in results) / len(results)
    print(f"\n  → mean reward={mean_reward:.2f}  mean answer_reward={mean_answer:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample rollouts over training time from multiple GRPO checkpoints."
    )
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument(
        "--run-dir",
        help="Directory containing step_N and/or final subdirs (auto-selects checkpoints).",
    )
    ckpt_group.add_argument(
        "--checkpoints",
        nargs="+",
        help="Explicit list of checkpoint paths to evaluate.",
    )
    parser.add_argument(
        "--n-checkpoints",
        type=int,
        default=6,
        help="Number of checkpoints to pick when using --run-dir (includes final if present).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy, best for qualitative comparison).",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Save full results as JSON (useful for the writeup).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    template = Path("student/prompts/countdown.prompt").read_text().strip()
    problems = DEFAULT_PROBLEMS

    # Resolve checkpoint list
    if args.run_dir:
        ckpt_paths = discover_checkpoints(args.run_dir, args.n_checkpoints)
    else:
        ckpt_paths = [Path(p) for p in args.checkpoints]

    print(f"Evaluating {len(ckpt_paths)} checkpoints on {len(problems)} problems "
          f"(temperature={args.temperature}):")
    for p in ckpt_paths:
        print(f"  {p}")

    all_results = {}
    for ckpt in ckpt_paths:
        label = ckpt.name
        print(f"\nLoading {ckpt} …")
        results = run_checkpoint(
            str(ckpt), problems, template,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        all_results[label] = results
        print_checkpoint_results(label, results)

    # Cross-checkpoint summary table
    print("\n\n" + "=" * 72)
    print("  SUMMARY: mean answer_reward over time")
    print("=" * 72)
    print(f"  {'Checkpoint':<20}  {'Answer reward':>14}  {'Format reward':>14}")
    print(f"  {'-'*20}  {'-'*14}  {'-'*14}")
    for label, results in all_results.items():
        ar = sum(r["answer_reward"] for r in results) / len(results)
        fr = sum(r["format_reward"] for r in results) / len(results)
        print(f"  {label:<20}  {ar:>14.3f}  {fr:>14.3f}")

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "checkpoints": [str(p) for p in ckpt_paths],
                    "problems": problems,
                    "results": all_results,
                },
                f,
                indent=2,
            )
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
