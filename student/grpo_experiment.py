import argparse
import json
import random
import re
import time
from pathlib import Path
from unittest.mock import patch

import torch
import wandb
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from vllm import LLM, SamplingParams

from student.sft import get_response_log_probs, tokenize_prompt_and_output
from student.grpo import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
)


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


def load_countdown_dataset(data_path: str) -> list[dict]:
    ds = load_from_disk(data_path)
    examples = []
    for ex in ds:
        numbers = ex.get("numbers", ex.get("nums", []))
        target = ex.get("target", ex.get("answer", 0))
        examples.append({"numbers": list(numbers), "target": int(target)})
    return examples


def format_countdown_prompt(example: dict, prompt_template: str) -> str:
    numbers = example["numbers"]
    target = example["target"]
    problem = f"Using the numbers in the list {numbers}, create an equation that equals {target}."
    return prompt_template + "\n" + problem


def make_ground_truth(example: dict) -> str:
    return json.dumps(
        {"target": example["target"], "numbers": sorted(example["numbers"])}
    )


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------


def init_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.8
) -> LLM:
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm(policy, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_countdown(llm: LLM, examples: list[dict], prompt_template: str) -> dict:
    prompts = [format_countdown_prompt(ex, prompt_template) for ex in examples]
    ground_truths = [make_ground_truth(ex) for ex in examples]

    params = SamplingParams(temperature=0.0, max_tokens=1024, stop=["</answer>"])
    outputs = llm.generate(prompts, params)

    total_reward = format_total = answer_total = 0
    for out, gt in zip(outputs, ground_truths):
        text = out.outputs[0].text + "</answer>"
        reward = countdown_reward_fn(text, gt)
        total_reward += reward["reward"]
        format_total += reward["format_reward"]
        answer_total += reward["answer_reward"]

    n = len(examples)
    return {
        "accuracy": total_reward / n,
        "format_accuracy": format_total / n,
        "answer_accuracy": answer_total / n,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO on Countdown")

    # Data
    parser.add_argument("--train-path", default="data/countdown/train")
    parser.add_argument("--val-path", default="data/countdown/dev")
    parser.add_argument("--test-path", default="data/countdown/test")

    # Model
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--output-dir", default="checkpoints/grpo")

    # GRPO core hyperparameters
    parser.add_argument("--n-grpo-steps", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--sampling-temperature", type=float, default=0.7)
    parser.add_argument("--sampling-min-tokens", type=int, default=4)
    parser.add_argument("--sampling-max-tokens", type=int, default=1024)

    # Loss type
    parser.add_argument(
        "--loss-type",
        default="reinforce_with_baseline",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    )

    # Normalization ablations
    parser.add_argument("--use-std-normalization", action="store_true", default=True)
    parser.add_argument(
        "--no-std-normalization", dest="use_std_normalization", action="store_false"
    )
    parser.add_argument(
        "--length-normalization",
        default="masked_mean",
        choices=["masked_mean", "masked_normalize"],
    )
    parser.add_argument(
        "--normalize-constant",
        type=float,
        default=None,
        help="Constant for masked_normalize; defaults to sampling-max-tokens when masked_normalize is used",
    )

    # Training
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)

    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--max-val-examples", type=int, default=256)

    # Logging
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm-device", type=str, default="cuda:0")
    parser.add_argument("--policy-device", type=str, default="cuda:1")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)

    return parser.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    device = torch.device(args.policy_device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derived hyperparameters
    assert args.rollout_batch_size % args.group_size == 0
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size
    train_batch_size = args.rollout_batch_size * args.epochs_per_rollout_batch
    assert train_batch_size % args.gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // args.gradient_accumulation_steps
    n_microbatches_per_rollout_batch = args.rollout_batch_size // micro_train_batch_size

    # Length normalization constant
    normalize_constant = None
    if args.length_normalization == "masked_normalize":
        normalize_constant = args.normalize_constant or float(args.sampling_max_tokens)

    print(f"n_prompts_per_rollout_batch: {n_prompts_per_rollout_batch}")
    print(f"train_batch_size: {train_batch_size}")
    print(f"micro_train_batch_size: {micro_train_batch_size}")
    print(
        f"n_microbatches_per_rollout_batch (per epoch): {n_microbatches_per_rollout_batch}"
    )

    # Load model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    policy.gradient_checkpointing_enable()

    # Load prompt template
    prompt_template = (Path("student/prompts/countdown.prompt")).read_text().strip()

    # Load datasets
    print("Loading datasets...")
    train_examples = load_countdown_dataset(args.train_path)
    val_examples = load_countdown_dataset(args.val_path)
    if args.max_val_examples:
        val_examples = val_examples[: args.max_val_examples]
    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    warmup_steps = int(args.warmup_ratio * args.n_grpo_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.n_grpo_steps,
    )

    # vLLM
    print("Initializing vLLM...")
    llm = init_vllm(
        args.model, args.vllm_device, args.seed, args.gpu_memory_utilization
    )

    sampling_params = SamplingParams(
        temperature=args.sampling_temperature,
        min_tokens=args.sampling_min_tokens,
        max_tokens=args.sampling_max_tokens,
        stop=["</answer>"],
    )

    # W&B
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            config=vars(args),
        )
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    # Initial eval
    print("\nRunning initial evaluation...")
    policy.eval()
    load_policy_into_vllm(policy, llm)
    eval_metrics = evaluate_countdown(llm, val_examples, prompt_template)
    tqdm.write(json.dumps({"step": 0, **eval_metrics}))
    eval_step = 0
    if args.wandb_project:
        wandb.log(
            {
                "eval/accuracy": eval_metrics["accuracy"],
                "eval/format_accuracy": eval_metrics["format_accuracy"],
                "eval/answer_accuracy": eval_metrics["answer_accuracy"],
                "eval_step": eval_step,
            }
        )

    # Training loop
    print(f"\nStarting GRPO for {args.n_grpo_steps} steps...")
    global_start = time.time()
    running = {
        "loss": 0.0,
        "reward": 0.0,
        "format_reward": 0.0,
        "answer_reward": 0.0,
        "entropy": 0.0,
        "grad_norm": 0.0,
        "clip_frac": 0.0,
        "count": 0,
        "opt_steps": 0,
    }

    progress = tqdm(range(args.n_grpo_steps), desc="GRPO")

    for grpo_step in progress:
        # ----------------------------------------------------------------
        # 1. Sample rollout questions
        # ----------------------------------------------------------------
        questions = random.sample(train_examples, n_prompts_per_rollout_batch)
        repeated_prompts = [
            format_countdown_prompt(q, prompt_template)
            for q in questions
            for _ in range(args.group_size)
        ]
        repeated_ground_truths = [
            make_ground_truth(q) for q in questions for _ in range(args.group_size)
        ]

        # ----------------------------------------------------------------
        # 2. Generate rollouts with vLLM
        # ----------------------------------------------------------------
        policy.eval()
        load_policy_into_vllm(policy, llm)
        outputs = llm.generate(repeated_prompts, sampling_params)
        rollout_responses = [out.outputs[0].text + "</answer>" for out in outputs]

        # ----------------------------------------------------------------
        # 3. Compute rewards and advantages
        # ----------------------------------------------------------------
        all_reward_dicts = [
            countdown_reward_fn(r, gt)
            for r, gt in zip(rollout_responses, repeated_ground_truths)
        ]
        step_format_reward = sum(d["format_reward"] for d in all_reward_dicts) / len(
            all_reward_dicts
        )
        step_answer_reward = sum(d["answer_reward"] for d in all_reward_dicts) / len(
            all_reward_dicts
        )
        _reward_iter = iter(all_reward_dicts)
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=lambda r, gt: next(_reward_iter),
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization,
        )

        # ----------------------------------------------------------------
        # 4. Tokenize all (prompt, response) pairs
        # ----------------------------------------------------------------
        tokenized = tokenize_prompt_and_output(
            repeated_prompts, rollout_responses, tokenizer
        )
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].float().to(device)

        # Shape for broadcasting: (rollout_batch_size, 1)
        adv_tensor = advantages.unsqueeze(1).to(device)
        raw_tensor = raw_rewards.unsqueeze(1).to(device)

        # ----------------------------------------------------------------
        # 5. Compute old log-probs (needed for grpo_clip or off-policy)
        # ----------------------------------------------------------------
        need_old_log_probs = (
            args.loss_type == "grpo_clip" or args.epochs_per_rollout_batch > 1
        )
        if need_old_log_probs:
            policy.eval()
            with torch.inference_mode():
                old_result = get_response_log_probs(policy, input_ids, labels)
                old_log_probs_all = old_result["log_probs"].detach()
        else:
            old_log_probs_all = None

        # ----------------------------------------------------------------
        # 6. Training over epochs
        # ----------------------------------------------------------------
        epoch_loss = epoch_entropy = epoch_clip_frac = 0.0
        epoch_grad_norm = 0.0
        opt_steps_this_rollout = 0

        for epoch in range(args.epochs_per_rollout_batch):
            optimizer.zero_grad()

            # Shuffle microbatch order within each epoch
            indices = list(range(args.rollout_batch_size))

            for mb_idx in range(n_microbatches_per_rollout_batch):
                mb_start = mb_idx * micro_train_batch_size
                mb_end = mb_start + micro_train_batch_size
                mb_ids = indices[mb_start:mb_end]

                mb_input_ids = input_ids[mb_ids]
                mb_labels = labels[mb_ids]
                mb_response_mask = response_mask[mb_ids]
                mb_adv = adv_tensor[mb_ids]
                mb_raw = raw_tensor[mb_ids]
                mb_old_lp = (
                    old_log_probs_all[mb_ids] if old_log_probs_all is not None else None
                )

                policy.train()
                result = get_response_log_probs(
                    policy, mb_input_ids, mb_labels, return_token_entropy=True
                )
                mb_log_probs = result["log_probs"]
                mb_entropy = result["token_entropy"]

                loss, meta = grpo_microbatch_train_step(
                    policy_log_probs=mb_log_probs,
                    response_mask=mb_response_mask,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    loss_type=args.loss_type,
                    raw_rewards=mb_raw,
                    advantages=mb_adv,
                    old_log_probs=mb_old_lp,
                    cliprange=args.cliprange,
                    normalize_constant=normalize_constant,
                )

                with torch.no_grad():
                    mean_entropy = (
                        (mb_entropy * mb_response_mask).sum()
                        / mb_response_mask.sum().clamp(min=1)
                    ).item()
                    epoch_entropy += mean_entropy

                if "is_clipped" in meta:
                    clip_frac = (
                        (meta["is_clipped"].float() * mb_response_mask).sum()
                        / mb_response_mask.sum().clamp(min=1)
                    ).item()
                    epoch_clip_frac += clip_frac

                epoch_loss += loss.item()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), args.grad_clip
            )
            optimizer.step()
            scheduler.step()
            epoch_grad_norm += grad_norm.item()
            opt_steps_this_rollout += 1

        n_mb_total = n_microbatches_per_rollout_batch * args.epochs_per_rollout_batch

        # ----------------------------------------------------------------
        # 7. Accumulate metrics
        # ----------------------------------------------------------------
        running["loss"] += epoch_loss / n_mb_total
        running["reward"] += reward_meta["mean_reward"]
        running["format_reward"] += step_format_reward
        running["answer_reward"] += step_answer_reward
        running["entropy"] += epoch_entropy / n_mb_total
        running["grad_norm"] += epoch_grad_norm / opt_steps_this_rollout
        running["clip_frac"] += epoch_clip_frac / n_mb_total
        running["count"] += 1

        # ----------------------------------------------------------------
        # 8. Logging
        # ----------------------------------------------------------------
        step = grpo_step + 1
        if step % args.log_interval == 0:
            c = running["count"]
            current_lr = scheduler.get_last_lr()[0]
            log = {
                "train/loss": running["loss"] / c,
                "train/mean_reward": running["reward"] / c,
                "train/format_reward": running["format_reward"] / c,
                "train/answer_reward": running["answer_reward"] / c,
                "train/token_entropy": running["entropy"] / c,
                "train/grad_norm": running["grad_norm"] / c,
                "train/clip_frac": running["clip_frac"] / c,
                "train/learning_rate": current_lr,
                "train/elapsed": time.time() - global_start,
            }
            tqdm.write(
                json.dumps(
                    {
                        "step": step,
                        "reward": round(running["reward"] / c, 4),
                        "loss": round(running["loss"] / c, 4),
                        "grad_norm": round(running["grad_norm"] / c, 4),
                    }
                )
            )
            if args.wandb_project:
                wandb.log({**log, "train_step": step})
            for k in running:
                running[k] = 0.0

        # ----------------------------------------------------------------
        # 9. Periodic evaluation
        # ----------------------------------------------------------------
        if step % args.eval_interval == 0:
            tqdm.write(f"\nEvaluating at step {step}...")
            policy.eval()
            load_policy_into_vllm(policy, llm)
            eval_metrics = evaluate_countdown(llm, val_examples, prompt_template)
            eval_step += 1
            tqdm.write(json.dumps({"step": step, **eval_metrics}))
            if args.wandb_project:
                wandb.log(
                    {
                        "eval/accuracy": eval_metrics["accuracy"],
                        "eval/format_accuracy": eval_metrics["format_accuracy"],
                        "eval/answer_accuracy": eval_metrics["answer_accuracy"],
                        "eval_step": eval_step,
                    }
                )

            policy.save_pretrained(str(output_dir / f"step_{step}"))
            tokenizer.save_pretrained(str(output_dir / f"step_{step}"))

        progress.set_postfix({"reward": f"{reward_meta['mean_reward']:.3f}"})

    progress.close()

    # Final save
    policy.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\nSaved final model to {output_dir / 'final'}")

    # Final test evaluation
    print("\nRunning final test evaluation...")
    policy.eval()
    test_examples = load_countdown_dataset(args.test_path)
    load_policy_into_vllm(policy, llm)
    test_metrics = evaluate_countdown(llm, test_examples, prompt_template)
    tqdm.write(json.dumps({"final_test": test_metrics}))
    if args.wandb_project:
        wandb.log(
            {
                "test/accuracy": test_metrics["accuracy"],
                "test/format_accuracy": test_metrics["format_accuracy"],
                "test/answer_accuracy": test_metrics["answer_accuracy"],
            }
        )
        wandb.finish()

    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args)
