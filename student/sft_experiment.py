"""SFT Training script for Qwen2.5-Math-1.5B on Prime Intellect dataset."""

import argparse
import json
import time
from itertools import cycle
from pathlib import Path
from unittest.mock import patch

import torch
import wandb
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from vllm import LLM

from student.sft import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from student.evaluate import evaluate


class IntellectDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_intellect_dataset(data_path, max_examples=None):
    dataset = load_from_disk(data_path)
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    examples = []
    for ex in tqdm(dataset, desc="Loading dataset", leave=False):
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assistant_msg = next(
            (m["content"] for m in msgs if m["role"] == "assistant"), ""
        )
        prompt = sys_msg + "\n\n" + user_msg if sys_msg else user_msg
        examples.append({"prompt": prompt, "output": assistant_msg})
    return examples


def collate_fn(batch, tokenizer):
    prompt_strs = [ex["prompt"] for ex in batch]
    output_strs = [ex["output"] for ex in batch]
    return tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)


def init_vllm(model_id, device, seed, gpu_memory_utilization=0.4):
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


def load_policy_into_vllm(policy, llm):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def run_eval(llm, val_examples):
    prompts = [ex["prompt"] for ex in val_examples]
    gts = [ex["ground_truth"] for ex in val_examples]
    return evaluate(llm, prompts, gts)


def run_math_eval(llm, math_prompts, math_gts):
    return evaluate(llm, math_prompts, math_gts)


def run_all_evals(policy, llm, val_examples, math_prompts, math_gts):
    """Load policy into vLLM once, then run both eval sets."""
    load_policy_into_vllm(policy, llm)
    intellect_acc = run_eval(llm, val_examples)
    math_acc = run_math_eval(llm, math_prompts, math_gts)
    return intellect_acc, math_acc


def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT Training on Prime Intellect dataset"
    )

    # Data
    parser.add_argument("--train-path", default="data/intellect_math/train")
    parser.add_argument("--val-path", default="data/intellect_math/dev")
    parser.add_argument("--test-path", default="data/intellect_math/test")
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Limit training examples (128/256/512/1024/None for full)",
    )

    # Model
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--output-dir", default="checkpoints/sft")

    # Training
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)

    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=100)

    # Logging
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name. If set, enables W&B logging.",
    )
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm-device", type=str, default="cuda:0")
    parser.add_argument("--policy-device", type=str, default="cuda:0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)

    return parser.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.policy_device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    policy.gradient_checkpointing_enable()

    # Load prompt template and MATH dataset once (reused at every eval)
    math_prompt_template = (
        (Path("student/prompts/intellect.prompt")).read_text().strip()
    )

    math_ds = load_dataset("hiyouga/math12k", split="test")
    math_prompts, math_gts = [], []
    for ex in tqdm(math_ds, desc="Loading MATH dataset", leave=False):
        math_prompts.append(math_prompt_template + "\n\n" + ex["problem"])
        math_gts.append(ex["answer"])

    # Load data
    print("Loading dataset...")
    train_examples = load_intellect_dataset(
        args.train_path, max_examples=args.max_train_examples
    )
    val_raw = load_from_disk(args.val_path)
    val_examples = []
    for ex in tqdm(val_raw, desc="Loading val examples", leave=False):
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prompt = sys_msg + "\n\n" + user_msg if sys_msg else user_msg
        val_examples.append(
            {"prompt": prompt, "ground_truth": ex.get("ground_truth", "")}
        )

    print(f"Train examples: {len(train_examples)}, Val examples: {len(val_examples)}")

    train_dataset = IntellectDataset(train_examples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    warmup_steps = int(args.warmup_ratio * args.total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.total_steps,
    )

    # vLLM
    print("Initializing vLLM for evaluation...")
    llm = init_vllm(
        args.model, args.vllm_device, args.seed, args.gpu_memory_utilization
    )

    # WandB setup
    if args.wandb_project is not None:
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

    # Training loop
    print(f"\nStarting SFT training for {args.total_steps} steps...")
    step = 0
    eval_step = 0
    running_loss = 0.0
    # running_entropy = 0.0
    running_grad_norm = 0.0
    running_count = 0  # microbatch count (for loss/entropy averaging)
    running_opt_steps = 0  # optimizer step count (for grad_norm averaging)
    global_start = time.time()

    print("\nRunning initial evaluation...")
    policy.eval()
    intellect_acc, math_acc = run_all_evals(
        policy, llm, val_examples, math_prompts, math_gts
    )
    tqdm.write(
        json.dumps(
            {
                "step": 0,
                "val_intellect_accuracy": round(intellect_acc, 4),
                "val_math_accuracy": round(math_acc, 4),
            }
        )
    )
    if args.wandb_project is not None:
        wandb.log(
            {
                "eval/intellect_accuracy": intellect_acc,
                "eval/math_accuracy": math_acc,
                "eval_step": 0,
            }
        )

    optimizer.zero_grad()
    progress_bar = tqdm(total=args.total_steps, desc="Training")

    for microbatch_idx, batch in enumerate(cycle(train_loader)):
        if step >= args.total_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        response_mask = batch["response_mask"].to(device)

        # Forward pass
        policy.train()
        result = get_response_log_probs(
            model=policy,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=False,
        )
        log_probs = result["log_probs"]

        # Mean entropy over response tokens only
        # token_entropy = result["token_entropy"]
        # n_response_tokens = response_mask.sum().clamp(min=1)
        # mean_entropy = (
        #     (token_entropy * response_mask).sum() / n_response_tokens
        # ).detach()

        # Microbatch train step
        loss, _ = sft_microbatch_train_step(
            policy_log_probs=log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            normalize_constant=1.0,
        )

        running_loss += loss.item()
        # running_entropy += mean_entropy.item()
        running_count += 1

        # Optimizer step every gradient_accumulation_steps microbatches
        if (microbatch_idx + 1) % args.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), args.grad_clip
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            running_grad_norm += grad_norm.item()
            running_opt_steps += 1

            # Logging
            if step % args.log_interval == 0:
                avg_loss = running_loss / running_count
                # avg_entropy = running_entropy / running_count
                avg_grad_norm = running_grad_norm / running_opt_steps
                elapsed = time.time() - global_start
                log = {
                    "train/loss": avg_loss,
                    # "train/entropy": avg_entropy,
                    "train/grad_norm": avg_grad_norm,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/step": step,
                    "train/elapsed": elapsed,
                }
                tqdm.write(
                    json.dumps(
                        {
                            "step": step,
                            "train_loss": round(avg_loss, 4),
                            # "train_entropy": round(avg_entropy, 4),
                            "grad_norm": round(avg_grad_norm, 4),
                        }
                    )
                )
                if args.wandb_project is not None:
                    wandb.log({**log, "train_step": step})
                running_loss = 0.0
                # running_entropy = 0.0
                running_grad_norm = 0.0
                running_count = 0
                running_opt_steps = 0

            # Evaluation
            if step % args.eval_interval == 0:
                tqdm.write(f"\nEvaluating at step {step}...")
                policy.eval()
                intellect_acc, math_acc = run_all_evals(
                    policy, llm, val_examples, math_prompts, math_gts
                )
                eval_step += 1
                tqdm.write(
                    json.dumps(
                        {
                            "step": step,
                            "val_intellect_accuracy": round(intellect_acc, 4),
                            "val_math_accuracy": round(math_acc, 4),
                        }
                    )
                )
                if args.wandb_project is not None:
                    wandb.log(
                        {
                            "eval/intellect_accuracy": intellect_acc,
                            "eval/math_accuracy": math_acc,
                            "eval_step": eval_step,
                        }
                    )

                # Save checkpoint
                policy.save_pretrained(str(output_dir / f"step_{step}"))
                tokenizer.save_pretrained(str(output_dir / f"step_{step}"))
                tqdm.write(f"Saved checkpoint: {output_dir / f'step_{step}'}")

            progress_bar.update(1)
            progress_bar.set_postfix({"step": step, "loss": f"{loss.item():.4f}"})

    # Save final model
    policy.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\nSaved final model to {output_dir / 'final'}")

    # Final test evaluation on both test sets
    print("\nRunning final test evaluation...")
    policy.eval()

    # Intellect test set
    test_raw = load_from_disk(args.test_path)
    test_examples = []
    for ex in tqdm(test_raw, desc="Loading test examples", leave=False):
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prompt = sys_msg + "\n\n" + user_msg if sys_msg else user_msg
        test_examples.append(
            {"prompt": prompt, "ground_truth": ex.get("ground_truth", "")}
        )

    load_policy_into_vllm(policy, llm)
    intellect_test_acc = evaluate(
        llm,
        [ex["prompt"] for ex in test_examples],
        [ex["ground_truth"] for ex in test_examples],
        save_path=str(output_dir / "intellect_test_results.json"),
    )
    math_test_acc = evaluate(
        llm,
        math_prompts,
        math_gts,
        save_path=str(output_dir / "math_test_results.json"),
    )

    tqdm.write(
        json.dumps(
            {
                "final_intellect_test_accuracy": round(intellect_test_acc, 4),
                "final_math_test_accuracy": round(math_test_acc, 4),
            }
        )
    )
    if args.wandb_project is not None:
        wandb.log(
            {
                "test/intellect_accuracy": intellect_test_acc,
                "test/math_accuracy": math_test_acc,
            }
        )
        wandb.finish()

    progress_bar.close()
    print("Training complete!")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args)
