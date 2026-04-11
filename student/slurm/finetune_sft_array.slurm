#!/bin/bash

#SBATCH --job-name=sft_finetuning
#SBATCH --account=torch_pr_647_cds
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --open-mode=append
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/scratch/ab12057/nyu-llm-reasoners-a3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ab12057@nyu.edu
#SBATCH --array=0-3%4

set -e

# Background PID for the GPU keepalive process on cuda:0 (vLLM device sits idle during training).
GPU_KEEPALIVE_PID=""

# Always stop the keepalive when the job exits (success, failure, or cancel).
trap 'echo "Shutting down GPU keepalive..."; kill $GPU_KEEPALIVE_PID 2>/dev/null; wait $GPU_KEEPALIVE_PID 2>/dev/null' EXIT

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Current working directory: $(pwd)"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

nvidia-smi

# Keep cuda:0 (vLLM eval device) above HPC 50% utilization threshold during training.
# The SFT policy trains on cuda:1; cuda:0 is only used during periodic evaluation.
# Without this, HPC cancels the job after 2+ hours of 0% GPU 0 utilization.
echo "Starting GPU keepalive on cuda:0..."
KEEPALIVE_LOG=$(mktemp)
uv run python -c "
import signal, sys, time, torch
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
t = torch.randn(4096, 4096, device='cuda:0', dtype=torch.bfloat16)
print('GPU keepalive running on cuda:0...', flush=True)
while True:
    torch.matmul(t, t)
    time.sleep(0.0005)
" > "$KEEPALIVE_LOG" 2>&1 &
GPU_KEEPALIVE_PID=$!

echo "Waiting for GPU keepalive to initialize..."
until grep -q "GPU keepalive running" "$KEEPALIVE_LOG" 2>/dev/null; do
    sleep 1
done
echo "GPU keepalive confirmed running (PID: $GPU_KEEPALIVE_PID)."

# Dataset size configurations (None = full dataset)
MAX_EXAMPLES_CONFIGS=(128 256 512 1024 "None")

# Parse config for this job
MAX_EXAMPLES=${MAX_EXAMPLES_CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Training hyperparameters
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=56
NUM_EPOCHS=35
EVAL_POINTS=3
LEARNING_RATE=2e-5
GRAD_CLIP=1.0
WARMUP_RATIO=0.1
LOG_INTERVAL=1
SEED=42
GPU_MEMORY_UTILIZATION=0.85

# Paths
MODEL="Qwen/Qwen2.5-Math-1.5B"
TRAIN_PATH="data/intellect_math/train"
VAL_PATH="data/intellect_math/dev"
TEST_PATH="data/intellect_math/test"

# Resolve actual number of examples (full dataset requires counting)
if [ "$MAX_EXAMPLES" = "None" ]; then
    RUN_NAME="sft-full"
    OUTPUT_DIR="checkpoints/sft/full"
    MAX_EXAMPLES_ARG=""
    N_EXAMPLES=$(uv run python -c "from datasets import load_from_disk; print(len(load_from_disk('$TRAIN_PATH')))")
else
    RUN_NAME="sft-${MAX_EXAMPLES}"
    OUTPUT_DIR="checkpoints/sft/${MAX_EXAMPLES}"
    MAX_EXAMPLES_ARG="--max-train-examples $MAX_EXAMPLES"
    N_EXAMPLES=$MAX_EXAMPLES
fi

# Dynamically compute steps: multiply before dividing to avoid early truncation
EFFECTIVE_BATCH_SIZE=$(( BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS ))
TOTAL_STEPS=$(( (N_EXAMPLES * NUM_EPOCHS) / EFFECTIVE_BATCH_SIZE ))
EVAL_INTERVAL=$(( TOTAL_STEPS / EVAL_POINTS ))
if [ "$EVAL_INTERVAL" -eq 0 ]; then EVAL_INTERVAL=1; fi

echo "=== SFT EXPERIMENT ==="
echo "Job ID: $SLURM_ARRAY_TASK_ID"
echo "Max train examples: $MAX_EXAMPLES (actual: $N_EXAMPLES)"
echo "Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS | Total steps: $TOTAL_STEPS"
echo "Eval interval: $EVAL_INTERVAL"
echo "Output dir: $OUTPUT_DIR"
echo "Run name: $RUN_NAME"
echo ""

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

source ~/.bashrc

export HF_HOME=/scratch/$USER/.cache/huggingface

echo "Starting training..."
uv run student/sft_experiment.py \
    --model "$MODEL" \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --test-path "$TEST_PATH" \
    $MAX_EXAMPLES_ARG \
    --output-dir "$OUTPUT_DIR" \
    --total-steps $TOTAL_STEPS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
    --learning-rate $LEARNING_RATE \
    --grad-clip $GRAD_CLIP \
    --warmup-ratio $WARMUP_RATIO \
    --eval-interval $EVAL_INTERVAL \
    --log-interval $LOG_INTERVAL \
    --seed $SEED \
    --vllm-device cuda:0 \
    --policy-device cuda:1 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --wandb-project "sft-intellect-math" \
    --wandb-entity "csciga3033-llm-reasoners" \
    --wandb-name "$RUN_NAME"

echo "############### End Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"