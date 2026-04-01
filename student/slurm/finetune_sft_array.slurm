#!/bin/bash

#SBATCH --job-name=nyu_llm_reasoners_sft
#SBATCH --account=torch_pr_647_cds
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --open-mode=append
#SBATCH --error=logs/%j.err
#SBATCH --output=logs/%j.out
#SBATCH --chdir=/scratch/ab12057/nyu-llm-reasoners-a3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ab12057@nyu.edu
#SBATCH --array=0-4%5

set -e

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Current working directory: $(pwd)"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

nvidia-smi

# Dataset size configurations (None = full dataset)
MAX_EXAMPLES_CONFIGS=(128 256 512 1024 "None")
# 25 epochs proportional to dataset size (effective batch size = 32)
TOTAL_STEPS_CONFIGS=(100 200 400 800 7813)
# ~10 eval points per run (total_steps / 10)
EVAL_INTERVAL_CONFIGS=(10 20 40 80 781)

# Parse config for this job
MAX_EXAMPLES=${MAX_EXAMPLES_CONFIGS[$SLURM_ARRAY_TASK_ID]}
TOTAL_STEPS=${TOTAL_STEPS_CONFIGS[$SLURM_ARRAY_TASK_ID]}
EVAL_INTERVAL=${EVAL_INTERVAL_CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Training hyperparameters
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=2e-5
GRAD_CLIP=1.0
LOG_INTERVAL=10
SEED=42
GPU_MEMORY_UTILIZATION=0.5

# Paths
MODEL="Qwen/Qwen2.5-Math-1.5B"
TRAIN_PATH="data/intellect_math/train"
VAL_PATH="data/intellect_math/dev"
TEST_PATH="data/intellect_math/test"

if [ "$MAX_EXAMPLES" = "None" ]; then
    RUN_NAME="sft-full"
    OUTPUT_DIR="checkpoints/sft/full"
    MAX_EXAMPLES_ARG=""
else
    RUN_NAME="sft-${MAX_EXAMPLES}"
    OUTPUT_DIR="checkpoints/sft/${MAX_EXAMPLES}"
    MAX_EXAMPLES_ARG="--max-train-examples $MAX_EXAMPLES"
fi

echo "=== SFT EXPERIMENT ==="
echo "Job ID: $SLURM_ARRAY_TASK_ID"
echo "Max train examples: $MAX_EXAMPLES"
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
    --eval-interval $EVAL_INTERVAL \
    --log-interval $LOG_INTERVAL \
    --seed $SEED \
    --vllm-device cuda:0 \
    --policy-device cuda:0 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --wandb-project "nyu-llm-reasoners-sft" \
    --wandb-name "$RUN_NAME"

echo "############### End Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"