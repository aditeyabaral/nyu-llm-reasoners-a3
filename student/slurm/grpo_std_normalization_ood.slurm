#!/bin/bash

#SBATCH --job-name=grpo_std_norm
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:15:00
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:a100:2
#SBATCH --open-mode=append
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --chdir=/scratch/ab12057/nyu-llm-reasoners-a3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ab12057@nyu.edu
#SBATCH --array=0-1%2

set -e

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
nvidia-smi

# std normalization configurations: 0=with_std, 1=no_std
STD_FLAGS=("--use-std-normalization" "--no-std-normalization")
STD_FLAG=${STD_FLAGS[$SLURM_ARRAY_TASK_ID]}
STD_NAMES=("with_std" "no_std")
STD_NAME=${STD_NAMES[$SLURM_ARRAY_TASK_ID]}

RUN_NAME="grpo-stdnorm-${STD_NAME}"
OUTPUT_DIR="checkpoints/grpo/std_norm/${STD_NAME}"

echo "=== GRPO STD NORMALIZATION ==="
echo "Std normalization flag: $STD_FLAG"
echo "Run name: $RUN_NAME"
echo ""

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

source ~/.bashrc
export HF_HOME=/scratch/$USER/.cache/huggingface

uv run student/grpo_experiment.py \
    --model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
    --train-path "data/countdown/dataset/train" \
    --val-path "data/countdown/dataset/dev" \
    --test-path "data/countdown/dataset/test" \
    --output-dir "$OUTPUT_DIR" \
    --n-grpo-steps 350 \
    --rollout-batch-size 16 \
    --group-size 8 \
    --epochs-per-rollout-batch 1 \
    --gradient-accumulation-steps 16 \
    --learning-rate 3e-5 \
    --loss-type "reinforce_with_baseline" \
    $STD_FLAG \
    --length-normalization "masked_mean" \
    --sampling-temperature 0.7 \
    --sampling-max-tokens 1024 \
    --grad-clip 1.0 \
    --warmup-ratio 0.0 \
    --eval-interval 10 \
    --log-interval 5 \
    --seed 42 \
    --vllm-device cuda:0 \
    --policy-device cuda:1 \
    --gpu-memory-utilization 0.85 \
    --wandb-project "grpo-countdown" \
    --wandb-entity "csciga3033-llm-reasoners" \
    --wandb-name "$RUN_NAME"

echo "############### End Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
