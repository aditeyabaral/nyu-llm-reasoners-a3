#!/bin/bash

#SBATCH --job-name=sample_rollouts
#SBATCH --account=torch_pr_647_cds
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --open-mode=append
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/scratch/ab12057/nyu-llm-reasoners-a3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ab12057@nyu.edu

set -e

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Current working directory: $(pwd)"
nvidia-smi

RUN_DIR="checkpoints/grpo/baselines/reinforce_with_baseline"
N_CHECKPOINTS=6
TEMPERATURE=0.0
MAX_TOKENS=512
OUTPUT_TAG="reinforce_with_baseline"

echo "=== ROLLOUT SAMPLING ==="
echo "Run dir:       $RUN_DIR"
echo "N checkpoints: $N_CHECKPOINTS"
echo "Temperature:   $TEMPERATURE"
echo "Max tokens:    $MAX_TOKENS"
echo "Output tag:    $OUTPUT_TAG"
echo ""

mkdir -p logs
mkdir -p out

source ~/.bashrc
export HF_HOME=/scratch/$USER/.cache/huggingface

uv run python -m student.sample_rollouts \
    --run-dir "$RUN_DIR" \
    --n-checkpoints $N_CHECKPOINTS \
    --temperature $TEMPERATURE \
    --max-tokens $MAX_TOKENS \
    --output-file "out/rollouts_${OUTPUT_TAG}.json"

echo "############### End Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
