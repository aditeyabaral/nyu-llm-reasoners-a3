#!/bin/bash

#SBATCH --job-name=evaluate
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:2
#SBATCH --open-mode=append
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/scratch/ab12057/nyu-llm-reasoners-a3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ab12057@nyu.edu

set -e

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
nvidia-smi

# Model to evaluate (override with --export MODEL=... or edit here)
MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
MAX_EXAMPLES=${MAX_EXAMPLES:-500}
OUTPUT_TAG=${OUTPUT_TAG:-"baseline"}

echo "=== EVALUATION ==="
echo "Model: $MODEL"
echo "Max examples: $MAX_EXAMPLES"
echo "Output tag: $OUTPUT_TAG"
echo ""

mkdir -p logs
mkdir -p results

source ~/.bashrc
export HF_HOME=/scratch/$USER/.cache/huggingface

uv run student/evaluate.py \
    --model "$MODEL" \
    --max-examples $MAX_EXAMPLES \
    --intellect-path "data/intellect_math/test" \
    --intellect-save-path "results/intellect_${OUTPUT_TAG}.json" \
    --math-save-path "results/math_${OUTPUT_TAG}.json" \
    --gpu-memory-utilization 0.85

echo "############### End Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
