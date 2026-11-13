#!/bin/bash
#SBATCH -J finetune_heterowear
#SBATCH -o logs/finetune_heterowear_%A_%a.out
#SBATCH -e logs/finetune_heterowear_%A_%a.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-4
# Adjust partition / account / QoS / time for your cluster (add e.g. "#SBATCH -A <account>").
# Submit from the repo root:  mkdir -p logs && sbatch jobs/finetune_heterowear.sbatch
set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

# Table IV: HW-Scratch vs HW-LODO under identical subject folds / schedule.
# PAMAP2 and HHAR run exhaustive nine-fold LOSO.
DATASETS=(pamap2 wisdm hhar harth uci_har)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
mkdir -p results/heterowear

python3 -u scripts/eval/finetune_heterowear.py \
  --dataset "$DS" \
  --lodo_checkpoint "checkpoints/heterowear_lodo/${DS}_step_10000.pt" \
  --data_root data/processed \
  --out "results/heterowear/${DS}_results.csv" --device cuda
