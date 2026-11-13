#!/bin/bash
#SBATCH -J linear_probe
#SBATCH -o logs/linear_probe_%j.out
#SBATCH -e logs/linear_probe_%j.err
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

python3 -u scripts/eval/linear_probe.py \
    --checkpoint checkpoints/pretrain/run1/step_5000.pt \
    --data_root data/processed \
    --out results/linear_probe/results.csv \
    --max_loso_folds 6 --device cuda
