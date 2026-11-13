#!/bin/bash
#SBATCH -J robustness_curve_v2
#SBATCH -o logs/robustness_curve_v2_%j.out
#SBATCH -e logs/robustness_curve_v2_%j.err
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

python3 -u scripts/eval/robustness_curve.py \
    --checkpoint checkpoints/pretrain/run2/step_10000.pt \
    --data_root data/processed \
    --out results/robustness/curve_v2.csv \
    --device cuda
