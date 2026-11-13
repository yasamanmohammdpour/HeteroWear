#!/bin/bash
#SBATCH -J linear_probe_v2
#SBATCH -o logs/linear_probe_v2_%j.out
#SBATCH -e logs/linear_probe_v2_%j.err
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

python3 -u scripts/eval/linear_probe.py \
    --checkpoint checkpoints/pretrain/run2/step_10000.pt \
    --data_root data/processed \
    --out results/linear_probe/results_v2.csv \
    --max_loso_folds 6 --device cuda
