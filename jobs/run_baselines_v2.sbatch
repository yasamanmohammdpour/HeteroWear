#!/bin/bash
#SBATCH -J baselines_v2
#SBATCH -o logs/baselines_v2_%j.out
#SBATCH -e logs/baselines_v2_%j.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

python3 -u scripts/analysis/run_har_baselines.py \
    --processed_root data/processed --out results/baselines/har_baselines_v2.csv \
    --max_loso_folds 6
