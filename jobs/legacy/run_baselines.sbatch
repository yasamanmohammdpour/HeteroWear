#!/bin/bash
#SBATCH -J hw_baselines
#SBATCH -o logs/baselines_%j.out
#SBATCH -e logs/baselines_%j.err
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -p general

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

python3 -u scripts/analysis/run_har_baselines.py \
    --processed_root data/processed --out results/baselines/har_baselines.csv \
    --max_loso_folds 6
