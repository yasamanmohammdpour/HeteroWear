#!/bin/bash
#SBATCH -J ppg_dalia_test
#SBATCH -o logs/ppg_dalia_test_%j.out
#SBATCH -e logs/ppg_dalia_test_%j.err
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -p general

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

python3 scripts/preprocess/03_preprocess.py --dataset ppg_dalia \
    --data_root data/raw --out_root data/processed \
    --config configs/data/ppg_dalia.yaml --limit_subjects 1
