#!/bin/bash
#SBATCH -J reexport_ppg_dalia
#SBATCH -o logs/reexport_ppg_dalia_%j.out
#SBATCH -e logs/reexport_ppg_dalia_%j.err
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -p general

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

python3 -u scripts/preprocess/04_export_raw_all_subjects.py --dataset ppg_dalia \
    --data_root data/raw --out_root data/processed --config configs/data/ppg_dalia.yaml
