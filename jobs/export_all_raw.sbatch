#!/bin/bash
#SBATCH -J hw_export_all
#SBATCH -o logs/export_all_%j.out
#SBATCH -e logs/export_all_%j.err
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -p general

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

for ds in ppg_dalia wesad pamap2 mmash wisdm hhar harth uci_har; do
    echo "=== $ds (full subject set) ==="
    python3 -u scripts/preprocess/04_export_raw_all_subjects.py --dataset "$ds" \
        --data_root data/raw --out_root data/processed --config "configs/data/$ds.yaml" \
        || echo "[$ds] FAILED — continuing to next dataset"
    echo
done
