#!/bin/bash
#SBATCH -J hw_adapter_test
#SBATCH -o logs/adapter_test_%j.out
#SBATCH -e logs/adapter_test_%j.err
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 00:45:00
#SBATCH -p general

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"
mkdir -p logs data/processed

for ds in ppg_dalia pamap2 wisdm hhar harth wesad mmash uci_har; do
    echo "=== $ds ==="
    python3 -u scripts/preprocess/03_preprocess.py --dataset "$ds" \
        --data_root data/raw \
        --out_root data/processed \
        --config "configs/data/$ds.yaml" --limit_subjects 1 \
        || echo "[$ds] FAILED — continuing to next dataset"
    echo
done
