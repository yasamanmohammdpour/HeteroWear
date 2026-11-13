#!/bin/bash
#SBATCH -J robustness_curve_heterowear
#SBATCH -o logs/robustness_hw_%j.out
#SBATCH -e logs/robustness_hw_%j.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

mkdir -p results/heterowear
for ds in pamap2 wisdm hhar harth uci_har; do
  echo "=== $ds ==="
  python3 -u scripts/eval/robustness_curve_heterowear.py \
    --dataset "$ds" \
    --lodo_checkpoint "checkpoints/heterowear_lodo/${ds}_step_10000.pt" \
    --data_root data/processed \
    --out "results/heterowear/${ds}_robustness.csv" \
    --device cuda
done
