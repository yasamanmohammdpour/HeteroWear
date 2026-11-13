#!/bin/bash
#SBATCH -J device_holdout
#SBATCH -o logs/device_holdout_%j.out
#SBATCH -e logs/device_holdout_%j.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1

# Adjust partition / account / QoS / time for your cluster (add e.g. "#SBATCH -A <account>").
# Submit from the repo root:  mkdir -p logs && sbatch jobs/device_holdout.sbatch
set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

# Table V: held-out-device transfer (phone <-> watch) for WISDM and HHAR.
# NOTE: HHAR is split by device CLASS (smartphone/smartwatch); see README "Known gaps".
mkdir -p results/heterowear
for DS in wisdm hhar; do
  python3 -u scripts/eval/device_holdout_eval.py \
    --dataset "$DS" \
    --lodo_checkpoint "checkpoints/heterowear_lodo/${DS}_step_10000.pt" \
    --data_root data/processed \
    --out "results/heterowear/${DS}_device_holdout.csv" --device cuda
done
