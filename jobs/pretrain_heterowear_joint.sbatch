#!/bin/bash
#SBATCH -J pretrain_heterowear_joint
#SBATCH -o logs/pretrain_heterowear_joint_%j.out
#SBATCH -e logs/pretrain_heterowear_joint_%j.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1

# Adjust partition / account / QoS / time for your cluster (add e.g. "#SBATCH -A <account>").
# Submit from the repo root:  mkdir -p logs && sbatch jobs/pretrain_heterowear_joint.sbatch
set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

# Joint pretraining on all eight datasets (the corpus-level reference checkpoint; no exclusion).
python3 -u scripts/train/pretrain_heterowear.py \
  --mode joint --data_root data/processed --out_dir checkpoints/heterowear_joint \
  --steps 10000 --device cuda
