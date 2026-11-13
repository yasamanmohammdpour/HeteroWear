#!/bin/bash
#SBATCH -J hw_finetune
#SBATCH -o logs/finetune_%j.out
#SBATCH -e logs/finetune_%j.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

python3 -u scripts/eval/finetune.py \
    --checkpoint checkpoints/pretrain/run1/step_5000.pt \
    --data_root data/processed \
    --out results/finetune/results.csv \
    --max_loso_folds 6 --epochs 5 --batch_size 32 --lr 1e-4 --max_steps 400 \
    --device cuda
