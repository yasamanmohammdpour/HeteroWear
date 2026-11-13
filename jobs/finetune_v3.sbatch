#!/bin/bash
#SBATCH -J finetune_v3
#SBATCH -o logs/finetune_v3_%j.out
#SBATCH -e logs/finetune_v3_%j.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

python3 -u scripts/eval/finetune.py \
    --checkpoint checkpoints/pretrain/run2/step_10000.pt \
    --data_root data/processed \
    --out results/finetune/results_v3.csv \
    --max_loso_folds 6 --epochs 15 --batch_size 32 \
    --head_lr 1e-3 --encoder_lr 5e-5 --warmup_frac 0.15 \
    --max_steps 1200 --eval_every 200 \
    --device cuda
