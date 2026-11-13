#!/bin/bash
#SBATCH -J finetune_ts_transformer
#SBATCH -o logs/finetune_ts_transformer_%j.out
#SBATCH -e logs/finetune_ts_transformer_%j.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -p htc
#SBATCH --gres=gpu:a100:1

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

python3 -u scripts/eval/finetune_ts_transformer.py \
    --checkpoint checkpoints/ts_pretrain/run1/step_5000.pt \
    --processed_root data/processed \
    --out results/ts_ssl/results.csv \
    --max_loso_folds 6 --warmup_steps 100 --total_steps 800 \
    --batch_size 256 --device cuda
