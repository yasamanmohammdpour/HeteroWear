#!/bin/bash
#SBATCH -J pretrain_ts_transformer
#SBATCH -o logs/pretrain_ts_transformer_%j.out
#SBATCH -e logs/pretrain_ts_transformer_%j.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:15:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

python3 -u scripts/train/pretrain_ts_transformer.py \
    --datasets pamap2 wisdm hhar harth uci_har \
    --data_root data/processed \
    --out checkpoints/ts_pretrain/run1 \
    --d_model 64 --n_layers 3 --n_heads 4 --patch_len 16 \
    --batch_size 64 --steps 5000 --lr 3e-4 --warmup_steps 300 --mask_ratio 0.4 \
    --log_every 50 --ckpt_every 1000 \
    --device cuda
