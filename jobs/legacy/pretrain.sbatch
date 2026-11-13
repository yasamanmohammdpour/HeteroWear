#!/bin/bash
#SBATCH -J hw_pretrain
#SBATCH -o logs/pretrain_%j.out
#SBATCH -e logs/pretrain_%j.err
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

python3 -u scripts/train/pretrain.py \
    --datasets ppg_dalia wesad pamap2 mmash wisdm hhar harth uci_har \
    --data_root data/processed \
    --out_dir checkpoints/pretrain/run1 \
    --d_model 128 --n_patches 8 --samples_per_patch 16 \
    --n_layers 4 --n_heads 4 \
    --batch_size 64 --steps 5000 --lr 3e-4 \
    --mask_ratio 0.4 --lambda_msm 1.0 \
    --p_modality 0.25 --p_location 0.25 --p_channel 0.2 \
    --log_every 20 --ckpt_every 500 \
    --device cuda
