#!/bin/bash
#SBATCH -J pretrain_heterowear_lodo
#SBATCH -o logs/pretrain_heterowear_lodo_%A_%a.out
#SBATCH -e logs/pretrain_heterowear_lodo_%A_%a.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-4
# Adjust partition / account / QoS / time for your cluster (add e.g. "#SBATCH -A <account>").
# Submit from the repo root:  mkdir -p logs && sbatch jobs/pretrain_heterowear_lodo.sbatch
set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

# Leave-one-dataset-out SSL pretraining (Sec. V-C): checkpoint k excludes downstream dataset k,
# and is trained on the other seven datasets only.
DATASETS=(pamap2 wisdm hhar harth uci_har)
DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

python3 -u scripts/train/pretrain_heterowear.py \
  --mode lodo --only_dataset "$DS" \
  --data_root data/processed --out_dir checkpoints/heterowear_lodo \
  --steps 10000 --batch_size 64 --lr 3e-4 --warmup_steps 400 --device cuda
