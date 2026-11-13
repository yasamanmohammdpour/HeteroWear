#!/bin/bash
#SBATCH -J seed_stability
#SBATCH -o logs/seed_stability_%j.out
#SBATCH -e logs/seed_stability_%j.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 72:00:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1

# Adjust partition / account / QoS / time for your cluster (add e.g. "#SBATCH -A <account>").
# Submit from the repo root:  mkdir -p logs && sbatch jobs/seed_stability.sbatch
set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

# Pretraining-seed stability: the whole LODO pipeline (pretrain + fine-tune) for three seeds.
mkdir -p results/heterowear/seeds
for seed in 42 123 7; do
  echo "=== seed $seed ==="
  python3 -u scripts/train/pretrain_heterowear.py \
    --mode lodo --data_root data/processed \
    --out_dir "checkpoints/heterowear_lodo_seed${seed}" \
    --steps 10000 --seed "$seed" --device cuda

  for ds in pamap2 wisdm hhar harth uci_har; do
    python3 -u scripts/eval/finetune_heterowear.py --skip_scratch \
      --dataset "$ds" \
      --lodo_checkpoint "checkpoints/heterowear_lodo_seed${seed}/${ds}_step_10000.pt" \
      --data_root data/processed \
      --out "results/heterowear/seeds/seed${seed}_${ds}.csv" --seed "$seed" --device cuda
  done
done
