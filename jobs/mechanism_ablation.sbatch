#!/bin/bash
#SBATCH -J mechanism_ablation
#SBATCH -o logs/mechanism_ablation_%j.out
#SBATCH -e logs/mechanism_ablation_%j.err
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH -p public
#SBATCH --gres=gpu:a100:1

# Adjust partition / account / QoS / time for your cluster (add e.g. "#SBATCH -A <account>").
# Submit from the repo root:  mkdir -p logs && sbatch jobs/mechanism_ablation.sbatch
set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

# Table VII: HHAR mechanism ablation. Every config is LODO-pretrained with HHAR excluded, then
#   (a) fine-tuned under 9-fold LOSO            -> results/heterowear/ablation/hhar_<cfg>.csv
#   (b) evaluated under 50% sensor removal      -> results/heterowear/ablation/hhar_<cfg>_robustness.csv
CKPT_ROOT=checkpoints/heterowear_ablation
OUT=results/heterowear/ablation
mkdir -p "$CKPT_ROOT" "$OUT"

declare -A CONFIGS=(
  ["msm_only"]="--lambda_cons 0 --p_modality 0 --p_location 0 --p_channel 0"   # MSM only
  ["no_dropout"]="--p_modality 0 --p_location 0 --p_channel 0"                  # MSM + consistency
  ["no_consistency"]="--lambda_cons 0"                                          # MSM + structured dropout
  ["full"]=""                                                                   # full objective
  ["no_context"]="--disable_context"                                            # full, minus sensor context
)

for name in msm_only no_dropout no_consistency full no_context; do
  echo "=== ablation config: $name ==="
  python3 -u scripts/train/pretrain_heterowear.py \
    --mode lodo --only_dataset hhar \
    --data_root data/processed --out_dir "$CKPT_ROOT/$name" \
    --steps 10000 --device cuda ${CONFIGS[$name]}

  python3 -u scripts/eval/finetune_heterowear.py --skip_scratch \
    --dataset hhar --lodo_checkpoint "$CKPT_ROOT/$name/hhar_step_10000.pt" \
    --data_root data/processed --out "$OUT/hhar_${name}.csv" --device cuda

  python3 -u scripts/eval/robustness_curve_heterowear.py --skip_scratch \
    --dataset hhar --lodo_checkpoint "$CKPT_ROOT/$name/hhar_step_10000.pt" \
    --data_root data/processed --out "$OUT/hhar_${name}_robustness.csv" --device cuda
done
