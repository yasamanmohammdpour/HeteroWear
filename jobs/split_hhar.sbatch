#!/bin/bash
#SBATCH -J hhar_split
#SBATCH -o logs/hhar_split_%j.out
#SBATCH -e logs/hhar_split_%j.err
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -p general

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"
python3 -u scripts/preprocess/01_split_hhar_by_user.py --hhar_dir data/raw/hhar
