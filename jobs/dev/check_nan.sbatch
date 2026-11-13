#!/bin/bash
#SBATCH -J check_nan
#SBATCH -o logs/check_nan_%j.out
#SBATCH -e logs/check_nan_%j.err
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 00:15:00
#SBATCH -p general

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

python3 -u -c "
import pickle, numpy as np
from pathlib import Path

for ds in ['ppg_dalia', 'wesad', 'pamap2', 'mmash', 'wisdm', 'hhar', 'harth', 'uci_har']:
    d = Path(f'data/processed/{ds}/all')
    if not d.exists():
        print(f'{ds}: no data'); continue
    files = sorted(d.glob('*.pkl'))[:2]
    any_nan = False
    total_checked = 0
    for f in files:
        with open(f, 'rb') as fh:
            windows = pickle.load(fh)
        for w in windows[:50]:
            for k in w['channel_keys']:
                arr = w[f'sig__{k}']
                total_checked += 1
                if np.isnan(arr).any():
                    any_nan = True
    print(f'{ds}: NaN found = {any_nan}  (sampled {total_checked} channel-arrays from {len(files)} subjects)', flush=True)
"
