#!/bin/bash
#SBATCH -J verify_ppg_dalia
#SBATCH -o logs/verify_ppg_dalia_%j.out
#SBATCH -e logs/verify_ppg_dalia_%j.err
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 00:10:00
#SBATCH -p general

set -euo pipefail
cd "${HETEROWEAR_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
source activate "${HETEROWEAR_ENV:-heterowear}"

python3 -u -c "
import pickle
from pathlib import Path

bad = []
for f in sorted(Path('data/processed/ppg_dalia/all').glob('*.pkl')):
    try:
        with open(f, 'rb') as fh:
            w = pickle.load(fh)
        print(f'{f.name}: {len(w)} windows OK', flush=True)
    except Exception as e:
        bad.append((f.name, str(e)))
        print(f'{f.name}: FAILED ({e})', flush=True)

print()
print(f'{len(bad)} bad file(s)' if bad else 'all files loaded cleanly')
"
