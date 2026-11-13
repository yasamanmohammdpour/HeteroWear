#!/usr/bin/env python3
"""
Raw-data audit: run this BEFORE preprocessing a freshly downloaded dataset.

For every subject the registered adapter can read, report per channel the
sample count, duration, implied sampling rate (median timestamp delta),
NaN fraction, and timestamp monotonicity. These are exactly the checks that
surfaced the data-quality problems described in Sec. III of the paper
(PAMAP2 NaN packets, WISDM non-monotonic timestamps, UCI-HAR overlapping
session clocks, HHAR multi-year timestamp jumps).

Usage:
    python3 scripts/preprocess/00_audit_raw_data.py --dataset pamap2 \
        --data_root data/raw --config configs/data/pamap2.yaml --limit_subjects 2
    python3 scripts/preprocess/00_audit_raw_data.py --dataset all --data_root data/raw

Exit status is 1 if any hard problem (NaNs, non-monotonic or >30-day timestamps,
implied rate off by >2x from the adapter's declared native_hz) is found.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.registry import DATASET_REGISTRY, get_adapter

MAX_SESSION_SEC = 30 * 86400.0


def load_config(path: Path, data_root: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f.read().replace("${DATA_ROOT}", str(data_root)))


def audit_channel(key, arr, meta, t):
    problems = []
    n = len(arr)
    nan_frac = float(np.isnan(arr).mean()) if n else 1.0
    t = np.asarray(t, dtype=np.float64)
    dur = float(t[-1] - t[0]) if n > 1 else 0.0
    dt = np.diff(t)
    implied_hz = float(1.0 / np.median(dt)) if n > 1 and np.median(dt) > 0 else float("nan")
    if nan_frac > 0:
        problems.append(f"NaN fraction {nan_frac:.4f}")
    if n > 1 and (dt < 0).any():
        problems.append(f"{int((dt < 0).sum())} non-monotonic timestamps")
    if dur > MAX_SESSION_SEC:
        problems.append(f"implausible duration {dur / 86400:.1f} days")
    if np.isfinite(implied_hz) and meta.native_hz > 0 and not (0.5 <= implied_hz / meta.native_hz <= 2.0):
        problems.append(f"implied {implied_hz:.1f} Hz vs declared {meta.native_hz:.1f} Hz")
    line = (f"    {key:34s} {meta.modality.value:5s} {meta.location.value:12s} {meta.device.value:12s} "
            f"n={n:>9d} dur={dur:>9.0f}s implied={implied_hz:>8.2f}Hz nan={nan_frac:.4f}")
    return line, problems


def audit_dataset(name, data_root, config_dir, limit):
    cfg_path = config_dir / f"{name}.yaml"
    cfg = load_config(cfg_path, data_root) if cfg_path.exists() else {}
    adapter = get_adapter(name, data_root / name, cfg)
    subjects = list(adapter.iter_subjects())
    print(f"\n=== {name}: {len(subjects)} subjects on disk ===")
    if not subjects:
        print("  no subjects found - check the download / path")
        return 0
    n_problems = 0
    for sid in subjects[:limit] if limit else subjects:
        try:
            raw = adapter.load_subject_raw(sid)
        except Exception as e:  # report, keep auditing the rest
            print(f"  [{sid}] FAILED to load: {type(e).__name__}: {e}")
            n_problems += 1
            continue
        labels = raw.get("activity_labels")
        print(f"  [{sid}] channels={len(raw['channels'])} "
              f"activity_segments={'n/a' if labels is None else len(labels)}")
        for key, (arr, meta, t) in raw["channels"].items():
            line, problems = audit_channel(key, arr, meta, t)
            print(line + ("   <-- " + "; ".join(problems) if problems else ""))
            n_problems += len(problems)
    return n_problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY) + ["all"])
    ap.add_argument("--data_root", required=True, type=Path, help="parent dir of per-dataset raw folders")
    ap.add_argument("--config_dir", type=Path, default=Path("configs/data"))
    ap.add_argument("--config", type=Path, default=None, help="(optional) overrides --config_dir/<dataset>.yaml")
    ap.add_argument("--limit_subjects", type=int, default=None)
    args = ap.parse_args()

    names = list(DATASET_REGISTRY) if args.dataset == "all" else [args.dataset]
    total = 0
    for n in names:
        cdir = args.config.parent if args.config else args.config_dir
        total += audit_dataset(n, args.data_root, cdir, args.limit_subjects)
    print(f"\naudit finished: {total} problem(s) flagged")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
