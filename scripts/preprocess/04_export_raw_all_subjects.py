#!/usr/bin/env python3
"""
Export ALL subjects for a dataset as raw (unnormalized) windows, one
pickle file per subject, no split baked in.

Why this exists separately from 03_preprocess.py: baking one fixed
train/val/test split's normalization stats into the stored files means
regenerating everything from scratch every time you want a different
eval scheme (within-dataset random split vs. LOSO vs. leave-one-dataset-out
later). Storing raw windows once and normalizing at training time (per
experiment, fit on whatever that experiment's train subjects are) avoids
that duplication and is the more correct approach anyway.

Usage:
    python3 scripts/preprocess/04_export_raw_all_subjects.py --dataset pamap2 \
        --data_root data/raw --out_root data/processed --config configs/data/pamap2.yaml
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.registry import get_adapter
from src.data.windowing import make_windows
from src.data.schema import WindowSample


def load_config(path: Path, data_root: Path) -> dict:
    with open(path) as f:
        raw = f.read().replace("${DATA_ROOT}", str(data_root))
    return yaml.safe_load(raw)


def serialize_window(w: WindowSample) -> dict:
    out = {
        "dataset": w.dataset, "subject_id": w.subject_id, "session_id": w.session_id,
        "window_start_sec": w.window_start_sec, "window_sec": w.window_sec,
        "activity_label": w.activity_label or "", "hr_bpm": w.hr_bpm if w.hr_bpm is not None else np.nan,
        "hr_source": w.hr_source or "", "channel_keys": list(w.signals.keys()),
    }
    for ch_key, arr in w.signals.items():
        out[f"sig__{ch_key}"] = arr
        meta = w.channel_meta[ch_key]
        out[f"meta__{ch_key}"] = (meta.modality.value, meta.location.value,
                                    meta.device.value, meta.native_hz, meta.n_axes, meta.is_derived)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=[
        "ppg_dalia", "wesad", "pamap2", "mmash", "wisdm", "hhar", "harth", "uci_har"])
    ap.add_argument("--data_root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--limit_subjects", type=int, default=None, help="debug only")
    args = ap.parse_args()

    cfg = load_config(args.config, args.data_root)
    raw_root = args.data_root / args.dataset
    adapter = get_adapter(args.dataset, raw_root, cfg)

    subjects = list(adapter.iter_subjects())
    if args.limit_subjects:
        subjects = subjects[: args.limit_subjects]
    if not subjects:
        print(f"[{args.dataset}] no subjects found under {raw_root}")
        return
    print(f"[{args.dataset}] exporting {len(subjects)} subjects (raw, no split, no normalization)")

    out_dir = args.out_root / args.dataset / "all"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_windows = 0
    failed = []
    for subj in subjects:
        try:
            raw = adapter.load_subject_raw(subj)
        except Exception as e:
            failed.append((subj, str(e)))
            print(f"  [FAIL] {subj}: {e}")
            continue
        windows = make_windows(
            dataset_name=args.dataset, subject_id=subj, session_id=raw["session_id"],
            channels=raw["channels"], window_sec=cfg["window_sec"], stride_sec=cfg["stride_sec"],
            min_coverage_frac=cfg.get("min_coverage_frac", 0.5),
            activity_labels=raw.get("activity_labels"), hr_stream=raw.get("hr_stream"),
        )
        window_dicts = [serialize_window(w) for w in windows]
        with open(out_dir / f"{subj}.pkl", "wb") as f:
            pickle.dump(window_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
        total_windows += len(windows)
        print(f"  [OK] {subj}: {len(windows)} windows -> {out_dir / f'{subj}.pkl'}")

    print(f"[{args.dataset}] TOTAL: {total_windows} windows across {len(subjects) - len(failed)} subjects, "
          f"{len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
