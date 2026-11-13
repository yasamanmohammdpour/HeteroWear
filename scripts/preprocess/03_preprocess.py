#!/usr/bin/env python3
"""
Turns raw data (via the registry/adapters) into windowed, normalized,
split-tagged samples on disk. This is the one script every downstream
experiment depends on — get this right once, don't reimplement per model.

Pipeline per dataset:
  1. adapter.iter_subjects() -> list of subject IDs actually on disk
  2. subject-level split is created ONCE and saved (never regenerated)
  3. adapter.load_subject_raw(subject) -> channels + activity/HR labels
  4. windowing.make_windows(...) -> list[WindowSample], per-modality native rate preserved
  5. normalization stats fit on TRAIN split only, applied to all splits
  6. windows serialized to data/processed/{dataset}/{subject}.npz

Usage:
    python 03_preprocess.py --dataset ppg_dalia --data_root <repo>/data/raw \
        --out_root <repo>/data/processed --config configs/data/ppg_dalia.yaml
"""

import argparse
import sys
from pathlib import Path

import pickle
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on path

from src.data.registry import get_adapter
from src.data.windowing import make_windows
from src.data.normalization import fit_normalization_stats, apply_normalization, save_stats
from src.data.splits import make_subject_split, make_loso_folds, save_split, assign_windows_to_split
from src.data.schema import WindowSample


def load_config(path: Path, data_root: Path) -> dict:
    with open(path) as f:
        raw = f.read().replace("${DATA_ROOT}", str(data_root))
    return yaml.safe_load(raw)


def serialize_window(w: WindowSample) -> dict:
    """Flatten a WindowSample into a dict npz can store (no nested objects)."""
    out = {
        "dataset": w.dataset, "subject_id": w.subject_id, "session_id": w.session_id,
        "window_start_sec": w.window_start_sec, "window_sec": w.window_sec,
        "activity_label": w.activity_label or "", "hr_bpm": w.hr_bpm if w.hr_bpm is not None else np.nan,
        "hr_source": w.hr_source or "", "channel_keys": list(w.signals.keys()),
    }
    for ch_key, arr in w.signals.items():
        out[f"sig__{ch_key}"] = arr
        meta = w.channel_meta[ch_key]
        out[f"meta__{ch_key}"] = np.array([meta.modality.value, meta.location.value,
                                            meta.device.value, meta.native_hz, meta.n_axes,
                                            meta.is_derived], dtype=object)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=[
        "ppg_dalia", "wesad", "pamap2", "mmash", "wisdm", "hhar", "harth", "uci_har"])
    ap.add_argument("--data_root", required=True, type=Path,
                     help="parent dir containing one subfolder per dataset (raw)")
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--eval_mode", choices=["subject_split", "loso"], default="subject_split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_subjects", type=int, default=None,
                     help="debug: only process first N subjects")
    args = ap.parse_args()

    cfg = load_config(args.config, args.data_root)
    raw_root = args.data_root / args.dataset
    adapter = get_adapter(args.dataset, raw_root, cfg)

    subjects = list(adapter.iter_subjects())
    if not subjects:
        print(f"[{args.dataset}] no subjects found under {raw_root} — check paths/downloads first.")
        return
    if args.limit_subjects:
        subjects = subjects[: args.limit_subjects]
    print(f"[{args.dataset}] {len(subjects)} subjects found: {subjects[:5]}{'...' if len(subjects) > 5 else ''}")

    # --- splits (computed once here, saved, never regenerated downstream) ---
    split_dir = args.out_root.parent / "splits" / args.dataset
    if args.eval_mode == "subject_split":
        split = make_subject_split(subjects, seed=args.seed)
        save_split(split, split_dir / "subject_split.json")
        splits_to_run = [("subject_split", split)]
    else:
        folds = make_loso_folds(subjects)
        splits_to_run = [(f"loso_{f['fold_subject']}", f) for f in folds]
        for name, fold in splits_to_run:
            save_split(fold, split_dir / f"{name}.json")

    # --- load raw + window every subject once, reused across split variants ---
    all_windows: list[WindowSample] = []
    failed = []
    import time as _time
    for subj in subjects:
        _t0 = _time.time()
        try:
            raw = adapter.load_subject_raw(subj)
            print(f"  [load] {subj}: {_time.time() - _t0:.1f}s", flush=True)
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
        all_windows.extend(windows)
        print(f"  [OK] {subj}: {len(windows)} windows")

    print(f"[{args.dataset}] total windows: {len(all_windows)}, failed subjects: {len(failed)}")
    if failed:
        print(f"[{args.dataset}] FAILED SUBJECTS (fix before trusting downstream numbers): {failed}")

    # --- normalize (fit on train only, per the chosen split) + write out ---
    for split_name, split in splits_to_run:
        train_windows = assign_windows_to_split(all_windows, split, "train") if "train" in split \
            else assign_windows_to_split(all_windows, split, "train")
        stats = fit_normalization_stats(train_windows, granularity=cfg.get("normalization", "dataset"))
        save_stats(stats, split_dir / f"{split_name}_norm_stats.json")

        for part in ("train", "val", "test") if "val" in split else ("train", "test"):
            part_windows = assign_windows_to_split(all_windows, split, part)
            out_dir = args.out_root / args.dataset / split_name / part
            out_dir.mkdir(parents=True, exist_ok=True)
            # ONE FILE PER SUBJECT, not per window — writing thousands of tiny
            # .npz files per subject was the real bottleneck (parallel/network
            # filesystem metadata overhead per file). A pickled list of
            # serialized windows per subject cuts file count by orders of
            # magnitude with no change to the actual data.
            by_subject: dict[str, list[dict]] = {}
            for w in part_windows:
                w = apply_normalization(w, stats, granularity=cfg.get("normalization", "dataset"))
                by_subject.setdefault(w.subject_id, []).append(serialize_window(w))
            for subj, window_dicts in by_subject.items():
                out_path = out_dir / f"{subj}.pkl"
                with open(out_path, "wb") as f:
                    pickle.dump(window_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  [{split_name}/{part}] wrote {len(part_windows)} windows across "
                  f"{len(by_subject)} subject file(s) to {out_dir}")


if __name__ == "__main__":
    main()
