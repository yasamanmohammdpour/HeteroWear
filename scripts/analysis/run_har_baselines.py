#!/usr/bin/env python3
"""
Runs RF, CNN, DeepConvLSTM, and Transformer baselines on the 5 activity-
labeled datasets under two eval schemes: within-dataset (random 70/30
subject split) and LOSO (capped by --max_loso_folds for datasets with
many subjects). Writes results incrementally to CSV so a timeout never
loses completed work.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.baseline_data import PRIMARY_CHANNEL_PREFIX, load_all_subjects, build_split
from src.data.splits import make_subject_split
from src.models_baseline.classical import run_rf_baseline
from src.models_baseline.deep import run_cnn_baseline
from src.models_baseline.deepconvlstm import run_deepconvlstm_baseline
from src.models_baseline.ts_transformer import run_transformer_baseline

TARGET_LEN = {
    "pamap2": int(10.0 * 100), "wisdm": int(5.0 * 20), "hhar": int(5.0 * 100),
    "harth": int(5.0 * 50), "uci_har": 128,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max_loso_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "eval", "model", "accuracy", "macro_f1", "n_train", "n_test", "device"]
    csv_file = open(args.out, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    csv_file.flush()

    def _write_row(row):
        writer.writerow(row)
        csv_file.flush()

    rows = []
    for dataset, prefix in PRIMARY_CHANNEL_PREFIX.items():
        dataset_dir = args.processed_root / dataset / "all"
        if not dataset_dir.exists():
            print(f"[{dataset}] no data at {dataset_dir}, skipping")
            continue
        per_subject = load_all_subjects(dataset_dir, prefix, TARGET_LEN[dataset])
        subjects = sorted(per_subject.keys())
        print(f"[{dataset}] {len(subjects)} subjects with usable windows")
        if len(subjects) < 3:
            print(f"[{dataset}] too few subjects, skipping")
            continue

        split = make_subject_split(subjects, seed=args.seed)
        result = build_split(per_subject, split["train"] + split["val"], split["test"])
        if result is not None:
            X_train, y_train, X_test, y_test, vocab = result
            n_classes = len(vocab)
            rf = run_rf_baseline(X_train, y_train, X_test, y_test, seed=args.seed)
            cnn = run_cnn_baseline(X_train, y_train, X_test, y_test, n_classes=n_classes, seed=args.seed)
            dcl = run_deepconvlstm_baseline(X_train, y_train, X_test, y_test, n_classes=n_classes, seed=args.seed)
            tsf = run_transformer_baseline(X_train, y_train, X_test, y_test, n_classes=n_classes, seed=args.seed)
            for model_name, res in [("RF", rf), ("CNN", cnn), ("DeepConvLSTM", dcl), ("Transformer", tsf)]:
                row = {"dataset": dataset, "eval": "within_dataset", "model": model_name, **res}
                rows.append(row); _write_row(row)
            print(f"  [within_dataset] RF={rf['macro_f1']:.3f} CNN={cnn['macro_f1']:.3f} "
                  f"DeepConvLSTM={dcl['macro_f1']:.3f} Transformer={tsf['macro_f1']:.3f}")

        loso_subjects = subjects[: args.max_loso_folds]
        rf_f1s, cnn_f1s, dcl_f1s, tsf_f1s = [], [], [], []
        for test_subj in loso_subjects:
            train_subj = [s for s in subjects if s != test_subj]
            result = build_split(per_subject, train_subj, [test_subj])
            if result is None:
                continue
            X_train, y_train, X_test, y_test, vocab = result
            n_classes = len(vocab)
            rf = run_rf_baseline(X_train, y_train, X_test, y_test, seed=args.seed)
            cnn = run_cnn_baseline(X_train, y_train, X_test, y_test, n_classes=n_classes, seed=args.seed)
            dcl = run_deepconvlstm_baseline(X_train, y_train, X_test, y_test, n_classes=n_classes, seed=args.seed)
            tsf = run_transformer_baseline(X_train, y_train, X_test, y_test, n_classes=n_classes, seed=args.seed)
            rf_f1s.append(rf["macro_f1"]); cnn_f1s.append(cnn["macro_f1"])
            dcl_f1s.append(dcl["macro_f1"]); tsf_f1s.append(tsf["macro_f1"])
            for model_name, res in [("RF", rf), ("CNN", cnn), ("DeepConvLSTM", dcl), ("Transformer", tsf)]:
                row = {"dataset": dataset, "eval": f"loso_{test_subj}", "model": model_name, **res}
                rows.append(row); _write_row(row)
        if rf_f1s:
            print(f"  [LOSO x{len(rf_f1s)}] RF={np.mean(rf_f1s):.3f} CNN={np.mean(cnn_f1s):.3f} "
                  f"DeepConvLSTM={np.mean(dcl_f1s):.3f} Transformer={np.mean(tsf_f1s):.3f}")
            for model_name, f1s in [("RF", rf_f1s), ("CNN", cnn_f1s), ("DeepConvLSTM", dcl_f1s), ("Transformer", tsf_f1s)]:
                row = {"dataset": dataset, "eval": "loso_mean", "model": model_name, "macro_f1": float(np.mean(f1s))}
                rows.append(row); _write_row(row)

    csv_file.close()
    print(f"\nwrote {len(rows)} result rows to {args.out}")


if __name__ == "__main__":
    main()
