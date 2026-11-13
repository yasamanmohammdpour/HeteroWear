#!/usr/bin/env python3
"""
Linear probe evaluation: freeze the pretrained HW-Flat encoder, extract
pooled embeddings for every window (full multimodal input, no dropout, no
masking), then train a plain logistic regression on top. This is the
standard test of SSL representation quality — if a linear classifier on
frozen features can't beat the supervised baselines, pretraining didn't
learn a useful representation, and that's the honest result to report.

Usage:
    python3 scripts/eval/linear_probe.py \
        --checkpoint checkpoints/pretrain/run1/step_5000.pt \
        --data_root data/processed --out results/linear_probe/results.csv \
        --device cuda
"""
import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.hw_flat import HWFlatEncoder, window_dict_to_channel_inputs
from src.data.splits import make_subject_split
from src.eval.baseline_data import build_split

ACTIVITY_DATASETS = ["pamap2", "wisdm", "hhar", "harth", "uci_har"]


def load_encoder(checkpoint_path: Path, device: str) -> HWFlatEncoder:
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt["args"]
    model = HWFlatEncoder(
        d_model=args["d_model"], n_patches=args["n_patches"],
        samples_per_patch=args["samples_per_patch"],
        n_layers=args["n_layers"], n_heads=args["n_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded checkpoint from step {ckpt['step']}")
    return model


def extract_embeddings(model, dataset_dir: Path, device: str, batch_size: int = 64) -> dict:
    per_subject = {}
    for pkl_path in sorted(dataset_dir.glob("*.pkl")):
        subj = pkl_path.stem
        with open(pkl_path, "rb") as fh:
            windows = pickle.load(fh)

        labeled = [w for w in windows if w["activity_label"]]
        if not labeled:
            continue

        embeddings, labels = [], []
        with torch.no_grad():
            for i in range(0, len(labeled), batch_size):
                chunk = labeled[i:i + batch_size]
                batch = [window_dict_to_channel_inputs(w) for w in chunk]
                pooled, _, _, _, _ = model(batch, mask_ratio=0.0)
                embeddings.append(pooled.cpu().numpy())
                labels.extend(w["activity_label"] for w in chunk)

        X = np.concatenate(embeddings, axis=0).astype(np.float32)
        y = np.array(labels, dtype=object)
        per_subject[subj] = (X, y)
        print(f"    {subj}: {len(y)} windows embedded", flush=True)
    return per_subject


def run_linear_probe(X_train, y_train, X_test, y_test, seed: int = 42) -> dict:
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "n_train": len(y_train), "n_test": len(y_test),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_loso_folds", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = load_encoder(args.checkpoint, args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "eval", "model", "accuracy", "macro_f1", "n_train", "n_test"]
    csv_file = open(args.out, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader(); csv_file.flush()

    def _write(row):
        writer.writerow(row); csv_file.flush()

    for dataset in ACTIVITY_DATASETS:
        dataset_dir = args.data_root / dataset / "all"
        if not dataset_dir.exists():
            print(f"[{dataset}] no data at {dataset_dir}, skipping")
            continue
        print(f"[{dataset}] extracting embeddings...")
        per_subject = extract_embeddings(model, dataset_dir, args.device, args.batch_size)
        subjects = sorted(per_subject.keys())
        if len(subjects) < 3:
            print(f"[{dataset}] too few subjects, skipping")
            continue

        split = make_subject_split(subjects, seed=args.seed)
        result = build_split(per_subject, split["train"] + split["val"], split["test"])
        if result is not None:
            X_train, y_train, X_test, y_test, vocab = result
            r = run_linear_probe(X_train, y_train, X_test, y_test, seed=args.seed)
            row = {"dataset": dataset, "eval": "within_dataset", "model": "HW-Flat_LinearProbe", **r}
            _write(row)
            print(f"  [within_dataset] macro_f1={r['macro_f1']:.3f}")

        loso_subjects = subjects[: args.max_loso_folds]
        f1s = []
        for test_subj in loso_subjects:
            train_subj = [s for s in subjects if s != test_subj]
            result = build_split(per_subject, train_subj, [test_subj])
            if result is None:
                continue
            X_train, y_train, X_test, y_test, vocab = result
            r = run_linear_probe(X_train, y_train, X_test, y_test, seed=args.seed)
            f1s.append(r["macro_f1"])
            _write({"dataset": dataset, "eval": f"loso_{test_subj}", "model": "HW-Flat_LinearProbe", **r})
        if f1s:
            print(f"  [LOSO x{len(f1s)}] macro_f1={np.mean(f1s):.3f}+/-{np.std(f1s):.3f}")
            _write({"dataset": dataset, "eval": "loso_mean", "model": "HW-Flat_LinearProbe",
                    "macro_f1": float(np.mean(f1s))})

    csv_file.close()
    print(f"\nwrote results to {args.out}")


if __name__ == "__main__":
    main()
