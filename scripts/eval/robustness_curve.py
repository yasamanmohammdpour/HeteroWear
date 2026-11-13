#!/usr/bin/env python3
"""
Tests the actual headline claim: does the pretrained HW-Flat
representation stay useful as sensors go missing at inference time?

A classifier is trained ONCE on full-channel embeddings (the normal
deployment scenario), then evaluated unchanged as increasing fractions of
channels are randomly removed from TEST windows before re-embedding.
This isolates test-time robustness — does performance degrade gracefully
or catastrophically as sensors fail after deployment — independent of
training procedure.

Datasets with more channels (PAMAP2: up to 9 — 3 locations x ACC/GYRO/MAG;
WISDM/HHAR: up to 4 — phone+watch x ACC/GYRO) give a more informative
curve than low-channel-count datasets (HARTH, UCI-HAR: 2 channels), where
there's only really one meaningful drop point. Worth noting honestly in
the writeup rather than averaging over datasets with very different
resolution on this axis.

Usage:
    python3 scripts/eval/robustness_curve.py \
        --checkpoint checkpoints/pretrain/run1/step_5000.pt \
        --data_root data/processed --out results/robustness/curve.csv \
        --device cuda
"""
import argparse
import csv
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.hw_flat import HWFlatEncoder, window_dict_to_channel_inputs
from src.data.splits import make_subject_split

ACTIVITY_DATASETS = ["pamap2", "wisdm", "hhar", "harth", "uci_har"]
DROP_RATES = [0.0, 0.25, 0.5, 0.75]


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


def random_channel_drop(channels: list, drop_rate: float, rng: random.Random) -> list:
    if drop_rate <= 0 or len(channels) <= 1:
        return channels
    n_keep = max(1, int(round(len(channels) * (1 - drop_rate))))
    return rng.sample(channels, n_keep)


def embed_windows(model: HWFlatEncoder, windows: list, device: str, batch_size: int,
                   drop_rate: float, rng: random.Random) -> np.ndarray:
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            chunk = windows[i:i + batch_size]
            batch = []
            for w in chunk:
                ch = window_dict_to_channel_inputs(w)
                if drop_rate > 0:
                    ch = random_channel_drop(ch, drop_rate, rng)
                batch.append(ch)
            pooled, _, _, _, _ = model(batch, mask_ratio=0.0)
            embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def load_dataset_windows(dataset_dir: Path) -> dict:
    per_subject = {}
    for pkl_path in sorted(dataset_dir.glob("*.pkl")):
        with open(pkl_path, "rb") as fh:
            windows = pickle.load(fh)
        labeled = [w for w in windows if w["activity_label"]]
        if labeled:
            per_subject[pkl_path.stem] = labeled
    return per_subject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    model = load_encoder(args.checkpoint, args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "drop_rate", "accuracy", "macro_f1", "n_train", "n_test",
                  "mean_channels_full", "mean_channels_at_drop_rate"]
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
        print(f"[{dataset}] loading windows...")
        per_subject = load_dataset_windows(dataset_dir)
        subjects = sorted(per_subject.keys())
        if len(subjects) < 4:
            print(f"[{dataset}] too few subjects, skipping")
            continue

        split = make_subject_split(subjects, seed=args.seed)
        train_windows = [w for s in split["train"] + split["val"] for w in per_subject.get(s, [])]
        test_windows = [w for s in split["test"] for w in per_subject.get(s, [])]
        if not train_windows or not test_windows:
            print(f"[{dataset}] empty split, skipping")
            continue

        mean_ch_full = float(np.mean([len(window_dict_to_channel_inputs(w)) for w in test_windows[:200]]))
        print(f"  mean channels/window (full): {mean_ch_full:.2f}")

        print("  embedding train set (full channels)...")
        X_train = embed_windows(model, train_windows, args.device, args.batch_size, 0.0, rng)
        y_train = np.array([w["activity_label"] for w in train_windows])
        clf = LogisticRegression(max_iter=2000, random_state=args.seed)
        clf.fit(X_train, y_train)

        for drop_rate in DROP_RATES:
            print(f"  embedding test set (drop_rate={drop_rate})...")
            X_test = embed_windows(model, test_windows, args.device, args.batch_size, drop_rate, rng)
            y_test = np.array([w["activity_label"] for w in test_windows])
            keep_mask = np.isin(y_test, clf.classes_)
            X_test_f, y_test_f = X_test[keep_mask], y_test[keep_mask]
            if len(y_test_f) == 0:
                continue
            pred = clf.predict(X_test_f)
            acc = float(accuracy_score(y_test_f, pred))
            f1 = float(f1_score(y_test_f, pred, average="macro", zero_division=0))

            mean_ch_dropped = mean_ch_full if drop_rate == 0.0 else float(np.mean(
                [len(random_channel_drop(window_dict_to_channel_inputs(w), drop_rate, rng))
                 for w in test_windows[:200]]))

            row = {"dataset": dataset, "drop_rate": drop_rate, "accuracy": acc, "macro_f1": f1,
                   "n_train": len(y_train), "n_test": len(y_test_f),
                   "mean_channels_full": round(mean_ch_full, 2),
                   "mean_channels_at_drop_rate": round(mean_ch_dropped, 2)}
            _write(row)
            print(f"    drop_rate={drop_rate:.2f}  macro_f1={f1:.3f}  "
                  f"(avg {mean_ch_dropped:.1f}/{mean_ch_full:.1f} channels)")

    csv_file.close()
    print(f"\nwrote results to {args.out}")


if __name__ == "__main__":
    main()
