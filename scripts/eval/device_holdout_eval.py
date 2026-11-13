#!/usr/bin/env python3
"""
Held-out-device evaluation: tests generalization to unseen hardware
rather than unseen subjects.

WISDM: phone-to-watch and watch-to-phone transfer, evaluated separately —
train on windows whose channels come from one device class, test on the
other, same subjects appearing in both (their two devices are worn
simultaneously in WISDM's collection protocol).

HHAR: complete device models held out from training entirely, used only
for testing. NOTE: this implementation splits by device CLASS
(smartphone vs smartwatch) since that's what's already tracked per
channel; if finer per-model device identifiers exist in your window
metadata, prefer those for a stricter test.

Usage:
    python3 scripts/eval/device_holdout_eval.py \
        --dataset wisdm --lodo_checkpoint checkpoints/heterowear_lodo/wisdm_step_10000.pt \
        --data_root data/processed --out results/heterowear/wisdm_device_holdout.csv \
        --device cuda
"""
import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.hw_flat import window_dict_to_channel_inputs
from src.models.heterowear import HeteroWearEncoder
from scripts.eval.finetune_heterowear import HeteroWearClassifier, build_lodo_encoder, build_scratch_encoder


def window_device_classes(window_dict: dict) -> set:
    devices = set()
    for k in window_dict["channel_keys"]:
        _, _, device, _, _, _ = window_dict[f"meta__{k}"]
        devices.add(device)
    return devices


def load_windows_by_device(dataset_dir: Path, device_class: str) -> list:
    """Windows where AT LEAST ONE present channel comes from device_class."""
    out = []
    for pkl_path in sorted(dataset_dir.glob("*.pkl")):
        with open(pkl_path, "rb") as fh:
            windows = pickle.load(fh)
        for w in windows:
            if w["activity_label"] and device_class in window_device_classes(w):
                out.append(w)
    return out


def build_label_vocab(window_dicts: list) -> dict:
    return {lbl: i for i, lbl in enumerate(sorted({w["activity_label"] for w in window_dicts}))}


def evaluate(model, windows, label_to_idx, batch_size, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            chunk = windows[i:i + batch_size]
            batch = [window_dict_to_channel_inputs(w) for w in chunk]
            logits = model(batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(np.array([label_to_idx[w["activity_label"]] for w in chunk]))
    preds = np.concatenate(preds); targets = np.concatenate(targets)
    return {"accuracy": float(accuracy_score(targets, preds)),
            "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0))}


def train_simple(model, train_windows, label_to_idx, steps, batch_size, lr, device, seed):
    import random
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        bw = rng.choices(train_windows, k=batch_size) if len(train_windows) < batch_size else rng.sample(train_windows, batch_size)
        batch = [window_dict_to_channel_inputs(w) for w in bw]
        labels = torch.tensor([label_to_idx[w["activity_label"]] for w in bw], device=device, dtype=torch.long)
        loss = loss_fn(model(batch), labels)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["wisdm", "hhar"])
    ap.add_argument("--lodo_checkpoint", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dataset_dir = args.data_root / args.dataset / "all"
    device_pairs = [("smartphone", "smartwatch"), ("smartwatch", "smartphone")]

    fieldnames = ["dataset", "model", "train_device", "test_device", "accuracy", "macro_f1",
                  "n_train", "n_test"]
    csv_file = open(args.out, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader(); csv_file.flush()

    _, arch_args = build_lodo_encoder(args.lodo_checkpoint, args.device)

    for train_dev, test_dev in device_pairs:
        train_windows = load_windows_by_device(dataset_dir, train_dev)
        test_windows = load_windows_by_device(dataset_dir, test_dev)
        label_to_idx = build_label_vocab(train_windows)
        test_windows = [w for w in test_windows if w["activity_label"] in label_to_idx]
        print(f"[{args.dataset}] train={train_dev} ({len(train_windows)}), "
              f"test={test_dev} ({len(test_windows)})")
        if not train_windows or not test_windows or len(label_to_idx) < 2:
            print("  skipping — insufficient data")
            continue

        for model_name in ["HW-Scratch", "HW-LODO"]:
            torch.manual_seed(args.seed)
            encoder = (build_scratch_encoder(arch_args, args.device) if model_name == "HW-Scratch"
                       else build_lodo_encoder(args.lodo_checkpoint, args.device)[0])
            model = HeteroWearClassifier(encoder, len(label_to_idx)).to(args.device)
            train_simple(model, train_windows, label_to_idx, args.steps, args.batch_size,
                         args.lr, args.device, args.seed)
            r = evaluate(model, test_windows, label_to_idx, args.batch_size, args.device)
            row = {"dataset": args.dataset, "model": model_name, "train_device": train_dev,
                   "test_device": test_dev, "n_train": len(train_windows), "n_test": len(test_windows), **r}
            writer.writerow(row); csv_file.flush()
            print(f"  [{model_name}] macro_f1={r['macro_f1']:.3f}")

    csv_file.close()
    print(f"\nwrote results to {args.out}")


if __name__ == "__main__":
    main()
