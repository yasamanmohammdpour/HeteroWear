#!/usr/bin/env python3
"""
Fine-tunes the SSL-pretrained TSTransformer (from pretrain_ts_transformer.py)
on each activity-labeled dataset, within-dataset and LOSO, using the exact
same subject splits and evaluation protocol as run_har_baselines.py — so
the resulting "Transformer_SSL" row is directly comparable to the
from-scratch "Transformer" row already in the results table.

Usage:
    python3 scripts/eval/finetune_ts_transformer.py \
        --checkpoint checkpoints/ts_pretrain/run1/step_5000.pt \
        --processed_root data/processed --out results/ts_ssl/results.csv \
        --device cuda
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models_baseline.ts_transformer import TSTransformer
from src.eval.baseline_data import PRIMARY_CHANNEL_PREFIX, load_all_subjects, build_split
from src.data.splits import make_subject_split

TARGET_LEN = {
    "pamap2": int(10.0 * 100), "wisdm": int(5.0 * 20), "hhar": int(5.0 * 100),
    "harth": int(5.0 * 50), "uci_har": 128,
}


def load_pretrained_backbone(checkpoint_path: Path, n_classes: int, device: str) -> TSTransformer:
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt["args"]
    model = TSTransformer(n_channels=ckpt_args["n_channels"], n_classes=n_classes,
                           d_model=ckpt_args["d_model"], n_heads=ckpt_args["n_heads"],
                           n_layers=ckpt_args["n_layers"], patch_len=ckpt_args["patch_len"]).to(device)
    filtered = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("head.")}
    model.load_state_dict(filtered, strict=False)
    return model


def cosine_lr(step: int, total_steps: int, base_lr: float) -> float:
    progress = step / max(1, total_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def run_ts_ssl_fold(checkpoint_path, X_train, y_train, X_test, y_test, n_classes: int,
                     head_lr: float, encoder_lr: float, warmup_steps: int, total_steps: int,
                     batch_size: int, device: str, seed: int) -> dict:
    torch.manual_seed(seed)
    model = load_pretrained_backbone(checkpoint_path, n_classes, device)
    loss_fn = nn.CrossEntropyLoss()

    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                           batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                          batch_size=batch_size, shuffle=False)

    for name, p in model.named_parameters():
        if not name.startswith("head."):
            p.requires_grad = False
    head_opt = torch.optim.AdamW(model.head.parameters(), lr=head_lr)

    model.train()
    step = 0
    while step < warmup_steps:
        for xb, yb in train_dl:
            if step >= warmup_steps:
                break
            xb, yb = xb.to(device), yb.to(device)
            head_opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            head_opt.step()
            step += 1

    for p in model.parameters():
        p.requires_grad = True
    full_opt = torch.optim.AdamW(model.parameters(), lr=encoder_lr)

    remaining = max(1, total_steps - warmup_steps)
    step = 0
    while step < remaining:
        for xb, yb in train_dl:
            if step >= remaining:
                break
            lr = cosine_lr(step, remaining, encoder_lr)
            for g in full_opt.param_groups:
                g["lr"] = lr
            xb, yb = xb.to(device), yb.to(device)
            full_opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            full_opt.step()
            step += 1

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            pred = model(xb.to(device)).argmax(dim=1).cpu().numpy()
            preds.append(pred); targets.append(yb.numpy())
    preds = np.concatenate(preds); targets = np.concatenate(targets)

    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "n_train": len(y_train), "n_test": len(y_test), "device": device,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--processed_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_loso_folds", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--total_steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "eval", "model", "accuracy", "macro_f1", "n_train", "n_test", "device"]
    csv_file = open(args.out, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader(); csv_file.flush()

    def _write(row):
        writer.writerow(row); csv_file.flush()

    for dataset, prefix in PRIMARY_CHANNEL_PREFIX.items():
        dataset_dir = args.processed_root / dataset / "all"
        if not dataset_dir.exists():
            print(f"[{dataset}] no data, skipping")
            continue
        per_subject = load_all_subjects(dataset_dir, prefix, TARGET_LEN[dataset])
        subjects = sorted(per_subject.keys())
        print(f"[{dataset}] {len(subjects)} subjects with usable windows")
        if len(subjects) < 3:
            continue

        split = make_subject_split(subjects, seed=args.seed)
        result = build_split(per_subject, split["train"] + split["val"], split["test"])
        if result is not None:
            X_train, y_train, X_test, y_test, vocab = result
            r = run_ts_ssl_fold(args.checkpoint, X_train, y_train, X_test, y_test, len(vocab),
                                 args.head_lr, args.encoder_lr, args.warmup_steps, args.total_steps,
                                 args.batch_size, args.device, args.seed)
            row = {"dataset": dataset, "eval": "within_dataset", "model": "Transformer_SSL", **r}
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
            r = run_ts_ssl_fold(args.checkpoint, X_train, y_train, X_test, y_test, len(vocab),
                                 args.head_lr, args.encoder_lr, args.warmup_steps, args.total_steps,
                                 args.batch_size, args.device, args.seed)
            f1s.append(r["macro_f1"])
            _write({"dataset": dataset, "eval": f"loso_{test_subj}", "model": "Transformer_SSL", **r})
        if f1s:
            print(f"  [LOSO x{len(f1s)}] macro_f1={np.mean(f1s):.3f}+/-{np.std(f1s):.3f}")
            _write({"dataset": dataset, "eval": "loso_mean", "model": "Transformer_SSL",
                    "macro_f1": float(np.mean(f1s))})

    csv_file.close()
    print(f"\nwrote results to {args.out}")


if __name__ == "__main__":
    main()
