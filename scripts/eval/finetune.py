#!/usr/bin/env python3
"""
Fine-tunes the pretrained HW-Flat encoder on each activity-labeled
dataset, within-dataset and LOSO. Publication-quality version — fixes vs.
the naive first pass:

  1. Two-phase training: encoder frozen while only the (randomly
     initialized) head trains first, THEN encoder unfrozen for joint
     fine-tuning at a much lower LR. Prevents the random head's early
     gradients from corrupting good pretrained features.
  2. Training steps scale with dataset size (epochs * n_train/batch_size,
     capped by --max_steps) instead of a flat cap that starved large
     datasets of training and left small ones possibly over-budget.
  3. Cosine LR decay during the fine-tuning phase.
  4. Gradient clipping for stability at the larger step budgets.
  5. Validation-based checkpoint selection: a real held-out validation
     set (never used for final scoring) picks the best checkpoint by
     macro-F1 during training, rather than reporting whatever the model
     happens to be at the last step.

Usage:
    python3 scripts/eval/finetune.py \
        --checkpoint checkpoints/pretrain/run1/step_5000.pt \
        --data_root data/processed --out results/finetune/results.csv \
        --device cuda
"""
import argparse
import csv
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.classifier import HWFlatClassifier, load_pretrained_encoder
from src.models.hw_flat import window_dict_to_channel_inputs
from src.data.splits import make_subject_split

ACTIVITY_DATASETS = ["pamap2", "wisdm", "hhar", "harth", "uci_har"]
VAL_MAX_WINDOWS = 600  # caps eval cost regardless of dataset size — a few hundred
                        # windows is statistically plenty for checkpoint selection


def load_dataset_windows(dataset_dir: Path) -> dict:
    per_subject = {}
    for pkl_path in sorted(dataset_dir.glob("*.pkl")):
        with open(pkl_path, "rb") as fh:
            windows = pickle.load(fh)
        labeled = [w for w in windows if w["activity_label"]]
        if labeled:
            per_subject[pkl_path.stem] = labeled
    return per_subject


def build_label_vocab(window_dicts: list) -> dict:
    labels = sorted({w["activity_label"] for w in window_dicts})
    return {lbl: i for i, lbl in enumerate(labels)}


def carve_val_subjects(train_subjects: list, val_frac: float, seed: int):
    """Splits a train-subject pool into (train, val) — used for LOSO folds,
    where make_loso_folds doesn't produce a val split and using the test
    subject for validation would leak."""
    subs = sorted(train_subjects)
    rng = random.Random(seed)
    rng.shuffle(subs)
    n_val = max(1, int(round(len(subs) * val_frac)))
    return subs[n_val:], subs[:n_val]


def evaluate(model, windows: list, label_to_idx: dict, batch_size: int, device: str) -> dict:
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
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def cosine_lr(step: int, total_steps: int, base_lr: float) -> float:
    progress = step / max(1, total_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def run_finetune_fold(checkpoint_path, train_windows: list, val_windows: list, test_windows: list,
                       epochs: int, batch_size: int, head_lr: float, encoder_lr: float,
                       warmup_frac: float, max_steps: int, eval_every: int,
                       device: str, seed: int, rng: random.Random) -> dict:
    label_to_idx = build_label_vocab(train_windows)
    n_classes = len(label_to_idx)
    val_windows = [w for w in val_windows if w["activity_label"] in label_to_idx]
    test_windows = [w for w in test_windows if w["activity_label"] in label_to_idx]
    if not train_windows or not val_windows or not test_windows or n_classes < 2:
        return None
    if len(val_windows) > VAL_MAX_WINDOWS:
        val_windows = rng.sample(val_windows, VAL_MAX_WINDOWS)

    torch.manual_seed(seed)
    encoder, _ = load_pretrained_encoder(checkpoint_path, device)
    model = HWFlatClassifier(encoder, n_classes=n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()

    n_train = len(train_windows)
    total_steps = min(max_steps, max(100, (n_train // batch_size) * epochs))
    warmup_steps = max(20, int(total_steps * warmup_frac))
    finetune_steps = total_steps - warmup_steps

    def sample_batch():
        bw = rng.choices(train_windows, k=batch_size) if n_train < batch_size else rng.sample(train_windows, batch_size)
        batch = [window_dict_to_channel_inputs(w) for w in bw]
        labels = torch.tensor([label_to_idx[w["activity_label"]] for w in bw], device=device, dtype=torch.long)
        return batch, labels

    # --- Phase 1: encoder frozen, head only (protects pretrained features) ---
    for p in model.encoder.parameters():
        p.requires_grad = False
    head_opt = torch.optim.AdamW(model.head.parameters(), lr=head_lr)

    model.train()
    for _ in range(warmup_steps):
        batch, labels = sample_batch()
        loss = loss_fn(model(batch), labels)
        head_opt.zero_grad(); loss.backward(); head_opt.step()

    # --- Phase 2: unfrozen joint fine-tuning, cosine-decayed LR, val-selected checkpoint ---
    for p in model.encoder.parameters():
        p.requires_grad = True
    full_opt = torch.optim.AdamW(model.parameters(), lr=encoder_lr)

    best_val_f1 = -1.0
    best_state = None
    for step in range(finetune_steps):
        lr = cosine_lr(step, finetune_steps, encoder_lr)
        for g in full_opt.param_groups:
            g["lr"] = lr

        model.train()
        batch, labels = sample_batch()
        loss = loss_fn(model(batch), labels)
        full_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        full_opt.step()

        if (step + 1) % eval_every == 0 or step == finetune_steps - 1:
            val_metrics = evaluate(model, val_windows, label_to_idx, batch_size, device)
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_windows, label_to_idx, batch_size, device)
    return {**test_metrics, "n_train": n_train, "n_val": len(val_windows), "n_test": len(test_windows),
            "steps": total_steps, "best_val_f1": best_val_f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_loso_folds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=5e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.15)
    ap.add_argument("--max_steps", type=int, default=1200)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--val_frac", type=float, default=0.15,
                     help="fraction of LOSO train subjects carved out for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "eval", "model", "accuracy", "macro_f1", "n_train", "n_val",
                  "n_test", "steps", "best_val_f1"]
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

        # --- within-dataset: use the real val split, never merged into train ---
        split = make_subject_split(subjects, seed=args.seed)
        train_windows = [w for s in split["train"] for w in per_subject.get(s, [])]
        val_windows = [w for s in split["val"] for w in per_subject.get(s, [])]
        test_windows = [w for s in split["test"] for w in per_subject.get(s, [])]
        r = run_finetune_fold(args.checkpoint, train_windows, val_windows, test_windows,
                               args.epochs, args.batch_size, args.head_lr, args.encoder_lr,
                               args.warmup_frac, args.max_steps, args.eval_every,
                               args.device, args.seed, rng)
        if r is not None:
            row = {"dataset": dataset, "eval": "within_dataset", "model": "HW-Flat_FineTuned", **r}
            _write(row)
            print(f"  [within_dataset] macro_f1={r['macro_f1']:.3f} "
                  f"(steps={r['steps']}, best_val_f1={r['best_val_f1']:.3f})")

        # --- LOSO: carve a val subset from each fold's train pool ---
        loso_subjects = subjects[: args.max_loso_folds]
        f1s = []
        for test_subj in loso_subjects:
            loso_train_pool = [s for s in subjects if s != test_subj]
            train_subj, val_subj = carve_val_subjects(loso_train_pool, args.val_frac, args.seed)
            train_windows = [w for s in train_subj for w in per_subject.get(s, [])]
            val_windows = [w for s in val_subj for w in per_subject.get(s, [])]
            test_windows = per_subject.get(test_subj, [])
            r = run_finetune_fold(args.checkpoint, train_windows, val_windows, test_windows,
                                   args.epochs, args.batch_size, args.head_lr, args.encoder_lr,
                                   args.warmup_frac, args.max_steps, args.eval_every,
                                   args.device, args.seed, rng)
            if r is None:
                continue
            f1s.append(r["macro_f1"])
            _write({"dataset": dataset, "eval": f"loso_{test_subj}", "model": "HW-Flat_FineTuned", **r})
        if f1s:
            print(f"  [LOSO x{len(f1s)}] macro_f1={np.mean(f1s):.3f}+/-{np.std(f1s):.3f}")
            _write({"dataset": dataset, "eval": "loso_mean", "model": "HW-Flat_FineTuned",
                    "macro_f1": float(np.mean(f1s))})

    csv_file.close()
    print(f"\nwrote results to {args.out}")


if __name__ == "__main__":
    main()
