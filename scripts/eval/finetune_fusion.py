#!/usr/bin/env python3
"""
Trains and evaluates the HW-Flat + supervised-Transformer FusionClassifier.

Requires a pretrained HW-Flat checkpoint. For each fold, a fresh
FusionClassifier is built (HW-Flat encoder re-initialized from the
pretrained checkpoint each time, TSTransformer backbone and fusion block
freshly initialized), then trained in two phases: first with the
HW-Flat encoder frozen (letting the untrained TF backbone and fusion
block catch up against a stable representation), then unfrozen for joint
fine-tuning at a lower, cosine-decayed learning rate — same discipline as
finetune.py, extended to the two-backbone case.

Usage:
    python3 scripts/eval/finetune_fusion.py \
        --checkpoint checkpoints/pretrain/run2/step_10000.pt \
        --data_root data/processed --out results/fusion/results.csv \
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

from src.models.hw_flat import window_dict_to_channel_inputs
from src.models.classifier import load_pretrained_encoder
from src.models.fusion import FusionClassifier
from src.data.splits import make_subject_split
from src.eval.baseline_data import _find_channel

ACTIVITY_DATASETS = ["pamap2", "wisdm", "hhar", "harth", "uci_har"]
PRIMARY_CHANNEL_PREFIX = {
    "pamap2": "hand_ACC", "wisdm": "wrist_ACC", "hhar": "wrist_ACC",
    "harth": "back_ACC", "uci_har": "phone_ACC",
}
TARGET_LEN = {
    "pamap2": int(10.0 * 100), "wisdm": int(5.0 * 20), "hhar": int(5.0 * 100),
    "harth": int(5.0 * 50), "uci_har": 128,
}
VAL_MAX_WINDOWS = 600


def extract_primary_channel_array(window_dict: dict, prefix: str, target_len: int):
    # reuses the PROVEN channel-matching logic from the baseline pipeline
    # (validated against real data: 44,212 usable HHAR windows) instead of
    # a separate reimplementation, after a hand-rolled version here was
    # found to silently drop 98% of HHAR's windows for reasons that
    # weren't worth chasing further once a known-correct alternative existed
    arr = _find_channel(window_dict, prefix)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    T = arr.shape[0]
    if T >= target_len:
        arr = arr[:target_len]
    else:
        pad = np.zeros((target_len - T, arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr


def load_dataset_windows(dataset_dir: Path, prefix: str, target_len: int) -> dict:
    per_subject = {}
    for pkl_path in sorted(dataset_dir.glob("*.pkl")):
        with open(pkl_path, "rb") as fh:
            windows = pickle.load(fh)
        usable = [w for w in windows
                  if w["activity_label"] and extract_primary_channel_array(w, prefix, target_len) is not None]
        if usable:
            per_subject[pkl_path.stem] = usable
    return per_subject


def build_label_vocab(window_dicts: list) -> dict:
    labels = sorted({w["activity_label"] for w in window_dicts})
    return {lbl: i for i, lbl in enumerate(labels)}


def carve_val_subjects(train_subjects: list, val_frac: float, seed: int):
    subs = sorted(train_subjects)
    rng = random.Random(seed)
    rng.shuffle(subs)
    n_val = max(1, int(round(len(subs) * val_frac)))
    return subs[n_val:], subs[:n_val]


def make_batch(window_dicts: list, prefix: str, target_len: int, device: str):
    cw_batch = [window_dict_to_channel_inputs(w) for w in window_dicts]
    tf_arrs = np.stack([extract_primary_channel_array(w, prefix, target_len) for w in window_dicts])
    tf_input = torch.from_numpy(tf_arrs).to(device)
    return cw_batch, tf_input


def evaluate(model, windows: list, prefix: str, target_len: int, label_to_idx: dict,
             batch_size: int, device: str) -> dict:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            chunk = windows[i:i + batch_size]
            cw_batch, tf_input = make_batch(chunk, prefix, target_len, device)
            logits = model(cw_batch, tf_input)
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


def run_fusion_fold(checkpoint_path, prefix: str, target_len: int,
                     train_windows: list, val_windows: list, test_windows: list,
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
    cw_encoder, _ = load_pretrained_encoder(checkpoint_path, device)
    n_channels = extract_primary_channel_array(train_windows[0], prefix, target_len).shape[1]
    model = FusionClassifier(cw_encoder, n_channels=n_channels, n_classes=n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()

    n_train = len(train_windows)
    total_steps = min(max_steps, max(100, (n_train // batch_size) * epochs))
    warmup_steps = max(20, int(total_steps * warmup_frac))
    finetune_steps = total_steps - warmup_steps

    def sample_windows():
        return rng.choices(train_windows, k=batch_size) if n_train < batch_size else rng.sample(train_windows, batch_size)

    def labels_for(bw):
        return torch.tensor([label_to_idx[w["activity_label"]] for w in bw], device=device, dtype=torch.long)

    for p in model.cw_encoder.parameters():
        p.requires_grad = False
    phase1_params = (list(model.tf_backbone.parameters()) + list(model.proj_cw.parameters())
                      + list(model.proj_tf.parameters()) + list(model.fusion.parameters())
                      + list(model.head.parameters()))
    head_opt = torch.optim.AdamW(phase1_params, lr=head_lr)

    model.train()
    for _ in range(warmup_steps):
        bw = sample_windows()
        cw_batch, tf_input = make_batch(bw, prefix, target_len, device)
        loss = loss_fn(model(cw_batch, tf_input), labels_for(bw))
        head_opt.zero_grad(); loss.backward(); head_opt.step()

    for p in model.cw_encoder.parameters():
        p.requires_grad = True
    full_opt = torch.optim.AdamW(model.parameters(), lr=encoder_lr)

    best_val_f1 = -1.0
    best_state = None
    for step in range(finetune_steps):
        lr = cosine_lr(step, finetune_steps, encoder_lr)
        for g in full_opt.param_groups:
            g["lr"] = lr

        model.train()
        bw = sample_windows()
        cw_batch, tf_input = make_batch(bw, prefix, target_len, device)
        loss = loss_fn(model(cw_batch, tf_input), labels_for(bw))
        full_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        full_opt.step()

        if (step + 1) % eval_every == 0 or step == finetune_steps - 1:
            val_metrics = evaluate(model, val_windows, prefix, target_len, label_to_idx, batch_size, device)
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_windows, prefix, target_len, label_to_idx, batch_size, device)
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
    ap.add_argument("--val_frac", type=float, default=0.15)
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
        prefix = PRIMARY_CHANNEL_PREFIX[dataset]
        target_len = TARGET_LEN[dataset]
        print(f"[{dataset}] loading windows (primary channel prefix={prefix})...")
        per_subject = load_dataset_windows(dataset_dir, prefix, target_len)
        subjects = sorted(per_subject.keys())
        if len(subjects) < 4:
            print(f"[{dataset}] too few subjects, skipping")
            continue

        split = make_subject_split(subjects, seed=args.seed)
        train_windows = [w for s in split["train"] for w in per_subject.get(s, [])]
        val_windows = [w for s in split["val"] for w in per_subject.get(s, [])]
        test_windows = [w for s in split["test"] for w in per_subject.get(s, [])]
        r = run_fusion_fold(args.checkpoint, prefix, target_len, train_windows, val_windows, test_windows,
                             args.epochs, args.batch_size, args.head_lr, args.encoder_lr,
                             args.warmup_frac, args.max_steps, args.eval_every,
                             args.device, args.seed, rng)
        if r is not None:
            row = {"dataset": dataset, "eval": "within_dataset", "model": "Fusion", **r}
            _write(row)
            print(f"  [within_dataset] macro_f1={r['macro_f1']:.3f} "
                  f"(steps={r['steps']}, best_val_f1={r['best_val_f1']:.3f})")

        loso_subjects = subjects[: args.max_loso_folds]
        f1s = []
        for test_subj in loso_subjects:
            loso_train_pool = [s for s in subjects if s != test_subj]
            train_subj, val_subj = carve_val_subjects(loso_train_pool, args.val_frac, args.seed)
            train_windows = [w for s in train_subj for w in per_subject.get(s, [])]
            val_windows = [w for s in val_subj for w in per_subject.get(s, [])]
            test_windows = per_subject.get(test_subj, [])
            r = run_fusion_fold(args.checkpoint, prefix, target_len, train_windows, val_windows, test_windows,
                                 args.epochs, args.batch_size, args.head_lr, args.encoder_lr,
                                 args.warmup_frac, args.max_steps, args.eval_every,
                                 args.device, args.seed, rng)
            if r is None:
                continue
            f1s.append(r["macro_f1"])
            _write({"dataset": dataset, "eval": f"loso_{test_subj}", "model": "Fusion", **r})
        if f1s:
            print(f"  [LOSO x{len(f1s)}] macro_f1={np.mean(f1s):.3f}+/-{np.std(f1s):.3f}")
            _write({"dataset": dataset, "eval": "loso_mean", "model": "Fusion",
                    "macro_f1": float(np.mean(f1s))})

    csv_file.close()
    print(f"\nwrote results to {args.out}")


if __name__ == "__main__":
    main()
