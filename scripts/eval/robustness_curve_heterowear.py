#!/usr/bin/env python3
"""
Missing-sensor robustness curve for HeteroWear: trains HW-Scratch and
HW-LODO once each (full-channel, single within-dataset split), freezes
both, then sweeps channel dropout at inference at d in {0,0.25,0.5,0.75}.

Per the spec's controlled-comparison requirement, HW-Scratch and HW-LODO
see IDENTICAL channel-removal masks for a given (window, drop_rate) pair
— achieved via a deterministic per-(window_index, drop_rate) seed rather
than a shared advancing RNG, so a difference in retention between the two
models can't be attributed to differing random masks.

R_50 = F1_at_50%_drop / F1_full * 100 is reported explicitly per model.

DESIGN NOTE: "fixed classifier trained using full-channel representations"
in the spec is implemented here as a full end-to-end HW-Scratch/HW-LODO
model (encoder + head), trained once via the same two-phase schedule used
in finetune_heterowear.py, then frozen — NOT a separate frozen-encoder
linear probe. If a linear-probe-style evaluation was intended instead,
this needs to change; flagging since the spec text doesn't fully
disambiguate.

Usage:
    python3 scripts/eval/robustness_curve_heterowear.py \
        --dataset pamap2 \
        --lodo_checkpoint checkpoints/heterowear_lodo/pamap2_step_10000.pt \
        --data_root data/processed \
        --out results/heterowear/pamap2_robustness.csv \
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
from src.data.splits import make_subject_split
from scripts.eval.finetune_heterowear import (
    HeteroWearClassifier, build_lodo_encoder, build_scratch_encoder,
    load_dataset_windows, build_label_vocab, cosine_lr,
)

DROP_RATES = [0.0, 0.25, 0.5, 0.75]


def apply_fixed_channel_dropout(channels: list, drop_rate: float, window_idx: int) -> list:
    """Deterministic per-(window_idx, drop_rate) channel removal — same
    call with the same arguments always drops the same channels, so two
    different models evaluated with the same (window_idx, drop_rate)
    always see an identical reduced channel set."""
    if drop_rate <= 0 or len(channels) <= 1:
        return channels
    n_drop = min(int(round(len(channels) * drop_rate)), len(channels) - 1)
    if n_drop <= 0:
        return channels
    seed = hash((window_idx, round(drop_rate, 4))) & 0xFFFFFFFF
    rng = random.Random(seed)
    idx_to_drop = set(rng.sample(range(len(channels)), n_drop))
    return [c for i, c in enumerate(channels) if i not in idx_to_drop]


def train_full_channel_model(encoder_builder, train_windows, val_windows, n_classes,
                              head_lr, encoder_lr, warmup_frac, max_steps, eval_every,
                              batch_size, device, seed, rng, label_to_idx):
    torch.manual_seed(seed)
    encoder = encoder_builder()
    model = HeteroWearClassifier(encoder, n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()

    n_train = len(train_windows)
    total_steps = min(max_steps, max(100, (n_train // batch_size) * 15))
    warmup_steps = max(20, int(total_steps * warmup_frac))
    finetune_steps = total_steps - warmup_steps

    def sample_windows():
        return rng.choices(train_windows, k=batch_size) if n_train < batch_size else rng.sample(train_windows, batch_size)

    def labels_for(bw):
        return torch.tensor([label_to_idx[w["activity_label"]] for w in bw], device=device, dtype=torch.long)

    for p in model.encoder.parameters():
        p.requires_grad = False
    head_opt = torch.optim.AdamW(model.head.parameters(), lr=head_lr)
    model.train()
    for _ in range(warmup_steps):
        bw = sample_windows()
        batch = [window_dict_to_channel_inputs(w) for w in bw]
        loss = loss_fn(model(batch), labels_for(bw))
        head_opt.zero_grad(); loss.backward(); head_opt.step()

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
        bw = sample_windows()
        batch = [window_dict_to_channel_inputs(w) for w in bw]
        loss = loss_fn(model(batch), labels_for(bw))
        full_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        full_opt.step()

        if (step + 1) % eval_every == 0 or step == finetune_steps - 1:
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for i in range(0, len(val_windows), batch_size):
                    chunk = val_windows[i:i + batch_size]
                    vb = [window_dict_to_channel_inputs(w) for w in chunk]
                    logits = model(vb)
                    preds.append(logits.argmax(dim=1).cpu().numpy())
                    targets.append(np.array([label_to_idx[w["activity_label"]] for w in chunk]))
            preds = np.concatenate(preds); targets = np.concatenate(targets)
            val_f1 = f1_score(targets, preds, average="macro", zero_division=0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_with_dropout(model, test_windows, label_to_idx, drop_rate, batch_size, device):
    preds, targets = [], []
    with torch.no_grad():
        for start in range(0, len(test_windows), batch_size):
            chunk = test_windows[start:start + batch_size]
            batch = []
            for i, w in enumerate(chunk):
                channels = window_dict_to_channel_inputs(w)
                channels = apply_fixed_channel_dropout(channels, drop_rate, start + i)
                batch.append(channels)
            logits = model(batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(np.array([label_to_idx[w["activity_label"]] for w in chunk]))
    preds = np.concatenate(preds); targets = np.concatenate(targets)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                     choices=["pamap2", "wisdm", "hhar", "harth", "uci_har"])
    ap.add_argument("--lodo_checkpoint", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=5e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.15)
    ap.add_argument("--max_steps", type=int, default=1200)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_scratch", action="store_true",
                     help="evaluate only HW-LODO (ablation sweep)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "model", "drop_rate", "accuracy", "macro_f1", "retention_pct",
                  "n_train", "n_val", "n_test"]
    csv_file = open(args.out, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader(); csv_file.flush()

    dataset_dir = args.data_root / args.dataset / "all"
    per_subject = load_dataset_windows(dataset_dir)
    subjects = sorted(per_subject.keys())
    print(f"[{args.dataset}] {len(subjects)} subjects with usable windows")

    split = make_subject_split(subjects, seed=args.seed)
    train_windows = [w for s in split["train"] for w in per_subject.get(s, [])]
    val_windows = [w for s in split["val"] for w in per_subject.get(s, [])]
    test_windows = [w for s in split["test"] for w in per_subject.get(s, [])]
    label_to_idx = build_label_vocab(train_windows)
    val_windows = [w for w in val_windows if w["activity_label"] in label_to_idx]
    test_windows = [w for w in test_windows if w["activity_label"] in label_to_idx]
    n_classes = len(label_to_idx)
    print(f"train={len(train_windows)} val={len(val_windows)} test={len(test_windows)} classes={n_classes}")

    _, lodo_arch_args = build_lodo_encoder(args.lodo_checkpoint, args.device)

    for model_name in (["HW-LODO"] if args.skip_scratch else ["HW-Scratch", "HW-LODO"]):
        print(f"\n=== training {model_name} (full-channel, single split) ===")
        if model_name == "HW-Scratch":
            builder = lambda: build_scratch_encoder(lodo_arch_args, args.device)
        else:
            builder = lambda: build_lodo_encoder(args.lodo_checkpoint, args.device)[0]

        model = train_full_channel_model(
            builder, train_windows, val_windows, n_classes,
            args.head_lr, args.encoder_lr, args.warmup_frac, args.max_steps, args.eval_every,
            args.batch_size, args.device, args.seed, rng, label_to_idx)

        full_f1 = None
        for d in DROP_RATES:
            r = evaluate_with_dropout(model, test_windows, label_to_idx, d, args.batch_size, args.device)
            if d == 0.0:
                full_f1 = r["macro_f1"]
            retention = (r["macro_f1"] / full_f1 * 100) if full_f1 and full_f1 > 0 else None
            row = {"dataset": args.dataset, "model": model_name, "drop_rate": d,
                   "retention_pct": retention, "n_train": len(train_windows),
                   "n_val": len(val_windows), "n_test": len(test_windows), **r}
            writer.writerow(row); csv_file.flush()
            ret_str = f"{retention:.1f}%" if retention is not None else "n/a"
            print(f"  [{model_name}] drop={d:.2f}  macro_f1={r['macro_f1']:.3f}  retention={ret_str}")

    csv_file.close()
    print(f"\nwrote results to {args.out}")


if __name__ == "__main__":
    main()
