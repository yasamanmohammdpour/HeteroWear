#!/usr/bin/env python3
"""
Self-supervised pretraining for the SAME single-channel TSTransformer
architecture used as the supervised baseline — masked patch reconstruction,
matching the LIMU-BERT convention (masked reconstruction on IMU data) that
the Transformer baseline currently beats HW-Flat without.

This isolates a real question the results so far don't answer: does the
supervised Transformer win because task-aligned supervision beats SSL, or
because single-channel input beats multimodal input? Pretraining the exact
same architecture, on the exact same primary channels HW-Flat draws
from, removes the multimodal-vs-single-channel confound entirely.

Usage:
    python3 scripts/train/pretrain_ts_transformer.py \
        --data_root data/processed --out checkpoints/ts_pretrain/run1 \
        --steps 5000 --device cuda
"""
import argparse
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models_baseline.ts_transformer import TSTransformer
from src.losses.masked_modeling import MSMHead, compute_msm_loss
from src.eval.baseline_data import _find_channel

PRIMARY_CHANNEL_PREFIX = {
    "pamap2": "hand_ACC", "wisdm": "wrist_ACC", "hhar": "wrist_ACC",
    "harth": "back_ACC", "uci_har": "phone_ACC",
}
TARGET_LEN = {
    "pamap2": int(10.0 * 100), "wisdm": int(5.0 * 20), "hhar": int(5.0 * 100),
    "harth": int(5.0 * 50), "uci_har": 128,
}


CANONICAL_LEN = 128  # shared across ALL datasets for joint pretraining batching —
                       # per-dataset TARGET_LEN values differ (1000/100/500/250/128),
                       # so mixing datasets in one batch requires a common length;
                       # interpolation (not padding) matches how HW-Flat's own
                       # tokenizer already solves this exact cross-dataset problem


def _interpolate_to_length(arr: np.ndarray, canonical_len: int) -> np.ndarray:
    """Resample [T, C] to [canonical_len, C] via linear interpolation."""
    T, C = arr.shape
    if T == canonical_len:
        return arr.astype(np.float32)
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, canonical_len)
    out = np.zeros((canonical_len, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = np.interp(x_new, x_old, arr[:, c])
    return out


def extract_primary(w: dict, prefix: str, canonical_len: int = CANONICAL_LEN):
    arr = _find_channel(w, prefix)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] < 2:
        return None  # can't interpolate a near-empty array
    return _interpolate_to_length(arr, canonical_len)


def load_all_primary_windows(data_root: Path, datasets: list) -> list:
    arrs = []
    for ds in datasets:
        prefix = PRIMARY_CHANNEL_PREFIX[ds]
        d = data_root / ds / "all"
        if not d.exists():
            print(f"[warn] {d} missing, skipping")
            continue
        n = 0
        for f in sorted(d.glob("*.pkl")):
            with open(f, "rb") as fh:
                windows = pickle.load(fh)
            for w in windows:
                a = extract_primary(w, prefix)
                if a is not None:
                    arrs.append(a)
                    n += 1
        print(f"[{ds}] {n} usable windows")
    print(f"total: {len(arrs)} windows")
    return arrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["pamap2", "wisdm", "hhar", "harth", "uci_har"])
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--patch_len", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--mask_ratio", type=float, default=0.4)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"device: {args.device}")
    windows = load_all_primary_windows(args.data_root, args.datasets)
    if not windows:
        print("no windows loaded, aborting")
        return

    n_channels = windows[0].shape[1]
    model = TSTransformer(n_channels=n_channels, n_classes=1, d_model=args.d_model,
                           n_heads=args.n_heads, n_layers=args.n_layers, patch_len=args.patch_len).to(args.device)
    head = MSMHead(d_model=args.d_model).to(args.device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr)

    model.train()
    t0 = time.time()
    running = 0.0
    for step in range(1, args.steps + 1):
        if step <= args.warmup_steps:
            lr = args.lr * step / max(1, args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            lr = 0.5 * args.lr * (1 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g["lr"] = lr

        batch_arrs = rng.sample(windows, min(args.batch_size, len(windows)))
        x = torch.from_numpy(np.stack(batch_arrs)).to(args.device)

        pooled, tok, mm_targets, mm_mask = model(x, mask_ratio=args.mask_ratio)
        loss = compute_msm_loss(head, tok, mm_targets, mm_mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), max_norm=1.0)
        opt.step()
        running += loss.item()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step:5d}/{args.steps}  msm_loss={running/args.log_every:.4f}  "
                  f"lr={lr:.2e}  ({elapsed:.1f}s elapsed)", flush=True)
            running = 0.0

        if step % args.ckpt_every == 0 or step == args.steps:
            ckpt_path = args.out / f"step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "args": {"d_model": args.d_model, "n_layers": args.n_layers, "n_heads": args.n_heads,
                         "patch_len": args.patch_len, "n_channels": n_channels},
            }, ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}", flush=True)

    print(f"training complete, total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
