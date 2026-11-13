#!/usr/bin/env python3
"""
HW-Flat SSL pretraining loop.

Loads windowed pickle files (from 04_export_raw_all_subjects.py) for one or
more datasets, and trains the encoder with:
  L = L_consistency + lambda_msm * L_msm
using structured sensor dropout to generate the two "views" the consistency
loss compares.

Usage:
    python3 scripts/train/pretrain.py --datasets uci_har harth pamap2 \
        --data_root data/processed --out_dir checkpoints/pretrain/run1 \
        --batch_size 32 --steps 2000 --device cuda
"""
import argparse
import math
import pickle
import random
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.hw_flat import HWFlatEncoder, window_dict_to_channel_inputs
from src.losses.structured_dropout import structured_dropout
from src.losses.consistency import cross_config_consistency_loss
from src.losses.masked_modeling import MSMHead, compute_msm_loss


def load_all_windows(data_root: Path, datasets: list[str]) -> list[dict]:
    windows = []
    for ds in datasets:
        d = data_root / ds / "all"
        if not d.exists():
            print(f"[warn] {d} does not exist, skipping {ds}")
            continue
        files = sorted(d.glob("*.pkl"))
        for f in files:
            with open(f, "rb") as fh:
                windows.extend(pickle.load(fh))
        print(f"[{ds}] loaded {len(files)} subject files")
    print(f"total windows loaded: {len(windows)}")
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True,
                     choices=["ppg_dalia", "wesad", "pamap2", "mmash", "wisdm", "hhar", "harth", "uci_har"])
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_patches", type=int, default=8)
    ap.add_argument("--samples_per_patch", type=int, default=16)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--mask_ratio", type=float, default=0.4)
    ap.add_argument("--lambda_msm", type=float, default=1.0)
    ap.add_argument("--p_modality", type=float, default=0.25)
    ap.add_argument("--p_location", type=float, default=0.25)
    ap.add_argument("--p_channel", type=float, default=0.2)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {args.device}")
    windows = load_all_windows(args.data_root, args.datasets)
    if not windows:
        print("no windows loaded, aborting")
        return

    model = HWFlatEncoder(d_model=args.d_model, n_patches=args.n_patches,
                               samples_per_patch=args.samples_per_patch,
                               n_layers=args.n_layers, n_heads=args.n_heads).to(args.device)
    msm_head = MSMHead(d_model=args.d_model).to(args.device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(msm_head.parameters()), lr=args.lr)

    model.train()
    t_start = time.time()
    running_loss, running_cons, running_msm = 0.0, 0.0, 0.0

    for step in range(1, args.steps + 1):
        if step <= args.warmup_steps:
            lr = args.lr * step / max(1, args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            lr = 0.5 * args.lr * (1 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g["lr"] = lr

        batch_windows = rng.sample(windows, min(args.batch_size, len(windows)))
        base_batch = [window_dict_to_channel_inputs(w) for w in batch_windows]

        view_a = [structured_dropout(c, args.p_modality, args.p_location, args.p_channel, rng)
                  for c in base_batch]
        view_b = [structured_dropout(c, args.p_modality, args.p_location, args.p_channel, rng)
                  for c in base_batch]

        pooled_a, _, _, _, _ = model(view_a, mask_ratio=0.0)
        pooled_b, tok_b, pad_b, mm_targets_b, mm_mask_b = model(view_b, mask_ratio=args.mask_ratio)

        cons_loss = cross_config_consistency_loss(pooled_a, pooled_b)
        msm_loss = compute_msm_loss(msm_head, tok_b, mm_targets_b, mm_mask_b)
        loss = cons_loss + args.lambda_msm * msm_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(msm_head.parameters()), max_norm=1.0)
        opt.step()

        running_loss += loss.item()
        running_cons += cons_loss.item()
        running_msm += msm_loss.item()

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            print(f"step {step:5d}/{args.steps}  loss={running_loss/args.log_every:.4f}  "
                  f"consistency={running_cons/args.log_every:.4f}  msm={running_msm/args.log_every:.4f}  "
                  f"lr={lr:.2e}  ({elapsed:.1f}s elapsed)", flush=True)
            running_loss, running_cons, running_msm = 0.0, 0.0, 0.0

        if step % args.ckpt_every == 0 or step == args.steps:
            ckpt_path = args.out_dir / f"step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "msm_head_state_dict": msm_head.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}", flush=True)

    print(f"training complete, total time {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
