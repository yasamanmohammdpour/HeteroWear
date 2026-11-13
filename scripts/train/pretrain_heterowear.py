#!/usr/bin/env python3
"""
Leave-one-dataset-out (LODO) pretraining for HeteroWear.

For each activity-labeled downstream dataset D in {pamap2, wisdm, hhar,
harth, uci_har}, pretrains a fresh HeteroWearEncoder on the OTHER seven
datasets only (all labeled-eval datasets except D, plus the three
pretrain-only datasets), then saves a checkpoint named for the excluded
dataset. This produces five checkpoints, one per downstream target,
distinguishing genuine cross-dataset transfer from transductive
pretraining on the eventual target distribution — D's windows are never
seen, labeled or unlabeled, during that checkpoint's pretraining.

Also supports --mode joint (no exclusion, all eight datasets — this is
the HW-Flat-style reference checkpoint) and --seed for the multi-seed
stability runs.

Usage:
    # LODO: produces 5 checkpoints, one per excluded downstream dataset
    python3 scripts/train/pretrain_heterowear.py \
        --mode lodo --data_root data/processed \
        --out_dir checkpoints/heterowear_lodo --steps 10000 --device cuda

    # joint reference checkpoint (all 8 datasets, no exclusion)
    python3 scripts/train/pretrain_heterowear.py \
        --mode joint --data_root data/processed \
        --out_dir checkpoints/heterowear_joint --steps 10000 --device cuda
"""
import argparse
import math
import pickle
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.hw_flat import window_dict_to_channel_inputs
from src.models.heterowear import HeteroWearEncoder
from src.losses.structured_dropout import structured_dropout
from src.losses.consistency import cross_config_consistency_loss
from src.losses.masked_modeling import MSMHead

ALL_DATASETS = ["ppg_dalia", "wesad", "pamap2", "mmash", "wisdm", "hhar", "harth", "uci_har"]
DOWNSTREAM_DATASETS = ["pamap2", "wisdm", "hhar", "harth", "uci_har"]


def load_all_windows(data_root: Path, datasets: list) -> list:
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
        print(f"  [{ds}] loaded {len(files)} subject files")
    return windows


def run_one_pretraining(datasets: list, args, out_dir: Path, run_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== pretraining run: {run_name} ===")
    print(f"datasets included: {datasets}")
    windows = load_all_windows(args.data_root, datasets)
    print(f"total windows loaded: {len(windows)}")
    if not windows:
        print(f"[warn] no windows loaded for {run_name}, skipping")
        return

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    model = HeteroWearEncoder(
        d_model=args.d_model, n_patches=args.n_patches, samples_per_patch=args.samples_per_patch,
        temporal_layers=args.temporal_layers, temporal_heads=args.temporal_heads,
        cross_sensor_layers=args.cross_sensor_layers, cross_sensor_heads=args.cross_sensor_heads,
        n_latent_queries=args.n_latent_queries, latent_layers=args.latent_layers,
        latent_heads=args.latent_heads, use_context=not args.disable_context,
    ).to(args.device)
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

        g_a, _, _ = model(view_a, mask_ratio=0.0)
        g_b, msm_reprs, msm_targets = model(view_b, mask_ratio=args.mask_ratio)

        cons_loss = cross_config_consistency_loss(g_a, g_b)
        if msm_reprs is not None and msm_reprs.shape[0] > 0:
            preds = msm_head.proj(msm_reprs)
            msm_loss = torch.nn.functional.mse_loss(preds, msm_targets)
        else:
            msm_loss = torch.tensor(0.0, device=args.device)

        loss = args.lambda_cons * cons_loss + args.lambda_msm * msm_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(msm_head.parameters()), max_norm=1.0)
        opt.step()

        running_loss += loss.item()
        running_cons += cons_loss.item()
        running_msm += msm_loss.item()

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            print(f"  [{run_name}] step {step:5d}/{args.steps}  loss={running_loss/args.log_every:.4f}  "
                  f"consistency={running_cons/args.log_every:.4f}  msm={running_msm/args.log_every:.4f}  "
                  f"lr={lr:.2e}  ({elapsed:.1f}s elapsed)", flush=True)
            running_loss, running_cons, running_msm = 0.0, 0.0, 0.0

        if step % args.ckpt_every == 0 or step == args.steps:
            ckpt_path = out_dir / f"{run_name}_step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "msm_head_state_dict": msm_head.state_dict(),
                "args": {"d_model": args.d_model, "n_patches": args.n_patches,
                         "samples_per_patch": args.samples_per_patch,
                         "temporal_layers": args.temporal_layers, "temporal_heads": args.temporal_heads,
                         "cross_sensor_layers": args.cross_sensor_layers,
                         "cross_sensor_heads": args.cross_sensor_heads,
                         "n_latent_queries": args.n_latent_queries, "latent_layers": args.latent_layers,
                         "latent_heads": args.latent_heads, "use_context": not args.disable_context},
                "excluded_dataset": run_name if run_name in DOWNSTREAM_DATASETS else None,
                "datasets_included": datasets,
            }, ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}", flush=True)

    print(f"=== {run_name} complete, total time {time.time() - t_start:.1f}s ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lodo", "joint"], required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_patches", type=int, default=8)
    ap.add_argument("--samples_per_patch", type=int, default=16)
    ap.add_argument("--temporal_layers", type=int, default=2)
    ap.add_argument("--temporal_heads", type=int, default=4)
    ap.add_argument("--cross_sensor_layers", type=int, default=4)
    ap.add_argument("--cross_sensor_heads", type=int, default=4)
    ap.add_argument("--n_latent_queries", type=int, default=4)
    ap.add_argument("--latent_layers", type=int, default=1)
    ap.add_argument("--latent_heads", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=400)
    ap.add_argument("--mask_ratio", type=float, default=0.4)
    ap.add_argument("--lambda_msm", type=float, default=1.0)
    ap.add_argument("--lambda_cons", type=float, default=1.0)
    ap.add_argument("--disable_context", action="store_true",
                     help="ablation: skip sensor context embeddings (modality/location/device/rate)")
    ap.add_argument("--p_modality", type=float, default=0.25)
    ap.add_argument("--p_location", type=float, default=0.25)
    ap.add_argument("--p_channel", type=float, default=0.2)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only_dataset", type=str, default=None,
                     help="in lodo mode, run only this one excluded-dataset config "
                          "(useful for splitting the 5 LODO runs across separate Slurm jobs)")
    args = ap.parse_args()

    print(f"device: {args.device}")

    if args.mode == "joint":
        run_one_pretraining(ALL_DATASETS, args, args.out_dir, "joint")
        return

    targets = [args.only_dataset] if args.only_dataset else DOWNSTREAM_DATASETS
    for excluded in targets:
        included = [d for d in ALL_DATASETS if d != excluded]
        run_one_pretraining(included, args, args.out_dir, excluded)


if __name__ == "__main__":
    main()
