#!/usr/bin/env python3
"""
Fine-tunes and evaluates HW-Scratch (random init, supervised only) versus
HW-LODO (initialized from a leave-one-dataset-out pretrained checkpoint),
under identical LOSO protocol, identical splits, identical fine-tuning
schedule — the two conditions differ ONLY in whether heterogeneous
self-supervised pretraining was used, isolating that variable per
Section IV.A of the spec.

PAMAP2 and HHAR use exhaustive nine-fold LOSO (both have exactly 9
subjects); other datasets cap at --max_loso_folds as elsewhere in this
project.

Usage:
    python3 scripts/eval/finetune_heterowear.py \
        --dataset pamap2 \
        --lodo_checkpoint checkpoints/heterowear_lodo/pamap2_step_10000.pt \
        --data_root data/processed \
        --out results/heterowear/pamap2_results.csv \
        --device cuda
"""
import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.hw_flat import window_dict_to_channel_inputs
from src.models.heterowear import HeteroWearEncoder
from src.data.splits import make_subject_split

EXHAUSTIVE_LOSO_DATASETS = {"pamap2", "hhar"}


class HeteroWearClassifier(nn.Module):
    def __init__(self, encoder: HeteroWearEncoder, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.d_model, n_classes)

    def forward(self, batch):
        g, _, _ = self.encoder(batch, mask_ratio=0.0)
        return self.head(g)


def build_scratch_encoder(arch_args: dict, device: str) -> HeteroWearEncoder:
    return HeteroWearEncoder(
        d_model=arch_args["d_model"], n_patches=arch_args["n_patches"],
        samples_per_patch=arch_args["samples_per_patch"],
        temporal_layers=arch_args["temporal_layers"], temporal_heads=arch_args["temporal_heads"],
        cross_sensor_layers=arch_args["cross_sensor_layers"],
        cross_sensor_heads=arch_args["cross_sensor_heads"],
        n_latent_queries=arch_args["n_latent_queries"], latent_layers=arch_args["latent_layers"],
        latent_heads=arch_args["latent_heads"], use_context=arch_args.get("use_context", True),
    ).to(device)


def build_lodo_encoder(checkpoint_path: Path, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    arch_args = ckpt["args"]
    encoder = build_scratch_encoder(arch_args, device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    return encoder, arch_args


def load_dataset_windows(dataset_dir: Path) -> dict:
    import pickle
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


def cosine_lr(step: int, total_steps: int, base_lr: float) -> float:
    progress = step / max(1, total_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


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


def run_fold(encoder_builder, train_windows, val_windows, test_windows,
             head_lr, encoder_lr, warmup_frac, max_steps, eval_every,
             batch_size, device, seed, rng) -> dict:
    label_to_idx = build_label_vocab(train_windows)
    n_classes = len(label_to_idx)
    val_windows = [w for w in val_windows if w["activity_label"] in label_to_idx]
    test_windows = [w for w in test_windows if w["activity_label"] in label_to_idx]
    if not train_windows or not val_windows or not test_windows or n_classes < 2:
        return None

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
    ap.add_argument("--dataset", required=True,
                     choices=["pamap2", "wisdm", "hhar", "harth", "uci_har"])
    ap.add_argument("--lodo_checkpoint", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_loso_folds", type=int, default=6,
                     help="ignored for pamap2/hhar, which always run exhaustive 9-fold")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=5e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.15)
    ap.add_argument("--max_steps", type=int, default=1200)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_scratch", action="store_true",
                     help="evaluate only HW-LODO (used by the ablation sweep, where HW-Scratch "
                          "is identical across configs and already reported elsewhere)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "model", "eval", "accuracy", "macro_f1", "n_train", "n_val",
                  "n_test", "steps", "best_val_f1"]
    csv_file = open(args.out, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader(); csv_file.flush()

    def _write(row):
        writer.writerow(row); csv_file.flush()

    dataset_dir = args.data_root / args.dataset / "all"
    per_subject = load_dataset_windows(dataset_dir)
    subjects = sorted(per_subject.keys())
    print(f"[{args.dataset}] {len(subjects)} subjects with usable windows")

    _, lodo_arch_args = build_lodo_encoder(args.lodo_checkpoint, args.device)

    loso_subjects = subjects if args.dataset in EXHAUSTIVE_LOSO_DATASETS else subjects[:args.max_loso_folds]

    for model_name in (["HW-LODO"] if args.skip_scratch else ["HW-Scratch", "HW-LODO"]):
        f1s = []
        for test_subj in loso_subjects:
            train_pool = [s for s in subjects if s != test_subj]
            val_n = max(1, int(round(len(train_pool) * args.val_frac)))
            shuffled = train_pool[:]
            random.Random(args.seed).shuffle(shuffled)
            val_subj, train_subj = shuffled[:val_n], shuffled[val_n:]

            train_windows = [w for s in train_subj for w in per_subject.get(s, [])]
            val_windows = [w for s in val_subj for w in per_subject.get(s, [])]
            test_windows = per_subject.get(test_subj, [])

            if model_name == "HW-Scratch":
                builder = lambda: build_scratch_encoder(lodo_arch_args, args.device)
            else:
                builder = lambda: build_lodo_encoder(args.lodo_checkpoint, args.device)[0]

            r = run_fold(builder, train_windows, val_windows, test_windows,
                         args.head_lr, args.encoder_lr, args.warmup_frac, args.max_steps,
                         args.eval_every, args.batch_size, args.device, args.seed, rng)
            if r is None:
                continue
            f1s.append(r["macro_f1"])
            _write({"dataset": args.dataset, "model": model_name, "eval": f"loso_{test_subj}", **r})
            print(f"  [{model_name}] fold {test_subj}: macro_f1={r['macro_f1']:.3f}")

        if f1s:
            print(f"[{model_name}] LOSO x{len(f1s)}: macro_f1={np.mean(f1s):.3f}+/-{np.std(f1s):.3f}")
            _write({"dataset": args.dataset, "model": model_name, "eval": "loso_mean",
                    "macro_f1": float(np.mean(f1s))})

    csv_file.close()
    print(f"\nwrote results to {args.out}")


if __name__ == "__main__":
    main()
