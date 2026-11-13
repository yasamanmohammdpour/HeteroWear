#!/usr/bin/env python3
"""
Grouped bar chart of LOSO macro-F1 across all five datasets, combining the
baseline suite, the single-channel SSL ablation, and HW-Flat (best of
linear probe or fine-tuned, per dataset).

Usage:
    python3 scripts/viz/plot_model_comparison.py \
        --baselines results/baselines/har_baselines_v2.csv \
        --ts_ssl results/ts_ssl/results.csv \
        --linear_probe results/linear_probe/results_v2.csv \
        --finetune results/finetune/results_v3.csv \
        --out figures/model_comparison.pdf
"""
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["pamap2", "wisdm", "hhar", "harth", "uci_har"]
DATASET_LABELS = ["PAMAP2", "WISDM", "HHAR", "HARTH", "UCI-HAR"]
MODEL_ORDER = ["RF", "CNN", "DeepConvLSTM", "Transformer", "Transformer_SSL", "HW-Flat"]
COLORS = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]


def read_loso_mean(csv_path: Path, model_filter=None) -> dict:
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("eval") != "loso_mean":
                continue
            model = row.get("model", model_filter)
            if model_filter and model != model_filter:
                continue
            try:
                out[(row["dataset"], model)] = float(row["macro_f1"])
            except (ValueError, KeyError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", type=Path, required=True)
    ap.add_argument("--ts_ssl", type=Path, required=True)
    ap.add_argument("--linear_probe", type=Path, required=True)
    ap.add_argument("--finetune", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    values = {}
    values.update(read_loso_mean(args.baselines))
    values.update(read_loso_mean(args.ts_ssl))

    probe = read_loso_mean(args.linear_probe, model_filter="HW-Flat_LinearProbe")
    ft = read_loso_mean(args.finetune, model_filter="HW-Flat_FineTuned")
    for ds in DATASETS:
        p = probe.get((ds, "HW-Flat_LinearProbe"))
        f = ft.get((ds, "HW-Flat_FineTuned"))
        candidates = [v for v in (p, f) if v is not None]
        if candidates:
            values[(ds, "HW-Flat")] = max(candidates)

    fig, ax = plt.subplots(figsize=(9, 4))
    n_models = len(MODEL_ORDER)
    width = 0.8 / n_models
    x = np.arange(len(DATASETS))

    for i, model in enumerate(MODEL_ORDER):
        heights = [values.get((ds, model), 0.0) for ds in DATASETS]
        offsets = x + (i - n_models / 2 + 0.5) * width
        ax.bar(offsets, heights, width=width * 0.9, label=model, color=COLORS[i])

    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS)
    ax.set_ylabel("LOSO macro-F1")
    ax.set_title("Model comparison across all five datasets")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"saved -> {args.out}")
    missing = [(ds, m) for ds in DATASETS for m in MODEL_ORDER if (ds, m) not in values]
    if missing:
        print(f"[warn] no data found for: {missing}")


if __name__ == "__main__":
    main()
