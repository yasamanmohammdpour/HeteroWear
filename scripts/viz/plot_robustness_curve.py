#!/usr/bin/env python3
"""
Plots macro-F1 vs. fraction of channels dropped at inference time, one line
per dataset, from the real robustness_curve.py output CSV.

Usage:
    python3 scripts/viz/plot_robustness_curve.py \
        --csv results/robustness/curve_v2.csv \
        --out figures/robustness_curve.pdf
"""
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "pamap2": "#1f77b4", "wisdm": "#ff7f0e", "hhar": "#2ca02c",
    "harth": "#d62728", "uci_har": "#9467bd",
}
LABELS = {
    "pamap2": "PAMAP2 (9 ch.)", "wisdm": "WISDM (4 ch.)", "hhar": "HHAR (2.1 ch.)",
    "harth": "HARTH (2 ch.)", "uci_har": "UCI-HAR (2 ch.)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = defaultdict(list)
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            data[row["dataset"]].append((float(row["drop_rate"]), float(row["macro_f1"])))

    fig, ax = plt.subplots(figsize=(5, 4))
    for ds, points in data.items():
        points.sort()
        x = [p[0] * 100 for p in points]
        y = [p[1] for p in points]
        ax.plot(x, y, marker="o", label=LABELS.get(ds, ds), color=COLORS.get(ds), linewidth=2)

    ax.set_xlabel("Channels dropped at inference (%)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Missing-modality robustness")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xlim(-3, 78)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
