#!/usr/bin/env python3
"""
Schematic architecture diagram. Pure schematic, no external data required.

Usage:
    python3 scripts/viz/plot_architecture.py --out figures/architecture.pdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


def box(ax, xy, w, h, text, fc="#e8eef7", ec="#2c3e6b", fontsize=8.5):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                        facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
             fontsize=fontsize, wrap=True)
    return b


def arrow(ax, start, end, color="#333333"):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                         color=color, linewidth=1.3)
    ax.add_patch(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ch_labels = ["ACC\nwrist", "GYRO\nchest", "PPG\nwrist", "ECG\nchest", "\u22ef"]
    for i, lbl in enumerate(ch_labels):
        box(ax, (0.2, 6.3 - i * 0.75), 1.3, 0.65, lbl, fc="#fdf3e7", ec="#b5651d", fontsize=7)

    box(ax, (2.1, 4.9), 1.9, 2.0, "Per-modality\npatch tokenizer\n(interpolate + strided conv,\nper-instance normalized)",
        fc="#eaf4ea", ec="#2e7d32")

    box(ax, (4.5, 4.9), 1.9, 2.0,
        "Context\nembeddings\n$E_{mod}{+}E_{loc}{+}E_{dev}$\n$+E_{rate}{+}E_{time}$",
        fc="#eef0fb", ec="#4b3f9e")

    box(ax, (6.9, 4.7), 2.4, 2.4,
        "Shared Transformer\nencoder\n(4 layers, 4 heads,\npadding-masked)",
        fc="#fdeaea", ec="#a31515")

    arrow(ax, (1.5, 5.6), (2.1, 5.8))
    arrow(ax, (4.0, 5.9), (4.5, 5.9))
    arrow(ax, (6.4, 5.9), (6.9, 5.9))

    obj_y = 1.0
    box(ax, (0.5, obj_y), 2.6, 1.1,
        "Structured sensor\ndropout\n(two augmented views)", fc="#fff6e0", ec="#c99400", fontsize=7.5)
    box(ax, (3.6, obj_y), 2.6, 1.1,
        "Cross-configuration\nconsistency\n(InfoNCE on the two views)", fc="#fff6e0", ec="#c99400", fontsize=7.5)
    box(ax, (6.7, obj_y), 2.6, 1.1,
        "Masked sensor\nmodeling\n(per-axis mean/std target)", fc="#fff6e0", ec="#c99400", fontsize=7.5)

    arrow(ax, (1.8, obj_y + 1.1), (2.6, 4.9), color="#c99400")
    arrow(ax, (4.9, obj_y + 1.1), (5.4, 4.9), color="#c99400")
    arrow(ax, (8.0, obj_y + 1.1), (8.0, 4.7), color="#c99400")

    legend_elems = [
        Line2D([0], [0], color="#333333", lw=1.3, label="forward pass"),
        Line2D([0], [0], color="#c99400", lw=1.3, label="pretraining objective"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8, frameon=False)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
