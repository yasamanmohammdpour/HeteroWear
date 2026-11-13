#!/usr/bin/env python3
"""
Plots consistency and masked-sensor-modeling loss vs. training step, parsed
directly from the pretraining job logs already on disk — no re-running
required. Handles both log formats seen in this project: the original
flat-learning-rate run (no "lr=" field) and the later cosine-scheduled run
(has "lr=" field), so the same script produces Figure 2 (training curve
comparison) from the real logs of both runs.

Usage:
    python3 plot_training_curves.py \
        --run1_log logs/pretrain_61263246.out \
        --run2_log logs/pretrain_v2_61766995.out \
        --out figures/training_curves.pdf
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s+loss=([\d.]+)\s+consistency=([\d.]+)\s+msm=([\d.]+)"
    r"(?:\s+lr=([\d.eE+-]+))?\s+\(([\d.]+)s elapsed\)"
)


def parse_log(path: Path):
    steps, cons, msm = [], [], []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                cons.append(float(m.group(4)))
                msm.append(float(m.group(5)))
    return steps, cons, msm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1_log", type=Path, required=True,
                     help="original flat-LR / pooled-target pretraining log")
    ap.add_argument("--run2_log", type=Path, required=True,
                     help="cosine-scheduled / per-axis-target pretraining log")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    s1, c1, m1 = parse_log(args.run1_log)
    s2, c2, m2 = parse_log(args.run2_log)
    print(f"run1: parsed {len(s1)} log lines from {args.run1_log}")
    print(f"run2: parsed {len(s2)} log lines from {args.run2_log}")
    if not s1 or not s2:
        raise SystemExit("no log lines matched — check the log file paths/format")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    axes[0].plot(s1, c1, label="Run 1: flat LR, pooled target", color="#888888", linewidth=1.5)
    axes[0].plot(s2, c2, label="Run 2: cosine LR, per-axis target", color="#1f77b4", linewidth=1.5)
    axes[0].axhline(y=4.159, color="gray", linestyle=":", linewidth=1, label=r"random chance ($\ln 64$)")
    if s1:
        axes[0].axvline(x=max(s1), color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Consistency loss")
    axes[0].set_title("Cross-configuration consistency")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(alpha=0.3)

    # Log scale is necessary here, not cosmetic: run1's MSM loss stays in the
    # ~1200-1600 range (unnormalized reconstruction target, never converges)
    # while run2's converges to a floor near 0.26 -- roughly a 5000x gap. On
    # a shared linear axis, run2's entire trajectory is crushed into a few
    # pixels against zero and its internal dynamics (does it actually
    # converge, or just start low?) become invisible. Log scale is the
    # standard fix for two curves spanning several orders of magnitude.
    axes[1].plot(s1, m1, label="Run 1: flat LR, pooled target", color="#888888", linewidth=1.5)
    axes[1].plot(s2, m2, label="Run 2: cosine LR, per-axis target", color="#d62728", linewidth=1.5)
    if s1:
        axes[1].axvline(x=max(s1), color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Masked-sensor-modeling loss (log scale)")
    axes[1].set_title("Masked sensor modeling")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
