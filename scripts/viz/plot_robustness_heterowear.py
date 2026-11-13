#!/usr/bin/env python3
"""
Fig. 2: robustness to sensor loss, HW-Scratch vs HW-LODO.
  (a) macro-F1 vs % of channels removed at test time (thin lines: individual datasets,
      thick lines: mean over datasets)
  (b) mean performance retention F1(d) / F1(0) for each model

Reads the per-dataset CSVs written by scripts/eval/robustness_curve_heterowear.py.

    python3 scripts/viz/plot_robustness_heterowear.py \
        --results_dir results/heterowear --out figures/robustness_heterowear.pdf
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = ["pamap2", "wisdm", "hhar", "harth", "uci_har"]
STYLE = {"HW-Scratch": dict(color="#1f77b4", ls="--", marker="o"),
         "HW-LODO": dict(color="#ff7f0e", ls="-", marker="s")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, default=Path("results/heterowear"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    f1 = defaultdict(dict)  # (model, dataset) -> {drop_rate: macro_f1}
    for ds in DS:
        p = args.results_dir / f"{ds}_robustness.csv"
        if not p.exists():
            print(f"[warn] missing {p}")
            continue
        with open(p) as fh:
            for r in csv.DictReader(fh):
                f1[(r["model"], ds)][float(r["drop_rate"])] = float(r["macro_f1"])
    if not f1:
        raise SystemExit("no robustness CSVs found")

    rates = sorted({d for v in f1.values() for d in v})
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.3))
    for model, st in STYLE.items():
        per_ds = [v for (m, _), v in f1.items() if m == model and all(d in v for d in rates)]
        if not per_ds:
            continue
        for v in per_ds:
            a.plot([d * 100 for d in rates], [v[d] for d in rates], color="0.8", lw=0.8, zorder=1)
        mean_f1 = [sum(v[d] for v in per_ds) / len(per_ds) for d in rates]
        mean_ret = [sum(v[d] / v[rates[0]] for v in per_ds if v[rates[0]] > 0) / len(per_ds) * 100 for d in rates]
        a.plot([d * 100 for d in rates], mean_f1, lw=2, label=model, markersize=4, zorder=3, **st)
        b.plot([d * 100 for d in rates], mean_ret, lw=2, label=model, markersize=4, **st)

    a.set_title("(a) Controlled sensor loss", fontsize=9)
    a.set_ylabel("Macro-F1")
    b.set_title("(b) Mean retention", fontsize=9)
    b.set_ylabel("Retention (%)")
    for ax in (a, b):
        ax.set_xlabel("Sensor channels removed (%)")
        ax.set_xticks([d * 100 for d in rates])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
