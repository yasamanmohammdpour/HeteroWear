#!/usr/bin/env python3
"""
Rebuilds the paper's result tables from the CSVs the experiment scripts write.

    python3 scripts/analysis/make_tables.py                       # print markdown to stdout
    python3 scripts/analysis/make_tables.py --out results/tables.md

Inputs (missing files are reported as "n/a", never silently skipped):

  results/reference/                      HW-Flat / baseline runs (local only, git-ignored)
      har_baselines_v2.csv                RF, CNN, DeepConvLSTM, Transformer     (Table III)
      linear_probe_v2.csv, finetune_v3.csv  HW-Flat probe / fine-tune           (Table III)
      ts_ssl.csv                          single-channel SSL ablation            (Table VIII)
      fusion.csv                          representation-fusion control          (Sec. VI-G)
  results/heterowear/                     written by the jobs in jobs/
      <ds>_results.csv                    HW-Scratch vs HW-LODO, LOSO            (Table IV)
      <ds>_device_holdout.csv             held-out-device transfer               (Table V)
      <ds>_robustness.csv                 sensor-removal curve                   (Table VI)
      ablation/hhar_<cfg>.csv, ablation/hhar_<cfg>_robustness.csv               (Table VII)
      seeds/seed<k>_<ds>.csv              pretraining-seed stability

Retention follows the paper's definition (Eq. 17 / Table VI): F1 at 50% channel removal divided
by the full-channel macro-F1 of the same model **from Table IV (LOSO mean)**. The robustness
script's own `retention_pct` column instead divides by the single-split full-channel F1 of that
run; both are shown in the Table VI block so the difference is visible.
"""
import argparse
import csv
import statistics as st
from pathlib import Path

DS = ["pamap2", "wisdm", "hhar", "harth", "uci_har"]
DS_NAME = {"pamap2": "PAMAP2", "wisdm": "WISDM", "hhar": "HHAR", "harth": "HARTH", "uci_har": "UCI-HAR"}


def read(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def pct(x):
    return "n/a" if x is None else f"{x:.1f}%"


def mean(xs):
    """Strict mean: None unless every value is present, so partial runs never
    produce a misleading 'Mean' row."""
    xs = list(xs)
    return None if not xs or any(x is None for x in xs) else sum(xs) / len(xs)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def loso_mean(rows, ds, model):
    if rows is None:
        return None
    for r in rows:
        if r["dataset"] == ds and r["model"] == model and r["eval"] == "loso_mean":
            return float(r["macro_f1"])
    return None


def table3(ref):
    base = read(ref / "har_baselines_v2.csv")
    probe = read(ref / "linear_probe_v2.csv")
    ft = read(ref / "finetune_v3.csv")
    rows = []
    for ds in DS:
        v = [loso_mean(base, ds, m) for m in ("RF", "CNN", "DeepConvLSTM", "Transformer")]
        p, t = loso_mean(probe, ds, "HW-Flat_LinearProbe") or loso_mean(probe, ds, "CardioWear_LinearProbe"), \
            loso_mean(ft, ds, "HW-Flat_FineTuned") or loso_mean(ft, ds, "CardioWear_FineTuned")
        flat = max([x for x in (p, t) if x is not None], default=None)  # "best of probe / fine-tune"
        rows.append([DS_NAME[ds]] + [f(x) for x in v] + [f(flat)])
    return "### Table III - reference comparison (subject-level macro-F1)\n" + \
        table(["Dataset", "RF", "CNN", "DCLSTM", "Transf.", "HW-Flat"], rows)


def table4(hw, ref):
    base = read(ref / "har_baselines_v2.csv")
    rows, d_ssl, d_t = [], [], []
    scratch, lodo = {}, {}
    for ds in DS:
        r = read(hw / f"{ds}_results.csv")
        tr = loso_mean(base, ds, "Transformer")
        s, l = loso_mean(r, ds, "HW-Scratch"), loso_mean(r, ds, "HW-LODO")
        scratch[ds], lodo[ds] = s, l
        ds_ssl = None if s is None or l is None else l - s
        ds_t = None if tr is None or l is None else l - tr
        d_ssl.append(ds_ssl); d_t.append(ds_t)
        rows.append([DS_NAME[ds], f(tr), f(s), f(l), "n/a" if ds_ssl is None else f"{ds_ssl:+.3f}",
                     "n/a" if ds_t is None else f"{ds_t:+.3f}"])
    ms, ml = mean(scratch.values()), mean(lodo.values())
    mt = mean([loso_mean(base, ds, "Transformer") for ds in DS])
    rows.append(["Mean", f(mt), f(ms), f(ml),
                 "n/a" if mean(d_ssl) is None else f"{mean(d_ssl):+.3f}",
                 "n/a" if mean(d_t) is None else f"{mean(d_t):+.3f}"])
    return "### Table IV - controlled downstream macro-F1 (LOSO)\n" + \
        table(["Data", "Trans.", "Scratch", "LODO", "dSSL", "dT"], rows), scratch, lodo


def table5(hw):
    rows, ds_all = [], []
    for ds in ("wisdm", "hhar"):
        r = read(hw / f"{ds}_device_holdout.csv")
        for tr, te, label in (("smartphone", "smartwatch", "phone -> watch"),
                              ("smartwatch", "smartphone", "watch -> phone")):
            vals = {}
            for row in r or []:
                if row["train_device"] == tr and row["test_device"] == te:
                    vals[row["model"]] = float(row["macro_f1"])
            s, l = vals.get("HW-Scratch"), vals.get("HW-LODO")
            d = None if s is None or l is None else l - s
            ds_all.append((s, l, d))
            rows.append([DS_NAME[ds], label, f(s), f(l), "n/a" if d is None else f"{d:+.3f}"])
    rows.append(["Mean", "-", f(mean([a for a, _, _ in ds_all])), f(mean([b for _, b, _ in ds_all])),
                 "n/a" if mean([c for _, _, c in ds_all]) is None else f"{mean([c for _, _, c in ds_all]):+.3f}"])
    return "### Table V - held-out-device macro-F1 (device-class split)\n" + \
        table(["Dataset", "Transfer", "HW-Scratch", "HeteroWear", "delta"], rows)


def f1_at(rows, model, rate):
    for r in rows or []:
        if r["model"] == model and abs(float(r["drop_rate"]) - rate) < 1e-9:
            return float(r["macro_f1"]), (float(r["retention_pct"]) if r["retention_pct"] not in ("", None) else None)
    return None, None


def table6(hw, scratch, lodo):
    rows, rs, rl = [], [], []
    for ds in DS:
        r = read(hw / f"{ds}_robustness.csv")
        s50, s_own = f1_at(r, "HW-Scratch", 0.5)
        l50, l_own = f1_at(r, "HW-LODO", 0.5)
        s_ret = None if s50 is None or not scratch.get(ds) else s50 / scratch[ds] * 100
        l_ret = None if l50 is None or not lodo.get(ds) else l50 / lodo[ds] * 100
        rs.append(s_ret); rl.append(l_ret)
        rows.append([DS_NAME[ds], f(scratch.get(ds)), f(s50), pct(s_ret), f(lodo.get(ds)), f(l50), pct(l_ret),
                     f"{pct(s_own)} / {pct(l_own)}"])
    rows.append(["Mean", "-", "-", pct(mean(rs)), "-", "-", pct(mean(rl)), "-"])
    return "### Table VI - robustness under 50% test-time sensor removal\n" + \
        table(["Data", "S-Full", "S-50", "S-Ret.", "L-Full", "L-50", "L-Ret.", "script retention S / L"], rows)


def table7(hw, scratch, lodo):
    abl = hw / "ablation"
    configs = [("HeteroWear, MSM only", "msm_only"),
               ("HW + consistency, no struct. dropout", "no_dropout"),
               ("HW + dropout, no consistency", "no_consistency"),
               ("Full HeteroWear", "full"),
               ("Full HW, no context embeddings", "no_context")]
    # first row: full-channel scratch (HW-Scratch from Tables IV / VI)
    hh = read(hw / "hhar_robustness.csv")
    s50, _ = f1_at(hh, "HW-Scratch", 0.5)
    s_full = scratch.get("hhar")
    rows = [["Full-channel scratch", f(s_full), f(s50),
             pct(None if not s50 or not s_full else s50 / s_full * 100)]]
    for label, cfg in configs:
        full = loso_mean(read(abl / f"hhar_{cfg}.csv"), "hhar", "HW-LODO")
        l50, _ = f1_at(read(abl / f"hhar_{cfg}_robustness.csv"), "HW-LODO", 0.5)
        rows.append([label, f(full), f(l50), pct(None if not l50 or not full else l50 / full * 100)])
    return "### Table VII - mechanism ablation on HHAR\n" + table(["Configuration", "Macro-F1", "F1 @50%", "Retention"], rows)


def seeds(hw):
    rows = []
    for ds in ("hhar", "pamap2", "wisdm"):
        vals = [loso_mean(read(hw / "seeds" / f"seed{s}_{ds}.csv"), ds, "HW-LODO") for s in (42, 123, 7)]
        ok = [v for v in vals if v is not None]
        rows.append([DS_NAME[ds]] + [f(v) for v in vals] +
                    [f"{st.mean(ok):.3f} +/- {st.stdev(ok):.3f}" if len(ok) > 1 else "n/a"])
    return "### Pretraining-seed stability (HW-LODO macro-F1)\n" + \
        table(["Dataset", "seed 42", "seed 123", "seed 7", "mean +/- std"], rows)


def table8_and_fusion(ref):
    tr = read(ref / "har_baselines_v2.csv")
    ssl = read(ref / "ts_ssl.csv")
    fus = read(ref / "fusion.csv")
    ft, pr = read(ref / "finetune_v3.csv"), read(ref / "linear_probe_v2.csv")
    rows8, rowsf = [], []
    for ds in DS:
        a, b = loso_mean(tr, ds, "Transformer"), loso_mean(ssl, ds, "Transformer_SSL")
        rows8.append([DS_NAME[ds], f(a), f(b), "n/a" if a is None or b is None else f"{b - a:+.3f}"])
        flat = max([x for x in (loso_mean(ft, ds, "HW-Flat_FineTuned") or loso_mean(ft, ds, "CardioWear_FineTuned"),
                                loso_mean(pr, ds, "HW-Flat_LinearProbe") or loso_mean(pr, ds, "CardioWear_LinearProbe"))
                    if x is not None], default=None)
        rowsf.append([DS_NAME[ds], f(flat), f(loso_mean(fus, ds, "Fusion"))])
    return ("### Table VIII - single-channel masked-reconstruction SSL\n" +
            table(["Dataset", "Transf. scratch", "Transf.SSL", "delta"], rows8) +
            "\n\n### Representation fusion control\n" + table(["Dataset", "HW-Flat", "Fusion"], rowsf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    ref, hw = args.results_dir / "reference", args.results_dir / "heterowear"

    t4, scratch, lodo = table4(hw, ref)
    blocks = [table3(ref), t4, table5(hw), table6(hw, scratch, lodo), table7(hw, scratch, lodo),
              seeds(hw), table8_and_fusion(ref)]
    text = "\n\n".join(blocks) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
