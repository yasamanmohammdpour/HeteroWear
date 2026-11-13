#!/usr/bin/env python3
"""
One-time split of HHAR's four monolithic CSVs (~3.2GB, 34M rows total)
into per-user files. Without this, extracting even one subject requires
a full scan of every file — and that cost repeats for all 9 subjects.
After this runs once, hhar.py's adapter reads a small per-user file
directly instead of re-scanning everything.

Run once, from the repo root:
    python3 scripts/preprocess/01_split_hhar_by_user.py \
        --hhar_dir data/raw/hhar
"""
import argparse
from pathlib import Path
import pandas as pd

FILES = ["Phones_accelerometer.csv", "Phones_gyroscope.csv",
         "Watch_accelerometer.csv", "Watch_gyroscope.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hhar_dir", required=True, type=Path)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    args = ap.parse_args()

    src_dir = args.hhar_dir / "Activity recognition exp"
    out_dir = args.hhar_dir / "by_user"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        fpath = src_dir / fname
        if not fpath.exists():
            print(f"[skip] {fpath} not found")
            continue
        print(f"=== splitting {fname} ===")
        header_written = set()
        n_chunks = 0
        for chunk in pd.read_csv(fpath, chunksize=args.chunksize):
            for user, grp in chunk.groupby("User"):
                user_dir = out_dir / str(user)
                user_dir.mkdir(parents=True, exist_ok=True)
                out_path = user_dir / fname
                write_header = user not in header_written
                grp.to_csv(out_path, mode="a", header=write_header, index=False)
                header_written.add(user)
            n_chunks += 1
            print(f"  chunk {n_chunks} processed ({args.chunksize * n_chunks:,} rows so far)")
        print(f"=== done {fname}: {len(header_written)} users ===")

    print("all files split. Per-user files under:", out_dir)


if __name__ == "__main__":
    main()
