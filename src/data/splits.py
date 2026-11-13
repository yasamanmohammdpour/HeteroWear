"""
Subject-disjoint splitting. This file exists specifically to make split
leakage structurally hard to cause by accident:

  - splits are computed on subject IDs, never on windows
  - splits are written to disk once and loaded thereafter (never
    regenerated inside a training script, which is how "reasonable
    random seed handling" quietly turns into leakage across reruns)
  - `assign_windows_to_split` is the ONLY function allowed to filter a
    window list by split, and it asserts the split file exists.
"""

from __future__ import annotations
import json
import random
from pathlib import Path

from src.data.schema import WindowSample


def make_subject_split(
    subject_ids: list[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """One fixed random subject-level train/val/test split. Deterministic given seed."""
    ids = sorted(subject_ids)  # sort first so shuffle is reproducible regardless of input order
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
        "seed": seed,
    }


def make_loso_folds(subject_ids: list[str]) -> list[dict]:
    """One fold per subject: that subject is test, everyone else is train.
    No val split within LOSO by default — add one only if you need it for
    early stopping, and carve it from train subjects, never from test."""
    ids = sorted(subject_ids)
    return [
        {"test": [sid], "train": [s for s in ids if s != sid], "fold_subject": sid}
        for sid in ids
    ]


def save_split(split: dict, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Split file {path} does not exist. Splits must be generated once via "
            f"scripts/preprocess/02_build_manifests.py and never regenerated ad hoc — "
            f"regenerating inside a training script risks a different shuffle each run."
        )
    with open(path) as f:
        return json.load(f)


def assign_windows_to_split(windows: list[WindowSample], split: dict, part: str) -> list[WindowSample]:
    assert part in split, f"'{part}' not in split keys {list(split.keys())}"
    allowed = set(split[part])
    out = [w for w in windows if w.subject_id in allowed]
    # Hard assertion: catch leakage immediately rather than downstream in a metric that's off by 2%.
    other_parts_ids = set()
    for k, v in split.items():
        if k != part and isinstance(v, list):
            other_parts_ids.update(v)
    leaked = {w.subject_id for w in out} & other_parts_ids
    assert not leaked, f"Subject leakage detected: {leaked} appear in both '{part}' and another split part"
    return out
