"""
Normalization statistics must be fit on the TRAIN split only, then applied
to val/test. Fitting on the full dataset (a common shortcut) leaks test
distribution information into preprocessing — small effect for accuracy
metrics, but a real one for HR MAE/RMSE where you're reporting a small
number to two decimal places.

Two granularities are supported:
  - "dataset": one mean/std per channel_key, pooled across train subjects.
    Use for datasets where per-subject stats would be too noisy (e.g. HAR
    datasets with few windows per subject).
  - "subject": per-subject per-channel stats — appropriate for HR/PPG work
    where absolute physiological baselines differ meaningfully per person
    (this is also the basis for the HR_z personalization score in the
     original proposal: (HR_t - mu_s) / sigma_s).
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from src.data.schema import WindowSample


def fit_normalization_stats(
    train_windows: list[WindowSample],
    granularity: str = "dataset",
) -> dict:
    """Returns {channel_key: {"mean": [...], "std": [...]}} or, for
    granularity="subject", {subject_id: {channel_key: {...}}}."""
    assert granularity in ("dataset", "subject")

    def _accumulate(bucket: dict, key: str, arr: np.ndarray):
        bucket.setdefault(key, []).append(arr)

    if granularity == "dataset":
        acc: dict[str, list[np.ndarray]] = {}
        for w in train_windows:
            for ch_key, arr in w.signals.items():
                _accumulate(acc, ch_key, arr)
        return {
            ch_key: {
                "mean": np.concatenate(arrs, axis=0).mean(axis=0).tolist(),
                "std": (np.concatenate(arrs, axis=0).std(axis=0) + 1e-8).tolist(),
            }
            for ch_key, arrs in acc.items()
        }

    # per-subject
    per_subject: dict[str, dict[str, list[np.ndarray]]] = {}
    for w in train_windows:
        subj_acc = per_subject.setdefault(w.subject_id, {})
        for ch_key, arr in w.signals.items():
            _accumulate(subj_acc, ch_key, arr)
    return {
        subj: {
            ch_key: {
                "mean": np.concatenate(arrs, axis=0).mean(axis=0).tolist(),
                "std": (np.concatenate(arrs, axis=0).std(axis=0) + 1e-8).tolist(),
            }
            for ch_key, arrs in acc.items()
        }
        for subj, acc in per_subject.items()
    }


def apply_normalization(window: WindowSample, stats: dict, granularity: str = "dataset") -> WindowSample:
    lookup = stats.get(window.subject_id, {}) if granularity == "subject" else stats
    for ch_key, arr in window.signals.items():
        if ch_key not in lookup:
            continue  # channel unseen in train stats (e.g. rare channel) — leave raw, flag downstream
        mean = np.array(lookup[ch_key]["mean"])
        std = np.array(lookup[ch_key]["std"])
        window.signals[ch_key] = (arr - mean) / std
    return window


def save_stats(stats: dict, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
