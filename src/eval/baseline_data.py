"""
Loads the raw "all subjects" export (from 04_export_raw_all_subjects.py)
and produces fixed-length (X, y) arrays per subject for a single
"primary channel" — this is what the classical/deep BASELINES train on
(a single accelerometer stream), not the full multimodal HW-Flat input.

Each dataset uses a different channel-key naming convention, so channels
are matched by PREFIX, not exact key, and picked as the first matching
channel present in a given window.
"""

from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np

# Undefined/transition labels — not real activity classes, standard practice
# is to exclude them from classification (e.g. PAMAP2 activity ID 0
# "transient" is 36% of the raw label stream and represents unlabeled
# transition periods, not a coherent 13th class).
EXCLUDED_LABELS = {"transient", "transition"}


def _is_excluded_label(label: str) -> bool:
    return label in EXCLUDED_LABELS or label.startswith("unknown_")


PRIMARY_CHANNEL_PREFIX = {
    "pamap2": "hand_ACC",       # PAMAP2 adapter uses "hand" not "wrist" for this location
    "wisdm": "wrist_ACC",       # watch accelerometer
    "hhar": "wrist_ACC",        # watch accelerometer
    "harth": "back_ACC",
    "uci_har": "phone_ACC",
}


def _find_channel(window_dict: dict, prefix: str):
    for key in window_dict["channel_keys"]:
        if key.startswith(prefix):
            return window_dict[f"sig__{key}"]
    return None


def load_subject_windows(pkl_path: Path, primary_prefix: str, target_len: int):
    with open(pkl_path, "rb") as f:
        window_dicts = pickle.load(f)

    X, y = [], []
    for w in window_dicts:
        if not w["activity_label"] or _is_excluded_label(w["activity_label"]):
            continue
        arr = _find_channel(w, primary_prefix)
        if arr is None:
            continue
        n = arr.shape[0]
        if n >= target_len:
            arr = arr[:target_len]
        else:
            pad = np.zeros((target_len - n, arr.shape[1]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=0)
        X.append(arr)
        y.append(w["activity_label"])

    if not X:
        return np.zeros((0, target_len, 3), dtype=np.float32), np.zeros((0,), dtype=object)
    return np.stack(X).astype(np.float32), np.array(y, dtype=object)


def load_all_subjects(dataset_dir: Path, primary_prefix: str, target_len: int):
    out = {}
    for pkl_path in sorted(dataset_dir.glob("*.pkl")):
        subj = pkl_path.stem
        X, y = load_subject_windows(pkl_path, primary_prefix, target_len)
        if len(X) > 0:
            out[subj] = (X, y)
    return out


def fit_normalize(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def build_split(per_subject: dict, train_subjects: list, test_subjects: list):
    def _concat(subj_list):
        Xs, ys = [], []
        for s in subj_list:
            if s in per_subject:
                X, y = per_subject[s]
                Xs.append(X); ys.append(y)
        if not Xs:
            return np.zeros((0,)), np.zeros((0,), dtype=object)
        return np.concatenate(Xs), np.concatenate(ys)

    X_train, y_train_str = _concat(train_subjects)
    X_test, y_test_str = _concat(test_subjects)

    if len(X_train) == 0 or len(X_test) == 0:
        return None

    label_vocab = sorted(set(y_train_str.tolist()))
    label_to_idx = {lbl: i for i, lbl in enumerate(label_vocab)}

    y_train = np.array([label_to_idx[l] for l in y_train_str], dtype=np.int64)
    keep_test = np.array([l in label_to_idx for l in y_test_str], dtype=bool)
    X_test = X_test[keep_test]
    y_test = np.array([label_to_idx[l] for l in y_test_str[keep_test]], dtype=np.int64)

    if len(X_test) == 0:
        return None
    X_train, X_test = fit_normalize(X_train, X_test)
    return X_train, y_train, X_test, y_test, label_vocab
