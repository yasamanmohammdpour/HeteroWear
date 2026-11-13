"""
Classical baseline: Random Forest on handcrafted per-window statistical
features (mean/std/min/max/energy per axis) — the reference point any
learned representation needs to beat.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def extract_features(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    energy = (X ** 2).mean(axis=1)
    return np.concatenate([mean, std, mn, mx, energy], axis=1)


def run_rf_baseline(X_train, y_train, X_test, y_test, seed: int = 42) -> dict:
    feat_train = extract_features(X_train)
    feat_test = extract_features(X_test)
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=seed, n_jobs=-1)
    clf.fit(feat_train, y_train)
    pred = clf.predict(feat_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "n_train": len(y_train), "n_test": len(y_test),
    }
