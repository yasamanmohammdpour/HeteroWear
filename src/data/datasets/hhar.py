"""
HHAR adapter.

Expected layout: {raw_root}/{Phones|Watch}_{accelerometer|gyroscope}.csv
    Columns (standard HHAR release): Index, Arrival_Time, Creation_Time, x, y, z,
    User, Model, Device, gt (ground-truth activity: bike/sit/stand/walk/
    stairsup/stairsdown/null)

Sampling is IRREGULAR (real device timestamps, not a fixed clock) — this
is HHAR's whole point (hardware/domain heterogeneity). We compute the
per-subject-per-device implied rate from median timestamp deltas rather
than assuming a fixed Hz; ChannelMeta.native_hz is therefore an estimate,
logged per subject so it's auditable.

These files are large (single CSV covers all users/devices) — filtering
per subject_id reads and discards a lot; for repeated use, consider having
02_build_manifests.py pre-split these into per-subject parquet files.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd

from src.data.datasets.base import BaseDatasetAdapter
from src.data.schema import ChannelMeta, DatasetMeta, Modality, BodyLocation, DeviceClass

def _drop_trailing_outliers(t, arr, max_session_sec=30 * 86400.0):
    """HHAR's Creation_Time occasionally contains a corrupted value (seen:
    a phone-sensor file whose sorted max timestamp implied a 45-YEAR span).
    Gap SIZE is not a safe signal here — HHAR users recorded each activity
    as a separate short burst across a day, so real inter-session gaps of
    several hours are normal and must NOT be truncated. Instead cap the
    absolute elapsed time from session start: no real HHAR session spans
    more than ~30 days even with gaps, so anything beyond that is almost
    certainly corrupted, not real."""
    bad = np.where(t > max_session_sec)[0]
    if len(bad) > 0:
        cutoff = bad[0]
        print(f"      [hhar outlier] dropping {len(t) - cutoff} trailing rows "
              f"(t jumped to {t[cutoff]:.0f}s = {t[cutoff]/86400:.1f} days)", flush=True)
        return t[:cutoff], arr[:cutoff]
    return t, arr


_GT_MAP = {"bike": "cycling", "sit": "sitting", "stand": "standing", "walk": "walking",
           "stairsup": "ascending_stairs", "stairsdown": "descending_stairs"}

_FILES = {
    ("Phones", "accelerometer"): (Modality.ACC, BodyLocation.PHONE_POCKET, DeviceClass.SMARTPHONE),
    ("Phones", "gyroscope"): (Modality.GYRO, BodyLocation.PHONE_POCKET, DeviceClass.SMARTPHONE),
    ("Watch", "accelerometer"): (Modality.ACC, BodyLocation.WRIST, DeviceClass.SMARTWATCH),
    ("Watch", "gyroscope"): (Modality.GYRO, BodyLocation.WRIST, DeviceClass.SMARTWATCH),
}


class HHARAdapter(BaseDatasetAdapter):
    name = "hhar"

    def meta(self) -> DatasetMeta:
        channels = [ChannelMeta(mod, loc, dev, hz=100.0, n_axes=3) for mod, loc, dev in _FILES.values()]
        return DatasetMeta(
            name=self.name, n_subjects=9, channels=channels,
            has_activity_labels=True, has_hr_labels=False, hr_ground_truth_source=None,
            notes="Irregular sampling by design (hardware heterogeneity); native_hz is an estimate.",
        )

    def _data_dir(self) -> Path:
        candidate = self.raw_root / "Activity recognition exp"
        return candidate if candidate.exists() else self.raw_root

    def _file_path(self, group: str, sensor: str) -> Path:
        return self._data_dir() / f"{group}_{sensor}.csv"

    def iter_subjects(self) -> Iterator[str]:
        split_dir = self.raw_root / "by_user"
        if split_dir.exists():
            for d in sorted(split_dir.iterdir()):
                if d.is_dir():
                    yield d.name
            return
        f = self._file_path("Phones", "accelerometer")
        if not f.exists():
            return
        users = pd.read_csv(f, usecols=["User"])["User"].unique()
        for u in sorted(users):
            yield str(u)

    def load_subject_raw(self, subject_id: str) -> dict:
        channels = {}
        label_candidates = []
        split_dir = self.raw_root / "by_user" / subject_id
        for (group, sensor), (mod, loc, dev) in _FILES.items():
            split_f = split_dir / f"{group}_{sensor}.csv"
            if split_f.exists():
                df = pd.read_csv(split_f)  # pre-split — fast, no scan needed
            else:
                f = self._file_path(group, sensor)
                if not f.exists():
                    continue
                chunks = []
                for chunk in pd.read_csv(f, chunksize=500_000):
                    match = chunk[chunk["User"].astype(str) == subject_id]
                    if not match.empty:
                        chunks.append(match)
                if not chunks:
                    continue
                df = pd.concat(chunks, ignore_index=True)
            df = df.sort_values("Creation_Time")
            t = df["Creation_Time"].to_numpy(dtype=np.float64) / 1e9  # ns -> s
            t = t - t[0]
            arr = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
            n_before = len(t)
            t, arr = _drop_trailing_outliers(t, arr)
            implied_hz = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 100.0
            key = f"{loc.value}_{mod.value}_{group.lower()}"
            channels[key] = (
                self.validate_channel_array(arr, 3, key),
                ChannelMeta(mod, loc, dev, float(implied_hz), n_axes=3),
                t,
            )
            # gt must be sliced with the SAME cutoff as t/arr — df is in the
            # same sorted row order they came from, so a prefix slice keeps
            # everything aligned (this was the bug: gt stayed full-length
            # while t got truncated, causing an out-of-bounds index later).
            gt = df["gt"].iloc[:len(t)].map(lambda g: _GT_MAP.get(g, None))
            label_candidates.append((t, gt.to_numpy()))

        activity_labels = None
        if label_candidates:
            ts_ref, lbl_ref = max(label_candidates, key=lambda tl: len(tl[0]))
            activity_labels = _labels_to_segments(lbl_ref, ts_ref)

        if not channels:
            raise FileNotFoundError(f"[{self.name}] no rows found for subject {subject_id}")

        return {"channels": channels, "activity_labels": activity_labels, "hr_stream": None, "session_id": "0"}


def _labels_to_segments(labels: np.ndarray, ts: np.ndarray) -> list[tuple[float, float, str]]:
    segments = []
    valid = [i for i, l in enumerate(labels) if l is not None and l == l]  # drop None/NaN
    if not valid:
        return segments
    start = valid[0]
    prev = valid[0]
    for i in valid[1:]:
        if labels[i] != labels[start] or i != prev + 1:
            segments.append((float(ts[start]), float(ts[prev]), str(labels[start])))
            start = i
        prev = i
    segments.append((float(ts[start]), float(ts[prev]), str(labels[start])))
    return segments
