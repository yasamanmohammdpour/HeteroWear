"""
WISDM (2019 dual-device) adapter.

Expected layout: {raw_root}/{device}/{sensor}/data/data_{user}_{sensor}_{device}.txt
    device in {"phone", "watch"}, sensor in {"accel", "gyro"}
    Each line: user,activity_code,timestamp_ns,x,y,z;  (trailing ';' common, comma-separated)
Nominal sampling rate ~20Hz for both devices/sensors (WISDM spec) — VERIFY,
some rows have irregular spacing; consider deriving actual rate from
timestamp deltas per subject if you see drift in the audit report.

Activity code -> label mapping below covers the subset of the 18-class
WISDM taxonomy most commonly used in HAR papers; VERIFY the full mapping
against the WISDM activity_key.txt file included in the download before
trusting labels beyond A/B/C/D/E.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd

from src.data.datasets.base import BaseDatasetAdapter
from src.data.schema import ChannelMeta, DatasetMeta, Modality, BodyLocation, DeviceClass

_NOMINAL_HZ = 20.0

_ACTIVITY_MAP = {
    "A": "walking", "B": "jogging", "C": "stairs", "D": "sitting", "E": "standing",
    "F": "typing", "G": "brushing_teeth", "H": "eating_soup", "I": "eating_chips",
    "J": "eating_pasta", "K": "drinking", "L": "eating_sandwich", "M": "kicking_ball",
    "O": "playing_catch", "P": "dribbling", "Q": "writing", "R": "clapping", "S": "folding_clothes",
}

_DEVICE_MAP = {"phone": DeviceClass.SMARTPHONE, "watch": DeviceClass.SMARTWATCH}
_LOCATION_MAP = {"phone": BodyLocation.PHONE_POCKET, "watch": BodyLocation.WRIST}


def _fix_monotonic_time(ts):
    """WISDM raw files concatenate multiple per-activity sessions without
    re-basing the device clock, so ts is not globally monotonic. Splice in
    the typical (median positive) sample spacing wherever the timestamp
    goes backward or stalls; real order/spacing preserved elsewhere."""
    import numpy as np
    d = np.diff(ts)
    positive = d[d > 0]
    typical_dt = float(np.median(positive)) if len(positive) else 0.05
    d_fixed = np.where(d <= 0, typical_dt, d)
    fixed = np.concatenate([[ts[0]], ts[0] + np.cumsum(d_fixed)])
    # also cap absolute elapsed time — a forward jump that's individually
    # "monotonic" can still be a corrupted timestamp (seen: WISDM subjects
    # with a few-hour real session but a single bad sample claiming days
    # later). No real WISDM session should exceed ~7 days.
    max_session_sec = 7 * 86400.0
    bad = np.where(fixed > max_session_sec)[0]
    if len(bad) > 0:
        cutoff = bad[0]
        print(f"    [wisdm outlier] dropping {len(fixed) - cutoff} trailing samples "
              f"(t reached {fixed[cutoff]/86400:.1f} days)", flush=True)
        fixed = fixed[:cutoff]
    return fixed


class WISDMAdapter(BaseDatasetAdapter):
    name = "wisdm"

    def meta(self) -> DatasetMeta:
        channels = []
        for device in ("phone", "watch"):
            for sensor, mod in (("accel", Modality.ACC), ("gyro", Modality.GYRO)):
                channels.append(ChannelMeta(mod, _LOCATION_MAP[device], _DEVICE_MAP[device],
                                             _NOMINAL_HZ, n_axes=3))
        return DatasetMeta(
            name=self.name, n_subjects=51, channels=channels,
            has_activity_labels=True, has_hr_labels=False, hr_ground_truth_source=None,
            notes="Dual-device (phone+watch) HAR, 18-class taxonomy (subset mapped here).",
        )

    def _data_root(self) -> Path:
        for candidate in (self.raw_root / "wisdm-dataset" / "raw", self.raw_root / "raw"):
            if candidate.exists():
                return candidate
        return self.raw_root

    def _subject_ids(self) -> list[str]:
        d = self._data_root() / "phone" / "accel" / "data"
        if not d.exists():
            d = self._data_root() / "phone" / "accel"
        ids = set()
        for f in d.glob("data_*_accel_phone.txt"):
            # filename: data_1600_accel_phone.txt
            parts = f.stem.split("_")
            if len(parts) >= 2:
                ids.add(parts[1])
        return sorted(ids)

    def iter_subjects(self) -> Iterator[str]:
        yield from self._subject_ids()

    def load_subject_raw(self, subject_id: str) -> dict:
        channels = {}
        all_labels = []
        for device in ("phone", "watch"):
            for sensor, mod in (("accel", Modality.ACC), ("gyro", Modality.GYRO)):
                d = self._data_root() / device / sensor / "data"
                if not d.exists():
                    d = self._data_root() / device / sensor
                f = d / f"data_{subject_id}_{sensor}_{device}.txt"
                if not f.exists():
                    continue  # not every subject has every device/sensor combo — real, not an error
                df = pd.read_csv(f, header=None, names=["user", "activity", "ts_ns", "x", "y", "z"],
                                  sep=",", engine="python", on_bad_lines="skip")
                df["z"] = df["z"].astype(str).str.replace(";", "", regex=False).astype(np.float32)
                ts = df["ts_ns"].to_numpy(dtype=np.float64) / 1e9
                ts = ts - ts[0]
                arr = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
                ts = _fix_monotonic_time(ts)
                arr = arr[:len(ts)]  # keep aligned after outlier truncation
                key = f"{_LOCATION_MAP[device].value}_{mod.value}_{device}"
                channels[key] = (
                    self.validate_channel_array(arr, 3, key),
                    ChannelMeta(mod, _LOCATION_MAP[device], _DEVICE_MAP[device], _NOMINAL_HZ, n_axes=3),
                    ts,
                )
                labels = df["activity"].map(lambda c: _ACTIVITY_MAP.get(c, f"unknown_{c}"))
                labels_arr = labels.to_numpy()[:len(ts)]  # keep aligned after outlier truncation
                all_labels.append((ts, labels_arr))

        activity_labels = None
        if all_labels:
            ts_ref, lbl_ref = max(all_labels, key=lambda tl: len(tl[0]))
            activity_labels = _labels_to_segments(lbl_ref, ts_ref)

        if not channels:
            raise FileNotFoundError(f"[{self.name}] no channel files found for subject {subject_id}")

        return {
            "channels": channels,
            "activity_labels": activity_labels,
            "hr_stream": None,
            "session_id": "0",
        }


def _labels_to_segments(labels: np.ndarray, ts: np.ndarray) -> list[tuple[float, float, str]]:
    segments = []
    if len(labels) == 0:
        return segments
    start_idx = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start_idx]:
            segments.append((float(ts[start_idx]), float(ts[i - 1]), str(labels[start_idx])))
            start_idx = i
    return segments
