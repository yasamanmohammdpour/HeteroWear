"""
PAMAP2 adapter.

Expected layout: {raw_root}/Protocol/subject{101..109}.dat (whitespace-separated,
54 columns, no header). Column layout per PAMAP2 documentation:
    0:  timestamp (s)
    1:  activityID
    2:  heart rate (bpm) — sparse, ~9Hz effective, NaN elsewhere
    3-19:  hand IMU  (temp, acc16g x3, acc6g x3, gyro x3, mag x3, orientation x4[invalid])
    20-36: chest IMU (same 17-column layout)
    37-53: ankle IMU (same 17-column layout)

We use the acc16g range (avoids clipping during vigorous activity — the
canonical choice in PAMAP2 papers) and drop the orientation columns, which
PAMAP2's own documentation marks as invalid in this data collection.
IMU sampling rate is 100Hz.

VERIFY column offsets against your actual .dat files before trusting this —
some PAMAP2 mirrors ship the "Optional" folder with a different subject
numbering; this adapter only reads "Protocol" by default.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd

from src.data.datasets.base import BaseDatasetAdapter
from src.data.schema import ChannelMeta, DatasetMeta, Modality, BodyLocation, DeviceClass

_ACTIVITY_MAP = {
    0: "transient", 1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
    6: "cycling", 7: "nordic_walking", 9: "watching_tv", 10: "computer_work", 11: "car_driving",
    12: "ascending_stairs", 13: "descending_stairs", 16: "vacuum_cleaning", 17: "ironing",
    18: "folding_laundry", 19: "house_cleaning", 20: "playing_soccer", 24: "rope_jumping",
}

_IMU_HZ = 100.0

# (location_name, BodyLocation, start_col_of_17block)
def _drop_nan_rows(arr, ts):
    """PAMAP2's raw IMU columns (not just HR) use literal 'NaN' to mark
    dropped wireless packets. Undetected, these silently poison every
    downstream mean/std/feature computation (confirmed: an entire
    baseline run produced NaN features and near-zero macro-F1 before
    this fix). Drop rows where this channel's own values are NaN."""
    valid = ~np.isnan(arr).any(axis=1)
    return arr[valid], ts[valid]


_IMU_BLOCKS = [
    ("hand", BodyLocation.WRIST, 3),
    ("chest", BodyLocation.CHEST, 20),
    ("ankle", BodyLocation.ANKLE, 37),
]


class PAMAP2Adapter(BaseDatasetAdapter):
    name = "pamap2"

    def meta(self) -> DatasetMeta:
        channels = []
        for loc_name, loc_enum, _ in _IMU_BLOCKS:
            channels.append(ChannelMeta(Modality.ACC, loc_enum, DeviceClass.RESEARCH_IMU, _IMU_HZ, n_axes=3))
            channels.append(ChannelMeta(Modality.GYRO, loc_enum, DeviceClass.RESEARCH_IMU, _IMU_HZ, n_axes=3))
            channels.append(ChannelMeta(Modality.MAG, loc_enum, DeviceClass.RESEARCH_IMU, _IMU_HZ, n_axes=3))
        return DatasetMeta(
            name=self.name, n_subjects=9, channels=channels,
            has_activity_labels=True, has_hr_labels=True,
            hr_ground_truth_source="chest_strap",
            notes="HR from chest strap monitor, sparse relative to 100Hz IMU stream.",
        )

    def _protocol_dir(self) -> Path:
        for candidate in (self.raw_root / "PAMAP2_Dataset" / "Protocol",
                           self.raw_root / "Protocol"):
            if candidate.exists():
                return candidate
        return self.raw_root

    def iter_subjects(self) -> Iterator[str]:
        for f in sorted(self._protocol_dir().glob("subject*.dat")):
            yield f.stem  # e.g. "subject101"

    def load_subject_raw(self, subject_id: str) -> dict:
        f = self._protocol_dir() / f"{subject_id}.dat"
        if not f.exists():
            raise FileNotFoundError(f"[{self.name}] missing {f}")

        df = pd.read_csv(f, sep=r"\s+", header=None, na_values="NaN")
        ts = df[0].to_numpy(dtype=np.float64)
        activity_ids = df[1].to_numpy(dtype=int)
        hr = df[2].to_numpy(dtype=np.float32)

        channels = {}
        for loc_name, loc_enum, start_col in _IMU_BLOCKS:
            acc16 = df.iloc[:, start_col + 1: start_col + 4].to_numpy(dtype=np.float32)
            gyro = df.iloc[:, start_col + 7: start_col + 10].to_numpy(dtype=np.float32)
            mag = df.iloc[:, start_col + 10: start_col + 13].to_numpy(dtype=np.float32)
            acc16, acc_ts = _drop_nan_rows(acc16, ts)
            gyro, gyro_ts = _drop_nan_rows(gyro, ts)
            mag, mag_ts = _drop_nan_rows(mag, ts)
            channels[f"{loc_name}_ACC"] = (
                self.validate_channel_array(acc16, 3, f"{loc_name}_ACC"),
                ChannelMeta(Modality.ACC, loc_enum, DeviceClass.RESEARCH_IMU, _IMU_HZ, n_axes=3), acc_ts)
            channels[f"{loc_name}_GYRO"] = (
                self.validate_channel_array(gyro, 3, f"{loc_name}_GYRO"),
                ChannelMeta(Modality.GYRO, loc_enum, DeviceClass.RESEARCH_IMU, _IMU_HZ, n_axes=3), gyro_ts)
            channels[f"{loc_name}_MAG"] = (
                self.validate_channel_array(mag, 3, f"{loc_name}_MAG"),
                ChannelMeta(Modality.MAG, loc_enum, DeviceClass.RESEARCH_IMU, _IMU_HZ, n_axes=3), mag_ts)

        activity_labels = _ids_to_segments(activity_ids, ts, _ACTIVITY_MAP)

        valid_hr = ~np.isnan(hr)
        hr_stream = (ts[valid_hr], hr[valid_hr], "chest_strap") if valid_hr.any() else None

        return {
            "channels": channels,
            "activity_labels": activity_labels,
            "hr_stream": hr_stream,
            "session_id": "0",
        }


def _ids_to_segments(ids: np.ndarray, ts: np.ndarray, id_map: dict) -> list[tuple[float, float, str]]:
    segments = []
    if len(ids) == 0:
        return segments
    start_idx = 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start_idx]:
            lbl = id_map.get(int(ids[start_idx]), f"unknown_{int(ids[start_idx])}")
            segments.append((float(ts[start_idx]), float(ts[i - 1]), lbl))
            start_idx = i
    return segments
