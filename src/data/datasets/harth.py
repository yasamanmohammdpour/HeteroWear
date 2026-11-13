"""
HARTH adapter.

Expected layout: {raw_root}/S*.csv, columns:
    timestamp, back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label
Sampling rate 50Hz. Label codes per HARTH documentation:
    1 walking, 2 running, 3 shuffling, 4 stairs_ascending, 5 stairs_descending,
    6 standing, 7 sitting, 8 lying, 13 cycling_sit, 14 cycling_stand,
    130 cycling_sit_inactive, 140 cycling_stand_inactive
VERIFY these codes against the HARTH README before trusting the full set —
they are transcribed from memory of the dataset paper, not re-derived here.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd

from src.data.datasets.base import BaseDatasetAdapter
from src.data.schema import ChannelMeta, DatasetMeta, Modality, BodyLocation, DeviceClass

_HZ = 50.0
_LABEL_MAP = {
    1: "walking", 2: "running", 3: "shuffling", 4: "ascending_stairs", 5: "descending_stairs",
    6: "standing", 7: "sitting", 8: "lying", 13: "cycling_sit", 14: "cycling_stand",
    130: "cycling_sit_inactive", 140: "cycling_stand_inactive",
}


class HARTHAdapter(BaseDatasetAdapter):
    name = "harth"

    def meta(self) -> DatasetMeta:
        channels = [
            ChannelMeta(Modality.ACC, BodyLocation.BACK, DeviceClass.RESEARCH_IMU, _HZ, n_axes=3),
            ChannelMeta(Modality.ACC, BodyLocation.THIGH, DeviceClass.RESEARCH_IMU, _HZ, n_axes=3),
        ]
        return DatasetMeta(
            name=self.name, n_subjects=22, channels=channels,
            has_activity_labels=True, has_hr_labels=False, hr_ground_truth_source=None,
            notes="Free-living recordings — your one free-living HAR dataset; small scale (~22 subjects) "
                  "relative to lab-protocol sets.",
        )

    def _data_dir(self) -> Path:
        nested = self.raw_root / "harth"
        return nested if nested.exists() else self.raw_root

    def iter_subjects(self) -> Iterator[str]:
        for f in sorted(self._data_dir().glob("S*.csv")):
            yield f.stem

    def load_subject_raw(self, subject_id: str) -> dict:
        f = self._data_dir() / f"{subject_id}.csv"
        if not f.exists():
            raise FileNotFoundError(f"[{self.name}] missing {f}")
        df = pd.read_csv(f)
        required = {"back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"[{self.name}] {f} missing expected columns: {missing}; "
                              f"actual columns: {list(df.columns)}")

        n = len(df)
        ts = np.arange(n) / _HZ
        back = df[["back_x", "back_y", "back_z"]].to_numpy(dtype=np.float32)
        thigh = df[["thigh_x", "thigh_y", "thigh_z"]].to_numpy(dtype=np.float32)
        channels = {
            "back_ACC": (self.validate_channel_array(back, 3, "back_ACC"),
                         ChannelMeta(Modality.ACC, BodyLocation.BACK, DeviceClass.RESEARCH_IMU, _HZ, n_axes=3), ts),
            "thigh_ACC": (self.validate_channel_array(thigh, 3, "thigh_ACC"),
                          ChannelMeta(Modality.ACC, BodyLocation.THIGH, DeviceClass.RESEARCH_IMU, _HZ, n_axes=3), ts),
        }

        labels = df["label"].to_numpy(dtype=int)
        activity_labels = _ids_to_segments(labels, ts, _LABEL_MAP)

        return {"channels": channels, "activity_labels": activity_labels, "hr_stream": None, "session_id": "0"}


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
