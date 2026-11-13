"""
MMASH adapter.

Expected layout: {raw_root}/extracted/user_{n}/ containing (per PhysioNet
distribution): RR.csv, Actigraph.csv, Activity.csv, sleep.csv, user_info.csv,
questionnaire.csv, saliva.csv. We only use RR.csv (beat-to-beat -> HR) and
Actigraph.csv (1Hz triaxial wrist accelerometer).

COLUMN NAMES ARE NOT 100% CONFIRMED FROM DOCUMENTATION ALONE — this adapter
tries a small set of known-plausible column name variants and raises a
clear error naming the columns it found vs. expected if none match. Run
this against one real subject folder first and patch `_RR_COL_CANDIDATES`
/ `_ACC_COL_CANDIDATES` to match before trusting output for all 22 subjects.

Important limitations to carry into the paper (see conversation history):
  - Actigraph.csv is 1Hz — this is activity-count-resolution, not raw IMU.
    Do not expect fine-grained motion/activity information from it.
  - MMASH is a single continuous ~24h session per subject, not multi-day.
    Calibration-budget experiments here are "within-session", not "multi-day".
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd

from src.data.datasets.base import BaseDatasetAdapter
from src.data.schema import ChannelMeta, DatasetMeta, Modality, BodyLocation, DeviceClass

_ACTIGRAPH_HZ = 1.0

_RR_COL_CANDIDATES = [("time", "ibi_s"), ("Time", "ibi_s"), ("time", "IBI")]
_ACC_COL_CANDIDATES = [("Axis1", "Axis2", "Axis3"), ("axis1", "axis2", "axis3"),
                        ("x", "y", "z")]


def _daytime_to_sec(day: np.ndarray, time_str) -> np.ndarray:
    """MMASH's 'time' column is HH:MM:SS wall-clock, 'day' handles midnight
    rollover across the ~24h session. Convert to elapsed seconds from the
    start of day 1 so RR.csv and Actigraph.csv share a consistent axis."""
    h = time_str.str.slice(0, 2).astype(int)
    m = time_str.str.slice(3, 5).astype(int)
    s = time_str.str.slice(6, 8).astype(int)
    return (day.astype(np.float64) - 1) * 86400.0 + h.to_numpy() * 3600.0 + \
        m.to_numpy() * 60.0 + s.to_numpy().astype(np.float64)


def _find_columns(df: pd.DataFrame, candidates: list[tuple]) -> tuple:
    for cand in candidates:
        if all(c in df.columns for c in cand):
            return cand
    raise ValueError(
        f"[mmash] none of the expected column sets {candidates} found; "
        f"actual columns are {list(df.columns)} — update _RR_COL_CANDIDATES / "
        f"_ACC_COL_CANDIDATES in mmash.py to match your download."
    )


class MMASHAdapter(BaseDatasetAdapter):
    name = "mmash"

    def meta(self) -> DatasetMeta:
        channels = [
            ChannelMeta(Modality.ACC, BodyLocation.WRIST, DeviceClass.ACTIGRAPH, _ACTIGRAPH_HZ,
                        n_axes=3, is_derived=True),
        ]
        return DatasetMeta(
            name=self.name, n_subjects=22, channels=channels,
            has_activity_labels=False, has_hr_labels=True,
            hr_ground_truth_source="chest_strap",  # Polar H7 R-R intervals
            notes="1Hz actigraphy only (no raw high-rate IMU); single ~24h session per subject; "
                  "all-male cohort — flag as a demographic limitation in the paper.",
        )

    def _data_dir(self) -> Path:
        for candidate in (self.raw_root / "extracted" / "DataPaper",
                           self.raw_root / "DataPaper",
                           self.raw_root / "extracted", self.raw_root):
            if candidate.exists() and any(candidate.glob("user_*")):
                return candidate
        return self.raw_root

    def iter_subjects(self) -> Iterator[str]:
        for d in sorted(self._data_dir().glob("user_*")):
            if d.is_dir():
                yield d.name

    def load_subject_raw(self, subject_id: str) -> dict:
        subj_dir = self._data_dir() / subject_id
        rr_path = subj_dir / "RR.csv"
        acc_path = subj_dir / "Actigraph.csv"
        if not rr_path.exists() or not acc_path.exists():
            raise FileNotFoundError(f"[{self.name}] missing RR.csv or Actigraph.csv under {subj_dir}")

        channels = {}
        rr_df = pd.read_csv(rr_path)
        n_before = len(rr_df)
        rr_df = rr_df[(rr_df["day"] >= 1) & (rr_df["day"] <= 5)].reset_index(drop=True)
        if len(rr_df) < n_before:
            print(f"    [mmash outlier] {subject_id}/RR.csv: dropped {n_before - len(rr_df)} rows "
                  f"with implausible day values (source data corruption, e.g. day<0)", flush=True)
        _, ibi_col = _find_columns(rr_df, _RR_COL_CANDIDATES)
        t = _daytime_to_sec(rr_df["day"].to_numpy(), rr_df["time"])
        ibi_s = rr_df[ibi_col].to_numpy(dtype=np.float32)
        hr_bpm = 60.0 / np.clip(ibi_s, 1e-3, None)
        hr_stream = (t, hr_bpm, "chest_strap")

        acc_df = pd.read_csv(acc_path)
        n_before = len(acc_df)
        acc_df = acc_df[(acc_df["day"] >= 1) & (acc_df["day"] <= 5)].reset_index(drop=True)
        if len(acc_df) < n_before:
            print(f"    [mmash outlier] {subject_id}/Actigraph.csv: dropped {n_before - len(acc_df)} rows "
                  f"with implausible day values", flush=True)
        ax_cols = _find_columns(acc_df, _ACC_COL_CANDIDATES)
        acc = acc_df[list(ax_cols)].to_numpy(dtype=np.float32)
        acc_ts = _daytime_to_sec(acc_df["day"].to_numpy(), acc_df["time"])
        channels["wrist_ACC"] = (
            self.validate_channel_array(acc, 3, "wrist_ACC"),
            ChannelMeta(Modality.ACC, BodyLocation.WRIST, DeviceClass.ACTIGRAPH, _ACTIGRAPH_HZ,
                        n_axes=3, is_derived=True),
            acc_ts,
        )

        return {
            "channels": channels,
            "activity_labels": None,   # Activity.csv is a coarse diary, not a per-sample label stream;
                                         # merge in preprocessing if you want coarse activity context.
            "hr_stream": hr_stream,
            "session_id": "0",
        }
