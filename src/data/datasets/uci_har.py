"""
UCI-HAR Raw adapter.

Expected layout: {raw_root}/RawData/
    acc_exp{EE}_user{UU}.txt, gyro_exp{EE}_user{UU}.txt  (space-separated x,y,z, 50Hz)
    labels.txt: exp_id, user_id, activity_id, start_sample, end_sample
Activity ids (base 6-class taxonomy; the "Raw" distribution's labels.txt
sometimes includes postural-transition ids 7-12 — mapped to a generic
"transition" bucket here since transitions aren't comparable to the other
datasets' steady-state activity labels):
    1 walking, 2 ascending_stairs, 3 descending_stairs, 4 sitting, 5 standing, 6 lying
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
    1: "walking", 2: "ascending_stairs", 3: "descending_stairs",
    4: "sitting", 5: "standing", 6: "lying",
}


class UCIHARAdapter(BaseDatasetAdapter):
    name = "uci_har"

    def meta(self) -> DatasetMeta:
        channels = [
            ChannelMeta(Modality.ACC, BodyLocation.PHONE_POCKET, DeviceClass.SMARTPHONE, _HZ, n_axes=3),
            ChannelMeta(Modality.GYRO, BodyLocation.PHONE_POCKET, DeviceClass.SMARTPHONE, _HZ, n_axes=3),
        ]
        return DatasetMeta(
            name=self.name, n_subjects=30, channels=channels,
            has_activity_labels=True, has_hr_labels=False, hr_ground_truth_source=None,
            notes="Standard HAR benchmark; single body location (phone), 6-class taxonomy.",
        )

    def _raw_dir(self) -> Path:
        d = self.raw_root / "RawData"
        return d if d.exists() else self.raw_root

    def _labels_df(self) -> pd.DataFrame:
        lf = self._raw_dir() / "labels.txt"
        return pd.read_csv(lf, sep=r"\s+", header=None,
                            names=["exp", "user", "activity", "start", "end"])

    def iter_subjects(self) -> Iterator[str]:
        labels = self._labels_df()
        for u in sorted(labels["user"].unique()):
            yield f"user{int(u):02d}"

    def load_subject_raw(self, subject_id: str) -> dict:
        user_num = int(subject_id.replace("user", ""))
        raw_dir = self._raw_dir()
        labels = self._labels_df()
        user_labels = labels[labels["user"] == user_num]
        if user_labels.empty:
            raise FileNotFoundError(f"[{self.name}] no label rows for {subject_id}")

        channels = {}
        activity_segments = []
        for _, row in user_labels.groupby("exp").first().iterrows():
            pass  # placeholder to keep structure explicit; real loop below

        for exp_id in sorted(user_labels["exp"].unique()):
            acc_f = raw_dir / f"acc_exp{int(exp_id):02d}_user{user_num:02d}.txt"
            gyro_f = raw_dir / f"gyro_exp{int(exp_id):02d}_user{user_num:02d}.txt"
            if not acc_f.exists() or not gyro_f.exists():
                continue
            acc = pd.read_csv(acc_f, sep=r"\s+", header=None).to_numpy(dtype=np.float32)
            gyro = pd.read_csv(gyro_f, sep=r"\s+", header=None).to_numpy(dtype=np.float32)
            n = min(len(acc), len(gyro))
            ts = np.arange(n) / _HZ
            channels[f"phone_ACC_exp{exp_id}"] = (
                self.validate_channel_array(acc[:n], 3, "phone_ACC"),
                ChannelMeta(Modality.ACC, BodyLocation.PHONE_POCKET, DeviceClass.SMARTPHONE, _HZ, n_axes=3), ts)
            channels[f"phone_GYRO_exp{exp_id}"] = (
                self.validate_channel_array(gyro[:n], 3, "phone_GYRO"),
                ChannelMeta(Modality.GYRO, BodyLocation.PHONE_POCKET, DeviceClass.SMARTPHONE, _HZ, n_axes=3), ts)

            exp_rows = user_labels[user_labels["exp"] == exp_id]
            for _, r in exp_rows.iterrows():
                lbl = _LABEL_MAP.get(int(r["activity"]), "transition")
                activity_segments.append((float(r["start"]) / _HZ, float(r["end"]) / _HZ, lbl))

        if not channels:
            raise FileNotFoundError(f"[{self.name}] no acc/gyro files found for {subject_id}")

        return {
            "channels": channels,
            "activity_labels": activity_segments,
            "hr_stream": None,
            "session_id": "0",
        }
