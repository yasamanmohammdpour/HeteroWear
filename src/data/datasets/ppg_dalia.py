"""
PPG-DaLiA adapter.

Expected layout: {raw_root}/S{n}/S{n}.pkl (pickled dict, latin1 encoding).
    data["signal"]["wrist"]  -> {"ACC": [T,3]@32Hz, "BVP": [T,1]@64Hz, "EDA": [T,1]@4Hz, "TEMP": [T,1]@4Hz}
    data["signal"]["chest"]  -> {"ACC": [T,3]@700Hz, "ECG": [T,1]@700Hz, "EMG": [T,1]@700Hz,
                                  "EDA": [T,1]@700Hz, "Temp": [T,1]@700Hz, "Resp": [T,1]@700Hz}
    data["label"]            -> HR reference [N] @ 0.5Hz (one value per 2s, from ECG R-peaks)
    data["activity"]         -> activity id per chest-ACC-rate sample (verify against your download;
                                 some distributions synchronize this to 4Hz instead of 700Hz)

VERIFY against your actual .pkl before trusting sampling rates below —
distributions of this dataset have varied slightly across mirrors.
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Iterator
import numpy as np

from src.data.datasets.base import BaseDatasetAdapter
from src.data.schema import ChannelMeta, DatasetMeta, Modality, BodyLocation, DeviceClass

# canonical PPG-DaLiA activity ids -> ontology label (verify against dataset README)
_ACTIVITY_MAP = {
    0: "transient", 1: "sitting", 2: "stairs", 3: "table_soccer", 4: "cycling",
    5: "driving", 6: "lunch_break", 7: "walking", 8: "working",
}

_WRIST_RATES = {"ACC": 32.0, "BVP": 64.0, "EDA": 4.0, "TEMP": 4.0}
_CHEST_RATES = {"ACC": 700.0, "ECG": 700.0, "EMG": 700.0, "EDA": 700.0, "Temp": 700.0, "Resp": 700.0}
_MODALITY_MAP = {"ACC": Modality.ACC, "BVP": Modality.PPG, "EDA": Modality.EDA,
                  "TEMP": Modality.TEMP, "Temp": Modality.TEMP,  # wrist key is "TEMP", chest key is "Temp"
                  "ECG": Modality.ECG, "EMG": Modality.EMG, "Resp": Modality.RESP}


class PPGDaLiAAdapter(BaseDatasetAdapter):
    name = "ppg_dalia"

    def meta(self) -> DatasetMeta:
        channels = [
            ChannelMeta(_MODALITY_MAP[k], BodyLocation.WRIST, DeviceClass.SMARTWATCH, hz,
                        n_axes=3 if k == "ACC" else 1)
            for k, hz in _WRIST_RATES.items()
        ] + [
            ChannelMeta(_MODALITY_MAP[k], BodyLocation.CHEST, DeviceClass.RESEARCH_IMU, hz,
                        n_axes=3 if k == "ACC" else 1)
            for k, hz in _CHEST_RATES.items()
        ]
        return DatasetMeta(
            name=self.name, n_subjects=15, channels=channels,
            has_activity_labels=True, has_hr_labels=True,
            hr_ground_truth_source="ecg",
            notes="HR label derived from chest ECG at 0.5Hz (every 2s).",
        )

    def _data_dir(self) -> Path:
        candidate = self.raw_root / "PPG_FieldStudy"
        return candidate if candidate.exists() else self.raw_root

    def iter_subjects(self) -> Iterator[str]:
        for d in sorted(self._data_dir().glob("S*")):
            if (d / f"{d.name}.pkl").exists():
                yield d.name

    def load_subject_raw(self, subject_id: str) -> dict:
        pkl_path = self._data_dir() / subject_id / f"{subject_id}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"[{self.name}] missing {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        channels = {}
        for loc_name, loc_enum, rates, device in (
            ("wrist", BodyLocation.WRIST, _WRIST_RATES, DeviceClass.SMARTWATCH),
            ("chest", BodyLocation.CHEST, _CHEST_RATES, DeviceClass.RESEARCH_IMU),
        ):
            sig = data.get("signal", {}).get(loc_name, {})
            for ch_name, hz in rates.items():
                if ch_name not in sig:
                    continue
                arr = np.asarray(sig[ch_name], dtype=np.float32)
                arr = self.validate_channel_array(arr, expected_axes=3 if ch_name == "ACC" else 1,
                                                    channel_key=f"{loc_name}_{ch_name}")
                ts = np.arange(len(arr)) / hz
                meta = ChannelMeta(_MODALITY_MAP[ch_name], loc_enum, device, hz,
                                    n_axes=arr.shape[1])
                channels[f"{loc_name}_{ch_name}"] = (arr, meta, ts)

        activity_labels = None
        if "activity" in data:
            act = np.asarray(data["activity"]).flatten()
            # activity stream assumed aligned to chest ACC rate — VERIFY for your download
            hz = _CHEST_RATES["ACC"]
            ts = np.arange(len(act)) / hz
            activity_labels = _runs_to_segments(act, ts, _ACTIVITY_MAP)

        hr_stream = None
        if "label" in data:
            hr_vals = np.asarray(data["label"], dtype=np.float32).flatten()
            hr_ts = np.arange(len(hr_vals)) * 2.0  # one value per 2s window
            hr_stream = (hr_ts, hr_vals, "ecg")

        return {
            "channels": channels,
            "activity_labels": activity_labels,
            "hr_stream": hr_stream,
            "session_id": "0",
        }


def _runs_to_segments(label_arr: np.ndarray, ts: np.ndarray, label_map: dict) -> list[tuple[float, float, str]]:
    """Collapse a per-sample label array into (start_sec, end_sec, label) runs."""
    segments = []
    if len(label_arr) == 0:
        return segments
    start_idx = 0
    for i in range(1, len(label_arr) + 1):
        if i == len(label_arr) or label_arr[i] != label_arr[start_idx]:
            lbl = label_map.get(int(label_arr[start_idx]), f"unknown_{int(label_arr[start_idx])}")
            segments.append((float(ts[start_idx]), float(ts[i - 1]), lbl))
            start_idx = i
    return segments
