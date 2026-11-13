"""
WESAD adapter.

Expected layout: {raw_root}/S{n}/S{n}.pkl (pickled dict, latin1).
    data["signal"]["wrist"] -> {"ACC":[T,3]@32Hz, "BVP":[T,1]@64Hz, "EDA":[T,1]@4Hz, "TEMP":[T,1]@4Hz}
    data["signal"]["chest"] -> {"ACC":[T,3]@700Hz, "ECG":[T,1]@700Hz, "EDA":[T,1]@700Hz,
                                 "EMG":[T,1]@700Hz, "Resp":[T,1]@700Hz, "Temp":[T,1]@700Hz}
    data["label"]           -> per-chest-sample (700Hz) integer affect-state code

IMPORTANT — do not treat WESAD's label as a physical "activity" the way
PAMAP2/WISDM/etc. are. It encodes affect/experimental condition
(baseline/stress/amusement/meditation), a different semantic axis. It is
stored in `extra["affect_state"]`, and `activity_label` is left None so
downstream cross-dataset activity-ontology code can't accidentally merge
the two label spaces.

WESAD does not ship a precomputed HR ground-truth stream like PPG-DaLiA's
`label` array — HR must be derived from chest ECG R-peaks in preprocessing
(scripts/preprocess/03_preprocess.py should call an R-peak detector, e.g.
neurokit2 or biosppy, on the "chest_ECG" channel; not implemented in this
adapter to keep dataset I/O and signal-processing concerns separate).
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Iterator
import numpy as np

from src.data.datasets.base import BaseDatasetAdapter
from src.data.schema import ChannelMeta, DatasetMeta, Modality, BodyLocation, DeviceClass

_AFFECT_MAP = {0: "transient", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation",
               5: "ignore", 6: "ignore", 7: "ignore"}

_WRIST_RATES = {"ACC": 32.0, "BVP": 64.0, "EDA": 4.0, "TEMP": 4.0}
_CHEST_RATES = {"ACC": 700.0, "ECG": 700.0, "EDA": 700.0, "EMG": 700.0, "Resp": 700.0, "Temp": 700.0}
_MODALITY_MAP = {"ACC": Modality.ACC, "BVP": Modality.PPG, "EDA": Modality.EDA,
                  "TEMP": Modality.TEMP, "Temp": Modality.TEMP, "ECG": Modality.ECG,
                  "EMG": Modality.EMG, "Resp": Modality.RESP}


class WESADAdapter(BaseDatasetAdapter):
    name = "wesad"

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
            has_activity_labels=False, has_hr_labels=True,
            hr_ground_truth_source="ecg_derived",
            notes="Affect-state labels only, no native activity taxonomy. "
                  "HR must be derived from chest ECG in preprocessing.",
        )

    def _data_dir(self) -> Path:
        for candidate in (self.raw_root / "extracted" / "WESAD",
                           self.raw_root / "WESAD", self.raw_root):
            if candidate.exists() and any(candidate.glob("S*")):
                return candidate
        return self.raw_root

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
                meta = ChannelMeta(_MODALITY_MAP[ch_name], loc_enum, device, hz, n_axes=arr.shape[1])
                channels[f"{loc_name}_{ch_name}"] = (arr, meta, ts)

        affect_segments = None
        if "label" in data:
            lab = np.asarray(data["label"]).flatten()
            hz = _CHEST_RATES["ECG"]
            ts = np.arange(len(lab)) / hz
            affect_segments = _runs_to_segments(lab, ts, _AFFECT_MAP)

        return {
            "channels": channels,
            "activity_labels": None,          # deliberately not populated — see module docstring
            "affect_segments": affect_segments,  # consumed separately by preprocessing into extra[]
            "hr_stream": None,                 # derive from chest_ECG in preprocessing
            "session_id": "0",
        }


def _runs_to_segments(label_arr: np.ndarray, ts: np.ndarray, label_map: dict) -> list[tuple[float, float, str]]:
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
