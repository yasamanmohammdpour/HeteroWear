"""
Canonical schema for HeteroWear.

Every dataset adapter must emit data in this shape. This is the contract
that makes the rest of the pipeline (windowing, tokenization, model)
dataset-agnostic. If a new dataset doesn't fit this schema, the fix goes
in the adapter, never by loosening this contract.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class Modality(str, Enum):
    ACC = "ACC"
    GYRO = "GYRO"
    MAG = "MAG"
    PPG = "PPG"          # BVP / PPG, interchangeable naming across datasets
    ECG = "ECG"
    EDA = "EDA"
    TEMP = "TEMP"
    RESP = "RESP"
    EMG = "EMG"
    HR = "HR"             # pre-computed HR/RR scalar stream (not raw waveform)


class BodyLocation(str, Enum):
    WRIST = "wrist"
    CHEST = "chest"
    ANKLE = "ankle"
    THIGH = "thigh"
    BACK = "back"
    HIP = "hip"
    PHONE_POCKET = "phone_pocket"
    UNKNOWN = "unknown"


class DeviceClass(str, Enum):
    RESEARCH_IMU = "research_imu"      # e.g. Shimmer, RespiBAN, Axivity
    SMARTWATCH = "smartwatch"          # Empatica E4, Galaxy Watch, etc.
    SMARTPHONE = "smartphone"
    CHEST_STRAP = "chest_strap"        # Polar H7/H10 style
    ACTIGRAPH = "actigraph"
    UNKNOWN = "unknown"


@dataclass
class ChannelMeta:
    """Describes one physical channel (e.g. wrist ACC x-axis group)."""
    modality: Modality
    location: BodyLocation
    device: DeviceClass
    native_hz: float
    n_axes: int = 1                    # 3 for tri-axial ACC/GYRO, 1 for PPG/EDA/etc.
    is_derived: bool = False           # True for e.g. MMASH's 1Hz actigraphy counts,
                                        # HR computed by the device rather than raw waveform


@dataclass
class WindowSample:
    """
    One training/eval example: a fixed-duration interval with an arbitrary
    subset of channels present. Missing channels are simply absent from
    `signals`, never zero-filled (zero-filling would be indistinguishable
    from a real all-zero reading and corrupts the missing-modality story).
    """
    dataset: str
    subject_id: str
    session_id: str
    window_start_sec: float
    window_sec: float
    signals: dict[str, np.ndarray] = field(default_factory=dict)   # channel_key -> [T, n_axes]
    channel_meta: dict[str, ChannelMeta] = field(default_factory=dict)  # channel_key -> meta
    activity_label: Optional[str] = None     # canonical ontology label, or None if unlabeled/not applicable
    hr_bpm: Optional[float] = None           # reference HR for this window, if available
    hr_source: Optional[str] = None          # "ecg", "chest_strap", "ppg_derived" — provenance matters
    extra: dict = field(default_factory=dict)  # dataset-specific extras (e.g. WESAD affect state)

    def channel_key(self, modality: Modality, location: BodyLocation) -> str:
        return f"{location.value}_{modality.value}"


@dataclass
class DatasetMeta:
    name: str
    n_subjects: int
    channels: list[ChannelMeta]
    has_activity_labels: bool
    has_hr_labels: bool
    hr_ground_truth_source: Optional[str]   # "ecg", "chest_strap", None
    notes: str = ""
