"""
Base interface for dataset adapters.

Design rule: adapters know how to READ their dataset's native format and
emit WindowSample objects. They do NOT resample, normalize, or window
across a common grid — that happens later in windowing.py so the logic
is shared and testable once instead of reimplemented per dataset.

Every adapter must implement:
    - meta() -> DatasetMeta
    - iter_subjects() -> iterator of subject IDs found on disk
    - load_subject_raw(subject_id) -> dict of {channel_key: (array, ChannelMeta)}
      plus optional activity/HR label streams (see load_subject_raw docstring)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator
import numpy as np

from src.data.schema import ChannelMeta, DatasetMeta


class BaseDatasetAdapter(ABC):
    name: str = "base"

    def __init__(self, raw_root: Path, config: dict):
        self.raw_root = Path(raw_root)
        self.config = config

    @abstractmethod
    def meta(self) -> DatasetMeta:
        ...

    @abstractmethod
    def iter_subjects(self) -> Iterator[str]:
        ...

    @abstractmethod
    def load_subject_raw(self, subject_id: str) -> dict:
        """
        Returns a dict with:
            "channels": {channel_key: (np.ndarray [T, n_axes], ChannelMeta, timestamps_sec [T])}
            "activity_labels": Optional[list of (start_sec, end_sec, canonical_label)]
            "hr_stream": Optional[(timestamps_sec [T], hr_bpm [T], source_str)]
            "session_id": str  (use "0" if the dataset has no session concept)
        Subclasses should be defensive: if a file is missing/malformed for
        one subject, raise a clear exception rather than silently skipping —
        silent skips are exactly the kind of methodological mistake this
        pipeline is designed to avoid.
        """
        ...

    def validate_channel_array(self, arr: np.ndarray, expected_axes: int, channel_key: str):
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] != expected_axes:
            raise ValueError(
                f"[{self.name}] channel '{channel_key}' expected {expected_axes} axes, "
                f"got shape {arr.shape}"
            )
        return arr
