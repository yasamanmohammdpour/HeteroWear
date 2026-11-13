"""
Central registry. New datasets are added here, nowhere else — this is
the single place that must change to add dataset #9, which keeps the
rest of the pipeline (windowing, splits, training) dataset-count-agnostic.
"""

from src.data.datasets.ppg_dalia import PPGDaLiAAdapter
from src.data.datasets.wesad import WESADAdapter
from src.data.datasets.pamap2 import PAMAP2Adapter
from src.data.datasets.mmash import MMASHAdapter
from src.data.datasets.wisdm import WISDMAdapter
from src.data.datasets.hhar import HHARAdapter
from src.data.datasets.harth import HARTHAdapter
from src.data.datasets.uci_har import UCIHARAdapter

DATASET_REGISTRY = {
    "ppg_dalia": PPGDaLiAAdapter,
    "wesad": WESADAdapter,
    "pamap2": PAMAP2Adapter,
    "mmash": MMASHAdapter,
    "wisdm": WISDMAdapter,
    "hhar": HHARAdapter,
    "harth": HARTHAdapter,
    "uci_har": UCIHARAdapter,
}


def get_adapter(name: str, raw_root, config: dict):
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Registered: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](raw_root=raw_root, config=config)
