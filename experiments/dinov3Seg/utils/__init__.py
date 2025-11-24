"""
Utility helpers for data preparation, optimization, and training glue.
"""

from .data import (
    PrecomputedDataset,
    extract_multiscale_features,
    prepare_data_tiles,
    subset_label_to_image_bounds,
    verify_and_clean_dataset_fast,
)
from .optim import EarlyStopping, Muon, zeropower_via_newtonschulz5

__all__ = [
    "PrecomputedDataset",
    "extract_multiscale_features",
    "prepare_data_tiles",
    "subset_label_to_image_bounds",
    "verify_and_clean_dataset_fast",
    "EarlyStopping",
    "Muon",
    "zeropower_via_newtonschulz5",
]
