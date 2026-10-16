"""Vendor adapters for NMR Workbench."""

from .bruker import Bruker1DReader, BrukerReadError, map_bruker_metadata
from .loader import (
    VendorDetectionError,
    detect_vendor,
    load_nmr_dataset,
    resolve_nmr_dataset_path,
)
from .varian import Varian1DReader, VarianReadError, map_varian_metadata

__all__ = [
    "Bruker1DReader",
    "BrukerReadError",
    "Varian1DReader",
    "VarianReadError",
    "VendorDetectionError",
    "detect_vendor",
    "load_nmr_dataset",
    "resolve_nmr_dataset_path",
    "map_bruker_metadata",
    "map_varian_metadata",
]

