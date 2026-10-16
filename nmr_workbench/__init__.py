"""Internal NMR Workbench module for PCS Analyzer."""

from .version import __version__

from .core import (
    ExperimentalConditions,
    NMRDataset,
    NMRObservable,
    NMRPeak,
    NMRSeries,
    NMRSeriesMember,
    ProcessingHistory,
    ProcessingStep,
    Spectrum1D,
)
from .io import (
    Bruker1DReader,
    Varian1DReader,
    VendorDetectionError,
    detect_vendor,
    load_nmr_dataset,
    resolve_nmr_dataset_path,
)

__all__ = [
    "__version__",
    "ExperimentalConditions",
    "NMRDataset",
    "NMRObservable",
    "NMRPeak",
    "NMRSeries",
    "NMRSeriesMember",
    "ProcessingHistory",
    "ProcessingStep",
    "Spectrum1D",
    "Bruker1DReader",
    "Varian1DReader",
    "VendorDetectionError",
    "detect_vendor",
    "load_nmr_dataset",
    "resolve_nmr_dataset_path",
]

