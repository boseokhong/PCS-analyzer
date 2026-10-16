"""Deterministic, GUI-independent processing for one-dimensional NMR data."""

from .baseline import BaselineCorrectionService, BaselineError
from .comparison import (
    PeakPositionComparison,
    SpectrumAxisSummary,
    compare_peak_positions,
    load_xy_spectrum_csv,
    summarize_xy_spectrum,
)
from .digital_filter import BrukerDigitalFilter, DigitalFilterError
from .processor import ProcessingError, SpectrumProcessor, ppm_to_fractional_index
from .phasing import AutoPhaseError, AutoPhaseEstimate, AutoPhaseService
from .recipe import (
    PROCESSING_RECIPE_SCHEMA_VERSION,
    ProcessingRecipeError,
    deserialize_processing_history,
    processing_history_from_json,
    processing_history_to_json,
    serialize_processing_history,
)
from .result import ProcessingResult

__all__ = [
    "BaselineCorrectionService",
    "BaselineError",
    "AutoPhaseError",
    "AutoPhaseEstimate",
    "AutoPhaseService",
    "BrukerDigitalFilter",
    "DigitalFilterError",
    "PeakPositionComparison",
    "ProcessingError",
    "ProcessingRecipeError",
    "ProcessingResult",
    "PROCESSING_RECIPE_SCHEMA_VERSION",
    "SpectrumProcessor",
    "SpectrumAxisSummary",
    "compare_peak_positions",
    "deserialize_processing_history",
    "ppm_to_fractional_index",
    "processing_history_from_json",
    "processing_history_to_json",
    "serialize_processing_history",
    "load_xy_spectrum_csv",
    "summarize_xy_spectrum",
]

