"""Lightweight comparisons against external x/y or peak-position exports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SpectrumAxisSummary:
    """Orientation and limits for an externally exported x/y spectrum."""

    point_count: int
    ppm_min: float
    ppm_max: float
    orientation: str
    intensity_abs_max: float


@dataclass(frozen=True)
class PeakPositionComparison:
    """Nearest unique peak-position comparison without performing peak picking."""

    reference_count: int
    candidate_count: int
    matched_count: int
    tolerance_ppm: float
    median_offset_ppm: float | None
    mean_absolute_error_ppm: float | None
    maximum_absolute_error_ppm: float | None


def load_xy_spectrum_csv(
    path: str | Path, *, delimiter: str = ","
) -> tuple[np.ndarray, np.ndarray]:
    """Load the first two numeric columns from a simple spectrum export.

    Text headers and blank rows are ignored. The source file is never modified.
    """

    x_values: list[float] = []
    y_values: list[float] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter=delimiter):
            if len(row) < 2:
                continue
            try:
                x_value = float(row[0].strip())
                y_value = float(row[1].strip())
            except ValueError:
                continue
            if np.isfinite(x_value) and np.isfinite(y_value):
                x_values.append(x_value)
                y_values.append(y_value)
    if len(x_values) < 2:
        raise ValueError("x/y spectrum export must contain at least two numeric rows")
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def summarize_xy_spectrum(
    axis_ppm: Iterable[float], intensities: Iterable[float]
) -> SpectrumAxisSummary:
    """Summarize ppm limits, stored orientation, and relative intensity scale."""

    axis = np.asarray(tuple(axis_ppm), dtype=float)
    values = np.asarray(tuple(intensities), dtype=float)
    if axis.ndim != 1 or values.ndim != 1 or axis.shape != values.shape or axis.size < 2:
        raise ValueError("axis and intensities must be matching 1D arrays")
    differences = np.diff(axis)
    if np.all(differences > 0):
        orientation = "ascending"
    elif np.all(differences < 0):
        orientation = "descending"
    else:
        orientation = "non_monotonic"
    return SpectrumAxisSummary(
        point_count=int(axis.size),
        ppm_min=float(np.min(axis)),
        ppm_max=float(np.max(axis)),
        orientation=orientation,
        intensity_abs_max=float(np.max(np.abs(values))),
    )


def compare_peak_positions(
    reference_ppm: Iterable[float],
    candidate_ppm: Iterable[float],
    *,
    tolerance_ppm: float = 0.02,
) -> PeakPositionComparison:
    """Greedily match supplied peak lists within a ppm tolerance.

    This function compares externally supplied or manually curated positions;
    it deliberately does not perform peak detection or assignment.
    """

    reference = _finite_positions(reference_ppm, "reference_ppm")
    candidate = _finite_positions(candidate_ppm, "candidate_ppm")
    tolerance = float(tolerance_ppm)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance_ppm must be finite and non-negative")

    available = list(range(candidate.size))
    offsets: list[float] = []
    for value in reference:
        if not available:
            break
        nearest = min(available, key=lambda index: abs(candidate[index] - value))
        offset = float(candidate[nearest] - value)
        if abs(offset) <= tolerance:
            offsets.append(offset)
            available.remove(nearest)
    if offsets:
        offset_array = np.asarray(offsets)
        median = float(np.median(offset_array))
        mean_absolute = float(np.mean(np.abs(offset_array)))
        maximum_absolute = float(np.max(np.abs(offset_array)))
    else:
        median = mean_absolute = maximum_absolute = None
    return PeakPositionComparison(
        reference_count=int(reference.size),
        candidate_count=int(candidate.size),
        matched_count=len(offsets),
        tolerance_ppm=tolerance,
        median_offset_ppm=median,
        mean_absolute_error_ppm=mean_absolute,
        maximum_absolute_error_ppm=maximum_absolute,
    )


def _finite_positions(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 1D sequence")
    return array
