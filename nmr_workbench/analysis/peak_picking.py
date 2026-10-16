from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from nmr_workbench.core import NMRPeak, Spectrum1D


class PeakPickingError(ValueError):
    """Raised when automatic 1D peak-picking parameters are invalid."""


@dataclass(frozen=True)
class PeakPickingParameters:
    """Parameters for positive-peak detection in a processed 1D spectrum."""

    height_fraction: float = 0.01
    prominence_fraction: float = 0.005
    distance_points: int = 3

    def __post_init__(self) -> None:
        height = float(self.height_fraction)
        prominence = float(self.prominence_fraction)

        if not np.isfinite(height) or height < 0.0 or height > 1.0:
            raise PeakPickingError("height_fraction must be between 0 and 1.")
        if not np.isfinite(prominence) or prominence < 0.0 or prominence > 1.0:
            raise PeakPickingError("prominence_fraction must be between 0 and 1.")
        if isinstance(self.distance_points, bool):
            raise PeakPickingError("distance_points must be an integer.")
        try:
            distance = int(self.distance_points)
        except (TypeError, ValueError) as exc:
            raise PeakPickingError("distance_points must be an integer.") from exc
        if distance < 1:
            raise PeakPickingError("distance_points must be at least 1.")

        object.__setattr__(self, "height_fraction", height)
        object.__setattr__(self, "prominence_fraction", prominence)
        object.__setattr__(self, "distance_points", distance)


def pick_positive_peaks_1d(
    spectrum: Spectrum1D,
    *,
    parameters: PeakPickingParameters | None = None,
) -> list[NMRPeak]:
    """Detect positive local maxima in a processed ppm-domain spectrum.

    The real component is used, matching the displayed phased spectrum.

    Height threshold:
        median(real intensity) + height_fraction * (max - median)

    Prominence threshold:
        prominence_fraction * (max - min)

    Fractions are dimensionless and therefore independent of absolute receiver gain.
    """

    if not isinstance(spectrum, Spectrum1D):
        raise TypeError("spectrum must be a Spectrum1D")
    if spectrum.domain != "frequency" or spectrum.axis_unit.lower() != "ppm":
        raise PeakPickingError(
            "Automatic peak picking requires a frequency-domain ppm spectrum."
        )

    params = parameters or PeakPickingParameters()
    axis = np.asarray(spectrum.axis, dtype=float)
    values = np.real(np.asarray(spectrum.intensities)).astype(float, copy=False)

    if axis.ndim != 1 or values.ndim != 1 or axis.size != values.size:
        raise PeakPickingError("Spectrum axis and intensities must be matching 1D arrays.")
    if axis.size < 3:
        return []

    finite = np.isfinite(axis) & np.isfinite(values)
    if np.count_nonzero(finite) < 3:
        return []

    working_axis = axis[finite]
    working_values = values[finite]
    original_indices = np.flatnonzero(finite)

    minimum = float(np.min(working_values))
    maximum = float(np.max(working_values))
    baseline = float(np.median(working_values))
    amplitude = maximum - minimum

    if not np.isfinite(amplitude) or amplitude <= 0.0:
        return []

    positive_span = max(maximum - baseline, 0.0)
    height = baseline + params.height_fraction * positive_span
    prominence = params.prominence_fraction * amplitude

    peak_indices, _ = find_peaks(
        working_values,
        height=height,
        prominence=prominence,
        distance=params.distance_points,
    )

    peaks: list[NMRPeak] = []
    for working_index in peak_indices:
        source_index = int(original_indices[int(working_index)])
        peaks.append(
            NMRPeak(
                position_ppm=float(axis[source_index]),
                height=float(values[source_index]),
                source="auto",
            )
        )

    peaks.sort(key=lambda peak: peak.position_ppm)
    return peaks
