"""Numerical one-dimensional spectrum representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

SpectrumDomain = Literal["time", "frequency"]
SpectrumState = Literal["raw", "processed"]


@dataclass
class Spectrum1D:
    """A vendor-neutral one-dimensional FID or frequency-domain spectrum."""

    axis: NDArray[np.floating]
    intensities: NDArray[np.number]
    axis_unit: str
    domain: SpectrumDomain
    state: SpectrumState

    def __post_init__(self) -> None:
        self.axis = np.asarray(self.axis, dtype=float)
        self.intensities = np.asarray(self.intensities)
        if self.axis.ndim != 1 or self.intensities.ndim != 1:
            raise ValueError("Spectrum1D arrays must be one-dimensional")
        if self.axis.size != self.intensities.size:
            raise ValueError("axis and intensities must have the same length")
        if self.domain not in {"time", "frequency"}:
            raise ValueError(f"unsupported spectrum domain: {self.domain}")
        if self.state not in {"raw", "processed"}:
            raise ValueError(f"unsupported spectrum state: {self.state}")

    @property
    def point_count(self) -> int:
        """Number of numerical points in the spectrum."""

        return int(self.axis.size)

