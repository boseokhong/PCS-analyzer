"""Result objects returned by NMR processing operations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from nmr_workbench.core import ProcessingHistory, Spectrum1D


@dataclass
class ProcessingResult:
    """A new spectrum, its complete recipe, and optional diagnostic arrays."""

    spectrum: Spectrum1D
    history: ProcessingHistory = field(default_factory=ProcessingHistory)
    baseline: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        self.history = ProcessingHistory(self.history.steps)
        if self.baseline is not None:
            baseline = np.asarray(self.baseline, dtype=float)
            if baseline.ndim != 1 or baseline.shape != self.spectrum.intensities.shape:
                raise ValueError("baseline must be one-dimensional and match the spectrum")
            self.baseline = baseline
