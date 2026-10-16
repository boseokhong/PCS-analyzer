"""Scientific observables derived from NMR spectra."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class NMRObservable:
    """A quantitative value kept separate from its source spectral peak."""

    nucleus: str
    chemical_shift_ppm: float
    assignment: str | None = None
    uncertainty_ppm: float | None = None
    dataset_id: str | None = None
    source_peak_id: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))
    included: bool = True
    provenance: dict = field(default_factory=dict)

