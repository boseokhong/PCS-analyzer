"""Derived experimental PCS values and provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class ExperimentalPCSValue:
    """One fitting-ready experimental PCS value.

    This object deliberately lives above raw NMR observables.  Its value may be
    produced from a simple delta-shift approximation, an FCS correction, a
    future VT/Bleaney model, or an external provider.
    """

    assignment: str
    pcs_ppm: float
    method: str
    uncertainty_ppm: float | None = None
    temperature_K: float | None = None
    source_dataset_ids: tuple[str, ...] = ()
    source_observable_ids: tuple[str, ...] = ()
    target_ref_id: int | None = None
    included: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
