"""Spectral peak domain objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class NMRPeak:
    """A detected or manually defined feature in a spectrum."""

    position_ppm: float
    height: float | None = None
    integral: float | None = None
    linewidth_hz: float | None = None
    source: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))

