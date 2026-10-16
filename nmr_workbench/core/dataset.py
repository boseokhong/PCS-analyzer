"""Aggregate root for one NMR experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .metadata import ExperimentalConditions
from .observables import NMRObservable
from .peaks import NMRPeak
from .processing_history import ProcessingHistory
from .spectrum import Spectrum1D


@dataclass
class NMRDataset:
    """Vendor-neutral data, metadata, provenance, peaks, and observables."""

    conditions: ExperimentalConditions = field(default_factory=ExperimentalConditions)
    raw_spectrum: Spectrum1D | None = None
    processed_spectrum: Spectrum1D | None = None
    processing_history: ProcessingHistory = field(default_factory=ProcessingHistory)
    acquisition_parameters: dict[str, Any] = field(default_factory=dict)
    vendor_metadata: dict[str, Any] = field(default_factory=dict)
    peaks: list[NMRPeak] = field(default_factory=list)
    observables: list[NMRObservable] = field(default_factory=list)
    source_path: Path | None = None
    vendor: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))
    experimental_role: str = "unspecified"
    spectral_revision: int = 0
    source_provenance: dict[str, Any] = field(default_factory=dict)
    restored_committed_spectrum: Spectrum1D | None = field(default=None, repr=False, compare=False)

    def mark_spectrum_changed(self) -> None:
        """Invalidate tracking snapshots after committed processing/reference edits.

        Preview and deterministic selection replay must not advance this counter.
        External spectrum writers should call this after a successful mutation.
        """
        self.spectral_revision += 1
        self.restored_committed_spectrum = None

    def __post_init__(self) -> None:
        if type(self.spectral_revision) is not int or self.spectral_revision < 0:
            raise ValueError("spectral_revision must be a non-negative integer")
        if self.raw_spectrum is not None and self.raw_spectrum.domain != "time":
            raise ValueError("raw_spectrum must be in the time domain")
        if self.processed_spectrum is not None and self.processed_spectrum.domain != "frequency":
            raise ValueError("processed_spectrum must be in the frequency domain")
        self.acquisition_parameters = dict(self.acquisition_parameters)
        self.vendor_metadata = dict(self.vendor_metadata)
        self.peaks = list(self.peaks)
        self.observables = list(self.observables)
        for observable in self.observables:
            if observable.dataset_id is None:
                observable.dataset_id = self.id
            elif observable.dataset_id != self.id:
                raise ValueError("owned observable dataset_id must match the dataset")
        role = str(self.experimental_role or "unspecified").strip().lower()
        if role not in {"unspecified", "paramagnetic", "diamagnetic"}:
            raise ValueError(
                "experimental_role must be 'unspecified', 'paramagnetic', or 'diamagnetic'"
            )
        self.experimental_role = role

        if self.source_path is not None:
            self.source_path = Path(self.source_path)

