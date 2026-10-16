"""Extensible PCS derivation providers used by PCS Preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

from ..core.experimental_pcs import ExperimentalPCSValue
from .shift_analysis import ShiftComparisonRow


class PCSDerivationError(ValueError):
    """Raised when a PCS value cannot be derived from the supplied inputs."""


@dataclass(frozen=True)
class PCSDerivationContext:
    paramagnetic_dataset_id: str
    diamagnetic_dataset_id: str
    temperature_K: float | None = None
    corrections_ppm: Mapping[str, float] = field(default_factory=dict)
    manual_pcs_ppm: Mapping[str, float] = field(default_factory=dict)
    uncertainties_ppm: Mapping[str, float] = field(default_factory=dict)


class PCSDerivationProvider(Protocol):
    provider_id: str
    display_name: str
    description: str

    def derive(
        self,
        row: ShiftComparisonRow,
        context: PCSDerivationContext,
    ) -> ExperimentalPCSValue:
        ...


def _matched_delta(row: ShiftComparisonRow) -> float:
    if row.status != "Matched" or row.delta_para_ppm is None:
        raise PCSDerivationError(
            f"{row.assignment}: a matched paramagnetic/diamagnetic shift pair is required."
        )
    return float(row.delta_para_ppm)


def _uncertainty(context: PCSDerivationContext, assignment: str) -> float | None:
    value = context.uncertainties_ppm.get(assignment)
    return None if value is None else float(value)


def _source_observable_ids(row: ShiftComparisonRow) -> tuple[str, ...]:
    return tuple(
        value
        for value in (
            row.paramagnetic_observable_id,
            row.diamagnetic_observable_id,
        )
        if value
    )


class DirectDeltaShiftProvider:
    provider_id = "direct_delta_shift"
    display_name = "Direct from Δshift"
    description = (
        "Treat δparamagnetic − δdiamagnetic directly as experimental PCS. "
        "Use only when this approximation is scientifically justified."
    )

    def derive(self, row, context):
        delta = _matched_delta(row)
        return ExperimentalPCSValue(
            assignment=row.assignment,
            pcs_ppm=delta,
            method=self.provider_id,
            uncertainty_ppm=_uncertainty(context, row.assignment),
            temperature_K=context.temperature_K,
            source_dataset_ids=(
                context.paramagnetic_dataset_id,
                context.diamagnetic_dataset_id,
            ),
            source_observable_ids=_source_observable_ids(row),
            provenance={
                "derivation": "delta_para_as_pcs",
                "delta_para_ppm": delta,
            },
        )


class ManualFCSProvider:
    provider_id = "manual_fcs"
    display_name = "FCS-corrected"
    description = "Calculate PCS = Δshift − manually supplied FCS/contact correction."

    def derive(self, row, context):
        delta = _matched_delta(row)
        if row.assignment not in context.corrections_ppm:
            raise PCSDerivationError(
                f"{row.assignment}: FCS correction is required."
            )
        correction = float(context.corrections_ppm[row.assignment])
        return ExperimentalPCSValue(
            assignment=row.assignment,
            pcs_ppm=delta - correction,
            method=self.provider_id,
            uncertainty_ppm=_uncertainty(context, row.assignment),
            temperature_K=context.temperature_K,
            source_dataset_ids=(
                context.paramagnetic_dataset_id,
                context.diamagnetic_dataset_id,
            ),
            source_observable_ids=_source_observable_ids(row),
            provenance={
                "derivation": "delta_para_minus_fcs",
                "delta_para_ppm": delta,
                "fcs_ppm": correction,
                "correction_source": "manual",
            },
        )


class ManualPCSProvider:
    provider_id = "manual_pcs"
    display_name = "Manual / Imported PCS"
    description = "Use a manually supplied or externally imported PCS value."

    def derive(self, row, context):
        # A matched row is still used as the assignment/provenance anchor, but
        # the supplied PCS value does not have to equal Δshift.
        _matched_delta(row)
        if row.assignment not in context.manual_pcs_ppm:
            raise PCSDerivationError(
                f"{row.assignment}: manual PCS value is required."
            )
        pcs = float(context.manual_pcs_ppm[row.assignment])
        return ExperimentalPCSValue(
            assignment=row.assignment,
            pcs_ppm=pcs,
            method=self.provider_id,
            uncertainty_ppm=_uncertainty(context, row.assignment),
            temperature_K=context.temperature_K,
            source_dataset_ids=(
                context.paramagnetic_dataset_id,
                context.diamagnetic_dataset_id,
            ),
            source_observable_ids=_source_observable_ids(row),
            provenance={
                "derivation": "manual_or_imported_pcs",
                "delta_para_ppm": float(row.delta_para_ppm),
                "manual_pcs_ppm": pcs,
            },
        )


def builtin_pcs_providers():
    """Return providers in the order presented to the user."""
    return (
        DirectDeltaShiftProvider(),
        ManualFCSProvider(),
        ManualPCSProvider(),
    )
