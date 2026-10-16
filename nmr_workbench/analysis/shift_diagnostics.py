"""Shift-analysis diagnostics for NMR Workbench.

This module bridges assignment-matched paramagnetic/diamagnetic NMR data with
the host PCS Analyzer *calculated* PCS layer.  It does not alter any host data.

Important:
    non_pcs_ppm = delta_para_ppm - pcs_calculated_ppm

is a diagnostic residual-like quantity.  It is not automatically interpreted
as a physically isolated FCS/contact contribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from ..integration.structured_pcs_prediction import predict_structured_pcs
from .shift_analysis import ShiftComparisonRow


@dataclass(frozen=True)
class ShiftDiagnosticRow:
    assignment: str
    paramagnetic_ppm: float | None
    diamagnetic_ppm: float | None
    delta_para_ppm: float | None
    pcs_calculated_ppm: float | None
    non_pcs_ppm: float | None
    target_ref_id: int | None
    status: str
    paramagnetic_observable_id: str | None = None
    diamagnetic_observable_id: str | None = None
    structured_target_id: str | None = None
    target_site_member_ref_ids: tuple[tuple[int, ...], ...] = ()
    target_member_ref_ids: tuple[int, ...] = ()
    target_scope: str | None = None
    matching_status: str | None = None
    structured_prediction_status: str | None = None


@dataclass(frozen=True)
class PredictedParamagneticPeak:
    """One non-destructive predicted position on a paramagnetic spectrum."""

    assignment: str
    position_ppm: float
    diamagnetic_ppm: float
    pcs_calculated_ppm: float
    fixed_correction_ppm: float
    target_ref_id: int | None
    diamagnetic_observable_id: str | None = None
    paramagnetic_observable_id: str | None = None
    structured_target_id: str | None = None
    target_site_member_ref_ids: tuple[tuple[int, ...], ...] = ()
    target_member_ref_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class AbsoluteSpectrumPosition:
    """An assignment-linked coordinate on an absolute ppm spectrum axis."""

    assignment: str
    position_ppm: float
    kind: str
    target_ref_id: int | None


@dataclass(frozen=True)
class ShiftDiagnosticQuantity:
    """An assignment-linked value on a shift-difference ppm axis."""

    assignment: str
    value_ppm: float
    kind: str
    target_ref_id: int | None


def _targets_from_state(state: dict | None):
    if not isinstance(state, dict):
        return []

    atom_by_id = state.get("atom_by_id", {}) or {}
    overrides = state.get("ref_label_overrides", {}) or {}
    selected = state.get("current_selected_ids", []) or []

    if selected:
        ids = [int(value) for value in selected]
    else:
        ids = sorted(
            {int(value) for value in atom_by_id}
            | {int(value) for value in overrides}
        )

    return [
        (
            ref_id,
            str(overrides.get(ref_id, atom_by_id.get(ref_id, "")) or ""),
        )
        for ref_id in ids
    ]


def suggest_ref_id(
    assignment: str | None,
    state: dict | None,
) -> int | None:
    text = (assignment or "").strip()
    if not text:
        return None

    targets = _targets_from_state(state)

    if text.isdigit():
        ref_id = int(text)
        if any(candidate == ref_id for candidate, _ in targets):
            return ref_id

    folded = text.casefold()
    matches = [
        ref_id
        for ref_id, label in targets
        if label.strip().casefold() == folded
    ]
    return matches[0] if len(matches) == 1 else None


def build_shift_diagnostics(
    rows: Iterable[ShiftComparisonRow],
    state: dict | None,
    *,
    structured_target_choices: Mapping[str, object] | None = None,
) -> list[ShiftDiagnosticRow]:
    pcs_by_id = (
        (state.get("pcs_by_id", {}) or {})
        if isinstance(state, dict)
        else {}
    )

    output: list[ShiftDiagnosticRow] = []
    for row in rows:
        target_choice = (structured_target_choices or {}).get(row.assignment)
        structured = predict_structured_pcs(
            row,
            state,
            target_choice=target_choice,
        )
        ref_id = (
            structured.primary_ref_id
            if structured is not None
            else suggest_ref_id(row.assignment, state)
        )
        pcs = None
        non_pcs = None
        status = row.status

        if structured is not None:
            pcs = structured.pcs_ppm
            if structured.unavailable_reason:
                status = (
                    f"{row.status} · PCS unavailable: "
                    f"{structured.unavailable_reason}"
                )
        elif ref_id is not None and ref_id in pcs_by_id:
            try:
                pcs = float(pcs_by_id[ref_id])
            except (TypeError, ValueError):
                pcs = None

        if row.delta_para_ppm is not None and pcs is not None:
            non_pcs = float(row.delta_para_ppm) - pcs

        output.append(
            ShiftDiagnosticRow(
                assignment=row.assignment,
                paramagnetic_ppm=row.paramagnetic_ppm,
                diamagnetic_ppm=row.diamagnetic_ppm,
                delta_para_ppm=row.delta_para_ppm,
                pcs_calculated_ppm=pcs,
                non_pcs_ppm=non_pcs,
                target_ref_id=ref_id,
                status=status,
                paramagnetic_observable_id=getattr(
                    row, "paramagnetic_observable_id", None
                ),
                diamagnetic_observable_id=getattr(
                    row, "diamagnetic_observable_id", None
                ),
                structured_target_id=(
                    None if structured is None else structured.target_id
                ),
                target_site_member_ref_ids=(
                    () if structured is None else structured.site_member_ref_ids
                ),
                target_member_ref_ids=(
                    () if structured is None else structured.member_ref_ids
                ),
                target_scope=(None if structured is None else structured.scope),
                matching_status=(
                    None if structured is None else structured.matching_status
                ),
                structured_prediction_status=(
                    None
                    if structured is None or structured.available
                    else structured.unavailable_reason
                ),
            )
        )
    return output


def build_predicted_paramagnetic_peaks(
    rows: Iterable[ShiftDiagnosticRow],
    *,
    fixed_corrections_ppm: Mapping[str, float] | None = None,
) -> list[PredictedParamagneticPeak]:
    """Build absolute predicted positions from dia + calculated PCS + correction.

    ``delta_para_ppm`` is deliberately not used: it is an observed shift
    difference, not an absolute spectrum coordinate. Missing or non-finite
    diamagnetic/PCS inputs are skipped. An absent or invalid optional fixed
    correction defaults to zero.
    """
    corrections = fixed_corrections_ppm or {}
    output: list[PredictedParamagneticPeak] = []

    for row in rows:
        if row.diamagnetic_ppm is None or row.pcs_calculated_ppm is None:
            continue
        try:
            dia = float(row.diamagnetic_ppm)
            pcs = float(row.pcs_calculated_ppm)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(dia) or not math.isfinite(pcs):
            continue

        try:
            correction = float(corrections.get(row.assignment, 0.0))
        except (TypeError, ValueError):
            correction = 0.0
        if not math.isfinite(correction):
            correction = 0.0

        position = dia + pcs + correction
        if not math.isfinite(position):
            continue
        output.append(
            PredictedParamagneticPeak(
                assignment=row.assignment,
                position_ppm=position,
                diamagnetic_ppm=dia,
                pcs_calculated_ppm=pcs,
                fixed_correction_ppm=correction,
                target_ref_id=row.target_ref_id,
                diamagnetic_observable_id=row.diamagnetic_observable_id,
                paramagnetic_observable_id=row.paramagnetic_observable_id,
                structured_target_id=row.structured_target_id,
                target_site_member_ref_ids=row.target_site_member_ref_ids,
                target_member_ref_ids=row.target_member_ref_ids,
            )
        )

    return output


def build_absolute_spectrum_positions(
    rows: Iterable[ShiftDiagnosticRow],
    *,
    fixed_corrections_ppm: Mapping[str, float] | None = None,
) -> list[AbsoluteSpectrumPosition]:
    """Return only quantities that are valid absolute spectrum coordinates."""
    row_list = list(rows)
    output: list[AbsoluteSpectrumPosition] = []
    for row in row_list:
        for kind, value in (
            ("observed_paramagnetic", row.paramagnetic_ppm),
            ("observed_diamagnetic", row.diamagnetic_ppm),
        ):
            if value is None:
                continue
            try:
                position = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(position):
                output.append(
                    AbsoluteSpectrumPosition(
                        assignment=row.assignment,
                        position_ppm=position,
                        kind=kind,
                        target_ref_id=row.target_ref_id,
                    )
                )

    output.extend(
        AbsoluteSpectrumPosition(
            assignment=prediction.assignment,
            position_ppm=prediction.position_ppm,
            kind="predicted_paramagnetic",
            target_ref_id=prediction.target_ref_id,
        )
        for prediction in build_predicted_paramagnetic_peaks(
            row_list,
            fixed_corrections_ppm=fixed_corrections_ppm,
        )
    )
    return output


def build_shift_diagnostic_quantities(
    rows: Iterable[ShiftDiagnosticRow],
    *,
    fixed_corrections_ppm: Mapping[str, float] | None = None,
) -> list[ShiftDiagnosticQuantity]:
    """Return only difference-space quantities used by Shift Diagnostics."""
    corrections = fixed_corrections_ppm or {}
    output: list[ShiftDiagnosticQuantity] = []

    for row in rows:
        values = (
            ("delta_para_observed", row.delta_para_ppm),
            ("pcs_calculated", row.pcs_calculated_ppm),
            ("implied_non_pcs", row.non_pcs_ppm),
        )
        for kind, value in values:
            if value is None:
                continue
            try:
                difference = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(difference):
                output.append(
                    ShiftDiagnosticQuantity(
                        assignment=row.assignment,
                        value_ppm=difference,
                        kind=kind,
                        target_ref_id=row.target_ref_id,
                    )
                )

        if row.assignment not in corrections:
            continue
        try:
            correction = float(corrections[row.assignment])
        except (TypeError, ValueError):
            continue
        if math.isfinite(correction):
            output.append(
                ShiftDiagnosticQuantity(
                    assignment=row.assignment,
                    value_ppm=correction,
                    kind="fixed_correction",
                    target_ref_id=row.target_ref_id,
                )
            )

    return output
