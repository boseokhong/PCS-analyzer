"""Adapter from persisted NMR targets to the canonical PCS forward engine.

The adapter is intentionally narrow: it translates the existing
``structured_v1`` observation payload into :class:`PCSStructureTarget`, asks
the shared forward engine for one reduced value, and returns metadata suitable
for Shift Diagnostics.  It never expands target members into NMR data rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from logic.pcs_forward_api import (
    PCSAtom,
    PCSPredictionRequest,
    PCSStructureTarget,
)
from logic.pcs_forward_model import PCSForwardEngine
from logic.pcs_prediction_model import (
    get_prediction_model,
    prediction_tensor_from_components,
)
from logic.pcs_tensor_state import get_canonical_tensor_state
from logic.pcs_structured_observations import structure_target_from_payload
from logic.symmetry_geometry import effective_geometry_factors

from .structure_aware_pcs_targets import PCSStructureTargetChoice


_ENGINE = PCSForwardEngine()
_OBSERVATIONS_KEY = "experimental_pcs_observations"


@dataclass(frozen=True)
class StructuredPCSPrediction:
    """One observable-level PCS prediction or a concise unavailable result."""

    target_id: str
    pcs_ppm: float | None
    site_member_ref_ids: tuple[tuple[int, ...], ...]
    member_ref_ids: tuple[int, ...]
    primary_ref_id: int | None
    scope: str | None
    averaging_hint: str | None
    matching_status: str | None
    site_values_ppm: tuple[float, ...] = ()
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.pcs_ppm is not None and self.unavailable_reason is None


def _row_observable_ids(row: Any) -> set[str]:
    return {
        str(value)
        for value in (
            getattr(row, "paramagnetic_observable_id", None),
            getattr(row, "diamagnetic_observable_id", None),
        )
        if value
    }


def _row_dataset_ids(row: Any) -> set[str]:
    return {
        str(value)
        for value in (
            getattr(row, "paramagnetic_dataset_id", None),
            getattr(row, "diamagnetic_dataset_id", None),
        )
        if value
    }


def resolve_structured_observation(
    row: Any,
    state: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Resolve a stored target by stable identity before assignment text.

    A non-unique match is deliberately left unresolved.  This prevents two
    datasets using the same assignment label from silently sharing a target.
    """

    if not isinstance(state, Mapping):
        return None
    records = tuple(
        record
        for record in (state.get(_OBSERVATIONS_KEY, ()) or ())
        if isinstance(record, Mapping) and isinstance(record.get("target"), Mapping)
    )
    if not records:
        return None

    observable_ids = _row_observable_ids(row)
    if observable_ids:
        matches = [
            record
            for record in records
            if observable_ids.issubset(
                {str(value) for value in (record.get("source_observable_ids", ()) or ())}
            )
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None

    dataset_ids = _row_dataset_ids(row)
    if dataset_ids:
        matches = [
            record
            for record in records
            if not (record.get("source_observable_ids", ()) or ())
            if dataset_ids.issubset(
                {str(value) for value in (record.get("source_dataset_ids", ()) or ())}
            )
            and str(record.get("assignment", "")).casefold()
            == str(getattr(row, "assignment", "")).casefold()
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None

    assignment = str(getattr(row, "assignment", "")).strip().casefold()
    matches = [
        record
        for record in records
        if not (record.get("source_observable_ids", ()) or ())
        and not (record.get("source_dataset_ids", ()) or ())
        if str(record.get("assignment", "")).strip().casefold() == assignment
    ]
    return matches[0] if len(matches) == 1 else None


def _payload_from_choice(choice: PCSStructureTargetChoice) -> dict[str, Any]:
    return {
        "scope": choice.scope,
        "choice_key": choice.key,
        "site_member_ref_ids": [list(group) for group in choice.site_member_ref_ids],
        "primary_ref_id": choice.primary_ref_id,
        "averaging_hint": choice.averaging,
        "matching_status": choice.matching_status,
        "site_ids": list(choice.site_ids),
        "source": choice.source,
    }


def _target_from_payload(
    target_payload: Mapping[str, Any],
    *,
    assignment: str,
    target_id: str,
) -> PCSStructureTarget:
    return structure_target_from_payload(
        target_payload,
        assignment=assignment,
        target_id=target_id,
    )


def _coordinate_map(state: Mapping[str, Any]) -> dict[int, tuple[float, float, float]]:
    result: dict[int, tuple[float, float, float]] = {}
    ref_ids = tuple(int(value) for value in (state.get("current_selected_ids", ()) or ()))
    coords = state.get("_pcs_last_rotated_coords", ())
    if coords is None:
        coords = ()
    for ref_id, coordinate in zip(ref_ids, coords):
        values = tuple(float(value) for value in coordinate)
        if len(values) == 3 and all(math.isfinite(value) for value in values):
            result[ref_id] = values
    for ref_id, coordinate in dict(state.get("last_rotated_raw_by_id", {}) or {}).items():
        values = tuple(float(value) for value in coordinate)
        if len(values) == 3 and all(math.isfinite(value) for value in values):
            result.setdefault(int(ref_id), values)
    return result


def _torsion_groups(state: Mapping[str, Any]):
    enabled = state.get("torsion_avg_enabled_var")
    try:
        enabled = bool(enabled.get())
    except AttributeError:
        enabled = bool(enabled)
    return (state.get("torsion_avg_groups", ()) or ()) if enabled else ()


def predict_structured_pcs(
    row: Any,
    state: dict | None,
    *,
    target_choice: PCSStructureTargetChoice | None = None,
) -> StructuredPCSPrediction | None:
    """Return one engine-reduced PCS for the row's structured target.

    ``None`` means no structured target is linked and callers may retain the
    legacy one-Ref path.  A returned unavailable result means a target was
    linked but could not be evaluated and must not silently fall back to a
    different atom.
    """

    if not isinstance(state, dict):
        return None
    record = None if target_choice is not None else resolve_structured_observation(row, state)
    if target_choice is None and record is None:
        return None

    target_payload = (
        _payload_from_choice(target_choice)
        if target_choice is not None
        else dict(record.get("target", {}) or {})
    )
    assignment = str(getattr(row, "assignment", "") or "")
    target_id = str(
        target_payload.get("choice_key")
        or (record or {}).get("id")
        or next(iter(_row_observable_ids(row)), "")
        or f"assignment:{assignment}"
    )
    raw_groups = target_payload.get("site_member_ref_ids", ()) or ()
    try:
        groups = tuple(tuple(int(value) for value in group) for group in raw_groups)
    except Exception:
        groups = ()
    flat = tuple(value for group in groups for value in group)
    primary = target_payload.get("primary_ref_id")
    try:
        primary_ref_id = int(primary) if primary is not None else (flat[0] if flat else None)
    except (TypeError, ValueError):
        primary_ref_id = flat[0] if flat else None

    def unavailable(reason: str) -> StructuredPCSPrediction:
        return StructuredPCSPrediction(
            target_id=target_id,
            pcs_ppm=None,
            site_member_ref_ids=groups,
            member_ref_ids=flat,
            primary_ref_id=primary_ref_id,
            scope=target_payload.get("scope"),
            averaging_hint=target_payload.get("averaging_hint"),
            matching_status=target_payload.get("matching_status"),
            unavailable_reason=reason,
        )

    matching_status = target_payload.get("matching_status")
    if matching_status not in {None, "", "unique"}:
        return unavailable(f"matching status is {matching_status}")

    try:
        target = _target_from_payload(
            target_payload,
            assignment=assignment,
            target_id=target_id,
        )
        coordinates = _coordinate_map(state)
        missing = [ref_id for ref_id in target.member_ref_ids if ref_id not in coordinates]
        if missing:
            return unavailable(
                "missing structure Ref " + ", ".join(map(str, missing))
            )

        ref_ids = target.member_ref_ids
        positions = tuple(coordinates[ref_id] for ref_id in ref_ids)
        atoms = tuple(
            PCSAtom(ref_id=ref_id, position=position)
            for ref_id, position in zip(ref_ids, positions)
        )
        canonical = get_canonical_tensor_state(state)
        tensor = prediction_tensor_from_components(
            dchi_ax=canonical.delta_chi_ax,
            dchi_rh=canonical.delta_chi_rh,
            rhombic_enabled=canonical.rhombic_enabled,
            model=get_prediction_model(state),
        )
        request = PCSPredictionRequest(
            tensor=tensor,
            atoms=atoms,
            targets=(target,),
            request_id=f"nmr-structured:{target_id}",
            metadata={"consumer": "nmr_workbench"},
        )
        _, _, _, g_ax, g_rh = effective_geometry_factors(
            ref_ids,
            np.asarray(positions, dtype=float),
            np.zeros(3, dtype=float),
            pseudo_members=state.get("symavg_members_by_pseudo_id", {}) or {},
            raw_coords_by_id=state.get("last_rotated_raw_by_id", {}) or {},
            torsion_groups=_torsion_groups(state),
        )
        effective = {
            ref_id: (float(g_ax[index]), float(g_rh[index]))
            for index, ref_id in enumerate(ref_ids)
        }
        result = _ENGINE.predict_with_effective_geometry(request, effective)
        predicted = result.target_predictions[0]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return unavailable(str(exc))

    return StructuredPCSPrediction(
        target_id=target_id,
        pcs_ppm=float(predicted.pcs_ppm),
        site_member_ref_ids=target.site_member_ref_ids,
        member_ref_ids=target.member_ref_ids,
        primary_ref_id=primary_ref_id,
        scope=target_payload.get("scope"),
        averaging_hint=target_payload.get("averaging_hint"),
        matching_status=matching_status,
        site_values_ppm=tuple(float(value) for value in predicted.site_values_ppm),
    )


__all__ = [
    "StructuredPCSPrediction",
    "predict_structured_pcs",
    "resolve_structured_observation",
]
