"""Validation and normalization for persisted structured PCS observations."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from logic.pcs_forward_api import PCSStructureTarget, TargetReduction


OBSERVATIONS_KEY = "experimental_pcs_observations"
SUPPORTED_SCOPES = {
    "Single atom",
    "NMR site",
    "Matching ligand sites",
    "Custom refs",
}
SUPPORTED_MATCHING_STATUSES = {None, "", "unique"}


def structure_target_from_payload(
    payload: Mapping[str, Any],
    *,
    assignment: str,
    target_id: str,
    experimental_pcs_ppm: float | None = None,
    uncertainty_ppm: float | None = None,
) -> PCSStructureTarget:
    """Build the canonical two-stage target used by UI and fitting consumers."""
    if not isinstance(payload, Mapping):
        raise ValueError("structured PCS target is missing or malformed")
    scope = payload.get("scope")
    if scope not in SUPPORTED_SCOPES:
        raise ValueError(f"unsupported structured PCS target scope {scope!r}")
    matching_status = payload.get("matching_status")
    if matching_status not in SUPPORTED_MATCHING_STATUSES:
        raise ValueError(f"structured PCS matching status is {matching_status!r}")
    raw_groups = payload.get("site_member_ref_ids")
    if not isinstance(raw_groups, (list, tuple)):
        raise ValueError("structured PCS target member list is missing or malformed")
    groups = []
    for group in raw_groups:
        if not isinstance(group, (list, tuple)):
            raise ValueError("structured PCS target members must be nested by site")
        groups.append(tuple(int(ref_id) for ref_id in group))
    hint = str(payload.get("averaging_hint", "") or "none")
    if hint not in {"none", "mean_members"}:
        raise ValueError(f"unsupported structured PCS averaging hint {hint!r}")
    member_reduction = (
        TargetReduction.ARITHMETIC_MEAN
        if hint == "mean_members" or any(len(group) > 1 for group in groups)
        else TargetReduction.NONE
    )
    site_reduction = (
        TargetReduction.ARITHMETIC_MEAN
        if len(groups) > 1
        else TargetReduction.NONE
    )
    return PCSStructureTarget(
        target_id=str(target_id),
        site_member_ref_ids=tuple(groups),
        member_reduction=member_reduction,
        site_reduction=site_reduction,
        assignment=str(assignment),
        experimental_pcs_ppm=experimental_pcs_ppm,
        uncertainty_ppm=uncertainty_ppm,
        metadata={
            "scope": scope,
            "choice_key": payload.get("choice_key"),
            "averaging_hint": hint,
            "matching_status": matching_status,
            "site_ids": tuple(str(value) for value in (payload.get("site_ids", ()) or ())),
            "source": payload.get("source"),
        },
    )


def canonical_provenance_json(value: Any) -> str:
    """Return an immutable deterministic copy of JSON-compatible provenance."""
    try:
        return json.dumps(value or {}, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("structured PCS provenance must be JSON-compatible") from exc


def finite_pcs(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("structured experimental PCS must be finite")
    return number


__all__ = [
    "OBSERVATIONS_KEY",
    "SUPPORTED_MATCHING_STATUSES",
    "SUPPORTED_SCOPES",
    "canonical_provenance_json",
    "finite_pcs",
    "structure_target_from_payload",
]
