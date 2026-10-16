"""Deterministic fingerprints for immutable standard-fit inputs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

from logic.pcs_fit_snapshot import StandardFitInputSnapshot


FIT_INPUT_PAYLOAD_SCHEMA = "pcs_fit_input_payload_v3"
FIT_INPUT_FINGERPRINT_SCHEMA = "pcs_fit_input_v3"
FIT_INPUT_FINGERPRINT_ALGORITHM = "sha256"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _finite(value: Any, category: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{category} fingerprint value must be finite")
    return number


def _persisted_decimal(value: Any, digits: int, category: str) -> str:
    """Match fixed-decimal .pcsp scientific persistence exactly."""
    return f"{_finite(value, category):.{digits}f}"


def _round_trip_number(value: Any, category: str) -> str:
    """Canonical binary-float text for settings persisted without rounding."""
    return format(_finite(value, category), ".17g")


def _parameter_category(name: str) -> str:
    if name in {"metal_x", "metal_y", "metal_z"}:
        return "coordinate"
    if name in {"theta", "alpha", "ax", "ay", "az"}:
        return "orientation_angle"
    return "numeric"


def _parameter_number(value: Any, name: str) -> str:
    if _parameter_category(name) == "coordinate":
        return _persisted_decimal(value, 8, "coordinate")
    if _parameter_category(name) == "orientation_angle":
        # Matrix-based XYZ adapters can differ by ~1e-14 degrees after an
        # otherwise exact project round trip. Twelve decimal places preserve
        # far more angular precision than is scientifically meaningful while
        # suppressing only that deterministic decomposition noise.
        return _persisted_decimal(value, 12, "orientation angle")
    return _round_trip_number(value, "optimizer parameter")


def _bound(value: Any, name: str) -> str:
    number = float(value)
    if math.isinf(number):
        return "+infinity" if number > 0 else "-infinity"
    return _parameter_number(number, name)


def _vector(values, formatter) -> list[str]:
    return [formatter(value) for value in values]


def _target_payload(observation) -> dict[str, Any]:
    target = observation.target
    site_ids = tuple(target.metadata.get("site_ids", ()))
    sites = [
        {
            "site_id": site_ids[index] if index < len(site_ids) else None,
            "member_ref_ids": sorted(group),
        }
        for index, group in enumerate(target.site_member_ref_ids)
    ]
    sites.sort(key=lambda row: (row["member_ref_ids"], str(row["site_id"])))
    return {
        "target_id": target.target_id,
        "sites": sites,
        "member_reduction": target.member_reduction.value,
        "site_reduction": target.site_reduction.value,
        "scope": target.metadata.get("scope"),
        "choice_key": target.metadata.get("choice_key"),
        "averaging_hint": target.metadata.get("averaging_hint"),
        "matching_status": target.metadata.get("matching_status"),
        "source": target.metadata.get("source"),
    }


def build_fit_input_fingerprint_payload(
    snapshot: StandardFitInputSnapshot,
) -> dict[str, Any]:
    """Build the versioned JSON-safe payload for one exact fit snapshot.

    Coordinates and legacy experimental PCS values use the eight decimal
    places preserved by .pcsp. Tensor, angle, torsion, and optimizer settings
    use round-trip binary-float text. Infinity in least-squares bounds is an
    explicit string token, never a non-standard JSON number.
    """
    if not isinstance(snapshot, StandardFitInputSnapshot):
        raise TypeError("snapshot must be a StandardFitInputSnapshot")
    geometry = snapshot.geometry
    parameter_names = snapshot.free_parameters
    if len(parameter_names) != len(snapshot.x0):
        raise ValueError("snapshot free parameters and x0 are inconsistent")
    if not (
        len(snapshot.lower_bounds)
        == len(snapshot.upper_bounds)
        == len(snapshot.x0)
    ):
        raise ValueError("snapshot optimizer bounds are inconsistent")

    def coordinates(rows):
        return [
            _vector(row, lambda value: _persisted_decimal(value, 8, "coordinate"))
            for row in rows
        ]

    optimizer = snapshot.optimizer
    de_bounds = None
    if optimizer.effective_de_bounds is not None:
        if len(optimizer.effective_de_bounds) != len(parameter_names):
            raise ValueError("snapshot DE bounds are inconsistent")
        de_bounds = [
            [_bound(lo, name), _bound(hi, name)]
            for name, (lo, hi) in zip(parameter_names, optimizer.effective_de_bounds)
        ]

    return {
        "payload_schema": FIT_INPUT_PAYLOAD_SCHEMA,
        "mode": snapshot.mode,
        "weighting_mode": snapshot.weighting_mode,
        "observations": [
            {
                "observable_id": item.observable_id,
                "source": item.source,
                "assignment": item.assignment,
                "legacy_ref_id": (
                    item.ref_id if item.source == "legacy_single_atom" else None
                ),
                "pcs": _persisted_decimal(item.pcs, 8, "experimental PCS"),
                "uncertainty_ppm": (
                    _persisted_decimal(
                        item.uncertainty_ppm, 8, "experimental PCS uncertainty"
                    )
                    if snapshot.weighting_mode == "experimental_uncertainty"
                    else None
                ),
                "source_dataset_ids": sorted(item.source_dataset_ids),
                "source_observable_ids": sorted(item.source_observable_ids),
                "target": _target_payload(item),
            }
            for item in snapshot.observations
        ],
        "geometry": {
            "effective_ref_ids": list(geometry.effective_ref_ids),
            "effective_coordinates": coordinates(geometry.effective_coordinates),
            "raw_ref_ids": list(geometry.raw_ref_ids),
            "raw_coordinates": coordinates(geometry.raw_coordinates),
            "metal_position": _vector(
                geometry.metal_position,
                lambda value: _persisted_decimal(value, 8, "metal coordinate"),
            ),
            "pseudo_members": [
                {"ref_id": ref_id, "members": list(members)}
                for ref_id, members in geometry.pseudo_members
            ],
            "torsion_enabled": geometry.torsion_enabled,
            "torsion_groups": [
                {
                    "enabled": group.enabled,
                    "axis_atoms": list(group.axis_atoms),
                    "rotating_atoms": list(group.rotating_atoms),
                    "n_samples": group.n_samples,
                    "range_mode": group.range_mode,
                    "range_start_deg": _round_trip_number(
                        group.range_start_deg, "torsion angle"
                    ),
                    "range_end_deg": _round_trip_number(
                        group.range_end_deg, "torsion angle"
                    ),
                }
                for group in geometry.torsion_groups
            ],
        },
        "mode_inputs": {
            "donor_ids": list(snapshot.donor_ids),
            "donor_coordinates": coordinates(snapshot.donor_coordinates),
            "axis_mode": snapshot.axis_mode,
            "mode_a_angle_z": _round_trip_number(
                snapshot.mode_a_angle_z, "Mode A angle_z"
            ),
            "use_current_tensor": snapshot.use_current_tensor,
        },
        "tensor": {
            "fit_delta_chi_ax": snapshot.tensor.fit_delta_chi_ax,
            "fit_delta_chi_rh": snapshot.tensor.fit_delta_chi_rh,
            "fixed_delta_chi_ax": _round_trip_number(
                snapshot.tensor.fixed_delta_chi_ax, "fixed delta_chi_ax"
            ),
            "fixed_delta_chi_rh": _round_trip_number(
                snapshot.tensor.fixed_delta_chi_rh, "fixed delta_chi_rh"
            ),
        },
        "optimizer": {
            "identity": optimizer.identity,
            "use_global_search": optimizer.use_global_search,
            "local_optimizer": optimizer.local_optimizer,
            "free_parameters": list(parameter_names),
            "x0": [
                _parameter_number(value, name)
                for name, value in zip(parameter_names, snapshot.x0)
            ],
            "lower_bounds": [
                _bound(value, name)
                for name, value in zip(parameter_names, snapshot.lower_bounds)
            ],
            "upper_bounds": [
                _bound(value, name)
                for name, value in zip(parameter_names, snapshot.upper_bounds)
            ],
            "de_effective_bounds": de_bounds,
            "de_seed": optimizer.de_seed,
            "de_polish": optimizer.de_polish,
            "de_default_span": (
                None
                if optimizer.de_default_span is None
                else _round_trip_number(optimizer.de_default_span, "DE span")
            ),
        },
    }


def fingerprint_fit_input_snapshot(
    snapshot: StandardFitInputSnapshot,
) -> dict[str, str]:
    payload = build_fit_input_fingerprint_payload(snapshot)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "fingerprint_schema": FIT_INPUT_FINGERPRINT_SCHEMA,
        "algorithm": FIT_INPUT_FINGERPRINT_ALGORITHM,
        "digest": hashlib.sha256(encoded).hexdigest(),
    }


def fit_input_reconstruction_config(
    snapshot: StandardFitInputSnapshot,
) -> dict[str, Any]:
    """Minimal saved arguments needed to recapture the same snapshot."""
    structured = any(item.source == "structured_v1" for item in snapshot.observations)
    return {
        "mode": snapshot.mode,
        "observation_model": "structured_v1" if structured else "legacy_single_atom",
        "observation_ids": (
            [item.observable_id for item in snapshot.observations] if structured else []
        ),
        "proton_ids": (
            [] if structured else [item.ref_id for item in snapshot.observations]
        ),
        "donor_ids": list(snapshot.donor_ids),
        "axis_mode": snapshot.axis_mode,
        "fit_delta_chi_ax": snapshot.tensor.fit_delta_chi_ax,
        "fit_delta_chi_rh": snapshot.tensor.fit_delta_chi_rh,
        "use_global_search": snapshot.optimizer.use_global_search,
        "use_current_tensor": snapshot.use_current_tensor,
        "weighting_mode": snapshot.weighting_mode,
    }


def validate_fit_input_fingerprint(value: Any) -> tuple[bool, str]:
    if not isinstance(value, Mapping):
        return False, "Fit input fingerprint is missing or invalid."
    if value.get("fingerprint_schema") != FIT_INPUT_FINGERPRINT_SCHEMA:
        return False, "Fit input fingerprint schema is missing or unsupported."
    if value.get("algorithm") != FIT_INPUT_FINGERPRINT_ALGORITHM:
        return False, "Fit input fingerprint algorithm is missing or unsupported."
    digest = value.get("digest")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        return False, "Fit input fingerprint digest is malformed."
    return True, ""


__all__ = [
    "FIT_INPUT_FINGERPRINT_ALGORITHM",
    "FIT_INPUT_FINGERPRINT_SCHEMA",
    "FIT_INPUT_PAYLOAD_SCHEMA",
    "build_fit_input_fingerprint_payload",
    "fingerprint_fit_input_snapshot",
    "fit_input_reconstruction_config",
    "validate_fit_input_fingerprint",
]
