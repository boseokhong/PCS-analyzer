"""Explicit standard-fit result application to canonical PCS tensor state."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Any, Mapping, MutableMapping

from logic.pcs_forward_api import AngleUnit, EulerConvention
from logic.pcs_model_state import update_canonical_tensor
from logic.pcs_orientation import matrix_to_euler, orientation_matrix
from logic.pcs_fit_result_schema import validate_fit_result_provenance
from logic.pcs_fit_result_status import (
    FIT_RESULT_STALE,
    STALE_FIT_RESULT_MESSAGE,
    evaluate_fit_result_status,
)
from logic.pcs_tensor_state import (
    ApplicationPCSTensor,
    PCS_TENSOR_STATE_KEY,
    get_canonical_tensor_state,
)


class FitResultApplicationError(ValueError):
    """Raised when a fit result cannot safely define a canonical tensor."""


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FitResultApplicationError(f"Fit result {name} is missing or invalid.") from exc
    if not isfinite(result):
        raise FitResultApplicationError(f"Fit result {name} must be finite.")
    return result


def _finite_triplet(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise FitResultApplicationError(f"Fit result {name} must contain three values.")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise FitResultApplicationError(
            f"Fit result {name} must contain three values."
        ) from exc
    if len(items) != 3:
        raise FitResultApplicationError(f"Fit result {name} must contain three values.")
    return tuple(
        _finite_float(item, f"{name}[{index}]")
        for index, item in enumerate(items)
    )


def _validated_result_values(
    result: Mapping[str, Any] | None,
) -> tuple[str, tuple[float, float, float], float, float | None]:
    if not isinstance(result, Mapping):
        raise FitResultApplicationError("No fit result is available.")

    mode = str(result.get("mode", ""))
    if mode == "theta_alpha_multi":
        raise FitResultApplicationError(
            "Mode A uses a donor-dependent orientation and cannot be applied "
            "to the canonical tensor."
        )
    try:
        validate_fit_result_provenance(result)
    except ValueError as exc:
        raise FitResultApplicationError(str(exc)) from exc
    if mode == "euler_global":
        xyz = tuple(
            _finite_float(result.get(key), key)
            for key in ("ax", "ay", "az")
        )
        return mode, xyz, _finite_float(result.get("delta_chi_ax"), "delta_chi_ax"), None
    if mode == "full_tensor":
        xyz = _finite_triplet(result.get("euler_deg"), "euler_deg")
        # A valid standard Mode C result includes its fitted metal position.
        # It is deliberately validated but not mapped into canonical tensor
        # state because L7 has no canonical metal-position model or event.
        _finite_triplet(result.get("metal_pos"), "metal_pos")
        return (
            mode,
            xyz,
            _finite_float(result.get("delta_chi_ax"), "delta_chi_ax"),
            _finite_float(result.get("delta_chi_rh"), "delta_chi_rh"),
        )
    raise FitResultApplicationError("Only Mode B and Mode C fit results are supported.")


def fit_result_application_status(
    result: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Return whether the result is safe to apply and a concise UI message."""
    try:
        mode, _xyz, _dchi_ax, _dchi_rh = _validated_result_values(result)
    except FitResultApplicationError as exc:
        return False, str(exc)
    if state is not None:
        status = evaluate_fit_result_status(result, state)
        if status["status"] == FIT_RESULT_STALE:
            return False, STALE_FIT_RESULT_MESSAGE
    if mode == "full_tensor":
        return (
            True,
            "Applies tensor/orientation only; fitted metal remains available via "
            "Preview fitted geometry.",
        )
    return True, "Applies the fitted axial tensor/orientation to canonical PCS state."


def canonical_tensor_from_fit_result(
    current: ApplicationPCSTensor,
    result: Mapping[str, Any],
) -> ApplicationPCSTensor:
    """Map a validated Mode B/C result into the current canonical representation."""
    if not isinstance(current, ApplicationPCSTensor):
        raise TypeError("current must be an ApplicationPCSTensor")
    mode, xyz_degrees, dchi_ax, dchi_rh = _validated_result_values(result)
    rotation = orientation_matrix(
        xyz_degrees,
        EulerConvention.EXTRINSIC_XYZ,
        AngleUnit.DEGREE,
    )
    canonical_angles = matrix_to_euler(
        rotation,
        current.frame_convention,
        current.angle_unit,
    )
    updates = {
        "delta_chi_ax": dchi_ax,
        "alpha": canonical_angles[0],
        "beta": canonical_angles[1],
        "gamma": canonical_angles[2],
    }
    if mode == "full_tensor":
        updates["delta_chi_rh"] = dchi_rh
    return replace(current, **updates)


def apply_fit_result_to_canonical_tensor(
    state: MutableMapping[str, Any],
    result: Mapping[str, Any] | None = None,
) -> bool:
    """Apply one fit result through the atomic canonical tensor command path.

    Any active legacy plot override is removed before the event-driven refresh
    so the fitted rotation is not applied once as geometry and again as the
    canonical tensor orientation. ``last_fit_result`` is intentionally kept.
    """
    selected = state.get("last_fit_result") if result is None else result
    # Validate the entire result before mutating any application state.
    _validated_result_values(selected)
    status = evaluate_fit_result_status(selected, state)
    if status["status"] == FIT_RESULT_STALE:
        raise FitResultApplicationError(STALE_FIT_RESULT_MESSAGE)
    current = get_canonical_tensor_state(state)
    updated = canonical_tensor_from_fit_result(current, selected)

    from logic.document_edit_history import FIT_KEYS, begin_edit, finish_edit
    edit_token = begin_edit(state, FIT_KEYS)
    previous_override = state.pop("fit_override", None)
    try:
        changed = update_canonical_tensor(
            state,
            updated,
            source="fitting:apply-result",
            legacy_callback=state.get("update_graph"),
        )
    except Exception:
        # Listener errors occur after canonical installation and must not
        # restore a now-incompatible geometry override. Pre-update failures do.
        if state.get(PCS_TENSOR_STATE_KEY) is current and previous_override is not None:
            state["fit_override"] = previous_override
        raise
    finally:
        mode_label = {"theta_alpha_multi": "Mode A", "euler_global": "Mode B",
                      "full_tensor": "Mode C"}.get(selected.get("mode"), "Fit")
        finish_edit(state, edit_token, f"Apply fit · {mode_label}")

    if not changed and previous_override is not None:
        refresh = state.get("update_graph")
        if callable(refresh):
            refresh()
    return changed


__all__ = [
    "FitResultApplicationError",
    "apply_fit_result_to_canonical_tensor",
    "canonical_tensor_from_fit_result",
    "fit_result_application_status",
]
