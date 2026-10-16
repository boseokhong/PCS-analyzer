"""Reconstruct and classify standard fit results against current inputs."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from logic.pcs_fit_fingerprint import (
    fingerprint_fit_input_snapshot,
    validate_fit_input_fingerprint,
)
from logic.pcs_tensor_state import (
    ApplicationPCSTensor,
    PCS_TENSOR_STATE_KEY,
    canonical_tensor_from_legacy,
)
from logic.pcs_fit_weighting import parse_fit_weighting_mode


FIT_RESULT_CURRENT = "current"
FIT_RESULT_STALE = "stale"
FIT_RESULT_UNKNOWN = "unknown"
STALE_FIT_RESULT_MESSAGE = "Fit inputs changed since this result was calculated."
UNKNOWN_FIT_RESULT_MESSAGE = (
    "Result validity cannot be verified against current fit inputs."
)


def _status(status, reason, *, stored_digest=None, current_digest=None):
    return {
        "status": status,
        "reason": reason,
        "stored_digest": stored_digest,
        "current_digest": current_digest,
    }


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"saved fit configuration {name} is invalid")
    return value


def _ids(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"saved fit configuration {name} is invalid")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"saved fit configuration {name} is invalid") from exc
    if any(float(item) != result[index] for index, item in enumerate(value)):
        raise ValueError(f"saved fit configuration {name} is invalid")
    return result


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"saved fit configuration {name} is invalid")
    try:
        result = tuple(str(item) for item in value)
    except TypeError as exc:
        raise ValueError(f"saved fit configuration {name} is invalid") from exc
    if any(not item for item in result):
        raise ValueError(f"saved fit configuration {name} is invalid")
    return result


def _reconstruction_arguments(result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = result.get("fit_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("fit metadata is missing")
    config = metadata.get("input_configuration")
    if not isinstance(config, Mapping):
        raise ValueError("saved fit input configuration is missing")
    mode = str(result.get("mode", ""))
    if config.get("mode") != mode:
        raise ValueError("saved fit mode configuration is inconsistent")

    observation_model = config.get("observation_model", "legacy_single_atom")
    if observation_model == "structured_v1":
        observation_ids = _strings(config.get("observation_ids", ()), "observation_ids")
        try:
            result_ids = tuple(
                str(row["observable_id"])
                for row in result.get("per_observable", ())
            )
        except (TypeError, KeyError) as exc:
            raise ValueError("saved structured fit observations are invalid") from exc
        if observation_ids != result_ids:
            raise ValueError("saved fit observation configuration is inconsistent")
        proton_ids = ()
    elif observation_model == "legacy_single_atom":
        observation_ids = None
        proton_ids = _ids(config.get("proton_ids"), "proton_ids")
        try:
            result_ids = tuple(int(row[0]) for row in result.get("per_point", ()))
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("saved fit observations are invalid") from exc
        if proton_ids != result_ids:
            raise ValueError("saved fit observation configuration is inconsistent")
    else:
        raise ValueError("saved fit observation model is unsupported")

    donor_ids = _ids(config.get("donor_ids", ()), "donor_ids")
    axis_mode = config.get("axis_mode")
    if mode == "theta_alpha_multi":
        if donor_ids != _ids(result.get("donor_ids", ()), "result donor_ids"):
            raise ValueError("saved donor configuration is inconsistent")
        if axis_mode != result.get("axis_mode"):
            raise ValueError("saved donor-axis configuration is inconsistent")
    elif mode not in {"euler_global", "full_tensor"}:
        raise ValueError("saved fit mode is unsupported")

    free = tuple(str(name) for name in metadata.get("free_parameters", ()))
    fit_ax = _bool(config.get("fit_delta_chi_ax"), "fit_delta_chi_ax")
    fit_rh = _bool(config.get("fit_delta_chi_rh"), "fit_delta_chi_rh")
    if fit_ax != ("delta_chi_ax" in free):
        raise ValueError("saved axial fitting configuration is inconsistent")
    if fit_rh != ("delta_chi_rh" in free):
        raise ValueError("saved rhombic fitting configuration is inconsistent")
    use_global = _bool(config.get("use_global_search"), "use_global_search")
    weighting_mode = parse_fit_weighting_mode(
        config.get("weighting_mode", "unweighted")
    ).value
    metadata_weighting = metadata.get("weighting_mode", "unweighted")
    if parse_fit_weighting_mode(metadata_weighting).value != weighting_mode:
        raise ValueError("saved fit weighting configuration is inconsistent")
    expected_optimizer = (
        "differential_evolution_then_least_squares"
        if use_global else "least_squares"
    )
    if metadata.get("optimizer") != expected_optimizer:
        raise ValueError("saved optimizer configuration is inconsistent")

    return {
        "mode": mode,
        "proton_ids": proton_ids,
        "observation_ids": observation_ids,
        "donor_ids": donor_ids,
        "axis_mode": axis_mode,
        "fit_delta_chi": fit_ax,
        "fit_delta_chi_rh": fit_rh,
        "use_global_search": use_global,
        "use_current_tensor": _bool(
            config.get("use_current_tensor"), "use_current_tensor"
        ),
        "weighting_mode": weighting_mode,
    }


def build_current_fit_snapshot_for_result(
    result: Mapping[str, Any],
    current_state: Mapping[str, Any],
):
    """Recapture saved fit dependencies without executing an optimizer."""
    if not isinstance(current_state, Mapping):
        raise ValueError("current application state is unavailable")
    arguments = _reconstruction_arguments(result)
    capture_state: MutableMapping[str, Any] = dict(current_state)
    current_weighting = current_state.get("fit_weighting_var")
    if current_weighting is not None:
        try:
            current_weighting = current_weighting.get()
        except AttributeError:
            pass
        arguments["weighting_mode"] = parse_fit_weighting_mode(
            current_weighting
        ).value

    if not (capture_state.get("atom_data_eff") or capture_state.get("atom_data")):
        raise ValueError("current fit geometry is unavailable")
    if not (
        capture_state.get("atom_ids_eff")
        or capture_state.get("atom_ids")
        or capture_state.get("atom_ids_raw")
    ):
        raise ValueError("current fit reference IDs are unavailable")
    if arguments["observation_ids"] is None:
        if not isinstance(capture_state.get("delta_exp_values"), Mapping):
            raise ValueError("current experimental PCS values are unavailable")
    elif not isinstance(capture_state.get("experimental_pcs_observations"), (list, tuple)):
        raise ValueError("current structured PCS observations are unavailable")
    if any(key not in capture_state for key in ("x0", "y0", "z0")):
        raise ValueError("current metal position is unavailable")
    if capture_state.get("tensor_entry") is None and "tensor" not in capture_state:
        raise ValueError("current axial tensor value is unavailable")

    if arguments["mode"] == "theta_alpha_multi":
        if (
            capture_state.get("angle_z_var") is None
            and "angle_z" not in capture_state
        ):
            raise ValueError("current Mode A z-angle is unavailable")
        if capture_state.get("angle_z_var") is None:
            capture_state["angle_z_var"] = capture_state["angle_z"]

    if arguments["use_current_tensor"] and not isinstance(
        capture_state.get(PCS_TENSOR_STATE_KEY), ApplicationPCSTensor
    ):
        # Build an ephemeral compatibility tensor in the copied mapping. This
        # never installs or mutates canonical state in the application.
        canonical_tensor_from_legacy(capture_state)

    from logic.fitting import build_standard_fit_input_snapshot

    return build_standard_fit_input_snapshot(capture_state, **arguments)


def evaluate_fit_result_status(
    result: Mapping[str, Any] | None,
    current_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return current/stale/unknown without changing application state."""
    if not isinstance(result, Mapping):
        return _status(FIT_RESULT_UNKNOWN, UNKNOWN_FIT_RESULT_MESSAGE)
    metadata = result.get("fit_metadata")
    fingerprint = metadata.get("input_fingerprint") if isinstance(metadata, Mapping) else None
    valid, validation_reason = validate_fit_input_fingerprint(fingerprint)
    if not valid:
        return _status(
            FIT_RESULT_UNKNOWN,
            f"{UNKNOWN_FIT_RESULT_MESSAGE} {validation_reason}",
            stored_digest=(fingerprint.get("digest") if isinstance(fingerprint, Mapping) else None),
        )
    stored_digest = fingerprint["digest"]
    try:
        snapshot = build_current_fit_snapshot_for_result(result, current_state)
        current = fingerprint_fit_input_snapshot(snapshot)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return _status(
            FIT_RESULT_UNKNOWN,
            f"{UNKNOWN_FIT_RESULT_MESSAGE} {exc}",
            stored_digest=stored_digest,
        )
    current_digest = current["digest"]
    if current_digest == stored_digest:
        return _status(
            FIT_RESULT_CURRENT,
            "Current scientific fit inputs match this result.",
            stored_digest=stored_digest,
            current_digest=current_digest,
        )
    return _status(
        FIT_RESULT_STALE,
        STALE_FIT_RESULT_MESSAGE,
        stored_digest=stored_digest,
        current_digest=current_digest,
    )


__all__ = [
    "FIT_RESULT_CURRENT",
    "FIT_RESULT_STALE",
    "FIT_RESULT_UNKNOWN",
    "STALE_FIT_RESULT_MESSAGE",
    "UNKNOWN_FIT_RESULT_MESSAGE",
    "build_current_fit_snapshot_for_result",
    "evaluate_fit_result_status",
]
