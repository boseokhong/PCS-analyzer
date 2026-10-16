"""Versioned JSON serialization for long-lived processing recipes.

Future schema migrations belong in this module, before construction of domain
``ProcessingStep`` objects.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from nmr_workbench.core import ProcessingHistory, ProcessingStep

PROCESSING_RECIPE_SCHEMA_VERSION = 1

_METHODS: dict[str, set[str | None]] = {
    "digital_filter": {"bruker_nmrglue"},
    "apodization": {"exponential", "gaussian"},
    "zero_fill": {None},
    "fourier_transform": {"numpy_fft", "numpy_fft_positive"},
    "phase": {"linear"},
    "baseline": {"bernstein", "whittaker_asls", "arpls"},
    "reference": {"axis_translation"},
}

_ALLOWED_PARAMETERS: dict[tuple[str, str | None], set[str]] = {
    ("digital_filter", "bruker_nmrglue"): {
        "truncate",
        "decimation",
        "firmware_version",
        "group_delay",
    },
    ("apodization", "exponential"): {"lb_hz"},
    ("apodization", "gaussian"): {"gb_hz"},
    ("zero_fill", None): {"target_points"},
    ("fourier_transform", "numpy_fft"): {
        "direction",
        "exponent_sign",
        "shift",
        "ordering",
    },
    ("fourier_transform", "numpy_fft_positive"): {
        "direction",
        "exponent_sign",
        "shift",
        "ordering",
    },
    ("phase", "linear"): {
        "p0_deg",
        "p1_deg",
        "pivot_kind",
        "pivot_ppm",
        "pivot_index",
    },
    ("baseline", "bernstein"): {"order", "regions"},
    ("baseline", "whittaker_asls"): {"lam", "p", "diff_order", "regions"},
    ("baseline", "arpls"): {"lam", "diff_order", "regions"},
    ("reference", "axis_translation"): {
        "observed_ppm",
        "target_ppm",
        "delta_ppm",
        "offset_ppm",
        "source_type",
        "preset_id",
        "resonance_id",
        "source_id",
    },
}


class ProcessingRecipeError(ValueError):
    """Raised when a processing recipe is malformed or unsupported."""


def serialize_processing_history(history: ProcessingHistory) -> dict[str, Any]:
    """Return the canonical schema-v1 dictionary for ``history``."""

    if not isinstance(history, ProcessingHistory):
        raise TypeError("history must be a ProcessingHistory")
    steps = []
    for step in history.steps:
        normalized_parameters = _to_json_value(
            dict(step.parameters), path="parameters"
        )
        _validate_step(step.operation, step.method, normalized_parameters)
        steps.append(
            {
                "operation": step.operation,
                "method": step.method,
                "parameters": normalized_parameters,
            }
        )
    return {"schema_version": PROCESSING_RECIPE_SCHEMA_VERSION, "steps": steps}


def deserialize_processing_history(payload: Mapping[str, Any]) -> ProcessingHistory:
    """Validate a schema-v1 mapping and reconstruct its ordered history."""

    if not isinstance(payload, Mapping):
        raise ProcessingRecipeError("processing recipe must be a JSON object")
    version = payload.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, (int, np.integer)):
        raise ProcessingRecipeError("schema_version must be an integer")
    if int(version) != PROCESSING_RECIPE_SCHEMA_VERSION:
        raise ProcessingRecipeError(
            f"unsupported processing recipe schema_version: {version}"
        )
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        raise ProcessingRecipeError("steps must be a JSON array")

    history = ProcessingHistory()
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            raise ProcessingRecipeError(f"steps[{index}] must be a JSON object")
        operation = raw_step.get("operation")
        method = raw_step.get("method")
        parameters = raw_step.get("parameters", {})
        if not isinstance(operation, str) or not operation.strip():
            raise ProcessingRecipeError(f"steps[{index}].operation must be a string")
        if method is not None and not isinstance(method, str):
            raise ProcessingRecipeError(f"steps[{index}].method must be a string or null")
        if not isinstance(parameters, Mapping):
            raise ProcessingRecipeError(f"steps[{index}].parameters must be an object")
        normalized_parameters = _to_json_value(
            dict(parameters), path=f"steps[{index}].parameters"
        )
        _validate_step(operation, method, normalized_parameters, index=index)
        history.append(ProcessingStep(operation, method, normalized_parameters))
    return history


def processing_history_to_json(
    history: ProcessingHistory, *, indent: int | None = 2
) -> str:
    """Serialize a processing history to deterministic, standards-compliant JSON."""

    return json.dumps(
        serialize_processing_history(history),
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )


def processing_history_from_json(text: str) -> ProcessingHistory:
    """Deserialize a JSON processing recipe."""

    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProcessingRecipeError(f"invalid processing recipe JSON: {exc}") from exc
    return deserialize_processing_history(payload)


def _validate_step(
    operation: str,
    method: str | None,
    parameters: Mapping[str, Any],
    *,
    index: int | None = None,
) -> None:
    location = "step" if index is None else f"steps[{index}]"
    if operation not in _METHODS:
        raise ProcessingRecipeError(f"{location}: unsupported operation '{operation}'")
    if method not in _METHODS[operation]:
        raise ProcessingRecipeError(
            f"{location}: unsupported method {method!r} for operation '{operation}'"
        )
    required: set[str]
    if operation == "apodization":
        required = {"lb_hz"} if method == "exponential" else {"gb_hz"}
    elif operation == "zero_fill":
        required = {"target_points"}
    elif operation == "phase":
        required = {"p0_deg", "p1_deg", "pivot_kind"}
        pivot_kind = parameters.get("pivot_kind")
        if pivot_kind == "ppm":
            required.add("pivot_ppm")
        elif pivot_kind == "index":
            required.add("pivot_index")
        else:
            raise ProcessingRecipeError(f"{location}: invalid phase pivot_kind")
    elif operation == "baseline":
        if method == "bernstein":
            required = {"order"}
        elif method == "arpls":
            required = {"lam", "diff_order"}
        else:
            required = {"lam", "p", "diff_order"}
    elif operation == "reference":
        required = {"observed_ppm", "target_ppm"}
    else:
        required = set()
    missing = sorted(required.difference(parameters))
    if missing:
        raise ProcessingRecipeError(
            f"{location}: missing required parameters: {', '.join(missing)}"
        )
    unknown = sorted(set(parameters).difference(_ALLOWED_PARAMETERS[(operation, method)]))
    if unknown:
        raise ProcessingRecipeError(
            f"{location}: unsupported parameters: {', '.join(unknown)}"
        )
    numeric_keys = {
        "lb_hz",
        "gb_hz",
        "group_delay",
        "p0_deg",
        "p1_deg",
        "pivot_ppm",
        "pivot_index",
        "lam",
        "p",
        "observed_ppm",
        "target_ppm",
        "delta_ppm",
        "offset_ppm",
    }
    for key in numeric_keys.intersection(parameters):
        value = parameters[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProcessingRecipeError(f"{location}: parameter '{key}' must be numeric")
    string_keys = {
        "source_type",
        "preset_id",
        "resonance_id",
        "source_id",
    }
    for key in string_keys.intersection(parameters):
        value = parameters[key]
        if not isinstance(value, str) or not value.strip():
            raise ProcessingRecipeError(
                f"{location}: parameter '{key}' must be a non-empty string"
            )
    integer_keys = {"target_points", "decimation", "firmware_version", "order", "diff_order"}
    for key in integer_keys.intersection(parameters):
        value = parameters[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProcessingRecipeError(f"{location}: parameter '{key}' must be an integer")


def _to_json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _to_json_value(value.item(), path=path)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ProcessingRecipeError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, np.ndarray):
        return _to_json_value(value.tolist(), path=path)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProcessingRecipeError(f"{path} contains a non-string object key")
            output[key] = _to_json_value(item, path=f"{path}.{key}")
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _to_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ProcessingRecipeError(
        f"{path} contains unsupported value type: {type(value).__name__}"
    )
