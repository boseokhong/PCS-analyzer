"""Additive provenance schema for standard PCS fitting results."""

from __future__ import annotations

from math import isclose, isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from logic.pcs_fit_weighting import FitWeightingMode, parse_fit_weighting_mode


PCS_FIT_RESULT_SCHEMA = "pcs_fit_result_v1"
PARAMETER_STATUSES = frozenset({"fitted", "fixed", "derived", "inherited"})
FITTER_ORIENTATION_CONVENTION = "active_extrinsic_XYZ"
FITTER_ORIENTATION_UNIT = "deg"
QUALITY_METRIC_FIELDS = ("rmsd", "r2", "q_factor", "chi2_n")
WEIGHTED_QUALITY_METRIC_FIELDS = ("chi_square", "chi_square_per_datum")


def _parameter(status: str, value: Any, **metadata: Any) -> dict[str, Any]:
    if status not in PARAMETER_STATUSES:
        raise ValueError(f"unsupported parameter status: {status}")
    return {"status": status, "value": value, **metadata}


def annotate_standard_fit_result(
    result: Mapping[str, Any],
    *,
    mode: str,
    use_global_search: bool,
    free_parameters: Sequence[str],
    optimizer: str | None = None,
    input_fingerprint: Mapping[str, Any] | None = None,
    input_configuration: Mapping[str, Any] | None = None,
    weighting_mode: str = "unweighted",
) -> dict[str, Any]:
    """Return a legacy-compatible result with explicit parameter provenance."""
    annotated = dict(result)
    normalized_weighting = parse_fit_weighting_mode(weighting_mode)
    free = tuple(str(name) for name in free_parameters)
    fitted = set(free)
    optimizer_name = optimizer or (
        "differential_evolution_then_least_squares"
        if use_global_search else "least_squares"
    )
    provenance: dict[str, dict[str, Any]] = {
        "delta_chi_ax": _parameter(
            "fitted" if "delta_chi_ax" in fitted else "fixed",
            float(annotated["delta_chi_ax"]),
            unit="1e-32_m3",
        ),
        "delta_chi_rh": _parameter(
            "fitted" if "delta_chi_rh" in fitted else "fixed",
            float(annotated["delta_chi_rh"]),
            unit="1e-32_m3",
        ),
    }

    if mode == "theta_alpha_multi":
        theta = float(annotated["theta"])
        alpha = float(annotated["alpha"])
        provenance.update(
            {
                "theta": _parameter("fitted", theta, unit="deg"),
                "alpha": _parameter("fitted", alpha, unit="deg"),
                "orientation": _parameter(
                    "fitted",
                    [theta, alpha],
                    parameterization="donor_axis_theta_alpha",
                    unit="deg",
                ),
                "donor_axis": _parameter(
                    "fixed",
                    {
                        "donor_ids": list(annotated.get("donor_ids", ())),
                        "axis_mode": annotated.get("axis_mode", "bisector"),
                    },
                    parameterization="ligand_donor_axis",
                ),
            }
        )
        mode_label = "A"
    elif mode == "euler_global":
        orientation = [
            float(annotated["ax"]),
            float(annotated["ay"]),
            float(annotated["az"]),
        ]
        provenance["orientation"] = _parameter(
            "fitted",
            orientation,
            parameterization=FITTER_ORIENTATION_CONVENTION,
            transformation="active",
            euler_type="extrinsic",
            axis_order="XYZ",
            unit=FITTER_ORIENTATION_UNIT,
        )
        mode_label = "B"
    elif mode == "full_tensor":
        orientation = [float(value) for value in annotated["euler_deg"]]
        metal_pos = [float(value) for value in annotated["metal_pos"]]
        provenance["orientation"] = _parameter(
            "fitted",
            orientation,
            parameterization=FITTER_ORIENTATION_CONVENTION,
            transformation="active",
            euler_type="extrinsic",
            axis_order="XYZ",
            unit=FITTER_ORIENTATION_UNIT,
        )
        provenance["metal_pos"] = _parameter(
            "fitted",
            metal_pos,
            unit="angstrom",
        )
        mode_label = "C"
    else:
        raise ValueError(f"unsupported standard fit mode: {mode}")

    annotated["result_schema"] = PCS_FIT_RESULT_SCHEMA
    annotated["fit_metadata"] = {
        "mode": mode,
        "mode_label": mode_label,
        "optimizer": optimizer_name,
        "free_parameters": list(free),
        "weighting_mode": normalized_weighting.value,
        "weighted_objective": (
            normalized_weighting is FitWeightingMode.EXPERIMENTAL_UNCERTAINTY
        ),
    }
    if input_fingerprint is not None:
        annotated["fit_metadata"]["input_fingerprint"] = dict(input_fingerprint)
    if input_configuration is not None:
        annotated["fit_metadata"]["input_configuration"] = dict(input_configuration)
        observation_model = input_configuration.get("observation_model")
        if observation_model:
            annotated["fit_metadata"]["observation_model"] = str(observation_model)
    annotated["parameter_provenance"] = provenance
    return annotated


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
            return False
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right)
        )
    try:
        return isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12)
    except (TypeError, ValueError):
        return left == right


def validate_fit_result_provenance(result: Mapping[str, Any]) -> None:
    """Validate v1 provenance against legacy flat values.

    Legacy payloads without ``result_schema`` remain valid and are deliberately
    ignored here. Structured payloads fail closed when metadata disagrees with
    their compatibility fields.
    """
    schema = result.get("result_schema")
    if schema is None:
        return
    if schema != PCS_FIT_RESULT_SCHEMA:
        raise ValueError(f"unsupported fit result schema: {schema}")
    metadata = result.get("fit_metadata")
    provenance = result.get("parameter_provenance")
    if not isinstance(metadata, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("fit result provenance metadata is incomplete")
    mode = str(result.get("mode", ""))
    if metadata.get("mode") != mode:
        raise ValueError("fit result mode provenance does not match legacy mode")
    free_parameters = metadata.get("free_parameters")
    if (
        not isinstance(free_parameters, Sequence)
        or isinstance(free_parameters, (str, bytes))
    ):
        raise ValueError("fit result free-parameter provenance is incomplete")
    free = {str(name) for name in free_parameters}

    expected_values: dict[str, Any] = {
        "delta_chi_ax": result.get("delta_chi_ax"),
        "delta_chi_rh": result.get("delta_chi_rh"),
    }
    if mode == "euler_global":
        expected_values["orientation"] = [
            result.get("ax"), result.get("ay"), result.get("az")
        ]
    elif mode == "full_tensor":
        expected_values["orientation"] = result.get("euler_deg")
        expected_values["metal_pos"] = result.get("metal_pos")
    elif mode == "theta_alpha_multi":
        expected_values.update(
            {
                "theta": result.get("theta"),
                "alpha": result.get("alpha"),
                "orientation": [result.get("theta"), result.get("alpha")],
                "donor_axis": {
                    "donor_ids": list(result.get("donor_ids", ())),
                    "axis_mode": result.get("axis_mode"),
                },
            }
        )

    for name, expected in expected_values.items():
        record = provenance.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"fit result provenance is missing {name}")
        if record.get("status") not in PARAMETER_STATUSES:
            raise ValueError(f"fit result provenance has invalid {name} status")
        if not _same_value(record.get("value"), expected):
            raise ValueError(f"fit result provenance value for {name} is inconsistent")

    expected_statuses = {
        "delta_chi_ax": "fitted" if "delta_chi_ax" in free else "fixed",
        "delta_chi_rh": "fitted" if "delta_chi_rh" in free else "fixed",
    }
    if mode == "theta_alpha_multi":
        expected_statuses.update(
            {
                "theta": "fitted",
                "alpha": "fitted",
                "donor_axis": "fixed",
            }
        )
        required_groups = {
            "orientation": {"theta", "alpha"},
        }
    elif mode == "euler_global":
        required_groups = {
            "orientation": {"ax", "ay", "az"},
        }
    elif mode == "full_tensor":
        required_groups = {
            "orientation": {"ax", "ay", "az"},
            "metal_pos": {"metal_x", "metal_y", "metal_z"},
        }
    else:
        raise ValueError(f"unsupported standard fit mode: {mode}")
    for name, components in required_groups.items():
        expected_statuses[name] = (
            "fitted" if components.issubset(free) else "fixed"
        )
    for name, expected_status in expected_statuses.items():
        record = provenance.get(name)
        if not isinstance(record, Mapping) or record.get("status") != expected_status:
            raise ValueError(f"fit result provenance status for {name} is inconsistent")

    if mode in {"euler_global", "full_tensor"}:
        orientation = provenance["orientation"]
        expected_orientation_metadata = {
            "parameterization": FITTER_ORIENTATION_CONVENTION,
            "transformation": "active",
            "euler_type": "extrinsic",
            "axis_order": "XYZ",
            "unit": FITTER_ORIENTATION_UNIT,
        }
        if any(
            orientation.get(key) != value
            for key, value in expected_orientation_metadata.items()
        ):
            raise ValueError("fit result orientation provenance is inconsistent")


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fit result {name} is missing or invalid") from exc
    if not isfinite(number):
        raise ValueError(f"fit result {name} must be finite")
    return number


def _numeric_vector(value: Any, name: str, length: int) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"fit result {name} must contain {length} values")
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(
            f"fit result {name} must contain {length} values"
        ) from exc
    if len(items) != length:
        raise ValueError(f"fit result {name} must contain {length} values")
    return [_finite_number(item, f"{name}[{index}]") for index, item in enumerate(items)]


def validate_standard_fit_result(result: Mapping[str, Any]) -> None:
    """Validate a complete standard result at its persistence boundary."""
    if not isinstance(result, Mapping):
        raise ValueError("fit result must be a mapping")
    if result.get("result_schema") != PCS_FIT_RESULT_SCHEMA:
        raise ValueError(
            f"unsupported fit result schema: {result.get('result_schema')}"
        )
    validate_fit_result_provenance(result)

    mode = str(result.get("mode", ""))
    metadata = result["fit_metadata"]
    expected_mode_labels = {
        "theta_alpha_multi": "A",
        "euler_global": "B",
        "full_tensor": "C",
    }
    if metadata.get("mode_label") != expected_mode_labels.get(mode):
        raise ValueError("fit result mode label is inconsistent")
    if not isinstance(metadata.get("optimizer"), str) or not metadata.get("optimizer"):
        raise ValueError("fit result optimizer metadata is missing or invalid")
    weighting_mode = parse_fit_weighting_mode(
        metadata.get("weighting_mode", FitWeightingMode.UNWEIGHTED.value)
    )
    weighted_objective = metadata.get(
        "weighted_objective",
        weighting_mode is FitWeightingMode.EXPERIMENTAL_UNCERTAINTY,
    )
    if bool(weighted_objective) != (
        weighting_mode is FitWeightingMode.EXPERIMENTAL_UNCERTAINTY
    ):
        raise ValueError("fit result weighting metadata is inconsistent")
    free = {str(name) for name in metadata["free_parameters"]}
    _finite_number(result.get("delta_chi_ax"), "delta_chi_ax")
    _finite_number(result.get("delta_chi_rh"), "delta_chi_rh")
    if mode == "theta_alpha_multi":
        required_free = {"theta", "alpha"}
        _finite_number(result.get("theta"), "theta")
        _finite_number(result.get("alpha"), "alpha")
        donor_ids = result.get("donor_ids")
        if (
            not isinstance(donor_ids, Sequence)
            or isinstance(donor_ids, (str, bytes))
        ):
            raise ValueError("fit result donor_ids must be a sequence")
        try:
            donor_numbers = [float(ref_id) for ref_id in donor_ids]
        except (TypeError, ValueError) as exc:
            raise ValueError("fit result donor_ids are invalid") from exc
        if not donor_numbers or any(
            not isfinite(ref_id) or int(ref_id) != ref_id
            for ref_id in donor_numbers
        ):
            raise ValueError("fit result donor_ids are invalid")
        if not isinstance(result.get("axis_mode"), str) or not result.get("axis_mode"):
            raise ValueError("fit result axis_mode is missing or invalid")
    elif mode == "euler_global":
        required_free = {"ax", "ay", "az"}
        for key in ("ax", "ay", "az"):
            _finite_number(result.get(key), key)
    elif mode == "full_tensor":
        required_free = {"ax", "ay", "az", "metal_x", "metal_y", "metal_z"}
        _numeric_vector(result.get("euler_deg"), "euler_deg", 3)
        _numeric_vector(result.get("metal_pos"), "metal_pos", 3)
    else:
        raise ValueError(f"unsupported standard fit mode: {mode}")
    if not required_free.issubset(free):
        raise ValueError("fit result free parameters do not match standard mode")

    per_point = result.get("per_point")
    if not isinstance(per_point, Sequence) or isinstance(per_point, (str, bytes)):
        raise ValueError("fit result per_point must be a sequence")
    for index, row in enumerate(per_point):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError(f"fit result per_point[{index}] must be a record")
        if len(row) != 4:
            raise ValueError(f"fit result per_point[{index}] must contain four values")
        try:
            ref_id = float(row[0])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"fit result per_point[{index}] has an invalid reference id"
            ) from exc
        if not isfinite(ref_id) or int(ref_id) != ref_id:
            raise ValueError(
                f"fit result per_point[{index}] has an invalid reference id"
            )
        for column, label in enumerate(("observed", "predicted", "residual"), start=1):
            _finite_number(row[column], f"per_point[{index}].{label}")

    count_number = _finite_number(result.get("n"), "n")
    count = int(count_number)
    if count != count_number or count < 0:
        raise ValueError("fit result n is missing or invalid")
    if count != len(per_point):
        raise ValueError("fit result n does not match per_point length")
    per_observable = result.get("per_observable")
    if per_observable is not None:
        if not isinstance(per_observable, Sequence) or isinstance(
            per_observable, (str, bytes)
        ):
            raise ValueError("fit result per_observable must be a sequence")
        if len(per_observable) != count:
            raise ValueError("fit result per_observable length does not match n")
        seen_ids = set()
        for index, row in enumerate(per_observable):
            if not isinstance(row, Mapping):
                raise ValueError(f"fit result per_observable[{index}] must be a record")
            observable_id = row.get("observable_id")
            if not isinstance(observable_id, str) or not observable_id:
                raise ValueError(f"fit result per_observable[{index}] has no observable id")
            if observable_id in seen_ids:
                raise ValueError("fit result per_observable ids must be unique")
            seen_ids.add(observable_id)
            for key in ("observed", "predicted", "residual"):
                _finite_number(row.get(key), f"per_observable[{index}].{key}")
            uncertainty = row.get("uncertainty_ppm")
            if uncertainty is not None:
                uncertainty = _finite_number(
                    uncertainty,
                    f"per_observable[{index}].uncertainty_ppm",
                )
            weighted_residual = row.get("weighted_residual")
            if weighting_mode is FitWeightingMode.EXPERIMENTAL_UNCERTAINTY:
                if uncertainty is None or uncertainty <= 0.0:
                    raise ValueError(
                        f"fit result per_observable[{index}] requires a finite "
                        "positive uncertainty_ppm for weighted fitting"
                    )
                weighted_value = _finite_number(
                    weighted_residual,
                    f"per_observable[{index}].weighted_residual",
                )
                expected_weighted = float(row["residual"]) / uncertainty
                if not isclose(
                    weighted_value,
                    expected_weighted,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    raise ValueError(
                        f"fit result per_observable[{index}] weighted residual "
                        "is inconsistent"
                    )
            elif weighted_residual is not None:
                raise ValueError(
                    f"fit result per_observable[{index}] has a weighted residual "
                    "for an unweighted fit"
                )
    for key in QUALITY_METRIC_FIELDS:
        if key not in result:
            raise ValueError(f"fit result {key} is missing")
        value = result[key]
        if value is not None:
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"fit result {key} is invalid") from exc
    for key in WEIGHTED_QUALITY_METRIC_FIELDS:
        if weighting_mode is FitWeightingMode.EXPERIMENTAL_UNCERTAINTY:
            value = _finite_number(result.get(key), key)
            if value < 0.0:
                raise ValueError(f"fit result {key} must be non-negative")
        elif key in result:
            raise ValueError(f"unweighted fit result must not contain {key}")


def _json_safe(value: Any, *, path: str = "result") -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), path=path)
    if isinstance(value, np.generic):
        return _json_safe(value.item(), path=path)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            normalized[key] = _json_safe(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if isfinite(value):
            return value
        top_level = path.removeprefix("result.").split(".", 1)[0]
        if top_level in QUALITY_METRIC_FIELDS:
            return None
        raise ValueError(f"{path} must be finite")
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


def serialize_fit_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy without mutating the live fit result."""
    normalized = _json_safe(result)
    validate_standard_fit_result(normalized)
    return normalized


def deserialize_fit_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and validate one persisted standard fit result."""
    if not isinstance(payload, Mapping):
        raise ValueError("persisted fit result must be a JSON object")
    normalized = _json_safe(payload)
    validate_standard_fit_result(normalized)
    return normalized


def parameter_status(result: Mapping[str, Any], name: str) -> str | None:
    """Read one status without requiring provenance on legacy payloads."""
    provenance = result.get("parameter_provenance")
    if not isinstance(provenance, Mapping):
        return None
    record = provenance.get(name)
    if not isinstance(record, Mapping):
        return None
    status = record.get("status")
    return str(status) if status in PARAMETER_STATUSES else None


__all__ = [
    "FITTER_ORIENTATION_CONVENTION",
    "FITTER_ORIENTATION_UNIT",
    "PARAMETER_STATUSES",
    "PCS_FIT_RESULT_SCHEMA",
    "QUALITY_METRIC_FIELDS",
    "WEIGHTED_QUALITY_METRIC_FIELDS",
    "annotate_standard_fit_result",
    "deserialize_fit_result",
    "parameter_status",
    "serialize_fit_result",
    "validate_standard_fit_result",
    "validate_fit_result_provenance",
]
