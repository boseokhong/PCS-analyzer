"""Stable, query-only public facade for PCS forward prediction.

This module is the UI-independent entry point for in-process consumers and
plugins.  It accepts either L0A DTOs or JSON-compatible primitive payloads and
delegates all scientific calculation to the L0B pure forward engine.

State-changing commands are deliberately not part of this facade.  Future
commands must live behind a separate interface rather than acquiring implicit
access to Tk objects or the application's mutable ``state`` dictionary.
"""

from __future__ import annotations

from enum import Enum
from math import isfinite
from typing import Any, Mapping, Sequence

from logic.pcs_forward_api import (
    AngleUnit,
    CoordinateUnit,
    EulerConvention,
    PCSAtom,
    PCSAtomPrediction,
    PCS_FORWARD_API_VERSION,
    PCS_FORWARD_SCHEMA_VERSION,
    PCSForwardModel,
    PCSFrame,
    PCSPredictionRequest,
    PCSPredictionResult,
    PCSStructureTarget,
    PCSTargetPrediction,
    PCSTensor,
    SusceptibilityUnit,
    TargetReduction,
)
from logic.pcs_forward_model import (
    PCS_FORWARD_ENGINE_ID,
    PCS_FORWARD_ENGINE_VERSION,
    PCSForwardEngine,
)


PCS_API_NAME = "pcs-forward"


class PCSPayloadError(ValueError):
    """Raised when a primitive API payload is malformed or unsupported."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PCSPayloadError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PCSPayloadError(f"{name} must be an array")
    return value


def _json_value(value: Any, name: str = "metadata") -> Any:
    """Copy a value into strict JSON-compatible primitive containers."""

    if isinstance(value, Enum):
        return _json_value(value.value, name)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise PCSPayloadError(f"{name} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PCSPayloadError(f"{name} object keys must be strings")
            copied[key] = _json_value(item, f"{name}.{key}")
        return copied
    if isinstance(value, (list, tuple)):
        return [_json_value(item, name) for item in value]
    raise PCSPayloadError(
        f"{name} contains unsupported value type {type(value).__name__}"
    )


def _metadata_from(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    return dict(_json_value(_mapping(metadata, "metadata"), "metadata"))


def _optional_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return None if value is None else float(value)


def serialize_request(request: PCSPredictionRequest) -> dict[str, Any]:
    """Return a detached JSON-compatible representation of a request DTO."""

    if not isinstance(request, PCSPredictionRequest):
        raise TypeError("request must be a PCSPredictionRequest")
    tensor = request.tensor
    frame = tensor.frame
    return {
        "api_version": PCS_FORWARD_API_VERSION,
        "schema_version": request.schema_version,
        "request_id": request.request_id,
        "tensor": {
            "dchi_ax": tensor.dchi_ax,
            "dchi_rh": tensor.dchi_rh,
            "rhombic_enabled": tensor.rhombic_enabled,
            "susceptibility_unit": tensor.susceptibility_unit.value,
            "frame": {
                "metal_position": list(frame.metal_position),
                "euler_angles": list(frame.euler_angles),
                "convention": frame.convention.value,
                "angle_unit": frame.angle_unit.value,
                "coordinate_unit": frame.coordinate_unit.value,
            },
        },
        "atoms": [
            {
                "ref_id": atom.ref_id,
                "position": list(atom.position),
                "label": atom.label,
                "coordinate_unit": atom.coordinate_unit.value,
                "metadata": _json_value(atom.metadata, "atom.metadata"),
            }
            for atom in request.atoms
        ],
        "targets": [
            {
                "target_id": target.target_id,
                "site_member_ref_ids": [
                    list(group) for group in target.site_member_ref_ids
                ],
                "member_reduction": target.member_reduction.value,
                "site_reduction": target.site_reduction.value,
                "assignment": target.assignment,
                "experimental_pcs_ppm": target.experimental_pcs_ppm,
                "uncertainty_ppm": target.uncertainty_ppm,
                "metadata": _json_value(target.metadata, "target.metadata"),
            }
            for target in request.targets
        ],
        "metadata": _json_value(request.metadata, "request.metadata"),
    }


def deserialize_request(payload: Mapping[str, Any]) -> PCSPredictionRequest:
    """Build and validate an L0A request from a primitive payload."""

    try:
        return _deserialize_request(payload)
    except PCSPayloadError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise PCSPayloadError(str(exc)) from exc


def _deserialize_request(payload: Mapping[str, Any]) -> PCSPredictionRequest:
    """Internal request decoder; public wrapper normalizes payload errors."""

    root = _mapping(payload, "request")
    api_version = root.get("api_version", PCS_FORWARD_API_VERSION)
    if str(api_version) != PCS_FORWARD_API_VERSION:
        raise PCSPayloadError(f"unsupported PCS forward api_version: {api_version!r}")

    tensor_payload = _mapping(root.get("tensor"), "tensor")
    frame_payload = _mapping(tensor_payload.get("frame"), "tensor.frame")
    frame = PCSFrame(
        metal_position=tuple(
            _sequence(frame_payload.get("metal_position"), "metal_position")
        ),
        euler_angles=tuple(
            _sequence(frame_payload.get("euler_angles"), "euler_angles")
        ),
        convention=EulerConvention(frame_payload.get("convention")),
        angle_unit=AngleUnit(frame_payload.get("angle_unit")),
        coordinate_unit=CoordinateUnit(frame_payload.get("coordinate_unit")),
    )
    rhombic_enabled = tensor_payload.get("rhombic_enabled")
    if not isinstance(rhombic_enabled, bool):
        raise PCSPayloadError("tensor.rhombic_enabled must be a boolean")
    tensor = PCSTensor(
        dchi_ax=float(tensor_payload.get("dchi_ax")),
        dchi_rh=float(tensor_payload.get("dchi_rh", 0.0)),
        rhombic_enabled=rhombic_enabled,
        frame=frame,
        susceptibility_unit=SusceptibilityUnit(
            tensor_payload.get("susceptibility_unit")
        ),
    )

    atoms = tuple(
        PCSAtom(
            ref_id=int(atom_payload.get("ref_id")),
            position=tuple(
                _sequence(atom_payload.get("position"), "atom.position")
            ),
            label=str(atom_payload.get("label", "")),
            coordinate_unit=CoordinateUnit(atom_payload.get("coordinate_unit")),
            metadata=_metadata_from(atom_payload),
        )
        for atom_payload in (
            _mapping(item, "atom")
            for item in _sequence(root.get("atoms"), "atoms")
        )
    )
    targets = tuple(
        PCSStructureTarget(
            target_id=str(target_payload.get("target_id", "")),
            site_member_ref_ids=tuple(
                tuple(int(ref_id) for ref_id in _sequence(group, "target site"))
                for group in _sequence(
                    target_payload.get("site_member_ref_ids"),
                    "target.site_member_ref_ids",
                )
            ),
            member_reduction=TargetReduction(
                target_payload.get("member_reduction")
            ),
            site_reduction=TargetReduction(target_payload.get("site_reduction")),
            assignment=target_payload.get("assignment"),
            experimental_pcs_ppm=_optional_float(
                target_payload, "experimental_pcs_ppm"
            ),
            uncertainty_ppm=_optional_float(target_payload, "uncertainty_ppm"),
            metadata=_metadata_from(target_payload),
        )
        for target_payload in (
            _mapping(item, "target")
            for item in _sequence(root.get("targets", ()), "targets")
        )
    )

    try:
        return PCSPredictionRequest(
            tensor=tensor,
            atoms=atoms,
            targets=targets,
            request_id=str(root.get("request_id", "")),
            schema_version=int(root.get("schema_version")),
            metadata=_metadata_from(root),
        )
    except (TypeError, ValueError) as exc:
        raise PCSPayloadError(str(exc)) from exc


def serialize_result(result: PCSPredictionResult) -> dict[str, Any]:
    """Return a detached JSON-compatible representation of a result DTO."""

    if not isinstance(result, PCSPredictionResult):
        raise TypeError("result must be a PCSPredictionResult")
    return {
        "api_version": result.api_version,
        "schema_version": result.schema_version,
        "request_id": result.request_id,
        "engine_id": result.engine_id,
        "engine_version": result.engine_version,
        "atom_predictions": [
            {
                "ref_id": item.ref_id,
                "pcs_ppm": item.pcs_ppm,
                "g_ax": item.g_ax,
                "g_rh": item.g_rh,
                "distance_angstrom": item.distance_angstrom,
            }
            for item in result.atom_predictions
        ],
        "target_predictions": [
            {
                "target_id": item.target_id,
                "pcs_ppm": item.pcs_ppm,
                "site_values_ppm": list(item.site_values_ppm),
                "member_ref_ids": list(item.member_ref_ids),
                "experimental_pcs_ppm": item.experimental_pcs_ppm,
                "residual_ppm": item.residual_ppm,
            }
            for item in result.target_predictions
        ],
        "metadata": _json_value(result.metadata, "result.metadata"),
    }


def deserialize_result(payload: Mapping[str, Any]) -> PCSPredictionResult:
    """Build and validate an L0A result from a primitive payload."""

    try:
        return _deserialize_result(payload)
    except PCSPayloadError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise PCSPayloadError(str(exc)) from exc


def _deserialize_result(payload: Mapping[str, Any]) -> PCSPredictionResult:
    """Internal result decoder; public wrapper normalizes payload errors."""

    root = _mapping(payload, "result")
    atoms = tuple(
        PCSAtomPrediction(
            ref_id=int(item.get("ref_id")),
            pcs_ppm=float(item.get("pcs_ppm")),
            g_ax=float(item.get("g_ax")),
            g_rh=float(item.get("g_rh")),
            distance_angstrom=_optional_float(item, "distance_angstrom"),
        )
        for item in (
            _mapping(value, "atom prediction")
            for value in _sequence(
                root.get("atom_predictions"), "atom_predictions"
            )
        )
    )
    targets = tuple(
        PCSTargetPrediction(
            target_id=str(item.get("target_id", "")),
            pcs_ppm=float(item.get("pcs_ppm")),
            site_values_ppm=tuple(
                float(value)
                for value in _sequence(
                    item.get("site_values_ppm"), "site_values_ppm"
                )
            ),
            member_ref_ids=tuple(
                int(value)
                for value in _sequence(item.get("member_ref_ids"), "member_ref_ids")
            ),
            experimental_pcs_ppm=_optional_float(item, "experimental_pcs_ppm"),
            residual_ppm=_optional_float(item, "residual_ppm"),
        )
        for item in (
            _mapping(value, "target prediction")
            for value in _sequence(
                root.get("target_predictions", ()), "target_predictions"
            )
        )
    )
    try:
        return PCSPredictionResult(
            request_id=str(root.get("request_id", "")),
            atom_predictions=atoms,
            target_predictions=targets,
            engine_id=str(root.get("engine_id", "")),
            engine_version=str(root.get("engine_version", "")),
            api_version=str(root.get("api_version", "")),
            schema_version=int(root.get("schema_version")),
            metadata=_metadata_from(root),
        )
    except (TypeError, ValueError) as exc:
        raise PCSPayloadError(str(exc)) from exc


class PCSQueryService:
    """Read-only facade; prediction is a pure query, not a command."""

    api_version = PCS_FORWARD_API_VERSION
    schema_version = PCS_FORWARD_SCHEMA_VERSION

    def __init__(self, engine: PCSForwardModel | None = None) -> None:
        if engine is not None and not isinstance(engine, PCSForwardModel):
            raise TypeError("engine must implement PCSForwardModel")
        self._engine = PCSForwardEngine() if engine is None else engine

    def get_api_info(self) -> dict[str, Any]:
        """Return detached version/capability information for discovery."""

        return {
            "name": PCS_API_NAME,
            "service_kind": "query_only",
            "api_version": self.api_version,
            "schema_version": self.schema_version,
            "engine_id": str(getattr(self._engine, "engine_id", "custom")),
            "engine_version": str(
                getattr(self._engine, "engine_version", "unspecified")
            ),
            "capabilities": [
                "typed_prediction",
                "payload_prediction",
                "structured_targets",
            ],
            "supports_commands": False,
        }

    def predict(self, request: PCSPredictionRequest) -> PCSPredictionResult:
        """Execute a typed prediction query without mutating the request."""

        return self._engine.predict(request)

    def predict_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute a prediction using only JSON-compatible primitives."""

        request = deserialize_request(payload)
        return serialize_result(self.predict(request))


_DEFAULT_QUERY_SERVICE = PCSQueryService()


def get_api_info() -> dict[str, Any]:
    """Return public PCS query API discovery information."""

    return _DEFAULT_QUERY_SERVICE.get_api_info()


def predict(request: PCSPredictionRequest) -> PCSPredictionResult:
    """Stable typed prediction entry point."""

    return _DEFAULT_QUERY_SERVICE.predict(request)


def predict_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Stable primitive-payload prediction entry point."""

    return _DEFAULT_QUERY_SERVICE.predict_payload(payload)


__all__ = [
    "PCS_API_NAME",
    "PCSPayloadError",
    "PCSQueryService",
    "deserialize_request",
    "deserialize_result",
    "get_api_info",
    "predict",
    "predict_payload",
    "serialize_request",
    "serialize_result",
]
