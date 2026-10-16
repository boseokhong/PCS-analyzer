"""Minimal UI-independent events for future PCS model consumers.

This is intentionally separate from :mod:`logic.selection_context`: selection
events describe view navigation, while these events describe scientific model
inputs and prediction snapshots. Application-level publication is owned by
:mod:`logic.pcs_model_state`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, ClassVar, Mapping, TypeAlias
from uuid import uuid4

from logic.pcs_api import (
    deserialize_request,
    deserialize_result,
    serialize_request,
    serialize_result,
)
from logic.pcs_forward_api import (
    AngleUnit,
    CoordinateUnit,
    EulerConvention,
    PCS_FORWARD_API_VERSION,
    PCS_FORWARD_SCHEMA_VERSION,
    PCSFrame,
    PCSPredictionRequest,
    PCSPredictionResult,
    PCSTensor,
    SusceptibilityUnit,
)
from logic.pcs_prediction_model import PredictionModel


class PCSModelEventType(str, Enum):
    TENSOR_CHANGED = "tensor_changed"
    PREDICTION_MODEL_CHANGED = "prediction_model_changed"
    MODEL_INPUT_CHANGED = "model_input_changed"
    PREDICTION_UPDATED = "prediction_updated"


@dataclass(frozen=True)
class PCSEventMetadata:
    """Origin and causality metadata for loop prevention and tracing."""

    source: str
    revision: int
    event_id: str = field(default_factory=lambda: str(uuid4()))
    correlation_id: str | None = None
    causation_id: str | None = None
    api_version: str = PCS_FORWARD_API_VERSION
    schema_version: int = PCS_FORWARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        source = str(self.source).strip()
        event_id = str(self.event_id).strip()
        revision = int(self.revision)
        if not source:
            raise ValueError("event source must not be blank")
        if not event_id:
            raise ValueError("event_id must not be blank")
        if revision < 0:
            raise ValueError("event revision must be non-negative")
        if str(self.api_version) != PCS_FORWARD_API_VERSION:
            raise ValueError(f"unsupported PCS event api_version: {self.api_version!r}")
        if int(self.schema_version) != PCS_FORWARD_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported PCS event schema_version: {self.schema_version!r}"
            )
        correlation = self.correlation_id
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(
            self,
            "correlation_id",
            event_id if correlation is None else str(correlation),
        )
        object.__setattr__(
            self,
            "causation_id",
            None if self.causation_id is None else str(self.causation_id),
        )
        object.__setattr__(self, "api_version", str(self.api_version))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def loop_key(self) -> tuple[str, int]:
        """Stable key consumers can remember to reject their own stale echoes."""

        return self.source, self.revision


@dataclass(frozen=True)
class PCSTensorChanged:
    metadata: PCSEventMetadata
    tensor: PCSTensor
    event_type: ClassVar[PCSModelEventType] = PCSModelEventType.TENSOR_CHANGED

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, PCSEventMetadata):
            raise TypeError("metadata must be PCSEventMetadata")
        if not isinstance(self.tensor, PCSTensor):
            raise TypeError("tensor must be PCSTensor")


@dataclass(frozen=True)
class PCSPredictionModelChanged:
    metadata: PCSEventMetadata
    prediction_model: PredictionModel
    event_type: ClassVar[PCSModelEventType] = (
        PCSModelEventType.PREDICTION_MODEL_CHANGED
    )

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, PCSEventMetadata):
            raise TypeError("metadata must be PCSEventMetadata")
        object.__setattr__(
            self,
            "prediction_model",
            PredictionModel(self.prediction_model),
        )


@dataclass(frozen=True)
class PCSModelInputChanged:
    metadata: PCSEventMetadata
    request: PCSPredictionRequest
    event_type: ClassVar[PCSModelEventType] = PCSModelEventType.MODEL_INPUT_CHANGED

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, PCSEventMetadata):
            raise TypeError("metadata must be PCSEventMetadata")
        if not isinstance(self.request, PCSPredictionRequest):
            raise TypeError("request must be PCSPredictionRequest")


@dataclass(frozen=True)
class PCSPredictionUpdated:
    metadata: PCSEventMetadata
    result: PCSPredictionResult
    event_type: ClassVar[PCSModelEventType] = PCSModelEventType.PREDICTION_UPDATED

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, PCSEventMetadata):
            raise TypeError("metadata must be PCSEventMetadata")
        if not isinstance(self.result, PCSPredictionResult):
            raise TypeError("result must be PCSPredictionResult")


PCSModelEvent: TypeAlias = (
    PCSTensorChanged
    | PCSPredictionModelChanged
    | PCSModelInputChanged
    | PCSPredictionUpdated
)
PCSModelEventListener: TypeAlias = Callable[[PCSModelEvent], None]


@dataclass(frozen=True)
class PCSDispatchFailure:
    token: int
    event: PCSModelEvent
    error: Exception


class PCSDispatchError(RuntimeError):
    """Reports listener failures after all subscribers and queued events run."""

    def __init__(self, failures: tuple[PCSDispatchFailure, ...]) -> None:
        self.failures = failures
        super().__init__(f"{len(failures)} PCS event listener failure(s)")


class PCSModelEventHub:
    """Synchronous PCS event hub with FIFO nested-publication queuing."""

    def __init__(self) -> None:
        self._listeners: dict[int, PCSModelEventListener] = {}
        self._next_token = 1
        self._publishing = False
        self._queue: deque[PCSModelEvent] = deque()

    def subscribe(self, listener: PCSModelEventListener) -> int:
        if not callable(listener):
            raise TypeError("listener must be callable")
        token = self._next_token
        self._next_token += 1
        self._listeners[token] = listener
        return token

    def unsubscribe(self, token: int) -> None:
        self._listeners.pop(int(token), None)

    def publish(self, event: PCSModelEvent) -> None:
        if not isinstance(
            event,
            (
                PCSTensorChanged,
                PCSPredictionModelChanged,
                PCSModelInputChanged,
                PCSPredictionUpdated,
            ),
        ):
            raise TypeError("event must be a typed PCSModelEvent")

        self._queue.append(event)
        if self._publishing:
            return

        failures: list[PCSDispatchFailure] = []
        self._publishing = True
        try:
            while self._queue:
                current = self._queue.popleft()
                for token, listener in tuple(self._listeners.items()):
                    try:
                        listener(current)
                    except Exception as exc:
                        failures.append(PCSDispatchFailure(token, current, exc))
        finally:
            self._publishing = False

        if failures:
            raise PCSDispatchError(tuple(failures))


class PCSEventPayloadError(ValueError):
    """Raised when a plugin-facing event payload is malformed."""


def _metadata_payload(metadata: PCSEventMetadata) -> dict[str, object]:
    return {
        "api_version": metadata.api_version,
        "schema_version": metadata.schema_version,
        "source": metadata.source,
        "revision": metadata.revision,
        "event_id": metadata.event_id,
        "correlation_id": metadata.correlation_id,
        "causation_id": metadata.causation_id,
    }


def _tensor_payload(tensor: PCSTensor) -> dict[str, object]:
    frame = tensor.frame
    return {
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
    }


def serialize_event(event: PCSModelEvent) -> dict[str, object]:
    """Return a detached JSON-compatible event payload."""

    payload = _metadata_payload(event.metadata)
    payload["event_type"] = event.event_type.value
    if isinstance(event, PCSTensorChanged):
        payload["data"] = {"tensor": _tensor_payload(event.tensor)}
    elif isinstance(event, PCSPredictionModelChanged):
        payload["data"] = {
            "prediction_model": event.prediction_model.value,
        }
    elif isinstance(event, PCSModelInputChanged):
        payload["data"] = {"request": serialize_request(event.request)}
    elif isinstance(event, PCSPredictionUpdated):
        payload["data"] = {"result": serialize_result(event.result)}
    else:
        raise TypeError("event must be a typed PCSModelEvent")
    return payload


def _event_metadata(payload: Mapping[str, object]) -> PCSEventMetadata:
    return PCSEventMetadata(
        source=str(payload.get("source", "")),
        revision=int(payload.get("revision")),
        event_id=str(payload.get("event_id", "")),
        correlation_id=payload.get("correlation_id"),
        causation_id=payload.get("causation_id"),
        api_version=str(payload.get("api_version", "")),
        schema_version=int(payload.get("schema_version")),
    )


def _tensor_from_payload(payload: Mapping[str, object]) -> PCSTensor:
    frame_payload = payload.get("frame")
    if not isinstance(frame_payload, Mapping):
        raise PCSEventPayloadError("tensor.frame must be an object")
    enabled = payload.get("rhombic_enabled")
    if not isinstance(enabled, bool):
        raise PCSEventPayloadError("tensor.rhombic_enabled must be a boolean")
    return PCSTensor(
        dchi_ax=float(payload.get("dchi_ax")),
        dchi_rh=float(payload.get("dchi_rh", 0.0)),
        rhombic_enabled=enabled,
        susceptibility_unit=SusceptibilityUnit(
            payload.get("susceptibility_unit")
        ),
        frame=PCSFrame(
            metal_position=tuple(frame_payload.get("metal_position", ())),
            euler_angles=tuple(frame_payload.get("euler_angles", ())),
            convention=EulerConvention(frame_payload.get("convention")),
            angle_unit=AngleUnit(frame_payload.get("angle_unit")),
            coordinate_unit=CoordinateUnit(frame_payload.get("coordinate_unit")),
        ),
    )


def deserialize_event(payload: Mapping[str, object]) -> PCSModelEvent:
    """Build a typed event from a plugin-facing primitive payload."""

    try:
        if not isinstance(payload, Mapping):
            raise PCSEventPayloadError("event must be an object")
        metadata = _event_metadata(payload)
        event_type = PCSModelEventType(payload.get("event_type"))
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise PCSEventPayloadError("event.data must be an object")

        if event_type is PCSModelEventType.TENSOR_CHANGED:
            tensor_payload = data.get("tensor")
            if not isinstance(tensor_payload, Mapping):
                raise PCSEventPayloadError("event.data.tensor must be an object")
            return PCSTensorChanged(metadata, _tensor_from_payload(tensor_payload))
        if event_type is PCSModelEventType.PREDICTION_MODEL_CHANGED:
            value = data.get("prediction_model")
            try:
                model = PredictionModel(value)
            except (TypeError, ValueError) as exc:
                raise PCSEventPayloadError(
                    "event.data.prediction_model is invalid"
                ) from exc
            return PCSPredictionModelChanged(metadata, model)
        if event_type is PCSModelEventType.MODEL_INPUT_CHANGED:
            request_payload = data.get("request")
            if not isinstance(request_payload, Mapping):
                raise PCSEventPayloadError("event.data.request must be an object")
            return PCSModelInputChanged(metadata, deserialize_request(request_payload))

        result_payload = data.get("result")
        if not isinstance(result_payload, Mapping):
            raise PCSEventPayloadError("event.data.result must be an object")
        return PCSPredictionUpdated(metadata, deserialize_result(result_payload))
    except PCSEventPayloadError:
        raise
    except (TypeError, ValueError) as exc:
        raise PCSEventPayloadError(str(exc)) from exc


__all__ = [
    "PCSDispatchError",
    "PCSDispatchFailure",
    "PCSEventMetadata",
    "PCSEventPayloadError",
    "PCSModelEvent",
    "PCSModelEventHub",
    "PCSModelEventListener",
    "PCSModelEventType",
    "PCSModelInputChanged",
    "PCSPredictionModelChanged",
    "PCSPredictionUpdated",
    "PCSTensorChanged",
    "deserialize_event",
    "serialize_event",
]
