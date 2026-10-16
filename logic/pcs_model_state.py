"""Application-level PCS tensor commands and event wiring.

The UI still owns the legacy split tensor inputs during M5.  This module is
the compatibility boundary that keeps :class:`ApplicationPCSTensor` current
and publishes one typed event for each effective tensor change.
"""

from __future__ import annotations

from typing import Any, Callable, MutableMapping

from logic.pcs_events import (
    PCSEventMetadata,
    PCSModelEventHub,
    PCSPredictionModelChanged,
    PCSTensorChanged,
)
from logic.pcs_prediction_model import (
    PredictionModel,
    get_prediction_model,
    install_prediction_model,
    parse_prediction_model,
)
from logic.pcs_tensor_state import (
    ApplicationPCSTensor,
    PCS_TENSOR_STATE_KEY,
    apply_canonical_tensor_to_legacy,
    canonical_tensor_from_legacy,
)
from logic.pcs_orientation import tensors_have_same_physical_state


PCS_MODEL_EVENT_HUB_KEY = "pcs_model_event_hub"
PCS_MODEL_REVISION_KEY = "pcs_model_revision"
PCS_MODEL_REFRESHED_REVISION_KEY = "pcs_model_refreshed_revision"


def ensure_pcs_model_event_hub(
    state: MutableMapping[str, Any],
) -> PCSModelEventHub:
    """Return the single PCS model event hub owned by this application state."""
    hub = state.get(PCS_MODEL_EVENT_HUB_KEY)
    if hub is None:
        hub = PCSModelEventHub()
        state[PCS_MODEL_EVENT_HUB_KEY] = hub
    elif not isinstance(hub, PCSModelEventHub):
        raise TypeError(f"{PCS_MODEL_EVENT_HUB_KEY} must be a PCSModelEventHub")
    state.setdefault(PCS_MODEL_REVISION_KEY, 0)
    return hub


def initialize_pcs_model_state(
    state: MutableMapping[str, Any],
) -> ApplicationPCSTensor:
    """Install the canonical baseline and hub without announcing a change."""
    ensure_pcs_model_event_hub(state)
    install_prediction_model(state, get_prediction_model(state))
    current = state.get(PCS_TENSOR_STATE_KEY)
    if isinstance(current, ApplicationPCSTensor):
        return current
    return canonical_tensor_from_legacy(state)


def update_prediction_model(
    state: MutableMapping[str, Any],
    prediction_model: PredictionModel | str,
    *,
    source: str,
    correlation_id: str | None = None,
    causation_id: str | None = None,
) -> bool:
    """Set the active forward-prediction policy and publish one change event."""
    requested = parse_prediction_model(prediction_model)
    previous = get_prediction_model(state)
    if requested is previous:
        install_prediction_model(state, previous)
        return False

    install_prediction_model(state, requested)
    hub = ensure_pcs_model_event_hub(state)
    revision = int(state.get(PCS_MODEL_REVISION_KEY, 0)) + 1
    state[PCS_MODEL_REVISION_KEY] = revision
    hub.publish(
        PCSPredictionModelChanged(
            metadata=PCSEventMetadata(
                source=source,
                revision=revision,
                correlation_id=correlation_id,
                causation_id=causation_id,
            ),
            prediction_model=requested,
        )
    )
    from logic.pcs_document import mark_document_mutated
    mark_document_mutated(state, "tensor_orientation")
    return True


def _publish_tensor_change(
    state: MutableMapping[str, Any],
    tensor: ApplicationPCSTensor,
    *,
    source: str,
    correlation_id: str | None = None,
    causation_id: str | None = None,
) -> PCSTensorChanged:
    hub = ensure_pcs_model_event_hub(state)
    revision = int(state.get(PCS_MODEL_REVISION_KEY, 0)) + 1
    state[PCS_MODEL_REVISION_KEY] = revision
    event = PCSTensorChanged(
        metadata=PCSEventMetadata(
            source=source,
            revision=revision,
            correlation_id=correlation_id,
            causation_id=causation_id,
        ),
        tensor=tensor.to_forward_tensor(),
    )
    hub.publish(event)
    from logic.pcs_document import mark_document_mutated
    mark_document_mutated(state, "tensor_orientation")
    return event


def sync_canonical_tensor_from_legacy(
    state: MutableMapping[str, Any],
    *,
    source: str,
    correlation_id: str | None = None,
    causation_id: str | None = None,
) -> ApplicationPCSTensor:
    """Import legacy inputs and publish only when their canonical value changed."""
    previous = state.get(PCS_TENSOR_STATE_KEY)
    current = canonical_tensor_from_legacy(state)

    # A first import establishes the baseline; initialization is not a model
    # change and must remain safe for read/setup paths.
    if not isinstance(previous, ApplicationPCSTensor):
        ensure_pcs_model_event_hub(state)
        return current
    if tensors_have_same_physical_state(current, previous):
        # Keep the existing canonical Euler representation.  A legacy redraw
        # may project it to XYZ, but that is not a physical model change.
        state[PCS_TENSOR_STATE_KEY] = previous
        return previous

    _publish_tensor_change(
        state,
        current,
        source=source,
        correlation_id=correlation_id,
        causation_id=causation_id,
    )
    return current


def update_canonical_tensor(
    state: MutableMapping[str, Any],
    tensor: ApplicationPCSTensor,
    *,
    source: str,
    legacy_callback: Callable[[], None] | None = None,
    correlation_id: str | None = None,
    causation_id: str | None = None,
) -> bool:
    """Apply a canonical tensor command and retain the legacy projection.

    The optional callback is the existing redraw bridge.  The canonical value
    is installed before it runs, so a callback that synchronizes legacy state
    cannot publish a duplicate event for the same logical command.
    """
    if not isinstance(tensor, ApplicationPCSTensor):
        raise TypeError("tensor must be an ApplicationPCSTensor")

    previous = initialize_pcs_model_state(state)
    if tensors_have_same_physical_state(tensor, previous):
        return False

    apply_canonical_tensor_to_legacy(state, tensor)
    try:
        _publish_tensor_change(
            state,
            tensor,
            source=source,
            correlation_id=correlation_id,
            causation_id=causation_id,
        )
    finally:
        # Event listener failures must not skip the compatibility redraw.
        event_refresh_handled = (
            state.get(PCS_MODEL_REFRESHED_REVISION_KEY)
            == state.get(PCS_MODEL_REVISION_KEY)
            and legacy_callback is state.get("update_graph")
        )
        if legacy_callback is not None and not event_refresh_handled:
            legacy_callback()
    return True


__all__ = [
    "PCS_MODEL_EVENT_HUB_KEY",
    "PCS_MODEL_REFRESHED_REVISION_KEY",
    "PCS_MODEL_REVISION_KEY",
    "ensure_pcs_model_event_hub",
    "initialize_pcs_model_state",
    "sync_canonical_tensor_from_legacy",
    "update_prediction_model",
    "update_canonical_tensor",
]
