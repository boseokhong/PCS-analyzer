"""Compatibility consumers for application-level PCS model events.

M6 keeps the existing refresh callbacks public, but canonical-origin tensor
changes reach them through the event hub. Legacy-origin events are ignored by
these consumers because their producer is already executing the old callback.
"""

from __future__ import annotations

from typing import Any, MutableMapping

from logic.nmr_delta_data_manager import push_layers_to_nmr_if_open
from logic.pcs_events import (
    PCSModelEvent,
    PCSPredictionModelChanged,
    PCSTensorChanged,
)
from logic.pcs_model_state import (
    PCS_MODEL_REFRESHED_REVISION_KEY,
    ensure_pcs_model_event_hub,
)


PCS_REFRESH_SUBSCRIPTIONS_KEY = "_pcs_refresh_subscription_tokens"
_PCS_REFRESH_HUB_KEY = "_pcs_refresh_subscription_hub"
PCS_EVENT_REFRESH_ACTIVE_KEY = "_pcs_event_refresh_active"


def _is_canonical_origin(event: PCSModelEvent) -> bool:
    return (
        isinstance(event, (PCSTensorChanged, PCSPredictionModelChanged))
        and not event.metadata.source.startswith("legacy:")
    )


def subscribe_pcs_refresh_consumers(
    state: MutableMapping[str, Any],
) -> tuple[int, int]:
    """Subscribe ordered view and NMR compatibility consumers once."""
    hub = ensure_pcs_model_event_hub(state)
    existing_hub = state.get(_PCS_REFRESH_HUB_KEY)
    existing_tokens = state.get(PCS_REFRESH_SUBSCRIPTIONS_KEY)
    if existing_hub is hub and isinstance(existing_tokens, tuple):
        return existing_tokens

    unsubscribe_pcs_refresh_consumers(state)

    def refresh_legacy_views(event: PCSModelEvent) -> None:
        if not _is_canonical_origin(event):
            return
        callback = state.get("update_graph")
        if not callable(callback):
            return

        state[PCS_EVENT_REFRESH_ACTIVE_KEY] = event.metadata.event_id
        try:
            callback()
        finally:
            state.pop(PCS_EVENT_REFRESH_ACTIVE_KEY, None)
        state[PCS_MODEL_REFRESHED_REVISION_KEY] = event.metadata.revision

    def refresh_nmr_compatibility(event: PCSModelEvent) -> None:
        if not _is_canonical_origin(event):
            return
        # The ordered view consumer must finish first so legacy pcs_by_id is
        # current before the compatibility NMR layer is rebuilt.
        if state.get(PCS_MODEL_REFRESHED_REVISION_KEY) != event.metadata.revision:
            return
        push_layers_to_nmr_if_open(state)

    tokens = (
        hub.subscribe(refresh_legacy_views),
        hub.subscribe(refresh_nmr_compatibility),
    )
    state[_PCS_REFRESH_HUB_KEY] = hub
    state[PCS_REFRESH_SUBSCRIPTIONS_KEY] = tokens
    return tokens


def unsubscribe_pcs_refresh_consumers(state: MutableMapping[str, Any]) -> None:
    """Remove application refresh consumers; safe to call repeatedly."""
    hub = state.pop(_PCS_REFRESH_HUB_KEY, None)
    tokens = state.pop(PCS_REFRESH_SUBSCRIPTIONS_KEY, ())
    if hub is not None:
        for token in tokens:
            hub.unsubscribe(token)
    state.pop(PCS_EVENT_REFRESH_ACTIVE_KEY, None)


__all__ = [
    "PCS_EVENT_REFRESH_ACTIVE_KEY",
    "PCS_REFRESH_SUBSCRIPTIONS_KEY",
    "subscribe_pcs_refresh_consumers",
    "unsubscribe_pcs_refresh_consumers",
]
