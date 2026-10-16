"""Optional bridge between NMR Workbench and PCS Analyzer SelectionHub.

NMR Workbench can still be imported in isolation. The PCS Analyzer-specific
selection dependency is resolved lazily only when a PCS state dictionary is
available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Any


@dataclass(frozen=True)
class WorkbenchSelection:
    assignment: str | None = None
    peak_ids: tuple[str, ...] = ()
    dataset_id: str | None = None
    ref_ids: tuple[int, ...] = ()
    source: str = "nmr_workbench"


def _selection_api():
    try:
        from logic.selection_context import ensure_selection_hub
    except Exception:
        return None
    return ensure_selection_hub


def shared_selection_hub(pcs_state: dict | None):
    """Return the PCS Analyzer SelectionHub, or None in standalone mode."""
    if not isinstance(pcs_state, dict):
        return None
    ensure = _selection_api()
    if ensure is None:
        return None
    try:
        return ensure(pcs_state)
    except Exception:
        return None


def publish_workbench_selection(
    pcs_state: dict | None,
    *,
    assignment: str | None,
    peak_ids: Iterable[str] = (),
    dataset_id: str | None = None,
    ref_ids: Iterable[int] = (),
    source: str = "nmr_workbench",
):
    """Publish semantic NMR selection without reaching into PCS Treeview."""
    hub = shared_selection_hub(pcs_state)
    if hub is None:
        return None

    assignment_text = str(assignment).strip() if assignment else None
    return hub.publish(
        ref_ids=tuple(int(value) for value in ref_ids),
        assignment=assignment_text or None,
        peak_ids=tuple(str(value) for value in peak_ids),
        dataset_id=str(dataset_id) if dataset_id is not None else None,
        source=str(source),
    )


def subscribe_workbench_selection(
    pcs_state: dict | None,
    callback: Callable[[Any], None],
    *,
    fire_immediately: bool = False,
) -> tuple[object, int] | None:
    """Subscribe to the shared selection stream."""
    hub = shared_selection_hub(pcs_state)
    if hub is None:
        return None
    token = hub.subscribe(callback, fire_immediately=fire_immediately)
    return hub, token


def unsubscribe_workbench_selection(subscription) -> None:
    if not subscription:
        return
    try:
        hub, token = subscription
        hub.unsubscribe(token)
    except Exception:
        pass
