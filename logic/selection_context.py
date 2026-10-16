"""Shared cross-view selection state for PCS Analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class SelectionContext:
    ref_ids: tuple[int, ...] = ()
    atom_ids: tuple[int, ...] = ()
    assignment: str | None = None
    peak_ids: tuple[str, ...] = ()
    dataset_id: str | None = None
    source: str | None = None
    revision: int = 0

    @property
    def primary_ref_id(self) -> int | None:
        return self.ref_ids[0] if self.ref_ids else None

    @property
    def primary_atom_id(self) -> int | None:
        return self.atom_ids[0] if self.atom_ids else None


SelectionListener = Callable[[SelectionContext], None]


class SelectionHub:
    """Synchronous event hub with nested-publication queuing."""

    def __init__(self) -> None:
        self._context = SelectionContext()
        self._listeners: dict[int, SelectionListener] = {}
        self._next_token = 1
        self._publishing = False
        self._queued: SelectionContext | None = None

    @property
    def context(self) -> SelectionContext:
        return self._context

    def subscribe(self, listener: SelectionListener, *, fire_immediately=False) -> int:
        token = self._next_token
        self._next_token += 1
        self._listeners[token] = listener
        if fire_immediately:
            listener(self._context)
        return token

    def unsubscribe(self, token: int) -> None:
        self._listeners.pop(int(token), None)

    def publish(
        self,
        *,
        ref_ids: Iterable[int] | None = None,
        atom_ids: Iterable[int] | None = None,
        assignment: str | None = None,
        peak_ids: Iterable[str] | None = None,
        dataset_id: str | None = None,
        source: str | None = None,
    ) -> SelectionContext:
        refs = _unique_int_tuple(ref_ids)
        atoms = _unique_int_tuple(atom_ids)

        if ref_ids is not None and atom_ids is None:
            atoms = refs
        elif atom_ids is not None and ref_ids is None:
            refs = atoms

        context = SelectionContext(
            ref_ids=refs if ref_ids is not None or atom_ids is not None else self._context.ref_ids,
            atom_ids=atoms if atom_ids is not None or ref_ids is not None else self._context.atom_ids,
            assignment=assignment,
            peak_ids=_unique_str_tuple(peak_ids),
            dataset_id=dataset_id,
            source=source,
            revision=self._context.revision + 1,
        )
        self._dispatch(context)
        return self._context

    def clear(self, *, source: str | None = None) -> SelectionContext:
        self._dispatch(
            SelectionContext(
                source=source,
                revision=self._context.revision + 1,
            )
        )
        return self._context

    def restore(self, context: SelectionContext) -> SelectionContext:
        """Activate a document-owned context without replacing listeners."""
        if not isinstance(context, SelectionContext):
            raise TypeError("context must be a SelectionContext")
        self._dispatch(context)
        return self._context

    def _dispatch(self, context: SelectionContext) -> None:
        if self._publishing:
            self._queued = context
            return
        self._publishing = True
        try:
            pending = context
            while pending is not None:
                self._context = pending
                self._queued = None
                for listener in tuple(self._listeners.values()):
                    listener(pending)
                pending = self._queued
        finally:
            self._publishing = False


def ensure_selection_hub(state: dict) -> SelectionHub:
    hub = state.get("selection_hub")
    if isinstance(hub, SelectionHub):
        return hub
    hub = SelectionHub()
    state["selection_hub"] = hub
    state["selection_context"] = hub.context

    def mirror(context: SelectionContext) -> None:
        state["selection_context"] = context
        try:
            from logic.pcs_document import active_document_view

            active_document_view(state)["selection_context"] = context
        except (KeyError, RuntimeError):
            pass

    state["_selection_state_mirror_token"] = hub.subscribe(mirror)
    return hub


def restore_active_document_selection(state: dict) -> SelectionContext:
    """Route the active document's selection through the stable listener hub."""
    from logic.pcs_document import active_document_view

    document = active_document_view(state)
    context = document.get("selection_context", SelectionContext())
    if not isinstance(context, SelectionContext):
        context = SelectionContext()
    return ensure_selection_hub(state).restore(context)


def publish_ref_selection(
    state: dict,
    ref_ids: Iterable[int],
    *,
    source: str,
    assignment: str | None = None,
) -> SelectionContext:
    return ensure_selection_hub(state).publish(
        ref_ids=ref_ids,
        assignment=assignment,
        source=source,
    )


def publish_tree_selection(state: dict, *, source: str = "pcs_table") -> SelectionContext:
    tree = state.get("tree")
    refs: list[int] = []
    if tree is not None:
        try:
            for item in tree.selection():
                values = tree.item(item, "values")
                if values:
                    refs.append(int(values[0]))
        except Exception:
            refs = []
    if refs:
        return publish_ref_selection(state, refs, source=source)
    return ensure_selection_hub(state).clear(source=source)


def apply_context_to_tree(state: dict, context: SelectionContext) -> bool:
    tree = state.get("tree")
    row_by_id = state.get("row_by_id", {}) or {}
    if tree is None:
        return False
    items = [row_by_id[r] for r in context.ref_ids if r in row_by_id]
    if not items:
        return False
    try:
        if tuple(tree.selection()) != tuple(items):
            tree.selection_set(items)
        tree.focus(items[0])
        tree.see(items[0])
        return True
    except Exception:
        return False


def _unique_int_tuple(values):
    if values is None:
        return ()
    seen, result = set(), []
    for value in values:
        number = int(value)
        if number not in seen:
            seen.add(number)
            result.append(number)
    return tuple(result)


def _unique_str_tuple(values):
    if values is None:
        return ()
    seen, result = set(), []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)
