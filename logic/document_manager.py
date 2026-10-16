"""Application-level PCS document lifecycle and ordering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from logic.pcs_document import (
    PCSDocument,
    capture_active_document,
    commit_pending_document_edits,
    default_document_data,
    hydrate_active_document,
    refresh_follow_active_surfaces,
)


@dataclass(frozen=True)
class ActiveDocumentChanged:
    previous_document_id: str | None
    document_id: str


def _normalized_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False)).casefold()


class DocumentManager:
    """Own PCS documents, active identity, and lifecycle sequencing."""

    def __init__(self, state: dict[str, Any], initial: PCSDocument) -> None:
        if not isinstance(initial, PCSDocument):
            raise TypeError("initial must be a PCSDocument")
        self._state = state
        self._documents: dict[str, PCSDocument] = {
            initial.document_id: initial
        }
        self._active_document_id = initial.document_id
        self._listeners: dict[int, Callable[[ActiveDocumentChanged], None]] = {}
        self._next_listener_token = 1
        self._next_untitled_number = 1

    @property
    def documents(self) -> Mapping[str, PCSDocument]:
        return MappingProxyType(self._documents)

    @property
    def compatibility_documents(self) -> dict[str, PCSDocument]:
        """Mutable legacy mirror; lifecycle code should use manager methods."""
        return self._documents

    @property
    def active_document(self) -> PCSDocument:
        return self._documents[self._active_document_id]

    @property
    def active_document_id(self) -> str:
        return self._active_document_id

    def document_by_id(self, document_id: str) -> PCSDocument | None:
        return self._documents.get(str(document_id))

    def set_default_snapshot(self, values: Mapping[str, Any]) -> None:
        """Compatibility no-op; blank documents never copy live state."""
        if not isinstance(values, Mapping):
            raise TypeError("values must be a mapping")

    def default_snapshot(self) -> dict[str, Any]:
        """Return explicit application defaults, never an active snapshot."""
        return default_document_data(self._state)

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(self._documents)

    def subscribe(
        self, callback: Callable[[ActiveDocumentChanged], None]
    ) -> int:
        if not callable(callback):
            raise TypeError("callback must be callable")
        token = self._next_listener_token
        self._next_listener_token += 1
        self._listeners[token] = callback
        return token

    def unsubscribe(self, token: int) -> None:
        self._listeners.pop(int(token), None)

    def _notify(self, previous_id: str | None, document_id: str) -> None:
        event = ActiveDocumentChanged(previous_id, document_id)
        failures = []
        for callback in tuple(self._listeners.values()):
            try:
                callback(event)
            except Exception as exc:  # lifecycle must survive presentation hooks
                failures.append(exc)
        self._state["_active_document_notification_failures"] = failures

    def _document_for_path(self, path: str | Path) -> PCSDocument | None:
        normalized = _normalized_path(path)
        for document in self._documents.values():
            if document.file_path is None:
                continue
            if _normalized_path(document.file_path) == normalized:
                return document
        return None

    def _next_title(self) -> str:
        existing = {document.title for document in self._documents.values()}
        while True:
            title = f"Untitled {self._next_untitled_number}"
            self._next_untitled_number += 1
            if title not in existing:
                return title

    def add_document(
        self, document: PCSDocument, *, activate: bool = False
    ) -> PCSDocument:
        if not isinstance(document, PCSDocument):
            raise TypeError("document must be a PCSDocument")
        existing_id = self._documents.get(document.document_id)
        if existing_id is not None and existing_id is not document:
            raise ValueError(f"duplicate document ID: {document.document_id}")
        if document.file_path is not None:
            existing_path = self._document_for_path(document.file_path)
            if existing_path is not None and existing_path is not document:
                if activate:
                    self.set_active_document(existing_path.document_id)
                return existing_path
        self._documents[document.document_id] = document
        if activate:
            self.set_active_document(document.document_id)
        return document

    def new_document(self, *, activate: bool = True) -> PCSDocument:
        document = PCSDocument(title=self._next_title())
        document.replace_data(self.default_snapshot())
        document.dirty = False
        document.revision = 0
        return self.add_document(document, activate=activate)

    def duplicate_document(
        self, document_id: str, *, activate: bool = True
    ) -> PCSDocument:
        source = self.document_by_id(document_id)
        if source is None:
            raise KeyError(f"unknown PCS document ID: {document_id}")
        values = source.snapshot()
        defaults = self.default_snapshot()
        # Selection and one-step undo are transient interaction state, not a
        # meaningful part of an explicit project duplicate.
        for key in (
            "current_selected_ids", "selection_context",
            "cartesian_selected_ids", "cartesian_selector_active",
            "cartesian_selector_kind", "_delta_exp_undo", "_delta_obs_undo",
            "_delta_dia_undo", "_experimental_pcs_observations_undo",
            "_nmr_workbench_delta_exp_provenance_undo",
        ):
            if key in defaults:
                values[key] = deepcopy(defaults[key])
            else:
                values.pop(key, None)
        base = source.project_name or Path(source.title).stem or "Untitled"
        document = PCSDocument(
            title=self._next_title(),
            project_name=f"{base} Copy",
            tab_color=source.tab_color,
            dirty=True,
            revision=1,
        )
        document.replace_data(values)
        return self.add_document(document, activate=activate)

    def set_project_name(self, document_id: str, value: str | None) -> None:
        document = self.document_by_id(document_id)
        if document is None:
            raise KeyError(f"unknown PCS document ID: {document_id}")
        normalized = str(value).strip() if value is not None else ""
        normalized = normalized or None
        if document.project_name != normalized:
            document.project_name = normalized
            document.mark_mutated()
            refresh = self._state.get("refresh_document_integration_titles")
            if callable(refresh):
                try:
                    refresh()
                except Exception:
                    pass

    def set_tab_color(self, document_id: str, value: str | None) -> None:
        document = self.document_by_id(document_id)
        if document is None:
            raise KeyError(f"unknown PCS document ID: {document_id}")
        normalized = str(value).strip() if value is not None else ""
        normalized = normalized or None
        if normalized and not (
            len(normalized) == 7
            and normalized.startswith("#")
            and all(ch in "0123456789abcdefABCDEF" for ch in normalized[1:])
        ):
            raise ValueError("tab color must be a #RRGGBB value")
        if document.tab_color != normalized:
            document.tab_color = normalized
            document.mark_mutated()

    def replace_documents(
        self,
        documents,
        *,
        active_document_id: str | None = None,
    ) -> PCSDocument:
        """Atomically install a detached, already-validated workspace set."""
        ordered: dict[str, PCSDocument] = {}
        seen_paths: set[str] = set()
        for document in documents:
            if not isinstance(document, PCSDocument):
                raise TypeError("documents must contain PCSDocument values")
            if document.document_id in ordered:
                raise ValueError(f"duplicate document ID: {document.document_id}")
            if document.file_path is not None:
                normalized = _normalized_path(document.file_path)
                if normalized in seen_paths:
                    continue
                seen_paths.add(normalized)
            ordered[document.document_id] = document
        if not ordered:
            fallback = PCSDocument(title="Untitled 1")
            fallback.replace_data(self.default_snapshot())
            ordered[fallback.document_id] = fallback
        selected = (
            active_document_id
            if active_document_id in ordered
            else next(iter(ordered))
        )
        previous_id = self._active_document_id
        self._documents.clear()
        self._documents.update(ordered)
        self._active_document_id = str(selected)
        active = self.active_document
        self._state["active_document"] = active
        self._hydrate_document(active)
        refresh_follow_active_surfaces(self._state, include_legacy=False)
        self._notify(previous_id, active.document_id)
        return active

    def move_document(self, document_id: str, new_index: int) -> None:
        """Change presentation order without activating or replacing a document."""
        document_id = str(document_id)
        if document_id not in self._documents:
            raise KeyError(f"unknown PCS document ID: {document_id}")
        ordered_ids = list(self._documents)
        ordered_ids.remove(document_id)
        index = max(0, min(int(new_index), len(ordered_ids)))
        ordered_ids.insert(index, document_id)
        ordered = [(key, self._documents[key]) for key in ordered_ids]
        # Mutate in place so the legacy compatibility mapping remains valid.
        self._documents.clear()
        self._documents.update(ordered)

    def _hydrate_document(self, document: PCSDocument) -> None:
        hydrate_active_document(
            self._state, document, refresh=True, _from_manager=True
        )

    def set_active_document(self, document_id: str) -> PCSDocument:
        incoming = self.document_by_id(document_id)
        if incoming is None:
            raise KeyError(f"unknown PCS document ID: {document_id}")
        outgoing = self.active_document
        if incoming is outgoing:
            return incoming

        # All fallible outgoing work occurs before the authoritative ID moves.
        commit_pending_document_edits(self._state)
        capture_active_document(self._state, outgoing)

        previous_id = outgoing.document_id
        self._active_document_id = incoming.document_id
        self._state["active_document"] = incoming
        try:
            self._hydrate_document(incoming)
        except Exception:
            self._active_document_id = previous_id
            self._state["active_document"] = outgoing
            self._hydrate_document(outgoing)
            raise

        # Legacy presenters were refreshed during hydration.  This second
        # pass only reaches follow-active secondary viewers.
        refresh_follow_active_surfaces(self._state, include_legacy=False)
        self._notify(previous_id, incoming.document_id)
        return incoming

    def open_document(
        self, path: str | Path, *, activate: bool = True
    ) -> PCSDocument:
        existing = self._document_for_path(path)
        if existing is not None:
            if activate:
                self.set_active_document(existing.document_id)
            return existing

        from logic.project_io import load_project_document

        document = load_project_document(
            path, self._state, defaults=self.default_snapshot()
        )
        self._documents[document.document_id] = document
        try:
            if activate:
                self.set_active_document(document.document_id)
        except Exception:
            self._documents.pop(document.document_id, None)
            raise
        return document

    def save_document(
        self,
        document_id: str | None = None,
        path: str | Path | None = None,
    ) -> PCSDocument:
        document = (
            self.active_document
            if document_id is None
            else self.document_by_id(document_id)
        )
        if document is None:
            raise KeyError(f"unknown PCS document ID: {document_id}")
        target = Path(path) if path is not None else document.file_path
        if target is None:
            raise ValueError("A path is required to save an untitled document")
        from logic.project_io import normalize_project_path, save_project_file

        target = normalize_project_path(target)
        duplicate = self._document_for_path(target)
        if duplicate is not None and duplicate is not document:
            raise ValueError("Another open PCS document already uses this path")

        if document is self.active_document:
            commit_pending_document_edits(self._state)
        save_project_file(target, self._state, document=document)
        refresh = self._state.get("refresh_document_integration_titles")
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass
        return document

    def save_all(
        self,
        path_provider: Callable[[PCSDocument], str | Path | None] | None = None,
    ) -> bool:
        """Save dirty documents in order, stopping safely on path cancellation."""
        for document in tuple(self._documents.values()):
            if not document.dirty:
                continue
            target = document.file_path
            if target is None:
                if path_provider is None:
                    raise ValueError(
                        f"A path is required to save {document.title}"
                    )
                target = path_provider(document)
                if not target:
                    return False
            self.save_document(document.document_id, target)
        return True

    def _resolve_dirty_document(
        self,
        document: PCSDocument,
        decision_hook: Callable[[PCSDocument], Any] | None,
        *,
        save_path: str | Path | None = None,
    ) -> bool:
        if not document.dirty:
            return True
        if decision_hook is None:
            raise ValueError("A dirty-close decision hook is required")
        decision = decision_hook(document)
        decision_path = save_path
        if isinstance(decision, tuple):
            decision, decision_path = decision
        action = str(decision).strip().lower()
        if action == "cancel":
            return False
        if action == "save":
            self.save_document(document.document_id, decision_path)
        elif action != "discard":
            raise ValueError(f"unsupported dirty-close decision: {decision}")
        return True

    def resolve_dirty_documents(
        self,
        decision_hook: Callable[[PCSDocument], Any],
    ) -> bool:
        """Resolve every dirty document for application exit, in tab order."""
        for document in tuple(self._documents.values()):
            if not self._resolve_dirty_document(document, decision_hook):
                return False
        return True

    def close_document(
        self,
        document_id: str,
        *,
        decision_hook: Callable[[PCSDocument], Any] | None = None,
        save_path: str | Path | None = None,
    ) -> bool:
        document = self.document_by_id(document_id)
        if document is None:
            raise KeyError(f"unknown PCS document ID: {document_id}")

        if not self._resolve_dirty_document(
            document, decision_hook, save_path=save_path
        ):
            return False

        if document is self.active_document:
            fallback = next(
                (
                    item for item in self._documents.values()
                    if item is not document
                ),
                None,
            )
            if fallback is None:
                fallback = self.new_document(activate=False)
            self.set_active_document(fallback.document_id)

        self._documents.pop(document.document_id, None)
        # The active fallback was presented before the closed document was
        # removed. Refresh once more so pinned integrations can detect that
        # their explicit origin no longer exists. Inactive closes do not
        # redraw follow-active viewers.
        refresh = self._state.get("refresh_document_integration_titles")
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass
        return True

    def close_documents(
        self,
        document_ids,
        *,
        decision_hook: Callable[[PCSDocument], Any] | None = None,
    ) -> bool:
        """Close documents sequentially; cancellation preserves the remainder."""
        for document_id in tuple(str(value) for value in document_ids):
            if self.document_by_id(document_id) is None:
                continue
            if not self.close_document(
                document_id, decision_hook=decision_hook
            ):
                return False
        return True

    def close_others(
        self,
        keep_document_id: str,
        *,
        decision_hook: Callable[[PCSDocument], Any] | None = None,
    ) -> bool:
        keep_document_id = str(keep_document_id)
        if keep_document_id not in self._documents:
            raise KeyError(f"unknown PCS document ID: {keep_document_id}")
        return self.close_documents(
            (
                document_id for document_id in tuple(self._documents)
                if document_id != keep_document_id
            ),
            decision_hook=decision_hook,
        )

    def close_all(
        self,
        *,
        decision_hook: Callable[[PCSDocument], Any] | None = None,
    ) -> bool:
        return self.close_documents(
            tuple(self._documents), decision_hook=decision_hook
        )


__all__ = ["ActiveDocumentChanged", "DocumentManager"]
