"""Explicit PCS-document affinity for long-lived NMR Workbench windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from logic.pcs_document import active_document, document_by_id


class InactivePCSDocumentError(RuntimeError):
    """Raised rather than silently retargeting a bound Workbench window."""


@dataclass
class PCSDocumentAffinity:
    app_state: dict[str, Any]
    document_id: str

    def resolve_document(self):
        document = document_by_id(self.app_state, self.document_id)
        if document is None:
            raise InactivePCSDocumentError(
                "The PCS document bound to this NMR Workbench is unavailable."
            )
        return document

    def resolve_state(self) -> dict[str, Any]:
        """Return the legacy surface only while it represents this document."""
        document = active_document(self.app_state)
        if document is None or document.document_id != self.document_id:
            raise InactivePCSDocumentError(
                "The NMR Workbench is bound to a different PCS document."
            )
        return self.app_state

    def rebind_to_active(self):
        """Explicitly replace the pinned ID with the current active document."""
        document = active_document(self.app_state)
        if document is None:
            raise InactivePCSDocumentError("No active PCS document is available.")
        self.document_id = document.document_id
        return document


def bind_to_active_pcs_document(
    state: dict[str, Any],
) -> PCSDocumentAffinity | None:
    document = active_document(state)
    if document is None:
        return None
    return PCSDocumentAffinity(state, document.document_id)


__all__ = [
    "InactivePCSDocumentError",
    "PCSDocumentAffinity",
    "bind_to_active_pcs_document",
]
