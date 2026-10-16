"""PCS Analyzer window lifecycle adapter."""

from __future__ import annotations

from typing import Any

from nmr_workbench.integration.pcs_document_affinity import (
    bind_to_active_pcs_document,
)


def open_nmr_workbench(state: dict[str, Any]) -> None:
    """Open or focus the single NMR Workbench window owned by app state."""

    from tkinter import messagebox

    from nmr_workbench.ui import NMRWorkbenchWindow

    window = state.get("nmr_workbench_window")
    if window is not None:
        try:
            if window.winfo_exists():
                refresh = getattr(window, "refresh_pcs_document_binding", None)
                if not callable(refresh):
                    refresh = getattr(window, "refresh_from_pcs_state", None)
                if callable(refresh):
                    refresh()
                window.lift()
                window.focus_force()
                return
        except Exception:
            pass
    try:
        affinity = bind_to_active_pcs_document(state)
        window = NMRWorkbenchWindow(state["root"], pcs_state=state)
        window._pcs_document_affinity = affinity
        window.bound_pcs_document_id = (
            affinity.document_id if affinity is not None else None
        )
        refresh_binding = getattr(window, "refresh_pcs_document_binding", None)
        if callable(refresh_binding):
            refresh_binding()
        elif affinity is not None:
            from logic.pcs_document import document_display_name

            document = affinity.resolve_document()
            set_title = getattr(window, "title", None)
            if callable(set_title):
                set_title(
                    f"NMR Workbench — PCS: {document_display_name(document)}"
                )
        state["nmr_workbench_window"] = window

        def close() -> None:
            try:
                window.destroy()
            finally:
                state["nmr_workbench_window"] = None

        window.protocol("WM_DELETE_WINDOW", close)
        window.lift()
        window.focus_force()
    except Exception as exc:
        messagebox.showerror("NMR Workbench", f"Could not open NMR Workbench:\n{exc}")

