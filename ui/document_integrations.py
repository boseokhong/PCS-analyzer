"""Presentation-only document integration for built-in secondary windows."""

from __future__ import annotations

from typing import Any

from logic.pcs_document import active_document, document_display_name


def active_document_title(state: dict[str, Any], prefix: str) -> str:
    return f"{prefix} — {document_display_name(active_document(state))}"


def _set_tk_title(window: Any, value: str) -> None:
    if window is None:
        return
    try:
        if window.winfo_exists():
            window.title(value)
    except Exception:
        pass


def _set_pyvista_title(plotter: Any, value: str) -> None:
    if plotter is None:
        return
    try:
        plotter.ren_win.SetWindowName(value)
    except Exception:
        pass


def refresh_document_integration_titles(state: dict[str, Any]) -> None:
    """Refresh titles/status without recreating or retargeting any window."""
    name = document_display_name(active_document(state))
    for key, prefix in (
        ("pcs_plot_popup", "2D PCS"),
        ("projection_popup", "Projection"),
        ("plot3d_popup", "3D Structure (Matplotlib)"),
        ("chemical_map_win", "Chemical Map"),
    ):
        _set_tk_title(state.get(key), f"{prefix} — {name}")

    _set_tk_title(
        state.get("pyvista_field_ctrl_win"), f"3D PCS — {name} — Controls"
    )
    _set_pyvista_title(state.get("pyvista_field_plotter"), f"3D PCS — {name}")

    workbench = state.get("nmr_workbench_window")
    refresh_binding = getattr(workbench, "refresh_pcs_document_binding", None)
    if callable(refresh_binding):
        refresh_binding()

    isolated_viewer = state.get("_isolated_molecule_viewer")
    if isolated_viewer is not None:
        isolated_viewer.request_refresh()


__all__ = ["active_document_title", "refresh_document_integration_titles"]
