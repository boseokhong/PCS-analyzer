"""Command registration for the persistent PCS experimental-data workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from logic.pcs_document import active_document_view
from ui.commands import Command, CommandRegistry


PCS_DATA_COMMAND_IDS = (
    "pcs.experimental.export_template",
    "pcs.experimental.import",
    "pcs.experimental.paste",
    "pcs.experimental.undo",
    "pcs.experimental.clear",
)


def _document_state(state: dict):
    try:
        return active_document_view(state)
    except RuntimeError:
        return state


def _has_experimental_data(state: dict) -> bool:
    return bool(_document_state(state).get("delta_exp_values", {}) or {})


def _has_undo_snapshot(state: dict) -> bool:
    return _document_state(state).get("_delta_exp_undo") is not None


def register_pcs_data_commands(
    registry: CommandRegistry,
    state: dict,
    callbacks: Mapping[str, Callable[[], object]],
) -> dict[str, Command]:
    """Register the five persistent PCS-data actions using existing callbacks.

    ``callbacks`` is keyed by the stable IDs in ``PCS_DATA_COMMAND_IDS``. The
    caller remains responsible for adapting the existing callback signatures;
    no data or scientific logic is implemented here.
    """
    missing = [key for key in PCS_DATA_COMMAND_IDS if not callable(callbacks.get(key))]
    if missing:
        raise ValueError(f"Missing PCS-data callbacks: {', '.join(missing)}")

    commands = {
        "pcs.experimental.export_template": registry.create(
            "pcs.experimental.export_template",
            "Export δ_Exp Template",
            callbacks["pcs.experimental.export_template"],
            icon="export",
            description="Export a CSV/XLSX template for the visible PCS table rows.",
            ribbon_label="Export Template",
            presentation="primary",
            category="Data",
            group="Experimental PCS",
        ),
        "pcs.experimental.import": registry.create(
            "pcs.experimental.import",
            "Import δ_Exp",
            callbacks["pcs.experimental.import"],
            icon="import",
            description="Import experimental shift values from CSV or Excel.",
            ribbon_label="Import PCS",
            presentation="primary",
            category="Data",
            group="Experimental PCS",
        ),
        "pcs.experimental.paste": registry.create(
            "pcs.experimental.paste",
            "Paste δ_Exp",
            callbacks["pcs.experimental.paste"],
            icon="paste",
            description="Paste experimental shift values and preview the mapping.",
            ribbon_label="Paste",
            category="Data",
            group="Experimental PCS",
        ),
        "pcs.experimental.undo": registry.create(
            "pcs.experimental.undo",
            "Undo",
            callbacks["pcs.experimental.undo"],
            icon="undo",
            description="Restore the previous imported experimental-shift state.",
            enabled=lambda: _has_undo_snapshot(state),
            category="Data",
            group="Experimental PCS",
        ),
        "pcs.experimental.clear": registry.create(
            "pcs.experimental.clear",
            "Clear δ_Exp",
            callbacks["pcs.experimental.clear"],
            icon="clear",
            description="Clear all experimental shift values from the PCS table.",
            ribbon_label="Clear",
            presentation="danger",
            enabled=lambda: _has_experimental_data(state),
            category="Data",
            group="Experimental PCS",
        ),
    }
    return commands


def refresh_pcs_data_commands(registry: CommandRegistry) -> None:
    """Refresh registered PCS-data readiness without affecting other commands."""
    for command_id in PCS_DATA_COMMAND_IDS:
        if command_id in registry:
            registry.refresh(command_id)


__all__ = [
    "PCS_DATA_COMMAND_IDS",
    "refresh_pcs_data_commands",
    "register_pcs_data_commands",
]
