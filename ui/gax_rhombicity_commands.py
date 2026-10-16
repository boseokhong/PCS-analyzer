"""Commands for the G_ax and Rhombicity contextual Ribbon surfaces."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from logic.pcs_document import active_document_view
from ui.commands import Command, CommandRegistry


GAX_RHOMBICITY_COMMAND_IDS = (
    "analysis.gax.selection.lasso",
    "analysis.gax.selection.box",
    "analysis.gax.regression.fit_selected",
    "analysis.gax.regression.force_origin",
    "analysis.gax.selection.clear",
    "analysis.gax.output.export",
    "analysis.rhombicity_check.update",
    "analysis.rhombicity_check.force_origin",
    "analysis.rhombicity_table.update",
    "analysis.rhombicity_table.calculate_tensor",
    "analysis.rhombicity_table.edit_orientation",
)


def _document_state(state: dict):
    try:
        return active_document_view(state)
    except RuntimeError:
        return state


def _plot_rows(state: dict) -> list:
    return list(state.get("cartesian_plot_rows", []) or [])


def _active_fit_row_count(state: dict) -> int:
    rows = _plot_rows(state)
    document = _document_state(state)
    selected = set(document.get("cartesian_selected_ids", set()) or set())
    if document.get("cartesian_selection_mode") == "selected" and selected:
        return sum(row.get("ref_id") in selected for row in rows)
    return len(rows)


def _selection_mode_is(state: dict, kind: str) -> bool:
    document = _document_state(state)
    return bool(document.get("cartesian_selector_active", False)) and (
        document.get("cartesian_selector_kind") == kind
    )


def _has_cartesian_selection(state: dict) -> bool:
    document = _document_state(state)
    if document.get("cartesian_selected_ids") or document.get(
        "cartesian_selector_active", False
    ):
        return True
    tree = state.get("tree")
    try:
        return bool(tree is not None and tree.selection())
    except Exception:
        return False


def _boolean_var_value(state: dict, key: str) -> bool:
    variable = state.get(key)
    try:
        return bool(variable.get())
    except Exception:
        return False


def _diagnostic_inputs_ready(state: dict) -> bool:
    document = _document_state(state)
    if not (document.get("atom_data_eff") or document.get("atom_data")):
        return False
    values = document.get("delta_exp_values", {}) or {}
    candidate_ids = set(values)
    fit_list = state.get("fit_proton_list")
    if fit_list is not None:
        try:
            selected_indices = fit_list.curselection()
            if selected_indices:
                candidate_ids = {
                    int(fit_list.get(index)) for index in selected_indices
                }
        except Exception:
            pass
    ids = (
        document.get("atom_ids_eff")
        or state.get("atom_ids")
        or document.get("atom_ids_raw")
        or ()
    )
    valid_ids = set(ids)
    return sum(
        ref_id in valid_ids and ref_id in values for ref_id in candidate_ids
    ) >= 3


def register_gax_rhombicity_commands(
    registry: CommandRegistry,
    state: dict,
    callbacks: Mapping[str, Callable[[], object]],
) -> dict[str, Command]:
    """Register M4 actions while leaving scientific state with current owners."""
    missing = [
        key
        for key in GAX_RHOMBICITY_COMMAND_IDS
        if not callable(callbacks.get(key))
    ]
    if missing:
        raise ValueError(f"Missing G_ax/Rhombicity callbacks: {', '.join(missing)}")

    commands = {
        "analysis.gax.selection.lasso": registry.create(
            "analysis.gax.selection.lasso",
            "Lasso",
            callbacks["analysis.gax.selection.lasso"],
            icon="lasso_select",
            description="Select G_ax plot points with a freeform lasso.",
            enabled=lambda: bool(_plot_rows(state)),
            checked=lambda: _selection_mode_is(state, "lasso"),
            category="G_ax Plot",
            group="Selection",
        ),
        "analysis.gax.selection.box": registry.create(
            "analysis.gax.selection.box",
            "Box",
            callbacks["analysis.gax.selection.box"],
            icon="box_select",
            description="Select G_ax plot points with a rectangular region.",
            enabled=lambda: bool(_plot_rows(state)),
            checked=lambda: _selection_mode_is(state, "rectangle"),
            category="G_ax Plot",
            group="Selection",
        ),
        "analysis.gax.regression.fit_selected": registry.create(
            "analysis.gax.regression.fit_selected",
            "Fit Selected",
            callbacks["analysis.gax.regression.fit_selected"],
            icon="run_fit",
            description="Use the current selected subset for the G_ax regression.",
            presentation="primary",
            ribbon_label="Fit",
            enabled=lambda: _active_fit_row_count(state) >= 2,
            category="G_ax Plot",
            group="Regression",
        ),
        "analysis.gax.regression.force_origin": registry.create(
            "analysis.gax.regression.force_origin",
            "Force fit through origin",
            callbacks["analysis.gax.regression.force_origin"],
            description="Constrain the G_ax regression intercept to zero.",
            ribbon_label="Force origin",
            checked=lambda: _boolean_var_value(state, "plot_force_origin_var"),
            category="G_ax Plot",
            group="Regression",
        ),
        "analysis.gax.selection.clear": registry.create(
            "analysis.gax.selection.clear",
            "Clear Selection",
            callbacks["analysis.gax.selection.clear"],
            icon="clear_selection",
            description="Clear the plot selection and return to all points.",
            enabled=lambda: _has_cartesian_selection(state),
            category="G_ax Plot",
            group="Selection",
        ),
        "analysis.gax.output.export": registry.create(
            "analysis.gax.output.export",
            "Export Plot",
            callbacks["analysis.gax.output.export"],
            icon="export",
            description="Export the current G_ax plot and its plotted data.",
            enabled=lambda: state.get("cartesian_figure") is not None,
            category="G_ax Plot",
            group="Output",
        ),
        "analysis.rhombicity_check.update": registry.create(
            "analysis.rhombicity_check.update",
            "Update Diagnostic",
            callbacks["analysis.rhombicity_check.update"],
            icon="update",
            description="Recompute the axial-fit rhombicity diagnostic.",
            presentation="primary",
            ribbon_label="Update",
            enabled=lambda: _diagnostic_inputs_ready(state),
            category="Rhombicity Check",
            group="Analysis",
        ),
        "analysis.rhombicity_check.force_origin": registry.create(
            "analysis.rhombicity_check.force_origin",
            "Force b = 0 (through origin)",
            callbacks["analysis.rhombicity_check.force_origin"],
            description="Constrain the diagnostic linear fit intercept to zero.",
            ribbon_label="Force b = 0",
            checked=lambda: _boolean_var_value(state, "diag_fit_intercept_var"),
            category="Rhombicity Check",
            group="Analysis",
        ),
        "analysis.rhombicity_table.update": registry.create(
            "analysis.rhombicity_table.update",
            "Update Table",
            callbacks["analysis.rhombicity_table.update"],
            icon="update",
            description="Apply the local rhombic input and refresh the table.",
            category="Rhombicity Table",
            group="Table",
        ),
        "analysis.rhombicity_table.calculate_tensor": registry.create(
            "analysis.rhombicity_table.calculate_tensor",
            "Calc χ(xx,yy,zz)",
            callbacks["analysis.rhombicity_table.calculate_tensor"],
            icon="calculation",
            description="Calculate tensor components from the existing susceptibility inputs.",
            ribbon_label="Calc Tensor",
            category="Rhombicity Table",
            group="Table",
        ),
        "analysis.rhombicity_table.edit_orientation": registry.create(
            "analysis.rhombicity_table.edit_orientation",
            "Edit in Control Panel",
            callbacks["analysis.rhombicity_table.edit_orientation"],
            description="Open the persistent tensor-orientation controls in the Control Panel.",
            ribbon_label="Edit Orientation",
            category="Rhombicity Table",
            group="View",
        ),
    }
    return commands


def refresh_gax_rhombicity_commands(registry: CommandRegistry) -> None:
    """Refresh only the M4 presentation state."""
    for command_id in GAX_RHOMBICITY_COMMAND_IDS:
        if command_id in registry:
            registry.refresh(command_id)


__all__ = [
    "GAX_RHOMBICITY_COMMAND_IDS",
    "refresh_gax_rhombicity_commands",
    "register_gax_rhombicity_commands",
]
