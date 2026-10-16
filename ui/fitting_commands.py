"""Command registration for standard and Advanced Fitting workspaces."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from logic.pcs_document import active_document_view
from ui.commands import Command, CommandRegistry


FITTING_COMMAND_IDS = (
    "analysis.fitting.run",
    "analysis.fitting.initialization.use_current_tensor",
    "analysis.fitting.initialization.fit_dchi_ax",
    "analysis.fitting.initialization.fit_dchi_rh",
    "analysis.fitting.initialization.global_search",
    "analysis.fitting.data.refresh",
    "analysis.fitting.results.apply_tensor",
    "analysis.fitting.results.preview_geometry",
    "analysis.fitting.results.clear_geometry",
    "analysis.fitting.results.history",
    "analysis.fitting.output.export_plot",
    "analysis.advanced.multiconformer.input.use_current",
    "analysis.advanced.multiconformer.input.load_xyz",
    "analysis.advanced.multiconformer.input.clear",
    "analysis.advanced.multiconformer.run",
    "analysis.advanced.multiconformer.results.apply",
    "analysis.advanced.rdc.input.load_csv",
    "analysis.advanced.rdc.run",
    "analysis.advanced.rdc.results.apply",
    "analysis.advanced.rdc.output.export_template",
    "analysis.advanced.pre.input.load_csv",
    "analysis.advanced.pre.run",
    "analysis.advanced.pre.output.export_template",
    "analysis.advanced.multilanthanide.input.use_current",
    "analysis.advanced.multilanthanide.input.add_csv",
    "analysis.advanced.multilanthanide.input.clear",
    "analysis.advanced.multilanthanide.run",
    "analysis.advanced.multiconformer.input.remove",
    "analysis.advanced.multiconformer.input.configure",
    "analysis.advanced.multiconformer.input.observations",
    "analysis.advanced.multiconformer.input.validate",
    "analysis.advanced.multiconformer.parameters.settings",
    "analysis.advanced.multiconformer.parameters.fit_ax",
    "analysis.advanced.multiconformer.parameters.fit_rh",
    "analysis.advanced.multiconformer.output.export",
    "analysis.advanced.multiconformer.results.reliability",
    "analysis.advanced.rdc.input.observations",
    "analysis.advanced.rdc.input.validate",
    "analysis.advanced.rdc.parameters.settings",
    "analysis.advanced.rdc.output.export",
    "analysis.advanced.pre.input.validate",
    "analysis.advanced.pre.output.export",
    "analysis.advanced.multilanthanide.input.remove",
    "analysis.advanced.multilanthanide.input.validate",
    "analysis.advanced.multilanthanide.results.transfer",
    "analysis.advanced.multilanthanide.output.export",
    "analysis.advanced.multilanthanide.output.template",
)


def _document_state(state: dict):
    try:
        return active_document_view(state)
    except RuntimeError:
        return state


def _has_structure(state: dict) -> bool:
    document = _document_state(state)
    return bool(document.get("atom_data_eff") or document.get("atom_data"))


def _has_delta_exp(state: dict) -> bool:
    return bool(_document_state(state).get("delta_exp_values", {}) or {})


def _document_value(state: dict, key: str, default=None):
    return _document_state(state).get(key, default)


def _has_listbox_selection(state: dict, key: str) -> bool:
    widget = state.get(key)
    try:
        return bool(widget is not None and widget.curselection())
    except Exception:
        return False


def _standard_fit_ready(state: dict) -> bool:
    if state.get("fit_is_running", False):
        return False
    if not _has_listbox_selection(state, "fit_proton_list"):
        return False
    mode_var = state.get("fit_mode_var")
    try:
        mode = mode_var.get()
    except Exception:
        mode = None
    return mode != "theta_alpha_multi" or _has_listbox_selection(
        state, "fit_donor_list"
    )


def _owner_predicate(state: dict, key: str) -> bool:
    supplier = state.get(key)
    return bool(supplier()) if callable(supplier) else False


def _boolean_var_value(state: dict, key: str) -> bool:
    variable = state.get(key)
    try:
        return bool(variable.get())
    except Exception:
        return False


def _current_tensor_initialization_supported(state: dict) -> bool:
    variable = state.get("fit_mode_var")
    try:
        return variable.get() in ("euler_global", "full_tensor")
    except Exception:
        return False


def register_fitting_commands(
    registry: CommandRegistry,
    state: dict,
    callbacks: Mapping[str, Callable[[], object]],
) -> dict[str, Command]:
    """Register M5 actions without taking ownership of fitting state."""
    missing = [
        key for key in FITTING_COMMAND_IDS if not callable(callbacks.get(key))
    ]
    if missing:
        raise ValueError(f"Missing Fitting callbacks: {', '.join(missing)}")

    commands = {
        "analysis.fitting.run": registry.create(
            "analysis.fitting.run", "Run Fit", callbacks["analysis.fitting.run"],
            icon="run_fit", description="Run the configured standard PCS fit.",
            presentation="primary",
            ribbon_label="Run Fit",
            enabled=lambda: _standard_fit_ready(state),
            category="Fitting", group="Fitting",
        ),
        "analysis.fitting.initialization.use_current_tensor": registry.create(
            "analysis.fitting.initialization.use_current_tensor",
            "Use current tensor as initial guess",
            callbacks["analysis.fitting.initialization.use_current_tensor"],
            description="Use the current tensor as the Mode B or Mode C initial guess.",
            ribbon_label="Use current tensor",
            enabled=lambda: _current_tensor_initialization_supported(state),
            checked=lambda: _boolean_var_value(
                state, "fit_use_current_tensor_var"
            ),
            category="Fitting", group="Initialization",
        ),
        "analysis.fitting.initialization.fit_dchi_ax": registry.create(
            "analysis.fitting.initialization.fit_dchi_ax",
            "Fit Δχ_ax",
            callbacks["analysis.fitting.initialization.fit_dchi_ax"],
            description="Include the axial susceptibility component in the fit.",
            checked=lambda: _boolean_var_value(state, "fit_dchi_var"),
            category="Fitting", group="Initialization",
        ),
        "analysis.fitting.initialization.fit_dchi_rh": registry.create(
            "analysis.fitting.initialization.fit_dchi_rh",
            "Fit Δχ_rh",
            callbacks["analysis.fitting.initialization.fit_dchi_rh"],
            description="Include the rhombic susceptibility component in the fit.",
            checked=lambda: _boolean_var_value(state, "fit_dchi_rh_var"),
            category="Fitting", group="Initialization",
        ),
        "analysis.fitting.initialization.global_search": registry.create(
            "analysis.fitting.initialization.global_search",
            "Global search (DE)",
            callbacks["analysis.fitting.initialization.global_search"],
            description="Use differential evolution for global initialization.",
            ribbon_label="Global search",
            checked=lambda: _boolean_var_value(state, "fit_global_search_var"),
            category="Fitting", group="Initialization",
        ),
        "analysis.fitting.data.refresh": registry.create(
            "analysis.fitting.data.refresh", "Refresh Inputs",
            callbacks["analysis.fitting.data.refresh"], icon="update",
            description="Refresh fitting observations and donor choices from the main state.",
            ribbon_label="Refresh",
            enabled=lambda: _has_structure(state),
            category="Fitting", group="Data",
        ),
        "analysis.fitting.results.apply_tensor": registry.create(
            "analysis.fitting.results.apply_tensor", "Apply to Tensor",
            callbacks["analysis.fitting.results.apply_tensor"], icon="apply",
            description="Apply a compatible fit result to the canonical tensor.",
            ribbon_label="Apply Tensor",
            enabled=lambda: _owner_predicate(state, "can_apply_fit_result"),
            category="Fitting", group="Results",
        ),
        "analysis.fitting.results.preview_geometry": registry.create(
            "analysis.fitting.results.preview_geometry", "Preview Geometry",
            callbacks["analysis.fitting.results.preview_geometry"], icon="eye",
            description="Temporarily preview the fitted geometry in existing views.",
            ribbon_label="Preview",
            enabled=lambda: _owner_predicate(state, "can_preview_fit_result"),
            category="Fitting", group="Results",
        ),
        "analysis.fitting.results.clear_geometry": registry.create(
            "analysis.fitting.results.clear_geometry", "Clear Geometry Preview",
            callbacks["analysis.fitting.results.clear_geometry"], icon="clear",
            description="Remove the temporary fitted geometry; Undo restores it.",
            ribbon_label="Clear Preview",
            enabled=lambda: bool(state.get("fit_override")),
            category="Fitting", group="Results",
        ),
        "analysis.fitting.results.history": registry.create(
            "analysis.fitting.results.history", "Fit History",
            callbacks["analysis.fitting.results.history"], icon="fit_history",
            description="Open the existing Fit History manager.",
            category="Fitting", group="Results",
        ),
        "analysis.fitting.output.export_plot": registry.create(
            "analysis.fitting.output.export_plot", "Export Fit Plot",
            callbacks["analysis.fitting.output.export_plot"], icon="export",
            description="Export the current standard-fit correlation plot.",
            ribbon_label="Export Plot",
            enabled=lambda: state.get("fit_corr_fig") is not None,
            category="Fitting", group="Output",
        ),
        "analysis.advanced.multiconformer.input.use_current": registry.create(
            "analysis.advanced.multiconformer.input.use_current",
            "Use Current Structure",
            callbacks["analysis.advanced.multiconformer.input.use_current"], icon="use_existing_data",
            description="Add the current main structure as a conformer.",
            ribbon_label="Use Current",
            enabled=lambda: _has_structure(state),
            category="Multi-Conformer", group="Input",
        ),
        "analysis.advanced.multiconformer.input.load_xyz": registry.create(
            "analysis.advanced.multiconformer.input.load_xyz", "Load XYZ",
            callbacks["analysis.advanced.multiconformer.input.load_xyz"],
            icon="open_folder", description="Load another conformer structure.",
            enabled=lambda: _has_structure(state),
            category="Multi-Conformer", group="Input",
        ),
        "analysis.advanced.multiconformer.input.clear": registry.create(
            "analysis.advanced.multiconformer.input.clear", "Clear",
            callbacks["analysis.advanced.multiconformer.input.clear"], icon="close",
            description="Clear the current conformer collection.",
            presentation="danger",
            enabled=lambda: bool(_document_value(state, "advfit_conformers")),
            category="Multi-Conformer", group="Input",
        ),
        "analysis.advanced.multiconformer.run": registry.create(
            "analysis.advanced.multiconformer.run", "Run Fit",
            callbacks["analysis.advanced.multiconformer.run"], icon="run_fit",
            description="Run the multi-conformer fit.",
            presentation="primary",
            enabled=lambda: bool(
                state["can_run_mc_fit"]() if callable(state.get("can_run_mc_fit")) else (
                not state.get("advfit_mc_running", False)
                and len(_document_value(state, "advfit_conformers", ()) or ()) >= 2
                and _has_delta_exp(state)
                and _has_structure(state))
            ),
            category="Multi-Conformer", group="Fitting",
        ),
        "analysis.advanced.multiconformer.results.apply": registry.create(
            "analysis.advanced.multiconformer.results.apply", "Apply to Main",
            callbacks["analysis.advanced.multiconformer.results.apply"], icon="apply",
            description="Apply the fitted multi-conformer tensor to the main state.",
            ribbon_label="Apply",
            enabled=lambda: (_owner_predicate(state, "can_apply_mc_result")
                             if callable(state.get("can_apply_mc_result"))
                             else bool(_document_value(state, "advfit_mc_last_result"))),
            category="Multi-Conformer", group="Results",
        ),
        "analysis.advanced.rdc.input.load_csv": registry.create(
            "analysis.advanced.rdc.input.load_csv", "Load RDC CSV",
            callbacks["analysis.advanced.rdc.input.load_csv"], icon="import",
            description="Load the RDC observations for a joint fit.",
            category="Joint PCS+RDC", group="Input",
        ),
        "analysis.advanced.rdc.run": registry.create(
            "analysis.advanced.rdc.run", "Run Fit",
            callbacks["analysis.advanced.rdc.run"], icon="run_fit",
            description="Run the configured joint PCS and RDC fit.",
            presentation="primary",
            enabled=lambda: bool(
                state["can_run_rdc_fit"]() if callable(state.get("can_run_rdc_fit")) else (
                not state.get("advfit_rdc_running", False)
                and _document_value(state, "advfit_rdc_rows")
                and _has_delta_exp(state)
                and _has_structure(state))
            ),
            category="Joint PCS+RDC", group="Fitting",
        ),
        "analysis.advanced.rdc.results.apply": registry.create(
            "analysis.advanced.rdc.results.apply", "Apply to Main",
            callbacks["analysis.advanced.rdc.results.apply"], icon="apply",
            description="Apply the fitted joint tensor to the main state.",
            ribbon_label="Apply",
            enabled=lambda: (_owner_predicate(state, "can_apply_rdc_result")
                             if callable(state.get("can_apply_rdc_result"))
                             else bool(_document_value(state, "advfit_rdc_last_result"))),
            category="Joint PCS+RDC", group="Results",
        ),
        "analysis.advanced.rdc.output.export_template": registry.create(
            "analysis.advanced.rdc.output.export_template", "Export Template",
            callbacks["analysis.advanced.rdc.output.export_template"], icon="export",
            description="Export the existing RDC CSV template.",
            category="Joint PCS+RDC", group="Output",
        ),
        "analysis.advanced.pre.input.load_csv": registry.create(
            "analysis.advanced.pre.input.load_csv", "Load PRE CSV",
            callbacks["analysis.advanced.pre.input.load_csv"], icon="import",
            description="Load PRE observations.",
            category="PRE", group="Input",
        ),
        "analysis.advanced.pre.run": registry.create(
            "analysis.advanced.pre.run", "Run Fit",
            callbacks["analysis.advanced.pre.run"], icon="run_fit",
            description="Run the configured PRE fit.",
            presentation="primary",
            enabled=lambda: (_owner_predicate(state, "can_run_pre_fit")
                             if callable(state.get("can_run_pre_fit")) else bool(
                not state.get("advfit_pre_running", False)
                and _document_value(state, "advfit_pre_obs")
                and _has_structure(state)
            )),
            category="PRE", group="Fitting",
        ),
        "analysis.advanced.pre.output.export_template": registry.create(
            "analysis.advanced.pre.output.export_template", "Export Template",
            callbacks["analysis.advanced.pre.output.export_template"], icon="export",
            description="Export the existing PRE CSV template.",
            category="PRE", group="Output",
        ),
        "analysis.advanced.multilanthanide.input.use_current": registry.create(
            "analysis.advanced.multilanthanide.input.use_current",
            "Use Current δ_Exp",
            callbacks["analysis.advanced.multilanthanide.input.use_current"], icon="use_existing_data",
            description="Add the current experimental PCS dataset.",
            ribbon_label="Use Current",
            enabled=lambda: _has_delta_exp(state),
            category="Multi-Lanthanide", group="Input",
        ),
        "analysis.advanced.multilanthanide.input.add_csv": registry.create(
            "analysis.advanced.multilanthanide.input.add_csv", "Add CSV",
            callbacks["analysis.advanced.multilanthanide.input.add_csv"], icon="import",
            description="Add an experimental PCS dataset from CSV.",
            category="Multi-Lanthanide", group="Input",
        ),
        "analysis.advanced.multilanthanide.input.clear": registry.create(
            "analysis.advanced.multilanthanide.input.clear", "Clear",
            callbacks["analysis.advanced.multilanthanide.input.clear"], icon="clear",
            description="Clear the multi-lanthanide datasets.",
            presentation="danger",
            enabled=lambda: bool(_document_value(state, "advfit_ml_datasets")),
            category="Multi-Lanthanide", group="Input",
        ),
        "analysis.advanced.multilanthanide.run": registry.create(
            "analysis.advanced.multilanthanide.run", "Run Analysis",
            callbacks["analysis.advanced.multilanthanide.run"], icon="run_fit",
            description="Run the selected Multi-Lanthanide analysis.",
            presentation="primary",
            enabled=lambda: (_owner_predicate(state, "can_run_ml_analysis") if callable(state.get("can_run_ml_analysis")) else bool(
                not state.get("advfit_ml_running", False)
                and len(_document_value(state, "advfit_ml_datasets", ()) or ()) >= 2
                and _has_structure(state)
            )),
            category="Multi-Lanthanide", group="Fitting",
        ),
    }
    def row_selected():
        tree = state.get("advfit_mc_tree")
        return bool(tree is not None and tree.selection())

    additions = (
        ("input.remove", "Remove selected", "Input", "Remove the selected conformer", "clear"),
        ("input.configure", "Map / frame", "Input", "Configure the selected conformer's explicit mapping and metal/frame", "settings"),
        ("input.observations", "Select PCS", "Data", "Select observation IDs without changing main atom selection", "select_check"),
        ("input.validate", "Validate inputs", "Data", "Check identity, frame and population constraints", "check"),
        ("parameters.settings", "Fit settings", "Parameters", "Configure target, fixed values, alignment, objective and multi-start", "settings"),
        ("parameters.fit_ax", "Fit Δχ_ax", "Parameters", "Fit axial magnitude or preserve its specified fixed value", None),
        ("parameters.fit_rh", "Fit Δχ_rh", "Parameters", "Fit rhombic magnitude or preserve its specified fixed value", None),
        ("output.export", "Export result", "Output", "Export recorded inputs, runs and conformer contributions", "export"),
        ("results.reliability", "Reliability", "Results", "Refit bootstrap, grouped cross-validation and population diagnostics", "compare"),
    )
    for suffix, label, group, description, icon in additions:
        command_id = "analysis.advanced.multiconformer." + suffix
        enabled = (row_selected if suffix in ("input.remove", "input.configure") else
                   lambda: not state.get("advfit_mc_running", False))
        if suffix == "output.export":
            enabled = lambda: bool(_document_value(state, "advfit_mc_last_result"))
        elif suffix == "results.reliability":
            enabled = lambda: bool(state.get("mc_workspace") and state["mc_workspace"].can_evaluate())
        checked = None
        if suffix in ("parameters.fit_ax", "parameters.fit_rh"):
            key = "fit_ax" if suffix.endswith("fit_ax") else "fit_rh"
            checked = lambda key=key: bool((_document_value(state, "advfit_mc_options", {}) or {}).get(key, key == "fit_ax"))
        commands[command_id] = registry.create(command_id, label, callbacks[command_id],
                                                icon=icon, description=description, category="Multi-Conformer",
                                                group=group, enabled=enabled, checked=checked)
    for suffix, label, group, icon in (
        ("input.observations", "Select PCS", "Input", "select_check"),
        ("input.validate", "Validate inputs", "Validation", "check"),
        ("parameters.settings", "Fit settings", "Parameters", "settings"),
        ("output.export", "Export result", "Output", "export"),
    ):
        command_id = "analysis.advanced.rdc." + suffix
        commands[command_id] = registry.create(command_id, label, callbacks[command_id],
            icon=icon, category="Joint PCS+RDC", group=group,
            enabled=(lambda: bool(_document_value(state, "advfit_rdc_last_result"))) if suffix == "output.export"
                    else (lambda: not state.get("advfit_rdc_running", False)))
    for suffix, label, group, icon in (
        ("input.validate", "Validate inputs", "Validation", "check"),
        ("output.export", "Export result", "Output", "export"),
    ):
        command_id = "analysis.advanced.pre." + suffix
        commands[command_id] = registry.create(command_id, label, callbacks[command_id],
            icon=icon, category="PRE", group=group,
            enabled=(lambda: bool(_document_value(state, "advfit_pre_last_result"))) if suffix == "output.export"
                    else (lambda: not state.get("advfit_pre_running", False)))
    for suffix, label, group, icon in (
        ("input.remove", "Remove selected", "Input", "clear"),
        ("input.validate", "Validate inputs", "Validation", "check"),
        ("results.transfer", "Review PCS transfer", "Results", "use_existing_data"),
        ("output.export", "Export result", "Output", "export"),
        ("output.template", "Export Template", "Output", "export"),
    ):
        command_id = "analysis.advanced.multilanthanide." + suffix
        commands[command_id] = registry.create(command_id, label, callbacks[command_id],
            icon=icon, category="Multi-Lanthanide", group=group,
            enabled=lambda: not state.get("advfit_ml_running", False))
    return commands


def refresh_fitting_commands(registry: CommandRegistry) -> None:
    """Refresh only standard and Advanced Fitting command state."""
    for command_id in FITTING_COMMAND_IDS:
        if command_id in registry:
            registry.refresh(command_id)


__all__ = [
    "FITTING_COMMAND_IDS",
    "refresh_fitting_commands",
    "register_fitting_commands",
]
