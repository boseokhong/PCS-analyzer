"""Command registration for the embedded Conformer Search workspace."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from logic.pcs_document import active_document_view
from ui.commands import Command, CommandRegistry


CONFORMER_COMMAND_IDS = (
    "analysis.conformer.fixed.apply_filter",
    "analysis.conformer.fixed.fix_all_shown",
    "analysis.conformer.fixed.unfix_all",
    "analysis.conformer.fixed.invert",
    "analysis.conformer.fixed.apply_locked_bonds",
    "analysis.conformer.bonds.select_all_free",
    "analysis.conformer.bonds.select_none",
    "analysis.conformer.bonds.invert",
    "analysis.conformer.bonds.skipped_contacts",
    "analysis.conformer.bonds.add",
    "analysis.conformer.bonds.remove",
    "analysis.conformer.setup.sync_main",
    "analysis.conformer.search.run",
    "analysis.conformer.results.view_structure",
    "analysis.conformer.results.select_candidate",
    "analysis.conformer.results.correlation_plot",
    "analysis.conformer.output.save_xyz",
    "analysis.conformer.output.save_report",
    "analysis.conformer.candidate.apply",
    "analysis.conformer.candidate.revert",
    "analysis.conformer.candidate.discard",
)


def _document_state(state: dict):
    try:
        return active_document_view(state)
    except RuntimeError:
        return state


def _main_inputs_ready(state: dict) -> bool:
    document = _document_state(state)
    return bool(
        (document.get("atom_data_raw") or document.get("atom_data"))
        and (document.get("delta_exp_values", {}) or {})
    )


def register_conformer_commands(
    registry: CommandRegistry,
    state: dict,
    adapter,
    callbacks: Mapping[str, Callable[[], object]],
) -> dict[str, Command]:
    """Register Conformer actions while retaining existing state owners."""
    missing = [
        command_id
        for command_id in CONFORMER_COMMAND_IDS
        if not callable(callbacks.get(command_id))
    ]
    if missing:
        raise ValueError(f"Missing Conformer callbacks: {', '.join(missing)}")

    def create(
        command_id,
        label,
        *,
        icon=None,
        description,
        enabled,
        group,
        presentation="secondary",
        ribbon_label=None,
    ):
        return registry.create(
            command_id,
            label,
            callbacks[command_id],
            icon=icon,
            description=description,
            enabled=enabled,
            category="Conformer Search",
            group=group,
            presentation=presentation,
            ribbon_label=ribbon_label,
        )

    commands = {
        "analysis.conformer.fixed.apply_filter": create(
            "analysis.conformer.fixed.apply_filter", "Apply Filter",
            icon="filter", description="Apply the local element filter to the atom checklist.",
            enabled=adapter.fixed_actions_ready, group="Selection",
        ),
        "analysis.conformer.fixed.fix_all_shown": create(
            "analysis.conformer.fixed.fix_all_shown", "Fix All Shown",
            icon="select_check", description="Mark every currently shown atom as fixed.",
            enabled=adapter.fixed_actions_ready, group="Selection",
        ),
        "analysis.conformer.fixed.unfix_all": create(
            "analysis.conformer.fixed.unfix_all", "Unfix All",
            icon="clear_selection", description="Unfix every currently shown atom.",
            enabled=adapter.fixed_actions_ready, group="Selection",
        ),
        "analysis.conformer.fixed.invert": create(
            "analysis.conformer.fixed.invert", "Invert",
            icon="invert", description="Invert fixed state for the currently shown atoms.",
            enabled=adapter.fixed_actions_ready, group="Selection",
        ),
        "analysis.conformer.fixed.apply_locked_bonds": create(
            "analysis.conformer.fixed.apply_locked_bonds", "Apply & Show Locked Bonds",
            icon="lock", description="Apply fixed atoms, rebuild bond locks, and show the bond step.",
            enabled=adapter.lock_geometry_ready, group="Geometry",
            ribbon_label="Apply Locks", presentation="primary",
        ),
        "analysis.conformer.bonds.select_all_free": create(
            "analysis.conformer.bonds.select_all_free", "Select All Free",
            icon="select_check", description="Select every currently free rotatable bond.",
            enabled=adapter.bond_selection_ready, group="Selection",
        ),
        "analysis.conformer.bonds.select_none": create(
            "analysis.conformer.bonds.select_none", "Select None",
            icon="clear_selection", description="Clear all rotatable-bond selections.",
            enabled=adapter.bond_selection_ready, group="Selection",
        ),
        "analysis.conformer.bonds.invert": create(
            "analysis.conformer.bonds.invert", "Invert",
            icon="invert", description="Invert the free rotatable-bond selection.",
            enabled=adapter.bond_selection_ready, group="Selection",
        ),
        "analysis.conformer.bonds.skipped_contacts": create(
            "analysis.conformer.bonds.skipped_contacts", "Skipped Contacts",
            icon="caution", description="Show the existing skipped-contact diagnostic dialog.",
            enabled=adapter.has_structure, group="Geometry",
        ),
        "analysis.conformer.bonds.add": create(
            "analysis.conformer.bonds.add", "Add Bond",
            icon="add", description="Open the existing dialog for adding a bond.",
            enabled=adapter.has_structure, group="Geometry",
        ),
        "analysis.conformer.bonds.remove": create(
            "analysis.conformer.bonds.remove", "Remove Bond",
            icon="remove", description="Open the existing dialog for removing a bond.",
            enabled=adapter.has_structure, group="Geometry",
        ),
        "analysis.conformer.setup.sync_main": create(
            "analysis.conformer.setup.sync_main", "Sync from Main",
            icon="update", description="Refresh structure, PCS data, tensor, and orientation from the main window.",
            enabled=lambda: _main_inputs_ready(state), group="Setup",
            ribbon_label="Sync Main", presentation="primary",
        ),
        "analysis.conformer.search.run": create(
            "analysis.conformer.search.run", "Run Search",
            icon="run", description="Run the configured conformer search.",
            enabled=adapter.run_ready, group="Search", presentation="primary",
        ),
        "analysis.conformer.results.view_structure": create(
            "analysis.conformer.results.view_structure", "View Structure",
            icon="eye", description="Open the existing candidate structure viewer.",
            enabled=adapter.results_available, group="Results",
            ribbon_label="View", presentation="primary",
        ),
        "analysis.conformer.results.select_candidate": create(
            "analysis.conformer.results.select_candidate", "Select Candidate",
            icon="point_select", description="Select a search result and send it to the main window as a preview; this does not commit it.",
            enabled=adapter.results_available, group="Results",
            ribbon_label="Select",
        ),
        "analysis.conformer.results.correlation_plot": create(
            "analysis.conformer.results.correlation_plot", "Correlation Plot",
            icon="compare", description="Open the existing PCS correlation and residual plot.",
            enabled=adapter.results_available, group="Display",
            ribbon_label="Correlation",
        ),
        "analysis.conformer.output.save_xyz": create(
            "analysis.conformer.output.save_xyz", "Save XYZ",
            icon="save", description="Export selected conformer candidates as XYZ.",
            enabled=adapter.results_available, group="Output",
        ),
        "analysis.conformer.output.save_report": create(
            "analysis.conformer.output.save_report", "Save Report",
            icon="export", description="Export the active candidate report.",
            enabled=adapter.results_available, group="Output",
        ),
        "analysis.conformer.candidate.apply": create(
            "analysis.conformer.candidate.apply", "Apply Candidate",
            icon="apply", description="Commit the current preview candidate to the main structure.",
            enabled=lambda: bool(_document_state(state).get("conformer_preview")) and not _document_state(state).get("conformer_applied", False)
                and (state.get("conformer_search_project_adapter") is None or state["conformer_search_project_adapter"].can_apply()),
            group="Candidate",
            ribbon_label="Apply",
        ),
        "analysis.conformer.candidate.revert": create(
            "analysis.conformer.candidate.revert", "Revert",
            icon="reset", description="Restore the original structure after a candidate was applied.",
            enabled=lambda: bool(_document_state(state).get("atom_data_original")) and bool(_document_state(state).get("conformer_applied", False)),
            group="Candidate",
        ),
        "analysis.conformer.candidate.discard": create(
            "analysis.conformer.candidate.discard", "Discard Preview",
            icon="delete", description="Discard the current preview without committing it.",
            enabled=lambda: bool(_document_state(state).get("conformer_preview")) and not _document_state(state).get("conformer_applied", False),
            group="Candidate", presentation="danger",
            ribbon_label="Discard",
        ),
    }
    return commands


def refresh_conformer_commands(registry: CommandRegistry) -> None:
    """Refresh only Conformer Search presentation state."""
    for command_id in CONFORMER_COMMAND_IDS:
        if command_id in registry:
            registry.refresh(command_id)


__all__ = [
    "CONFORMER_COMMAND_IDS",
    "refresh_conformer_commands",
    "register_conformer_commands",
]
