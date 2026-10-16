"""Shared commands for the stable main-window Ribbon pages and menus.

The commands in this module are presentation adapters only.  Their callbacks
remain the existing application functions supplied by ``ui.components``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ui.commands import Command, CommandRegistry
from ui.keyboard_shortcuts import MENU_SHORTCUTS


MAIN_WINDOW_COMMAND_IDS = (
    "app.file.new_document",
    "app.file.close_document",
    "app.file.next_document",
    "app.file.previous_document",
    "app.help.documentation",
    "app.help.shortcuts",
    "app.file.load_xyz",
    "app.file.open_project",
    "app.file.save_project",
    "app.file.save_project_as",
    "app.file.save_all",
    "app.file.open_workspace",
    "app.file.save_workspace",
    "app.file.save_workspace_as",
    "app.file.save_xyz",
    "app.file.save_plot_data",
    "app.file.save_table",
    "app.view.pcs_2d",
    "app.view.projection",
    "app.view.structure_3d",
    "app.view.chemical_map",
    "app.view.pcs_3d",
    "app.view.reset_control_panel",
    "app.tools.nmr_workbench",
    "app.tools.pcs_workbench",
    "app.tools.module_manager",
    "app.tools.settings",
)


def register_main_window_commands(
    registry: CommandRegistry,
    callbacks: Mapping[str, Callable[[], object]],
) -> dict[str, Command]:
    """Register stable-page commands over authoritative existing callbacks."""

    definitions = (
        ("app.file.new_document", "New PCS Document", "New Document", "open_folder", "File", "Documents", "secondary", "Create a new PCS document tab."),
        ("app.file.close_document", "Close PCS Document", "Close Document", "clear", "File", "Documents", "secondary", "Close the current document with its existing unsaved-change check."),
        ("app.file.next_document", "Next Document", "Next Document", None, "View", "Documents", "secondary", "Select the next document in tab order, wrapping at the end."),
        ("app.file.previous_document", "Previous Document", "Previous Document", None, "View", "Documents", "secondary", "Select the previous document in tab order, wrapping at the beginning."),
        ("app.help.documentation", "Documentation", "Documentation", None, "Help", "Help", "secondary", "Open the online PCS Analyzer manual in the default browser."),
        ("app.help.shortcuts", "Keyboard Shortcuts...", "Keyboard Shortcuts", None, "Help", "Help", "secondary", "Show the read-only keyboard shortcut reference."),
        ("app.file.load_xyz", "Load XYZ...", "Load XYZ", "open_folder", "Home", "Project", "primary", "Load a molecular structure from XYZ or a supported calculation output."),
        ("app.file.open_project", "Open PCS Project...", "Open Project", "open_folder", "Home", "Project", "primary", "Open an existing PCS Analyzer project in a document tab."),
        ("app.file.save_project", "Save PCS Project", "Save Project", "save", "Home", "Project", "primary", "Save the active PCS Analyzer project to its current path, or choose a path if it is untitled."),
        ("app.file.save_project_as", "Save PCS Project As...", "Save As", "save", "Home", "Project", "secondary", "Save the active PCS Analyzer project under a new path."),
        ("app.file.save_all", "Save All", "Save All", "save", "Home", "Project", "secondary", "Save every modified PCS project document."),
        ("app.file.open_workspace", "Open Workspace...", "Open Workspace", "open_folder", "Home", "Workspace", "secondary", "Open a PCS workspace and replace the current document set after resolving unsaved changes."),
        ("app.file.save_workspace", "Save Workspace", "Save Workspace", "save", "Home", "Workspace", "secondary", "Save the current set and order of project references to the active workspace path."),
        ("app.file.save_workspace_as", "Save Workspace As...", "Workspace As", "save", "Home", "Workspace", "secondary", "Save the current workspace under a new path."),
        ("app.file.save_xyz", "Save current XYZ...", "Save XYZ", "save", "Home", "Output", "secondary", "Export the currently visible structure as XYZ."),
        ("app.file.save_plot_data", "Save plot data...", "Plot Data", "export", "Home", "Output", "secondary", "Export the active plot data or PCS plot image."),
        ("app.file.save_table", "Save table...", "Table", "export", "Home", "Output", "secondary", "Export the current PCS and rhombicity tables."),
        ("app.view.pcs_2d", "2D PCS Plot", "2D PCS", "pcs_2d", "View", "PCS Views", "primary", "Open or focus the two-dimensional PCS plot."),
        ("app.view.projection", "Projection", "Projection", "projection", "View", "PCS Views", "primary", "Open the projection viewer."),
        ("app.view.pcs_3d", "3D PCS Plot", "3D PCS", "pcs_3d", "View", "PCS Views", "primary", "Open the three-dimensional PCS field viewer."),
        ("app.view.structure_3d", "3D Structure", "3D Structure", "3d_structure", "View", "Structure", "primary", "Open the molecular structure viewer."),
        ("app.view.chemical_map", "Chemical Map", "Chemical Map", "chemical_map", "View", "Structure", "primary", "Open the Chemical Map window."),
        ("app.view.reset_control_panel", "Reset Control Panel", "Reset Panel", "reset", "View", "Layout", "secondary", "Restore the default Control Panel section order and expansion state."),
        ("app.tools.nmr_workbench", "NMR Workbench", "NMR Workbench", "nmr_workbench", "Tools", "Workbenches", "primary", "Open the integrated NMR Workbench."),
        ("app.tools.pcs_workbench", "PCS Workbench", "PCS Workbench", "pcs_workbench", "Tools", "Workbenches", "primary", "Open the PCS Workbench."),
        ("app.tools.module_manager", "Module Manager...", "Module Manager", "toolbox", "Tools", "Modules", "secondary", "Open the module and plugin manager."),
        ("app.tools.settings", "Open settings...", "Settings", "settings", "Tools", "Application", "secondary", "Open PCS Analyzer application settings."),
    )

    commands: dict[str, Command] = {}
    for command_id, label, ribbon_label, icon, category, group, presentation, description in definitions:
        callback = callbacks.get(command_id)
        if not callable(callback):
            raise KeyError(f"Missing callback for {command_id}")
        commands[command_id] = registry.create(
            command_id,
            label,
            callback,
            icon=icon,
            description=description,
            category=category,
            group=group,
            presentation=presentation,
            ribbon_label=ribbon_label,
            shortcut=MENU_SHORTCUTS.get(command_id),
        )
    return commands


__all__ = ["MAIN_WINDOW_COMMAND_IDS", "register_main_window_commands"]
