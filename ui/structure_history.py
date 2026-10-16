"""Rebuild structure-dependent presentation after a scoped replacement/restore."""
from logic.structure_edits import invalidate_structure_caches
from logic.selection_context import ensure_selection_hub


def capture_structure_controls(state):
    variable = state.get("torsion_avg_enabled_var")
    return {"torsion_avg_enabled_var": bool(variable.get())} if variable is not None else {}


def project_structure_controls(state, controls):
    for key, value in controls.items():
        variable = state.get(key)
        if variable is not None:
            if callable(getattr(variable, "set", None)):
                variable.set(value)
            else:
                variable.delete(0, "end")
                variable.insert(0, value)


def prepare_structure_views(state):
    """Invalidate old Ref caches and rebuild adapters before calculating plots."""
    invalidate_structure_caches(state)
    state["check_vars"] = {}
    state["ligand_check_vars"] = {}
    state.pop("current_selected_ids", None)
    state["cartesian_selected_ids"] = []
    tree = state.get("tree")
    if tree is not None:
        rows = tree.get_children()
        if rows:
            tree.delete(*rows)
    ensure_selection_hub(state).clear(source="structure:replace_or_restore")
    for key, args in (("create_checklist", (state,)), ("apply_symavg_to_state", (state,)),
                      ("refresh_averaging_status", ()), ("refresh_conformer_search_ui", ())):
        callback = state.get(key)
        if callable(callback):
            callback(*args)
    variable = state.get("conformer_status_var")
    if variable is not None:
        variable.set("Geometry restored." if state.get("conformer_preview") else
                     "Structure loaded. Click 'Sync' after loading δ_Exp")
