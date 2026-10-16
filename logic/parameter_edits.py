"""Validated assigned-centre edits over plain PCS document values."""
from math import isfinite

from logic.document_edit_history import begin_edit, finish_edit
from logic.pcs_document import mark_document_mutated


CENTER_KEYS = ("metal_ref_id", "x0", "y0", "z0", "fit_override")


def assign_metal_center(state, ref_id):
    """Assign an existing raw Ref at its absolute Å coordinates, clearing preview."""
    atoms = state.get("atom_data_raw") or state.get("atom_data") or []
    ids = state.get("atom_ids_raw") or list(range(1, len(atoms) + 1))
    if isinstance(ref_id, bool) or not isinstance(ref_id, int):
        raise ValueError("Select a valid raw atom Ref")
    matches = [i for i, value in enumerate(ids) if value == ref_id]
    if len(ids) != len(atoms) or len(matches) != 1:
        raise ValueError("The selected Ref is not a unique raw atom")
    xyz = tuple(float(v) for v in atoms[matches[0]][1:])
    if len(xyz) != 3 or not all(isfinite(v) for v in xyz):
        raise ValueError("The selected atom has invalid coordinates")
    if (state.get("metal_ref_id") == ref_id and
            tuple(state.get(k) for k in ("x0", "y0", "z0")) == xyz and not state.get("fit_override")):
        return False
    token = begin_edit(state, CENTER_KEYS)
    state["metal_ref_id"] = ref_id
    state["x0"], state["y0"], state["z0"] = xyz
    state.pop("fit_override", None)
    state.pop("structure_graph", None)
    finish_edit(state, token, f"Assign metal center · Ref {ref_id}")
    mark_document_mutated(state, "assigned_center")
    return True
