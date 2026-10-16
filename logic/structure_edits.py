"""Detached structure-replacement values; no widgets, parsing or visualization."""
from math import isfinite
from pathlib import Path

from logic.document_edit_history import PCS_KEYS, STRUCTURE_KEYS


REPLACEMENT_KEYS = tuple(dict.fromkeys(STRUCTURE_KEYS + PCS_KEYS + (
    "atom_data_original", "metal_ref_id", "x0", "y0", "z0", "fit_override",
    "current_structure_path", "torsion_avg_groups", "torsion_avg_phase",
    "_visibility_selected_elements", "_visibility_selected_ligands",
    "conformer_preview", "conformer_preview_coords", "conformer_preview_elements",
    "conformer_preview_report",
)))

STRUCTURE_CACHE_KEYS = (
    "structure_graph", "atom_data_eff", "atom_ids_eff", "ref_label_overrides",
    "symavg_pseudo_ref_ids", "symavg_records", "symavg_members_by_pseudo_id",
    "last_rotated_coords", "row_by_id", "atom_by_id", "pcs_by_id", "geom_by_id",
    "delta_para_values", "pcsp_table_atoms_snapshot", "ligand_analysis",
    "ligand_analysis_summary", "ligand_visibility_error",
    "torsion_avg_display_atom_data", "torsion_reference_polar_data",
    "torsion_reference_ref_ids", "torsion_reference_rotated_coords",
)


def validate_structure_atoms(atoms):
    """Normalize existing parser output without changing coordinates or units."""
    values = []
    for element, x, y, z in atoms:
        xyz = tuple(float(v) for v in (x, y, z))
        if not str(element) or not all(isfinite(v) for v in xyz):
            raise ValueError("Structure atoms must have an element and finite coordinates.")
        values.append((str(element), *xyz))
    if not values:
        raise ValueError("No usable atoms were found in the structure file.")
    return values


def replacement_values(atoms, center_element, path):
    """Choose the first matching element, as in the existing main loader."""
    atoms = validate_structure_atoms(atoms)
    center = next((i for i, atom in enumerate(atoms) if atom[0] == center_element), None)
    if center is None:
        raise ValueError(f"Atom {center_element} not found in the file.")
    x, y, z = atoms[center][1:]
    return dict(atom_data_original=list(atoms), atom_data_raw=list(atoms), atom_data=list(atoms),
                atom_ids_raw=list(range(1, len(atoms) + 1)), metal_ref_id=center + 1,
                x0=x, y0=y, z0=z, current_structure_path=str(Path(path).resolve()),
                delta_exp_values={}, delta_obs_values={}, delta_dia_values={},
                experimental_pcs_observations=[], torsion_avg_groups=[],
                _visibility_selected_elements=sorted({atom[0] for atom in atoms}),
                conformer_applied=False, conformer_preview=None,
                conformer_preview_coords=None, conformer_preview_elements=None,
                conformer_preview_report="")


def invalidate_structure_caches(state):
    for key in STRUCTURE_CACHE_KEYS:
        state.pop(key, None)
