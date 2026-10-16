"""Plain values affected by the main Control Panel Reset action."""
from math import isfinite

from logic.document_edit_history import FIT_KEYS
from logic.structure_edits import REPLACEMENT_KEYS
from logic.pcs_tensor_state import ApplicationPCSTensor


RESET_KEYS = tuple(dict.fromkeys(REPLACEMENT_KEYS + FIT_KEYS + (
    "pcs_prediction_model", "fit_history", "active_fit_result_id", "latest_fit_result_id",
    "last_fit_result", "_last_fit_result_restored", "_last_fit_result_restore_error",
    "_last_fit_result_load_processed",
)))
RESET_ENTRY_KEYS = ("pcs_min_entry", "pcs_max_entry", "pcs_interval_entry")


def reset_document_values(settings):
    """Build defaults without altering settings, fit conditions or other tools."""
    controls = dict(zip(RESET_ENTRY_KEYS, (str(settings.get(k, default)) for k, default in
                    (("default_pcs_min", -10.), ("default_pcs_max", 10.), ("default_pcs_interval", .5)))))
    numbers = tuple(float(v) for v in controls.values())
    if not all(isfinite(v) for v in numbers) or numbers[2] <= 0:
        raise ValueError("Reset requires finite PCS range defaults and a positive interval.")
    tensor = ApplicationPCSTensor(float(settings.get("default_dchi_ax", -2.)))
    values = dict(pcs_tensor_state=tensor, pcs_prediction_model="axial", atom_data=[],
                  atom_data_raw=None, atom_data_original=None, atom_ids_raw=[],
                  metal_ref_id=None, x0=0., y0=0., z0=0.,
                  delta_exp_values={}, delta_obs_values={}, delta_dia_values={},
                  experimental_pcs_observations=[], torsion_avg_groups=[], torsion_avg_phase=2,
                  _visibility_selected_elements=[], _visibility_selected_ligands=[],
                  atom_data_conformer=None, conformer_applied=False, conformer_preview=None,
                  conformer_preview_coords=None, conformer_preview_elements=None,
                  conformer_preview_report="", fit_history=[])
    return values, controls
