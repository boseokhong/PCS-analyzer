"""Single-document ownership boundary for PCS Analyzer.

The current UI remains a stable legacy-state surface.  This module owns plain
document values and explicitly transfers them to/from that surface without
placing widgets, menus, dialogs, or command presenters in a document.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, MutableMapping
from uuid import uuid4


DOCUMENT_LOCAL_KEY_MANIFEST: dict[str, tuple[str, ...]] = {
    "structure": (
        "atom_data_original", "atom_data_raw", "atom_data", "atom_ids_raw",
        "atom_data_eff", "atom_ids_eff", "metal_ref_id", "x0", "y0", "z0",
        "ref_label_overrides", "symavg_pseudo_ref_ids", "symavg_records",
        "symavg_members_by_pseudo_id", "structure_graph", "pcsp_table_atoms_snapshot",
        "current_structure_path", "current_project_source_path",
    ),
    "pcs_data": (
        "delta_exp_values", "delta_obs_values", "delta_dia_values",
        "experimental_pcs_observations", "pcs_values", "pcs_by_id", "geom_by_id",
        "nmr_workbench_delta_exp_provenance",
        "atom_by_id", "row_by_id", "current_selected_ids",
    ),
    "tensor_orientation": (
        "tensor", "rh_dchi_rh", "rh_calc_enabled", "angle_x", "angle_y", "angle_z",
        "pcs_prediction_model", "pcs_tensor_state", "pcs_tensor_frame_convention",
        "pcs_tensor_angle_unit", "pcs_model_revision", "fit_override",
    ),
    "averaging": (
        "torsion_avg_groups", "torsion_avg_phase", "torsion_avg_display_atom_data",
        "torsion_reference_polar_data", "torsion_reference_ref_ids",
        "torsion_reference_rotated_coords",
    ),
    "standard_fitting": (
        "fit_history", "active_fit_result_id", "latest_fit_result_id", "last_fit_result",
        "_last_fit_result_load_processed", "_last_fit_result_restore_error",
        "_last_fit_result_restored", "_last_fit_result_status",
    ),
    "advanced_fitting": (
        "advfit_conformers", "advfit_mc_last_result", "advfit_mc_options", "advfit_rdc_rows",
        "advfit_rdc_last_result", "advfit_rdc_options", "advfit_pre_obs", "advfit_pre_last_result", "advfit_pre_options",
        "advfit_ml_datasets", "advfit_ml_last_result", "advfit_ml_options", "advfit_ml_results",
    ),
    "conformer_search": (
        "conformer_search_session",
        "conformer_preview", "conformer_preview_coords", "conformer_preview_elements",
        "conformer_preview_report", "conformer_applied", "last_conformer_result",
        "atom_data_conformer",
    ),
    "selection_undo": (
        "selection_context", "cartesian_selected_ids", "cartesian_selection_mode",
        "cartesian_selector_active", "cartesian_selector_kind", "_delta_exp_undo",
        "_delta_obs_undo", "_delta_dia_undo", "_experimental_pcs_observations_undo",
        "_nmr_workbench_delta_exp_provenance_undo",
    ),
    "view_settings": (
        "residual_thr_ok", "residual_thr_warn", "pcs_scene_kwargs",
        "chemical_map_background", "_visibility_selected_elements",
        "_visibility_selected_ligands",
    ),
}

DOCUMENT_TK_VARIABLE_KEYS = (
    "angle_x_var", "angle_y_var", "angle_z_var", "rh_dchi_rh_var",
    "rh_calc_enabled_var", "pcs_prediction_model_var", "plot_90_var",
    "symavg_enabled_var", "symavg_methyl_enabled_var", "symavg_cf3_enabled_var",
    "symavg_keep_original_var", "torsion_avg_enabled_var",
    "torsion_avg_detection_mode_var", "torsion_avg_keep_reference_var",
    "torsion_avg_planarity_var", "residual_color_enabled_var",
    "plot_force_origin_var", "diag_fit_intercept_var", "fit_mode_var", "axis_mode_var",
    "fit_use_visible_var", "fit_dchi_var", "fit_dchi_rh_var", "fit_global_search_var",
    "fit_use_current_tensor_var", "fit_weighting_var", "plot3d_color_mode_var",
    "plot3d_show_labels_var", "projection_mode_var", "projection_r_var",
    "projection_show_atoms_var", "projection_show_h_var",
    "advfit_mc_fit_rh_var", "advfit_mc_fit_weights_var", "advfit_rdc_B0_var",
    "advfit_rdc_T_var", "advfit_rdc_weight_var", "advfit_pre_B0_var",
    "advfit_pre_mu_var", "advfit_pre_ln_var", "advfit_ml_ln_var",
)

DOCUMENT_ENTRY_KEYS = (
    "tensor_entry", "pcs_min_entry", "pcs_max_entry", "pcs_interval_entry",
    "angle_x_entry", "angle_y_entry", "angle_z_entry",
)

DOCUMENT_OWNED_KEYS = frozenset(
    key for keys in DOCUMENT_LOCAL_KEY_MANIFEST.values() for key in keys
)

_SUPPRESS_KEY = "_pcs_document_tracking_suppressed"


_DEFAULT_TK_VALUES: dict[str, Any] = {
    "angle_x_var": 0.0,
    "angle_y_var": 0.0,
    "angle_z_var": 0.0,
    "rh_dchi_rh_var": "0",
    "rh_calc_enabled_var": False,
    "pcs_prediction_model_var": "axial",
    "plot_90_var": False,
    "symavg_enabled_var": False,
    "symavg_methyl_enabled_var": True,
    "symavg_cf3_enabled_var": True,
    "symavg_keep_original_var": False,
    "torsion_avg_enabled_var": False,
    "torsion_avg_detection_mode_var": "auto",
    "torsion_avg_keep_reference_var": False,
    "torsion_avg_planarity_var": 0.10,
    "residual_color_enabled_var": False,
    "plot_force_origin_var": False,
    "diag_fit_intercept_var": False,
    "fit_mode_var": "theta_alpha_multi",
    "axis_mode_var": "bisector",
    "fit_use_visible_var": True,
    "fit_dchi_var": False,
    "fit_dchi_rh_var": False,
    "fit_global_search_var": False,
    "fit_use_current_tensor_var": False,
    "fit_weighting_var": "Unweighted",
    "plot3d_color_mode_var": "Element",
    "plot3d_show_labels_var": False,
    "projection_mode_var": "φ / cos(θ)",
    "projection_r_var": "10.0",
    "projection_show_atoms_var": True,
    "projection_show_h_var": True,
    "advfit_mc_fit_rh_var": False,
    "advfit_mc_fit_weights_var": True,
    "advfit_rdc_B0_var": "14.1",
    "advfit_rdc_T_var": "298",
    "advfit_rdc_weight_var": "1.0",
    "advfit_pre_B0_var": "14.1",
    "advfit_pre_mu_var": "3.68",
    "advfit_pre_ln_var": "Nd",
    "advfit_ml_ln_var": "Nd",
}


def _install_manager_bindings(
    state: MutableMapping[str, Any], manager: Any
) -> None:
    state["document_manager"] = manager
    state["pcs_documents"] = manager.compatibility_documents
    state["set_active_document"] = manager.set_active_document
    state["subscribe_active_document_changed"] = manager.subscribe
    state["unsubscribe_active_document_changed"] = manager.unsubscribe


@dataclass
class PCSDocument:
    """Minimal owner for one PCS project/complex."""

    title: str = "Untitled 1"
    file_path: Path | None = None
    dirty: bool = False
    revision: int = 0
    project_name: str | None = None
    tab_color: str | None = None
    document_id: str = field(default_factory=lambda: str(uuid4()))
    _data: dict[str, Any] = field(default_factory=dict, repr=False)
    edit_history: Any = field(default=None, repr=False)

    @property
    def data(self) -> Mapping[str, Any]:
        return MappingProxyType(self._data)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def replace_data(self, values: Mapping[str, Any]) -> None:
        self._data = deepcopy(dict(values))

    def update_data(self, values: Mapping[str, Any]) -> None:
        """Apply detached worker results without projecting this document."""
        self._data.update(deepcopy(dict(values)))

    def mark_mutated(self) -> None:
        self.revision += 1
        self.dirty = True

    def mark_clean(self, file_path: str | Path | None = None) -> None:
        if file_path is not None:
            self.file_path = Path(file_path)
            self.title = self.file_path.name
        self.dirty = False


def document_display_name(document: PCSDocument | None) -> str:
    """Return the concise user-facing name shared by document integrations."""
    if document is None:
        return "Untitled"
    project_name = str(document.project_name or "").strip()
    if project_name:
        return project_name
    if document.file_path is not None:
        return document.file_path.name
    return str(document.title or "Untitled").strip() or "Untitled"


@dataclass(frozen=True)
class DocumentWorkerIdentity:
    document_id: str
    revision: int


class _DetachedWorkerValue:
    def __init__(self, value: Any) -> None:
        self.value = _copy(value)

    def get(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = _copy(value)

    def delete(self, _start: Any, _end: Any) -> None:
        self.value = ""

    def insert(self, _index: Any, value: Any) -> None:
        self.value = str(value)


class ActiveDocumentAdapter(MutableMapping[str, Any]):
    """Narrow mapping over manifest-approved values of the active document.

    This is deliberately not a proxy for the application state.  Widgets,
    callbacks, services, settings, and unknown keys remain accessible only
    through the original application mapping.  Writes mirror into that stable
    mapping during the compatibility period so existing UI callbacks continue
    to observe the current single visible document.
    """

    def __init__(self, application_state: MutableMapping[str, Any]) -> None:
        self._application_state = application_state

    @property
    def application_state(self) -> MutableMapping[str, Any]:
        return self._application_state

    @property
    def document(self) -> PCSDocument:
        return require_active_document(self._application_state)

    def _reviewed_key(self, key: object) -> str:
        text = str(key)
        if text not in DOCUMENT_OWNED_KEYS:
            raise KeyError(
                f"Key is not classified as document-local: {text}"
            )
        return text

    def __getitem__(self, key: str) -> Any:
        reviewed = self._reviewed_key(key)
        return self.document._data[reviewed]

    def __setitem__(self, key: str, value: Any) -> None:
        reviewed = self._reviewed_key(key)
        copied = _copy(value)
        self.document._data[reviewed] = copied
        self._application_state[reviewed] = copied

    def __delitem__(self, key: str) -> None:
        reviewed = self._reviewed_key(key)
        if reviewed not in self.document._data:
            raise KeyError(reviewed)
        del self.document._data[reviewed]
        self._application_state.pop(reviewed, None)

    def __iter__(self):
        return (
            key for key in DOCUMENT_OWNED_KEYS if key in self.document._data
        )

    def __len__(self) -> int:
        return sum(key in self.document._data for key in DOCUMENT_OWNED_KEYS)

    def __contains__(self, key: object) -> bool:
        return key in DOCUMENT_OWNED_KEYS and key in self.document._data


def install_single_document(
    state: MutableMapping[str, Any], document: PCSDocument | None = None
) -> PCSDocument:
    """Install the one active document without introducing a multi-doc UI."""
    manager = state.get("document_manager")
    managed = getattr(manager, "active_document", None)
    if isinstance(managed, PCSDocument):
        return managed
    existing = state.get("active_document")
    if isinstance(existing, PCSDocument):
        from logic.document_manager import DocumentManager

        manager = DocumentManager(state, existing)
        _install_manager_bindings(state, manager)
        if not isinstance(
            state.get("active_document_adapter"), ActiveDocumentAdapter
        ):
            state["active_document_adapter"] = ActiveDocumentAdapter(state)
        state.setdefault(
            "refresh_active_document_commands",
            lambda: refresh_active_document_commands(state),
        )
        return existing
    document = document or PCSDocument()
    from logic.document_manager import DocumentManager

    manager = DocumentManager(state, document)
    _install_manager_bindings(state, manager)
    state["active_document"] = document
    state["current_project_path"] = (
        str(document.file_path) if document.file_path is not None else ""
    )
    state["active_document_adapter"] = ActiveDocumentAdapter(state)
    state["refresh_active_document_commands"] = (
        lambda: refresh_active_document_commands(state)
    )
    return document


def active_document(state: Mapping[str, Any]) -> PCSDocument | None:
    if isinstance(state, ActiveDocumentAdapter):
        return state.document
    manager = state.get("document_manager")
    document = getattr(manager, "active_document", None)
    if isinstance(document, PCSDocument):
        return document
    document = state.get("active_document")
    return document if isinstance(document, PCSDocument) else None


def require_active_document(state: Mapping[str, Any]) -> PCSDocument:
    """Return the explicit active document or fail at the access boundary."""
    application_state = (
        state._application_state
        if isinstance(state, ActiveDocumentAdapter)
        else state
    )
    document = active_document(application_state)
    if not isinstance(document, PCSDocument):
        raise RuntimeError("No active PCSDocument is installed")
    return document


def active_document_id(state: Mapping[str, Any]) -> str | None:
    document = active_document(state)
    return document.document_id if document is not None else None


def active_document_path(state: Mapping[str, Any]) -> Path | None:
    document = active_document(state)
    return document.file_path if document is not None else None


def active_document_view(
    state: MutableMapping[str, Any],
) -> ActiveDocumentAdapter:
    """Return the stable manifest-only compatibility view for this app."""
    if isinstance(state, ActiveDocumentAdapter):
        return state
    adapter = state.get("active_document_adapter")
    if not isinstance(adapter, ActiveDocumentAdapter):
        require_active_document(state)
        adapter = ActiveDocumentAdapter(state)
        state["active_document_adapter"] = adapter
    return adapter


def document_by_id(state: Mapping[str, Any], document_id: str) -> PCSDocument | None:
    manager = state.get("document_manager")
    lookup = getattr(manager, "document_by_id", None)
    if callable(lookup):
        document = lookup(str(document_id))
        return document if isinstance(document, PCSDocument) else None
    documents = state.get("pcs_documents", {}) or {}
    document = documents.get(str(document_id)) if isinstance(documents, Mapping) else None
    return document if isinstance(document, PCSDocument) else None


@contextmanager
def suspend_document_tracking(state: MutableMapping[str, Any]) -> Iterator[None]:
    depth = int(state.get(_SUPPRESS_KEY, 0) or 0)
    state[_SUPPRESS_KEY] = depth + 1
    try:
        yield
    finally:
        if depth:
            state[_SUPPRESS_KEY] = depth
        else:
            state.pop(_SUPPRESS_KEY, None)


def _copy(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception:
        return value


def default_document_data(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a clean non-inheriting fallback for an untitled document."""
    cfg = state.get("app_settings", {}) or {}
    values: dict[str, Any] = {
        "atom_data_original": [],
        "atom_data_raw": [],
        "atom_data": [],
        "atom_ids_raw": [],
        "atom_data_eff": [],
        "atom_ids_eff": [],
        "ref_label_overrides": {},
        "symavg_pseudo_ref_ids": [],
        "symavg_records": [],
        "symavg_members_by_pseudo_id": {},
        "structure_graph": None,
        "pcsp_table_atoms_snapshot": [],
        "x0": 0.0,
        "y0": 0.0,
        "z0": 0.0,
        "delta_exp_values": {},
        "delta_obs_values": {},
        "delta_dia_values": {},
        "experimental_pcs_observations": [],
        "pcs_values": [],
        "pcs_by_id": {},
        "geom_by_id": {},
        "atom_by_id": {},
        "row_by_id": {},
        "current_selected_ids": [],
        "fit_history": [],
        "active_fit_result_id": None,
        "latest_fit_result_id": None,
        "last_fit_result": None,
        "advfit_conformers": [],
        "advfit_mc_last_result": None,
        "advfit_mc_options": {},
        "advfit_rdc_rows": [],
        "advfit_rdc_last_result": None,
        "advfit_rdc_options": {},
        "advfit_pre_obs": [],
        "advfit_pre_options": {},
        "advfit_pre_last_result": None,
        "advfit_ml_datasets": [],
        "advfit_ml_options": {},
        "advfit_ml_results": {},
        "advfit_ml_last_result": None,
        "selection_context": None,
        "cartesian_selected_ids": set(),
        "cartesian_selection_mode": "all",
        "cartesian_selector_active": False,
        "cartesian_selector_kind": None,
        "_delta_exp_undo": None,
        "_delta_obs_undo": None,
        "_delta_dia_undo": None,
        "_experimental_pcs_observations_undo": None,
        "_nmr_workbench_delta_exp_provenance_undo": None,
        "conformer_preview": None,
        "conformer_preview_coords": None,
        "conformer_preview_elements": None,
        "conformer_preview_report": "",
        "conformer_applied": False,
        "last_conformer_result": None,
        "atom_data_conformer": None,
        "torsion_avg_groups": [],
        "torsion_avg_phase": 2,
        "torsion_avg_display_atom_data": [],
        "torsion_reference_polar_data": [],
        "torsion_reference_ref_ids": [],
        "torsion_reference_rotated_coords": [],
        "tensor": float(cfg.get("default_dchi_ax", -2.0)),
        "rh_dchi_rh": 0.0,
        "rh_calc_enabled": False,
        "angle_x": 0.0,
        "angle_y": 0.0,
        "angle_z": 0.0,
        "pcs_prediction_model": "axial",
        "tensor_entry": str(cfg.get("default_dchi_ax", -2.0)),
        "pcs_min_entry": str(cfg.get("default_pcs_min", -10.0)),
        "pcs_max_entry": str(cfg.get("default_pcs_max", 10.0)),
        "pcs_interval_entry": str(cfg.get("default_pcs_interval", 0.5)),
        "angle_x_entry": "0.0",
        "angle_y_entry": "0.0",
        "angle_z_entry": "0.0",
        "residual_thr_ok": 0.10,
        "residual_thr_warn": 0.30,
        "pcs_scene_kwargs": {},
        "chemical_map_background": "#FFFFFF",
        # ``None`` means the visibility filter has not been initialized yet;
        # once a structure/UI exists, an empty tuple intentionally means none.
        "_visibility_selected_elements": None,
        "_visibility_selected_ligands": None,
    }
    values.update(_DEFAULT_TK_VALUES)
    return values


def commit_pending_document_edits(state: MutableMapping[str, Any]) -> tuple[str, ...]:
    """Commit narrow Entry-backed scalar edits without invoking analysis."""
    committed: list[str] = []
    finish_gesture = state.get("finish_parameter_gesture")
    if callable(finish_gesture):
        finish_gesture()
    capture_search = state.get("capture_conformer_search_state")
    if callable(capture_search):
        capture_search()
    unchanged_input = state.get("is_history_input_unchanged")
    def is_unchanged(key):
        return callable(unchanged_input) and unchanged_input(key)
    with suspend_document_tracking(state):
        tensor_entry = state.get("tensor_entry")
        try:
            if not is_unchanged("tensor_entry"):
                state["tensor"] = float(tensor_entry.get())
                committed.append("tensor")
        except (AttributeError, TypeError, ValueError):
            pass

        rh_var = state.get("rh_dchi_rh_var")
        try:
            state["rh_dchi_rh"] = float(rh_var.get())
            committed.append("rh_dchi_rh")
        except (AttributeError, TypeError, ValueError):
            pass

        for axis in "xyz":
            if is_unchanged(f"angle_{axis}_entry"):
                continue
            entry = state.get(f"angle_{axis}_entry")
            variable = state.get(f"angle_{axis}_var")
            try:
                value = float(entry.get())
            except (AttributeError, TypeError, ValueError):
                continue
            state[f"angle_{axis}"] = value
            setter = getattr(variable, "set", None)
            if callable(setter):
                setter(value)
            committed.append(f"angle_{axis}")
    return tuple(committed)


def capture_active_document(
    state: Mapping[str, Any], document: PCSDocument | None = None
) -> PCSDocument:
    """Capture plain document values from the stable legacy UI/state."""
    document = document or active_document(state)
    if document is None:
        raise RuntimeError("No active PCSDocument is installed")
    values: dict[str, Any] = {}
    for key in DOCUMENT_OWNED_KEYS:
        if key in state:
            values[key] = _copy(state[key])
    for key in DOCUMENT_TK_VARIABLE_KEYS:
        variable = state.get(key)
        if variable is not None and callable(getattr(variable, "get", None)):
            try:
                values[key] = _copy(variable.get())
            except Exception:
                pass
    for key in DOCUMENT_ENTRY_KEYS:
        entry = state.get(key)
        if entry is not None and callable(getattr(entry, "get", None)):
            try:
                values[key] = str(entry.get())
            except Exception:
                pass
    # Visibility is document-owned plain data.  Tk variables remain a UI
    # adapter and are captured only when the corresponding control set exists.
    from logic.visibility import sync_visibility_state_from_vars

    sync_visibility_state_from_vars(state)
    for key in ("_visibility_selected_elements", "_visibility_selected_ligands"):
        if key in state:
            values[key] = _copy(state.get(key))
    document.replace_data(values)
    return document


def hydrate_active_document(
    state: MutableMapping[str, Any],
    document: PCSDocument,
    *,
    refresh: bool = False,
    _from_manager: bool = False,
) -> None:
    """Project one document into existing keys, variables, entries, and views."""
    manager = state.get("document_manager")
    if manager is not None and not _from_manager:
        if active_document(state) is document:
            hydrate_active_document(
                state, document, refresh=False, _from_manager=True
            )
            if refresh:
                refresh_follow_active_surfaces(state)
            return
        if document_by_id(state, document.document_id) is None:
            manager.add_document(document)
        manager.set_active_document(document.document_id)
        if refresh:
            refresh_follow_active_surfaces(state)
        return
    values = document.data
    with suspend_document_tracking(state):
        state["active_document"] = document
        state.setdefault("pcs_documents", {})[document.document_id] = document
        for key in DOCUMENT_OWNED_KEYS:
            if key in values:
                state[key] = _copy(values[key])
            else:
                state.pop(key, None)
        for key in DOCUMENT_TK_VARIABLE_KEYS:
            if key not in values:
                continue
            variable = state.get(key)
            setter = getattr(variable, "set", None)
            if callable(setter):
                setter(_copy(values[key]))
            else:
                state[key] = _copy(values[key])
        for key in DOCUMENT_ENTRY_KEYS:
            if key not in values:
                continue
            entry = state.get(key)
            if entry is None:
                state[key] = _copy(values[key])
                continue
            try:
                entry.delete(0, "end")
                entry.insert(0, str(values[key]))
            except Exception:
                pass
        state["current_project_path"] = (
            str(document.file_path) if document.file_path is not None else ""
        )
        # Ligand analysis is a transient cache over the active structure.  It
        # must never bleed across document tabs; the visibility selection itself
        # is plain document data and is rebuilt against the incoming structure.
        state.pop("ligand_analysis", None)
        state.pop("ligand_analysis_summary", None)
        state.pop("ligand_visibility_error", None)

        def apply_visibility_variables() -> None:
            selected_elements = values.get("_visibility_selected_elements")
            check_vars = state.get("check_vars", {}) or {}
            if selected_elements is not None and isinstance(check_vars, Mapping):
                selected = {str(value) for value in selected_elements}
                for element, variable in check_vars.items():
                    setter = getattr(variable, "set", None)
                    if callable(setter):
                        setter(str(element) in selected)

            selected_ligands = values.get("_visibility_selected_ligands")
            ligand_vars = state.get("ligand_check_vars", {}) or {}
            if selected_ligands is not None and isinstance(ligand_vars, Mapping):
                selected = {str(value) for value in selected_ligands}
                for key, variable in ligand_vars.items():
                    setter = getattr(variable, "set", None)
                    if callable(setter):
                        setter(str(key) in selected)

        apply_visibility_variables()
        if refresh:
            _clear_document_presentations(state)
            _refresh_legacy_views(state)
            apply_visibility_variables()
        try:
            from logic.selection_context import restore_active_document_selection

            restore_active_document_selection(state)
        except Exception:
            pass
        refresh_active_document_commands(state)


def _clear_document_presentations(state: MutableMapping[str, Any]) -> None:
    """Clear document-owned presenters before their refresh callbacks run.

    Several legacy refreshers deliberately return early for an empty model.
    Clearing centrally prevents the outgoing document from remaining visible.
    """
    for key in ("tree", "rh_tree", "fit_history_tree"):
        widget = state.get(key)
        if widget is None:
            continue
        try:
            children = widget.get_children()
            if children:
                widget.delete(*children)
            selection_remove = getattr(widget, "selection_remove", None)
            if callable(selection_remove):
                selection_remove(*widget.selection())
        except Exception:
            pass
    for key in ("fit_donor_list", "fit_proton_list"):
        widget = state.get(key)
        try:
            widget.delete(0, "end")
        except Exception:
            pass
    for key in ("fit_result_box", "conformer_result_box"):
        widget = state.get(key)
        try:
            previous_state = None
            cget = getattr(widget, "cget", None)
            if callable(cget):
                previous_state = cget("state")
            if previous_state == "disabled":
                widget.configure(state="normal")
            widget.delete("1.0", "end")
            if previous_state == "disabled":
                widget.configure(state="disabled")
        except Exception:
            pass
    for key, value in (
        ("fit_latest_run_summary_var", "No fit runs yet."),
        ("conformer_status_var", "Ready."),
    ):
        setter = getattr(state.get(key), "set", None)
        if callable(setter):
            try:
                setter(value)
            except Exception:
                pass


def _refresh_legacy_views(state: MutableMapping[str, Any]) -> None:
    for key, pass_state in (
        ("apply_symavg_to_state", True),
        ("create_checklist", True),
        ("update_graph", False),
        ("plot_cartesian", True),
        ("rh_refresh_table", False),
        ("refresh_residual_controls", False),
        ("populate_fitting_controls", True),
        ("refresh_last_fit_result_ui", False),
        ("refresh_fit_history_ui", False),
        ("refresh_conformer_search_ui", False),
        ("refresh_pcs_data_commands", False),
        ("refresh_gax_rhombicity_commands", False),
        ("refresh_fitting_commands", False),
        ("refresh_prediction_model_ui", False),
        ("refresh_conformer_commands", False),
        ("refresh_edit_history", False),
    ):
        callback = state.get(key)
        if not callable(callback):
            continue
        try:
            callback(state) if pass_state else callback()
        except Exception:
            pass


def refresh_follow_active_surfaces(
    state: MutableMapping[str, Any], *, include_legacy: bool = True
) -> None:
    """Refresh existing follow-active views without recreating their windows."""
    if include_legacy:
        _refresh_legacy_views(state)
    chemical_map = state.get("chemical_map_win")
    refresh_structure = getattr(chemical_map, "refresh_structure", None)
    if callable(refresh_structure):
        try:
            refresh_structure()
        except Exception as exc:
            state["_chemical_map_document_refresh_error"] = exc
    for callback in tuple(state.get("follow_active_document_refresh_hooks", ()) or ()):
        try:
            callback()
        except Exception:
            pass
    refresh_integrations = state.get("refresh_document_integration_titles")
    if callable(refresh_integrations):
        try:
            refresh_integrations()
        except Exception:
            pass


def mark_document_mutated(state: MutableMapping[str, Any], domain: str) -> bool:
    """Capture a controlled scientific mutation and advance its document."""
    application_state = (
        state.application_state
        if isinstance(state, ActiveDocumentAdapter)
        else state
    )
    if int(application_state.get(_SUPPRESS_KEY, 0) or 0):
        return False
    document = active_document(application_state)
    if document is None:
        return False
    capture_active_document(application_state, document)
    document.mark_mutated()
    application_state["last_document_mutation_domain"] = str(domain)
    refresh_history = application_state.get("refresh_edit_history")
    if callable(refresh_history):
        refresh_history()
    refresh_tabs = application_state.get("refresh_document_tabs")
    if callable(refresh_tabs):
        try:
            refresh_tabs()
        except Exception:
            pass
    return True


def capture_worker_identity(state: Mapping[str, Any]) -> DocumentWorkerIdentity | None:
    document = active_document(state)
    if document is None:
        return None
    return DocumentWorkerIdentity(document.document_id, document.revision)


def worker_document_state(
    state: MutableMapping[str, Any], document: PCSDocument | None = None
) -> dict[str, Any]:
    """Return a detached legacy-compatible input mapping for a worker."""
    document = document or active_document(state)
    if document is None:
        return dict(state)
    capture_active_document(state, document)
    values = document.snapshot()
    detached = dict(state)
    detached.update(values)
    for key in (*DOCUMENT_TK_VARIABLE_KEYS, *DOCUMENT_ENTRY_KEYS):
        if key in values:
            detached[key] = _DetachedWorkerValue(values[key])
    detached.pop("filter_atoms", None)
    return detached


def worker_completion_matches(
    state: Mapping[str, Any],
    identity: DocumentWorkerIdentity | None,
    *,
    require_revision: bool = True,
) -> bool:
    """Return whether a worker may still apply to its origin document."""
    if not isinstance(identity, DocumentWorkerIdentity):
        return False
    document = document_by_id(state, identity.document_id)
    if document is None:
        return False
    return not require_revision or document.revision == identity.revision


def apply_worker_document_updates(
    state: MutableMapping[str, Any],
    identity: DocumentWorkerIdentity | None,
    values: Mapping[str, Any],
    *,
    domain: str,
) -> PCSDocument | None:
    """Commit a valid worker payload to its origin without changing tabs."""
    if not worker_completion_matches(state, identity):
        return None
    document = document_by_id(state, identity.document_id)
    if document is None:
        return None
    document.update_data(values)
    document.mark_mutated()
    state["last_document_mutation_domain"] = str(domain)
    refresh_tabs = state.get("refresh_document_tabs")
    if callable(refresh_tabs):
        try:
            refresh_tabs()
        except Exception:
            pass
    return document


def refresh_active_document_commands(state: Mapping[str, Any]) -> bool:
    """Re-evaluate command predicates without rebuilding any presenters."""
    registry = state.get("command_registry")
    refresh = getattr(registry, "refresh", None)
    if not callable(refresh):
        return False
    refresh(force=True)
    return True


__all__ = [
    "DOCUMENT_ENTRY_KEYS",
    "DOCUMENT_LOCAL_KEY_MANIFEST",
    "DOCUMENT_OWNED_KEYS",
    "DOCUMENT_TK_VARIABLE_KEYS",
    "ActiveDocumentAdapter",
    "DocumentWorkerIdentity",
    "PCSDocument",
    "active_document",
    "active_document_id",
    "active_document_path",
    "active_document_view",
    "apply_worker_document_updates",
    "capture_active_document",
    "capture_worker_identity",
    "commit_pending_document_edits",
    "default_document_data",
    "document_by_id",
    "document_display_name",
    "hydrate_active_document",
    "install_single_document",
    "mark_document_mutated",
    "refresh_active_document_commands",
    "refresh_follow_active_surfaces",
    "require_active_document",
    "suspend_document_tracking",
    "worker_completion_matches",
    "worker_document_state",
]
