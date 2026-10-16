"""Tk boundary for document-owned Conformer Search records."""
from copy import deepcopy
from types import SimpleNamespace
import math
import numpy as np

from logic.conformer_search_project import (plain, parse_payload, candidate_values,
                                           preview_payload, context_matches)
from logic.pcs_document import active_document, mark_document_mutated

CONTROL_NAMES = ("_metal_idx_var", "_mx_var", "_my_var", "_mz_var", "_dchi_ax_var",
    "_dchi_rh_var", "_ea_var", "_eb_var", "_eg_var", "_search_mode_var",
    "_constraint_mode_var", "_anchor_mode_var", "_use_global_var", "_de_maxiter_var",
    "_de_popsize_var", "_opt_level_var", "_output_mode_var", "_n_candidates_var",
    "_dedup_tol_var", "_rb_axis_mode_var", "_rb_lock_mode_var", "_rb_grid_var",
    "_rb_bounds_lo_var", "_rb_bounds_hi_var")


def host_context(state):
    def read(key, fallback=0):
        value = state.get(key, fallback)
        return value.get() if hasattr(value, "get") else value
    def number(key, fallback):
        try:
            return float(read(key, fallback))
        except (TypeError, ValueError):
            return float(fallback)
    return plain(dict(
        atoms=state.get("atom_data_raw") or state.get("atom_data") or [],
        refs=state.get("atom_ids_raw") or [],
        pcs=sorted((int(k), float(v)) for k, v in state.get("delta_exp_values", {}).items()),
        metal=[state.get("metal_ref_id"), *(state.get(k, 0.) for k in ("x0", "y0", "z0"))],
        tensor=[number("tensor_entry", state.get("tensor", 0.)),
                float(state.get("rh_dchi_rh", 0.)), bool(state.get("rh_calc_enabled", False)),
                state.get("pcs_prediction_model", "axial")],
        orientation=[number(f"angle_{a}_var", state.get(f"angle_{a}", 0.)) for a in "xyz"],
        geometry_preview=state.get("fit_override"),
    ))


class ConformerSearchProjectAdapter:
    def __init__(self, state, app):
        self.state, self.app = state, app
        self.restoring = False
        self.document_id = None
        self.restored_record = None
        self.defaults = self.controls()

    def controls(self):
        return {name: getattr(self.app, name).get() for name in CONTROL_NAMES}

    def begin_run(self, atoms, values):
        app = self.app
        document = active_document(self.state)
        self.document_id = document.document_id if document else None
        return dict(context=host_context(self.state), inputs=plain(dict(
            elements=list(app.mol.elements), coords=app.mol.coords,
            metal_index=app.metal_idx, pcs=list(zip(atoms, values)),
            fixed=sorted(app._get_fixed_set()),
            selected_bonds=[b for b, v, locked in zip(app.rotatable_all, app.bond_vars,
                                                    app.bond_locked) if v.get() and not locked],
            bond_options=app.mol.bond_options,
            manual_add=sorted(app.mol.manual_add_bonds), manual_remove=sorted(app.mol.manual_remove_bonds),
            controls=self.controls(),
        )))

    def export(self, *, candidates=None, run=None, preview_index=None):
        from ui.conformer_search import format_report, format_rigid_body_report
        app = self.app
        candidates = candidates if candidates is not None else getattr(app, "candidates", None)
        run = run or getattr(app, "_recorded_run", None)
        if not candidates or run is None:
            return None
        inputs = run["inputs"]
        own_run = run is getattr(app, "_recorded_run", None)
        reports = getattr(app, "_recorded_reports", None)
        if reports is None or len(reports) != len(candidates) or run is not getattr(app, "_recorded_run", None):
            # Report formatters need atom labels only, not bond detection.
            mol = SimpleNamespace(elements=inputs["elements"])
            reports = [(format_rigid_body_report(c, mol, rank=i+1, n_total=len(candidates))
                        if c.get("mode") == "rigid" else format_report(c, c["selected_bonds"], mol,
                        c.get("locked_bonds"), rank=i+1, n_total=len(candidates)))
                       for i, c in enumerate(candidates)]
            if run is getattr(app, "_recorded_run", None):
                app._recorded_reports = list(reports)
        selected = getattr(app, "_candidate_var", None)
        index = int(selected.get()) if selected is not None and own_run else 0
        if not 0 <= index < len(candidates):
            index = 0
        selecting_preview = preview_index is not None
        if preview_index is None:
            preview_index = -1
            preview = self.state.get("conformer_preview")
            if preview:
                for i, c in enumerate(candidates):
                    if np.array_equal(c["coords_opt"], preview["coords_opt"]):
                        preview_index = i
                        break
        return parse_payload(plain(dict(schema_version=1, inputs=inputs,
            context=run["context"], candidates=candidates, reports=reports,
            controls=self.controls() if own_run else inputs["controls"],
            selected_index=index, preview_index=preview_index,
            subtab_index=app._nb.index("current") if own_run else 3,
            editor_graph=plain(dict(bond_options=app.mol.bond_options,
                manual_add=sorted(app.mol.manual_add_bonds), manual_remove=sorted(app.mol.manual_remove_bonds),
                metal_index=app.metal_idx)) if own_run else {k:inputs[k] for k in
                ("bond_options", "manual_add", "manual_remove", "metal_index")},
            ui_fixed=sorted(app._get_fixed_set()) if own_run else inputs["fixed"],
            ui_bonds=[b for b, v in zip(app.rotatable_all, app.bond_vars) if v.get()] if own_run else inputs["selected_bonds"],
            applied=False if selecting_preview else bool(self.state.get("conformer_applied", False)))))

    def capture(self):
        if self.restoring:
            return
        document = active_document(self.state)
        if document is not None and self.document_id != document.document_id:
            return
        record = self.export()
        if record != self.state.get("conformer_search_session"):
            self.state["conformer_search_session"] = record
            self.restored_record = record
            mark_document_mutated(self.state, "conformer_search_session")

    def can_apply(self):
        if self.state.get("_conformer_search_restore_error"):
            return False
        record = self.state.get("conformer_search_session")
        if record is None:
            return True
        if not context_matches(record, host_context(self.state)):
            return False
        app, inputs = self.app, record["inputs"]
        if app.mol is None or app.mol.elements != inputs["elements"] or app.metal_idx != inputs["metal_index"]:
            return False
        if not np.allclose(app.mol.coords, inputs["coords"], rtol=0, atol=1e-7):
            return False
        graph = plain(dict(bond_options=app.mol.bond_options, manual_add=sorted(app.mol.manual_add_bonds),
                           manual_remove=sorted(app.mol.manual_remove_bonds)))
        if any(graph[k] != inputs[k] for k in graph) or sorted(app._get_fixed_set()) != inputs["fixed"]:
            return False
        if inputs["controls"].get("_search_mode_var") == "dihedral":
            bonds = [list(b) for b,v,locked in zip(app.rotatable_all,app.bond_vars,app.bond_locked) if v.get() and not locked]
            if bonds != inputs["selected_bonds"]:
                return False
        fields = ["_metal_idx_var", "_mx_var", "_my_var", "_mz_var", "_dchi_ax_var", "_dchi_rh_var",
                  "_ea_var", "_eb_var", "_eg_var", "_search_mode_var"]
        fields += (["_constraint_mode_var", "_anchor_mode_var"] if inputs["controls"].get("_search_mode_var") == "dihedral"
                   else ["_rb_axis_mode_var", "_rb_lock_mode_var"])
        if inputs["controls"].get("_rb_lock_mode_var") == "angular_radial_free" and inputs["controls"].get("_search_mode_var") == "rigid":
            fields += ["_rb_bounds_lo_var", "_rb_bounds_hi_var"]
        current = self.controls()
        for key in fields:
            if key not in inputs["controls"]:
                return False
            old, new = inputs["controls"][key], current[key]
            try:
                equal = math.isclose(float(old), float(new), rel_tol=1e-10, abs_tol=1e-7)
            except (TypeError, ValueError):
                equal = old == new
            if not equal:
                return False
        return True

    def reset_for_sync(self):
        """Replacing search inputs explicitly invalidates the previous set."""
        self.state["conformer_search_session"] = None
        for key in ("conformer_preview", "conformer_preview_coords", "conformer_preview_elements", "conformer_preview_report"):
            self.state.pop(key, None)
        self.restored_record = object()
        self.restore()
        mark_document_mutated(self.state, "conformer_search_inputs")

    def restore(self):
        from ui.conformer_search import Molecule
        document = active_document(self.state)
        doc_id = document.document_id if document else None
        record = self.state.get("conformer_search_session")
        if record is self.restored_record and self.document_id == doc_id:
            return
        app = self.app
        self.restoring = True
        self.state.pop("_conformer_search_restore_error", None)
        try:
            self.document_id = doc_id
            self.restored_record = record
            app.candidates, app.result = [], None
            app._recorded_run = None
            app._recorded_reports = None
            app._restored_pcs = None
            if not app._is_running:
                app._worker_document_identity = None
            app.mol = None
            app.metal_idx = None
            app.rotatable_all, app.bond_vars, app.bond_locked, app._fix_rows = [], [], [], []
            for panel in (app._fix_inner, app._bond_inner, app._candidate_selector_frame):
                for widget in panel.winfo_children():
                    widget.destroy()
            app._candidate_selector_frame.pack_forget()
            app._result_box.config(state="normal")
            app._result_box.delete("1.0", "end")
            app._result_box.config(state="disabled")
            if record is None:
                for name, value in self.defaults.items():
                    getattr(app, name).set(value)
                app._xyz_var.set("")
                app._pcs_var.set("")
                app._on_search_mode_change()
                app._status_var.set("Load structure and PCS, then sync or run a search.")
                return
            record = parse_payload(record)
            inputs = record["inputs"]
            graph = record["editor_graph"]
            app.mol = Molecule(inputs["elements"], np.asarray(inputs["coords"], float),
                               bond_options=graph["bond_options"],
                               manual_add_bonds={tuple(b) for b in graph["manual_add"]},
                               manual_remove_bonds={tuple(b) for b in graph["manual_remove"]})
            for name, value in record["controls"].items():
                if name in CONTROL_NAMES:
                    getattr(app, name).set(value)
            app._on_search_mode_change()
            editor_metal_text = app._metal_idx_var.get()
            app._metal_idx_var.set(str(graph["metal_index"]+1))
            app._xyz_var.set("Saved project structure")
            app._pcs_var.set("Saved project PCS")
            app._restored_pcs = deepcopy(inputs["pcs"])
            app._recorded_run = dict(inputs=deepcopy(inputs), context=deepcopy(record["context"]))
            app._detect_all()
            app._metal_idx_var.set(editor_metal_text)
            for variable, _, idx, _ in app._fix_rows:
                variable.set(idx in record["ui_fixed"])
            app._refresh_bond_locks()
            chosen = {tuple(b) for b in record["ui_bonds"]}
            for bond, variable in zip(app.rotatable_all, app.bond_vars):
                variable.set(tuple(bond) in chosen)
            app.candidates = candidate_values(record)
            app._recorded_reports = list(record["reports"])
            if len(app.candidates) > 1:
                app._candidate_selector_frame.pack(fill="x", padx=8, pady=(6, 2))
                app._build_candidate_selector()
            if hasattr(app, "_candidate_var"):
                app._candidate_var.set(record["selected_index"])
            app._display_candidate_report(record["selected_index"])
            app._nb.select(record["subtab_index"])
            app._refresh_imported_summary()
            app._refresh_search_setup_summary()
            app._status_var.set("Restored search result. " +
                ("Current inputs match." if self.can_apply() else "Inputs changed; Apply is unavailable."))
            status = self.state.get("conformer_status_var")
            if status is not None:
                index = record["preview_index"]
                status.set("Restored results; no candidate queued." if index < 0 else
                    f"Restored {'applied' if record['applied'] else 'preview'} candidate #{index+1}.")
        except Exception as exc:
            self.state["_conformer_search_restore_error"] = str(exc)
            self.restored_record = None
            app._status_var.set(f"Search results could not be restored: {exc}")
            raise
        finally:
            self.restoring = False
