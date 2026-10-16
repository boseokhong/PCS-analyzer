"""Narrow presentation adapter for the embedded Conformer Search UI."""

from __future__ import annotations

import os


class ConformerSearchPresentationAdapter:
    """Expose command callbacks/readiness without moving search state ownership."""

    def __init__(self, app):
        self.app = app

    @property
    def notebook(self):
        return self.app._nb

    def set_state_changed_callback(self, callback):
        self.app._presentation_state_changed = callback

    def _invoke(self, callback, *args):
        try:
            return callback(*args)
        finally:
            self.app._notify_presentation_state()

    def has_structure(self) -> bool:
        return self.app.mol is not None

    def fixed_actions_ready(self) -> bool:
        return self.has_structure() and bool(self.app._fix_rows)

    def lock_geometry_ready(self) -> bool:
        return self.has_structure() and bool(self.app.rotatable_all)

    def bond_selection_ready(self) -> bool:
        return self.has_structure() and bool(self.app.bond_vars)

    def results_available(self) -> bool:
        return bool(getattr(self.app, "candidates", None)) and self.app.result is not None

    def run_ready(self) -> bool:
        if self.app._is_running or not self.has_structure():
            return False
        pcs_path = self.app._pcs_var.get().strip()
        if not getattr(self.app, "_restored_pcs", None) and (not pcs_path or not os.path.isfile(pcs_path)):
            return False
        mode = self.app._search_mode_var.get().strip().lower()
        if mode == "rigid":
            return bool(
                set(self.app._get_fixed_set()) - {self.app.metal_idx}
            )
        return any(
            var.get() and not locked
            for var, locked in zip(self.app.bond_vars, self.app.bond_locked)
        )

    def callbacks(self):
        return {
            "analysis.conformer.fixed.apply_filter": (
                lambda: self._invoke(self.app._apply_fixed_filter)
            ),
            "analysis.conformer.fixed.fix_all_shown": (
                lambda: self._invoke(self.app._set_fixed_all, True)
            ),
            "analysis.conformer.fixed.unfix_all": (
                lambda: self._invoke(self.app._set_fixed_all, False)
            ),
            "analysis.conformer.fixed.invert": (
                lambda: self._invoke(self.app._invert_fixed)
            ),
            "analysis.conformer.fixed.apply_locked_bonds": (
                lambda: self._invoke(self.app._refresh_bond_locks)
            ),
            "analysis.conformer.bonds.select_all_free": (
                lambda: self._set_bonds("all")
            ),
            "analysis.conformer.bonds.select_none": (
                lambda: self._set_bonds("none")
            ),
            "analysis.conformer.bonds.invert": (
                lambda: self._set_bonds("invert")
            ),
            "analysis.conformer.bonds.skipped_contacts": (
                lambda: self._invoke(self.app._show_skipped_contacts)
            ),
            "analysis.conformer.bonds.add": (
                lambda: self._invoke(self.app._add_manual_bond_dialog)
            ),
            "analysis.conformer.bonds.remove": (
                lambda: self._invoke(self.app._remove_manual_bond_dialog)
            ),
            "analysis.conformer.search.run": (
                lambda: self._invoke(self.app._run_search)
            ),
            "analysis.conformer.results.view_structure": (
                lambda: self._invoke(self.app._show_structure_plot)
            ),
            "analysis.conformer.results.select_candidate": (
                lambda: self._invoke(self.app._apply_selected)
            ),
            "analysis.conformer.results.correlation_plot": (
                lambda: self._invoke(self.app._show_plot)
            ),
            "analysis.conformer.output.save_xyz": (
                lambda: self._invoke(self.app._save_xyz)
            ),
            "analysis.conformer.output.save_report": (
                lambda: self._invoke(self.app._save_report)
            ),
        }

    def _set_bonds(self, operation: str):
        for var, locked in zip(self.app.bond_vars, self.app.bond_locked):
            if operation == "none":
                var.set(False)
            elif not locked and operation == "all":
                var.set(True)
            elif not locked and operation == "invert":
                var.set(not var.get())
        self.app._refresh_search_setup_summary()
        self.app._notify_presentation_state()


__all__ = ["ConformerSearchPresentationAdapter"]
