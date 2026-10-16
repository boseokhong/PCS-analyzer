"""MC workspace and thin main-thread adapters to detached numerical inputs."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
import queue
import threading
from uuid import uuid4
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from logic.conformer_fitting import ConformerSnapshot, MultiConformerSnapshot, ConformerBasisCache, population_start
from logic.conformer_workflow import (
    default_options, normalize_options, execute_fit, result_status, fitted_scalar_changes,
    snapshot_from_payload,
)
from logic.pcs_fit_snapshot import FitGeometrySnapshot, FitTensorSnapshot, snapshot_torsion_group
from logic.pcs_document import (
    active_document, active_document_view, mark_document_mutated, capture_worker_identity,
    apply_worker_document_updates, worker_completion_matches,
)
from logic.pcs_tensor_state import ApplicationPCSTensor, PCS_TENSOR_STATE_KEY, canonical_tensor_from_legacy
from logic.pcs_model_state import update_canonical_tensor
from logic.xyz_loader import load_structure
from logic.fitting import _fit_observations
from ui.scrollable_workspace import VerticalScrolledWorkspace
from ui.widget_style import ui_font_name, ui_native_colors
from ui.style import get_app_fonts


PREFIX = "analysis.advanced.multiconformer."


def current_geometry(state):
    """Full raw/effective geometry, independent of main visibility/selection filters."""
    raw = state.get("atom_data_raw") or state.get("atom_data") or []
    refs = state.get("atom_ids_raw") or state.get("atom_ids") or []
    effective = state.get("atom_data_eff") or raw
    effective_refs = state.get("atom_ids_eff") or refs
    if not raw or len(raw) != len(refs) or len(effective) != len(effective_refs):
        raise ValueError("Load a main structure with explicit Ref IDs first")
    enabled = state.get("torsion_avg_enabled_var")
    enabled = enabled.get() if hasattr(enabled, "get") else bool(enabled)
    geometry = FitGeometrySnapshot(tuple(refs), (), tuple(refs), (), (0., 0., 0.), (), False, ())
    geometry = replace(geometry, effective_ref_ids=tuple(effective_refs),
                       effective_coordinates=tuple(tuple(a[1:4]) for a in effective),
                       raw_coordinates=tuple(tuple(a[1:4]) for a in raw),
                       metal_position=tuple(float(state.get(k, 0.)) for k in ("x0", "y0", "z0")),
                       pseudo_members=tuple((int(r), tuple(m)) for r, m in
                                            (state.get("symavg_members_by_pseudo_id") or {}).items()),
                       torsion_enabled=bool(enabled),
                       torsion_groups=tuple(snapshot_torsion_group(g) for g in state.get("torsion_avg_groups", [])))
    return geometry, tuple(a[0] for a in raw)


def collect_observations(state, selected=None):
    geometry, _ = current_geometry(state)
    available = set(geometry.raw_ref_ids) | set(geometry.effective_ref_ids)
    if state.get("experimental_pcs_observations"):
        return _fit_observations(state, proton_ids=(), observation_ids=selected, available_ref_ids=available)
    values = state.get("delta_exp_values") or {}
    if selected is not None and set(selected) - {f"legacy-ref:{r}" for r in values if r in available}:
        raise ValueError("Selected legacy PCS observations are no longer available")
    refs = [r for r in values if selected is None or f"legacy-ref:{r}" in selected]
    return _fit_observations(state, proton_ids=refs, observation_ids=None, available_ref_ids=available)


def compile_snapshot(state, options=None):
    """Called only on the main thread; workers receive its immutable return value."""
    options = normalize_options(state.get("advfit_mc_options") if options is None else options)
    rows = state.get("advfit_conformers", [])
    mapped = []
    for row in rows:
        if isinstance(row, ConformerSnapshot):
            mapped.append(row)
        elif isinstance(row, dict) and not row.get("included", True):
            continue
        else:
            raise ValueError("Configure atom mapping, metal and frame for every included conformer")
    if len([c for c in mapped if c.included]) < 2:
        raise ValueError("At least two included, mapped conformers are required")
    if state.get("fit_override"):
        raise ValueError("An active geometry override is unsupported by this fixed-frame model; clear it before fitting")
    if options["alignment"] == "confirmed_common_frame" and any(
            c.included and c.frame_provenance.startswith("raw_for_core") for c in mapped):
        raise ValueError("Choose metal-centered core alignment in Fit settings for raw-frame conformers")
    observations = collect_observations(state, options["observation_ids"])
    geometry, elements = current_geometry(state)
    main_identity = dict(zip(geometry.raw_ref_ids, elements))
    if any(c.included and dict(zip(c.geometry.raw_ref_ids, c.elements)) != main_identity for c in mapped):
        raise ValueError("Conformer Ref/element mappings differ from the main reference structure")
    canonical = state.get(PCS_TENSOR_STATE_KEY)
    if not isinstance(canonical, ApplicationPCSTensor):
        canonical = canonical_tensor_from_legacy(dict(state))
    document = active_document(state)
    snapshot = MultiConformerSnapshot(
        tuple(mapped), observations, orientation=canonical.to_forward_tensor().frame,
        tensor=FitTensorSnapshot(options["fit_ax"], options["fit_rh"], options["dchi_ax"], options["dchi_rh"]),
        alignment_mode=options["alignment"], core_ref_ids=tuple(options["core_ids"]),
        weighting_mode=options["weighting"], run_id=str(uuid4()),
        document_id=document.document_id if document else "", document_revision=document.revision if document else 0)
    if len(observations) < 3:
        raise ValueError("Select at least three matched PCS observations")
    population_start(snapshot, options["initialization"])
    return snapshot


class MultiConformerWorkspace:
    def __init__(self, state, notebook):
        self.state, self.root, self.alive = state, state["root"], True
        self._poll_id = None
        self._last_plot = None
        self._font_refresh_id = None
        self._basis_cache = ConformerBasisCache()
        self._basis_lock = threading.Lock()
        self.tab = ttk.Frame(notebook, padding=0)
        notebook.add(self.tab, text="Multi-Conformer")
        state.setdefault("advfit_conformers", [])
        state.setdefault("advfit_mc_options", default_options())
        self.scroll = VerticalScrolledWorkspace(self.tab, padding=0)
        self.scroll.pack(fill=tk.BOTH, expand=True)
        ui_native_colors(self.scroll.canvas, background="surface")
        content = self.scroll.content
        header = tk.Frame(content, borderwidth=0)
        self.header = header
        ui_native_colors(header, background="selected")
        header.pack(fill=tk.X, pady=(0, 3))
        self.toggle = tk.Button(header, text="▾ Conformers", command=self.collapse,
                               anchor="w", relief="flat", borderwidth=0, highlightthickness=1,
                               padx=10, pady=6, font=ui_font_name(header, "ui_bold"))
        ui_native_colors(self.toggle, background="selected", foreground="text", activebackground="pressed",
                         activeforeground="text", highlightbackground="selected", highlightcolor="accent")
        self.toggle.pack(side=tk.LEFT)
        # A sibling native Label avoids native Button overpainting its text.
        self.summary = tk.Label(header, anchor="e", borderwidth=0, font=ui_font_name(header, "ui"))
        ui_native_colors(self.summary, background="selected", foreground="text")
        self.summary.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8)
        self.summary.bind("<Button-1>", lambda _e: self.collapse())
        header.bind("<Button-1>", lambda _e: self.collapse())
        self.toggle.bind("<Return>", lambda _e: self.collapse())
        self.split = ttk.Panedwindow(content, orient=tk.VERTICAL, style="PCS.TPanedwindow", height=570)
        self.split.pack(fill=tk.BOTH, expand=True)
        self.table_frame = ttk.Frame(self.split, padding=0, height=185, width=1)
        self.table_frame.pack_propagate(False)
        self.report_frame = ttk.Frame(self.split, padding=0, height=145, width=1)
        self.report_frame.pack_propagate(False)
        self.plot_frame = ttk.Frame(self.split, padding=0, height=235, width=1)
        self.plot_frame.pack_propagate(False)
        self.split.add(self.table_frame, weight=0)
        self.split.add(self.report_frame, weight=0)
        self.split.add(self.plot_frame, weight=1)
        self.tree = ttk.Treeview(self.table_frame, columns=("use", "name", "mapping", "mode", "initial", "fitted"),
                                 show="headings", height=7, selectmode="browse")
        for key, title, width in (("use", "Included", 65), ("name", "Conformer", 190),
                                  ("mapping", "Mapping / frame", 165), ("mode", "Population", 85),
                                  ("initial", "Initial / fixed", 100), ("fitted", "Recorded fit", 100)):
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, minwidth=50, anchor="w" if key in ("name", "mapping") else "center")
        yscroll = ttk.Scrollbar(self.table_frame, command=self.tree.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=yscroll.set)
        xscroll = ttk.Scrollbar(self.table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        xscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.configure(xscrollcommand=xscroll.set)
        ttk.Label(self.table_frame, text="Double-click or press Enter to edit a row. Shared settings and actions are in Ribbon.",
                  padding=(4, 2)).pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", lambda _e: self.edit_selected())
        self.tree.bind("<Return>", lambda _e: self.edit_selected())
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.refresh_commands())
        ttk.Label(self.report_frame, text="Result", font=ui_font_name(content, "ui_bold"),
                  padding=(4, 3)).pack(anchor="w")
        self.report = tk.Text(self.report_frame, height=6, width=1, wrap="none", padx=8, pady=4,
                              font=ui_font_name(content, "report"), state="disabled")
        ry = ttk.Scrollbar(self.report_frame, command=self.report.yview)
        ry.pack(side=tk.RIGHT, fill=tk.Y)
        rx = ttk.Scrollbar(self.report_frame, orient=tk.HORIZONTAL, command=self.report.xview)
        rx.pack(side=tk.BOTTOM, fill=tk.X)
        self.report.configure(yscrollcommand=ry.set, xscrollcommand=rx.set)
        self.report.pack(fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(8, 2.6), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().configure(width=1, height=1)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.collapsed, self.saved_table_height = False, 185
        self.split.bind("<Configure>", self._restore_layout, add="+")
        self.split.bind("<ButtonRelease-1>", self._remember_layout, add="+")
        self._table_height, self._report_height = 185, 145
        self.scroll.canvas.bind("<Configure>", self._fit_viewport, add="+")
        self.header.bind("<Configure>", self._fit_viewport, add="+")
        self.tab.bind("<Destroy>", self._destroy, add="+")
        self.tab.bind("<<ThemeChanged>>", self._theme_changed, add="+")
        state["advfit_mc_tree"] = self.tree
        state["advfit_mc_result_text"] = self.report
        state["advfit_mc_fig"], state["advfit_mc_canvas"] = self.figure, self.canvas
        state["mc_workspace"] = self
        state["can_run_mc_fit"], state["can_apply_mc_result"] = self.can_run, self.can_apply
        state.setdefault("follow_active_document_refresh_hooks", []).append(self.refresh)
        self.refresh()

    def _restore_layout(self, _event=None):
        # Restore during parent Configure, before a larger intermediate plot allocation.
        if self.collapsed:
            self.split.sashpos(0, self._report_height)
        else:
            self.split.sashpos(0, self._table_height)
            self.split.sashpos(1, self._table_height + self._report_height)

    def _remember_layout(self, _event=None):
        if self.collapsed:
            self._report_height = max(80, self.split.sashpos(0))
        else:
            self._table_height = max(100, self.split.sashpos(0))
            self._report_height = max(80, self.split.sashpos(1) - self._table_height)
        self._fit_viewport()

    def _fit_viewport(self, _event=None):
        """Allocate unused viewport height to plots; keep short-window scrolling."""
        minimum = self._report_height + 235 + (0 if self.collapsed else self._table_height)
        available = self.scroll.canvas.winfo_height() - self.header.winfo_reqheight() - 3
        height = max(minimum, available)
        if int(self.split.cget("height")) != height:
            self.split.configure(height=height)

    def collapse(self):
        if self.collapsed:
            self.split.insert(0, self.table_frame, weight=0)
            self.toggle.configure(text="▾ Conformers")
        else:
            self._remember_layout()
            self.split.forget(self.table_frame)
            self.toggle.configure(text="▸ Conformers")
        self.collapsed = not self.collapsed
        self._fit_viewport()
        self._restore_layout()

    def _destroy(self, event):
        if event.widget is self.tab:
            self.alive = False
            if self._poll_id:
                self.root.after_cancel(self._poll_id)
            if self._font_refresh_id:
                self.root.after_cancel(self._font_refresh_id)
            hooks = self.state.get("follow_active_document_refresh_hooks", [])
            if self.refresh in hooks:
                hooks.remove(self.refresh)

    def _theme_changed(self, _event=None):
        if self._font_refresh_id is None and self.alive:
            def refresh_fonts():
                self._font_refresh_id = None
                fonts = get_app_fonts(self.state)
                key = tuple(fonts.get(k) for k in ("plot_family", "plot_title", "plot_label", "plot_tick"))
                if key != getattr(self, "_plot_font_key", None):
                    self.draw_result(self.state.get("advfit_mc_last_result"))
            self._font_refresh_id = self.root.after_idle(refresh_fonts)

    def options(self):
        return normalize_options(self.state.get("advfit_mc_options"))

    def mutate(self, key, value):
        try:
            active_document_view(self.state)[key] = value
        except RuntimeError:
            self.state[key] = value
        mark_document_mutated(self.state, "multi_conformer_inputs")
        self.refresh()
        self.refresh_commands()

    def refresh_commands(self):
        callback = self.state.get("refresh_fitting_commands")
        if callable(callback):
            callback()

    def selected_row(self):
        ids = self.tree.selection()
        if not ids:
            return None
        return next((c for c in self.state.get("advfit_conformers", []) if self.row_id(c) == ids[0]), None)

    @staticmethod
    def row_id(row):
        return row.conformer_id if isinstance(row, ConformerSnapshot) else row["conformer_id"]

    def can_run(self):
        if self.state.get("advfit_mc_running"):
            return False
        try:
            compile_snapshot(self.state)
            return True
        except (ValueError, RuntimeError, KeyError, TypeError):
            return False

    def current_status(self):
        result = self.state.get("advfit_mc_last_result")
        try:
            snapshot = compile_snapshot(self.state)
        except (ValueError, RuntimeError, KeyError, TypeError):
            snapshot = None
        return result_status(result, snapshot, self.options())

    def can_apply(self):
        result = self.state.get("advfit_mc_last_result")
        return bool(not self.state.get("advfit_mc_running") and fitted_scalar_changes(result)
                    and self.current_status() == "Current")

    def can_evaluate(self):
        result = self.state.get("advfit_mc_last_result")
        if (self.state.get("advfit_mc_running") or not result or not result.get("success")
                or result.get("format") != "pcs_multiconformer_result_v1"):
            return False
        try:
            current = compile_snapshot(self.state)
        except (ValueError, RuntimeError, KeyError, TypeError):
            return False
        # Unweighted fit objectives ignore sigma; uncertainty evaluation does not.
        return bool(result_status(result, current, self.options()) == "Current" and
                    [o.uncertainty_ppm for o in current.observations] ==
                    [o["uncertainty_ppm"] for o in result["input_snapshot"]["observations"]])

    def refresh(self):
        if not self.alive:
            return
        selected = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        result = self.state.get("advfit_mc_last_result")
        fitted = dict(zip(result["conformer_ids"], result["best"]["populations"])) if result and result.get("best") else {}
        for row in self.state.get("advfit_conformers", []):
            rid = self.row_id(row)
            if isinstance(row, ConformerSnapshot):
                values = ("Yes" if row.included else "No", row.name, "Mapped / " + ("core" if row.frame_provenance.startswith("raw_for_core") else "confirmed"),
                          "Fixed" if row.population_fixed else "Fit", f"{row.initial_population:.6g}",
                          f"{fitted[rid]:.4f}" if rid in fitted else "—")
            else:
                values = ("Yes" if row.get("included", True) else "No", row["name"], "Needs mapping / frame", "Fit", "1", "—")
            self.tree.insert("", "end", iid=rid, values=values)
        if selected and self.tree.exists(selected[0]):
            self.tree.selection_set(selected[0])
        options = self.options()
        try:
            count = len(collect_observations(self.state, options["observation_ids"]))
        except (ValueError, RuntimeError, KeyError, TypeError):
            count = 0
        names = {"both": "Tensor + populations", "tensor": "Tensor only", "populations": "Populations only"}
        self.summary.configure(text=f"{names[options['target']]} / {len(self.state.get('advfit_conformers', []))} conformers / {count} observations")
        if self.state.get("advfit_mc_running"):
            self.summary.configure(text=self.summary.cget("text") + " · " + self.state.get("_mc_job_label", "Fitting…"))
        self.present_result(result)

    def present_result(self, result):
        lines = ["Add conformer structures, configure their mappings/frames and select PCS observations.",
                 "Choose fit conditions in Ribbon, then click Run Fit."]
        if result and result.get("format") == "pcs_multiconformer_result_v1":
            snapshot = snapshot_from_payload(result["input_snapshot"])
            options = result["options"]
            lines = [f"Multi-Conformer · {self.current_status()} · {'Completed' if result['success'] else 'Failed'}",
                     f"Recorded target: {options['target']} / weighting: {options['weighting']}",
                     f"Fixed orientation: {snapshot.orientation.euler_angles} / {snapshot.orientation.convention.value}",
                     f"Alignment: {snapshot.alignment_mode} / core Ref: {snapshot.core_ref_ids or 'none'}",
                     f"Observations: {len(snapshot.observations)} / seed: {options['seed']} / attempted starts: {len(result['runs'])}"]
            best = result.get("best")
            if best:
                lines += [f"Δχ_ax = {best['dchi_ax']:+.6g} E-32 m³ ({'fitted' if snapshot.tensor.fit_delta_chi_ax and options['target'] != 'populations' else 'fixed'})",
                          f"Δχ_rh = {best['dchi_rh']:+.6g} E-32 m³ ({'fitted' if snapshot.tensor.fit_delta_chi_rh and options['target'] != 'populations' else 'fixed'})",
                          f"RMSD = {best['rmsd']:.6g} ppm / objective = {best['objective']:.6g}", best["message"],
                          f"Population ambiguity: {result['population_ambiguous']} / alternative starts: {result['alternative_indices']}", "", "Recorded conformer conditions:"]
                ids = result["conformer_ids"]
                for c in snapshot.conformers:
                    if c.included:
                        index = ids.index(c.conformer_id)
                        start = best["initial_populations"][index]
                        final = best["populations"][index]
                        lines.append(f"C{index+1} {c.name}: supplied={c.initial_population:.6g}, start={start:.6g}, fitted={final:.6g} ({'fixed' if c.population_fixed or options['target']=='tensor' else 'fit'})")
                        lines.append(f"  Metal: {c.geometry.metal_position} Å / input frame: {c.coordinate_frame} / pseudo groups: {len(c.geometry.pseudo_members)} / torsion: {c.geometry.torsion_enabled}")
                        if snapshot.alignment_mode == "metal_centered_core":
                            lines.append(f"  Alignment core RMSD: {result['alignment'][index]['rmsd_angstrom']:.6g} Å")
                if result.get("reliability"):
                    lines += self.reliability_report(result["reliability"])
                else:
                    lines += ["", "Reliability: not evaluated. Use Reliability in Ribbon after reviewing the fit."]
                lines += ["", "Observation / member Ref IDs / experimental / predicted / residual (ppm)"]
                for obs, pred, res in zip(snapshot.observations, best["predictions"], best["residuals"]):
                    lines.append(f"{obs.assignment or obs.observable_id}\t{obs.target.member_ref_ids}\t{obs.pcs:+.6g}\t{pred:+.6g}\t{res:+.6g}")
            else:
                lines += [f"Start {i+1}: {r['message']} / objective={r['objective']:.6g}" for i, r in enumerate(result["runs"])]
        elif result:
            lines = ["Historical legacy result: Unknown. Refit with explicit mapping/frame before Apply."]
        if self.state.get("advfit_mc_running"):
            document = active_document(self.state)
            elsewhere = document is not None and self.state.get("_mc_running_origin") != document.document_id
            lines.insert(0, "A Multi-Conformer job is running in another document." if elsewhere else
                         self.state.get("_mc_job_label", "Fitting…") + " Previous recorded results remain below.")
        self.report.configure(state="normal")
        self.report.delete("1.0", tk.END)
        self.report.insert(tk.END, "\n".join(lines))
        self.report.configure(state="disabled")
        plot_key = json.dumps(result, sort_keys=True, default=str) if result else "empty"
        if plot_key != self._last_plot:
            self._last_plot = plot_key
            self.draw_result(result)

    def draw_result(self, result):
        self.figure.clear()
        fonts = get_app_fonts(self.state)
        self._plot_font_key = tuple(fonts.get(k) for k in ("plot_family", "plot_title", "plot_label", "plot_tick"))
        best = result.get("best") if result else None
        if not best:
            self.figure.text(.5, .55, "Multi-Conformer PCS fitting", ha="center", fontsize=fonts.get("plot_title", 9), fontfamily=fonts.get("plot_family"))
            self.figure.text(.5, .4, "Prepare mapped structures and PCS observations.\nThen click Run Fit in Ribbon.",
                             ha="center", va="center", fontsize=fonts.get("plot_label", 8), fontfamily=fonts.get("plot_family"))
        else:
            snapshot = snapshot_from_payload(result["input_snapshot"])
            exp = np.array([o.pcs for o in snapshot.observations])
            pred = np.array(best["predictions"])
            axes = self.figure.subplots(1, 3)
            lo, hi = float(min(exp.min(), pred.min())), float(max(exp.max(), pred.max()))
            margin = max((hi-lo)*.05, .01)
            axes[0].plot((lo-margin, hi+margin), (lo-margin, hi+margin), "--", color="gray", linewidth=1)
            axes[0].scatter(exp, pred, s=18)
            axes[0].set(title="PCS correlation", xlabel=r"Experimental $\delta_{Exp}$ (ppm)", ylabel=r"Predicted $\delta_{Pred}$ (ppm)")
            axes[1].bar(range(1, len(exp)+1), best["residuals"])
            axes[1].axhline(0, color="gray", linewidth=.8)
            axes[1].set(title="PCS residuals", xlabel="Observation", ylabel=r"$\delta_{Pred}-\delta_{Exp}$ (ppm)")
            axes[2].bar([f"C{i+1}" for i in range(len(best["populations"]))], best["populations"])
            reliability = result.get("reliability")
            if reliability:
                for i, item in enumerate(reliability["bootstrap"]["intervals"]["populations"]):
                    if item["interval"] is not None:
                        lower, upper = item["interval"]
                        axes[2].vlines(i, lower, upper, color="#23354d", linewidth=1.2)
                        axes[2].hlines((lower, upper), i-.07, i+.07, color="#23354d", linewidth=1.2)
            axes[2].set(title="Conformer populations", ylabel="Population", ylim=(0, 1))
            if reliability and any(item["interval"] is not None for item in reliability["bootstrap"]["intervals"]["populations"]):
                axes[2].set_title(f"Populations ({reliability['options']['confidence']*100:g}% intervals)")
            for ax in axes:
                ax.title.set_fontsize(fonts.get("plot_title", 9))
                ax.title.set_fontfamily(fonts.get("plot_family", "DejaVu Sans"))
                ax.xaxis.label.set_fontsize(fonts.get("plot_label", 8))
                ax.yaxis.label.set_fontsize(fonts.get("plot_label", 8))
                ax.xaxis.label.set_fontfamily(fonts.get("plot_family", "DejaVu Sans"))
                ax.yaxis.label.set_fontfamily(fonts.get("plot_family", "DejaVu Sans"))
                ax.tick_params(labelsize=fonts.get("plot_tick", 7))
                for label in ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()]:
                    label.set_fontfamily(fonts.get("plot_family", "DejaVu Sans"))
                    label.set_fontsize(fonts.get("plot_tick", 7))
        self.canvas.draw_idle()

    @staticmethod
    def reliability_report(reliability):
        bootstrap, cv = reliability["bootstrap"], reliability["cross_validation"]
        confidence = reliability["options"]["confidence"]*100
        lines = ["", "Reliability evaluation (recorded fit)",
                 f"Bootstrap: {reliability['options']['method']} / {bootstrap['successful']}/{bootstrap['requested']} successful / seed {reliability['options']['seed']}",
                 reliability["assumption"], f"Approximate marginal {confidence:g}% percentile intervals:"]
        for name, item in bootstrap["intervals"]["tensor"].items():
            interval = item["interval"]
            text = "fixed / not estimated" if item["fixed"] else (f"[{interval[0]:+.6g}, {interval[1]:+.6g}] E-32 m³" if interval else "withheld")
            lines.append(f"  {name}: {text}")
        for item in bootstrap["intervals"]["populations"]:
            interval = item["interval"]
            text = "fixed / not estimated" if item["fixed"] else (f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval else "withheld")
            lines.append(f"  {item['conformer_id']}: {text}")
        metric = f"{cv['rmsd_ppm']:.6g} ppm" if cv["rmsd_ppm"] is not None else "unavailable"
        lines += [f"Grouped CV: {cv['successful_folds']}/{len(cv['folds'])} folds / {cv['predicted_observations']} observations / RMSD {metric}" + (" (partial)" if not cv["complete"] else "")]
        for group in reliability["suggested_groups"]:
            interval = group["bootstrap_range"]
            text = f" / descriptive range [{interval[0]:.4f}, {interval[1]:.4f}]" if interval else ""
            lines.append(f"Similar PCS group {', '.join(group['conformer_ids'])}: combined population {group['population']:.4f}{text}")
        lines += ["Caution: " + warning for warning in reliability["warnings"]]
        return lines

    def use_current(self):
        try:
            geometry, elements = current_geometry(self.state)
            c = ConformerSnapshot(str(uuid4()), "Current structure", geometry, elements,
                                  "Main structure explicit Ref mapping", "main_raw_structure")
            self.mutate("advfit_conformers", list(self.state.get("advfit_conformers", [])) + [c])
        except (ValueError, RuntimeError, KeyError) as exc:
            self.error(exc)

    def load_xyz(self):
        path = filedialog.askopenfilename(parent=self.root, title="Load conformer structure",
                                         filetypes=[("Structure files", "*.xyz *.out *.log"), ("All files", "*.*")])
        if not path:
            return
        try:
            atoms = load_structure(path)
            row = {"format": "unmapped_xyz_v1", "conformer_id": str(uuid4()), "name": Path(path).stem,
                   "atoms": atoms, "source_path": path, "source_hash": sha256(Path(path).read_bytes()).hexdigest(), "included": True}
            from logic.conformer_workflow import conformer_payload
            row = conformer_payload(row)
            self.mutate("advfit_conformers", list(self.state.get("advfit_conformers", [])) + [row])
            self.tree.selection_set(row["conformer_id"])
            self.edit_selected()
        except (ValueError, RuntimeError, OSError) as exc:
            self.error(exc)

    def clear(self):
        self.mutate("advfit_conformers", [])

    def remove_selected(self):
        row = self.selected_row()
        if row is not None:
            self.mutate("advfit_conformers", [c for c in self.state.get("advfit_conformers", []) if self.row_id(c) != self.row_id(row)])

    def toggle_parameter(self, name):
        options = self.options()
        options[name] = not options[name]
        self.mutate("advfit_mc_options", options)

    def validate(self):
        try:
            snapshot = compile_snapshot(self.state)
            messagebox.showinfo("Multi-Conformer validation", f"{sum(c.included for c in snapshot.conformers)} mapped conformers / {len(snapshot.observations)} observations\nMapping, metal/frame and population conditions are valid.", parent=self.root)
        except (ValueError, RuntimeError, KeyError) as exc:
            self.error(exc)

    def run(self):
        if self.state.get("advfit_mc_running"):
            return
        try:
            snapshot, options = compile_snapshot(self.state), self.options()
            from logic.conformer_workflow import conformer_payload, conformer_from_payload
            rows = tuple(conformer_from_payload(conformer_payload(c)) for c in self.state.get("advfit_conformers", []))
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            self.error(exc)
            return
        cache, cache_lock = self._basis_cache, self._basis_lock

        def work(_progress):
            with cache_lock:
                basis = cache.get(snapshot)
            return execute_fit(snapshot, options, input_rows=rows, basis=basis)
        self._start_worker(work, "Fitting…")

    def _start_worker(self, work, label):
        identity = capture_worker_identity(self.state)
        self.state["advfit_mc_running"] = True
        self.state["_mc_running_origin"] = identity.document_id if identity else None
        self.state["_mc_job_label"] = label
        self.refresh()
        self.refresh_commands()
        results = queue.Queue()
        def worker():
            try:
                progress = lambda stage, done, total: results.put(("progress", stage, done, total))
                results.put(("result", work(progress), None))
            except Exception as exc:
                results.put(("result", None, f"{type(exc).__name__}: {exc}"))

        def poll():
            self._poll_id = None
            if not self.alive:
                return
            try:
                item = results.get_nowait()
                while item[0] == "progress":
                    _, stage, done, total = item
                    self.state["_mc_job_label"] = f"{stage.replace('_', ' ').capitalize()} {done}/{total}…"
                    if active_document(self.state) is None or self.state.get("_mc_running_origin") == active_document(self.state).document_id:
                        self.summary.configure(text=self.state["_mc_job_label"])
                    item = results.get_nowait()
                _, payload, error = item
            except queue.Empty:
                self._poll_id = self.root.after(40, poll)
                return
            self.complete(identity, payload, error)

        try:
            threading.Thread(target=worker, daemon=True).start()
            self._poll_id = self.root.after(40, poll)
        except RuntimeError as exc:
            self.state["advfit_mc_running"] = False
            self.state.pop("_mc_running_origin", None)
            self.state.pop("_mc_job_label", None)
            self.refresh_commands()
            self.error(exc)

    def complete(self, identity, payload, error=None):
        self.state["advfit_mc_running"] = False
        self.state.pop("_mc_running_origin", None)
        self.state.pop("_mc_job_label", None)
        if identity is not None and not worker_completion_matches(self.state, identity):
            self.refresh()
            self.refresh_commands()
            return False
        if error:
            if identity is None or (active_document(self.state) and active_document(self.state).document_id == identity.document_id):
                self.error(error)
            self.refresh()
            self.refresh_commands()
            return False
        if identity is None:
            self.state["advfit_mc_last_result"] = payload
        else:
            origin = apply_worker_document_updates(self.state, identity, {"advfit_mc_last_result": payload}, domain="multi_conformer_result")
            if origin is not active_document(self.state):
                self.refresh()
                self.refresh_commands()
                return origin is not None
            self.state["advfit_mc_last_result"] = payload
        self.refresh()
        self.refresh_commands()
        return True

    def apply(self):
        if not self.can_apply():
            self.error("Only a current, successful result with fitted tensor fields can be applied")
            return
        result = self.state["advfit_mc_last_result"]
        changes = fitted_scalar_changes(result)
        tensor = self.state.get(PCS_TENSOR_STATE_KEY)
        if not isinstance(tensor, ApplicationPCSTensor):
            tensor = canonical_tensor_from_legacy(self.state)
        update_canonical_tensor(self.state, replace(tensor, **changes), source="multi_conformer_apply",
                                legacy_callback=self.state.get("update_graph"))
        messagebox.showinfo("Multi-Conformer", "Applied fitted scalar fields only. Orientation, metal, rhombic enable state and prediction policy are preserved. Main views retain their single structure.", parent=self.root)
        self.refresh()
        self.refresh_commands()

    def export(self):
        result = self.state.get("advfit_mc_last_result")
        if not result:
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="Export recorded MC result", defaultextension=".json",
                                          filetypes=[("JSON", "*.json")])
        if path:
            try:
                Path(path).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
            except (OSError, ValueError) as exc:
                self.error(exc)

    def error(self, exc):
        messagebox.showerror("Multi-Conformer", str(exc), parent=self.root)

    def callbacks(self):
        return {PREFIX+key: callback for key, callback in {
            "input.use_current": self.use_current, "input.load_xyz": self.load_xyz,
            "input.clear": self.clear, "input.remove": self.remove_selected,
            "input.configure": self.edit_selected, "input.observations": self.select_observations,
            "input.validate": self.validate, "parameters.settings": self.settings,
            "parameters.fit_ax": lambda: self.toggle_parameter("fit_ax"),
            "parameters.fit_rh": lambda: self.toggle_parameter("fit_rh"),
            "results.reliability": self.reliability_settings,
            "run": self.run, "results.apply": self.apply, "output.export": self.export}.items()}

    def dialog(self, title):
        win = tk.Toplevel(self.root)
        win.withdraw()
        win.title(title)
        win.transient(self.root)
        body = ttk.Frame(win, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)
        return win, body, active_document(self.state)

    def show_dialog(self, win):
        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_reqwidth()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_reqheight()) // 2
        win.geometry(f"+{max(0, x)}+{max(0, y)}")
        win.deiconify()
        win.grab_set()
        win.bind("<Escape>", lambda _e: win.destroy())

    def edit_selected(self):
        row = self.selected_row()
        if row is None:
            return
        try:
            reference, _ = current_geometry(self.state)
        except ValueError as exc:
            self.error(exc)
            return
        mapped = isinstance(row, ConformerSnapshot)
        atoms = [(el, *xyz) for el, xyz in zip(row.elements, row.geometry.raw_coordinates)] if mapped else row["atoms"]
        initial_ids = row.geometry.raw_ref_ids if mapped else reference.raw_ref_ids if len(atoms) == len(reference.raw_ref_ids) else ()
        win, body, origin = self.dialog("Conformer mapping and population")
        fields = {}
        values = {
            "Name": row.name if mapped else row["name"],
            "Ref IDs in file row order": " ".join(map(str, initial_ids)),
            "Metal Ref (or blank for XYZ)": "" if mapped else str(self.state.get("metal_ref_id") or ""),
            "Metal XYZ (Å)": " ".join(map(str, row.geometry.metal_position if mapped else (0., 0., 0.))),
            "Initial / fixed population": str(row.initial_population if mapped else row.get("initial_population", 1.)),
        }
        for i, (label, value) in enumerate(values.items()):
            ttk.Label(body, text=label, padding=0).grid(row=i, column=0, sticky="w", padx=(0, 8), pady=3)
            var = tk.StringVar(master=win, value=value)
            ttk.Entry(body, textvariable=var, width=64 if i == 1 else 30).grid(row=i, column=1, sticky="ew", pady=3)
            fields[label] = var
        included = tk.BooleanVar(master=win, value=row.included if mapped else row.get("included", True))
        fixed = tk.BooleanVar(master=win, value=row.population_fixed if mapped else row.get("population_fixed", False))
        confirmed = tk.BooleanVar(master=win, value=mapped)
        verified = tk.BooleanVar(master=win, value=mapped)
        core = mapped and row.frame_provenance.startswith("raw_for_core")
        frame = tk.StringVar(master=win, value="Already tensor-oriented" if mapped and row.coordinate_frame == "tensor"
                             else "Raw; align using common core" if core else "Confirmed common frame")
        ttk.Label(body, text="Coordinate frame", padding=0).grid(row=5, column=0, sticky="w")
        ttk.Combobox(body, textvariable=frame, state="readonly",
                     values=("Confirmed common frame", "Raw; align using common core", "Already tensor-oriented")).grid(row=5, column=1, sticky="ew")
        ttk.Checkbutton(body, text="Include in fit", variable=included).grid(row=6, column=1, sticky="w")
        ttk.Checkbutton(body, text="Fixed population", variable=fixed).grid(row=7, column=1, sticky="w")
        ttk.Checkbutton(body, text="I confirm each Ref maps to the same chemical atom", variable=confirmed).grid(row=8, column=1, sticky="w")
        ttk.Checkbutton(body, text="I verified this file's own metal origin and frame", variable=verified).grid(row=9, column=1, sticky="w")
        ttk.Label(body, text="Equal atom counts/elements do not establish identity. Core alignment is configured in Ribbon Fit settings.",
                  wraplength=540, padding=(0, 6)).grid(row=10, column=0, columnspan=2, sticky="w")

        def save():
            if origin is not active_document(self.state):
                self.error("The active document changed; reopen this editor")
                return
            try:
                name = fields["Name"].get().strip()
                population = float(fields["Initial / fixed population"].get())
                if not name or not np.isfinite(population) or population < 0:
                    raise ValueError("A name and finite nonnegative population are required")
                if not mapped and not included.get() and not confirmed.get():
                    updated = dict(row, name=name, included=False, initial_population=population, population_fixed=fixed.get())
                else:
                    if not confirmed.get() or not verified.get():
                        raise ValueError("Explicit chemical mapping and metal/frame confirmation are required")
                    refs = tuple(int(x) for x in fields["Ref IDs in file row order"].get().replace(",", " ").split())
                    if len(refs) != len(atoms) or len(set(refs)) != len(refs):
                        raise ValueError("Provide one unique Ref ID for every file row")
                    raw_map = dict(zip(refs, [a[1:4] for a in atoms]))
                    metal_ref = fields["Metal Ref (or blank for XYZ)"].get().strip()
                    metal = raw_map[int(metal_ref)] if metal_ref else tuple(float(x) for x in fields["Metal XYZ (Å)"].get().split())
                    pseudo = reference.pseudo_member_mapping()
                    effective = []
                    for rid in reference.effective_ref_ids:
                        effective.append(tuple(np.mean([raw_map[m] for m in pseudo[rid]], axis=0)) if rid in pseudo else tuple(raw_map[rid]))
                    geometry = replace(reference, raw_ref_ids=refs, raw_coordinates=tuple(tuple(a[1:4]) for a in atoms),
                                       effective_coordinates=tuple(effective), metal_position=tuple(metal))
                    updated = ConformerSnapshot(self.row_id(row), name, geometry, tuple(a[0] for a in atoms),
                                                "User-confirmed explicit Ref mapping",
                                                "raw_for_core:user_verified" if frame.get().startswith("Raw") else "confirmed_common_frame:user_verified",
                                                coordinate_frame="tensor" if frame.get() == "Already tensor-oriented" else "structure",
                                                source_hash=row.source_hash if mapped else row["source_hash"],
                                                included=included.get(), initial_population=population, population_fixed=fixed.get())
                rows = [updated if self.row_id(c) == self.row_id(row) else c for c in self.state.get("advfit_conformers", [])]
                self.mutate("advfit_conformers", rows)
                win.destroy()
            except (ValueError, KeyError, TypeError) as exc:
                messagebox.showerror("Conformer mapping", str(exc), parent=win)
        actions = ttk.Frame(body, padding=0)
        actions.grid(row=11, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(actions, text="Save", command=save).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions, text="Cancel", command=win.destroy).pack(side=tk.LEFT, padx=3)
        self.show_dialog(win)

    def settings(self):
        options = self.options()
        win, body, origin = self.dialog("Multi-Conformer fit settings")
        fields = {}
        labels = (("target", "Fit target", ("both", "tensor", "populations")),
                  ("weighting", "Objective weighting", ("unweighted", "experimental_uncertainty")),
                  ("alignment", "Alignment", ("confirmed_common_frame", "metal_centered_core")),
                  ("initialization", "Population initialization", ("user", "uniform")),
                  ("dchi_ax", "Δχ_ax initial / fixed (E-32 m³)", None),
                  ("dchi_rh", "Δχ_rh initial / fixed (E-32 m³)", None),
                  ("core_ids", "Core Ref IDs (at least 3)", None), ("starts", "Multi-start count (1–20)", None),
                  ("seed", "Random seed", None))
        for i, (key, label, values) in enumerate(labels):
            var = tk.StringVar(master=win, value=" ".join(map(str, options[key])) if key == "core_ids" else str(options[key]))
            ttk.Label(body, text=label, padding=0).grid(row=i, column=0, sticky="w", padx=(0, 12), pady=3)
            control = ttk.Combobox(body, textvariable=var, values=values, state="readonly") if values else ttk.Entry(body, textvariable=var)
            control.grid(row=i, column=1, sticky="ew", pady=3)
            fields[key] = var
        flags = {key: tk.BooleanVar(master=win, value=options[key]) for key in ("fit_ax", "fit_rh")}
        for i, key in enumerate(flags, 9):
            ttk.Checkbutton(body, text="Fit Δχ_ax" if key == "fit_ax" else "Fit Δχ_rh", variable=flags[key]).grid(row=i, column=1, sticky="w")
        ttk.Label(body, text="Orientation stays fixed at the main tensor. Populations only fixes both scalar inputs; Tensor only fixes supplied populations. Evaluate reliability separately after fitting.",
                  wraplength=500, padding=(0, 6)).grid(row=11, column=0, columnspan=2)
        def save():
            if origin is not active_document(self.state):
                self.error("The active document changed; reopen settings")
                return
            try:
                updated = dict(options, **{k: v.get() for k, v in fields.items()}, **{k: v.get() for k, v in flags.items()})
                updated["core_ids"] = [int(x) for x in fields["core_ids"].get().replace(",", " ").split()]
                updated["starts"], updated["seed"] = int(updated["starts"]), int(updated["seed"])
                self.mutate("advfit_mc_options", normalize_options(updated))
                win.destroy()
            except ValueError as exc:
                messagebox.showerror("Fit settings", str(exc), parent=win)
        actions = ttk.Frame(body, padding=0)
        actions.grid(row=12, column=0, columnspan=2, sticky="e")
        ttk.Button(actions, text="Save", command=save).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions, text="Cancel", command=win.destroy).pack(side=tk.LEFT, padx=3)
        self.show_dialog(win)

    def reliability_settings(self):
        if not self.can_evaluate():
            self.error("Refit current inputs successfully before evaluating reliability")
            return
        from logic.conformer_reliability import ReliabilityOptions, observation_groups
        result = json.loads(json.dumps(self.state["advfit_mc_last_result"], allow_nan=False))
        snapshot = snapshot_from_payload(result["input_snapshot"])
        independent = all(len(g) == 1 for g in observation_groups(snapshot))
        known_sigma = all(o.uncertainty_ppm is not None and o.uncertainty_ppm > 0 for o in snapshot.observations)
        defaults = result.get("reliability", {}).get("options", {}) or dict(
            method="parametric" if known_sigma and independent else "cases", replicates=100,
            folds=max(2, min(5, len(observation_groups(snapshot)))), seed=2026, confidence=.95, similarity_relative=.01)
        win, body, origin = self.dialog("Multi-Conformer reliability")
        fields = {}
        specifications = (("method", "Bootstrap method", ("parametric", "cases")),
                          ("replicates", "Bootstrap refits (20–2000)", None),
                          ("folds", "Grouped CV folds (2–20)", None), ("seed", "Random seed", None),
                          ("confidence", "Marginal confidence (0–1)", None),
                          ("similarity_relative", "Relative PCS similarity threshold", None))
        for i, (key, title, choices) in enumerate(specifications):
            var = tk.StringVar(master=win, value=str(defaults[key]))
            fields[key] = var
            ttk.Label(body, text=title, padding=0).grid(row=i, column=0, sticky="w", padx=(0, 12), pady=3)
            control = ttk.Combobox(body, textvariable=var, values=choices, state="readonly") if choices else ttk.Entry(body, textvariable=var)
            control.grid(row=i, column=1, sticky="ew", pady=3)
        ttk.Label(body, text="Parametric: independent Gaussian errors with recorded positive σ. Cases: resample independent observation groups. All selected free parameters are refitted. Geometry/orientation and fixed values stay fixed. Percentile intervals are approximate; this can take several minutes.",
                  wraplength=520, padding=(0, 8)).grid(row=6, column=0, columnspan=2)
        def evaluate():
            if origin is not active_document(self.state) or not self.can_evaluate() or self.state["advfit_mc_last_result"] != result:
                self.error("The document or recorded fit changed; reopen Reliability")
                return
            try:
                options = ReliabilityOptions(method=fields["method"].get(),
                    **{key: int(fields[key].get()) for key in ("replicates", "folds", "seed")},
                    **{key: float(fields[key].get()) for key in ("confidence", "similarity_relative")})
                if options.folds > len(observation_groups(snapshot)):
                    raise ValueError("Too few independent observation groups for requested folds")
                if options.method == "parametric" and not (known_sigma and independent):
                    raise ValueError("Parametric bootstrap needs independent observations with positive known σ; use cases otherwise")
            except ValueError as exc:
                messagebox.showerror("Reliability", str(exc), parent=win)
                return
            win.destroy()
            self.run_reliability(result, options)
        actions = ttk.Frame(body, padding=0)
        actions.grid(row=7, column=0, columnspan=2, sticky="e")
        ttk.Button(actions, text="Evaluate", command=evaluate).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions, text="Cancel", command=win.destroy).pack(side=tk.LEFT, padx=3)
        self.show_dialog(win)

    def run_reliability(self, result, options):
        if not self.can_evaluate():
            return
        from logic.conformer_reliability import evaluate_reliability
        recorded = json.loads(json.dumps(result, allow_nan=False))
        def work(progress):
            recorded["reliability"] = evaluate_reliability(recorded, options, progress=progress)
            return recorded
        self._start_worker(work, "Evaluating reliability…")

    def select_observations(self):
        try:
            observations = collect_observations(self.state)
        except (ValueError, RuntimeError) as exc:
            self.error(exc)
            return
        win, body, origin = self.dialog("Select PCS observations")
        listing = tk.Listbox(body, selectmode=tk.EXTENDED, exportselection=False, height=12, width=70,
                             font=ui_font_name(body, "ui"))
        scrollbar = ttk.Scrollbar(body, command=listing.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        listing.configure(yscrollcommand=scrollbar.set)
        listing.grid(row=0, column=0, sticky="nsew")
        selected = self.options()["observation_ids"]
        for i, obs in enumerate(observations):
            listing.insert(tk.END, f"{obs.assignment or obs.observable_id} / Ref {obs.target.member_ref_ids} / {obs.pcs:+.6g} ppm")
            if selected is None or obs.observable_id in selected:
                listing.selection_set(i)
        def save():
            if origin is not active_document(self.state):
                self.error("The active document changed; reopen selection")
                return
            options = self.options()
            options["observation_ids"] = [observations[i].observable_id for i in listing.curselection()]
            self.mutate("advfit_mc_options", options)
            win.destroy()
        actions = ttk.Frame(body, padding=(0, 6))
        actions.grid(row=1, column=0, columnspan=2, sticky="e")
        ttk.Button(actions, text="Select all", command=lambda: listing.selection_set(0, tk.END)).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions, text="Save", command=save).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions, text="Cancel", command=win.destroy).pack(side=tk.LEFT, padx=3)
        self.show_dialog(win)
