"""VT plots, local fit selections and exports; numerical work stays in M6."""

import tkinter as tk
from tkinter import filedialog, ttk

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

from .presentation import WorkbenchFigureCanvas
from .vt_analysis_model import COORDINATES, FIT_MODELS, VTAnalysisModel, format_value
from nmr_workbench.analysis.vt_models import VTModelResult, MODELS, model_curve
from .actions import action, action_menu, divider, polish


class VTAnalysisWindow(tk.Toplevel):
    def __init__(self, parent, state, *, on_activate, on_changed, on_show_stack):
        super().__init__(parent)
        self.title("VT Analysis — observed chemical shifts")
        self.transient(parent)
        self.geometry("1180x820")
        self.minsize(940, 680)
        self._presentation = parent._presentation
        self.model = VTAnalysisModel(state)
        self.on_activate, self.on_changed, self.on_show_stack = on_activate, on_changed, on_show_stack
        self._refreshing = False
        self._series_ids, self._track_ids, self._row_ids = [], [], {}
        self._targets, self._press = [], None
        self._plot_key = None
        self.coordinate = tk.StringVar(self, value=self.model.coordinate)
        self.fit_model = tk.StringVar(self, value="Linear (coordinate)")
        self.weighting = tk.StringVar(self, value="unweighted")
        self.theta_low, self.theta_high = tk.StringVar(self), tk.StringVar(self)
        self.compare_models = {m: tk.BooleanVar(self) for m in MODELS}
        self.low, self.high = tk.StringVar(self), tk.StringVar(self)
        self.status = tk.StringVar(self, value="Observed δ only; no PCS/contact interpretation. Fit masks are analysis-local.")
        body = ttk.Frame(self, padding=8)
        body.pack(fill="both", expand=True)
        controls = ttk.Frame(body)
        controls.pack(fill="x")
        self.export_menu = action_menu(controls, "Export", "export", (
            ("Analysis table CSV…", self._export_csv), ("Plot image / PDF…", self._export_plot),
            ("Results JSON…", self._export_json), None, ("Keep results in session", self._save_evidence)))
        self.export_menu.pack(side="right", padx=3)
        self.workflow_menu = action_menu(controls, "Workflow", "toolbox", (
            ("Assignment review…", parent._open_vt_observables), ("VT PCS preparation…", parent._open_vt_pcs),
            ("Assignments / saved results…", parent._open_data_exchange), None,
            ("Workflow & file guide…", parent._open_workflow_help)))
        self.workflow_menu.pack(side="right", padx=3)
        action(controls, "Refresh", "update", self.refresh, description="Refresh analysis").pack(side="right", padx=4)
        ttk.Label(controls, text="Series").pack(side="left", padx=(0, 6))
        self.series_selector = ttk.Combobox(controls, state="readonly", width=19)
        self.series_selector.pack(side="left", fill="x", expand=True)
        self.series_selector.bind("<<ComboboxSelected>>", self._series_changed)
        coordinates = ttk.Frame(body)
        coordinates.pack(fill="x", pady=(8, 0))
        ttk.Label(coordinates, text="Coordinate").pack(side="left", padx=(0, 4))
        selector = ttk.Combobox(coordinates, state="readonly", width=15, values=tuple(COORDINATES), textvariable=self.coordinate)
        selector.pack(side="left", padx=4)
        selector.bind("<<ComboboxSelected>>", self._coordinate_changed)
        divider(coordinates)
        ttk.Label(coordinates, text="Fit range / K").pack(side="left", padx=4)
        for variable in (self.low, self.high):
            ttk.Entry(coordinates, width=7, textvariable=variable).pack(side="left", padx=2)
        action(coordinates, "Apply", "apply", self._range_changed, description="Apply fit range").pack(side="left", padx=4)
        action(coordinates, "Reset", "reset", self._reset, description="Reset masks and range").pack(side="left")
        models = ttk.Frame(body)
        models.pack(fill="x", pady=4)
        ttk.Label(models, text="Fit observed δ:").pack(side="left")
        model_selector = ttk.Combobox(models, state="readonly", width=20, values=tuple(FIT_MODELS), textvariable=self.fit_model)
        model_selector.pack(side="left", padx=4)
        model_selector.bind("<<ComboboxSelected>>", self._model_changed)
        weights = ttk.Combobox(models, state="readonly", width=16, values=("unweighted", "measured_sigma"), textvariable=self.weighting)
        weights.pack(side="left")
        weights.bind("<<ComboboxSelected>>", self._model_changed)
        ttk.Label(models, text="θ bounds / K").pack(side="left", padx=4)
        for variable in (self.theta_low, self.theta_high):
            ttk.Entry(models, width=7, textvariable=variable).pack(side="left", padx=2)
        action(models, "Apply", "apply", self._model_changed,
               description="Apply model options. Blank θ bounds use guarded defaults.").pack(side="left", padx=4)
        middle = ttk.Frame(body)
        middle.pack(fill="both", expand=True, pady=6)
        sidebar = ttk.Frame(middle, width=250)
        sidebar.pack(side="left", fill="y", padx=(0, 8))
        ttk.Label(sidebar, text="Tracks — Ctrl/Shift for overlay").pack(anchor="w")
        self.tracks = tk.Listbox(sidebar, selectmode="extended", exportselection=False, height=6, width=28)
        self.tracks.pack(fill="x", pady=4)
        self.tracks.bind("<<ListboxSelect>>", self._tracks_changed)
        self.summary = ttk.Treeview(sidebar, columns=("value",), show="tree headings", height=9)
        self.summary.heading("#0", text="Track / fit")
        self.summary.heading("value", text="Value")
        self.summary.column("#0", width=115)
        self.summary.column("value", width=105)
        self.summary.pack(fill="both", expand=True)
        plot = ttk.Frame(middle)
        plot.pack(side="left", fill="both", expand=True)
        self.figure = Figure(figsize=(8, 5))
        self.figure.subplots_adjust(left=.14, right=.98, bottom=.12, top=.94, hspace=.15)
        self.axes, self.residual_axes = self.figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": (3, 1)})
        self.canvas = WorkbenchFigureCanvas(self.figure, master=plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot, pack_toolbar=False)
        self.toolbar.pack(fill="x")
        self._cids = (self.canvas.mpl_connect("button_press_event", self._on_press),
                      self.canvas.mpl_connect("button_release_event", self._on_release))
        tabs = ttk.Notebook(body)
        tabs.pack(fill="x")
        observations, warnings = ttk.Frame(tabs), ttk.Frame(tabs)
        tabs.add(observations, text="Observations / fit mask")
        tabs.add(warnings, text="Diagnostics")
        comparison = ttk.Frame(tabs)
        tabs.add(comparison, text="Compare models (raw δ / ppm)")
        comparison_controls = ttk.Frame(comparison)
        comparison_controls.pack(fill="x")
        for model_id, label in MODELS.items():
            ttk.Checkbutton(comparison_controls, text=label, variable=self.compare_models[model_id],
                command=self._comparison_changed).pack(side="left", padx=5)
        ttk.Label(comparison_controls, text="Same points/weights; fit quality is not physical proof.").pack(side="left")
        comparison_table = ttk.Frame(comparison)
        comparison_table.pack(fill="both", expand=True)
        self.comparison = ttk.Treeview(comparison_table, columns=("track", "model", "rss", "rmse", "r2", "aic", "aicc", "n", "status"), show="headings", height=4)
        for name in self.comparison["columns"]:
            self.comparison.heading(name, text=name.upper())
            self.comparison.column(name, width=100)
        comparison_scroll = ttk.Scrollbar(comparison_table, command=self.comparison.yview)
        self.comparison.configure(yscrollcommand=comparison_scroll.set)
        comparison_scroll.pack(side="right", fill="y")
        self.comparison.pack(fill="both", expand=True)
        self.points = ttk.Treeview(observations, columns=("track", "member", "temperature", "shift", "x", "y", "state", "fit", "residual", "reason"),
                                  show="headings", height=4, selectmode="browse")
        for name, title, width in (("track", "Track", 125), ("member", "Member", 115), ("temperature", "T / K", 70),
                ("shift", "δ / ppm", 75), ("x", "x", 80), ("y", "y", 80), ("state", "State", 80),
                ("fit", "Fit", 50), ("residual", "Residual", 80), ("reason", "Exclusion / review", 230)):
            self.points.heading(name, text=title)
            self.points.column(name, width=width, stretch=name in {"track", "member", "reason"})
        self.points.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(observations, command=self.points.yview)
        scroll.pack(side="right", fill="y")
        self.points.configure(yscrollcommand=scroll.set)
        self.points.bind("<<TreeviewSelect>>", self._point_selected)
        self.points.tag_configure("excluded", foreground="#B45309")
        self.diagnostics = tk.Text(warnings, height=6, wrap="word", state="disabled")
        self.diagnostics.pack(fill="both", expand=True)
        actions = ttk.Frame(body)
        actions.pack(side="bottom", fill="x", pady=4, before=middle)
        action(actions, "Include in fit", "check", lambda: self._mask(True)).pack(side="left", padx=3)
        action(actions, "Exclude from fit", "remove", lambda: self._mask(False)).pack(side="left", padx=3)
        action(actions, "Show spectrum", "nmr_workbench", self._show_stack).pack(side="right")
        ttk.Label(body, textvariable=self.status, wraplength=1130, style="NMR.Muted.TLabel").pack(side="bottom", fill="x", before=actions)
        # Reserve the review table/footer before giving the plot remaining space.
        tabs.pack_configure(side="bottom", before=middle)
        polish(self)
        self.refresh()

    def _run(self, action):
        try:
            action()
        except (ValueError, KeyError, OSError) as exc:
            self.status.set(str(exc))
            return False
        self.refresh()
        return True

    def refresh(self):
        self.model.refresh()
        self._refreshing = True
        try:
            state, series = self.model.state, self.model.active_series
            self._series_ids = [s.id for s in state.series]
            self.series_selector.configure(values=[s.name for s in state.series])
            self.series_selector.set("")
            if state.selected_series_id in self._series_ids:
                self.series_selector.current(self._series_ids.index(state.selected_series_id))
            self._track_ids = [t.id for t in series.tracks] if series else []
            labels = [f"{t.label or 'Track'} [{t.id[:8]}]" for t in series.tracks] if series else []
            self.tracks.delete(0, "end")
            for index, label in enumerate(labels):
                self.tracks.insert("end", label)
                if self._track_ids[index] in self.model.selected_track_ids:
                    self.tracks.selection_set(index)
            bounds = self.model.temperature_range_k
            if self.focus_get() not in (None,) and self.focus_get().winfo_class() == "TEntry":
                pass  # Preserve an unfinished range edit during host refreshes.
            else:
                self.low.set(str(bounds[0]) if bounds else "")
                self.high.set(str(bounds[1]) if bounds else "")
            self.summary.delete(*self.summary.get_children())
            self.points.delete(*self.points.get_children())
            self._row_ids = {}
            warnings = []
            self.comparison.delete(*self.comparison.get_children())
            for group in self.model.comparison_results:
                for fit in group:
                    self.comparison.insert("", "end", values=(fit.coordinates.data.track_label or fit.coordinates.data.track_id[:8],
                        MODELS[fit.config.model_id], format_value(fit.rss), format_value(fit.rmse), format_value(fit.r_squared),
                        format_value(fit.aic), format_value(fit.aicc), fit.point_count, fit.status))
                    warnings.extend(f"{fit.coordinates.data.track_id[:8]} / {fit.config.model_id}: {d.code} — {d.message}" for d in fit.diagnostics)
            for result in self.model.results:
                data = result.coordinates.data
                label = f"{data.track_label or 'Track'} [{data.track_id[:8]}]"
                item = self.summary.insert("", "end", text=label, open=True)
                for title, value in (("Slope", format_value(result.slope)), ("Intercept", format_value(result.intercept)),
                        ("R²", format_value(result.r_squared)), ("n", str(result.point_count)), ("Status", result.status)):
                    self.summary.insert(item, "end", text=title, values=(value,))
                if isinstance(result, VTModelResult):
                    self.summary.insert(item, "end", text="Model / weights", values=(f"{MODELS[result.config.model_id]} / {result.config.weighting}",))
                    for parameter in result.parameters:
                        self.summary.insert(item, "end", text=f"{parameter.name} / {parameter.unit}",
                            values=(f"{format_value(parameter.value)} ± {format_value(parameter.standard_error)}",))
                    for title, value in (("RSS / ppm²", result.rss), ("RMSE / ppm", result.rmse), ("AIC", result.aic), ("AICc", result.aicc)):
                        self.summary.insert(item, "end", text=title, values=(format_value(value),))
                    warnings.extend(f"{label}: Assumption — {text}" for text in result.assumptions)
                residuals = dict(zip(result.included_identities, result.residuals))
                for point in result.coordinates.points:
                    identity, observed = point.observation.identity, point.observation
                    iid = str(len(self._row_ids))
                    self._row_ids[iid] = identity
                    reason = ", ".join(dict.fromkeys(point.exclusion_reasons + tuple(d.code for d in observed.diagnostics)))
                    self.points.insert("", "end", iid=iid, values=(label,
                        observed.member_label or identity.member_id[:8], format_value(observed.temperature_k),
                        format_value(observed.chemical_shift_ppm), format_value(point.x), format_value(point.y),
                        observed.point_state, "yes" if point.included else "no", format_value(residuals.get(identity)), reason),
                        tags=() if point.included else ("excluded",))
                    if identity == self.model.selected_identity:
                        self.points.selection_set(iid)
                        self.points.see(iid)
                warnings.extend(f"{label}: {d.code} — {d.message}" for d in result.diagnostics)
            self.diagnostics.configure(state="normal")
            self.diagnostics.delete("1.0", "end")
            self.diagnostics.insert("1.0", "\n".join(warnings) or "No diagnostics. Select a series and tracks to analyze.")
            self.diagnostics.configure(state="disabled")
            self._draw()
        finally:
            self._refreshing = False

    def _draw(self):
        key = (self.model.results, self.model.selected_identity)
        if key == self._plot_key:
            return
        same_data = self._plot_key is not None and self._plot_key[0] == self.model.results
        limits = (self.axes.get_xlim(), self.axes.get_ylim(), self.residual_axes.get_ylim()) if same_data else None
        self._plot_key = key
        self.axes.clear()
        self.residual_axes.clear()
        self._targets = []
        colors = ("#2459A6", "#D97706", "#15803D", "#7C3AED", "#DC2626", "#0891B2")
        for index, result in enumerate(self.model.results):
            color = colors[index % len(colors)]
            label = f"{result.coordinates.data.track_label or 'Track'} [{result.coordinates.data.track_id[:8]}]"
            for included, marker in ((True, "o"), (False, "x")):
                points = [p for p in result.coordinates.points if p.included == included and p.x is not None and p.y is not None]
                if points:
                    self.axes.plot([p.x for p in points], [p.y for p in points], linestyle="none", marker=marker,
                        color=color, markersize=5, alpha=1 if included else .55, label=label + ("" if included else " (not fitted)"))
                    self._targets.extend((self.axes, p.x, p.y, p.observation.identity) for p in points)
            fitted = dict(zip(result.included_identities, zip(result.fitted_values, result.residuals)))
            ordered = sorted(((p.x, *fitted[p.observation.identity], p.observation.identity)
                for p in result.coordinates.points if p.observation.identity in fitted), key=lambda row: row[0])
            if ordered:
                curve_x, curve_y = model_curve(result) if isinstance(result, VTModelResult) else ([p[0] for p in ordered], [p[1] for p in ordered])
                self.axes.plot(curve_x, curve_y, color=color, linewidth=1.2)
                self.residual_axes.plot([p[0] for p in ordered], [p[2] for p in ordered], linestyle="none", marker="o", color=color, markersize=4)
                self._targets.extend((self.residual_axes, p[0], p[2], p[3]) for p in ordered)
        domain = self.model.results[0].domain if self.model.results else None
        self.axes.set_title(f"Observed chemical shifts · {self.fit_model.get()} · × not fitted")
        self.axes.set_ylabel("δT / ppm K" if domain and domain.name == "deltaT_vs_inverse_T" else "δ / ppm")
        self.residual_axes.set_xlabel("1/T / K⁻¹" if domain and domain.x_quantity == "inverse_temperature" else "T / K")
        self.residual_axes.set_ylabel("Residual\n" + (domain.y_unit if domain else "ppm"))
        self.residual_axes.axhline(0, linewidth=.8, color="#6B7280")
        for axis in (self.axes, self.residual_axes):
            axis.grid(True, alpha=.15)
            axis.margins(.08)
        if self.axes.get_legend_handles_labels()[0]:
            self.axes.legend(fontsize="small")
        else:
            self.axes.text(.5, .5, "Select tracks with plottable observations.\nInvalid rows remain available in the table.", transform=self.axes.transAxes, ha="center")
        for axis, x, y, identity in self._targets:
            if identity == self.model.selected_identity:
                axis.plot([x], [y], marker="o", linestyle="none", markersize=10,
                    markerfacecolor="none", color="#111827")
        if limits:
            self.axes.set_xlim(limits[0])
            self.axes.set_ylim(limits[1])
            self.residual_axes.set_ylim(limits[2])
        else:
            self.toolbar.update()
            self.toolbar.push_current()
        self.canvas.draw_idle()

    def _series_changed(self, event=None):
        index = self.series_selector.current()
        if index >= 0:
            self.model.select_series(self._series_ids[index])
            self.refresh()
            self.on_changed()

    def _tracks_changed(self, event=None):
        if self._refreshing:
            return
        ids = tuple(self._track_ids[i] for i in self.tracks.curselection())
        if ids == self.model.selected_track_ids:
            return
        self.model.select_tracks(ids)
        self.refresh()
        self.on_changed()

    def _coordinate_changed(self, event=None):
        self._run(lambda: self.model.set_coordinate(self.coordinate.get()))

    def _model_changed(self, event=None):
        def apply():
            bounds = None if not self.theta_low.get().strip() and not self.theta_high.get().strip() else (float(self.theta_low.get()), float(self.theta_high.get()))
            self.model.set_model(FIT_MODELS[self.fit_model.get()], weighting=self.weighting.get(), theta_bounds_k=bounds)
        if not self._run(apply):
            self.fit_model.set(next(label for label, model_id in FIT_MODELS.items() if model_id == self.model.fit_model))
            self.weighting.set(self.model.weighting)

    def _comparison_changed(self):
        self._run(lambda: self.model.set_comparison([m for m, selected in self.compare_models.items() if selected.get()]))

    def _save_evidence(self):
        def save():
            record = self.model.save_evidence()
            self.on_changed()
            self.status.set(f"Saved historical fit evidence {record.id[:8]}; save the NMR session to retain it.")
        self._run(save)

    def _export_json(self):
        path = filedialog.asksaveasfilename(parent=self, title="Export VT fit evidence", defaultextension=".json", filetypes=(("JSON", "*.json"),))
        if path:
            def export():
                self.model.export_json(path)
                self.status.set("Exported results JSON. Workflow → Assignments / saved results opens historical evidence; live fits stay separate.")
            self._run(export)

    def _range_changed(self):
        def apply():
            bounds = None if not self.low.get().strip() and not self.high.get().strip() else (float(self.low.get()), float(self.high.get()))
            self.model.set_range(bounds)
        self._run(apply)

    def _reset(self):
        self._plot_key = None
        self._run(self.model.reset_masks)

    def _identity(self):
        selection = self.points.selection()
        return self._row_ids.get(selection[0]) if selection else None

    def _point_selected(self, event=None):
        identity = self._identity()
        if self._refreshing or identity is None or identity == self.model.selected_identity:
            return
        self.model.selected_identity = identity
        def activate():
            dataset_id, peak_id = self.model.navigation(identity)
            self.on_activate(dataset_id, peak_id)
            self.status.set(f"Track {identity.track_id[:8]} · member {identity.member_id[:8]} · dataset {dataset_id[:8]} · peak {peak_id or 'unavailable'}")
        self._run(activate)

    def _mask(self, included):
        identity = self._identity()
        if identity:
            self.model.selected_identity = identity
            self._run(lambda: self.model.set_included(identity, included))

    def _show_stack(self):
        identity = self._identity()
        if identity:
            def show():
                dataset_id, peak_id = self.model.navigation(identity)
                self.on_activate(dataset_id, peak_id)
                self.on_show_stack()
            self._run(show)

    def _on_press(self, event):
        self._press = (event.x, event.y) if event.button == 1 and event.inaxes in (self.axes, self.residual_axes) and not self.toolbar.mode else None

    def _on_release(self, event):
        press, self._press = self._press, None
        if press is None or event.button != 1 or self.toolbar.mode or abs(event.x-press[0]) + abs(event.y-press[1]) > 4:
            return
        targets = []
        for axis, x, y, identity in self._targets:
            if axis is event.inaxes:
                px, py = axis.transData.transform((x, y))
                distance = (event.x-px)**2 + (event.y-py)**2
                if distance <= 64:
                    targets.append((distance, identity))
        if targets:
            identity = min(targets, key=lambda item: item[0])[1]
            iid = next(k for k, value in self._row_ids.items() if value == identity)
            self.points.selection_set(iid)
            self.points.see(iid)
            self._point_selected()

    def _export_csv(self):
        path = filedialog.asksaveasfilename(parent=self, title="Export current VT analysis", defaultextension=".csv", filetypes=(("CSV", "*.csv"),))
        if path:
            def export():
                self.model.export_csv(path)
                self.status.set("Exported analysis report CSV with assignments, identities, masks and residuals. Save session to resume scientific work.")
            self._run(export)

    def save_plot(self, path):
        self.canvas.draw()  # Apply the same Workbench presentation used on screen.
        self.figure.savefig(path, dpi=300, bbox_inches="tight", facecolor=self.figure.get_facecolor())

    def _export_plot(self):
        path = filedialog.asksaveasfilename(parent=self, title="Export current VT plot", defaultextension=".png",
            filetypes=(("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")))
        if path:
            self._run(lambda: self.save_plot(path))

    def destroy(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        super().destroy()
