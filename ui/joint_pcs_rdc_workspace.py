"""Joint PCS/RDC workspace; all Tk access stays on the main thread."""
from dataclasses import replace
from pathlib import Path
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from logic.joint_pcs_rdc_workflow import (
    JointSnapshot, RDCObservation, default_options, normalize_options,
    parse_rdc_csv, execute_fit, result_status,
)
from logic.pcs_document import (active_document, active_document_view, mark_document_mutated,
    capture_worker_identity, worker_completion_matches, apply_worker_document_updates)
from logic.pcs_tensor_state import ApplicationPCSTensor, PCS_TENSOR_STATE_KEY, canonical_tensor_from_legacy
from logic.pcs_model_state import update_canonical_tensor
from ui.multiconformer_workspace import current_geometry, collect_observations
from ui.scrollable_workspace import VerticalScrolledWorkspace
from ui.widget_style import ui_font_name, ui_native_colors
from ui.style import get_app_fonts

PREFIX = "analysis.advanced.rdc."


def compile_snapshot(state):
    """Capture raw Ref geometry and structured observations without visibility filters."""
    options = normalize_options(state.get("advfit_rdc_options"))
    if state.get("fit_override"):
        raise ValueError("Clear the geometry override before fixed-frame joint fitting")
    geometry, elements = current_geometry(state)
    observations = tuple(collect_observations(state, options["observation_ids"]))
    tensor = state.get(PCS_TENSOR_STATE_KEY)
    if not isinstance(tensor, ApplicationPCSTensor):
        tensor = canonical_tensor_from_legacy(dict(state))
    labels = {}
    overrides = state.get("ref_label_overrides") or {}
    for rid, element in zip(geometry.raw_ref_ids, elements):
        label = str(overrides.get(rid) or f"{element}{rid}")
        labels.setdefault(label, []).append(rid)
    available, bonds, skipped = set(geometry.raw_ref_ids), [], []
    for index, row in enumerate(state.get("advfit_rdc_rows") or [], 1):
        refs = []
        for side in (1, 2):
            if f"ref{side}" in row:
                refs.append(int(row[f"ref{side}"]))
            else:
                candidates = labels.get(str(row.get(f"atom{side}", "")), [])
                if len(candidates) > 1:
                    raise ValueError("Ambiguous RDC atom label; use explicit ref1/ref2 columns")
                refs.append(candidates[0] if candidates else None)
        source_row = int(row.get("source_row", index))
        if any(r not in available for r in refs):
            skipped.append(f"RDC row {source_row}: unmatched raw atom Ref/label")
            continue
        bonds.append(RDCObservation(*refs, float(row["rdc_exp"]), float(row.get("rdc_err", .5)),
                    str(row.get("nuc1", "H")).upper(), str(row.get("nuc2", "N")).upper(), source_row))
    return JointSnapshot(geometry, tensor.to_forward_tensor().frame, observations, tuple(bonds),
                         options["B0"], options["T"], options["w_rdc"], options["use_pcs_errors"], tuple(skipped))


class JointPCSRDCWorkspace:
    def __init__(self, state, notebook):
        self.state, self.root, self.alive = state, state["root"], True
        self.poll_id = self.font_id = None
        self.last_plot = None
        self.view = "Correlation"
        self.collapsed = False
        self.table_height, self.report_height = 190, 165
        self.tab = ttk.Frame(notebook)
        notebook.add(self.tab, text="Joint PCS+RDC")
        state.setdefault("advfit_rdc_rows", [])
        state.setdefault("advfit_rdc_options", default_options())
        self.scroll = VerticalScrolledWorkspace(self.tab, padding=0)
        self.scroll.pack(fill=tk.BOTH, expand=True)
        ui_native_colors(self.scroll.canvas, background="surface")
        content = self.scroll.content
        header = tk.Frame(content, borderwidth=0)
        self.header = header
        ui_native_colors(header, background="selected")
        header.pack(fill=tk.X, pady=(0, 3))
        self.toggle = tk.Button(header, text="▾ Input data", command=self.collapse, anchor="w",
                               relief="flat", borderwidth=0, highlightthickness=1, padx=10, pady=6,
                               font=ui_font_name(header, "ui_bold"))
        ui_native_colors(self.toggle, background="selected", foreground="text", activebackground="pressed",
                         activeforeground="text", highlightbackground="selected", highlightcolor="accent")
        self.toggle.pack(side=tk.LEFT)
        self.summary = tk.Label(header, anchor="e", font=ui_font_name(header, "ui"), borderwidth=0)
        ui_native_colors(self.summary, background="selected", foreground="text")
        self.summary.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8)
        self.summary.bind("<Button-1>", lambda _e: self.collapse())
        self.toggle.bind("<Return>", lambda _e: self.collapse())
        self.split = ttk.Panedwindow(content, orient=tk.VERTICAL, style="PCS.TPanedwindow", height=630)
        self.split.pack(fill=tk.BOTH, expand=True)
        self.input_frame = ttk.Frame(self.split, height=190, width=1)
        self.report_frame = ttk.Frame(self.split, height=165, width=1)
        self.plot_frame = ttk.Frame(self.split, height=275, width=1)
        for pane, weight in ((self.input_frame, 0), (self.report_frame, 0), (self.plot_frame, 1)):
            pane.pack_propagate(False)
            self.split.add(pane, weight=weight)
        self.input_frame.columnconfigure((0, 1), weight=1, uniform="joint-input")
        self.input_frame.rowconfigure(0, weight=1)
        self.pcs_tree = self.build_tree(0, "PCS observations", ("id", "value", "sigma"),
                                        ("Observation / Ref", "δ_Exp (ppm)", "σ (ppm)"))
        self.rdc_tree = self.build_tree(1, "RDC observations", ("id", "value", "sigma"),
                                        ("Raw Ref pair", "D_Exp (Hz)", "σ (Hz)"))
        self.matching = ttk.Label(self.input_frame, text="Load RDC data from the Ribbon", padding=(4, 3))
        self.matching.grid(row=1, column=0, columnspan=2, sticky="ew")
        report_header = ttk.Frame(self.report_frame)
        report_header.pack(fill=tk.X)
        ttk.Label(report_header, text="Result", font=ui_font_name(content, "ui_bold"),
                  padding=(4, 3)).pack(side=tk.LEFT)
        self.status = ttk.Label(report_header, text="Not fitted", padding=4)
        self.status.pack(side=tk.RIGHT)
        self.report = tk.Text(self.report_frame, height=6, width=1, wrap="none", padx=8, pady=4,
                              font=ui_font_name(content, "report"), state="disabled")
        self.text_scrollbars(self.report_frame, self.report)
        self.report.pack(fill=tk.BOTH, expand=True)
        bar = ttk.Frame(self.plot_frame)
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="Diagnostics", font=ui_font_name(content, "ui_bold"), padding=4).pack(side=tk.LEFT)
        self.view_var = tk.StringVar(master=self.root, value=self.view)
        views = ttk.Combobox(bar, textvariable=self.view_var, values=("Correlation", "Residuals"),
                             state="readonly", width=14)
        views.pack(side=tk.RIGHT, padx=4)
        views.bind("<<ComboboxSelected>>", lambda _e: self.draw_result(force=True))
        self.figure = Figure(figsize=(8, 2.8), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().configure(width=1, height=1)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.split.bind("<Configure>", self.restore_layout, add="+")
        self.split.bind("<ButtonRelease-1>", self.remember_layout, add="+")
        self.scroll.canvas.bind("<Configure>", self.fit_viewport, add="+")
        self.header.bind("<Configure>", self.fit_viewport, add="+")
        self.tab.bind("<Destroy>", self.destroy, add="+")
        self.tab.bind("<<ThemeChanged>>", self.theme_changed, add="+")
        state.update(rdc_workspace=self, can_run_rdc_fit=self.can_run, can_apply_rdc_result=self.can_apply,
                     advfit_rdc_result_text=self.report, advfit_rdc_fig=self.figure, advfit_rdc_canvas=self.canvas)
        state.setdefault("follow_active_document_refresh_hooks", []).append(self.refresh)
        self.refresh()

    @staticmethod
    def text_scrollbars(parent, widget):
        y = ttk.Scrollbar(parent, command=widget.yview)
        y.pack(side=tk.RIGHT, fill=tk.Y)
        x = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=widget.xview)
        x.pack(side=tk.BOTTOM, fill=tk.X)
        widget.configure(yscrollcommand=y.set, xscrollcommand=x.set)

    def build_tree(self, column, title, columns, headings):
        frame = ttk.LabelFrame(self.input_frame, text=title, padding=4)
        frame.grid(row=0, column=column, sticky="nsew", padx=(0, 4) if column == 0 else (4, 0))
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=6, selectmode="browse")
        for index, (key, heading) in enumerate(zip(columns, headings)):
            tree.heading(key, text=heading)
            tree.column(key, width=145 if index == 0 else 95, minwidth=65, anchor="w" if index == 0 else "e")
        self.text_scrollbars(frame, tree)
        tree.pack(fill=tk.BOTH, expand=True)
        return tree

    def restore_layout(self, _event=None):
        if self.collapsed:
            self.split.sashpos(0, self.report_height)
        else:
            self.split.sashpos(0, self.table_height)
            self.split.sashpos(1, self.table_height + self.report_height)

    def remember_layout(self, _event=None):
        if self.collapsed:
            self.report_height = max(85, self.split.sashpos(0))
        else:
            self.table_height = max(110, self.split.sashpos(0))
            self.report_height = max(85, self.split.sashpos(1) - self.table_height)
        self.fit_viewport()

    def fit_viewport(self, _event=None):
        """Give surplus viewport height to plots, retaining overflow in short windows."""
        minimum = self.report_height + 275 + (0 if self.collapsed else self.table_height)
        available = self.scroll.canvas.winfo_height() - self.header.winfo_reqheight() - 3
        height = max(minimum, available)
        if int(self.split.cget("height")) != height:
            self.split.configure(height=height)

    def collapse(self):
        if self.collapsed:
            self.split.insert(0, self.input_frame, weight=0)
        else:
            self.remember_layout()
            self.split.forget(self.input_frame)
        self.collapsed = not self.collapsed
        self.toggle.configure(text=("▸" if self.collapsed else "▾") + " Input data")
        self.fit_viewport()
        self.restore_layout()

    def result_status(self):
        result = self.state.get("advfit_rdc_last_result")
        if not result:
            return "Not fitted"
        if "fingerprint" not in result:
            return "Unknown · refit required"
        try:
            return result_status(result, compile_snapshot(self.state))
        except (ValueError, TypeError, KeyError, RuntimeError):
            return "Stale"

    def can_run(self):
        return bool(not self.state.get("advfit_rdc_running") and self.state.get("advfit_rdc_rows")
                    and (self.state.get("delta_exp_values") or self.state.get("experimental_pcs_observations"))
                    and (self.state.get("atom_data_raw") or self.state.get("atom_data")))

    def can_apply(self):
        return bool(not self.state.get("advfit_rdc_running") and self.result_status() == "Current"
                    and self.state.get("advfit_rdc_last_result", {}).get("success"))

    def refresh_commands(self):
        callback = self.state.get("refresh_fitting_commands")
        if callable(callback):
            callback()

    def mutate(self, key, value):
        try:
            active_document_view(self.state)[key] = value
        except RuntimeError:
            self.state[key] = value
        mark_document_mutated(self.state, "joint_pcs_rdc_inputs")
        self.refresh()
        self.refresh_commands()

    def refresh(self):
        if not self.alive:
            return
        try:
            snapshot = compile_snapshot(self.state)
            options = normalize_options(self.state.get("advfit_rdc_options"))
            self.summary.configure(text=f"{len(snapshot.observations)} PCS · {len(snapshot.bonds)} RDC · Fixed frame · {options['B0']:g} T · {options['T']:g} K")
            pcs_rows = [(o.observable_id, f"{o.pcs:.6g}", "—" if o.uncertainty_ppm is None else f"{o.uncertainty_ppm:.4g}") for o in snapshot.observations]
            rdc_rows = [(f"{b.ref1} – {b.ref2}", f"{b.value:.6g}", f"{b.uncertainty:.4g}") for b in snapshot.bonds]
            rdc_rows += [(reason, "Unmatched", "—") for reason in snapshot.skipped_rows]
            self.matching.configure(text=f"{len(snapshot.observations)} PCS / {len(snapshot.bonds)} matched RDC · {len(snapshot.skipped_rows)} skipped RDC · σ {'supplied' if options['use_pcs_errors'] else 'legacy unit-ppm scale'}")
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            pcs_rows, rdc_rows = [], []
            self.summary.configure(text="Prepare structure and input data")
            self.matching.configure(text=str(exc))
        for tree, rows in ((self.pcs_tree, pcs_rows), (self.rdc_tree, rdc_rows)):
            if rows != getattr(tree, "_joint_rows", None):
                tree.delete(*tree.get_children())
                for row in rows:
                    tree.insert("", tk.END, values=row)
                tree._joint_rows = rows
        status = self.result_status()
        self.status.configure(text="Fitting…" if self.state.get("advfit_rdc_running") else status)
        self.render_report()
        self.draw_result()

    def render_report(self):
        result = self.state.get("advfit_rdc_last_result")
        if not result or "input_snapshot" not in result:
            text = "No verified joint-fit result.\nPrepare structure and PCS observations, load RDC data,\nvalidate inputs and select Run Fit."
        else:
            inputs = result["input_snapshot"]
            lines = ["JOINT PCS+RDC — FIXED FRAME", f"Result state: {self.result_status()}",
                f"Recorded conditions: B₀ {inputs['B0']:g} T / T {inputs['temperature']:g} K",
                "Geometry: fixed main metal and tensor orientation",
                f"PCS weighting: {'supplied uncertainties' if inputs['use_pcs_errors'] else 'legacy unit-ppm residual scale'}",
                f"RDC residual multiplier: {inputs['rdc_multiplier']:g}; error floor: 0.1 Hz",
                f"Data used: {len(result['per_point_pcs'])} PCS / {len(result['rdc_results'])} RDC", "",
                f"Δχ_ax: {result['dchi_ax']:+.6g} E-32 m³", f"Δχ_rh: {result['dchi_rh']:+.6g} E-32 m³",
                f"PCS RMSD: {result['rmsd_pcs']:.6g} ppm", f"RDC RMSD: {result['rmsd_rdc']:.6g} Hz",
                f"Rank: {result['rank']} / 2; condition number: {result['condition_number']:.5g}",
                f"Weighted residual sum of squares: {result['weighted_sum_squares']:.6g}",
                "No confidence intervals are estimated by this fit.", "",
                "PCS observation\tExperimental\tPredicted\tPred − Exp (ppm)"]
            lines += [f"{p['observable_id']}\t{p['experimental']:+.6g}\t{p['predicted']:+.6g}\t{p['residual']:+.6g}" for p in result['per_point_pcs']]
            lines += ["", "RDC Ref pair\tExperimental\tPredicted\tPred − Exp (Hz)"]
            lines += [f"{p['ref1']}–{p['ref2']}\t{p['experimental']:+.6g}\t{p['predicted']:+.6g}\t{p['residual']:+.6g}" for p in result['rdc_results']]
            lines += ["", "Skipped input: " + ("; ".join(result['skipped_rows']) or "none")]
            text = "\n".join(lines)
        if text != getattr(self, "last_report", None):
            self.report.configure(state="normal")
            self.report.delete("1.0", tk.END)
            self.report.insert("1.0", text)
            self.report.configure(state="disabled")
            self.last_report = text

    def draw_result(self, force=False):
        result = self.state.get("advfit_rdc_last_result")
        fonts = get_app_fonts(self.state)
        key = (id(result), self.view_var.get(), tuple(fonts.get(k) for k in ("plot_family", "plot_title", "plot_label", "plot_tick")))
        if not force and key == self.last_plot:
            return
        self.last_plot = key
        self.figure.clear()
        if not result or "input_snapshot" not in result:
            ax = self.figure.add_subplot(111)
            ax.set_axis_off()
            ax.text(.5, .5, "Prepare the structure and PCS observations, then load RDC data.\nValidate the inputs and select Run Fit to evaluate the joint model.",
                    ha="center", va="center", transform=ax.transAxes, wrap=True,
                    fontsize=fonts.get("plot_label", 10), fontfamily=fonts.get("plot_family"))
        else:
            for index, (title, unit, points, symbol) in enumerate((
                ("PCS", "ppm", result['per_point_pcs'], r"\delta"),
                ("RDC", "Hz", result['rdc_results'], "D")), 1):
                ax = self.figure.add_subplot(1, 2, index)
                exp = [p['experimental'] for p in points]
                pred = [p['predicted'] for p in points]
                if self.view_var.get() == "Residuals":
                    ax.scatter(exp, [p['residual'] for p in points], s=24)
                    ax.axhline(0, color="0.45", ls="--", lw=1)
                    ax.set_ylabel(rf"${symbol}_{{\mathrm{{Pred}}}}-{symbol}_{{\mathrm{{Exp}}}}$ ({unit})", fontsize=fonts.get("plot_label", 10))
                else:
                    ax.scatter(exp, pred, s=24)
                    lo, hi = min(exp+pred), max(exp+pred)
                    pad = max((hi-lo)*.08, .01)
                    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color="0.45", ls="--", lw=1)
                    ax.set_ylabel(rf"${symbol}_{{\mathrm{{Pred}}}}$ ({unit})", fontsize=fonts.get("plot_label", 10))
                ax.set_xlabel(rf"${symbol}_{{\mathrm{{Exp}}}}$ ({unit})", fontsize=fonts.get("plot_label", 10))
                ax.set_title(f"{title} {self.view_var.get().lower()}", fontsize=fonts.get("plot_title", 11))
                ax.tick_params(labelsize=fonts.get("plot_tick", 9))
                ax.grid(alpha=.25)
                for label in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
                    if fonts.get("plot_family"):
                        label.set_fontfamily(fonts['plot_family'])
        self.canvas.draw_idle()

    def error(self, error):
        messagebox.showerror("Joint PCS+RDC", str(error), parent=self.root)

    def load_csv(self):
        path = filedialog.askopenfilename(parent=self.root, title="Load RDC data", filetypes=[("CSV", "*.csv")])
        if path:
            try:
                with open(path, encoding="utf-8-sig", newline="") as stream:
                    rows = parse_rdc_csv(stream)
                self.mutate("advfit_rdc_rows", rows)
            except (OSError, ValueError) as exc:
                self.error(exc)

    def validate(self):
        try:
            result = execute_fit(compile_snapshot(self.state))
            messagebox.showinfo("Joint PCS+RDC validation", f"{len(result['per_point_pcs'])} PCS / {len(result['rdc_results'])} RDC\nRank {result['rank']} / 2; {len(result['skipped_rows'])} skipped RDC rows\nInputs and fixed-frame model are valid. No result was applied.", parent=self.root)
        except (ValueError, RuntimeError, TypeError, KeyError) as exc:
            self.error(exc)

    def run(self):
        if self.state.get("advfit_rdc_running"):
            return
        try:
            snapshot = compile_snapshot(self.state)
            identity = capture_worker_identity(self.state)
        except (ValueError, RuntimeError, TypeError, KeyError) as exc:
            self.error(exc)
            return
        results = queue.Queue()
        def work():
            try:
                results.put((execute_fit(snapshot), None))
            except Exception as exc:
                results.put((None, f"{type(exc).__name__}: {exc}"))
        self.state["advfit_rdc_running"] = True
        self.refresh()
        self.refresh_commands()
        def poll():
            self.poll_id = None
            if not self.alive:
                return
            try:
                payload, error = results.get_nowait()
            except queue.Empty:
                self.poll_id = self.root.after(40, poll)
                return
            self.state["advfit_rdc_running"] = False
            if identity is not None and not worker_completion_matches(self.state, identity):
                self.refresh()
                self.refresh_commands()
                return
            if error:
                if identity is None or (active_document(self.state) and active_document(self.state).document_id == identity.document_id):
                    self.error(error)
            elif identity is None:
                self.state["advfit_rdc_last_result"] = payload
            else:
                origin = apply_worker_document_updates(self.state, identity,
                    {"advfit_rdc_last_result": payload}, domain="joint_pcs_rdc_result")
                if origin is active_document(self.state):
                    self.state["advfit_rdc_last_result"] = payload
            self.refresh()
            self.refresh_commands()
        try:
            threading.Thread(target=work, daemon=True).start()
            self.poll_id = self.root.after(40, poll)
        except RuntimeError as exc:
            self.state["advfit_rdc_running"] = False
            self.refresh_commands()
            self.error(exc)

    def apply(self):
        if not self.can_apply():
            self.error("Only a current successful joint-fit result can be applied")
            return
        result = self.state['advfit_rdc_last_result']
        tensor = self.state.get(PCS_TENSOR_STATE_KEY)
        if not isinstance(tensor, ApplicationPCSTensor):
            tensor = canonical_tensor_from_legacy(self.state)
        update_canonical_tensor(self.state, replace(tensor, delta_chi_ax=result['dchi_ax'], delta_chi_rh=result['dchi_rh']),
                                source="joint_pcs_rdc_apply", legacy_callback=self.state.get("update_graph"))
        self.refresh()
        self.refresh_commands()

    def export(self):
        result = self.state.get("advfit_rdc_last_result")
        if not result:
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="Export recorded joint result", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            try:
                Path(path).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            except (OSError, ValueError) as exc:
                self.error(exc)

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
        x = self.root.winfo_rootx() + (self.root.winfo_width()-win.winfo_reqwidth())//2
        y = self.root.winfo_rooty() + (self.root.winfo_height()-win.winfo_reqheight())//2
        win.geometry(f"+{max(0,x)}+{max(0,y)}")
        win.deiconify()
        win.grab_set()
        win.bind("<Escape>", lambda _e: win.destroy())

    def settings(self):
        options = normalize_options(self.state.get("advfit_rdc_options"))
        win, body, origin = self.dialog("Joint PCS+RDC fit settings")
        fields = {}
        for i, (key, label) in enumerate((("B0", "B₀ (T)"), ("T", "Temperature (K)"), ("w_rdc", "RDC residual multiplier"))):
            fields[key] = tk.StringVar(master=win, value=str(options[key]))
            ttk.Label(body, text=label).grid(row=i, column=0, sticky="w", padx=(0, 12), pady=4)
            ttk.Entry(body, textvariable=fields[key]).grid(row=i, column=1, sticky="ew", pady=4)
        use_errors = tk.BooleanVar(master=win, value=options['use_pcs_errors'])
        ttk.Checkbutton(body, text="Use supplied PCS uncertainties", variable=use_errors).grid(row=3, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(body, text="Fixed main metal and tensor orientation; fit Δχ_ax and Δχ_rh.\nRDC: magnetic self-alignment only; error floor 0.1 Hz.\nWithout PCS σ, the legacy unit-ppm residual scale is used.\nThe RDC multiplier scales residuals and is squared in the objective.", wraplength=460).grid(row=4, column=0, columnspan=2, sticky="w", pady=8)
        def save():
            if origin is not active_document(self.state):
                self.error("The active document changed; reopen Fit settings")
                return
            try:
                updated = normalize_options(dict(options, **{k: v.get() for k,v in fields.items()}, use_pcs_errors=use_errors.get()))
                self.mutate('advfit_rdc_options', updated)
                win.destroy()
            except ValueError as exc:
                messagebox.showerror("Fit settings", str(exc), parent=win)
        actions = ttk.Frame(body)
        actions.grid(row=5, column=0, columnspan=2, sticky="e")
        ttk.Button(actions, text="Save", command=save).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions, text="Cancel", command=win.destroy).pack(side=tk.LEFT, padx=3)
        self.show_dialog(win)

    def select_observations(self):
        try:
            observations = tuple(collect_observations(self.state))
        except (ValueError, RuntimeError) as exc:
            self.error(exc)
            return
        win, body, origin = self.dialog("Select PCS observations")
        listing = tk.Listbox(body, selectmode=tk.EXTENDED, exportselection=False, height=12, width=65,
                             font=ui_font_name(body, "ui"))
        listing.grid(row=0, column=0, columnspan=2, sticky="nsew")
        scrollbar = ttk.Scrollbar(body, command=listing.yview)
        scrollbar.grid(row=0, column=2, sticky="ns")
        listing.configure(yscrollcommand=scrollbar.set)
        selected = normalize_options(self.state.get("advfit_rdc_options"))['observation_ids']
        for i, obs in enumerate(observations):
            listing.insert(tk.END, f"{obs.observable_id} · {obs.assignment or 'Ref '+str(obs.ref_id)} · {obs.pcs:+.6g} ppm")
            if selected is None or obs.observable_id in selected:
                listing.selection_set(i)
        def save():
            if origin is not active_document(self.state):
                self.error("The active document changed; reopen Select PCS")
                return
            options = normalize_options(self.state.get("advfit_rdc_options"))
            options['observation_ids'] = [observations[i].observable_id for i in listing.curselection()]
            self.mutate('advfit_rdc_options', options)
            win.destroy()
        actions = ttk.Frame(body)
        actions.grid(row=1, column=0, columnspan=2, sticky="e", pady=(8,0))
        ttk.Button(actions, text="Select all", command=lambda: listing.selection_set(0, tk.END)).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions, text="Save", command=save).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions, text="Cancel", command=win.destroy).pack(side=tk.LEFT, padx=3)
        self.show_dialog(win)

    def callbacks(self):
        return {PREFIX+key: method for key, method in {
            'input.load_csv': self.load_csv, 'input.observations': self.select_observations,
            'input.validate': self.validate, 'parameters.settings': self.settings,
            'run': self.run, 'results.apply': self.apply, 'output.export': self.export}.items()}

    def theme_changed(self, _event=None):
        if self.font_id is None and self.alive:
            def refresh():
                self.font_id = None
                self.draw_result()
            self.font_id = self.root.after_idle(refresh)

    def destroy(self, event):
        if event.widget is not self.tab:
            return
        self.alive = False
        self.state['advfit_rdc_running'] = False
        for task in (self.poll_id, self.font_id):
            if task:
                self.root.after_cancel(task)
        hooks = self.state.get('follow_active_document_refresh_hooks', [])
        if self.refresh in hooks:
            hooks.remove(self.refresh)
