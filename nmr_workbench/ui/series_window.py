"""Focused, modeless VT-series management; the host still renders one spectrum."""

from __future__ import annotations

from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from .series_manager import SeriesManager, discover_experiments, load_experiments
from .view_model import dataset_display_name
from .actions import action, action_menu, polish
from .workflow_help import show_workflow_help


class SeriesManagementWindow(tk.Toplevel):
    def __init__(self, parent, model, *, on_activate, on_datasets_changed, on_changed=None):
        super().__init__(parent)
        self.title("VT Series — NMR Workbench")
        self.transient(parent)
        self.geometry("900x560")
        self.minsize(720, 450)
        self.manager = SeriesManager(model)
        self.on_activate = on_activate
        self.on_datasets_changed = on_datasets_changed
        self.on_changed = on_changed
        self._refreshing = False
        self._busy = False
        self._poll_id = None
        self._buttons = []
        self._series_ids = []
        self._dataset_ids = []
        self.order = tk.StringVar(self, value="Acquisition")
        self.temperature = tk.StringVar(self)
        self.label = tk.StringVar(self)
        self.included = tk.BooleanVar(self, value=True)
        self.status = tk.StringVar(self, value="Create a series or load definitions to begin.")
        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        toolbar = ttk.Frame(body)
        toolbar.pack(fill="x", pady=(0, 8))
        self.selector = ttk.Combobox(toolbar, state="readonly", width=24)
        self.selector.pack(side="left", fill="x", expand=True)
        self.selector.bind("<<ComboboxSelected>>", self._select_series)
        self._button(toolbar, "New", self._new, icon="add", description="Create a new series")
        self._buttons.append(action_menu(toolbar, "Series", "settings", (
            ("Rename series…", lambda: self._run(self._rename)),
            ("Delete series…", lambda: self._run(self._delete)), None,
            ("Workflow & file guide…", lambda: show_workflow_help(self)))))
        self._buttons[-1].pack(side="left", padx=5)
        self._buttons.append(action_menu(toolbar, "Files", "open_folder", (
            ("Save NMR session…", lambda: self._run(self._save_session)),
            ("Open NMR session…", lambda: self._run(self._open_session)), None,
            ("Export series definitions…", lambda: self._run(self._save)),
            ("Import series definitions…", lambda: self._run(self._load)), None,
            ("Rebind loaded source…", lambda: self._run(self._rebind_source)))))
        self._buttons[-1].pack(side="left")
        ordering = ttk.Frame(body)
        ordering.pack(fill="x", pady=(0, 6))
        ttk.Label(ordering, text="Member order").pack(side="left")
        order = ttk.Combobox(ordering, textvariable=self.order, state="readonly", width=18,
                             values=("Acquisition", "Temperature ↑", "Temperature ↓"))
        order.pack(side="left", padx=6)
        order.bind("<<ComboboxSelected>>", lambda event: self.refresh())
        table = ttk.Frame(body)
        table.pack(fill="both", expand=True)
        columns = ("order", "dataset", "temperature", "source", "label", "included", "diagnostic")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", selectmode="browse",
                                 style="NMR.Inspector.Treeview")
        for column, title, width in zip(columns,
                ("Order", "Experiment", "T / K", "T source", "Label", "Use", "Diagnostic"),
                (50, 180, 75, 120, 130, 45, 130)):
            self.tree.heading(column, text=title)
            self.tree.column(column, width=width, minwidth=40)
        vertical = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        horizontal = ttk.Scrollbar(table, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", self._select_member)
        editor = ttk.Frame(body)
        editor.pack(fill="x", pady=8)
        ttk.Label(editor, text="Override / K").pack(side="left")
        ttk.Entry(editor, textvariable=self.temperature, width=10).pack(side="left", padx=5)
        ttk.Label(editor, text="Label").pack(side="left")
        ttk.Entry(editor, textvariable=self.label, width=22).pack(side="left", padx=5, fill="x", expand=True)
        ttk.Checkbutton(editor, text="Include", variable=self.included).pack(side="left")
        self._button(editor, "Apply edit", self._edit, icon="apply", primary=True)
        ttk.Label(body, text="Blank override uses series override, calibrated, then nominal temperature.",
                  style="NMR.Muted.TLabel").pack(anchor="w")
        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=8)
        self.loaded = ttk.Combobox(actions, state="readonly", width=24)
        self.loaded.pack(side="left", fill="x", expand=True)
        self._button(actions, "Add loaded", self._add, icon="add")
        self._button(actions, "Remove", self._remove, icon="remove", description="Remove the selected member from this series")
        self._buttons.append(action_menu(actions, "Import", "import", (
            ("FID experiments…", lambda: self._run(self._import_fids)),
            ("Folder group…", lambda: self._run(self._import_folder)))))
        self._buttons[-1].pack(side="left", padx=5)
        ttk.Label(body, text="Files → Save NMR session resumes committed work. Series definitions contain links only.",
                  wraplength=820, style="NMR.Muted.TLabel").pack(anchor="w")
        ttk.Label(body, textvariable=self.status, wraplength=820, justify="left").pack(fill="x")
        polish(self)
        self.refresh()

    def _button(self, parent, text, command, *, icon="apply", primary=False, description=None):
        button = action(parent, text, icon, lambda: self._run(command), primary=primary, description=description)
        button.pack(side="left", padx=(5, 0))
        self._buttons.append(button)
        return button

    def _run(self, command):
        try:
            command()
        except (ValueError, KeyError, OSError) as exc:
            messagebox.showerror("VT Series", str(exc), parent=self)

    def _active(self):
        series = self.manager.active_series
        if series is None:
            raise ValueError("Select or create a series first.")
        return series

    def _member(self):
        series = self._active()
        selected = self.tree.selection()
        return next((m for m in series.members if selected and m.id == selected[0]), None)

    def refresh(self):
        self._refreshing = True
        try:
            state = self.manager.state
            self._series_ids = [s.id for s in state.series]
            self.selector.configure(values=[s.name for s in state.series])
            self.selector.set("")
            if state.selected_series_id in self._series_ids:
                self.selector.current(self._series_ids.index(state.selected_series_id))
            self._dataset_ids = [d.id for d in state.datasets]
            self.loaded.configure(values=[f"{dataset_display_name(d)} [{d.id[:8]}]" for d in state.datasets])
            self.loaded.set("")
            if state.selected_dataset_id in self._dataset_ids:
                self.loaded.current(self._dataset_ids.index(state.selected_dataset_id))
            self.tree.delete(*self.tree.get_children())
            series = self.manager.active_series
            self.temperature.set("")
            self.label.set("")
            self.included.set(True)
            if series is None:
                self.status.set("Create a series or load definitions to begin.")
                return
            issues = series.validate_temperatures()
            by_member = {}
            for issue in issues:
                for member_id in issue.member_ids:
                    by_member.setdefault(member_id, []).append(issue.code)
            for member in self.manager.members(series.id, self.order.get()):
                resolved = member.resolve_temperature(series_override_k=series.temperature_override_k)
                value = f"{resolved.temperature_k:g}" if resolved.temperature_k is not None else "—"
                source = {"member_override": "Member override", "series_override": "Series override",
                          "calibrated": "Calibrated", "nominal": "Nominal", None: "Missing"}[resolved.source]
                self.tree.insert("", "end", iid=member.id, values=(
                    member.acquisition_order, dataset_display_name(member.dataset), value,
                    source, member.label or "", "Yes" if member.included else "No",
                    ", ".join(by_member.get(member.id, ())) or "OK"))
            if state.selected_member_id and self.tree.exists(state.selected_member_id):
                self.tree.selection_set(state.selected_member_id)
                self.tree.see(state.selected_member_id)
                self._fill_editor(self._member())
            counts = {code: sum(len(i.member_ids) for i in issues if i.code == code)
                      for code in ("missing", "invalid", "duplicate")}
            self.status.set(f"{len(series.members)} members · " + " · ".join(
                f"{counts[code]} {code}" for code in counts) + " (includes excluded members)")
        finally:
            self._refreshing = False
            if self.on_changed is not None:
                self.on_changed()

    def _fill_editor(self, member):
        if member is not None:
            self.temperature.set("" if member.temperature_override_k is None else str(member.temperature_override_k))
            self.label.set(member.label or "")
            self.included.set(member.included)

    def _select_series(self, event=None):
        index = self.selector.current()
        if not self._busy and index >= 0:
            self.manager.select_series(self._series_ids[index])
            self.refresh()

    def _select_member(self, event=None):
        if self._refreshing or self._busy:
            return
        member = self._member() if self.manager.active_series else None
        if member is not None:
            self._fill_editor(member)
            state = self.manager.state
            if state.selected_member_id != member.id or state.selected_dataset_id != member.dataset_id:
                self.manager.select_member(self._active().id, member.id)
                self.on_activate(member.dataset_id)

    def _new(self):
        name = simpledialog.askstring("New VT series", "Series name", parent=self)
        if name is not None:
            self.manager.create_series(name)
            self.refresh()

    def _rename(self):
        series = self._active()
        name = simpledialog.askstring("Rename VT series", "Series name", initialvalue=series.name, parent=self)
        if name is not None:
            self.manager.rename_series(series.id, name)
            self.refresh()

    def _delete(self):
        self.manager.delete_series(self._active().id)
        self.refresh()

    def _add(self):
        index = self.loaded.current()
        if index < 0:
            raise ValueError("Choose a loaded experiment to add.")
        member = self.manager.add_dataset(self._active().id, self._dataset_ids[index])
        self.manager.select_member(self._active().id, member.id)
        self.on_activate(member.dataset_id)
        self.refresh()

    def _remove(self):
        member = self._member()
        if member is not None:
            self.manager.remove_member(self._active().id, member.id)
            self.refresh()

    def _edit(self):
        member = self._member()
        if member is None:
            raise ValueError("Select a member to edit.")
        value = self.temperature.get().strip()
        self.manager.edit_member(self._active().id, member.id, label=self.label.get() or None,
                                 included=self.included.get(), temperature_override_k=float(value) if value else None)
        self.refresh()

    def _import_fids(self):
        self._active()
        paths = filedialog.askopenfilenames(parent=self, title="Select experiment FID files",
                                            filetypes=(("Raw FID", "fid"), ("All files", "*")))
        if paths:
            self._start_import(lambda: paths)

    def _import_folder(self):
        self._active()
        root = filedialog.askdirectory(parent=self, title="Choose a folder containing independent 1D experiments")
        if root:
            self._start_import(lambda: (root,))

    def _start_import(self, paths):
        series_id = self._active().id
        queue = Queue()
        self._busy = True
        self.selector.configure(state="disabled")
        for button in self._buttons:
            button.configure(state="disabled")
        self.status.set("Importing independent 1D experiments…")

        def worker():
            try:
                queue.put(load_experiments(paths()))
            except Exception as exc:
                queue.put(exc)
        Thread(target=worker, daemon=True).start()

        def poll():
            try:
                result = queue.get_nowait()
            except Empty:
                self._poll_id = self.after(100, poll)
                return
            self._poll_id = None
            self._busy = False
            self.selector.configure(state="readonly")
            for button in self._buttons:
                button.configure(state="normal")
            if isinstance(result, Exception):
                messagebox.showerror("VT Series import", str(result), parent=self)
                self.refresh()
                return
            report = self.manager.apply_import(series_id, result)
            self.on_datasets_changed()
            self.refresh()
            summary = f"Added {len(report.added_member_ids)} members; {len(report.failures)} failed; {len(report.skipped)} skipped."
            if not result.loaded and not result.failures:
                summary = "No independent 1D experiments were found in the selected folder."
            self.status.set(summary + " " + self.status.get())
            details = [f"{path}: {error}" for path, error in report.failures] + list(report.skipped)
            if details:
                messagebox.showwarning("VT Series import", summary + "\n\n" + "\n".join(details), parent=self)
        self._poll_id = self.after(100, poll)

    def _save(self):
        path = filedialog.asksaveasfilename(parent=self, title="Save series definitions", defaultextension=".nmrseries",
                                           filetypes=(("NMR series definitions", "*.nmrseries"),))
        if path:
            self.manager.save_definitions(path)
            self.status.set("Saved series definitions. Spectra and processing sessions are not included.")

    def _load(self):
        path = filedialog.askopenfilename(parent=self, title="Load definitions for loaded experiments",
                                         filetypes=(("NMR series definitions", "*.nmrseries"),))
        if path:
            self.manager.load_definitions(path)
            self.refresh()
            diagnostics = self.manager.restoration_diagnostics
            if diagnostics:
                self.status.set(f"Definitions loaded with {len(diagnostics)} stale track links; review in Tracks….")
                messagebox.showwarning("VT Series definitions", "No links were relinked or reconfirmed.\n\n" +
                    "\n".join(diagnostics), parent=self)

    def destroy(self):
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
            self._poll_id = None
        super().destroy()

    def _save_session(self):
        path = filedialog.asksaveasfilename(parent=self, title="Save recoverable NMR session",
            defaultextension=".nmrsession", filetypes=(("NMR Workbench session", "*.nmrsession"),))
        if path:
            omitted = self.manager.save_session(path, processing_session=getattr(self.master, "processing_session", None))
            self.status.set("Saved committed NMR session; previews and UI selections are not included.")
            if omitted:
                messagebox.showwarning("NMR session ownership", "\n".join(omitted), parent=self)

    def _open_session(self):
        path = filedialog.askopenfilename(parent=self, title="Open recoverable NMR session",
            filetypes=(("NMR Workbench session", "*.nmrsession"),))
        if path:
            result = self.manager.load_session(path)
            self.on_datasets_changed()
            self.refresh()
            warnings = [d for d in result.diagnostics if d.code != "source_unchanged"]
            self.status.set(f"Recovered {len(result.datasets)} datasets and {len(result.series)} series; {len(warnings)} diagnostics.")
            if warnings:
                messagebox.showwarning("NMR session recovery", "\n".join(
                    f"{d.code}: dataset={d.dataset_id}: {d.message}" for d in warnings), parent=self)

    def _rebind_source(self):
        from nmr_workbench.io.session import rebind_source
        index = self.loaded.current()
        if index < 0:
            raise ValueError("Choose a loaded experiment whose source should be rebound.")
        dataset = next(d for d in self.manager.state.datasets if d.id == self._dataset_ids[index])
        path = filedialog.askdirectory(parent=self, title="Locate vendor experiment source folder")
        if path:
            result = rebind_source(dataset, path)
            self.status.set(f"{result.code}: {result.message}")
            if result.code != "source_moved":
                messagebox.showwarning("NMR source binding", self.status.get(), parent=self)
