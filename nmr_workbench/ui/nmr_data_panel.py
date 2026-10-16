"""Compact observation table over existing dataset-local NMRObservable records."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from dataclasses import replace

from ui.icons import IconManager, TOOLBAR_ICON_SIZE
from ui.ribbon.icon_service import nearest_packaged_icon_size, tk_scale_factor
from ui.tooltips import ToolTip
from .presentation import size_workbench_action
from .view_model import dataset_display_name
from ..analysis.nmr_data import (data_rows, new_data_dataset, edit_observable,
    set_observable_use, remove_observable, read_only)
from ..io.observable_table import (table_reader, default_mapping, parse_observation_table,
    apply_observation_import, export_observation_table, observation_state_signature)
from ..io.exchange_files import atomic_text_export


class NMRDataPanel(ttk.Frame):
    def __init__(self, parent, model, *, on_changed, on_select, on_map, on_legacy=None, structure_actions=(),
                 on_peak_assignment=None, on_peak_remove=None, on_compare=None, on_clear_peaks=None,
                 peak_assignment_draft=None):
        super().__init__(parent)
        self.model = model
        self.on_changed, self.on_select, self.on_map = on_changed, on_select, on_map
        self.on_peak_assignment, self.on_peak_remove = on_peak_assignment, on_peak_remove
        self.peak_assignment_draft = peak_assignment_draft
        self.selected_key = None
        self.rows = {}
        self._signature = None
        self.icons = getattr(self._root(), "_icon_manager", None) or IconManager(self._root())
        self._images = []
        self._inline_editor = None
        self._sort_key = None
        self._sort_reverse = False
        self.experiment = tk.StringVar(value="Active experiment")
        self.show = tk.StringVar(value="All")
        filters = ttk.Frame(self)
        filters.pack(fill="x", pady=(3, 4))
        ttk.Label(filters, text="Experiment").grid(row=0, column=0, sticky="w")
        self.experiment_box = ttk.Combobox(filters, textvariable=self.experiment,
            values=("Active experiment", "All experiments"), state="readonly", width=21)
        self.experiment_box.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(filters, text="Show").grid(row=0, column=2, sticky="w")
        show_box = ttk.Combobox(filters, textvariable=self.show, values=("All", "Assigned", "Unassigned"), state="readonly", width=10)
        show_box.grid(row=0, column=3, sticky="w", padx=(4, 0))
        filters.columnconfigure(1, weight=1)
        for box in (self.experiment_box, show_box):
            box.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(3, 4))
        self._button(bar, "Add", "add", self.edit).pack(side="left")
        imports = ttk.Menubutton(bar, text="Import", style="NMR.Toolbar.TMenubutton")
        self._menu_icon(imports, "import")
        menu = tk.Menu(imports, tearoff=False)
        menu.add_command(label="Observed shifts CSV / TSV…", command=self.import_file)
        menu.add_command(label="Paste observed shifts…", command=self.paste)
        if on_legacy:
            menu.add_separator()
            menu.add_command(label="PCS project observed shifts…", command=lambda: on_legacy("obs", False))
            menu.add_command(label="PCS project diamagnetic shifts…", command=lambda: on_legacy("dia", False))
            menu.add_command(label="Paste PCS project observed shifts…", command=lambda: on_legacy("obs", True))
            menu.add_command(label="Paste PCS project diamagnetic shifts…", command=lambda: on_legacy("dia", True))
        imports.configure(menu=menu)
        size_workbench_action(imports)
        imports.pack(side="left", padx=3)
        exports = ttk.Menubutton(bar, text="Export", style="NMR.Toolbar.TMenubutton")
        self._menu_icon(exports, "export")
        export_menu = tk.Menu(exports, tearoff=False)
        export_menu.add_command(label="All observations CSV…", command=self.export)
        exports.configure(menu=export_menu)
        size_workbench_action(exports)
        exports.pack(side="left")
        more = ttk.Menubutton(bar, text="More", style="NMR.Toolbar.TMenubutton")
        self._menu_icon(more, "toolbox")
        self.actions_menu = tk.Menu(more, tearoff=False)
        self.actions_menu.add_command(label="Edit details…", command=lambda: self.edit(existing=True))
        self.actions_menu.add_command(label="Remove selected…", command=self.remove)
        if on_clear_peaks:
            self.actions_menu.add_command(label="Clear picked peaks in active experiment…", command=on_clear_peaks)
        self.actions_menu.add_command(label="Copy experiment values…", command=self.copy_values)
        self.actions_menu.add_separator()
        self.actions_menu.add_command(label="Show in Chemical Map", command=self.show_map)
        for label, callback in structure_actions:
            self.actions_menu.add_command(label=label, command=lambda action=callback: self.structure_action(action))
        more.configure(menu=self.actions_menu)
        size_workbench_action(more)
        more.pack(side="right")
        self.status = tk.StringVar(value="Observed shifts from spectra or prepared data")
        ttk.Label(self, textvariable=self.status, style="NMR.Muted.TLabel", wraplength=360).pack(fill="x", pady=(0, 4))
        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True)
        columns = ("use", "assignment", "shift", "source", "status", "experiment", "role", "nucleus", "temperature", "sigma", "intensity", "peak_source")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse", style="NMR.Inspector.Treeview")
        for key, title, width in zip(columns,
            ("Use", "Assignment", "Shift / ppm", "Source", "Status", "Experiment", "Role", "Nucleus", "T / K", "σ / ppm", "Intensity", "Picking"),
            (36, 100, 85, 75, 210, 130, 100, 60, 70, 70, 90, 90)):
            self.tree.heading(key, text=title, command=lambda k=key: self._sort(k))
            self.tree.column(key, width=width, minwidth=width, stretch=key == "status", anchor="center" if key == "use" else "w")
        scroll = ttk.Scrollbar(frame, command=self.tree.yview)
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=scroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", self._selected)
        self.tree.bind("<Button-1>", self._click)
        self.tree.bind("<Double-1>", self._double_click)
        self.tree.bind("<Return>", lambda e: self.edit_assignment())
        self.tree.bind("<Delete>", lambda e: self.remove())
        self.tree.bind("<Button-3>", self._context_menu)
        self.tree.bind("<space>", lambda e: self.toggle_use())
        footer = ttk.Frame(self)
        footer.pack(fill="x", pady=4)
        if on_compare:
            self._button(footer, "Compare shifts →", "compare", on_compare).pack(side="right")
        ToolTip(self.tree, "Click Use to include/exclude assigned observations from Shift Analysis and PCS preparation.\nDouble-click Assignment or press Enter to assign; Tab moves to the next row.\nSpectrum shifts remain dataset-peak measurements; PCS project records are read-only.\nVT fitting keeps its own masks and member temperatures.")
        self.refresh()

    def _button(self, parent, text, icon, command):
        size = nearest_packaged_icon_size(TOOLBAR_ICON_SIZE, tk_scale_factor(self._root()))
        image = self.icons.get(icon, size)
        options = dict(text=text, command=command, style="NMR.Secondary.TButton")
        if image:
            self._images.append(image)
            options.update(image=image, compound="left")
        button = ttk.Button(parent, **options)
        size_workbench_action(button)
        return button

    def _menu_icon(self, widget, icon):
        size = nearest_packaged_icon_size(TOOLBAR_ICON_SIZE, tk_scale_factor(self._root()))
        image = self.icons.get(icon, size)
        if image:
            self._images.append(image)
            widget.configure(image=image, compound="left")

    def structure_action(self, callback):
        try:
            dataset, item = self._record()
            if not item.assignment:
                raise ValueError("Assign this observation first")
            self.on_select(dataset.id, item.id)
            callback(item.assignment)
        except ValueError as exc:
            self._error(exc)

    def _record(self):
        if not self.selected_key:
            raise ValueError("Select an observation first")
        dataset_id, record_id = self.selected_key
        dataset = next((d for d in self.model.datasets if d.id == dataset_id), None)
        item = next((o for o in dataset.observables if o.id == record_id or
            (record_id.startswith("peak:") and o.source_peak_id == record_id[5:])), None) if dataset else None
        if item is None:
            raise ValueError("The selected observation no longer exists")
        return dataset, item

    @staticmethod
    def _key(row):
        return (row.dataset_id, "peak:" + row.peak_id if row.peak_id else row.observable_id)

    def _selected_row(self):
        return next((r for r in self.rows.values() if self._key(r) == self.selected_key), None)

    def _data_rows(self):
        rows = data_rows(self.model.datasets, include_unassigned_peaks=True)
        if self.peak_assignment_draft:
            def project(row):
                if row.peak_id and not row.observable_id:
                    label = self.peak_assignment_draft(row.dataset_id, row.peak_id)
                    if label:
                        return replace(row, assignment=label, diagnostics=("Assignment draft — complete experiment metadata before analysis",))
                return row
            rows = tuple(project(row) for row in rows)
        return rows

    def _cancel_inline_editor(self):
        editor, self._inline_editor = self._inline_editor, None
        if editor is not None:
            editor.destroy()

    def _sort(self, key):
        self._sort_reverse = not self._sort_reverse if self._sort_key == key else False
        self._sort_key = key
        self.refresh()

    def _context_menu(self, event):
        iid = self.tree.identify_row(event.y)
        if iid:
            self.tree.selection_set(iid)
            self.selected_key = self._key(self.rows[iid])
            self.on_select(*self.selected_key)
            self.actions_menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _double_click(self, event):
        iid = self.tree.identify_row(event.y)
        if not iid:
            return "break"
        self.selected_key = self._key(self.rows[iid])
        column = self.tree.identify_column(event.x)
        if column == "#2":
            self.edit_assignment()
        elif column != "#1":
            self.edit(existing=True)
        return "break"

    def edit_assignment(self):
        row = self._selected_row()
        if row is None:
            return "break"
        dataset = next(d for d in self.model.datasets if d.id == row.dataset_id)
        if read_only(dataset):
            self._error(ValueError("PCS project rows are read-only; copy values to edit them independently."))
            return "break"
        self._cancel_inline_editor()
        iid = next(i for i, r in self.rows.items() if self._key(r) == self.selected_key)
        self.tree.see(iid)
        bounds = self.tree.bbox(iid, "assignment")
        if not bounds:
            self.tree.xview_moveto(0)
            self.update_idletasks()
            bounds = self.tree.bbox(iid, "assignment")
        if not bounds:
            return "break"
        value = tk.StringVar(value=row.assignment or "")
        editor = self._inline_editor = ttk.Entry(self.tree, textvariable=value)
        editor.place(x=bounds[0], y=bounds[1], width=bounds[2], height=bounds[3])
        editor.focus_set()
        editor.select_range(0, "end")
        def finish(commit=True, advance=0):
            if self._inline_editor is not editor:
                return "break"
            label = value.get().strip()
            order = [self._key(r) for r in self.rows.values()]
            old_key = self._key(row)
            self._inline_editor = None
            editor.destroy()
            if commit:
                try:
                    if row.peak_id:
                        self.on_peak_assignment(row.dataset_id, row.peak_id, label)
                    else:
                        item = next(o for o in dataset.observables if o.id == row.observable_id)
                        edit_observable(dataset, observable_id=item.id, assignment=label,
                            shift_ppm=item.chemical_shift_ppm, uncertainty_ppm=item.uncertainty_ppm, included=item.included)
                        self.on_changed()
                except (ValueError, StopIteration) as exc:
                    self._error(exc)
            if advance and old_key in order:
                target_index = order.index(old_key) + advance
                if 0 <= target_index < len(order):
                    self.selected_key = order[target_index]
                    self._project_selection()
                    self.on_select(*self.selected_key)
                    self.edit_assignment()
            return "break"
        editor.bind("<Return>", lambda e: finish())
        editor.bind("<Escape>", lambda e: finish(False))
        editor.bind("<Tab>", lambda e: finish(advance=1))
        editor.bind("<Shift-Tab>", lambda e: finish(advance=-1))
        editor.bind("<FocusOut>", lambda e: finish())
        return "break"

    def refresh(self):
        all_rows = self._data_rows()
        names = {d.id: dataset_display_name(d) for d in self.model.datasets}
        roles = {d.id: d.experimental_role for d in self.model.datasets}
        choices = {f"{name} · {dataset_id[:8]}": dataset_id for dataset_id, name in names.items()}
        self.experiment_box.configure(values=("Active experiment", "All experiments", *choices))
        selected_id = self.model.selected_dataset_id if self.experiment.get() == "Active experiment" else choices.get(self.experiment.get())
        rows = tuple(r for r in all_rows if (self.experiment.get() == "All experiments" or r.dataset_id == selected_id)
            and (self.show.get() == "All" or bool(r.assignment) == (self.show.get() == "Assigned")))
        if self._sort_key:
            attrs = {"use": "included", "shift": "shift_ppm", "temperature": "temperature_k", "sigma": "uncertainty_ppm"}
            def value(row):
                key = self._sort_key
                result = names[row.dataset_id] if key == "experiment" else roles[row.dataset_id] if key == "role" else "; ".join(row.diagnostics) if key == "status" else getattr(row, attrs.get(key, key))
                return (result is None, result.casefold() if isinstance(result, str) else result if result is not None else 0)
            rows = tuple(sorted(rows, key=value, reverse=self._sort_reverse))
        signature = (rows, tuple(names.items()), tuple(roles.items()), self.experiment.get(), self.show.get())
        if signature == self._signature:
            return
        self._cancel_inline_editor()
        self._signature = signature
        self.tree.delete(*self.tree.get_children())
        self.rows = {}
        for index, row in enumerate(rows):
            iid = f"observation-{index}"
            self.rows[iid] = row
            number = lambda value: "—" if value is None else f"{value:.8g}"
            self.tree.insert("", "end", iid=iid, values=(("☑" if row.included else "☐") if row.observable_id and row.assignment else "—",
                row.assignment or "—", number(row.shift_ppm), row.source, "; ".join(row.diagnostics) or "Ready",
                names[row.dataset_id], roles[row.dataset_id], row.nucleus,
                number(row.temperature_k), number(row.uncertainty_ppm),
                number(row.intensity), row.peak_source or "—",
                ))
        self._project_selection()
        if rows:
            self.status.set(f"{len(rows)} observations · {sum(r.included for r in rows)} checked for shift comparison / PCS preparation")
        else:
            self.status.set("No matching data. Pick peaks, Add values or Import a shift table.")

    def _project_selection(self):
        for iid, row in self.rows.items():
            if self._key(row) == self.selected_key:
                self.tree.selection_set(iid)
                self.tree.focus(iid)
                self.tree.see(iid)
                return
        # A filtered-out semantic selection remains valid; only remove broken identities.
        if self.selected_key and not any(self._key(r) == self.selected_key
            for r in self._data_rows()):
            self.selected_key = None

    def select_identity(self, *, dataset_id=None, peak_id=None, assignment=None):
        matches = [self._key(row) for row in self._data_rows()
                   if (dataset_id is None or row.dataset_id == dataset_id)
                   and ((peak_id is not None and row.peak_id == peak_id)
                        or (peak_id is None and assignment and row.assignment == assignment))]
        if len(matches) == 1:
            self.selected_key = matches[0]
            self._project_selection()

    def _selected(self, event=None):
        selection = self.tree.selection()
        if not selection:
            return  # rebuild events do not discard semantic selection
        row = self.rows.get(selection[0])
        if row is None or self._key(row) == self.selected_key:
            return
        self.selected_key = self._key(row)
        self.status.set(f"{row.source} · " + ("; ".join(row.diagnostics) or "Ready for shift comparison"))
        self.on_select(*self.selected_key)

    def _click(self, event):
        iid = self.tree.identify_row(event.y)
        if iid and self.tree.identify_column(event.x) == "#1":
            row = self.rows[iid]
            self.selected_key = self._key(row)
            self.toggle_use()
            return "break"

    def _error(self, exc):
        messagebox.showerror("NMR Data", str(exc), parent=self.winfo_toplevel())

    def toggle_use(self):
        try:
            row = self._selected_row()
            if row is not None and (not row.observable_id or not row.assignment):
                self.status.set("Enter an assignment before choosing this row for analysis.")
                return "break"
            dataset, item = self._record()
            set_observable_use(dataset, item.id, not item.included)
            self.on_changed()
        except ValueError as exc:
            self._error(exc)
        return "break"

    def show_map(self):
        try:
            dataset, item = self._record()
            if not item.assignment:
                raise ValueError("Assign this observation before selecting a structural target")
            self.on_select(dataset.id, item.id)
            self.on_map(item.assignment)
        except ValueError as exc:
            self._error(exc)

    def remove(self):
        try:
            row = self._selected_row()
            if row is None:
                raise ValueError("Select a visible row to remove")
            if row is not None and row.peak_id:
                if messagebox.askyesno("Remove peak", "Remove this source peak and its linked observation? Track links may become stale.", parent=self.winfo_toplevel()):
                    self.on_peak_remove(row.dataset_id, row.peak_id)
                return
            dataset, item = self._record()
            if messagebox.askyesno("Remove observation", "Remove this independent observation?", parent=self.winfo_toplevel()):
                remove_observable(dataset, item.id)
                self.selected_key = None
                self.on_changed()
        except ValueError as exc:
            self._error(exc)

    def edit(self, *, existing=False):
        row = self._selected_row()
        if existing and row is not None and not row.observable_id:
            return self.edit_assignment()
        try:
            dataset, item = self._record() if existing else (self.model.current_dataset, None)
            if existing and read_only(dataset):
                raise ValueError("Edit PCS project values through the existing project import, or use Copy values to create independent session data.")
        except ValueError as exc:
            self._error(exc)
            return
        top = tk.Toplevel(self)
        top.title("Edit observation" if existing else "Add observed shift")
        top.transient(self.winfo_toplevel())
        body = ttk.Frame(top, padding=10)
        body.pack(fill="both", expand=True)
        datasets = {f"{dataset_display_name(d)} · {d.id[:8]}": d for d in self.model.datasets if not read_only(d)}
        choice = tk.StringVar(value=next((name for name, d in datasets.items() if d is dataset), "New experiment"))
        selected = ttk.Combobox(body, textvariable=choice, values=("New experiment", *datasets), state="disabled" if existing else "readonly", width=32)
        ttk.Label(body, text="Experiment").grid(row=0, column=0, sticky="w")
        selected.grid(row=0, column=1, sticky="w", pady=3)
        variables = {}
        defaults = {"Name": "Prepared shifts", "Nucleus": "1H", "Role": "unspecified", "Temperature / K": "",
            "Assignment": item.assignment or "" if item else "", "Shift / ppm": f"{item.chemical_shift_ppm:.12g}" if item else "",
            "Uncertainty / ppm": "" if item is None or item.uncertainty_ppm is None else f"{item.uncertainty_ppm:.12g}"}
        metadata_fields = []
        for index, (label, value) in enumerate(defaults.items(), 1):
            variable = variables[label] = tk.StringVar(value=value)
            ttk.Label(body, text=label).grid(row=index, column=0, sticky="w", padx=(0, 8), pady=3)
            widget = (ttk.Combobox(body, textvariable=variable, values=("unspecified", "paramagnetic", "diamagnetic"), state="readonly", width=20)
                      if label == "Role" else ttk.Entry(body, textvariable=variable, width=24))
            widget.grid(row=index, column=1, sticky="w", pady=3)
            if index <= 4:
                metadata_fields.append(widget)
            if label == "Shift / ppm" and item and item.source_peak_id:
                widget.configure(state="readonly")
        use = tk.BooleanVar(value=item.included if item else True)
        ttk.Checkbutton(body, text="Use in shift comparison", variable=use).grid(row=8, column=1, sticky="w", pady=4)
        def update_fields(event=None):
            current = datasets.get(choice.get())
            for widget in metadata_fields:
                widget.configure(state="disabled" if current else "readonly" if isinstance(widget, ttk.Combobox) else "normal")
            if current:
                variables["Name"].set(dataset_display_name(current))
                variables["Nucleus"].set(current.conditions.nucleus or "")
                variables["Role"].set(current.experimental_role)
                variables["Temperature / K"].set(current.conditions.temperature_k or "")
        selected.bind("<<ComboboxSelected>>", update_fields)
        update_fields()
        def save():
            try:
                target = datasets.get(choice.get())
                added = target is None
                if added:
                    target = new_data_dataset(variables["Name"].get(), variables["Nucleus"].get(),
                        variables["Role"].get(), variables["Temperature / K"].get())
                observation = edit_observable(target, observable_id=item.id if item else None,
                    assignment=variables["Assignment"].get(),
                    shift_ppm=None if item and item.source_peak_id else variables["Shift / ppm"].get(),
                    uncertainty_ppm=variables["Uncertainty / ppm"].get(), included=use.get())
                if added:
                    self.model.datasets.append(target)
                self.selected_key = (target.id, "peak:" + observation.source_peak_id if observation.source_peak_id else observation.id)
                self.on_changed()
                self.on_select(target.id, observation.id)
                top.destroy()
            except ValueError as exc:
                messagebox.showerror("NMR Data", str(exc), parent=top)
        actions = ttk.Frame(body)
        actions.grid(row=9, column=0, columnspan=2, sticky="e", pady=(6, 0))
        self._button(actions, "Save", "save", save).pack(side="left")
        self._button(actions, "Cancel", "cancel", top.destroy).pack(side="left", padx=3)

    def import_file(self):
        path = filedialog.askopenfilename(parent=self.winfo_toplevel(), title="Import observed shifts",
            filetypes=(("Shift tables", "*.csv *.tsv *.txt"), ("All files", "*.*")))
        if path:
            try:
                self.import_dialog(Path(path).read_text(encoding="utf-8-sig"), source=path)
            except (OSError, UnicodeError) as exc:
                self._error(exc)

    def paste(self):
        try:
            text = self.clipboard_get()
        except tk.TclError:
            text = "Assignment,Shift\n"
        self.import_dialog(text, source="Pasted table")

    def copy_values(self):
        try:
            dataset, item = self._record()
            self.import_dialog(export_observation_table((dataset,)), source="Copied values", detach=True)
        except ValueError as exc:
            self._error(exc)

    def export(self):
        path = filedialog.asksaveasfilename(parent=self.winfo_toplevel(), title="Export NMR observations",
            defaultextension=".csv", filetypes=(("CSV", "*.csv"),))
        if path:
            try:
                with atomic_text_export(path, encoding="utf-8-sig", newline="") as handle:
                    handle.write(export_observation_table(self.model.datasets))
                self.status.set("Exported all observations, including excluded values and source identities.")
            except (OSError, ValueError) as exc:
                self._error(exc)

    def import_dialog(self, text, *, source, detach=False):
        top = tk.Toplevel(self)
        top.title("Review observed-shift import")
        top.geometry("760x580")
        top.transient(self.winfo_toplevel())
        body = ttk.Frame(top, padding=8)
        body.pack(fill="both", expand=True)
        settings = ttk.Frame(body)
        settings.pack(fill="x")
        name, nucleus, role = tk.StringVar(value=Path(source).stem), tk.StringVar(value="1H"), tk.StringVar(value="unspecified")
        for label, variable in (("Name", name), ("Nucleus", nucleus), ("Role", role)):
            ttk.Label(settings, text=label).pack(side="left", padx=(0, 4))
            widget = (ttk.Combobox(settings, textvariable=variable, values=("unspecified", "paramagnetic", "diamagnetic"), state="readonly", width=14)
                      if label == "Role" else ttk.Entry(settings, textvariable=variable, width=20 if label == "Name" else 5))
            widget.pack(side="left", padx=(0, 8))
        ttk.Label(body, text="Observed absolute shifts only. Legacy δ_obs / δ_dia columns are supported; δ_para is derived by Shift Analysis.",
            wraplength=720, style="NMR.Muted.TLabel").pack(fill="x", pady=5)
        editor = tk.Text(body, height=9, wrap="none")
        editor.pack(fill="both", expand=True)
        editor.insert("1.0", text)
        mappingbar = ttk.Frame(body)
        mappingbar.pack(fill="x", pady=5)
        assignment, shift = tk.StringVar(), tk.StringVar()
        boxes = []
        for label, variable in (("Assignment", assignment), ("Shift", shift)):
            ttk.Label(mappingbar, text=label).pack(side="left", padx=(0, 4))
            box = ttk.Combobox(mappingbar, textvariable=variable, state="readonly", width=20)
            box.pack(side="left", padx=(0, 8))
            boxes.append(box)
        review = ttk.Treeview(body, columns=("experiment", "assignment", "shift", "role"), show="headings", height=6)
        for column in review["columns"]:
            review.heading(column, text=column.capitalize())
            review.column(column, width=130)
        review.pack(fill="both", expand=True)
        options = ttk.Frame(body)
        options.pack(fill="x", pady=5)
        updates, detached = tk.BooleanVar(value=False), tk.BooleanVar(value=detach)
        override_nucleus = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text="Update exact loaded IDs", variable=updates).pack(side="left")
        ttk.Checkbutton(options, text="Import detached values (new identities)", variable=detached).pack(side="left", padx=8)
        ttk.Checkbutton(body, text="Use the nucleus above for all imported values", variable=override_nucleus).pack(anchor="w")
        ToolTip(options, "ID updates require unchanged source links and experiment conditions.\nDetached copies retain source evidence but have no authoritative spectrum link.")
        info = tk.StringVar(value="Read columns, preview the values, then explicitly Apply.")
        ttk.Label(body, textvariable=info, wraplength=720).pack(fill="x", pady=4)
        actions = ttk.Frame(body)
        actions.pack(fill="x")
        prepared = []
        reviewed_signature = [None]
        def invalidate(*args):
            prepared.clear()
            apply_button.configure(state="disabled")
        def read_columns():
            try:
                headers, _ = table_reader(editor.get("1.0", "end"))
                mapping = default_mapping(headers)
                for box in boxes:
                    box.configure(values=headers)
                assignment.set(mapping["assignment"])
                shift.set(mapping["shift_ppm"])
                return headers
            except ValueError as exc:
                info.set(str(exc))
                return ()
        def preview():
            invalidate()
            try:
                if not boxes[0]["values"]:
                    read_columns()
                records = parse_observation_table(editor.get("1.0", "end"), name=name.get(),
                    nucleus=nucleus.get(), role=role.get(), override_nucleus=override_nucleus.get(),
                    mapping={"assignment": assignment.get(), "shift_ppm": shift.get()})
                review.delete(*review.get_children())
                for record in records:
                    review.insert("", "end", values=(record.experiment, record.assignment or "—", f"{record.shift_ppm:.8g}", record.role))
                prepared.extend(records)
                reviewed_signature[0] = observation_state_signature(self.model)
                info.set(f"{len(records)} observations reviewed. Repeated assignments are retained and diagnosed; no automatic pairing or PCS transfer.")
                apply_button.configure(state="normal")
            except (ValueError, TypeError) as exc:
                info.set(str(exc))
        def apply():
            try:
                if not prepared:
                    return
                added = apply_observation_import(self.model, prepared, source=source,
                    update_existing=updates.get(), detach=detached.get(), reviewed_signature=reviewed_signature[0])
                self.on_changed()
                if added and added[0].observables:
                    item = added[0].observables[0]
                    self.selected_key = (added[0].id, "peak:" + item.source_peak_id if item.source_peak_id else item.id)
                    self.on_select(added[0].id, item.id)
                self.status.set(f"Imported {len(prepared)} observations. Open Shift Analysis and choose observed/reference experiments.")
                top.destroy()
            except ValueError as exc:
                info.set(str(exc))
        self._button(actions, "Read columns", "data_table", read_columns).pack(side="left")
        self._button(actions, "Preview", "preview", preview).pack(side="left", padx=3)
        apply_button = self._button(actions, "Apply", "apply", apply)
        apply_button.configure(state="disabled")
        apply_button.pack(side="right")
        self._button(actions, "Cancel", "cancel", top.destroy).pack(side="right", padx=3)
        for variable in (name, nucleus, role, assignment, shift, updates, detached, override_nucleus):
            variable.trace_add("write", invalidate)
        editor.bind("<KeyRelease>", invalidate)
        editor.bind("<<Paste>>", invalidate)
        read_columns()
