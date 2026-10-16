"""Compact recipe editor and isolated preview/review, with explicit batch commit."""
import json
from queue import Queue, Empty
from threading import Thread
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure

from ..core.batch_recipe import BatchRecipe, OPERATIONS
from ..processing.batch import prepare_batch, run_batch_preview, apply_batch
from .series_manager import SeriesManager
from .view_model import dataset_display_name
from .presentation import WorkbenchFigureCanvas, WorkbenchPanedWindow
from .actions import action, polish
from ui.style import get_app_fonts, table_row_height

DEFAULTS = {
    "apodization": {"method": "exponential", "lb_hz": 0.3, "gb_hz": None},
    "zero_fill": {"target_points": 32768},
    "fourier_transform": {},
    "phase": {"mode": "auto", "pivot_mode": "biggest_peak", "algorithm": "acme", "p0_deg": 0., "p1_deg": 0., "pivot_ppm": None, "pivot_index": None},
    "baseline": {"method": "bernstein", "order": 8, "lam": 1e6, "p": .001},
    "reference": {"target_ppm": None, "window_low_ppm": None, "window_high_ppm": None,
                  "preset_id": None, "resonance_id": None},
    "peak_pick": {"height_fraction": .01, "prominence_fraction": .005, "distance_points": 3},
}

OPERATION_LABELS = {
    "apodization": "Apodization", "zero_fill": "Zero Fill",
    "fourier_transform": "Fourier Transform", "phase": "Phase",
    "baseline": "Baseline", "reference": "Reference", "peak_pick": "Peak Picking",
}


class _BatchPreviewCanvas(WorkbenchFigureCanvas):
    """Keep the existing bitmap during a resize; render after sizing settles."""

    def __init__(self, figure, *, master, on_resize):
        self._resize_timer = None
        self._pending_resize = None
        self._last_size = None
        self._on_resize = on_resize
        super().__init__(figure, master=master)
        self.get_tk_widget().bind("<Destroy>", self._cancel_resize, add="+")

    def resize(self, event):
        if (event.width, event.height) == self._last_size:
            self._pending_resize = None
            return
        self._pending_resize = event
        widget = self.get_tk_widget()
        if self._resize_timer is not None:
            widget.after_cancel(self._resize_timer)
        if self._idle_draw_id is not None:
            widget.after_cancel(self._idle_draw_id)
            self._idle_draw_id = None
        self._resize_timer = widget.after(120, self._resize_latest)

    def _resize_latest(self):
        self._resize_timer = None
        event, self._pending_resize = self._pending_resize, None
        if event is not None:
            self._last_size = (event.width, event.height)
            super().resize(event)
            self._on_resize(event)

    def _cancel_resize(self, event):
        if event.widget is self.get_tk_widget():
            if self._resize_timer is not None:
                self.get_tk_widget().after_cancel(self._resize_timer)
                self._resize_timer = None
            self._pending_resize = None


class BatchProcessingWindow(tk.Toplevel):
    def __init__(self, parent, state, *, on_committed, committed_spectra, blocked_members, context=None):
        super().__init__(parent)
        self.title("VT Batch Processing")
        self.geometry("1080x790")
        self.minsize(900, 620)
        self.transient(parent)
        self._presentation = parent._presentation
        self.context = context
        active = next((s for s in state.series if s.id == state.selected_series_id), None)
        self._source_members = [(active, m.id, m.included) for m in active.members
            if context is not None and any(c.dataset is m.dataset for c in context.members)] if active else []
        class ContextManager(SeriesManager):
            @property
            def active_series(manager):
                return context if context is not None else super().active_series
        self.manager = ContextManager(state)
        if context is not None:
            self.title("Batch Processing — selected experiments")
        self.on_committed, self.committed_spectra, self.blocked_members = on_committed, committed_spectra, blocked_members
        self.preview = None
        self._poll_id = None
        self._generation = 0
        self._checked_members = set()
        self._series_id = None
        self.rows = [{"operation": op, "enabled": False, "parameters": dict(DEFAULTS[op])} for op in OPERATIONS]
        self.overrides = {}
        self._fields = {}
        self.name = tk.StringVar(self, value="Batch preset")
        self.start_from_raw = tk.BooleanVar(self, value=False)
        self.initialize_unprocessed = tk.BooleanVar(self, value=True)
        self.advanced_processing = tk.BooleanVar(self, value=False)
        self.status = tk.StringVar(self, value="Check experiments, enable operations, then preview. No changes until Apply.")
        body = ttk.Frame(self, padding=6)
        body.pack(fill="both", expand=True)
        top = ttk.Frame(body)
        top.pack(fill="x")
        ttk.Label(top, text="Series").pack(side="left", padx=(0, 4))
        self.selector = ttk.Combobox(top, state="readonly", width=24)
        self.selector.pack(side="left")
        self.selector.bind("<<ComboboxSelected>>", self._select_series)
        from .workflow_help import show_workflow_help
        action(top, "Guide", "search", lambda: show_workflow_help(self)).pack(side="right")
        footer = ttk.Frame(body)
        footer.pack(side="bottom", fill="x", pady=(4, 0))
        self.status_label = ttk.Label(footer, textvariable=self.status, wraplength=1030,
                                      style="NMR.Muted.TLabel")
        self.status_label.pack(fill="x", pady=(0, 4))
        footer.bind("<Configure>", self._resize_status)
        self.sections = WorkbenchPanedWindow(body, orient="vertical")
        self.sections.pack(fill="both", expand=True, pady=(4, 0))
        experiments = ttk.Frame(self.sections, padding=6)
        processing = ttk.Frame(self.sections, padding=6)
        preview_panel = ttk.Frame(self.sections, padding=6)
        for panel, weight in ((experiments, 1), (processing, 1), (preview_panel, 3)):
            self.sections.add(panel, weight=weight)
        self._section_layout_id = None
        self._sections_initialized = False
        for event in ("<Configure>", "<ButtonRelease-1>"):
            self.sections.bind(event, self._queue_section_layout, add="+")
        member_heading = ttk.Frame(experiments)
        member_heading.pack(fill="x", pady=(0, 4))
        ttk.Label(member_heading, text="Experiments", style="NMR.Section.TLabel").pack(side="left")
        self.target_summary = tk.StringVar(self)
        ttk.Label(member_heading, textvariable=self.target_summary).pack(side="right")
        member_table = ttk.Frame(experiments)
        member_table.pack(fill="both", expand=True)
        self.members = ttk.Treeview(member_table, columns=("process", "member", "use", "result"), show="headings", selectmode="browse", height=4, style="NMR.Batch.Treeview")
        for key, title, width in (("process", "", 65), ("member", "Experiment", 250), ("use", "Series status", 100), ("result", "Result", 570)):
            self.members.heading(key, text=title)
            self.members.column(key, width=width)
        self.members.column("process", stretch=False, minwidth=65)
        self.members.column("process", anchor="center")
        self.members.heading("process", command=self._toggle_all_members)
        self._all_checked = tk.BooleanVar(self, value=False)
        self._all_check = ttk.Checkbutton(self.members, text="All", variable=self._all_checked,
                                          command=self._toggle_all_members, style="NMR.BatchHeader.TCheckbutton")
        self._header_layout_id = None
        self._header_bounds = None
        self._header_style_signature = None
        self.members.bind("<Configure>", self._queue_header_layout)
        self.members.bind("<ButtonPress-1>", self._member_press)
        self.members.bind("<space>", self._member_space)
        member_scroll = ttk.Scrollbar(member_table, command=self.members.yview)
        def member_scroll_changed(*args):
            member_scroll.set(*args)
        self.members.configure(yscrollcommand=member_scroll_changed)
        member_scroll.pack(side="right", fill="y")
        self.members.pack(fill="both", expand=True)
        from ui.tooltips import ToolTip
        ToolTip(self.members, "Check experiments to process; select a row to inspect its spectrum. Series inclusion is unchanged.")
        source = ttk.Frame(processing)
        source.pack(fill="x", pady=(0, 4))
        ttk.Label(source, text="Processing", style="NMR.Section.TLabel").pack(side="left", padx=(0, 12))
        current_source = ttk.Radiobutton(source, text="Current spectrum", variable=self.start_from_raw, value=False, command=self._change_source)
        current_source.pack(side="left")
        raw_source = ttk.Radiobutton(source, text="Original FID", variable=self.start_from_raw, value=True, command=self._change_source)
        raw_source.pack(side="left", padx=8)
        self.source_note = tk.StringVar(self, value="Uses the current spectrum; unprocessed FIDs are prepared automatically.")
        self._source_tooltip = ToolTip(current_source, self.source_note.get())
        ToolTip(raw_source, "Restart from the original FID. Review the FT step and operation order; only Apply commits changes.")
        preset = ttk.Frame(processing)
        preset.pack(fill="x", pady=(0, 4))
        ttk.Label(preset, text="Preset").pack(side="left", padx=(0, 4))
        self.saved = ttk.Combobox(preset, state="readonly", width=24)
        self.saved.pack(side="left")
        self.saved.bind("<<ComboboxSelected>>", self._load_recipe)
        ttk.Label(preset, text="Name").pack(side="left", padx=(12, 4))
        ttk.Entry(preset, textvariable=self.name, width=22).pack(side="left")
        self.save_button = action(preset, "Save", "save", self._save_recipe, description="Save processing preset")
        self.save_button.pack(side="left", padx=(6, 4))
        self.delete_button = action(preset, "Delete", "delete", self._delete_recipe, state="disabled", description="Delete selected preset")
        self.delete_button.pack(side="left")
        edit = ttk.Frame(processing)
        edit.pack(fill="both", expand=True)
        operation_panel = ttk.Frame(edit)
        operation_panel.pack(side="left", fill="y")
        self.operations = ttk.Treeview(operation_panel, columns=("enabled", "operation"), show="headings", selectmode="browse", height=6)
        self.operations.heading("enabled", text="Enabled")
        self.operations.heading("operation", text="Operation")
        self.operations.column("enabled", width=72, minwidth=72)
        self.operations.column("operation", width=150)
        operation_scroll = ttk.Scrollbar(operation_panel, command=self.operations.yview)
        operation_scroll.pack(side="right", fill="y")
        self.operations.pack(side="left", fill="both", expand=True)
        self.operations.bind("<<TreeviewSelect>>", lambda event: self._edit_parameters())
        self._enabled_editors = {}
        self._enabled_bounds = {}
        self._enabled_layout_id = None
        self._operation_drag = None
        def operation_scroll_changed(*args):
            operation_scroll.set(*args)
            self._queue_enabled_layout()
        self.operations.configure(style="NMR.Batch.Treeview",
                                  yscrollcommand=operation_scroll_changed)
        self.operations.bind("<Configure>", self._queue_enabled_layout)
        self.operations.bind("<ButtonPress-1>", self._operation_press)
        self.operations.bind("<B1-Motion>", self._operation_motion)
        self.operations.bind("<ButtonRelease-1>", self._operation_release)
        self.operations.bind("<Return>", lambda event: self._edit_enabled(self._selected_operation()))
        self.operations.bind("<Escape>", self._cancel_operation_edit)
        self.operations.tag_configure("drop_target", foreground="#2459A6")
        ToolTip(self.operations, "Drag an operation row to change processing order. On/Off enables or disables the step.")
        parameter_panel = ttk.Frame(edit)
        parameter_panel.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self._parameter_canvas = tk.Canvas(parameter_panel, highlightthickness=0, borderwidth=0, height=150, width=1)
        parameter_scroll = ttk.Scrollbar(parameter_panel, command=self._parameter_canvas.yview)
        parameter_scroll.pack(side="right", fill="y")
        self._parameter_canvas.pack(side="left", fill="both", expand=True)
        self._parameter_canvas.configure(yscrollcommand=parameter_scroll.set)
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._parameter_canvas.bind(sequence, self._scroll_parameters)
        self.parameters = ttk.Frame(self._parameter_canvas)
        self._parameter_window = self._parameter_canvas.create_window((0, 0), window=self.parameters, anchor="nw")
        self.parameters.bind("<Configure>", self._parameter_content_changed)
        self._parameter_canvas.bind("<Configure>", self._parameter_viewport_changed)
        actions = ttk.Frame(preview_panel)
        actions.pack(fill="x", pady=(0, 4))
        ttk.Label(actions, text="Spectrum preview", style="NMR.Section.TLabel").pack(side="left")
        self.preview_member = ttk.Combobox(actions, state="readonly", width=30)
        self.preview_member.pack(side="right")
        self.preview_member.bind("<<ComboboxSelected>>", self._plot_preview)
        self.plot_mode = tk.StringVar(self, value="Overlay")
        plot_choice = ttk.Combobox(actions, textvariable=self.plot_mode, values=("Committed", "Preview", "Overlay"), state="readonly", width=11)
        plot_choice.pack(side="right", padx=6)
        plot_choice.bind("<<ComboboxSelected>>", self._plot_preview)
        ToolTip(plot_choice, "Committed = before; Preview = after; Overlay = both. Preview is temporary until Apply.")
        self.figure = Figure(figsize=(9, 2.1))
        self.axes = self.figure.add_subplot(111)
        self._preview_margin_width = None
        self.canvas = _BatchPreviewCanvas(self.figure, master=preview_panel, on_resize=self._fit_preview_width)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        commit = ttk.Frame(footer)
        commit.pack(fill="x")
        action(commit, "Preview checked", "preview", self._preview).pack(side="left")
        action(commit, "Cancel", "cancel", self._cancel, description="Cancel preview").pack(side="left", padx=5)
        self.apply_button = action(commit, "Apply reviewed previews", "apply", self._apply, primary=True, state="disabled")
        self.apply_button.pack(side="right")
        self.members.bind("<<TreeviewSelect>>", self._plot_preview, add="+")
        self._draw_operations()
        self._edit_parameters()
        polish(self)
        self.refresh()

    def _queue_section_layout(self, event=None):
        if self._section_layout_id is None:
            self._section_layout_id = self.after_idle(self._layout_sections)

    def _resize_status(self, event):
        width = max(200, event.width)
        if int(self.status_label.cget("wraplength")) != width:
            self.status_label.configure(wraplength=width)

    def _layout_sections(self):
        self._section_layout_id = None
        height = self.sections.winfo_height()
        if height <= 1:
            return
        from ui.ribbon.icon_service import tk_scale_factor
        scale = tk_scale_factor(self._root())
        sash = int(ttk.Style(self).lookup("Sash", "sashthickness") or 6)
        minimums = [round(value * scale) for value in (100, 130, 140)]
        available = max(1, height - 2 * sash)
        if sum(minimums) > available:
            minimums = [max(1, int(value * available / sum(minimums))) for value in minimums]
        if not self._sections_initialized:
            first = max(minimums[0], self.members.winfo_reqheight() + round(38 * scale))
            second = first + sash
            # Allow all six standard operations to fit at the default split.
            # Users can still shrink Processing and scroll its operation list.
            processing_height = self.operations.winfo_reqheight() + round(78 * scale)
            second = max(second, first + sash + processing_height)
            self._sections_initialized = True
        else:
            first, second = self.sections.sashpos(0), self.sections.sashpos(1)
        first = max(minimums[0], min(first, available - minimums[1] - minimums[2]))
        second = max(first + sash + minimums[1], min(second, height - sash - minimums[2]))
        if self.sections.sashpos(0) != first:
            self.sections.sashpos(0, first)
        if self.sections.sashpos(1) != second:
            self.sections.sashpos(1, second)

    def _parameter_viewport_changed(self, event):
        if int(float(self._parameter_canvas.itemcget(self._parameter_window, "width"))) != event.width:
            self._parameter_canvas.itemconfigure(self._parameter_window, width=event.width)

    def _parameter_content_changed(self, event=None):
        bounds = self._parameter_canvas.bbox("all")
        if bounds != getattr(self, "_parameter_scroll_bounds", None):
            self._parameter_scroll_bounds = bounds
            self._parameter_canvas.configure(scrollregion=bounds)

    def _bind_parameter_wheel(self):
        # Local bindings only: no global wheel handlers and no interception of
        # comboboxes' native selection/popup behavior.
        def bind_children(widget):
            if not isinstance(widget, (ttk.Combobox, ttk.Treeview, ttk.Spinbox)):
                for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    if not widget.bind(sequence):
                        widget.bind(sequence, self._scroll_parameters)
            for child in widget.winfo_children():
                bind_children(child)
        bind_children(self.parameters)

    def _scroll_parameters(self, event):
        bounds = self._parameter_canvas.bbox("all")
        if not bounds or bounds[3] <= self._parameter_canvas.winfo_height():
            return
        number = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        if number == 4:
            units = -1
        elif number == 5:
            units = 1
        elif delta:
            units = -1 if delta > 0 else 1
        else:
            return
        self._parameter_canvas.yview_scroll(units * 3, "units")
        return "break"

    def _invalidate(self):
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
            self._poll_id = None
        self._generation += 1
        self.preview = None
        self.apply_button.configure(state="disabled")
        self._plot_preview()

    def _target_member_ids(self):
        return [member_id for member_id in self.members.get_children()
                if member_id in self._checked_members]

    def _update_target_summary(self):
        total = len(self.members.get_children())
        count = len(self._target_member_ids())
        self.target_summary.set(f"{count} / {total} to process")
        self.members.heading("process", text="")
        self._all_checked.set(bool(total and count == total))
        self._all_check.state(["alternate"] if 0 < count < total else ["!alternate"])

    def _set_checked_members(self, ids):
        self._checked_members = set(ids) & set(self.members.get_children())
        for member_id in self.members.get_children():
            self.members.set(member_id, "process", "☑" if member_id in self._checked_members else "☐")
        self._update_target_summary()

    def _toggle_member(self, member_id):
        if member_id in self._checked_members:
            self._checked_members.discard(member_id)
        else:
            self._checked_members.add(member_id)
        self.members.set(member_id, "process", "☑" if member_id in self._checked_members else "☐")
        self._update_target_summary()
        self._invalidate()

    def _toggle_all_members(self):
        ids = self.members.get_children()
        self._set_checked_members([] if len(self._checked_members) == len(ids) else ids)
        self._invalidate()

    def _member_press(self, event):
        # Native cells are clipped/scrolled by Treeview; no overlapping widgets.
        if (self.members.identify_region(event.x, event.y) == "cell"
                and self.members.identify_column(event.x) == "#1"):
            member_id = self.members.identify_row(event.y)
            if member_id:
                self._toggle_member(member_id)
                return "break"

    def _queue_header_layout(self, event=None):
        if self._header_layout_id is None:
            self._header_layout_id = self.after_idle(self._position_header_check)

    def _position_header_check(self):
        self._header_layout_id = None
        width = self.members.column("process", "width")
        style = ttk.Style(self)
        signature = (style.theme_use(), style.lookup("NMR.Batch.Treeview.Heading", "font"),
                     style.lookup("NMR.Batch.Treeview.Heading", "padding"))
        if self._header_bounds is not None and self._header_bounds[2] == width - 2 and signature == self._header_style_signature:
            return
        # Measure the native heading itself, never a scrolled/partial row bbox.
        bottom = 0
        for y in range(1, min(self.members.winfo_height(), 100)):
            if self.members.identify_region(width // 2, y) == "heading":
                bottom = y + 1
            elif bottom:
                break
        bounds = (1, 1, width - 2, max(1, bottom - 2))
        if bottom and bounds != self._header_bounds:
            self._header_bounds = bounds
            self._all_check.place(x=bounds[0], y=bounds[1], width=bounds[2], height=bounds[3])
        if bottom:
            self._header_style_signature = signature

    def _member_space(self, event):
        selected = self.members.selection()
        if selected:
            self._toggle_member(selected[0])
        return "break"

    def refresh(self):
        self._invalidate()
        self._series_ids = [s.id for s in self.manager.state.series]
        self.selector.configure(values=[s.name for s in self.manager.state.series])
        series = self.manager.active_series
        if self.context is not None:
            self.selector.configure(values=(self.context.name,), state="disabled")
            self.selector.current(0)
        elif series:
            self.selector.current(self._series_ids.index(series.id))
        else:
            self.selector.set("")
        self._checked_members.clear()
        self.members.delete(*self.members.get_children())
        self.saved.set("")
        self.delete_button.configure(state="disabled")
        if not series:
            self.save_button.configure(state="disabled")
            self._update_target_summary()
            return
        self.save_button.configure(state="normal")
        if self._series_id != series.id:
            self.overrides = {}
        self._series_id = series.id
        for m in series.members:
            self.members.insert("", "end", iid=m.id, values=("", m.label or dataset_display_name(m.dataset), "Included" if m.included else "Excluded", "Not previewed"))
        self._set_checked_members([m.id for m in series.members if m.included])
        self._refresh_saved_presets()
        self.saved.set("")
        self._plot_ids = [m.id for m in series.members]
        self.preview_member.configure(values=[m.label or dataset_display_name(m.dataset) for m in series.members])
        if self._plot_ids:
            self.preview_member.current(0)
        self._plot_preview()

    def _select_series(self, event=None):
        self.manager.select_series(self._series_ids[self.selector.current()])
        self.refresh()

    def _select_included(self):
        series = self.manager.active_series
        self._set_checked_members([m.id for m in series.members if m.included] if series else [])
        self._invalidate()

    def _draw_operations(self):
        for editor in self._enabled_editors.values():
            editor.destroy()
        self._enabled_editors.clear()
        self._enabled_bounds.clear()
        self.operations.delete(*self.operations.get_children())
        for row in self.rows:
            if row["operation"] == "fourier_transform" and not self.advanced_processing.get() and not row["enabled"]:
                continue
            operation = row["operation"]
            self.operations.insert("", "end", iid=operation, values=("", OPERATION_LABELS[operation]))
            editor = ttk.Combobox(self.operations, state="readonly", width=4, values=("On", "Off"))
            editor.set("On" if row["enabled"] else "Off")
            editor.bind("<<ComboboxSelected>>",
                        lambda event, op=operation: self._commit_enabled(op))
            editor.bind("<FocusIn>", lambda event, op=operation: self.operations.selection_set(op))
            self._enabled_editors[operation] = editor
        self._queue_enabled_layout()

    def _queue_enabled_layout(self, event=None):
        if self._enabled_layout_id is None:
            self._enabled_layout_id = self.after_idle(self._position_enabled_editors)

    def _position_enabled_editors(self, event=None):
        self._enabled_layout_id = None
        # Treeview has no native cell widgets; keep real comboboxes over each
        # visible cell, including after resize, scrolling or row reordering.
        if self._enabled_editors:
            style = ttk.Style(self)
            fonts = get_app_fonts(self)
            height = max(
                max(editor.winfo_reqheight() for editor in self._enabled_editors.values()) + 4,
                table_row_height(self, fonts["table"], fonts["table_rowheight"]),
            )
            if str(style.lookup("NMR.Batch.Treeview", "rowheight")) != str(height):
                style.configure("NMR.Batch.Treeview", rowheight=height)
        for operation, editor in self._enabled_editors.items():
            bounds = self.operations.bbox(operation, "enabled")
            if bounds == self._enabled_bounds.get(operation):
                continue
            self._enabled_bounds[operation] = bounds
            if bounds:
                editor.place(x=bounds[0] + 2, y=bounds[1] + 2,
                             width=bounds[2] - 4, height=bounds[3] - 4)
            else:
                editor.place_forget()

    def _commit_enabled(self, operation):
        editor = self._enabled_editors[operation]
        row = next(r for r in self.rows if r["operation"] == operation)
        enabled = editor.get() == "On"
        self.operations.selection_set(operation)
        if enabled != row["enabled"]:
            row["enabled"] = enabled
            self._invalidate()

    def _edit_enabled(self, operation):
        if operation in self._enabled_editors:
            self.operations.selection_set(operation)
            self.operations.see(operation)
            self._queue_enabled_layout()
            self._enabled_editors[operation].focus_set()
        return "break"

    def _cancel_operation_edit(self, event=None):
        self._operation_drag = None
        self.operations.configure(cursor="")
        for item in self.operations.get_children():
            self.operations.item(item, tags=())
        return "break"

    def _operation_press(self, event):
        self._cancel_operation_edit()
        item = self.operations.identify_row(event.y)
        if self.operations.identify_region(event.x, event.y) != "cell" or not item:
            return
        if self.operations.identify_column(event.x) == "#1":
            return self._edit_enabled(item)
        self._operation_drag = (item, event.y, False)

    def _operation_motion(self, event):
        if self._operation_drag is None:
            return
        item, start, moved = self._operation_drag
        moved = moved or abs(event.y - start) > 4
        self._operation_drag = (item, start, moved)
        if moved:
            self.operations.configure(cursor="sb_v_double_arrow")
            target = self.operations.identify_row(event.y)
            for row in self.operations.get_children():
                self.operations.item(row, tags=("drop_target",) if row == target else ())
            return "break"

    def _operation_release(self, event):
        drag = self._operation_drag
        self._cancel_operation_edit()
        if drag is None or not drag[2]:
            return
        source = drag[0]
        target = self.operations.identify_row(event.y)
        if not target or source == target:
            return
        bounds = self.operations.bbox(target)
        after = event.y >= bounds[1] + bounds[3] / 2
        row = next(r for r in self.rows if r["operation"] == source)
        self.rows.remove(row)
        index = next(i for i, r in enumerate(self.rows) if r["operation"] == target)
        self.rows.insert(index + int(after), row)
        self._invalidate()
        self._draw_operations()
        self.operations.selection_set(source)
        self.operations.focus(source)
        return "break"

    def _fit_preview_width(self, event):
        if event.width > 200 and event.width != self._preview_margin_width:
            self._preview_margin_width = event.width
            from ui.ribbon.icon_service import tk_scale_factor
            scale = tk_scale_factor(self._root())
            self.figure.subplots_adjust(left=min(.2, 62 * scale / event.width),
                                        right=1 - 12 * scale / event.width)
            # The coalesced canvas resize already schedules the final draw.

    def _show_advanced_processing(self):
        selected = self._selected_operation()
        self._draw_operations()
        if selected and self.operations.exists(selected):
            self.operations.selection_set(selected)

    def _change_source(self):
        self.advanced_processing.set(self.start_from_raw.get())
        self.initialize_unprocessed.set(True)
        # Raw processing needs an explicit FT step, visible in the reviewed
        # operation list. Current-spectrum mode must not repeat FT by default.
        for row in self.rows:
            if row["operation"] == "fourier_transform":
                row["enabled"] = self.start_from_raw.get()
        self._update_source_note()
        self._show_advanced_processing()
        self._invalidate()

    def _update_source_note(self):
        if self.start_from_raw.get():
            text = "Restart from the original FID; replaces existing processing only after Preview and Apply."
        elif not self.initialize_unprocessed.get():
            text = "Uses the current spectrum. Loaded legacy preset disables automatic FID preparation."
        else:
            text = "Uses the current spectrum; unprocessed FIDs are prepared automatically."
        self.source_note.set(text)
        self._source_tooltip.update_text(text)

    def _selected_operation(self):
        return self.operations.selection()[0] if self.operations.selection() else None

    def _toggle(self):
        op = self._selected_operation()
        if op:
            row = next(r for r in self.rows if r["operation"] == op)
            row["enabled"] = not row["enabled"]
            self._invalidate()
            self._draw_operations()
            self.operations.selection_set(op)

    def _move(self, delta):
        op = self._selected_operation()
        if op:
            visible = list(self.operations.get_children())
            neighbor = visible[max(0, min(visible.index(op) + delta, len(visible) - 1))]
            index = next(i for i, r in enumerate(self.rows) if r["operation"] == op)
            target = next(i for i, r in enumerate(self.rows) if r["operation"] == neighbor)
            self.rows.insert(target, self.rows.pop(index))
            self._invalidate()
            self._draw_operations()
            self.operations.selection_set(op)

    def _edit_parameters(self):
        self._parameter_canvas.yview_moveto(0)
        for widget in self.parameters.winfo_children():
            widget.destroy()
        self._fields = {}
        op = self._selected_operation()
        if not op:
            ttk.Label(self.parameters, text="Select a processing step\nConfigure its parameters, then enable it.\nPreview first; Apply only after review.",
                      style="NMR.Muted.TLabel", padding=12).pack(anchor="w")
            self._bind_parameter_wheel()
            return
        row = next(r for r in self.rows if r["operation"] == op)
        values = {**DEFAULTS[op], **row["parameters"]}
        if op == "phase":
            # Older/manual dispatched recipes keep their explicit pivot policy.
            values["pivot_mode"] = row["parameters"].get("pivot_mode", "explicit")
            values["mode"] = row["parameters"].get("mode", "manual")
        if op in {"apodization", "zero_fill", "peak_pick"}:
            self._build_simple_parameters(op, values)
            self._bind_parameter_wheel()
            self._plot_preview()
            return
        if op == "reference":
            from .batch_reference_editor import BatchReferenceEditor
            series = self.manager.active_series
            ids = set(self._target_member_ids())
            targets = [m for m in series.members if not ids or m.id in ids] if series else []
            nuclei = {m.dataset.conditions.nucleus for m in targets if m.dataset.conditions.nucleus}
            self._reference_editor = BatchReferenceEditor(self.parameters, values, nuclei, self._invalidate)
            self._reference_editor.grid(row=0, column=0, columnspan=6, sticky="ew")
            self._fields = self._reference_editor.fields
            actions = ttk.Frame(self.parameters)
            actions.grid(row=1, column=0, columnspan=6, sticky="w", pady=5)
            action(actions, "Set common", "apply", lambda: self._store_parameters(False), description="Store common settings for this operation; Preview/Apply commits changes").pack(side="left", padx=3)
            action(actions, "Set for selected", "settings", lambda: self._store_parameters(True), description="Store this operation's settings for the selected experiment").pack(side="left", padx=3)
            action(actions, "Clear selected overrides", "clear", self._clear_overrides, description="Clear ALL operation overrides for the selected experiment; use common settings").pack(side="left", padx=3)
            self._bind_parameter_wheel()
            self._plot_preview()
            return
        parameter_widgets = {}
        for i, (key, value) in enumerate(values.items()):
            variable = tk.StringVar(self, value="" if value is None else str(value))
            self._fields[key] = variable
            labels = {"lb_hz": "Line broadening / Hz", "gb_hz": "Gaussian width / Hz", "target_points": "Output points",
                      "p0_deg": "Constant phase / °", "p1_deg": "Phase slope / °", "pivot_ppm": "Pivot / ppm",
                      "pivot_index": "Pivot sample index", "pivot_mode": "Pivot strategy", "height_fraction": "Height fraction",
                      "prominence_fraction": "Prominence fraction", "distance_points": "Min. separation / points",
                      "order": "Polynomial order", "lam": "Smoothness λ", "p": "Asymmetry p"}
            ttk.Label(self.parameters, text=labels.get(key, key.replace("_", " "))).grid(row=i // 3, column=2 * (i % 3), sticky="w")
            choices = {"mode": ("manual", "auto"), "pivot_mode": ("biggest_peak", "explicit"), "algorithm": ("peak_minima", "acme")}
            if op == "baseline":
                choices["method"] = ("bernstein", "whittaker", "asls", "arpls")
            widget = (ttk.Combobox(self.parameters, textvariable=variable, values=choices[key], state="readonly", width=12)
                      if key in choices else ttk.Entry(self.parameters, textvariable=variable, width=12))
            parameter_widgets[key] = widget
            widget.grid(row=i // 3, column=2 * (i % 3) + 1, padx=4, pady=2)
            variable.trace_add("write", lambda *args: self._invalidate())
        actions = ttk.Frame(self.parameters)
        actions.grid(row=5, column=0, columnspan=6, sticky="w")
        action(actions, "Set common", "apply", lambda: self._store_parameters(False), description="Store common settings for this operation; Preview/Apply commits changes").pack(side="left")
        action(actions, "Set for selected", "settings", lambda: self._store_parameters(True), description="Store this operation's settings for the selected experiment").pack(side="left", padx=5)
        action(actions, "Clear selected overrides", "clear", self._clear_overrides, description="Clear ALL operation overrides for the selected experiment; use common settings").pack(side="left")
        if op == "phase":
            def update_phase_controls(*args):
                automatic = self._fields["mode"].get() == "auto"
                biggest = self._fields["pivot_mode"].get() == "biggest_peak"
                for key in ("p0_deg", "p1_deg"):
                    parameter_widgets[key].configure(state="disabled" if automatic else "normal")
                for key in ("pivot_ppm", "pivot_index"):
                    parameter_widgets[key].configure(state="disabled" if biggest else "normal")
                parameter_widgets["algorithm"].configure(state="readonly" if automatic else "disabled")
            for key in ("mode", "pivot_mode"):
                self._fields[key].trace_add("write", update_phase_controls)
            update_phase_controls()
            ttk.Label(self.parameters, text="Auto: estimate P0/P1 per member. Biggest peak: independent pivot; explicit: use ppm/index. "
                      "Manual: use common P0/P1 or selected-member overrides. Review every preview before Apply.",
                      wraplength=650).grid(row=4, column=0, columnspan=6, sticky="w", pady=4)
        self._bind_parameter_wheel()
        self._plot_preview()

    def _build_simple_parameters(self, operation, values):
        """Presentation only; retain recipe keys, units and stored parameters."""
        from ui.tooltips import ToolTip
        ttk.Label(self.parameters, text=OPERATION_LABELS[operation],
                  style="NMR.Section.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        captions = {
            "method": ("Window", "", "Exponential uses line broadening; Gaussian uses Gaussian width."),
            "lb_hz": ("Line broadening", "Hz", "Exponential window width in Hz; greater width broadens peaks."),
            "gb_hz": ("Gaussian width", "Hz", "Gaussian window width in Hz."),
            "target_points": ("Output points", "points", "Target spectrum size after zero filling; no new measured data are added."),
            "height_fraction": ("Height threshold", "fraction", "Fraction of signal height: 0.01 means 1%."),
            "prominence_fraction": ("Prominence threshold", "fraction", "Fraction used for peak prominence: 0.005 means 0.5%."),
            "distance_points": ("Minimum separation", "points", "Minimum distance between picked peaks in native samples, not ppm."),
        }
        field_rows = {}
        for index, (key, value) in enumerate(values.items(), start=1):
            variable = tk.StringVar(self, value="" if value is None else str(value))
            self._fields[key] = variable
            caption, unit, tip = captions.get(key, (key.replace("_", " ").title(), "", ""))
            label = ttk.Label(self.parameters, text=caption)
            label.grid(row=index, column=0, sticky="w", padx=(0, 10), pady=2)
            if key == "method":
                widget = ttk.Combobox(self.parameters, state="readonly",
                                      values=("Exponential", "Gaussian"), width=14)
                widget.set(variable.get().capitalize())
                widget.bind("<<ComboboxSelected>>", lambda event, target=variable: target.set(event.widget.get().lower()))
            else:
                widget = ttk.Entry(self.parameters, textvariable=variable, width=10)
            widget.grid(row=index, column=1, sticky="w", pady=2)
            unit_label = ttk.Label(self.parameters, text=unit, style="NMR.Muted.TLabel")
            unit_label.grid(row=index, column=2, sticky="w", padx=(6, 0), pady=2)
            field_rows[key] = (label, widget, unit_label)
            if tip:
                ToolTip(label, tip)
                ToolTip(widget, tip)
            variable.trace_add("write", lambda *args: self._invalidate())
        if operation == "apodization":
            def show_window_fields(*args):
                method = self._fields["method"].get().lower()
                for key, visible in (("lb_hz", method == "exponential"), ("gb_hz", method == "gaussian")):
                    for widget in field_rows[key]:
                        widget.grid() if visible else widget.grid_remove()
            self._fields["method"].trace_add("write", show_window_fields)
            show_window_fields()
        footer_row = len(values) + 1
        ttk.Separator(self.parameters).grid(row=footer_row, column=0, columnspan=3, sticky="ew", pady=(6, 4))
        actions = ttk.Frame(self.parameters)
        actions.grid(row=footer_row + 1, column=0, columnspan=3, sticky="w")
        action(actions, "Set common", "apply", lambda: self._store_parameters(False),
               description="Store these common settings; Preview and Apply commit spectrum changes").pack(side="left")
        action(actions, "Set for selected", "settings", lambda: self._store_parameters(True),
               description="Store settings for the selected experiment only").pack(side="left", padx=4)
        action(actions, "Clear selected overrides", "clear", self._clear_overrides,
               description="Clear ALL operation overrides for the selected experiment; use common settings").pack(side="left")

    def _store_parameters(self, override):
        try:
            op = self._selected_operation()
            params = {}
            if op == "reference":
                self._reference_editor.validate_choice()
            for key, variable in self._fields.items():
                text = variable.get().strip()
                if not text:
                    continue
                params[key] = text if key in {"method", "preset_id", "resonance_id", "mode", "pivot_mode", "algorithm"} else (int(text) if key in {"order", "target_points", "distance_points"} else float(text))
            if not override and "observed_ppm" in params:
                raise ValueError("Observed reference ppm belongs to per-member overrides")
            if override:
                if not self.members.selection():
                    raise ValueError("Select members to override")
                for member_id in self.members.selection():
                    self.overrides.setdefault(member_id, {})[op] = params
            else:
                next(r for r in self.rows if r["operation"] == op)["parameters"] = params
            self._invalidate()
            self.status.set("Parameters stored. Member overrides: " + json.dumps(self.overrides))
        except (ValueError, StopIteration) as exc:
            self.status.set(str(exc))

    def _clear_overrides(self):
        for member_id in self.members.selection():
            self.overrides.pop(member_id, None)
        self._invalidate()
        self.status.set("Selected member overrides cleared")

    def _recipe(self):
        ids = self._target_member_ids()
        if not ids:
            raise ValueError("Check at least one experiment to process")
        # A checked excluded member is explicit permission for this batch only.
        return BatchRecipe.create(self.name.get(), self.rows, member_ids=ids,
            overrides=self.overrides, include_excluded=True, start_from_raw=self.start_from_raw.get(),
            initialize_unprocessed=self.initialize_unprocessed.get())

    def _save_recipe(self):
        try:
            recipe = self._recipe()
            series = self.manager.active_series
            if series is None:
                raise ValueError("Select a series")
            current = self.saved.current()
            selected_id = self._saved_ids[current] if current >= 0 else None
            existing = next((r for r in self._saved_presets() if r.id == selected_id and r.name == recipe.name), None)
            common = self.context is not None or (existing is not None and any(r.id == existing.id for r in self.manager.state.batch_presets))
            if common:
                config = recipe.configuration
                config.update(member_ids=[], overrides={}, include_excluded=False)
                recipe = BatchRecipe.create(recipe.name, **config)
            if existing is not None:
                recipe = BatchRecipe(recipe.name, recipe.configuration_json, existing.id)
            if self.context is not None or any(r.id == recipe.id for r in self.manager.state.batch_presets):
                presets = self.manager.state.batch_presets
                presets[:] = [r for r in presets if r.id != recipe.id] + [recipe]
            else:
                series.set_batch_recipe(recipe)
            self._refresh_saved_presets()
            self.saved.current(self._saved_ids.index(recipe.id))
            self.delete_button.configure(state="normal")
            self.status.set("Common preset saved to Workbench; selection/member overrides are not stored. Save session persists it."
                            if common else "Preset saved; Save session persists it. No processing performed.")
        except ValueError as exc:
            self.status.set(str(exc))

    def _load_recipe(self, event=None):
        series = self.manager.active_series
        if self.saved.current() < 0:
            return
        recipe = next(r for r in self._saved_presets() if r.id == self._saved_ids[self.saved.current()])
        common = any(r.id == recipe.id for r in self.manager.state.batch_presets)
        config = recipe.configuration
        self.name.set(recipe.name)
        self.rows = config["operations"]
        self.overrides = {} if common else config.get("overrides", {})
        self.start_from_raw.set(config.get("start_from_raw", False))
        self.initialize_unprocessed.set(config.get("initialize_unprocessed", False))
        self.advanced_processing.set(self.start_from_raw.get() or not self.initialize_unprocessed.get()
                                     or any(r["operation"] == "fourier_transform" and r.get("enabled", True) for r in self.rows))
        self._show_advanced_processing()
        self._update_source_note()
        ids = config.get("member_ids") or list(self.members.get_children())
        missing = [i for i in ids if not self.members.exists(i)]
        allowed = {m.id for m in series.members if m.included or config.get("include_excluded", False)}
        if not common:
            self._set_checked_members([i for i in ids if i in allowed])
        else:
            missing = []
        self.delete_button.configure(state="normal")
        self._invalidate()
        self._draw_operations()
        self.status.set(("Common preset loaded; checked targets kept, member overrides cleared. Preview required."
                         if common else "Preset loaded; preview required.")
                        + (f" Missing target members: {missing}" if missing else ""))

    def _delete_recipe(self):
        if self.saved.current() >= 0:
            series = self.manager.active_series
            recipe_id = self._saved_ids[self.saved.current()]
            if any(r.id == recipe_id for r in self.manager.state.batch_presets):
                self.manager.state.batch_presets[:] = [r for r in self.manager.state.batch_presets if r.id != recipe_id]
            else:
                series.remove_batch_recipe(recipe_id)
            self._refresh_saved_presets()
            self.saved.set("")
            self.delete_button.configure(state="disabled")
            self.status.set("Saved preset deleted; current settings and checked experiments are unchanged.")

    def _saved_presets(self):
        series = self.manager.active_series
        return tuple(self.manager.state.batch_presets) + (series.batch_recipes if series is not None and self.context is None else ())

    def _refresh_saved_presets(self):
        presets = self._saved_presets()
        self._saved_ids = [r.id for r in presets]
        self.saved.configure(state="readonly", values=[r.name for r in presets])

    def _preview(self):
        self._invalidate()
        try:
            recipe = self._recipe()
            self._validate_context()
            blocked = set(recipe.configuration["member_ids"]) & set(self.blocked_members())
            if blocked:
                raise ValueError("Apply/cancel the single-spectrum preview before processing that member")
            request = prepare_batch(self.manager.active_series, recipe, committed_by_dataset=self.committed_spectra())
        except (ValueError, TypeError) as exc:
            self.status.set(str(exc))
            return
        generation = self._generation
        queue = Queue()
        def worker():
            try:
                queue.put(run_batch_preview(request))
            except Exception as exc:
                queue.put(exc)
        Thread(target=worker, daemon=True).start()
        self.status.set("Preparing isolated previews… Cancel discards results; datasets remain unchanged.")
        def poll():
            self._poll_id = None
            try:
                result = queue.get_nowait()
            except Empty:
                if generation == self._generation:
                    self._poll_id = self.after(40, poll)
                return
            if generation != self._generation:
                return
            if isinstance(result, Exception):
                self.status.set(f"Preview failed: {result}")
                return
            self.preview = result
            self._present(result.results)
            self.apply_button.configure(state="normal" if any(r.status == "ready" for r in result.results) else "disabled")
            self._plot_preview()
        self._poll_id = self.after(40, poll)

    def _present(self, results):
        for result in results:
            if self.members.exists(result.member_id):
                values = self.members.item(result.member_id, "values")
                self.members.item(result.member_id, values=(*values[:3], f"{result.status}: {result.message}; " + "; ".join(result.warnings)))
        counts = {state: sum(r.status == state for r in results) for state in ("ready", "succeeded", "failed", "skipped")}
        self.status.set(", ".join(f"{value} {key}" for key, value in counts.items()) + ". Inspect rows and preview before committing.")

    def _plot_preview(self, event=None):
        span = getattr(self, "_reference_span", None)
        if span is not None:
            span.disconnect_events()
            self._reference_span = None
        self.axes.clear()
        ids = getattr(self, "_plot_ids", [])
        if event is not None and event.widget is self.members:
            selected = self.members.selection()
            focused = self.members.focus()
            member_id = focused if focused in selected else (selected[0] if selected else None)
            if member_id in ids:
                self.preview_member.current(ids.index(member_id))
        index = self.preview_member.current()
        member_id = ids[index] if 0 <= index < len(ids) else None
        if member_id is not None and self.members.exists(member_id):
            if self.members.selection() != (member_id,):
                self.members.selection_set(member_id)
            self.members.focus(member_id)
        series = self.manager.active_series
        member = next((m for m in series.members if m.id == member_id), None) if series else None
        if member is not None and member.dataset is not None:
            dataset = member.dataset
            result = next((r for r in self.preview.results if r.member_id == member_id and r.status == "ready"), None) if self.preview else None
            original = result.input_spectrum if result else (self.committed_spectra().get(dataset.id)
                or dataset.restored_committed_spectrum or dataset.processed_spectrum or dataset.raw_spectrum)
            processed = result.spectrum if result else None
            mode = self.plot_mode.get()
            same_axes = original is not None and processed is not None and (original.domain, original.axis_unit) == (processed.domain, processed.axis_unit)
            # Time and frequency data cannot share one scientific axis.
            spectra = []
            if original is not None and (mode == "Committed" or processed is None or (mode == "Overlay" and same_axes)):
                spectra.append((original, "Committed input", .6 if processed is not None else 1.))
            if processed is not None and mode != "Committed":
                spectra.append((processed, "Batch preview", 1.))
            for spectrum, label, alpha in spectra:
                self.axes.plot(spectrum.axis, np.real(spectrum.intensities), label=label, alpha=alpha)
            if spectra:
                shown = spectra[-1][0]
                self.axes.set_xlabel(shown.axis_unit)
                if shown.axis_unit.lower() == "ppm":
                    self.axes.invert_xaxis()
                self.axes.legend()
            title = member.label or dataset_display_name(dataset)
            if processed is not None and mode == "Overlay" and not same_axes:
                title += " — different domains; choose Committed to inspect input"
            elif processed is None and mode == "Preview":
                title += " — no ready preview; showing committed data"
            self.axes.set_title(title, fontsize=9)
        if self._selected_operation() == "reference" and self.axes.get_xlabel() == "ppm":
            from matplotlib.widgets import SpanSelector
            self._reference_span = SpanSelector(self.axes, self._reference_region_selected, "horizontal",
                useblit=False, props={"facecolor": "gold", "alpha": .25}, interactive=True, button=1)
            try:
                low = float(self._fields["window_low_ppm"].get())
                high = float(self._fields["window_high_ppm"].get())
                self._reference_span.extents = (low, high)
            except (KeyError, ValueError):
                pass
        self.canvas.draw_idle()

    def _reference_region_selected(self, first, second):
        if self._selected_operation() == "reference" and np.isfinite(first) and np.isfinite(second) and first != second:
            low, high = sorted((float(first), float(second)))
            self._fields["window_low_ppm"].set(f"{low:.12g}")
            self._fields["window_high_ppm"].set(f"{high:.12g}")
            self.status.set("Reference search region selected. Choose Set common or Set for selected, then Preview.")

    def _apply(self):
        preview = self.preview
        if preview is None:
            return
        count = sum(r.status == "ready" for r in preview.results)
        if not messagebox.askyesno("Commit VT batch", f"Commit reviewed previews to {count} checked experiments?\nFailed/skipped experiments remain unchanged; affected tracks become stale.", parent=self):
            return
        try:
            self._validate_context()
            if preview is not self.preview or set(self.blocked_members()) & set(preview.inputs):
                raise ValueError("Selection/processing preview changed; review batch again")
            results = apply_batch(preview, self.manager.active_series, all_series=self.manager.state.series, confirmed=True)
            self._invalidate()
            refresh_error = None
            try:
                self.on_committed(results)
            except Exception as exc:
                refresh_error = str(exc)
            self._present(results)
            self._plot_preview()
            if refresh_error:
                self.status.set(self.status.get() + f" Committed data retained; display refresh failed: {refresh_error}")
        except (ValueError, RuntimeError) as exc:
            self._invalidate()
            self.status.set(str(exc))

    def _cancel(self):
        self._invalidate()
        self.status.set("Batch preview cancelled; committed datasets unchanged")
        self._plot_preview()

    def _validate_context(self):
        if self.context is None:
            return
        loaded = {d.id:d for d in self.manager.state.datasets}
        if any(loaded.get(m.dataset_id) is not m.dataset for m in self.context.members):
            raise ValueError("Selected experiment was removed/replaced; reopen Batch selected")
        for series, member_id, included in self._source_members:
            member = next((m for m in series.members if m.id == member_id), None)
            if not any(s is series for s in self.manager.state.series) or member is None or member.included != included:
                raise ValueError("Source series membership/inclusion changed; reopen Batch selected")

    def destroy(self):
        self._generation += 1
        if self._section_layout_id is not None:
            self.after_cancel(self._section_layout_id)
        if self._enabled_layout_id is not None:
            self.after_cancel(self._enabled_layout_id)
        if self._header_layout_id is not None:
            self.after_cancel(self._header_layout_id)
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
        span = getattr(self, "_reference_span", None)
        if span is not None:
            span.disconnect_events()
        self.figure.clear()
        super().destroy()
