"""Shared residual editor with per-view drafts and document-owned accepted values."""
import tkinter as tk
from tkinter import ttk

from logic.residual_display import mark_display_modified, retag_residual_tables, validate_thresholds
from ui.tooltips import ToolTip
from ui.widget_style import ui_label_style


class ResidualControls(ttk.Frame):
    def __init__(self, master, state):
        super().__init__(master, padding=0)
        self.owner = state
        self.variables = [tk.StringVar(master=self), tk.StringVar(master=self)]
        self.entries = []
        self.values = []
        self.color_button = ttk.Checkbutton(
            self, text="Color", variable=state["residual_color_enabled_var"], command=self.toggle_color
        )
        self.color_button.grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Separator(self, orient="vertical").grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=3)
        self.caption = ttk.Label(self, text="Residual:")
        self.caption.grid(row=0, column=2, padx=(0, 4))
        self.legend_labels = [self.caption]
        self.legend = ttk.Frame(self, padding=0)
        self.legend.grid(row=0, column=3, sticky="e")
        self.groups = []
        for index, (color, variable) in enumerate(zip(("#2ca02c", "#ff7f0e"), self.variables)):
            group = ttk.Frame(self.legend, padding=0)
            group.grid(row=0, column=index, padx=(0, 10))
            dot = ttk.Label(group, text="●", foreground=color)
            dot.grid(row=0, column=0)
            comparison = ttk.Label(group, text="≤")
            comparison.grid(row=0, column=1, padx=(2, 2))
            self.legend_labels.append(comparison)
            slot = ttk.Frame(group, padding=0)
            slot.grid(row=0, column=2)
            entry = ttk.Entry(slot, textvariable=variable, width=4, justify="right")
            entry.grid(row=0, column=0, sticky="nsew")
            value = ttk.Label(slot, textvariable=variable, anchor="w", padding=0,
                              style=ui_label_style(self, "muted"))
            value.grid(row=0, column=0, sticky="nsew")
            self.values.append(value)
            self.groups.append((dot, comparison, entry, value))
            entry.bind("<Return>", self.commit)
            entry.bind("<KP_Enter>", self.commit)
            entry.bind("<FocusOut>", self.commit)
            entry.bind("<Escape>", lambda event: self.refresh())
            entry._residual_tooltip = ToolTip(entry, "Absolute residual cutoff (ppm); 0 ≤ green < orange.")
            self.entries.append(entry)
        group = ttk.Frame(self.legend, padding=0)
        group.grid(row=0, column=2)
        dot = ttk.Label(group, text="●", foreground="#d62728")
        dot.grid(row=0, column=0)
        comparison = ttk.Label(group, text=">")
        comparison.grid(row=0, column=1, padx=(2, 2))
        self.legend_labels.append(comparison)
        self.red_label = tk.StringVar(master=self)
        red = ttk.Label(group, textvariable=self.red_label, anchor="w", padding=0)
        red.grid(row=0, column=2)
        self.groups.append((dot, comparison, red))
        self.legend_labels.append(red)
        self._reserved_geometry = None
        self.bind("<Configure>", self.reserve_geometry, add="+")
        state.setdefault("_residual_controls", []).append(self)
        state["refresh_residual_controls"] = lambda: refresh_residual_controls(state)
        self.refresh()

    def refresh(self):
        for variable, key, default in zip(self.variables,
                                         ("residual_thr_ok", "residual_thr_warn"), (0.1, 0.3)):
            variable.set(str(float(self.owner.get(key, default))))
        self.red_label.set(f"{float(self.owner.get('residual_thr_warn', 0.3))} ppm")
        for entry in self.entries:
            entry.state(["!invalid"])
            entry._residual_tooltip.update_text("Absolute residual cutoff (ppm); 0 ≤ green < orange.")
        self.refresh_enabled()

    def refresh_enabled(self):
        enabled = bool(self.owner["residual_color_enabled_var"].get())
        for entry, value in zip(self.entries, self.values):
            entry.configure(takefocus=enabled)
            entry.state(["!disabled"] if enabled else ["disabled"])
            (value if enabled else entry).grid_remove()
            (entry if enabled else value).grid()
        for label in self.legend_labels:
            label.configure(style="TLabel" if enabled else ui_label_style(self, "muted"))
        self.reserve_geometry()

    def reserve_geometry(self, _event=None):
        # Reserve surplus width before the right-aligned legend, never between
        # a sign and its value. Keep ppm's right edge fixed on Color toggles.
        width = 20  # two equal inter-group gaps
        height = self.color_button.winfo_reqheight()
        for dot, comparison, *values in self.groups:
            width += dot.winfo_reqwidth() + comparison.winfo_reqwidth() + 4
            width += max(widget.winfo_reqwidth() for widget in values)
            height = max(height, *(widget.winfo_reqheight() for widget in values))
        geometry = width, height
        if geometry != self._reserved_geometry:
            self._reserved_geometry = geometry
            self.columnconfigure(3, minsize=width)
            self.rowconfigure(0, minsize=height)

    def commit(self, _event=None):
        if self.owner.get("_pcs_document_tracking_suppressed"):
            return
        if not self.owner["residual_color_enabled_var"].get():
            self.refresh()
            return
        try:
            ok, warn = validate_thresholds(*(variable.get() for variable in self.variables))
        except ValueError as exc:
            for entry in self.entries:
                entry.state(["invalid"])
                entry._residual_tooltip.update_text(str(exc))
            return "break"
        changed = (ok, warn) != (self.owner.get("residual_thr_ok", 0.1), self.owner.get("residual_thr_warn", 0.3))
        self.owner["residual_thr_ok"], self.owner["residual_thr_warn"] = ok, warn
        refresh_residual_controls(self.owner)
        if changed:
            retag_residual_tables(self.owner)
            mark_display_modified(self.owner)
        return "break" if _event is not None and _event.type == tk.EventType.KeyPress else None

    def toggle_color(self):
        refresh_residual_controls(self.owner)
        retag_residual_tables(self.owner)
        mark_display_modified(self.owner)


def refresh_residual_controls(state):
    for control in state.get("_residual_controls", ()):
        if control.winfo_exists():
            control.refresh()


def install_rhombicity_control_layout(state, host, inputs, rotation, summary, status):
    """Reflow stable controls; the rotation/summary choice remains owned by orientation."""
    host.columnconfigure(2, weight=1)
    first_separator = ttk.Separator(host, orient="vertical")
    second_separator = ttk.Separator(host, orient="vertical")
    row_separator = ttk.Separator(host, orient="horizontal")
    state["rh_control_separators"] = (first_separator, second_separator, row_separator)
    last = [None]

    def layout(_event=None):
        orientation = state.get("orientation_mode_var")
        tensor = orientation is not None and orientation.get() == "tensor"
        middle = summary if tensor else rotation
        unused = rotation if tensor else summary
        unused.grid_remove()
        required = inputs.winfo_reqwidth() + status.winfo_reqwidth() + middle.winfo_reqwidth() + 40
        width = host.winfo_width()
        mode = "wide" if width >= required else "two" if width >= inputs.winfo_reqwidth() + status.winfo_reqwidth() + 20 else "three"
        current = (mode, tensor)
        if last[0] == current:
            return
        last[0] = current
        inputs.grid(row=0, column=0, sticky="w")
        for separator in (first_separator, second_separator, row_separator):
            separator.grid_remove()
        if mode == "wide":
            first_separator.grid(row=0, column=1, sticky="ns", padx=10, pady=3)
            middle.grid(row=0, column=2, columnspan=1, sticky="ew", padx=0, pady=0)
            second_separator.grid(row=0, column=3, sticky="ns", padx=10, pady=3)
            status.grid(row=0, column=4, columnspan=1, sticky="e", pady=0)
        elif mode == "two":
            first_separator.grid(row=0, column=1, sticky="ns", padx=8, pady=3)
            status.grid(row=0, column=2, columnspan=3, sticky="e", pady=0)
            row_separator.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(5, 0))
            middle.grid(row=2, column=0, columnspan=5, sticky="ew", padx=0, pady=(5, 0))
        else:
            middle.grid(row=1, column=0, columnspan=5, sticky="ew", padx=0, pady=(5, 0))
            row_separator.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(5, 0))
            status.grid(row=3, column=0, columnspan=5, sticky="e", pady=(5, 0))
    host.bind("<Configure>", layout, add="+")
    state["layout_rhombicity_controls"] = layout
    host.after_idle(layout)
