"""Small vertical-only scrolling host for dense main-window workspaces."""

from __future__ import annotations

import tkinter as tk
import tkinter.ttk as ttk


class VerticalScrolledWorkspace(ttk.Frame):
    """A width-following workspace with an automatically hidden scrollbar.

    The public ``content`` frame is the parent for normal ttk workspace
    widgets.  Mouse-wheel bindings are installed only on non-scrolling child
    controls, so list boxes, text widgets, and similar controls retain their
    native wheel behavior.
    """

    _NATIVE_WHEEL_CLASSES = {
        "Canvas",
        "Entry",
        "Listbox",
        "Scale",
        "Spinbox",
        "Text",
        "TCombobox",
        "TEntry",
        "Treeview",
        "TScale",
        "TSpinbox",
    }

    def __init__(self, parent, *, background: str | None = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        if background is None:
            background = ttk.Style(self).lookup("TFrame", "background")
        self.canvas = tk.Canvas(
            self,
            borderwidth=0,
            highlightthickness=0,
            background=background or "#f5f6fa",
        )
        self.scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview,
        )
        self.canvas.configure(yscrollcommand=self._set_scrollbar)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.content = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )
        self.content.bind("<Configure>", self._content_changed, add="+")
        self.canvas.bind("<Configure>", self._canvas_changed, add="+")
        self.canvas.bind("<MouseWheel>", self._mousewheel, add="+")
        self.after_idle(self._install_wheel_bindings)

    def _content_changed(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.after_idle(self._install_wheel_bindings)

    def _canvas_changed(self, event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)

    def _set_scrollbar(self, first: str, last: str) -> None:
        self.scrollbar.set(first, last)
        if float(first) <= 0.0 and float(last) >= 1.0:
            self.scrollbar.grid_remove()
        else:
            self.scrollbar.grid()

    def _install_wheel_bindings(self) -> None:
        def visit(widget) -> None:
            # Nested workspaces own their own controls and wheel bindings.
            if isinstance(widget, VerticalScrolledWorkspace):
                return
            if (
                widget.winfo_class() not in self._NATIVE_WHEEL_CLASSES
                and getattr(widget, "_pcs_workspace_wheel_owner", None) is not self
            ):
                widget.bind("<MouseWheel>", self._mousewheel, add="+")
                widget._pcs_workspace_wheel_owner = self
            for child in widget.winfo_children():
                visit(child)

        if self.winfo_exists():
            visit(self.content)

    def _mousewheel(self, event):
        if not self.canvas.winfo_exists():
            return None
        delta = int(-event.delta / 120) if event.delta else 0
        if delta:
            self.canvas.yview_scroll(delta, "units")
            return "break"
        return None
