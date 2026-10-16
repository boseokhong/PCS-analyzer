"""Three-column pane geometry using the PCS Control Panel boundary grammar."""

import tkinter as tk
from tkinter import ttk

from ui.control_panel_style import ControlPanelMetrics, control_panel_palette
from ui.pane_gesture import control_panel_drag_should_toggle, control_panel_release_action
from .presentation import WorkbenchPanedWindow


# Existing Workbench working width and smallest Inspector allocation validated in M4.
LEFT_PANEL_PREFERRED_WIDTH = 280
INSPECTOR_PREFERRED_WIDTH = 410
INSPECTOR_MIN_WIDTH = 320
SPECTRUM_MIN_WIDTH = 240
WORKSPACE_OUTER_X = 6


class WorkbenchWorkspace(WorkbenchPanedWindow):
    """Stable side panes, elastic spectrum, and shared release-to-snap gestures.

    Panel widgets stay alive when removed from the geometry manager. Preferences
    are runtime-only; no scientific/application state or settings writer is owned.
    """

    def __init__(self, master):
        super().__init__(master, orient="horizontal")
        self.metrics = ControlPanelMetrics.from_widget(self)
        self.left_width = max(1, round(LEFT_PANEL_PREFERRED_WIDTH * self.metrics.scale))
        self.inspector_width = INSPECTOR_PREFERRED_WIDTH
        self.collapsed = {"left": False, "right": False}
        self.gutters = {}
        self._pending_layout = None
        self._gesture = None
        self._drag_width = None
        self._hovered = set()
        self.bind("<Configure>", self._queue_layout, add="+")
        self.bind("<Destroy>", self._destroy_layout, add="+")

    def attach_panes(self, left, spectrum, inspector):
        self.side_panes = {"left": left, "right": inspector}
        self.spectrum = spectrum
        self._spectrum_padding = tuple(int(str(value)) for value in
                                       self.tk.splitlist(spectrum.cget("padding")))
        for pane in (left, spectrum, inspector):
            pane.bind("<Configure>", self._queue_layout, add="+")
        for side in self.side_panes:
            gutter = tk.Canvas(self, highlightthickness=0, borderwidth=0,
                               cursor="sb_h_double_arrow", takefocus=False)
            self.gutters[side] = gutter
            gutter.bind("<Enter>", lambda event, s=side: self._hover(s, True))
            gutter.bind("<Leave>", lambda event, s=side: self._hover(s, False))
            gutter.bind("<ButtonPress-1>", lambda event, s=side: self._press(s, event))
            gutter.bind("<B1-Motion>", lambda event, s=side: self._motion(s, event))
            gutter.bind("<ButtonRelease-1>", lambda event, s=side: self._release(s, event))
        # Keep both tools usable even when the application is opened at higher scaling.
        thickness = int(ttk.Style(self).lookup("Sash", "sashthickness") or 6)
        window = self.winfo_toplevel()
        old_min_width, old_min_height = window.minsize()
        window.minsize(max(old_min_width, self.left_width + INSPECTOR_MIN_WIDTH +
                          SPECTRUM_MIN_WIDTH + 2 * thickness + 2 * WORKSPACE_OUTER_X),
                       old_min_height)
        self._queue_layout()

    def _queue_layout(self, _event=None):
        if hasattr(self, "side_panes") and self._pending_layout is None:
            self._pending_layout = self.after_idle(self._layout)

    def _destroy_layout(self, event):
        if event.widget is self and self._pending_layout is not None:
            self.after_cancel(self._pending_layout)
            self._pending_layout = None

    def _inspector_limit(self):
        thickness = int(ttk.Style(self).lookup("Sash", "sashthickness") or 6)
        left = 0 if self.collapsed["left"] else self.left_width + thickness
        return max(INSPECTOR_MIN_WIDTH, self.winfo_width() - left - thickness - SPECTRUM_MIN_WIDTH)

    def _layout(self):
        self._pending_layout = None
        if self.winfo_width() <= 1:
            return
        # At the collapsed edge the full restore gutter sits over Spectrum.
        # Reserve its width, then return to the normal inset on expansion.
        padding = (self._spectrum_padding[0] +
                   (self.metrics.gutter_hit if self.collapsed["left"] else 0),
                   *self._spectrum_padding[1:])
        if tuple(int(str(value)) for value in self.tk.splitlist(self.spectrum.cget("padding"))) != padding:
            self.spectrum.configure(padding=padding)
        thickness = int(ttk.Style(self).lookup("Sash", "sashthickness") or 6)
        if not self.collapsed["left"]:
            if self.sashpos(0) != self.left_width:
                self.sashpos(0, self.left_width)
        if not self.collapsed["right"]:
            width = self.inspector_width if self._drag_width is None else self._drag_width
            width = min(max(INSPECTOR_MIN_WIDTH, width), self._inspector_limit())
            index = len(self.panes()) - 2
            target = self.winfo_width() - width - thickness
            if self.sashpos(index) != target:
                self.sashpos(index, target)
        hit = self.metrics.gutter_hit
        for side, gutter in self.gutters.items():
            if self.collapsed[side]:
                x = 0 if side == "left" else self.winfo_width() - hit
            else:
                index = 0 if side == "left" else len(self.panes()) - 2
                x = self.sashpos(index) + thickness // 2 - hit // 2
            x = max(0, min(self.winfo_width() - hit, x))
            gutter.place(x=x, y=0, width=hit, relheight=1)
            tk.Misc.lift(gutter)
            self._draw_gutter(side)

    def _hover(self, side, hover):
        if hover:
            self._hovered.add(side)
        else:
            self._hovered.discard(side)
        self._draw_gutter(side)

    def _draw_gutter(self, side):
        gutter = self.gutters[side]
        palette = control_panel_palette(self)
        bg = ttk.Style(self).lookup("TFrame", "background") or self._root().cget("background")
        gutter.configure(background=bg)
        gutter.delete("all")
        armed = bool(self._gesture and self._gesture["side"] == side and self._gesture["armed"])
        dragging = bool(self._gesture and self._gesture["side"] == side)
        color = palette["gutter_armed" if armed else "gutter_hover" if side in self._hovered or dragging else "separator"]
        gutter.create_line(self.metrics.gutter_hit // 2, 0,
                           self.metrics.gutter_hit // 2, self.winfo_height(), fill=color, width=1)

    def set_collapsed(self, side, collapsed):
        if self.collapsed[side] == collapsed:
            return
        pane = self.side_panes[side]
        if collapsed:
            # Do not move focus: existing Processing/peak-editor FocusOut semantics
            # must not be triggered by an artificial focus_set on the workspace.
            self.forget(pane)
        else:
            self.insert(0 if side == "left" else "end", pane, weight=0)
        self.collapsed[side] = collapsed
        self._queue_layout()

    def _press(self, side, event):
        width = self.left_width if side == "left" else self.inspector_width
        if not self.collapsed[side]:
            width = self.side_panes[side].winfo_width()
        self._gesture = dict(side=side, start=event.x_root, width=width,
                             collapsed=self.collapsed[side], armed=False)
        self.gutters[side].grab_set()
        self._draw_gutter(side)

    def _motion(self, side, event):
        gesture = self._gesture
        if gesture is None or gesture["side"] != side:
            return
        delta = event.x_root - gesture["start"]
        gesture["armed"] = control_panel_drag_should_toggle(
            delta, gesture["width"], collapsed=gesture["collapsed"], dock_side=side)
        if side == "right" and not gesture["collapsed"]:
            self._drag_width = max(INSPECTOR_MIN_WIDTH, gesture["width"] - delta)
            self._queue_layout()
        self._draw_gutter(side)

    def _release(self, side, event):
        gesture = self._gesture
        if gesture is None or gesture["side"] != side:
            return
        delta = event.x_root - gesture["start"]
        action = control_panel_release_action(
            delta, gesture["width"], collapsed=gesture["collapsed"],
            dock_side=side, scale=self.metrics.scale)
        self.gutters[side].grab_release()
        if action == "toggle":
            self.set_collapsed(side, not gesture["collapsed"])
        elif action == "expand":
            self.set_collapsed(side, False)
        elif side == "right" and not gesture["collapsed"] and abs(delta) > 3 * self.metrics.scale:
            self.inspector_width = min(max(INSPECTOR_MIN_WIDTH, gesture["width"] - delta),
                                       self._inspector_limit())
        self._gesture = None
        self._drag_width = None
        self._queue_layout()
        self._draw_gutter(side)
