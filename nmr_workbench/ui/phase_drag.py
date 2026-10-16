"""Relative vertical angle editing; no scientific processing or bounded scale."""
import math
from tkinter import ttk
from ui.tooltips import ToolTip
from .presentation import size_workbench_action
from .actions import _image


class PhaseDrag(ttk.Button):
    def __init__(self, parent, *, read_value, change_value, report_error):
        image = _image(parent, "phase_drag_vertical")
        super().__init__(parent, text="" if image else "Drag", image=image,
                         width=0, cursor="sb_v_double_arrow",
                         style="NMR.Phase.Secondary.TButton", takefocus=True)
        self._drag_image = image
        self.read_value, self.change_value, self.report_error = read_value, change_value, report_error
        self._last_y = self._start_value = None
        size_workbench_action(self)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Escape>", self._escape)
        self.bind("<Up>", lambda e: self._step(e, 1))
        self.bind("<Down>", lambda e: self._step(e, -1))
        self._tip = ToolTip(self, "Drag up to increase, down to decrease (0.5°/pixel). Shift: fine adjustment (0.05°/pixel). Arrow keys also adjust. Escape cancels this drag.")

    def _value(self):
        value = float(self.read_value())
        if not math.isfinite(value):
            raise ValueError("Phase angle must be finite.")
        return value

    def _press(self, event):
        if self.instate(["disabled"]):
            return "break"
        try:
            self._start_value = self._value()
        except ValueError as exc:
            self.report_error(exc)
            return "break"
        self.focus_set()
        self._tip.hide()
        self._last_y = event.y_root
        self.grab_set()
        self.state(["pressed"])
        return "break"

    def _motion(self, event):
        if self._last_y is None:
            return "break"
        if self.instate(["disabled"]):
            self.cancel_drag()
            return "break"
        delta = self._last_y - event.y_root
        self._last_y = event.y_root
        if delta:
            self._adjust(delta * (0.05 if event.state & 1 else 0.5))
        return "break"

    def _adjust(self, delta):
        try:
            value = self._value() + delta
            if not math.isfinite(value):
                raise ValueError("Phase angle must be finite.")
            self.change_value(value)
        except ValueError as exc:
            self.cancel_drag()
            self.report_error(exc)

    def _step(self, event, direction):
        if not self.instate(["disabled"]):
            self._adjust(direction * (0.1 if event.state & 1 else 1.0))
        return "break"

    def _release(self, event):
        self.cancel_drag()
        return "break"

    def _escape(self, event):
        value = self._start_value
        self.cancel_drag()
        if value is not None and not self.instate(["disabled"]):
            self.change_value(value)
        return "break"

    def cancel_drag(self):
        self._last_y = self._start_value = None
        self.state(["!pressed"])
        if self.grab_current() is self:
            self.grab_release()
