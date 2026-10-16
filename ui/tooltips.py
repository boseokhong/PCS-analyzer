"""Shared lightweight Tk tooltip support."""

from __future__ import annotations

import tkinter as tk


def command_tooltip_text(command) -> str:
    """Build concise tooltip text from shared command metadata."""
    parts = [str(command.label or "").strip()]
    description = str(command.description or "").strip()
    if description:
        parts.append(description)
    shortcut = str(command.shortcut or "").strip()
    if shortcut:
        parts.append(f"Shortcut: {shortcut}")
    return "\n".join(part for part in parts if part)


class ToolTip:
    """Delayed plain-text tooltip with safe widget-lifecycle cleanup."""

    def __init__(self, widget, text="", delay=550, wraplength=340):
        self.widget = widget
        self.text = str(text or "")
        self.delay = max(0, int(delay))
        self.wraplength = max(120, int(wraplength))
        self.tipwindow = None
        self.after_id = None
        self._destroyed = False
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")
        widget.bind("<Destroy>", self._on_destroy, add="+")

    def update_text(self, text) -> None:
        self.text = str(text or "")
        if self.tipwindow is not None:
            self.hide()

    def _schedule(self, _event=None):
        if self._destroyed or not self.text:
            return
        self._cancel()
        try:
            self.after_id = self.widget.after(self.delay, self.show)
        except tk.TclError:
            self.after_id = None

    def _cancel(self):
        if self.after_id is not None:
            try:
                self.widget.after_cancel(self.after_id)
            except (tk.TclError, ValueError):
                pass
            self.after_id = None

    def show(self):
        self.after_id = None
        if self._destroyed or self.tipwindow is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            root = self.widget.winfo_toplevel()
            variant = getattr(root, "_theme_variant", "light")
            font = (getattr(root, "_app_fonts", {}) or {}).get(
                "ui_small", ("Segoe UI", 9)
            )
            app_root = self.widget._root()
            palette = getattr(app_root, "_app_palette", {})
            font = (getattr(app_root, "_app_fonts", {}) or {}).get("ui_small", font)
            background = palette.get("field", "#272B31" if variant == "dark" else "#F8FAFC")
            foreground = palette.get("text", "#F2F4F7" if variant == "dark" else "#1C232B")
            tip = tk.Toplevel(self.widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{x}+{y}")
            tk.Label(
                tip,
                text=self.text,
                justify=tk.LEFT,
                background=background,
                foreground=foreground,
                relief=tk.SOLID,
                borderwidth=1,
                wraplength=self.wraplength,
                font=font,
                padx=8,
                pady=6,
            ).pack()
            self.tipwindow = tip
        except tk.TclError:
            self.tipwindow = None

    def hide(self, _event=None):
        self._cancel()
        tip = self.tipwindow
        self.tipwindow = None
        if tip is not None:
            try:
                tip.destroy()
            except tk.TclError:
                pass

    def destroy(self) -> None:
        self._destroyed = True
        self.hide()

    def _on_destroy(self, event) -> None:
        if event.widget is self.widget:
            self.destroy()


__all__ = ["ToolTip", "command_tooltip_text"]
