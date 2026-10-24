"""Content-sized Tk dialogs with a screen-limited scrolling form."""

import sys
from tkinter import ttk

from ui.scrollable_workspace import VerticalScrolledWorkspace
from ui.widget_style import ui_native_colors


def dialog_work_area(window):
    """Return the parent's monitor working area (taskbar excluded on Windows)."""
    anchor = (window.master or window).winfo_toplevel()
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class MonitorInfo(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        monitor_at = user32.MonitorFromPoint
        monitor_at.argtypes = [wintypes.POINT, wintypes.DWORD]
        monitor_at.restype = wintypes.HANDLE
        get_info = user32.GetMonitorInfoW
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(MonitorInfo)]
        get_info.restype = wintypes.BOOL
        point = wintypes.POINT(anchor.winfo_rootx(), anchor.winfo_rooty())
        info = MonitorInfo()
        info.cbSize = ctypes.sizeof(info)
        if get_info(monitor_at(point, 2), ctypes.byref(info)):
            rect = info.rcWork
            return rect.left, rect.top, rect.right, rect.bottom
    left, top = anchor.winfo_vrootx(), anchor.winfo_vrooty()
    return left, top, left + anchor.winfo_vrootwidth(), top + anchor.winfo_vrootheight()


def fit_dialog(window, *, form=None, max_height_fraction=1.0):
    """Fit once at opening; subsequent user resizing remains under user control."""
    # Measure mapped widget geometry invisibly. This avoids showing the interim
    # size-only geometry at the window manager's default (often top-left) location.
    opacity = window.attributes("-alpha")
    window.attributes("-alpha", 0)
    try:
        window.update_idletasks()
        if form is not None:
            # A Canvas otherwise requests its default size, irrespective of its form.
            form.canvas.configure(width=form.content.winfo_reqwidth(),
                                  height=form.content.winfo_reqheight())
            window.update_idletasks()
        left, top, right, bottom = dialog_work_area(window)
        # Reserve room for the native title bar and window borders as well as a gap.
        gap = max(12, round(float(window.tk.call("tk", "scaling")) * 10))
        available_width = max(1, right - left - 2 * gap)
        available_height = max(1, bottom - top - 4 * gap)
        available_height = min(available_height, max(1, round((bottom - top) * max_height_fraction)))
        width = min(window.winfo_reqwidth(), available_width)
        height = min(window.winfo_reqheight(), available_height)
        anchor = (window.master or window).winfo_toplevel()
        if form is not None:
            form.enable_wrapping()
        # A capped height can reveal a scrollbar, which itself needs horizontal room.
        # Settle both dimensions before establishing the final minimum and position.
        for _ in range(3):
            window.geometry(f"{width}x{height}")
            window.update_idletasks()
            if form is not None:
                form.wrap_content()
                window.update_idletasks()
                natural_height = (form.content.winfo_reqheight()
                                  + window.winfo_reqheight() - form.canvas.winfo_reqheight())
                height = min(natural_height, available_height)
                width = min(window.winfo_reqwidth(), available_width)
        window.minsize(width, min(height, max(120, available_height // 3)))
        if anchor.winfo_viewable():
            center_x = anchor.winfo_rootx() + anchor.winfo_width() // 2
            center_y = anchor.winfo_rooty() + anchor.winfo_height() // 2
        else:
            center_x, center_y = (left + right) // 2, (top + bottom) // 2
        x = max(left + gap, min(center_x - width // 2,
                               right - width - gap))
        y = max(top + 2 * gap, min(center_y - height // 2,
                                 bottom - height - 2 * gap))
        window.geometry(f"{width}x{height}+{x}+{y}")
        # Commit the final position before revealing the window, not on a later event.
        window.update_idletasks()
    finally:
        if window.winfo_exists():
            window.attributes("-alpha", opacity)


class DialogForm(VerticalScrolledWorkspace):
    """Natural-sized form; actions belong in a sibling footer outside its scroll."""

    def __init__(self, parent, *, padding=12):
        super().__init__(parent, padding=0)
        ui_native_colors(self.canvas, background="surface")
        self.content.configure(padding=padding)

    def enable_wrapping(self):
        """Let full-width explanatory labels follow the viewport after fitting."""
        self.content.bind("<Configure>", self.wrap_content, add="+")

    def wrap_content(self, _event=None):
        def visit(parent):
            # Tk versions may return a Tcl list string, Python values or Tcl_Obj
            # distance values. Let Tk parse distances (including units) as pixels.
            padding = parent.tk.splitlist(parent.cget("padding"))
            inset = (parent.winfo_pixels(str(padding[0]))
                     + parent.winfo_pixels(str(padding[2] if len(padding) > 2 else padding[0]))) if padding else 0
            width = max(1, parent.winfo_width() - inset)
            for child in parent.winfo_children():
                if isinstance(child, ttk.Frame):
                    visit(child)
                elif (isinstance(child, ttk.Label) and child.winfo_manager() == "grid"
                        and int(child.grid_info().get("columnspan", 1)) > 1
                        and child.winfo_pixels(child.cget("wraplength") or 0) > 0):
                    if child.winfo_pixels(child.cget("wraplength")) != width:
                        child.configure(wraplength=width)
        visit(self.content)


def dialog_actions(parent):
    """A fixed footer matching the form's outer horizontal inset."""
    return ttk.Frame(parent, padding=(12, 6, 12, 12))
