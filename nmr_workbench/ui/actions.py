"""Small presentation-only action widgets using the host's packaged icons."""
import tkinter as tk
from tkinter import ttk

from ui.icons import IconManager, TOOLBAR_ICON_SIZE
from ui.ribbon.icon_service import nearest_packaged_icon_size, tk_scale_factor
from ui.tooltips import ToolTip
from .presentation import size_workbench_action


def _image(parent, icon):
    root = parent._root()
    manager = getattr(root, "_icon_manager", None)
    if manager is None:
        manager = IconManager(root)
    size = nearest_packaged_icon_size(TOOLBAR_ICON_SIZE, tk_scale_factor(root))
    return manager.get(icon, size=size)


def _describe(widget, label, description):
    widget._action_label = label
    tip = ToolTip(widget, description or label)
    widget._action_tooltip = tip
    # Icon controls remain discoverable when navigating with Tab.
    widget.bind("<FocusIn>", lambda event=None: tip.show(), add="+")
    widget.bind("<FocusOut>", tip.hide, add="+")


def action(parent, label, icon, command, *, primary=False,
           description=None, state="normal"):
    image = _image(parent, icon)
    widget = ttk.Button(parent, text=label,
        image=image, compound="left", command=command, state=state, takefocus=True,
        style="NMR.Primary.TButton" if primary else "NMR.Toolbar.TButton",
        padding=(5, 1))
    size_workbench_action(widget)
    widget._action_image = image  # retain even when the host refreshes its cache
    _describe(widget, label, description)
    return widget


def action_menu(parent, label, icon, entries, *, description=None):
    """Entries are (label, callback), with None for a separator; no state ownership."""
    image = _image(parent, icon)
    widget = ttk.Menubutton(parent, text=label, image=image, compound="left",
                           takefocus=True, padding=(5, 1), style="NMR.Toolbar.TMenubutton")
    size_workbench_action(widget)
    widget._action_image = image
    menu = tk.Menu(widget, tearoff=False)
    for entry in entries:
        if entry is None:
            menu.add_separator()
        else:
            text, command = entry
            menu.add_command(label=text, command=command)
    widget.configure(menu=menu)
    widget._action_menu = menu
    _describe(widget, label, description)
    return widget


def divider(parent):
    ttk.Separator(parent, orient="vertical").pack(side="left", fill="y", padx=6, pady=2)


def polish(window):
    """Apply the existing palette to a newly built auxiliary window."""
    from .presentation import WorkbenchPresentation
    window._presentation = WorkbenchPresentation(window)
    window._presentation.refresh(force=True)
