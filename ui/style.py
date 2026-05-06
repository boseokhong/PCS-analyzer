# ui/style.py

import tkinter as tk
from tkinter import ttk

def apply_style(root, variant="light", accent="blue"):
    used_bootstrap = False
    try:
        import ttkbootstrap as tb
        theme = {
            ("light", "blue"): "flatly",
            ("light", "green"): "minty",
            ("light", "orange"): "journal",
            ("light", "purple"): "yeti",
            ("dark",  "blue"): "cyborg",
            ("dark",  "green"): "darkly",
            ("dark",  "orange"): "superhero",
            ("dark",  "purple"): "solar",
        }.get((variant, accent), "flatly" if variant == "light" else "darkly")
        style = tb.Style(theme)
        used_bootstrap = True
    except Exception:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

    # -------------------------------------------------
    # Base colors
    # -------------------------------------------------
    if variant == "light":
        bg = "#F5F6FA"
        fg = "#111111"

        panel_bg = bg
        input_bg = "#FFFFFF"
        input_fg = "#111111"
        input_disabled_bg = "#ECEEF2"

        tree_bg = "#FFFFFF"
        tree_fg = "#111111"
        tree_odd_bg = "#EEF1F6"
        tree_sel_bg = "#AFCBF3"
        tree_sel_fg = "#111111"

        scale_trough = "#FFFFFF"
        scale_active = "#808080"
    else:
        bg = "#1F2125"
        fg = "#EEEEEE"

        panel_bg = bg
        input_bg = "#2B2F36"
        input_fg = "#EEEEEE"
        input_disabled_bg = "#3A3F46"

        tree_bg = "#202428"
        tree_fg = "#EEEEEE"
        tree_odd_bg = "#25292E"
        tree_sel_bg = "#58779B"
        tree_sel_fg = "#FFFFFF"

        scale_trough = "#2B2F36"
        scale_active = "#A0A0A0"

    if used_bootstrap:
        try:
            root.configure(bg=bg)
        except Exception:
            pass
    else:
        root.configure(bg=bg)

    base_font = ("Segoe UI", 10)

    # -------------------------------------------------
    # Global defaults
    # -------------------------------------------------
    style.configure(".", padding=0)

    style.configure("TFrame", padding=6, background=panel_bg)
    style.configure("TLabelframe", background=panel_bg)
    style.configure("TLabelframe.Label", background=panel_bg, foreground=fg)
    style.configure("TLabel", padding=(2, 1), background=panel_bg, foreground=fg)

    style.configure("TNotebook", background=panel_bg)
    style.configure("TNotebook.Tab", padding=(6, 2), background=panel_bg, foreground=fg)
    style.map("TNotebook.Tab", expand=[("selected", [1, 1, 1, 0])])

    style.configure("TButton", padding=(6, 2))

    # -------------------------------------------------
    # Entry / Combobox
    # -------------------------------------------------
    style.configure(
        "TEntry",
        padding=(4, 1),
        fieldbackground=input_bg,
        background=input_bg,
        foreground=input_fg,
    )
    style.map(
        "TEntry",
        fieldbackground=[("disabled", input_disabled_bg)],
        foreground=[("disabled", "#888888" if variant == "light" else "#BBBBBB")],
    )

    style.configure(
        "TCombobox",
        padding=(4, 1),
        fieldbackground=input_bg,
        background=input_bg,
        foreground=input_fg,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", input_bg), ("disabled", input_disabled_bg)],
        foreground=[("readonly", input_fg), ("disabled", "#888888" if variant == "light" else "#BBBBBB")],
    )

    # -------------------------------------------------
    # Check / Radio
    # -------------------------------------------------
    disabled_fg = "#8A8F98" if variant == "light" else "#7C828A"
    disabled_bg = panel_bg

    style.configure("TCheckbutton", background=panel_bg, foreground=fg)
    style.map(
        "TCheckbutton",
        background=[
            ("disabled", disabled_bg),
            ("active", panel_bg),
            ("selected", panel_bg),
        ],
        foreground=[
            ("disabled", disabled_fg),
            ("active", fg),
            ("selected", fg),
        ],
    )

    style.configure("TRadiobutton", background=panel_bg, foreground=fg)
    style.map(
        "TRadiobutton",
        background=[
            ("disabled", disabled_bg),
            ("active", panel_bg),
            ("selected", panel_bg),
        ],
        foreground=[
            ("disabled", disabled_fg),
            ("active", fg),
            ("selected", fg),
        ],
    )

    # -------------------------------------------------
    # Treeview
    # -------------------------------------------------
    style.configure(
        "Treeview",
        rowheight=18,
        font=base_font,
        background=tree_bg,
        fieldbackground=tree_bg,
        foreground=tree_fg,
        borderwidth=0,
    )
    style.map(
        "Treeview",
        background=[("selected", tree_sel_bg)],
        foreground=[("selected", tree_sel_fg)],
    )
    style.configure(
        "Treeview.Heading",
        padding=(2, 1),
        font=(base_font[0], base_font[1], "bold"),
    )

    # -------------------------------------------------
    # Scrollbar
    # -------------------------------------------------
    try:
        style.configure("Vertical.TScrollbar", arrowsize=10)
        style.configure("Horizontal.TScrollbar", arrowsize=10)
    except tk.TclError:
        pass

    # -------------------------------------------------
    # Option database for tk widgets
    # -------------------------------------------------
    root.option_add("*Font", base_font)
    root.option_add("*Entry.Font", base_font)
    root.option_add("*TCombobox*Listbox.Font", base_font)

    root.option_add("*Scale.troughColor", scale_trough)
    root.option_add("*Scale.activeBackground", scale_active)
    root.option_add("*Scale.highlightThickness", 0)
    root.option_add("*Scale.sliderRelief", "raised")

    # -------------------------------------------------
    # Treeview stripe helper
    # -------------------------------------------------
    def stripe_treeview(tv: ttk.Treeview):
        tv.tag_configure("oddrow", background=tree_odd_bg)

        for i, item in enumerate(tv.get_children("")):
            current_tags = list(tv.item(item, "tags") or [])
            current_tags = [t for t in current_tags if t != "oddrow"]
            if i % 2:
                current_tags.append("oddrow")
            tv.item(item, tags=tuple(current_tags))

    setattr(root, "_stripe_treeview", stripe_treeview)
    setattr(root, "_app_bg", bg)
    setattr(root, "_app_fg", fg)

    return style