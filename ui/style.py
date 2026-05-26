# ui/style.py

import tkinter as tk
from tkinter import ttk


# Default:
#   False -> use plain ttk/clam theme for stable IDE/exe appearance.
#
# Optional legacy mode:
#   True  -> use ttkbootstrap themes again.
#           This may change widget spacing/layout between themes.
USE_TTKBOOTSTRAP = False


def _scaled_size(value, scale=1.0, minimum=6, maximum=72):
    """Return a safe integer font size after applying the global font scale."""
    try:
        size = int(round(float(value) * float(scale)))
    except Exception:
        size = int(minimum)
    return max(int(minimum), min(int(maximum), size))


def build_fonts(settings=None):
    """
    Build all application font roles from app_settings.

    Only a small set of user-facing settings is stored:
      - UI font family/size
      - Table font family/size
      - Report font family/size
      - Plot font family/size
      - Global font scale

    Other roles are derived here so individual UI files do not hard-code
    platform-specific font tuples such as ("Segoe UI", 9) or ("Consolas", 9).
    """
    settings = settings or {}

    try:
        scale = float(settings.get("font_scale", 1.0) or 1.0)
    except Exception:
        scale = 1.0
    scale = max(0.5, min(2.5, scale))

    ui_family = str(settings.get("font_family_ui", "Segoe UI") or "Segoe UI").strip()
    ui_size = _scaled_size(settings.get("font_size_ui", 10), scale, 7)

    table_family = str(settings.get("font_family_table", ui_family) or ui_family).strip()
    table_size = _scaled_size(settings.get("font_size_table", ui_size), scale, 7)

    report_family = str(settings.get("font_family_report", "Consolas") or "Consolas").strip()
    report_size = _scaled_size(settings.get("font_size_report", 9), scale, 7)

    plot_family = str(settings.get("font_family_plot", "DejaVu Sans") or "DejaVu Sans").strip()
    plot_size = _scaled_size(settings.get("font_size_plot", 9), scale, 6)

    return {
        # Tk / ttk UI fonts
        "ui": (ui_family, ui_size),
        "ui_bold": (ui_family, ui_size, "bold"),
        "ui_small": (ui_family, max(ui_size - 1, 7)),
        "ui_small_bold": (ui_family, max(ui_size - 1, 7), "bold"),

        # Structural labels / headers
        "section": (ui_family, ui_size, "bold"),
        "section_large": (ui_family, ui_size + 1, "bold"),
        "title": (ui_family, ui_size + 2, "bold"),
        "link": (ui_family, max(ui_size - 1, 7), "underline"),

        # Treeview / tables
        "table": (table_family, table_size),
        "table_heading": (table_family, table_size, "bold"),
        "table_rowheight": max(18, int(round(table_size * 2.0))),

        # Report/log/result text boxes
        "report": (report_family, report_size),
        "report_small": (report_family, max(report_size - 1, 7)),

        # Matplotlib / PyVista text sizes
        "plot_family": plot_family,
        "plot_base": plot_size,
        "plot_title": plot_size + 1,
        "plot_label": plot_size,
        "plot_tick": max(plot_size - 1, 6),
        "plot_legend": max(plot_size - 2, 6),
        "plot_annotation": max(plot_size - 2, 6),
        "viewer_label_size": plot_size + 1,
    }


def get_app_fonts(state_or_root=None):
    """
    Return the current font-role dictionary from a state dict or Tk root.
    Falls back to default fonts if no runtime font dictionary exists.
    """
    if isinstance(state_or_root, dict):
        fonts = state_or_root.get("fonts")
        if isinstance(fonts, dict):
            return fonts
        root = state_or_root.get("root")
        if root is not None:
            fonts = getattr(root, "_app_fonts", None)
            if isinstance(fonts, dict):
                return fonts
        return build_fonts(state_or_root.get("app_settings", {}))

    if state_or_root is not None:
        fonts = getattr(state_or_root, "_app_fonts", None)
        if isinstance(fonts, dict):
            return fonts

        # Tk child widgets do not carry _app_fonts themselves. Try their root.
        try:
            root = state_or_root._root()
            fonts = getattr(root, "_app_fonts", None)
            if isinstance(fonts, dict):
                return fonts
        except Exception:
            pass

    return build_fonts({})


def apply_style(root, variant="light", accent=None, settings=None):
    """
    Apply application-wide ttk/tk styling.

    variant:
        "light" or "dark"

    accent:
        Kept for backward compatibility with app_settings.
        In the default plain ttk mode, accent is intentionally ignored.

    settings:
        Optional app_settings dictionary. Font-related keys are read from here.
    """

    variant = str(variant or "light").lower().strip()
    if variant not in ("light", "dark"):
        variant = "light"

    accent = str(accent or "blue").lower().strip()
    fonts = build_fonts(settings or {})
    base_font = fonts["ui"]

    # -------------------------------------------------
    # Theme engine
    # -------------------------------------------------
    if USE_TTKBOOTSTRAP:
        # -------------------------------------------------
        # Legacy ttkbootstrap mode
        # -------------------------------------------------
        # This block restores the old ttkbootstrap behavior.
        # Note: accent changes the whole ttkbootstrap theme, not only color.
        # Therefore, widget spacing/layout can change between accent values.
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

            style = tb.Style()
            try:
                style.theme_use(theme)
            except Exception:
                style = tb.Style(theme=theme)

        except Exception:
            style = ttk.Style(root)
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass

    else:
        # -------------------------------------------------
        # Stable plain ttk mode
        # -------------------------------------------------
        # This keeps the appearance closer between IDE and PyInstaller exe.
        # Accent is ignored in this mode.
        style = ttk.Style(root)
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

        heading_bg = "#E4E2DC"

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

        heading_bg = "#D8D5CC"

        scale_trough = "#2B2F36"
        scale_active = "#A0A0A0"

    try:
        root.configure(bg=bg)
    except Exception:
        pass

    # -------------------------------------------------
    # Global defaults
    # -------------------------------------------------
    style.configure(".", padding=0, font=fonts["ui"])

    style.configure("TFrame", padding=6, background=panel_bg)
    style.configure("TLabelframe", background=panel_bg)
    style.configure("TLabelframe.Label", background=panel_bg, foreground=fg, font=fonts["section"])
    style.configure("TLabel", padding=(2, 1), background=panel_bg, foreground=fg, font=fonts["ui"])

    style.configure("TNotebook", background=panel_bg)
    style.configure(
        "TNotebook.Tab",
        padding=(6, 2),
        background=heading_bg if variant == "light" else "#D8D5CC",
        foreground="#111111" if variant == "light" else "#111111",
        font=fonts["ui"],
    )
    style.map(
        "TNotebook.Tab",
        background=[
            ("selected", "#FFFFFF" if variant == "light" else "#2B2F36"),
            ("active", "#FFFFFF" if variant == "light" else "#3A3F46"),
        ],
        foreground=[
            ("selected", "#111111" if variant == "light" else "#FFFFFF"),
            ("active", "#111111" if variant == "light" else "#FFFFFF"),
        ],
        expand=[("selected", [1, 1, 1, 0])],
    )

    style.configure("TButton", padding=(6, 2), font=fonts["ui"])

    # -------------------------------------------------
    # Entry / Combobox
    # -------------------------------------------------
    style.configure(
        "TEntry",
        padding=(4, 1),
        fieldbackground=input_bg,
        background=input_bg,
        foreground=input_fg,
        insertcolor=input_fg,
        font=fonts["ui"],
    )
    style.map(
        "TEntry",
        fieldbackground=[
            ("disabled", input_disabled_bg),
            ("readonly", input_bg),
        ],
        foreground=[
            ("disabled", "#888888" if variant == "light" else "#BBBBBB"),
            ("readonly", input_fg),
        ],
    )

    style.configure(
        "TCombobox",
        padding=(4, 1),
        fieldbackground=input_bg,
        background=input_bg,
        foreground=input_fg,
        arrowcolor=fg,
        font=fonts["ui"],
    )
    style.map(
        "TCombobox",
        fieldbackground=[
            ("readonly", input_bg),
            ("disabled", input_disabled_bg),
        ],
        foreground=[
            ("readonly", input_fg),
            ("disabled", "#888888" if variant == "light" else "#BBBBBB"),
        ],
    )

    # -------------------------------------------------
    # Check / Radio
    # -------------------------------------------------
    disabled_fg = "#8A8F98" if variant == "light" else "#7C828A"
    disabled_bg = panel_bg

    style.configure("TCheckbutton", background=panel_bg, foreground=fg, font=fonts["ui"])
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

    style.configure("TRadiobutton", background=panel_bg, foreground=fg, font=fonts["ui"])
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
        rowheight=fonts["table_rowheight"],
        font=fonts["table"],
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
        font=fonts["table_heading"],
        background=heading_bg,
        foreground="#111111",
    )

    # -------------------------------------------------
    # Scrollbar
    # -------------------------------------------------
    try:
        style.configure("Vertical.TScrollbar", arrowsize=10, width=12)
        style.configure("Horizontal.TScrollbar", arrowsize=10, width=12)
    except tk.TclError:
        pass

    # -------------------------------------------------
    # Option database for tk widgets
    # -------------------------------------------------
    root.option_add("*Font", base_font)
    root.option_add("*Entry.Font", base_font)
    root.option_add("*Text.Font", fonts["report"])
    root.option_add("*Listbox.Font", fonts["ui"])
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
    setattr(root, "_theme_variant", variant)
    setattr(root, "_app_fonts", fonts)

    return style
