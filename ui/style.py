# ui/style.py

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from ui.theme_palette import base_palette, normalize_theme, theme_mode

from ui.widget_style import (presentation_palette, button_skin, selection_skin,
                             notebook_skin, scrollbar_skin, scale_skin, sash_skin, compact_button_skin, refresh_prose_styles, ui_font_name)


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
    ui_size = _scaled_size(settings.get("font_size_ui", 9), scale, 7)

    table_family = str(settings.get("font_family_table", ui_family) or ui_family).strip()
    table_size = _scaled_size(settings.get("font_size_table", 10), scale, 7)

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


def table_row_height(root, font, minimum=18):
    """Fit the rendered font in a pixel row, without scaling points twice."""
    linespace = tkfont.Font(root=root, font=font).metrics("linespace")
    padding = max(3, round(float(root.tk.call("tk", "scaling")) * 2))
    return max(minimum, linespace + padding)


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

    selected_theme = normalize_theme(variant)
    variant = theme_mode(selected_theme)

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
    colors = base_palette(selected_theme)
    bg = colors["bg"]
    fg = colors["fg"]
    panel_bg = colors["panel_bg"]
    input_bg = colors["input_bg"]
    input_fg = colors["input_fg"]
    input_disabled_bg = colors["input_disabled_bg"]
    tree_bg = colors["tree_bg"]
    tree_fg = colors["tree_fg"]
    tree_odd_bg = colors["tree_odd_bg"]
    tree_sel_bg = colors["tree_sel_bg"]
    tree_sel_fg = colors["tree_sel_fg"]
    heading_bg = colors["heading_bg"]
    scale_trough = colors["scale_trough"]
    scale_active = colors["scale_active"]

    palette = presentation_palette(selected_theme)
    bg = panel_bg = palette["surface"]
    fg = input_fg = tree_fg = palette["text"]
    input_bg = tree_bg = palette["field"]
    input_disabled_bg = palette["disabled"]
    tree_sel_bg, tree_sel_fg = palette["selected"], palette["text"]
    heading_bg = palette["button"]
    root._app_palette = palette
    if not hasattr(root, "_pcs_native_button_layout"):
        root._pcs_native_button_layout = style.layout("TButton")

    try:
        root.configure(bg=bg)
    except Exception:
        pass

    # -------------------------------------------------
    # Global defaults
    # -------------------------------------------------
    style.configure(".", padding=0, font=fonts["ui"])

    style.configure("TFrame", padding=6, background=panel_bg)
    style.configure("TLabelframe", background=panel_bg, borderwidth=1,
                    bordercolor=palette["edge"], lightcolor=panel_bg, darkcolor=panel_bg)
    style.configure("TLabelframe.Label", background=panel_bg, foreground=fg, font=fonts["section"])
    style.configure("TLabel", padding=(2, 1), background=panel_bg, foreground=fg, font=fonts["ui"])

    style.configure("TNotebook", background=panel_bg, borderwidth=1,
                    bordercolor=palette["edge"], lightcolor=panel_bg, darkcolor=panel_bg)
    style.configure(
        "TNotebook.Tab",
        padding=(6, 2),
        background=palette["surface"],
        foreground=fg,
        font=fonts["ui"],
    )
    style.configure("TButton", padding=(6, 2), font=fonts["ui"])

    # -------------------------------------------------
    # PCS Analyzer action hierarchy
    # -------------------------------------------------
    # Fonts and compact metrics stay here; shared faces are installed below.
    style.configure("Primary.TButton", padding=(6, 2), font=fonts["ui_bold"])
    style.configure("Secondary.TButton", padding=(6, 2), font=fonts["ui"])
    style.configure("Utility.TButton", padding=(6, 2), font=fonts["ui"])
    style.configure("Danger.TButton", padding=(6, 2), font=fonts["ui"])

    # Control-panel section headers remain neutral gray rather than adopting an
    # action/accent color.
    header_hover = palette["hover"]
    style.configure(
        "ControlPanel.Section.TFrame",
        background=panel_bg,
        relief="flat",
        borderwidth=0,
    )
    style.configure(
        "ControlPanel.Header.TFrame",
        background=heading_bg,
    )
    style.configure(
        "ControlPanel.Header.TButton",
        padding=(6, 2),
        background=heading_bg,
        foreground=fg,
        font=fonts["ui_bold"],
        anchor="w",
    )
    style.map(
        "ControlPanel.Header.TButton",
        background=[("active", header_hover), ("pressed", header_hover)],
    )

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
            ("disabled", input_disabled_bg),
            ("readonly", input_bg),
        ],
        foreground=[
            ("disabled", palette["disabled_text"]),
            ("readonly", input_fg),
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
        rowheight=table_row_height(root, fonts["table"], fonts["table_rowheight"]),
        font=fonts["table"],
        background=tree_bg,
        fieldbackground=tree_bg,
        foreground=tree_fg,
        borderwidth=0,
        bordercolor=palette["edge"],
        lightcolor=tree_bg,
        darkcolor=tree_bg,
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
        foreground=fg,
        bordercolor=palette["edge"],
        lightcolor=heading_bg,
        darkcolor=heading_bg,
    )
    style.map("Treeview.Heading", background=[("active", palette["hover"])],
              foreground=[("!disabled", fg)])

    # -------------------------------------------------
    # Option database for tk widgets
    # -------------------------------------------------
    root._app_fonts = fonts
    live_ui_font = ui_font_name(root, "ui")
    root.option_add("*Font", live_ui_font)
    root.option_add("*Entry.Font", live_ui_font)
    root.option_add("*Text.Font", fonts["report"])
    root.option_add("*Text.Background", palette["field"])
    root.option_add("*Text.Foreground", palette["text"])
    root.option_add("*Text.InsertBackground", palette["text"])
    root.option_add("*Text.SelectBackground", palette["selected"])
    root.option_add("*Text.SelectForeground", palette["text"])
    root.option_add("*Listbox.Font", live_ui_font)
    root.option_add("*TCombobox*Listbox.Font", live_ui_font)
    root.option_add("*TCombobox*Listbox.Background", input_bg)
    root.option_add("*TCombobox*Listbox.Foreground", fg)
    root.option_add("*TCombobox*Listbox.SelectBackground", palette["selected"])
    root.option_add("*TCombobox*Listbox.SelectForeground", fg)

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
    setattr(root, "_theme_id", selected_theme)
    setattr(root, "_app_fonts", fonts)

    notebook_skin(root, style, palette)
    for name, role in (("TButton", "Secondary"), ("Primary.TButton", "Primary"),
                       ("Secondary.TButton", "Secondary"), ("Utility.TButton", "Utility"),
                       ("Danger.TButton", "Danger")):
        button_skin(root, style, name, palette, role)
    # Disclosure/header controls retain their compact, neutral native geometry.
    compact_button_skin(root, style, "ControlPanel.Header.TButton", palette)
    for kind in ("TCheckbutton", "TRadiobutton"):
        selection_skin(root, style, kind, palette)
    scrollbar_skin(root, style, palette)
    scale_skin(root, style, palette)
    sash_skin(root, style, palette)
    style.configure("TSeparator", background=palette["edge"])
    for orientation in ("Horizontal", "Vertical"):
        style.configure(orientation + ".TSeparator", background=palette["edge"])
        style.configure(orientation + ".TProgressbar",
                        troughcolor=palette["field"], background=palette["accent"],
                        bordercolor=palette["edge"], lightcolor=palette["accent"],
                        darkcolor=palette["accent"], borderwidth=1)
    button_skin(root, style, "TMenubutton", palette)
    for kind in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(kind, bordercolor=palette["edge"], lightcolor=input_bg,
                        darkcolor=input_bg, fieldbackground=input_bg, foreground=fg)
        style.map(kind, bordercolor=[("disabled", palette["edge"]), ("invalid", palette["danger"]), ("focus", palette["accent"])],
                  fieldbackground=[("disabled", palette["disabled"]), ("readonly", input_bg)],
                  foreground=[("disabled", palette["disabled_text"]), ("!disabled", fg)])
    style.configure("Muted.TLabel", foreground=palette["muted"], font=fonts["ui_small"])
    style.configure("Warning.TLabel", foreground=palette["warning"], font=fonts["ui_small"])

    refresh_prose_styles(root, style)
    return style
