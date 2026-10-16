"""Centralized Ribbon-only ttk style definitions."""

from __future__ import annotations

from tkinter import ttk
from ui.theme_palette import ribbon_palette, theme_id


from .icon_service import tk_scale_factor, nearest_packaged_icon_size


def _smaller_font(font, *, fallback=("Segoe UI", 7)):
    """Return a one-step smaller tuple font without mutating app-global fonts."""
    if isinstance(font, (tuple, list)) and len(font) >= 2:
        try:
            size = int(font[1])
        except (TypeError, ValueError):
            return fallback
        values = list(font)
        values[1] = size - 1 if size > 0 else size + 1
        return tuple(values)
    return fallback




def _layout_without_focus(layout):
    """Return a ttk layout with native focus-ring elements removed.

    Keyboard focus is preserved; only the platform-drawn dotted Button.focus
    rectangle is suppressed for Ribbon tabs, where the selected underline is
    the visual state indicator.
    """
    cleaned = []
    for element, options in layout:
        options = dict(options)
        children = options.get("children", [])
        children = _layout_without_focus(children) if children else []
        if children:
            options["children"] = children
        else:
            options.pop("children", None)
        if str(element).lower().endswith(".focus"):
            cleaned.extend(children)
        else:
            cleaned.append((element, options))
    return cleaned

def configure_ribbon_styles(widget) -> None:
    """Configure a compact Office-like Ribbon without changing global ttk defaults."""
    style = ttk.Style(widget)
    root = widget.winfo_toplevel()
    variant = getattr(root, "_theme_variant", "light")
    fonts = getattr(root, "_app_fonts", {}) or {}
    ui_font = fonts.get("ui", ("Segoe UI", 9))
    small_font = fonts.get("ui_small", ("Segoe UI", 8))
    bold_font = fonts.get("ui_bold", ("Segoe UI", 9, "bold"))
    caption_font = _smaller_font(small_font)
    scale = tk_scale_factor(root)
    # 76 logical px Primary cells leave about 63 logical px for text after
    # border/padding. Let the ttk label element wrap to that width rather than
    # inserting manual newlines, which keeps wrapping DPI-aware and centered.
    primary_text_wrap = max(1, int(round(63 * scale)))

    colors = ribbon_palette(theme_id(root))
    tab_selected = colors["tab_selected"]
    tab_foreground = colors["tab_foreground"]
    tab_selected_foreground = colors["tab_selected_foreground"]
    tab_hover = colors["tab_hover"]
    tab_accent = colors["tab_accent"]
    tab_context_bg = colors["tab_context_bg"]
    tab_context_hover = colors["tab_context_hover"]
    tab_context_selected = colors["tab_context_selected"]
    tab_context_foreground = colors["tab_context_foreground"]
    tab_context_line = colors["tab_context_line"]
    tab_context_selected_foreground = colors["tab_context_selected_foreground"]
    bar_bg = colors["bar_bg"]
    page_bg = colors["page_bg"]
    hover_bg = colors["hover_bg"]
    pressed_bg = colors["pressed_bg"]
    foreground = colors["foreground"]
    muted = colors["muted"]
    separator = colors["separator"]
    bottom_border = colors["bottom_border"]
    accent = colors["accent"]
    context_bg = colors["context_bg"]
    context_hover = colors["context_hover"]
    context_selected = colors["context_selected"]
    context_foreground = colors["context_foreground"]
    context_underline_muted = colors["context_underline_muted"]
    primary_hover = colors["primary_hover"]
    danger_hover = colors["danger_hover"]
    focus = colors["focus"]

    # Continuous Ribbon surfaces.
    style.configure("Ribbon.Bar.TFrame", background=bar_bg, padding=0)
    style.configure("Ribbon.TabStrip.TFrame", background=bar_bg, padding=0)
    style.configure("Ribbon.Tab.Container.TFrame", background=bar_bg, padding=0)
    style.configure("Ribbon.Tab.Selected.Container.TFrame",
                    background=bar_bg if variant == "dark" else tab_selected, padding=0)
    style.configure("Ribbon.Tab.Underline.TFrame", background=bar_bg, padding=0)
    style.configure("Ribbon.Tab.Selected.Underline.TFrame", background=tab_accent, padding=0)
    style.configure("Ribbon.Tab.Hover.Underline.TFrame", background=colors["tab_hover_line"], padding=0)
    style.configure("Ribbon.ContextTab.Container.TFrame", background=tab_context_bg, padding=0)
    style.configure(
        "Ribbon.ContextTab.Selected.Container.TFrame",
        background=tab_context_selected,
        padding=0,
    )
    style.configure(
        "Ribbon.ContextTab.Underline.TFrame",
        background=tab_context_line,
        padding=0,
    )
    style.configure(
        "Ribbon.ContextTab.Selected.Underline.TFrame",
        background=tab_accent,
        padding=0,
    )
    style.configure("Ribbon.PageHost.TFrame", background=page_bg, padding=0)
    style.configure("Ribbon.Page.TFrame", background=page_bg, padding=0)
    style.configure("Ribbon.Group.TFrame", background=page_bg, padding=0)
    style.configure("Ribbon.BottomBorder.TFrame", background=bottom_border, padding=0)
    # Native separator elements can ignore the palette and paint a dark bevel.
    style.layout("Ribbon.Separator.TSeparator", [("Frame.border", {"sticky": "nswe"})])
    style.configure("Ribbon.Separator.TSeparator", background=colors["group_separator"],
                    borderwidth=0, relief="flat")
    style.configure(
        "Ribbon.GroupCaption.TLabel",
        background=page_bg,
        foreground=muted,
        font=caption_font,
        padding=(2, 0, 2, 0),
    )

    def button_style(
        name, background, font=ui_font, padding=(5, 2), *,
        anchor="center", justify="center", wraplength=0,
    ):
        native_layout = getattr(root, "_pcs_native_button_layout", None)
        if native_layout:
            style.layout(name, _layout_without_focus(native_layout))
        style.configure(
            name,
            background=background,
            foreground=foreground,
            font=font,
            padding=padding,
            relief="flat",
            borderwidth=1,
            anchor=anchor,
            justify=justify,
            wraplength=wraplength,
        )
        style.map(
            name,
            background=[
                ("disabled", page_bg),
                ("pressed", pressed_bg),
                ("active", hover_bg),
            ],
            foreground=[("disabled", muted)],
            relief=[("focus", "solid")],
            bordercolor=[("focus", focus)],
        )

    button_style("Ribbon.TButton", page_bg)
    button_style(
        "Ribbon.Small.TButton", page_bg, small_font, (3, 0),
        anchor="w", justify="left",
    )

    # Primary commands no longer render as persistent blue tiles. Their larger
    # icon/geometry establishes hierarchy; the surface appears only on hover.
    button_style(
        "Ribbon.Primary.TButton", page_bg, small_font, (4, 1),
        anchor="center", justify="center", wraplength=primary_text_wrap,
    )
    style.map(
        "Ribbon.Primary.TButton",
        background=[
            ("disabled", page_bg),
            ("pressed", pressed_bg),
            ("active", primary_hover),
        ],
    )

    # Danger remains compact like Secondary; destructive meaning appears on hover.
    button_style(
        "Ribbon.Danger.TButton", page_bg, small_font, (3, 0),
        anchor="w", justify="left",
    )
    style.map(
        "Ribbon.Danger.TButton",
        background=[
            ("disabled", page_bg),
            ("pressed", pressed_bg),
            ("active", danger_hover),
        ],
    )

    style.configure(
        "Ribbon.Toggle.TCheckbutton",
        padding=(3, 0),
        background=page_bg,
        foreground=foreground,
        font=small_font,
        indicatoron=False,
        relief="flat",
        borderwidth=1,
        anchor="w",
    )
    style.layout(
        "Ribbon.Toggle.TCheckbutton",
        [
            (
                "Checkbutton.padding",
                {
                    "sticky": "nswe",
                    "children": [("Checkbutton.label", {"sticky": "nswe"})],
                },
            )
        ],
    )
    style.map(
        "Ribbon.Toggle.TCheckbutton",
        background=[
            ("disabled", page_bg),
            ("selected", context_selected),
            ("pressed", pressed_bg),
            ("active", hover_bg),
        ],
        foreground=[("disabled", muted)],
        relief=[("selected", "sunken"), ("focus", "solid")],
        bordercolor=[("focus", focus)],
    )

    style.configure(
        "Ribbon.Checkbutton.TCheckbutton",
        padding=(2, 0),
        background=page_bg,
        foreground=foreground,
        font=small_font,
        anchor="w",
    )
    style.map(
        "Ribbon.Checkbutton.TCheckbutton",
        background=[("disabled", page_bg), ("active", hover_bg)],
        foreground=[("disabled", muted)],
        focuscolor=[("focus", focus)],
    )

    from ui.widget_style import selection_skin, presentation_palette
    check_palette = presentation_palette(theme_id(root))
    check_palette.update(text=foreground, disabled_text=muted, accent=focus)
    selection_skin(widget._root(), style, "Ribbon.Checkbutton.TCheckbutton", check_palette, page_bg)

    # Mixed compact columns share a leading icon slot. Retain the native check
    # indicator and focus skin; only its margins differ from standalone options.
    icon_slot = nearest_packaged_icon_size(20, scale)
    indicator = style.lookup("Ribbon.Checkbutton.TCheckbutton", "indicatorsize") or 10
    indicator_pixels = root.winfo_pixels(indicator)
    left_margin = max(0, (icon_slot - indicator_pixels) // 2)
    label_gap = 4  # ttk compound-label spacing, shared with compact buttons
    for name in ("Ribbon.Small.TButton", "Ribbon.Danger.TButton"):
        style.configure(name, space=label_gap)
    style.layout("Aligned.Ribbon.Checkbutton.TCheckbutton",
                 style.layout("Ribbon.Checkbutton.TCheckbutton"))
    style.configure("Aligned.Ribbon.Checkbutton.TCheckbutton", padding=(3, 0),
                    indicatormargin=(left_margin, 1,
                                     max(0, icon_slot + label_gap - indicator_pixels - left_margin), 1))

    # Compact tab strip: selected state is primarily communicated by underline,
    # while contextual tabs retain only a restrained tint and stronger type.
    tab_specs = (
        ("Ribbon.Tab.TButton", bar_bg, ui_font, (7, 2), tab_foreground),
        ("Ribbon.Tab.Selected.TButton", tab_selected, ui_font, (7, 2), tab_selected_foreground),
        ("Ribbon.ContextTab.TButton", tab_context_bg, bold_font, (10, 2), tab_context_foreground),
        (
            "Ribbon.ContextTab.Selected.TButton",
            tab_context_selected,
            bold_font,
            (10, 2),
            tab_context_selected_foreground,
        ),
    )
    for name, background, font, padding, text_color in tab_specs:
        style.configure(
            name,
            padding=padding,
            background=background,
            foreground=text_color,
            font=font,
            relief="flat",
            borderwidth=1,
        )
        active = (
            background
            if "Selected" in name
            else tab_context_hover if "Context" in name else tab_hover
        )
        style.map(
            name,
            background=[("pressed", pressed_bg), ("active", active)],
            foreground=[("disabled", muted)],
            bordercolor=[("pressed", tab_accent)],
        )

    # Keep tabs keyboard-focusable but suppress the native dotted focus rectangle.
    # Copy the active theme's Button layout so this remains theme-portable.
    try:
        tab_layout = _layout_without_focus(getattr(root, "_pcs_native_button_layout", style.layout("TButton")))
        for tab_style in (
            "Ribbon.Tab.TButton",
            "Ribbon.Tab.Selected.TButton",
            "Ribbon.ContextTab.TButton",
            "Ribbon.ContextTab.Selected.TButton",
        ):
            style.layout(tab_style, tab_layout)
    except Exception:
        pass

    button_style("Ribbon.Collapse.TButton", bar_bg, bold_font, (4, 0))
    button_style("Ribbon.Overflow.TButton", page_bg, bold_font, (7, 1))


__all__ = ["configure_ribbon_styles"]
