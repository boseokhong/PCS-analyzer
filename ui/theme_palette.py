"""Light/Dark UI color definitions; no widget geometry or scientific colors.

Palette functions return fresh dictionaries so scoped styling can customize copies.
The accent token is a theme-owned color, not a user-selectable setting.
"""

THEME_LABELS = {
    "light": "Light", "dark": "Dark",
    "light_teal": "Light Teal", "dark_teal": "Dark Teal",
    "light_violet": "Light Violet", "dark_violet": "Dark Violet",
    "light_slate": "Light Slate", "dark_slate": "Dark Slate",
}


def normalize_theme(value):
    name = str(value or "light").strip().lower().replace(" ", "_")
    return name if name in THEME_LABELS else "light"


def theme_mode(value):
    return "dark" if normalize_theme(value).startswith("dark") else "light"


def theme_id(widget):
    root = widget.winfo_toplevel()
    app_root = root._root() if hasattr(root, "_root") else root
    return getattr(root, "_theme_id", getattr(app_root, "_theme_id", getattr(root, "_theme_variant", "light")))

def _blue_presentation_palette(variant):
    if variant == "dark":
        return dict(surface="#20262D", field="#1E252D", text="#EDF2F7",
                    muted="#B0BECE", edge="#566779", hover="#354657",
                    pressed="#435C76", accent="#91BDEA", selected="#354F6A",
                    disabled="#343D47", disabled_text="#98A6B5", button="#3B4857",
                    primary="#91BDEA", primary_hover="#AED0F0", primary_pressed="#79A9D9",
                    primary_text="#14283C", danger="#F3A5B2", danger_bg="#49303A", warning="#F0C27B",
                    scroll="#526172", scroll_hover="#91A7BE")
    return dict(surface="#EEF1F5", field="#FFFFFF", text="#1D2731",
                muted="#596879", edge="#B8C4D1", hover="#E3EBF4",
                pressed="#CCDDED", accent="#366B9B", selected="#D9E9F8",
                disabled="#E6EAF0", disabled_text="#727D89", button="#E8EEF5",
                primary="#366B9B", primary_hover="#285982", primary_pressed="#20496D",
                primary_text="#FFFFFF", danger="#A13343", danger_bg="#FAE9ED", warning="#8A570D",
                scroll="#B5C2CF", scroll_hover="#768DA5")

def _blue_panel_palette(variant):
    if variant == "dark":
        return {
            "panel": "#23272D",
            "body": "#2B3037",
            "header": "#30363E",
            "header_hover": "#39414B",
            "foreground": "#F0F2F5",
            "muted": "#AEB7C2",
            "subheading": "#CCD4DD",
            "separator": "#46505A",
            "entry": "#2B2F36",
            "primary": "#5B8FC1",
            "primary_hover": "#6A9DCE",
            "primary_pressed": "#4B7DAE",
            "primary_foreground": "#FFFFFF",
            "danger": "#F0B0B0",
            "danger_bg": "#4A3033",
            "danger_hover": "#5A393D",
            "slider_track": "#4A525D",
            "slider_thumb": "#C2CBD5",
            "slider_thumb_hover": "#E0E6EC",
            "slider_thumb_disabled": "#707983",
            "slider_outline": "#1D2228",
            "gutter_idle": "#46505A",
            "gutter_hover": "#7FA5C9",
            "gutter_armed": "#6FA7D8",
            "chip_selected_bg": "#3A424C",
            "chip_selected_hover": "#434D58",
            "chip_selected_border": "#65717E",
            "chip_unselected_bg": "#2B3037",
            "chip_unselected_border": "#46505A",
            "chip_unselected_text": "#7F8995",
            "chip_count": "#AEB7C2",
            "chip_dot_off": "#7F8995",
            "chip_dot_outline": "#B9C2CC",
            "chip_hover_border": "#7FA5C9",
        }
    return {
        "panel": "#EEF1F5",
        "body": "#F8FAFC",
        "header": "#E4E9EF",
        "header_hover": "#D8E0E8",
        "foreground": "#1D2731",
        "muted": "#66717E",
        "subheading": "#4A5968",
        "separator": "#CBD3DC",
        "entry": "#FFFFFF",
        "primary": "#3E74A5",
        "primary_hover": "#4B83B8",
        "primary_pressed": "#32638F",
        "primary_foreground": "#FFFFFF",
        "danger": "#8F4148",
        "danger_bg": "#F7E8E9",
        "danger_hover": "#F2D9DB",
        "slider_track": "#CCD5DF",
        "slider_thumb": "#66788B",
        "slider_thumb_hover": "#4F6478",
        "slider_thumb_disabled": "#ADB6C0",
        "slider_outline": "#F8FAFC",
        "gutter_idle": "#D2D9E1",
        "gutter_hover": "#7F9DB8",
        "gutter_armed": "#356F9F",
        "chip_selected_bg": "#E8EEF4",
        "chip_selected_hover": "#DEE7F0",
        "chip_selected_border": "#AEBBC8",
        "chip_unselected_bg": "#F8FAFC",
        "chip_unselected_border": "#D5DCE4",
        "chip_unselected_text": "#8893A0",
        "chip_count": "#66717E",
        "chip_dot_off": "#9AA5B1",
        "chip_dot_outline": "#F8FAFC",
        "chip_hover_border": "#3E74A5",
    }

def _blue_ribbon_palette(variant):
    if variant == "dark":
        bar_bg = "#23272D"
        page_bg = "#2A2F35"
        hover_bg = "#373E47"
        pressed_bg = "#434D58"
        foreground = "#F0F2F5"
        muted = "#A9B2BD"
        separator = "#46505A"
        bottom_border = "#515A64"
        accent = "#789FC2"
        context_bg = "#293846"
        context_hover = "#344B60"
        context_selected = "#31465A"
        context_foreground = "#D9EAF8"
        context_underline_muted = "#49647C"
        primary_hover = "#354A5D"
        danger_hover = "#584044"
        focus = "#8CB9E2"
    else:
        bar_bg = "#E9EDF2"
        page_bg = "#F7F9FB"
        hover_bg = "#E4EAF1"
        pressed_bg = "#D7E0E9"
        foreground = "#1D2731"
        muted = "#6C7681"
        separator = "#D3DAE2"
        bottom_border = "#BCC6D0"
        accent = "#4C7EA8"
        context_bg = "#EAF1F7"
        context_hover = "#DEEAF4"
        context_selected = "#F0F5F9"
        context_foreground = "#2C638E"
        context_underline_muted = "#AEC4D6"
        primary_hover = "#E2EDF6"
        danger_hover = "#F4E3E4"
        focus = "#4C7EA8"

    # Approved Light tab palette only; command/group surfaces keep their colors.
    tab_selected = page_bg
    tab_foreground = tab_selected_foreground = foreground
    tab_hover = hover_bg
    tab_accent = accent
    tab_context_bg, tab_context_hover = context_bg, context_hover
    tab_context_selected, tab_context_foreground = context_selected, context_foreground
    tab_context_line = context_underline_muted
    tab_context_selected_foreground = context_foreground
    if variant != "dark":
        tab_selected = tab_context_selected = "#DCEAF7"
        tab_foreground = "#596879"
        tab_selected_foreground = tab_context_selected_foreground = "#204D73"
        tab_hover = tab_context_hover = "#E0E8F1"
        tab_accent = "#356F9F"
        tab_context_bg, tab_context_foreground = "#EFF4F9", "#55718A"
        tab_context_line = "#D0DCE6"
    return {
        "tab_selected": tab_selected,
        "tab_foreground": tab_foreground,
        "tab_selected_foreground": tab_selected_foreground,
        "tab_hover": tab_hover,
        "tab_accent": tab_accent,
        "tab_context_bg": tab_context_bg,
        "tab_context_hover": tab_context_hover,
        "tab_context_selected": tab_context_selected,
        "tab_context_foreground": tab_context_foreground,
        "tab_context_line": tab_context_line,
        "tab_context_selected_foreground": tab_context_selected_foreground,
        "bar_bg": bar_bg,
        "page_bg": page_bg,
        "hover_bg": hover_bg,
        "pressed_bg": pressed_bg,
        "foreground": foreground,
        "muted": muted,
        "separator": separator,
        "bottom_border": bottom_border,
        "accent": accent,
        "context_bg": context_bg,
        "context_hover": context_hover,
        "context_selected": context_selected,
        "context_foreground": context_foreground,
        "context_underline_muted": context_underline_muted,
        "primary_hover": primary_hover,
        "danger_hover": danger_hover,
        "focus": focus,
    }


def _blue_base_palette(variant):
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
    return {
        "bg": bg,
        "fg": fg,
        "panel_bg": panel_bg,
        "input_bg": input_bg,
        "input_fg": input_fg,
        "input_disabled_bg": input_disabled_bg,
        "tree_bg": tree_bg,
        "tree_fg": tree_fg,
        "tree_odd_bg": tree_odd_bg,
        "tree_sel_bg": tree_sel_bg,
        "tree_sel_fg": tree_sel_fg,
        "heading_bg": heading_bg,
        "scale_trough": scale_trough,
        "scale_active": scale_active,
    }


def _blue_workspace_palette(variant):
    if variant == "dark":
        return dict(line="#46505A", hover="#7FA5C9", armed="#6FA7D8")
    return dict(line="#CBD3DC", hover="#607F9B", armed="#3E74A5")


def _blue_document_palette(variant):
    dark = variant == "dark"
    return dict(
        strip="#f4f5f7", foreground="#202124",
        inactive="#192027" if dark else "#dce2e9",
        inactive_hover="#2b3540" if dark else "#e3e8ee",
        inactive_foreground="#98a6b5" if dark else "#596879",
        active="#202428" if dark else "#ffffff",
        active_hover="#202428" if dark else "#ffffff",
        border="#71808f" if dark else "#8f9aa6",
    )


def _blue_document_arrow_palette(variant):
    dark = variant == "dark"
    return dict(
        background="#2c3b4d" if dark else "#e4ebf3",
        disabled_background="#222b35" if dark else "#edf1f5",
        edge="#637c96" if dark else "#9aaec2",
        disabled_edge="#46515e" if dark else "#bcc6d0",
        foreground="#b9cfe3" if dark else "#365572",
        hover="#3b526b" if dark else "#d2e0ef",
        hover_foreground="#e2edf8" if dark else "#244b70",
        disabled_foreground="#657587" if dark else "#a4afba",
    )


# Approved mockup colors. Status/danger/scientific colors remain independent.
_COLOR_THEMES = {
    "light_slate": dict(surface="#F0F2F4", page="#F9FAFB", field="#FFFFFF", header="#E3E8ED", selected="#DCE4EC", inactive="#E1E6EB", hover="#DDE5EC", text="#29343F", muted="#667583", edge="#BDC8D2", line="#D3DBE2", accent="#586D80", primary_text="#FFFFFF", primary_hover="#485E72"),
    "dark_slate": dict(surface="#252A30", page="#2D343C", field="#20262C", header="#343F4A", selected="#3B4C5C", inactive="#2B333B", hover="#3A4855", text="#E8EDF2", muted="#ADB9C5", edge="#586B7D", line="#414F5C", accent="#9EB2C5", primary_text="#202E3B", primary_hover="#B3C3D3"),
    "light_teal": dict(surface="#EEF4F4", page="#F8FBFB", field="#FFFFFF", header="#E0ECEC", selected="#D4E9E9", inactive="#DFE8E8", hover="#D5E5E5", text="#243638", muted="#617576", edge="#B7C9CA", line="#D0DDDD", accent="#287D83", primary_text="#FFFFFF", primary_hover="#216B70"),
    "light_violet": dict(surface="#F2F0F6", page="#FAF9FC", field="#FFFFFF", header="#E9E4F1", selected="#E3DBF1", inactive="#E5E1EC", hover="#E5DFF0", text="#302B3C", muted="#70677C", edge="#C6BDCF", line="#DCD5E4", accent="#7160A3", primary_text="#FFFFFF", primary_hover="#60508F"),
    "dark_teal": dict(surface="#222B2D", page="#293436", field="#1D2628", header="#303E40", selected="#334E51", inactive="#273335", hover="#354A4D", text="#E5EFEF", muted="#A3B7B8", edge="#50676A", line="#3D5053", accent="#79C3C5", primary_text="#163437", primary_hover="#94D3D4"),
    "dark_violet": dict(surface="#29262F", page="#302C38", field="#231F2A", header="#3A3444", selected="#483D5B", inactive="#2E2937", hover="#443A53", text="#ECE7F3", muted="#B3A8C2", edge="#665C76", line="#494150", accent="#B5A3DD", primary_text="#302342", primary_hover="#C7B7E8"),
}


def presentation_palette(variant):
    result = _blue_presentation_palette(theme_mode(variant))
    c = _COLOR_THEMES.get(normalize_theme(variant))
    if c:
        result.update({key: c[key] for key in ("surface", "field", "text", "muted", "edge", "hover", "selected", "accent", "primary_text", "primary_hover")})
        result.update(primary=c["accent"], primary_pressed=c["primary_hover"],
                      pressed=c["selected"], button=c["header"], disabled=c["inactive"],
                      disabled_text=c["muted"], scroll=c["edge"], scroll_hover=c["muted"])
    return result


def history_palette(variant):
    """Matching history icons on the menu surface; emphasis appears on hover."""
    p = presentation_palette(variant)
    dark = theme_mode(variant) == "dark"
    fg = "#91BDEA" if dark else "#356F9F"
    return {"undo": (p["field"], fg), "redo": (p["field"], fg),
            "disabled_foreground": "#74808D" if dark else "#929CA6",
            "hover": p["hover"], "pressed": p["pressed"],
            "edge": p["edge"], "focus": p["accent"]}


def panel_palette(variant):
    result = _blue_panel_palette(theme_mode(variant))
    c = _COLOR_THEMES.get(normalize_theme(variant))
    if c:
        result.update(panel=c["surface"], body=c["page"], header=c["header"],
                      header_hover=c["hover"], foreground=c["text"], muted=c["muted"],
                      subheading=c["text"], separator=c["edge"], entry=c["field"],
                      primary=c["accent"], primary_hover=c["primary_hover"],
                      primary_pressed=c["primary_hover"], primary_foreground=c["primary_text"],
                      slider_track=c["edge"], slider_thumb=c["muted"],
                      slider_thumb_hover=c["accent"], slider_thumb_disabled=c["edge"],
                      slider_outline=c["page"], gutter_idle=c["line"],
                      gutter_hover=c["edge"], gutter_armed=c["accent"],
                      chip_selected_bg=c["selected"], chip_selected_hover=c["hover"],
                      chip_selected_border=c["muted"], chip_unselected_bg=c["page"],
                      chip_unselected_border=c["edge"], chip_unselected_text=c["muted"],
                      chip_count=c["muted"], chip_dot_off=c["muted"],
                      chip_dot_outline=c["edge"], chip_hover_border=c["accent"])
    return result


def ribbon_palette(variant):
    result = _blue_ribbon_palette(theme_mode(variant))
    c = _COLOR_THEMES.get(normalize_theme(variant))
    if c:
        result.update(bar_bg=c["header"], page_bg=c["page"], hover_bg=c["hover"],
                      pressed_bg=c["selected"], foreground=c["text"], muted=c["muted"],
                      separator=c["edge"], bottom_border=c["edge"], accent=c["accent"],
                      context_bg=c["page"], context_hover=c["hover"],
                      context_selected=c["selected"], context_foreground=c["accent"],
                      context_underline_muted=c["edge"], primary_hover=c["hover"], focus=c["accent"],
                      tab_selected=c["selected"], tab_foreground=c["muted"],
                      tab_selected_foreground=c["accent"], tab_hover=c["hover"],
                      tab_accent=c["accent"], tab_context_bg=c["page"],
                      tab_context_hover=c["hover"], tab_context_selected=c["selected"],
                      tab_context_foreground=c["muted"], tab_context_line=c["edge"],
                      tab_context_selected_foreground=c["accent"])
    result["tab_hover_line"] = "#5E6B79" if theme_mode(variant) == "dark" else "#B8C4D0"
    result["group_separator"] = "#46505A" if theme_mode(variant) == "dark" else "#D0DAE3"
    if c:
        result.update(tab_hover_line=c["edge"], group_separator=c["line"])
    return result


def base_palette(variant):
    result = _blue_base_palette(theme_mode(variant))
    c = _COLOR_THEMES.get(normalize_theme(variant))
    if c:
        result.update(tree_odd_bg=c["page"], scale_trough=c["field"], scale_active=c["muted"])
    return result


def workspace_palette(variant):
    c = _COLOR_THEMES.get(normalize_theme(variant))
    return dict(line=c["line"], hover=c["edge"], armed=c["accent"]) if c else _blue_workspace_palette(theme_mode(variant))


def document_palette(variant):
    result = _blue_document_palette(theme_mode(variant))
    c = _COLOR_THEMES.get(normalize_theme(variant))
    if c:
        result.update(strip=c["surface"], foreground=c["text"], inactive=c["inactive"],
                      inactive_hover=c["hover"], inactive_foreground=c["muted"],
                      active=c["field"], active_hover=c["field"], border=c["edge"])
    return result


def document_arrow_palette(variant):
    result = _blue_document_arrow_palette(theme_mode(variant))
    c = _COLOR_THEMES.get(normalize_theme(variant))
    if c:
        result.update(background=c["header"], disabled_background=c["surface"],
                      edge=c["edge"], disabled_edge=c["line"], foreground=c["muted"],
                      hover=c["hover"], hover_foreground=c["accent"], disabled_foreground=c["muted"])
    return result


def nmr_analysis_accent(variant):
    c = _COLOR_THEMES.get(normalize_theme(variant))
    return c["accent"] if c else ("#85AED4" if theme_mode(variant) == "dark" else "#4F84B5")
