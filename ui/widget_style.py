"""Shared ttk image skins. Native widgets, bindings and data owners are retained."""

import tkinter as tk
from tkinter import ttk

from ui.ribbon.icon_service import tk_scale_factor


# Compatibility import for existing presentation consumers.
from ui.theme_palette import presentation_palette, theme_id


def ui_text_color(widget, role="muted"):
    """Resolve ordinary UI prose without touching scientific color encodings."""
    root = widget._root()
    palette = getattr(root, "_app_palette", None)
    if palette is None:
        palette = presentation_palette(theme_id(root))
    return palette[role]


def ui_font_name(widget, role):
    """A stable Tk font name keeps already-open labels/reports in sync."""
    from ui.style import get_app_fonts
    root = widget._root()
    name = "PCS_role_" + role
    if name not in root.tk.splitlist(root.tk.call("font", "names")):
        root.tk.call("font", "create", name)
    spec = get_app_fonts(widget)[role]
    actual = root.tk.call("font", "configure", name)
    current = dict(zip(actual[::2], actual[1::2]))
    weight = "bold" if "bold" in spec[2:] else "normal"
    if (current["-family"], int(current["-size"]), current["-weight"]) != (spec[0], spec[1], weight):
        root.tk.call("font", "configure", name, "-family", spec[0], "-size", spec[1],
                     "-weight", weight)
    roles = getattr(root, "_pcs_font_roles", None)
    if roles is None:
        roles = root._pcs_font_roles = set()
    roles.add(role)
    return name


def ui_label_style(widget, role="muted", base="TLabel"):
    """Compose semantic prose color with its existing background/density style."""
    root = widget._root()
    name = "PCS." + role + "." + base
    style = ttk.Style(widget)
    color = ui_text_color(widget, role)
    if style.configure(name, "foreground") != color:
        style.configure(name, foreground=color)
    roles = getattr(root, "_pcs_label_roles", None)
    if roles is None:
        roles = root._pcs_label_roles = set()
    roles.add((name, role))
    return name


def ui_native_colors(widget, **roles):
    """Enroll an ordinary native UI surface in palette refresh explicitly."""
    widget._pcs_color_roles = roles
    widget.configure(**{option: ui_text_color(widget, role) for option, role in roles.items()})
    return widget


def refresh_prose_styles(root, style):
    for role in getattr(root, "_pcs_font_roles", ()):
        ui_font_name(root, role)
    for name, role in getattr(root, "_pcs_label_roles", ()):
        color = ui_text_color(root, role)
        if style.configure(name, "foreground") != color:
            style.configure(name, foreground=color)
    # Only host reports explicitly enrolled via a PCS font name are refreshed.
    # Third-party/plugin Text widgets and scientific canvases are not recolored.
    pending = list(root.winfo_children())
    palette = root._app_palette
    input_font = tuple(root._app_fonts["ui"])
    input_font_changed = getattr(root, "_pcs_input_font_spec", None) != input_font
    root._pcs_input_font_spec = input_font
    while pending:
        widget = pending.pop()
        pending.extend(widget.winfo_children())
        roles = getattr(widget, "_pcs_color_roles", {})
        if roles:
            widget.configure(**{option: palette[role] for option, role in roles.items()})
        if input_font_changed and widget.winfo_class() in ("Entry", "TEntry", "TCombobox", "TSpinbox"):
            font = widget.cget("font")
            if str(font) == "PCS_role_ui":
                # Recompute existing glyph positions after named-font changes.
                widget.configure(font=font)
        if widget.winfo_class() == "TCombobox":
            popup = str(widget) + ".popdown.f.l"
            if root.tk.call("winfo", "exists", popup):
                root.tk.call(popup, "configure", "-background", style.lookup("TEntry", "fieldbackground"),
                             "-foreground", palette["text"], "-selectbackground", palette["selected"],
                             "-selectforeground", palette["text"], "-font", root._app_fonts["ui"])
        if widget.winfo_class() == "Text" and str(widget.cget("font")).startswith("PCS_role_"):
            widget.configure(background=palette["field"], foreground=palette["text"],
                             insertbackground=palette["text"],
                             selectbackground=palette["selected"], selectforeground=palette["text"])


def _images(root, style, element, size, painter, states, border=0, minimum_size=None):
    # Update images in place: widgets/styles retain them across theme refreshes.
    # Large flat tiles avoid thousands of native image blits. Their drawing
    # size can differ from the element's minimum geometry.
    cache = getattr(root, "_pcs_widget_images", None)
    if cache is None:
        cache = root._pcs_widget_images = {}
    key = (style.theme_use(), element, size)
    if key not in cache:
        cache[key] = tuple(tk.PhotoImage(master=root, width=size[0], height=size[1])
                           for _ in states)
    images = cache[key]
    paints = getattr(root, "_pcs_widget_paints", None)
    if paints is None:
        paints = root._pcs_widget_paints = {}
    signature = tuple(states)
    if paints.get(key) != signature:
        for image, (_, paint) in zip(images, states):
            image.blank()
            painter(image, paint)
        paints[key] = signature
    if element not in style.element_names():
        geometry = {} if minimum_size is None else dict(width=minimum_size[0], height=minimum_size[1])
        style.element_create(element, "image", images[0],
                             *((state, image) for (state, _), image in zip(states[1:], images[1:])),
                             border=border, sticky="nswe", **geometry)
    return element


def _face(image, paint):
    fill, edge, parent, underline, rounded = paint
    border = 4 if rounded else 2
    width, height = image.width(), image.height()
    # Expand only the uniform portions of the original 11px nine-slice.
    # Its corner pixels, border insets and underline remain unchanged.
    def source_coordinate(value, extent):
        return value if value < border else value - extent + 11 if value >= extent - border else border
    rows = []
    for iy in range(height):
        y = source_coordinate(iy, height)
        row = []
        for ix in range(width):
            x = source_coordinate(ix, width)
            dx, dy = x - min(7, max(3, x)), y - min(7, max(3, y))
            if rounded and dx * dx + dy * dy > 9:
                row.append(parent)
                continue
            inner = 1 <= x <= 9 and 1 <= y <= 9 and (not rounded or dx * dx + dy * dy <= 4)
            color = fill if inner else edge
            if underline and y >= 9:
                color = underline
            row.append(color)
        rows.append("{" + " ".join(row) + "}")
    image.put(" ".join(rows))


def button_skin(root, style, name, palette, role="Secondary", parent=None):
    """Only change the face; retain the caller's font, padding and anchor."""
    p = palette
    parent = parent or p["surface"]
    fill = p["primary"] if role == "Primary" else p["danger_bg"] if role == "Danger" else parent if role == "Utility" else p["button"]
    text = p["primary_text"] if role == "Primary" else p["danger"] if role == "Danger" else p["text"]
    hover = p["primary_hover"] if role == "Primary" else p["hover"]
    pressed = p["primary_pressed"] if role == "Primary" else p["pressed"]
    edge = p["primary_pressed"] if role == "Primary" else p["danger"] if role == "Danger" else p["edge"]
    element = "PCS." + name + ".face"
    states = [("", (fill, edge, parent, None, True)),
              ("disabled", (p["disabled"], p["edge"], parent, None, True)),
              ("pressed", (pressed, p["accent"], parent, None, True)),
              ("active focus", (hover, p["accent"], parent, None, True)),
              ("focus", (fill, p["accent"], parent, None, True)),
              ("active", (hover, p["accent"], parent, None, True))]
    _images(root, style, element, (64, 64), _face, states, border=4, minimum_size=(11, 11))
    style.configure(name, background=fill, foreground=text, bordercolor=edge, borderwidth=1)
    style.map(name, foreground=[("disabled", p["disabled_text"])],
              background=[("disabled", p["disabled"]), ("pressed", pressed), ("active", hover)],
              bordercolor=[("focus", p["accent"])])
    kind = "Menubutton" if name.endswith("TMenubutton") else "Button"
    children = [(kind + ".label", {"sticky": "nswe"})]
    if kind == "Menubutton":
        children.insert(0, ("Menubutton.indicator", {"side": "right", "sticky": ""}))
    style.layout(name, [(element, {"sticky": "nswe", "children": [
        (kind + ".padding", {"sticky": "nswe", "children": children})]})])


def _without_focus(layout):
    result = []
    for element, options in layout:
        options = dict(options)
        children = _without_focus(options.pop("children", []))
        if element.lower().endswith(".focus"):
            result.extend(children)
        else:
            if children:
                options["children"] = children
            result.append((element, options))
    return result


def compact_button_skin(root, style, name, palette):
    """Keep disclosure/toolbar fills and metrics; replace native dotted focus."""
    layout = getattr(root, "_pcs_native_button_layout", style.layout("TButton"))
    style.layout(name, _without_focus(layout))
    style.configure(name, borderwidth=1)
    style.map(name, relief=[("focus", "solid"), ("!focus", "flat")],
              bordercolor=[("focus", palette["accent"])])


def selection_skin(root, style, name, palette, background=None):
    """Replace dotted focus with an outer line, preserving native indicators."""
    p = palette
    bg = background or p["surface"]
    kind = "TCheckbutton" if name.endswith("TCheckbutton") else "TRadiobutton"
    base_layouts = getattr(root, "_pcs_selection_layouts", None)
    if base_layouts is None:
        base_layouts = root._pcs_selection_layouts = {}
    key = (style.theme_use(), kind)
    if key not in base_layouts:
        base_layouts[key] = _without_focus(style.layout(kind))
    def paint(image, color):
        if color:
            width, height = image.width(), image.height()
            image.put(color, to=(0, 0, width, 1))
            image.put(color, to=(0, height - 1, width, height))
            image.put(color, to=(0, 0, 1, height))
            image.put(color, to=(width - 1, 0, width, height))
    element = "PCS." + name + ".focusline"
    _images(root, style, element, (64, 64), paint,
            [("", None), ("disabled", None), ("focus", p["accent"])], border=2,
            minimum_size=(11, 11))
    style.layout(name, [(element, {"sticky": "nswe", "children": base_layouts[key]})])
    style.configure(name, background=bg, foreground=p["text"])
    style.map(name, background=[("!disabled", bg), ("disabled", bg)],
              foreground=[("disabled", p["disabled_text"]), ("!disabled", p["text"])])


def notebook_skin(root, style, palette):
    p = palette
    element = "PCS.Notebook.face"
    _images(root, style, element, (64, 64), _face,
            [("", (p["surface"], p["edge"], p["surface"], None, False)),
             ("disabled", (p["disabled"], p["edge"], p["surface"], None, False)),
             ("selected focus", (p["field"], p["accent"], p["surface"], p["accent"], False)),
             ("selected", (p["field"], p["edge"], p["surface"], p["accent"], False)),
             ("focus", (p["surface"], p["accent"], p["surface"], None, False)),
             ("active", (p["hover"], p["edge"], p["surface"], None, False))], border=2,
            minimum_size=(11, 11))
    style.layout("TNotebook.Tab", [(element, {"sticky": "nswe", "children": [
        ("Notebook.padding", {"sticky": "nswe", "children": [("Notebook.label", {"sticky": "nswe"})]})]})])
    style.map("TNotebook.Tab", expand=[("selected", (0, 0, 0, 0)), ("!selected", (0, 0, 0, 0))],
              foreground=[("disabled", p["disabled_text"]), ("!disabled", p["text"])],
              background=[("selected", p["field"]), ("active", p["hover"]), ("!selected", p["surface"])])


def scrollbar_skin(root, style, palette):
    p = palette
    scale = tk_scale_factor(root)
    slot, length = max(10, round(10 * scale)), max(26, round(26 * scale))
    thickness = max(4, round(4 * scale))
    for orient in ("Vertical", "Horizontal"):
        size = (slot, length) if orient == "Vertical" else (length, slot)
        element = f"PCS.{orient}{slot}.Scrollbar.thumb"
        def paint(image, color, orient=orient):
            radius = thickness / 2
            for y in range(size[1]):
                for x in range(size[0]):
                    across, along = (x, y) if orient == "Vertical" else (y, x)
                    a = across - (slot - 1) / 2
                    b = along - min(length - radius - 1, max(radius, along))
                    if a * a + b * b <= radius * radius:
                        image.put(color, (x, y))
        _images(root, style, element, size, paint,
                [("", p["scroll"]), ("disabled", p["scroll"]),
                 ("pressed", p["accent"]), ("active", p["scroll_hover"])],
                border=(0, round(thickness)) if orient == "Vertical" else (round(thickness), 0))
        name = orient + ".TScrollbar"
        # Clam's native trough retains bevel colors even at borderwidth=0.
        # A flat image keeps the native *.trough binding identity without a rail.
        trough = f"PCS.{orient}{slot}.Scrollbar.trough"
        _images(root, style, trough, (64, 64),
                lambda image, color: image.put(color, to=(0, 0, 64, 64)),
                [("", p["surface"])], minimum_size=(1, 1))
        style.configure(name, background=p["surface"], troughcolor=p["surface"],
                        borderwidth=0, width=slot, arrowsize=0)
        # Keep native semantic element names for ttk identify/drag/page bindings.
        style.layout(name, [(trough, {"sticky": "nswe", "children": [
            (element, {"sticky": "ns" if orient == "Vertical" else "we", "expand": "1"})]})])


def scale_skin(root, style, palette):
    """Round the native horizontal Scale thumb; do not replace its bindings."""
    p = palette
    size = max(14, round(14 * tk_scale_factor(root)))
    def paint(image, color):
        radius = size / 2 - 1
        for y in range(size):
            for x in range(size):
                if (x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2 <= radius ** 2:
                    image.put(color, (x, y))
    element = f"PCS.Horizontal{size}.Scale.slider"
    _images(root, style, element, (size, size), paint,
            [("", p["muted"]), ("disabled", p["disabled_text"]),
             ("pressed", p["accent"]), ("focus", p["accent"]), ("active", p["text"])])
    def paint_track(image, paint):
        background, color = paint
        image.put(background, to=(0, 0, 64, size))
        middle = size // 2
        half = max(1, round(tk_scale_factor(root)))
        image.put(color, to=(0, middle - half, 64, middle + half + 1))
    trough = f"PCS.Horizontal{size}.Scale.trough"
    _images(root, style, trough, (64, size), paint_track,
            [("", (p["surface"], p["edge"]))], minimum_size=(1, size))
    style.configure("PCS.Horizontal.TScale", troughcolor=p["edge"], borderwidth=0)
    style.layout("PCS.Horizontal.TScale", [(trough, {
        "sticky": "we", "children": [(element, {"side": "left", "sticky": ""})]})])


def sash_skin(root, style, palette):
    """Thin divider in a native Panedwindow sash with unchanged drag semantics."""
    thickness = max(8, round(8 * tk_scale_factor(root)))
    for orient in ("Horizontal", "Vertical"):
        size = (64, thickness) if orient == "Horizontal" else (thickness, 64)
        def paint(image, paint, orient=orient):
            background, color = paint
            image.put(background, to=(0, 0, size[0], size[1]))
            middle = thickness // 2
            stripe = (0, middle, 64, middle + 1) if orient == "Horizontal" else (middle, 0, middle + 1, 64)
            image.put(color, to=stripe)
        element = f"PCS.{orient}{thickness}.divider"
        _images(root, style, element, size, paint,
                [("", (palette["surface"], palette["edge"])),
                 ("active", (palette["surface"], palette["accent"]))],
                minimum_size=(1, thickness) if orient == "Horizontal" else (thickness, 1))
        style.layout("PCS.TPanedwindow." + orient + ".Sash", [(element, {"sticky": "nswe"})])
    style.configure("PCS.TPanedwindow", background=palette["surface"])
