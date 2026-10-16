"""Workbench-only presentation, derived from the application's ttk palette.

No command, binding, processing state or plot-data ownership lives here.
"""

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from ui.style import get_app_fonts, table_row_height
from ui.theme_palette import workspace_palette, theme_id, nmr_analysis_accent
from ui.widget_style import button_skin, presentation_palette, compact_button_skin
from ui.icons import TOOLBAR_ICON_SIZE
from ui.ribbon.icon_service import nearest_packaged_icon_size, tk_scale_factor


def size_workbench_action(widget):
    """Match text and icon actions without fixed pixel heights or clipped DPI fonts."""
    style = ttk.Style(widget)
    scale = tk_scale_factor(widget._root())
    font = style.lookup(widget.cget("style"), "font") or get_app_fonts(widget._root())["ui"]
    text_height = tkfont.Font(root=widget, font=font).metrics("linespace")
    content_height = text_height
    images = widget.cget("image")
    if images:
        image_name = widget.tk.splitlist(images)[0]
        content_height = max(content_height, int(widget.tk.call("image", "height", image_name)))
    # Plain-text and missing-icon actions reserve the same content height.
    target = max(text_height, nearest_packaged_icon_size(TOOLBAR_ICON_SIZE, scale))
    extra = max(0, target - content_height)
    compact = str(widget.cget("style")).startswith("NMR.Phase.")
    inset = 0 if compact else max(1, round(scale))
    horizontal = max(3, round((4 if compact else 5) * scale))
    widget.configure(padding=(horizontal, inset + extra // 2,
                              horizontal, inset + (extra + 1) // 2))


def _blend(widget, first, second, amount):
    a, b = widget.winfo_rgb(first), widget.winfo_rgb(second)
    return "#" + "".join(f"{round((x * (1 - amount) + y * amount) / 257):02x}"
                         for x, y in zip(a, b))


def _style_splitters(widget):
    """Paint a one-pixel divider inside the unchanged native sash footprint."""
    root = widget._root()
    style = ttk.Style(root)
    bg = style.lookup("TFrame", "background") or root.cget("background")
    bg = _blend(widget, bg, bg, 0)
    dark = getattr(root, "_theme_variant", "light") == "dark"
    # Match ui.components' main workspace divider, without its collapse logic.
    divider = workspace_palette(theme_id(root))
    line, hover = divider["line"], divider["hover"]
    thickness = int(style.lookup("Sash", "sashthickness") or 6)
    images = getattr(root, "_nmr_sash_images", None)
    if images is None:
        images = root._nmr_sash_images = {}
    paints = getattr(root, "_nmr_sash_paints", None)
    if paints is None:
        paints = root._nmr_sash_paints = {}
    for orient in ("Horizontal", "Vertical"):
        key = (style.theme_use(), orient, thickness)
        if key not in images:
            width, height = ((64, thickness) if orient == "Horizontal" else (thickness, 64))
            images[key] = tuple(tk.PhotoImage(master=root, width=width, height=height)
                                for _ in range(2))
        normal, active = images[key]
        signature = (bg, line, hover)
        if paints.get(key) != signature:
            for image, color in ((normal, line), (active, hover)):
                image.put(bg, to=(0, 0, image.width(), image.height()))
                center = thickness // 2
                stripe = ((0, center, image.width(), center + 1) if orient == "Horizontal"
                          else (center, 0, center + 1, image.height()))
                image.put(color, to=stripe)
            paints[key] = signature
        element = f"NMR.{orient}.divider{thickness}"
        if element not in style.element_names():
            minimum = (1, thickness) if orient == "Horizontal" else (thickness, 1)
            style.element_create(element, "image", normal, ("active", active), sticky="nswe",
                                 width=minimum[0], height=minimum[1])
        name = f"NMR.TPanedwindow.{orient}.Sash"
        layout = [(element, {"sticky": "nswe"})]
        if style.layout(name) != layout:
            style.layout(name, layout)
    if style.configure("NMR.TPanedwindow", "background") != bg:
        style.configure("NMR.TPanedwindow", background=bg)


class WorkbenchPanedWindow(ttk.PanedWindow):
    """Keep ttk's class drag handlers; add only a visual hover projection."""

    def __init__(self, master, **kwargs):
        _style_splitters(master)
        super().__init__(master, style="NMR.TPanedwindow", **kwargs)
        self.bind("<Motion>", self._highlight_sash, add="+")
        self.bind("<Leave>", self._clear_highlight, add="+")

    def _highlight_sash(self, event):
        over_sash = str(self.identify(event.x, event.y)).isdigit()
        self.state(["active" if over_sash else "!active"])

    def _clear_highlight(self, _event):
        self.state(["!active"])


class AccordionHeader(ttk.Button):
    """Style the existing arrow/text projection without owning section state."""

    def configure(self, cnf=None, **kwargs):
        options = dict(cnf or {}) if isinstance(cnf, dict) else {}
        options.update(kwargs)
        if "text" in options:
            options["style"] = ("NMR.OpenSection.TButton"
                                if str(options["text"]).startswith("▾")
                                else "NMR.Section.TButton")
        return super().configure(cnf, **options)

    config = configure


class WorkbenchPresentation:
    def __init__(self, window):
        self.window = window
        self.palette = None

    def refresh(self, *, force=False):
        window = self.window
        style = ttk.Style(window)
        bg = style.lookup("TFrame", "background") or window._root().cget("background")
        fg = style.lookup("TLabel", "foreground") or "SystemWindowText"
        field = style.lookup("TEntry", "fieldbackground") or bg
        selected = style.lookup("Treeview", "background", ("selected",)) or field
        selected_fg = style.lookup("Treeview", "foreground", ("selected",)) or fg
        disabled = style.lookup("TEntry", "foreground", ("disabled",)) or fg
        # Native Tk system-color names are valid for widgets but not Matplotlib.
        bg, fg, field, selected, selected_fg, disabled = (
            _blend(window, color, color, 0)
            for color in (bg, fg, field, selected, selected_fg, disabled)
        )
        fonts = get_app_fonts(window._root())
        palette = (bg, fg, field, selected, selected_fg, disabled, repr(fonts))
        if not force and palette == self.palette:
            return
        self.palette = palette
        muted = _blend(window, fg, bg, .30)
        edge = _blend(window, bg, fg, .18)
        hover = _blend(window, bg, selected, .35)
        primary = _blend(window, bg, selected, .55)
        secondary = _blend(window, bg, field, .65)
        if secondary == bg:
            secondary = _blend(window, bg, fg, .06)
        section_hover = style.lookup("ControlPanel.Header.TButton", "background", ("active",)) or hover
        section = style.lookup("ControlPanel.Header.TButton", "background") or secondary
        if getattr(window._root(), "_theme_variant", "light") == "dark":
            # The legacy header's light fill needs a darker surface for readable text.
            section = _blend(window, bg, section_hover, .60)
        section_open = _blend(window, section, fg, .04)
        _style_splitters(window)
        window.configure(background=bg)
        # Shared tooltip helper reads these attributes from the Toplevel.
        window._theme_variant = getattr(window._root(), "_theme_variant", "light")
        window._app_fonts = fonts

        for role in ("Toolbar", "Primary", "Secondary", "Danger", "Section",
                     "OpenSection", "Advanced"):
            name = f"NMR.{role}.TButton"
            fill = primary if role == "Primary" else secondary
            if role == "Advanced":
                fill = bg
            elif role == "Section":
                fill = section
            elif role == "OpenSection":
                fill = section_open
            style.configure(name, background=fill, foreground=fg, bordercolor=edge,
                            lightcolor=fill, darkcolor=fill, relief="flat", borderwidth=1,
                            focuscolor=selected, focusthickness=1, width=0,
                            padding=(5, 1),
                            font=fonts["ui_bold" if role in ("Primary", "Section", "OpenSection")
                                       else "ui_small_bold" if role == "Advanced" else "ui"])
            style.map(name,
                      background=[("disabled", bg), ("pressed", selected), ("active", hover)],
                      foreground=[("disabled", disabled), ("pressed", selected_fg)],
                      bordercolor=[("focus", selected), ("active", edge)],
                      relief=[("pressed", "sunken"), ("!pressed", "flat")])
        for role in ("Section", "OpenSection"):
            style.configure(f"NMR.{role}.TButton", anchor="w", padding=(5, 2))
            style.map(f"NMR.{role}.TButton",
                      background=[("disabled", bg), ("pressed", section_hover), ("active", section_hover)],
                      foreground=[("disabled", disabled), ("!disabled", fg)])
        style.configure("NMR.Danger.TButton", foreground=muted)
        for role in ("Primary", "Secondary", "Danger", "Advanced"):
            source, target = f"NMR.{role}.TButton", f"NMR.Phase.{role}.TButton"
            style.configure(target, **style.configure(source))
            style.map(target, **style.map(source))
            style.configure(target, padding=(4, 0))
        shared = presentation_palette(theme_id(window._root()))
        for role in ("Primary", "Secondary", "Danger", "Advanced", "Toolbar"):
            visual_role = role if role in ("Primary", "Secondary", "Danger") else "Utility"
            for prefix in ("NMR.", "NMR.Phase."):
                button_skin(window._root(), style, prefix + role + ".TButton",
                            shared, visual_role, parent=bg)
        native_layout = getattr(window._root(), "_pcs_native_button_layout", None)
        if native_layout:
            for role in ("Section", "OpenSection"):
                compact_button_skin(window._root(), style, "NMR." + role + ".TButton", shared)

        style.configure("NMR.Toolbar.TMenubutton", background=secondary, foreground=fg,
                        bordercolor=edge, lightcolor=secondary, darkcolor=secondary,
                        relief="flat", borderwidth=1, width=0, padding=(5, 1), font=fonts["ui"])
        style.map("NMR.Toolbar.TMenubutton",
                  background=[("disabled", bg), ("pressed", selected), ("active", hover)],
                  foreground=[("disabled", disabled), ("pressed", selected_fg)],
                  bordercolor=[("focus", selected), ("active", edge)])
        button_skin(window._root(), style, "NMR.Toolbar.TMenubutton", shared, parent=bg)
        style.configure("NMR.Advanced.TButton", anchor="w", foreground=muted)
        style.configure("NMR.Muted.TLabel", foreground=muted, font=fonts["ui_small"])
        style.configure("NMR.Section.TLabel", foreground=fg, background=bg, font=fonts["ui_bold"])
        style.configure("NMR.Mode.TLabel", foreground=fg, background=primary,
                        padding=(6, 2), font=fonts["ui_small_bold"])
        for kind in ("TEntry", "TCombobox"):
            style.configure(f"NMR.Compact.{kind}", padding=(3, 1), bordercolor=edge,
                            lightcolor=field, darkcolor=field)
            style.map(f"NMR.Compact.{kind}", bordercolor=[("focus", selected)],
                      fieldbackground=[("disabled", style.lookup("TEntry", "fieldbackground", ("disabled",))),
                                       ("readonly", field)],
                      foreground=[("disabled", disabled), ("readonly", fg)])
        for name in ("NMR.Panel.TLabelframe", "NMR.Spectrum.TLabelframe",
                     "NMR.Role.TLabelframe", "NMR.Reference.TLabelframe"):
            style.configure(name, borderwidth=0, relief="flat", background=bg)
        style.configure("NMR.Inspector.TNotebook", borderwidth=0, padding=0, background=bg)
        style.configure("NMR.Inspector.TNotebook.Tab", padding=(4, 2), background=bg,
                        foreground=muted, bordercolor=edge, lightcolor=bg, darkcolor=bg,
                        font=fonts["ui_small"])
        style.map("NMR.Inspector.TNotebook.Tab",
                  background=[("selected", field), ("active", hover)],
                  foreground=[("selected", fg), ("active", fg)],
                  lightcolor=[("selected", selected)],
                  expand=[("selected", [0, 0, 0, 0])])
        notebook = getattr(window, "_inspector_notebook", None)
        analysis_tabs = getattr(window, "_inspector_analysis_tabs", ())
        if notebook is not None and analysis_tabs:
            # ttk exposes no per-tab background style. Keep its native tab
            # labels, focus and selection behavior and add a subtle shared tint.
            scale = tk_scale_factor(window._root())
            width, height = max(2, round(3 * scale)), max(8, round(12 * scale))
            accent = getattr(window, "_inspector_analysis_accent", None)
            if accent is None or (accent.width(), accent.height()) != (width, height):
                accent = tk.PhotoImage(master=window, width=width, height=height)
                window._inspector_analysis_accent = accent
            dark = getattr(window._root(), "_theme_variant", "light") == "dark"
            color = _blend(window, bg, nmr_analysis_accent(theme_id(window._root())), .55 if dark else .35)
            accent.put(color, to=(0, 0, width, height))
            for tab in analysis_tabs:
                notebook.tab(tab, image=accent, compound="left")
        style.configure("NMR.Experiments.Treeview",
                        rowheight=table_row_height(window, fonts["table"], fonts["table_rowheight"]) + 2)
        for name in ("NMR.Experiments.Treeview", "NMR.Inspector.Treeview", "NMR.Batch.Treeview"):
            style.configure(name, borderwidth=0, relief="flat")
            style.map(name, background=[("selected", selected)],
                      foreground=[("selected", selected_fg)])
            style.configure(name + ".Heading", background=bg, foreground=fg,
                            relief="flat", bordercolor=edge, lightcolor=bg, darkcolor=bg,
                            font=fonts["table_heading"])
            style.map(name + ".Heading", background=[("active", hover)])
        style.configure("NMR.Batch.Treeview.Heading", padding=(6, 3))
        style.configure("NMR.BatchHeader.TCheckbutton", padding=(3, 0), background=bg,
                        font=fonts["table_heading"])
        style.map("NMR.BatchHeader.TCheckbutton", background=[("!disabled", bg)])

        def visit(widget):
            if isinstance(widget, ttk.Combobox) and not widget.cget("style"):
                widget.configure(style="NMR.Compact.TCombobox")
            elif isinstance(widget, ttk.Entry) and not widget.cget("style"):
                widget.configure(style="NMR.Compact.TEntry")
            elif isinstance(widget, ttk.Button) and not widget.cget("style"):
                widget.configure(style="NMR.Secondary.TButton")
            elif isinstance(widget, ttk.Treeview) and not widget.cget("style"):
                widget.configure(style="NMR.Inspector.Treeview")
            elif isinstance(widget, ttk.LabelFrame) and not widget.cget("style"):
                widget.configure(style="NMR.Panel.TLabelframe")
            elif isinstance(widget, tk.Text):
                widget.configure(background=field, foreground=fg, insertbackground=fg,
                                 selectbackground=selected, selectforeground=selected_fg,
                                 highlightthickness=0)
            elif isinstance(widget, (tk.Canvas, tk.Frame, tk.Label, tk.Button, tk.Checkbutton)):
                widget.configure(background=bg)
                if isinstance(widget, (tk.Label, tk.Button, tk.Checkbutton)):
                    widget.configure(foreground=fg)
                if isinstance(widget, (tk.Button, tk.Checkbutton)):
                    widget.configure(activebackground=hover, activeforeground=fg,
                                     disabledforeground=disabled, highlightbackground=bg,
                                     relief="flat", borderwidth=0)
                if isinstance(widget, tk.Checkbutton):
                    widget.configure(selectcolor=field)
                if (isinstance(widget.master, NavigationToolbar2Tk)
                        and isinstance(widget, (tk.Button, tk.Checkbutton))
                        and getattr(widget, "_image_file", None)):
                    # Matplotlib chooses its contrasting icon when it creates
                    # a button, before this owner applies the current theme.
                    widget.master._set_image_for_button(widget)
            if isinstance(widget, (ttk.Button, ttk.Menubutton)) and str(widget.cget("style")).startswith("NMR."):
                if widget.cget("style") not in ("NMR.Section.TButton", "NMR.OpenSection.TButton"):
                    size_workbench_action(widget)
            for child in widget.winfo_children():
                visit(child)
        visit(window)

    def style_figure(self, figure):
        self.refresh()
        bg, fg, field = self.palette[:3]
        figure.set_facecolor(bg)
        for axes in figure.axes:
            axes.set_facecolor(field)
            axes.tick_params(axis="both", colors=fg)
            axes.xaxis.label.set_color(fg)
            axes.yaxis.label.set_color(fg)
            axes.xaxis.get_offset_text().set_color(fg)
            axes.yaxis.get_offset_text().set_color(fg)
            axes.title.set_color(fg)
            for text in axes.texts:
                if text.get_gid() == "nmr-empty-hint":
                    text.set_color(_blend(self.window, fg, bg, .30))
            for spine in axes.spines.values():
                spine.set_color(_blend(self.window, bg, fg, .30))


class WorkbenchFigureCanvas(FigureCanvasTkAgg):
    """Reapply plot chrome after Axes.clear(), using the normal draw path."""

    def draw(self):
        presentation = self.get_tk_widget().winfo_toplevel()._presentation
        presentation.style_figure(self.figure)
        super().draw()
