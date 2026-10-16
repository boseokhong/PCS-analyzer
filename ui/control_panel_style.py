"""Control Panel-only metrics, styles, and compact presentation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

from ui.ribbon.icon_service import tk_scale_factor
from ui.style import get_app_fonts
from ui.theme_palette import panel_palette, theme_id
from ui.widget_style import button_skin, selection_skin, presentation_palette, compact_button_skin


PANEL_LOGICAL_WIDTH = 300


@dataclass(frozen=True)
class ControlPanelMetrics:
    """Logical metrics for the narrow scientific inspector."""

    scale: float
    panel_width: int
    outer_x: int
    outer_y: int
    section_gap: int
    header_x: int
    header_y: int
    body_x: int
    body_y: int
    row_y: int
    column_gap: int
    action_gap: int
    help_wrap: int
    gutter_hit: int

    @classmethod
    def from_widget(cls, widget) -> "ControlPanelMetrics":
        scale = tk_scale_factor(widget.winfo_toplevel())
        # The inspector's labels, units and numeric fields grow with UI text.
        # Reserve matching horizontal space instead of shrinking the fields.
        font_size = abs(int(get_app_fonts(widget)["ui"][1]))
        text_scale = max(1.0, font_size / 9.0)

        def px(value: float, minimum: int = 1) -> int:
            return max(minimum, int(round(value * scale)))

        return cls(
            scale=scale,
            panel_width=px(PANEL_LOGICAL_WIDTH * text_scale),
            outer_x=px(4),
            outer_y=px(5),
            section_gap=px(4),
            header_x=px(7),
            header_y=px(4),
            body_x=px(8),
            body_y=px(6),
            row_y=px(2),
            column_gap=px(5),
            action_gap=px(4),
            help_wrap=px(275),
            gutter_hit=px(10),
        )


def control_panel_palette(widget) -> dict[str, str]:
    return panel_palette(theme_id(widget))


def _configure_named_hierarchy_font(
    widget,
    name: str,
    base_font,
    *,
    size_delta: int = 0,
):
    """Create/update one persistent named font used by Control Panel hierarchy.

    ttk styles may outlive temporary ``tkinter.font.Font`` wrappers.  Using a
    stable Tcl font name avoids a style silently falling back after a temporary
    font object is replaced or garbage-collected during theme/font refreshes.
    """
    root = widget.winfo_toplevel()
    names = set(root.tk.call("font", "names"))
    if name not in names:
        # Create the Tcl named font directly.  ``tkfont.Font(..., exists=False)``
        # owns the font and deletes it when that Python wrapper is collected;
        # that is exactly what made the previous hierarchy font disappear after
        # style refreshes.  ``nametofont`` returns a non-owning wrapper.
        root.tk.call("font", "create", name)
    font = tkfont.nametofont(name, root=root)

    source = tkfont.Font(root=root, font=base_font)
    actual = source.actual()
    size = int(actual.get("size", 9) or 9)
    if size_delta:
        # Negative Tk font sizes are pixel sizes.  Increasing visual size means
        # increasing the absolute value while preserving the sign.
        size = size - size_delta if size < 0 else size + size_delta

    font.configure(
        family=actual.get("family"),
        size=size,
        weight="bold",
        slant=actual.get("slant", "roman"),
        underline=int(actual.get("underline", 0)),
        overstrike=int(actual.get("overstrike", 0)),
    )
    return font


def control_panel_hierarchy_fonts(widget):
    """Return stable named fonts for accordion headers and body subheadings."""
    root = widget.winfo_toplevel()
    fonts = get_app_fonts(widget)
    header = _configure_named_hierarchy_font(
        widget,
        "PCSControlPanelHeaderFont",
        fonts["ui_bold"],
        size_delta=0,
    )
    subheading = _configure_named_hierarchy_font(
        widget,
        "PCSControlPanelSubheadingFont",
        fonts["ui_bold"],
        size_delta=0,
    )
    variants = {"header": header, "subheading": subheading}
    # Keep Python wrappers alive as well as the persistent Tcl font names.
    root._control_panel_hierarchy_fonts = variants
    return variants


def configure_control_panel_styles(widget) -> None:
    """Configure scoped Control Panel styles without changing global defaults."""
    style = ttk.Style(widget)
    palette = control_panel_palette(widget)
    fonts = get_app_fonts(widget)
    hierarchy_fonts = control_panel_hierarchy_fonts(widget)

    style.configure("ControlPanel.Root.TFrame", background=palette["panel"], padding=0)
    style.configure(
        "ControlPanel.Title.TLabel",
        background=palette["panel"],
        foreground=palette["foreground"],
        font=fonts["section_large"],
        padding=0,
    )
    style.configure(
        "ControlPanel.Section.TFrame",
        background=palette["body"],
        borderwidth=0,
        relief="flat",
        padding=0,
    )
    style.configure("ControlPanel.Body.TFrame", background=palette["body"], padding=0)
    style.configure("ControlPanel.Form.TFrame", background=palette["body"], padding=0)
    style.configure("ControlPanel.Header.TFrame", background=palette["header"], padding=0)
    style.configure(
        "ControlPanel.Header.TLabel",
        background=palette["header"],
        foreground=palette["foreground"],
        font=hierarchy_fonts["header"],
        padding=0,
    )
    style.configure(
        "ControlPanel.Handle.TLabel",
        background=palette["header"],
        foreground=palette["muted"],
        font=fonts["ui"],
        padding=0,
    )
    style.configure(
        "ControlPanel.Chevron.TButton",
        background=palette["header"],
        foreground=palette["foreground"],
        font=fonts["ui_bold"],
        padding=(2, 0),
        relief="flat",
        borderwidth=0,
    )
    style.map(
        "ControlPanel.Chevron.TButton",
        background=[("pressed", palette["header_hover"]), ("active", palette["header_hover"])],
    )
    style.configure(
        "ControlPanel.HeaderUtility.TButton",
        background=palette["header"],
        foreground=palette["foreground"],
        font=fonts["ui_small"],
        padding=(3, 1),
        relief="flat",
        borderwidth=0,
    )
    style.map(
        "ControlPanel.HeaderUtility.TButton",
        background=[("pressed", palette["header_hover"]), ("active", palette["header_hover"])],
        foreground=[("disabled", palette["muted"])],
    )
    style.configure(
        "ControlPanel.TitleUtility.TButton",
        background=palette["panel"],
        foreground=palette["muted"],
        font=fonts["ui_bold"],
        padding=(4, 1),
        relief="flat",
        borderwidth=0,
    )
    style.map(
        "ControlPanel.TitleUtility.TButton",
        background=[("pressed", palette["header_hover"]), ("active", palette["header_hover"])],
        foreground=[("active", palette["foreground"]), ("disabled", palette["muted"])],
    )
    style.configure(
        "ControlPanel.TLabel",
        background=palette["body"],
        foreground=palette["foreground"],
        font=fonts["ui"],
        padding=0,
    )
    style.configure(
        "ControlPanel.Muted.TLabel",
        background=palette["body"],
        foreground=palette["muted"],
        font=fonts["ui_small"],
        padding=0,
    )
    style.configure(
        "ControlPanel.Subheading.TLabel",
        background=palette["body"],
        foreground=palette["subheading"],
        font=hierarchy_fonts["subheading"],
        padding=0,
    )
    style.configure(
        "ControlPanel.ElementSummary.TLabel",
        background=palette["body"],
        foreground=palette["muted"],
        font=fonts["ui_small"],
        padding=0,
    )
    style.configure(
        "ControlPanel.ElementSummaryStrong.TLabel",
        background=palette["body"],
        foreground=palette["foreground"],
        font=fonts["ui_small_bold"],
        padding=0,
    )
    style.configure(
        "ControlPanel.ElementUtility.TButton",
        background=palette["body"],
        foreground=palette["muted"],
        font=fonts["ui_small"],
        padding=(4, 1),
        relief="flat",
        borderwidth=0,
    )
    style.map(
        "ControlPanel.ElementUtility.TButton",
        background=[("pressed", palette["header_hover"]), ("active", palette["header_hover"])],
        foreground=[("active", palette["foreground"]), ("disabled", palette["muted"])],
    )
    style.configure(
        "ControlPanel.Unit.TLabel",
        background=palette["body"],
        foreground=palette["muted"],
        font=fonts["ui_small"],
        padding=0,
    )
    style.configure(
        "ControlPanel.TCheckbutton",
        background=palette["body"],
        foreground=palette["foreground"],
        font=fonts["ui"],
        padding=0,
    )
    style.map(
        "ControlPanel.TCheckbutton",
        background=[("active", palette["body"]), ("selected", palette["body"])],
        foreground=[("disabled", palette["muted"])],
    )
    style.configure(
        "ControlPanel.TRadiobutton",
        background=palette["body"],
        foreground=palette["foreground"],
        font=fonts["ui_small"],
        padding=0,
    )
    style.map(
        "ControlPanel.TRadiobutton",
        background=[("active", palette["body"]), ("selected", palette["body"])],
        foreground=[("disabled", palette["muted"])],
    )
    style.configure("ControlPanel.Secondary.TButton", padding=(5, 1), font=fonts["ui_small"])
    style.configure(
        "ControlPanel.Action.TButton",
        padding=(7, 2),
        font=fonts["ui_bold"],
        background=palette["primary"],
        foreground=palette["primary_foreground"],
        borderwidth=0,
    )
    style.map(
        "ControlPanel.Action.TButton",
        background=[
            ("pressed", palette["primary_pressed"]),
            ("active", palette["primary_hover"]),
        ],
        foreground=[("disabled", palette["muted"])],
    )
    style.configure(
        "ControlPanel.Danger.TButton",
        padding=(7, 2),
        font=fonts["ui"],
        foreground=palette["danger"],
        background=palette["danger_bg"],
        borderwidth=0,
    )
    style.map(
        "ControlPanel.Danger.TButton",
        background=[("pressed", palette["danger_hover"]), ("active", palette["danger_hover"])],
        foreground=[("disabled", palette["muted"])],
    )
    style.configure("ControlPanel.TSeparator", background=palette["separator"])
    style.configure(
        "ControlPanel.Placeholder.TEntry",
        fieldbackground=palette["entry"],
        foreground=palette["muted"],
        font=fonts["ui"],
    )

    root = widget._root()
    shared = presentation_palette(theme_id(root))
    shared.update(text=palette["foreground"], muted=palette["muted"])
    for name, role in (("Action", "Primary"), ("Secondary", "Secondary"),
                       ("Danger", "Danger"), ("ElementUtility", "Utility")):
        button_skin(root, style, "ControlPanel." + name + ".TButton", shared,
                    role, parent=palette["body"])
    for name in ("Header", "Chevron", "HeaderUtility", "TitleUtility"):
        compact_button_skin(root, style, "ControlPanel." + name + ".TButton", shared)
    for kind in ("TCheckbutton", "TRadiobutton"):
        selection_skin(root, style, "ControlPanel." + kind, shared, palette["body"])


class ControlPanelElementChip(tk.Canvas):
    """Compact element visibility toggle with count and restrained CPK identity."""

    def __init__(
        self,
        parent,
        *,
        element: str,
        count: int,
        variable,
        cpk_color: str,
        command=None,
        metrics: ControlPanelMetrics | None = None,
    ) -> None:
        self.metrics = metrics or ControlPanelMetrics.from_widget(parent)
        self.element = str(element)
        self.count = int(count)
        self.variable = variable
        self.cpk_color = str(cpk_color)
        self.command = command
        self._palette = control_panel_palette(parent)
        self._hover = False
        self._focused = False
        self._fonts = get_app_fonts(parent)

        height = max(24, int(round(27 * self.metrics.scale)))
        width = max(54, int(round(60 * self.metrics.scale)))
        super().__init__(
            parent,
            width=width,
            height=height,
            background=self._palette["body"],
            highlightthickness=0,
            borderwidth=0,
            relief="flat",
            cursor="hand2",
            takefocus=1,
        )

        self._variable_trace = self.variable.trace_add("write", self._variable_changed)
        self.bind("<Configure>", lambda _event: self._draw(), add="+")
        self.bind("<Enter>", self._enter, add="+")
        self.bind("<Leave>", self._leave, add="+")
        self.bind("<FocusIn>", self._focus_in, add="+")
        self.bind("<FocusOut>", self._focus_out, add="+")
        self.bind("<Button-1>", self._activate, add="+")
        self.bind("<space>", self._activate, add="+")
        self.bind("<Return>", self._activate, add="+")
        self.after_idle(self._draw)

    def destroy(self):
        try:
            self.variable.trace_remove("write", self._variable_trace)
        except Exception:
            pass
        super().destroy()

    def apply_theme(self, palette: dict[str, str] | None = None) -> None:
        self._palette = palette or control_panel_palette(self)
        self._fonts = get_app_fonts(self)
        super().configure(background=self._palette["body"])
        self._draw()

    def _variable_changed(self, *_args) -> None:
        self.after_idle(self._draw)

    def _enter(self, _event=None) -> None:
        self._hover = True
        self._draw()

    def _leave(self, _event=None) -> None:
        self._hover = False
        self._draw()

    def _focus_in(self, _event=None) -> None:
        self._focused = True
        self._draw()

    def _focus_out(self, _event=None) -> None:
        self._focused = False
        self._draw()

    def _activate(self, _event=None):
        self.focus_set()
        self.variable.set(not bool(self.variable.get()))
        if callable(self.command):
            self.command()
        return "break"

    def _draw(self) -> None:
        if not self.winfo_exists():
            return
        self.delete("all")
        palette = self._palette
        selected = bool(self.variable.get())
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        scale = self.metrics.scale
        inset = max(1, int(round(scale)))

        if selected:
            fill = palette["chip_selected_hover"] if self._hover else palette["chip_selected_bg"]
            border = (
                palette["chip_hover_border"]
                if (self._hover or self._focused)
                else palette["chip_selected_border"]
            )
            text_color = palette["foreground"]
            count_color = palette["chip_count"]
        else:
            fill = palette["header_hover"] if self._hover else palette["chip_unselected_bg"]
            border = (
                palette["chip_hover_border"]
                if (self._hover or self._focused)
                else palette["chip_unselected_border"]
            )
            text_color = palette["chip_unselected_text"]
            count_color = palette["chip_unselected_text"]

        self.create_rectangle(
            inset,
            inset,
            width - inset,
            height - inset,
            fill=fill,
            outline=border,
            width=max(1, int(round(scale))),
        )

        cy = height / 2.0
        dot_r = max(3, int(round(3.5 * scale)))
        dot_x = max(8, int(round(10 * scale)))
        if selected:
            self.create_oval(
                dot_x - dot_r,
                cy - dot_r,
                dot_x + dot_r,
                cy + dot_r,
                fill=self.cpk_color,
                outline=palette["chip_dot_outline"],
                width=max(1, int(round(scale))),
            )
        else:
            self.create_oval(
                dot_x - dot_r,
                cy - dot_r,
                dot_x + dot_r,
                cy + dot_r,
                fill=fill,
                outline=palette["chip_dot_off"],
                width=max(1, int(round(scale))),
            )

        symbol_x = max(17, int(round(20 * scale)))
        self.create_text(
            symbol_x,
            cy,
            text=self.element,
            anchor="w",
            fill=text_color,
            font=self._fonts["ui_bold"],
        )
        self.create_text(
            width - max(6, int(round(7 * scale))),
            cy,
            text=str(self.count),
            anchor="e",
            fill=count_color,
            font=self._fonts["ui_small"],
        )

class ControlPanelSlider(tk.Canvas):
    """Compact, theme-aware slider with a clearly visible custom thumb."""

    def __init__(
        self,
        parent,
        *,
        variable,
        from_: float,
        to: float,
        command=None,
        resolution: float = 0.1,
        metrics: ControlPanelMetrics | None = None,
        length: int | None = None,
    ) -> None:
        self.metrics = metrics or ControlPanelMetrics.from_widget(parent)
        self._scale_factor = self.metrics.scale
        self._variable = variable
        self._from = float(from_)
        self._to = float(to)
        self._command = command
        self._resolution = abs(float(resolution)) if resolution else 0.0
        self._palette = control_panel_palette(parent)
        self._hover = False
        self._dragging = False
        self._pcs_control_slider = True

        height = max(18, int(round(18 * self._scale_factor)))
        request_width = (
            int(length)
            if length is not None
            else max(88, int(round(112 * self._scale_factor)))
        )
        super().__init__(
            parent,
            width=request_width,
            height=height,
            background=self._palette["body"],
            highlightthickness=0,
            borderwidth=0,
            relief="flat",
            takefocus=1,
            cursor="hand2",
        )

        self._variable_trace = self._variable.trace_add("write", self._variable_changed)
        self.bind("<Configure>", lambda _event: self._draw(), add="+")
        self.bind("<Enter>", self._enter, add="+")
        self.bind("<Leave>", self._leave, add="+")
        self.bind("<Button-1>", self._press, add="+")
        self.bind("<B1-Motion>", self._drag, add="+")
        self.bind("<ButtonRelease-1>", self._release, add="+")
        self.bind("<Left>", lambda _event: self._step(-1), add="+")
        self.bind("<Right>", lambda _event: self._step(1), add="+")
        self.bind("<Home>", lambda _event: self._set_value(self._from, notify=True), add="+")
        self.bind("<End>", lambda _event: self._set_value(self._to, notify=True), add="+")
        self.after_idle(self._draw)

    def destroy(self):
        try:
            self._variable.trace_remove("write", self._variable_trace)
        except Exception:
            pass
        super().destroy()

    def configure(self, cnf=None, **kwargs):
        result = super().configure(cnf, **kwargs)
        if kwargs and "state" in kwargs:
            self.after_idle(self._draw)
        return result

    config = configure

    def apply_theme(self, palette: dict[str, str] | None = None) -> None:
        self._palette = palette or control_panel_palette(self)
        super().configure(background=self._palette["body"])
        self._draw()

    def _is_disabled(self) -> bool:
        try:
            return str(self.cget("state")) == str(tk.DISABLED)
        except Exception:
            return False

    def _variable_changed(self, *_args) -> None:
        self.after_idle(self._draw)

    def _normalized_value(self, value: float) -> float:
        lo = min(self._from, self._to)
        hi = max(self._from, self._to)
        value = min(hi, max(lo, float(value)))
        if self._resolution > 0:
            value = self._from + round((value - self._from) / self._resolution) * self._resolution
            value = min(hi, max(lo, value))
        return value

    def _set_value(self, value: float, *, notify: bool) -> None:
        if self._is_disabled():
            return
        value = self._normalized_value(value)
        self._variable.set(value)
        if notify and callable(self._command):
            self._command(f"{value:.12g}")
        self._draw()

    def _value_from_x(self, x: float) -> float:
        width = max(1, self.winfo_width())
        radius = max(6, int(round(6 * self._scale_factor)))
        left = radius + max(2, int(round(2 * self._scale_factor)))
        right = max(left + 1, width - left)
        fraction = (float(x) - left) / max(1, right - left)
        fraction = min(1.0, max(0.0, fraction))
        return self._from + fraction * (self._to - self._from)

    def _press(self, event):
        if self._is_disabled():
            return "break"
        start = getattr(self, "_edit_gesture_start", None)
        if callable(start):
            start()
        self.focus_set()
        self._dragging = True
        self._set_value(self._value_from_x(event.x), notify=True)
        return "break"

    def _drag(self, event):
        if self._dragging and not self._is_disabled():
            self._set_value(self._value_from_x(event.x), notify=True)
        return "break"

    def _release(self, _event):
        self._dragging = False
        end = getattr(self, "_edit_gesture_end", None)
        if callable(end):
            end()
        self._draw()
        return "break"

    def _enter(self, _event):
        self._hover = True
        self._draw()

    def _leave(self, _event):
        self._hover = False
        if not self._dragging:
            self._draw()

    def _step(self, direction: int):
        if self._is_disabled():
            return "break"
        try:
            current = float(self._variable.get())
        except Exception:
            current = self._from
        step = self._resolution if self._resolution > 0 else abs(self._to - self._from) / 100.0
        self._set_value(current + direction * step, notify=True)
        return "break"

    def _draw(self) -> None:
        if not self.winfo_exists():
            return
        self.delete("all")
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        y = height / 2.0
        radius = max(5, int(round(6 * self._scale_factor)))
        left = radius + max(2, int(round(2 * self._scale_factor)))
        right = max(left + 1, width - left)
        track_width = max(2, int(round(3 * self._scale_factor)))

        self.create_line(
            left,
            y,
            right,
            y,
            fill=self._palette["slider_track"],
            width=track_width,
            capstyle=tk.ROUND,
        )

        try:
            value = float(self._variable.get())
        except Exception:
            value = self._from
        span = self._to - self._from
        fraction = 0.0 if span == 0 else (value - self._from) / span
        fraction = min(1.0, max(0.0, fraction))
        x = left + fraction * (right - left)

        if self._is_disabled():
            thumb = self._palette["slider_thumb_disabled"]
        elif self._hover or self._dragging:
            thumb = self._palette["slider_thumb_hover"]
        else:
            thumb = self._palette["slider_thumb"]

        self.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=thumb,
            outline=self._palette["slider_outline"],
            width=max(1, int(round(self._scale_factor))),
        )


class ControlPanelForm:
    """A presentation-only three-column form builder for inspector sections."""

    def __init__(self, parent, metrics: ControlPanelMetrics):
        self.metrics = metrics
        self.frame = ttk.Frame(parent, style="ControlPanel.Form.TFrame")
        self.frame.pack(fill="x")
        self.frame.columnconfigure(0, minsize=int(round(94 * metrics.scale)))
        self.frame.columnconfigure(1, weight=1)
        self.row = 0

    def _next(self) -> int:
        row = self.row
        self.row += 1
        return row

    def add_control(
        self,
        label: str,
        control,
        unit: str = "",
        *,
        unitvariable=None,
        label_style="ControlPanel.TLabel",
    ):
        row = self._next()
        label_widget = ttk.Label(self.frame, text=label, style=label_style, anchor="w")
        label_widget.grid(row=row, column=0, sticky="w", pady=self.metrics.row_y)
        control.grid(
            row=row,
            column=1,
            sticky="ew",
            padx=(self.metrics.column_gap, 0),
            pady=self.metrics.row_y,
        )
        unit_widget = None
        if unit or unitvariable is not None:
            unit_widget = ttk.Label(
                self.frame,
                text=unit,
                textvariable=unitvariable,
                style="ControlPanel.Unit.TLabel",
                anchor="w",
            )
            unit_widget.grid(
                row=row,
                column=2,
                sticky="w",
                padx=(self.metrics.column_gap, 0),
                pady=self.metrics.row_y,
            )
        return label_widget, unit_widget

    def numeric(self, label: str, *, unit: str = "", width: int = 8, textvariable=None):
        entry = ttk.Entry(self.frame, width=width, textvariable=textvariable, justify="right")
        label_widget, unit_widget = self.add_control(label, entry, unit)
        return entry, label_widget, unit_widget

    def checkbox(self, text: str, *, variable, command=None):
        row = self._next()
        check = ttk.Checkbutton(
            self.frame,
            text=text,
            variable=variable,
            command=command,
            style="ControlPanel.TCheckbutton",
        )
        check.grid(row=row, column=0, columnspan=3, sticky="w", pady=self.metrics.row_y)
        return check

    def slider(
        self,
        label: str,
        *,
        variable,
        from_: float,
        to: float,
        command,
        unit: str = "°",
        entry_width: int = 7,
        entry_textvariable=None,
    ):
        controls = ttk.Frame(self.frame, style="ControlPanel.Form.TFrame")
        controls.columnconfigure(0, weight=1)
        slider = ControlPanelSlider(
            controls,
            from_=from_,
            to=to,
            variable=variable,
            resolution=0.1,
            command=command,
            metrics=self.metrics,
        )
        slider.grid(row=0, column=0, sticky="ew")
        entry = ttk.Entry(
            controls,
            width=entry_width,
            textvariable=entry_textvariable,
            justify="right",
        )
        entry.grid(row=0, column=1, padx=(self.metrics.column_gap, 0))
        label_widget, unit_widget = self.add_control(label, controls, unit)
        return slider, entry, label_widget, unit_widget

    def heading(self, text: str):
        row = self._next()
        label = ttk.Label(
            self.frame,
            text=text,
            style="ControlPanel.Subheading.TLabel",
            font=control_panel_hierarchy_fonts(self.frame)["subheading"],
            anchor="w",
        )
        label.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(self.metrics.row_y + max(2, int(round(2 * self.metrics.scale))), self.metrics.row_y),
        )
        return label

    def heading_row(self, text: str):
        """Subheading with a compact action area on the right."""
        row = self._next()
        frame = ttk.Frame(self.frame, style="ControlPanel.Form.TFrame")
        frame.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(self.metrics.row_y + max(2, int(round(2 * self.metrics.scale))), self.metrics.row_y),
        )
        frame.columnconfigure(0, weight=1)
        label = ttk.Label(
            frame,
            text=text,
            style="ControlPanel.Subheading.TLabel",
            font=control_panel_hierarchy_fonts(frame)["subheading"],
            anchor="w",
        )
        label.grid(row=0, column=0, sticky="w")
        actions = ttk.Frame(frame, style="ControlPanel.Form.TFrame")
        actions.grid(row=0, column=1, sticky="e", padx=(self.metrics.column_gap, 0))
        return label, actions

    def status_action(
        self,
        label: str,
        *,
        textvariable,
        action_text: str,
        command,
    ):
        """Three-part status row: label | flexible status | fixed right action."""
        row = self._next()
        label_widget = ttk.Label(
            self.frame,
            text=label,
            style="ControlPanel.TLabel",
            anchor="w",
        )
        label_widget.grid(
            row=row, column=0, sticky="w", pady=self.metrics.row_y
        )
        value_widget = ttk.Label(
            self.frame,
            textvariable=textvariable,
            style="ControlPanel.TLabel",
            anchor="w",
        )
        value_widget.grid(
            row=row,
            column=1,
            sticky="ew",
            padx=(self.metrics.column_gap, self.metrics.column_gap),
            pady=self.metrics.row_y,
        )
        action = ttk.Button(
            self.frame,
            text=action_text,
            style="ControlPanel.Secondary.TButton",
            command=command,
        )
        action.grid(
            row=row, column=2, sticky="e", pady=self.metrics.row_y
        )
        return value_widget, label_widget, action

    def result(self, label: str, *, textvariable=None, text="", unit: str = "", unitvariable=None):
        value = ttk.Label(
            self.frame,
            text=text,
            textvariable=textvariable,
            style="ControlPanel.TLabel",
            anchor="e",
        )
        label_widget, unit_widget = self.add_control(
            label, value, unit, unitvariable=unitvariable
        )
        return value, label_widget, unit_widget

    def legacy_result(self, text: str):
        """Add a compact result whose full text remains a legacy data contract."""
        row = self._next()
        value = ttk.Label(
            self.frame,
            text=text,
            style="ControlPanel.TLabel",
            anchor="w",
        )
        value.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=self.metrics.row_y,
        )
        return value

    def help(self, text: str, *, textvariable=None):
        row = self._next()
        label = ttk.Label(
            self.frame,
            text=text,
            textvariable=textvariable,
            style="ControlPanel.Muted.TLabel",
            anchor="w",
            justify="left",
            wraplength=self.metrics.help_wrap,
        )
        label.grid(row=row, column=0, columnspan=3, sticky="ew", pady=self.metrics.row_y)
        return label

    def actions(self, *, align: str = "center"):
        row = self._next()
        frame = ttk.Frame(self.frame, style="ControlPanel.Form.TFrame")
        sticky = {"left": "w", "right": "e", "center": ""}.get(align, "")
        frame.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky=sticky,
            pady=(self.metrics.row_y + 1, 0),
        )
        return frame


__all__ = [
    "ControlPanelElementChip",
    "ControlPanelForm",
    "ControlPanelMetrics",
    "ControlPanelSlider",
    "PANEL_LOGICAL_WIDTH",
    "configure_control_panel_styles",
    "control_panel_hierarchy_fonts",
    "control_panel_palette",
]
