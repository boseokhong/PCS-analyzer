"""PCS color presentation and compact inspector; no calculated-value mutation."""
from logic.molecule_viewer_pcs import DEFAULTS, PALETTES, display_limits


def pcs_presentation(presentation, options):
    if presentation is None or presentation["color_mode"] != "PCS":
        return presentation
    result = dict(presentation)
    result.update(pcs_legend_visible=options["show_legend"], pcs_font_pt=options["legend_font"],
                  pcs_tick_step=options["tick_step"])
    values = presentation.get("pcs_values")
    if values is None:  # Old snapshots retain their authoritative host colors.
        return result
    limits = display_limits(presentation, options)
    if limits is not None and options["tick_step"] and (limits[1] - limits[0]) / options["tick_step"] > 63:
        result["pcs_tick_step"] = 0.
        result["pcs_tick_auto_fallback"] = True
    if not any(value is not None for value in values):
        if options["missing_color"] != DEFAULTS["missing_color"]:
            from matplotlib.colors import to_rgba
            result["atom_rgba"] = [list(to_rgba(options["missing_color"])) for _ in values]
        return result
    unchanged = all(options[key] == DEFAULTS[key] for key in
        ("range_mode", "centre_zero", "palette", "reverse", "missing_color"))
    if unchanged:
        return result  # Preserve exact legacy colors and symmetric Auto scale.
    from matplotlib import colormaps
    from matplotlib.colors import Normalize, to_rgba
    cmap = colormaps[options["palette"]]
    if options["reverse"]:
        cmap = cmap.reversed()
    norm = Normalize(*limits, clip=True)
    missing = (.75, .75, .75, 1.) if options["missing_color"] == DEFAULTS["missing_color"] else to_rgba(options["missing_color"])
    result["atom_rgba"] = [list(cmap(norm(value))) if value is not None else list(missing) for value in values]
    result["color_scale"] = dict(range=list(limits), title="PCS (ppm)",
                                  rgba=[list(cmap(index / 255)) for index in range(256)])
    return result


class PcsColorControls:
    def __init__(self, parent, widgets, core, gui, renderer, changed):
        from ui.molecule_viewer_style import InspectorSection
        from ui.molecule_viewer_color import ColorControl
        self.widgets, self.renderer, self.changed = widgets, renderer, changed
        self.gui, self.core = gui, core
        self.model = renderer.pcs_colors
        self.section = InspectorSection(parent, widgets, "PCS colors", core=core)
        self.widget, self.form = self.section.widget, self.section.form
        self.controls = {}
        def choice(key, title, values, tooltip):
            control = widgets.QComboBox(self.section.body)
            control.addItems(values)
            control.setToolTip(tooltip)
            self.form.addRow(title, control)
            self.controls[key] = (control, "setCurrentText")
            control.currentTextChanged.connect(lambda value: self.change(key, value))
            return control
        choice("range_mode", "Range", ("Auto", "Manual"), "Auto follows visible calculated PCS values; Manual clips colors to your limits. PCS values are unchanged.")
        for key, title in (("minimum", "Min (ppm)"), ("maximum", "Max (ppm)")):
            control = widgets.QLineEdit(self.section.body)
            control.setAlignment(core.Qt.AlignmentFlag.AlignRight)
            control.setToolTip("Color limit in ppm. Scientific notation is accepted. Values beyond the range use endpoint colors.")
            self.form.addRow(title, control)
            self.controls[key] = (control, "setText")
            control.editingFinished.connect(lambda key=key, control=control: self.numeric(key, control))
        def check(key, title, tooltip):
            control = widgets.QCheckBox(title, self.section.body)
            control.setToolTip(tooltip)
            self.form.addRow(control)
            self.controls[key] = (control, "setChecked")
            control.toggled.connect(lambda value: self.change(key, value))
        check("centre_zero", "Centre at zero", "Auto range uses symmetric limits around zero. Manual limits are used exactly as entered.")
        palette = choice("palette", "Palette", PALETTES, "Gradient previews use the same palette as atom colors and the legend.")
        self.palette_reverse = None
        palette.setIconSize(core.QSize(40, 12))
        check("reverse", "Reverse", "Reverse the palette direction without changing the PCS scale.")
        self.missing = ColorControl(self.section.body, widgets, gui, "Missing PCS color",
            lambda: self.model.values["missing_color"], lambda: self.model.generation,
            lambda value: self.change("missing_color", value))
        self.form.addRow("Missing PCS", self.missing.widget)
        check("show_legend", "Legend", "Show or hide the entire PCS legend. Figure Include → PCS legend can exclude it separately.")
        control = widgets.QLineEdit(self.section.body)
        control.setAlignment(core.Qt.AlignmentFlag.AlignRight)
        control.setToolTip("Tick interval in ppm. Auto or 0 uses automatic spacing. More than 64 ticks falls back to Auto.")
        self.form.addRow("Tick step", control)
        self.controls["tick_step"] = (control, "setText")
        control.editingFinished.connect(lambda: self.numeric("tick_step", self.controls["tick_step"][0]))
        control = widgets.QSpinBox(self.section.body)
        control.setRange(0, 96)
        control.setSpecialValueText("Auto")
        control.setSuffix(" pt")
        control.setToolTip("Legend title and tick font in points. Auto uses host plot fonts; Figure Label size can override output only.")
        self.form.addRow("Legend font", control)
        self.controls["legend_font"] = (control, "setValue")
        control.valueChanged.connect(lambda value: self.change("legend_font", value))
        self.note = widgets.QLabel(self.section.body)
        self.note.setWordWrap(True)
        self.form.addRow(self.note)
        self.section.add_reset(widgets, self.reset)
        self.sync()

    def apply_icons(self, assets):
        from ui.molecule_viewer_style import compact_button
        compact_button(assets, self.section.reset_button, "reset", "Reset PCS colors",
                       "Reset PCS colors and legend only. Calculated PCS, camera and other display settings are retained.")

    def update_palette_preview(self, reverse):
        if self.palette_reverse == reverse:
            return
        from matplotlib import colormaps
        gui = self.gui
        for index, name in enumerate(PALETTES):
            pixmap = gui.QPixmap(40, 12)
            gradient = gui.QLinearGradient(0, 0, 40, 0)
            cmap = colormaps[name].reversed() if reverse else colormaps[name]
            for sample in range(17):
                gradient.setColorAt(sample / 16, gui.QColor.fromRgbF(*cmap(sample / 16)))
            painter = gui.QPainter(pixmap)
            try:
                painter.fillRect(pixmap.rect(), gui.QBrush(gradient))
            finally:
                painter.end()
            self.controls["palette"][0].setItemIcon(index, gui.QIcon(pixmap))
        self.palette_reverse = reverse

    def sync(self):
        presentation = (self.renderer.scene or {}).get("presentation")
        visible = presentation is not None and presentation["color_mode"] == "PCS"
        self.widget.setVisible(visible)
        options = self.model.options()
        limits = display_limits(presentation, options) if visible else None
        for key, (control, setter) in self.controls.items():
            value = options[key]
            if setter == "setText":
                if control.hasFocus() and control.isModified():
                    continue
                if key in ("minimum", "maximum") and options["range_mode"] == "Auto" and limits:
                    value = limits[key == "maximum"]
                value = format(value, ".8g")
                if key == "tick_step" and options[key] == 0:
                    value = "Auto"
            blocked = control.blockSignals(True)
            getattr(control, setter)(value)
            control.blockSignals(blocked)
        manual = options["range_mode"] == "Manual"
        for key in ("minimum", "maximum"):
            self.controls[key][0].setEnabled(manual)
        self.controls["centre_zero"][0].setEnabled(not manual)
        self.controls["centre_zero"][0].setVisible(not manual)
        for key in ("tick_step", "legend_font"):
            self.controls[key][0].setEnabled(options["show_legend"])
        self.missing.sync()
        self.update_palette_preview(options["reverse"])
        self.section.summary(options["palette"] + " · " + options["range_mode"])
        available = visible and any(value is not None for value in presentation.get("pcs_values", []))
        fallback = visible and pcs_presentation(presentation, options).get("pcs_tick_auto_fallback")
        self.note.setText("No calculated PCS values." if not available else "Tick step too small for this range; using Auto." if fallback else "")
        self.note.setVisible(not available or bool(fallback))

    def numeric(self, key, control):
        if not control.isModified():
            return
        control.setModified(False)
        try:
            text = control.text().strip()
            self.change(key, 0. if key == "tick_step" and text.lower() in ("auto", "") else float(text))
        except ValueError:
            self.widgets.QMessageBox.warning(self.widget, "PCS colors", "Enter a finite number; scientific notation is accepted.")
            self.sync()

    def change(self, key, value):
        previous = self.model.options()
        candidate = dict(previous, **{key: value})
        if key == "range_mode" and value == "Manual":
            presentation = (self.renderer.scene or {}).get("presentation")
            limits = display_limits(presentation, previous) if presentation else None
            if limits:
                candidate.update(minimum=limits[0], maximum=limits[1])
        self.apply_options(candidate)

    def apply_options(self, candidate):
        previous = self.model.options()
        attempted = False
        try:
            self.model.restore(candidate)
            attempted = True
            if self.changed() is False:
                raise RuntimeError("The PCS display could not be updated.")
        except Exception as error:
            self.model.restore(previous)
            if attempted:
                try:
                    self.changed()
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception("PCS display rollback rendering failed")
            self.widgets.QMessageBox.warning(self.widget, "PCS colors", str(error)[:768])
        self.sync()

    def reset(self):
        self.apply_options(DEFAULTS)
