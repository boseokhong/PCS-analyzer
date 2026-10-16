"""Child-local surface options, independent of scientific scene/persistence."""

import math


def camera_density(camera, height, aspect=1.):
    """Projected pixels per world unit at the focal plane, without DPI factors."""
    from ui.molecule_viewer_camera import vertical_view_factor
    if not math.isfinite(height) or height <= 0:
        return None
    if camera.GetParallelProjection():
        half_height = camera.GetParallelScale()
    else:
        distance = math.dist(camera.GetPosition(), camera.GetFocalPoint())
        half_height = distance * vertical_view_factor(camera.GetViewAngle(),
            bool(camera.GetUseHorizontalViewAngle()), aspect)
    if not math.isfinite(half_height) or half_height <= 0:
        return None
    return height / (2 * half_height)


def projected_line_width(diameter, camera, height, aspect=1.):
    density = camera_density(camera, height, aspect)
    return max(.1, min(128., diameter * density)) if density is not None else None


class BondLineSizing:
    """Keep line thickness proportional to zoom relative to the initial view."""
    def __init__(self):
        self.document_id = None
        self.baseline = None

    def document(self, document_id):
        if self.document_id != document_id:
            self.document_id, self.baseline = document_id, None

    def calibrate(self, camera, renderer):
        if self.baseline is None:
            self.baseline = camera_density(camera, renderer.GetSize()[1], renderer.GetTiledAspectRatio())

    def diameter(self, base_width):
        return base_width / self.baseline if self.baseline else None


FINISHES = ("Default", "Matte", "Glossy")
DETAILS = {"Low": 16, "Standard": 32, "High": 64}
MATERIAL_FIELDS = ("Ambient", "Diffuse", "Specular", "SpecularPower")
OPTION_KEYS = ("atom_finish", "bond_finish", "bond_color", "background_color", "detail", "smooth", "bond_mode",
               "line_width", "bond_radius", "preset", "ambient", "diffuse",
               "specular", "specular_power", "light_intensity", "anti_aliasing")


class ViewportAppearance:
    def __init__(self):
        self.document_id = None
        self.document_generation = 0
        self.baselines = {}
        self.reset()

    def reset(self):
        self.atom_finish = self.bond_finish = "Default"
        self.detail = "Standard"
        self.smooth = True
        self.bond_mode = "Line"
        self.line_width = 4.
        self.bond_color = "#808080"
        self.background_color = "#FFFFFF"
        self.bond_radius = .08
        self.preset = "Basic"
        self.ambient = self.diffuse = self.specular = self.specular_power = -1.
        self.light_intensity = 1.
        self.anti_aliasing = "Default"

    def options(self):
        return {key: getattr(self, key) for key in OPTION_KEYS}

    def restore(self, options):
        for key, value in options.items():
            setattr(self, key, value)

    def set_preset(self, preset):
        if preset not in ("Basic", "Figure", "Custom"):
            raise ValueError("Unknown appearance preset")
        if preset != "Custom":
            self.reset()
        if preset == "Figure":
            self.detail = "High"
            self.bond_mode = "Cylinder"
            self.atom_finish = self.bond_finish = "Matte"
            self.ambient, self.diffuse = .18, .82
            self.specular, self.specular_power = .08, 30.
            self.anti_aliasing = "FXAA"
        self.preset = preset

    def document(self, document_id):
        if self.document_id != document_id:
            self.document_id = document_id
            self.document_generation += 1
            self.baselines.clear()
            self.reset()

    def forget(self, actor):
        self.baselines.pop(actor.GetAddressAsString(""), None)

    def material(self, actor, finish, *, surface=True):
        prop = actor.GetProperty()
        key = actor.GetAddressAsString("")
        if key not in self.baselines:
            self.baselines[key] = tuple(getattr(prop, "Get" + field)() for field in MATERIAL_FIELDS)
        values = self.baselines[key] if finish == "Default" else {
            "Matte": (.1, .9, 0., 1.), "Glossy": (.1, .8, .35, 40.)}[finish]
        for field, value in zip(MATERIAL_FIELDS, values):
            override = getattr(self, {"Ambient": "ambient", "Diffuse": "diffuse",
                                     "Specular": "specular", "SpecularPower": "specular_power"}[field])
            if override >= 0:
                value = override
            getattr(prop, "Set" + field)(value)
        if surface:
            if self.smooth:
                prop.SetInterpolationToPhong()
            else:
                prop.SetInterpolationToFlat()


class ViewportRendering:
    """Capture and restore child-local light/AA defaults without window resizing."""
    def __init__(self, plotter):
        self.plotter = plotter
        self.samples = plotter.render_window.GetMultiSamples()
        self.fxaa = plotter.renderer.GetUseFXAA()
        self.lights = [(light, light.GetIntensity()) for light in plotter.renderer.lights]
        self.mode = "Default"

    def apply(self, appearance):
        mode = appearance.anti_aliasing
        if mode not in ("Default", "Off", "FXAA"):
            raise ValueError("Unknown anti-aliasing mode")
        if mode != self.mode:
            window, renderer = self.plotter.render_window, self.plotter.renderer
            old_samples, old_fxaa = window.GetMultiSamples(), renderer.GetUseFXAA()
            try:
                window.SetMultiSamples(self.samples if mode == "Default" else 0)
                renderer.SetUseFXAA(self.fxaa if mode == "Default" else mode == "FXAA")
            except Exception:
                window.SetMultiSamples(old_samples)
                renderer.SetUseFXAA(old_fxaa)
                raise
            self.mode = mode
        for light, intensity in self.lights:
            light.SetIntensity(intensity * appearance.light_intensity)


class AppearanceControls:
    def __init__(self, parent, widgets, core, appearance, changed, gui=None):
        self.appearance, self.changed = appearance, changed
        self.widget = widgets.QWidget(parent)
        form = self.form = widgets.QFormLayout(self.widget)
        from ui.molecule_viewer_style import inspector_form, InspectorSection
        inspector_form(form, widgets)
        self.controls = {}
        preset = widgets.QComboBox(self.widget)
        preset.addItems(("Basic", "Figure", "Custom"))
        primary = widgets.QWidget(self.widget)
        primary_row = widgets.QHBoxLayout(primary)
        primary_row.setContentsMargins(0, 0, 0, 0)
        primary_row.setSpacing(4)
        primary_row.addWidget(preset, 1)
        form.addRow("Preset", primary)
        from ui.molecule_viewer_color import ColorControl
        self.background = ColorControl(self.widget, widgets, gui, "Viewport background",
            lambda: appearance.background_color,
            lambda: (appearance.document_generation, appearance.background_color),
            lambda value: self.change("background_color", value))
        form.addRow("Background", self.background.widget)
        atoms = InspectorSection(self.widget, widgets, "Atoms", gui=gui, core=core)
        bonds = InspectorSection(self.widget, widgets, "Bonds", gui=gui, core=core)
        lighting = InspectorSection(self.widget, widgets, "Lighting", gui=gui, core=core)
        self.sections = (atoms, bonds, lighting)
        self.field_forms = {}
        self.group_resets = []
        for section, keys in ((atoms, ("atom_finish", "detail", "smooth")),
                (bonds, ("bond_finish", "bond_color", "bond_mode", "line_width", "bond_radius")),
                (lighting, ("ambient", "diffuse", "specular", "specular_power", "light_intensity", "anti_aliasing"))):
            button = section.add_reset(widgets, lambda checked=False, keys=keys: self.reset_group(keys))
            self.group_resets.append((button, section.title))
        for section in self.sections:
            form.addRow(section.widget)
        self.controls["preset"] = (preset, "setCurrentText")
        preset.currentTextChanged.connect(self.choose_preset)
        for key, label, choices in (("atom_finish", "Finish", FINISHES),
                ("bond_finish", "Finish", FINISHES),
                ("detail", "Detail", tuple(DETAILS)),
                ("bond_mode", "Display", ("Line", "Cylinder"))):
            control = widgets.QComboBox(self.widget)
            control.addItems(choices)
            (bonds.form if key.startswith("bond_") else atoms.form).addRow(label, control)
            self.controls[key] = (control, "setCurrentText")
            control.currentTextChanged.connect(lambda value, key=key: self.change(key, value))
        from ui.molecule_viewer_color import ColorControl
        self.bond_color = ColorControl(self.widget, widgets, gui, "Bond color",
            lambda: appearance.bond_color,
            lambda: (appearance.document_generation, appearance.bond_color),
            lambda value: self.change("bond_color", value))
        bonds.form.addRow("Color", self.bond_color.widget)
        for key, label, bounds, step, suffix in (
                ("line_width", "Thickness", (1., 12.), .5, " px"),
                ("bond_radius", "Radius", (.01, .5), .01, " Å")):
            control = widgets.QDoubleSpinBox(self.widget)
            control.setRange(*bounds)
            control.setSingleStep(step)
            control.setSuffix(suffix)
            control.setAlignment(core.Qt.AlignmentFlag.AlignRight)
            bonds.form.addRow(label, control)
            self.field_forms[key] = bonds.form
            self.controls[key] = (control, "setValue")
            if key == "line_width":
                control.setToolTip("Thickness at the document's initial fitted view. Zoom scales it with the atoms; cylinders use their radius instead.")
            control.valueChanged.connect(lambda value, key=key: self.change(key, value))
        for key, label, maximum, step in (("ambient", "Ambient", 1., .05),
                ("diffuse", "Diffuse", 1., .05), ("specular", "Specular", 1., .05),
                ("specular_power", "Sharpness", 128., 1.),
                ("light_intensity", "Intensity", 2., .1)):
            control = widgets.QDoubleSpinBox(self.widget)
            control.setDecimals(2)
            control.setRange(0 if key == "light_intensity" else -.01, maximum)
            if key != "light_intensity":
                control.setSpecialValueText("Auto")
                control.setToolTip("Auto uses the selected material finish. An explicit value overrides it for ordinary atoms and bonds.")
            else:
                control.setToolTip("Multiplier of the existing viewport lights; 1.00 preserves their original intensities.")
            control.setSingleStep(step)
            control.setAlignment(core.Qt.AlignmentFlag.AlignRight)
            lighting.form.addRow(label, control)
            self.controls[key] = (control, "setValue")
            control.valueChanged.connect(lambda value, key=key: self.change(key, value))
        aa = widgets.QComboBox(self.widget)
        aa.addItems(("Default", "Off", "FXAA"))
        aa.setToolTip("Default restores startup anti-aliasing. FXAA smooths viewport pixels and may soften small text; check PNG and vector output separately.")
        lighting.form.addRow("Anti-aliasing", aa)
        self.controls["anti_aliasing"] = (aa, "setCurrentText")
        aa.currentTextChanged.connect(lambda value: self.change("anti_aliasing", value))
        smooth = widgets.QCheckBox("Smooth", self.widget)
        smooth.setToolTip("Smooth shading for ordinary atoms and bond cylinders; off shows flat facets.")
        atoms.form.addRow(smooth)
        self.controls["smooth"] = (smooth, "setChecked")
        smooth.toggled.connect(lambda value: self.change("smooth", value))
        self.widget.setToolTip("Appearance only. Bond tolerance determines connectivity; thickness, radius and color change its drawing. Averaged/reference markers retain their meaning.")
        self.controls["detail"][0].setToolTip("Sphere resolution 16/32/64 and cylinder sides 8/16/32. High uses more geometry and can reach vector-export limits sooner.")
        self.controls["bond_finish"][0].setToolTip("Surface reflection for cylinders; line rendering may show little difference.")
        reset = widgets.QPushButton("Reset", self.widget)
        self.reset_button = reset
        reset.clicked.connect(self.reset)
        primary_row.addWidget(reset)
        self.sync()

    def sync(self):
        self.background.sync()
        self.bond_color.sync()
        for key, (control, setter) in self.controls.items():
            blocked = control.blockSignals(True)
            getattr(control, setter)(getattr(self.appearance, key))
            control.blockSignals(blocked)
        cylinders = self.appearance.bond_mode == "Cylinder"
        self.controls["line_width"][0].setEnabled(not cylinders)
        self.controls["bond_radius"][0].setEnabled(cylinders)
        from ui.molecule_viewer_style import set_form_row_visible
        for key, visible in (("line_width", not cylinders), ("bond_radius", cylinders)):
            set_form_row_visible(self.field_forms[key], self.controls[key][0], visible)
        self.sections[0].summary(self.appearance.atom_finish)
        self.sections[1].summary(self.appearance.bond_mode)
        self.sections[2].summary(self.appearance.anti_aliasing)

    def arrange(self, display, axes):
        """Move existing fields, retaining their signal/validation owners."""
        from ui.molecule_viewer_style import move_form_row
        move_form_row(display.panel.layout(), display.atom, self.form, "Atom size", index=2)
        width = self.controls["line_width"][0]
        move_form_row(self.field_forms["line_width"], width, self.form, "Bond thickness", index=3)
        self.field_forms["line_width"] = self.form
        move_form_row(display.panel.layout(), display.bond, self.sections[1].form, "Tolerance", index=0)
        self.form.insertRow(6, axes.widget)
        self.sync()

    def apply_icons(self, assets):
        from ui.molecule_viewer_style import compact_button
        for button, title in self.group_resets:
            tooltip = {"Atoms": "Reset atom finish, detail and smoothness; shared atom size is retained.",
                       "Bonds": "Reset bond display, finish, color, thickness and radius; shared tolerance is retained.",
                       "Lighting": "Reset lighting/material overrides and anti-aliasing only."}[title]
            compact_button(assets, button, "reset", "Reset", tooltip)

    def reset_group(self, keys):
        previous = self.appearance.options()
        defaults = ViewportAppearance().options()
        for key in keys:
            setattr(self.appearance, key, defaults[key])
        self.appearance.preset = "Custom"
        self.notify(previous)

    def change(self, key, value):
        previous = self.appearance.options()
        setattr(self.appearance, key, value)
        self.appearance.preset = "Custom"
        self.notify(previous)

    def choose_preset(self, preset):
        previous = self.appearance.options()
        self.appearance.set_preset(preset)
        self.notify(previous)

    def notify(self, previous):
        generation = self.appearance.document_generation
        self.sync()
        if self.changed() is False and generation == self.appearance.document_generation:
            self.appearance.restore(previous)
            self.sync()

    def reset(self):
        previous = self.appearance.options()
        self.appearance.reset()
        self.notify(previous)
