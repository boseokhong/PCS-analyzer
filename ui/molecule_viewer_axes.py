"""Presentation-only molecular axes; never rotate scientific coordinates."""


MODES = ("Metal centre", "Screen corner", "Both", "Hidden")
CORNERS = ("Bottom left", "Bottom right", "Top left", "Top right")


def corner_viewport(corner, size):
    if corner not in CORNERS or not .10 <= size <= .35:
        raise ValueError("Invalid corner axes viewport")
    x = .02 if corner.endswith("left") else .98 - size
    y = .02 if corner.startswith("Bottom") else .98 - size
    return (x, y, x + size, y + size)


class ViewportAxes:
    def __init__(self, plotter):
        self.plotter = plotter
        self.actors = []
        self.widget = None
        self.document_id = None
        self.reset()

    def reset(self):
        self.mode = "Metal centre"
        self.corner = "Bottom left"
        self.length = 0.  # Auto uses the original host-derived world length.
        self.size = .18
        self.world_labels = self.corner_labels = True
        self.width = 1.
        self.head_width = 1.
        self.head_length = .25
        self.world_font_size = self.corner_font_size = 0
        self.colors = ["blue", "green", "red"]
        self.visible = [True, True, True]

    def document(self, document_id):
        if document_id != self.document_id:
            self.document_id = document_id
            self.reset()

    def clear(self):
        if self.widget is not None:
            self.widget.SetEnabled(0)
            self.widget.SetInteractor(None)
            self.widget = None
        for actor in self.actors:
            self.plotter.remove_actor(actor, reset_camera=False, render=False)
        self.actors.clear()

    def apply(self, presentation):
        self.clear()
        if presentation is None:
            return
        from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
        import pyvista as pv
        length = self.length or presentation["axis_length"]
        if self.mode in ("Metal centre", "Both") and length is not None:
            for enabled, axis, direction, color in zip(self.visible, "XYZ", ((1, 0, 0), (0, 1, 0), (0, 0, 1)), self.colors):
                if not enabled:
                    continue
                arrow = pv.Arrow(start=(0, 0, 0), direction=direction, scale=length,
                                 shaft_radius=.05 * self.width,
                                 tip_radius=.10 * self.width * self.head_width,
                                 tip_length=self.head_length)
                self.actors.append(self.plotter.add_mesh(arrow, color=color, name=f"axis_{axis}",
                    pickable=False, reset_camera=False, render=False))
                if self.world_labels:
                    self.actors.append(self.plotter.add_point_labels(
                        [[component * length for component in direction]], [axis],
                        name=f"axis_label_{axis}", text_color=color, bold=True,
                        font_size=self.world_font_size or presentation["axis_label_size"], font_file=presentation["font_file"],
                        show_points=False, shape=None, always_visible=True, pickable=False,
                        reset_camera=False, render=False))
        if self.mode in ("Screen corner", "Both") and any(self.visible):
            actor = vtkAxesActor()
            actor.SetPickable(False)
            actor.SetTotalLength(*(1. if enabled else 0. for enabled in self.visible))
            actor.SetAxisLabels(self.corner_labels)
            for axis, color_value in zip("XYZ", self.colors):
                # Preserve the original corner colors until explicitly customized.
                color = {"blue": (0, 0, 1), "green": (0, 1, 0), "red": (1, 0, 0)}.get(color_value)
                if color is None:
                    color = tuple(int(color_value[offset:offset + 2], 16) / 255. for offset in (1, 3, 5))
                for part in ("Shaft", "Tip"):
                    getattr(actor, f"Get{axis}Axis{part}Property")().SetColor(*color)
                caption = getattr(actor, f"Get{axis}AxisCaptionActor2D")()
                caption.SetPickable(False)
                prop = caption.GetCaptionTextProperty()
                prop.SetColor(*color)
                prop.SetFontFamilyAsString("File")
                prop.SetFontFile(presentation["font_file"])
                prop.SetFontSize(self.corner_font_size or presentation["axis_label_size"])
                if self.corner_font_size:
                    caption.GetTextActor().SetTextScaleModeToNone()
                prop.ShadowOff()
            self.widget = self.plotter.add_orientation_widget(actor, interactive=False,
                viewport=corner_viewport(self.corner, self.size))


class AxesControls:
    def __init__(self, parent, widgets, axes, changed, gui=None, core=None):
        self.axes, self.changed = axes, changed
        self.widgets, self.gui = widgets, gui
        from ui.molecule_viewer_style import InspectorSection
        self.section = InspectorSection(parent, widgets, "Axes", gui=gui, core=core)
        self.widget = self.section.widget
        form = self.section.form
        self.mode = widgets.QComboBox(self.widget)
        self.mode.addItems(MODES)
        self.corner = widgets.QComboBox(self.widget)
        self.corner.addItems(CORNERS)
        self.length = widgets.QDoubleSpinBox(self.widget)
        self.length.setRange(0, 10000)
        self.length.setSpecialValueText("Auto")
        self.length.setSuffix(" Å")
        self.size = widgets.QDoubleSpinBox(self.widget)
        self.size.setRange(.10, .35)
        self.size.setSingleStep(.01)
        self.width = widgets.QDoubleSpinBox(self.widget)
        self.width.setRange(.1, 5)
        self.width.setSingleStep(.1)
        self.head_width = widgets.QDoubleSpinBox(self.widget)
        self.head_width.setRange(.1, 3)
        self.head_width.setSingleStep(.1)
        self.head_length = widgets.QDoubleSpinBox(self.widget)
        self.head_length.setRange(.05, .75)
        self.head_length.setSingleStep(.05)
        self.world_font_size = widgets.QSpinBox(self.widget)
        self.corner_font_size = widgets.QSpinBox(self.widget)
        for control in (self.world_font_size, self.corner_font_size):
            control.setRange(0, 96)
            control.setSpecialValueText("Auto")
            control.setSuffix(" px")
            control.setToolTip("Renderer font size in pixels; Auto retains the existing plot-font behavior. Check the exported result separately.")
        self.world_labels = widgets.QCheckBox("Labels", self.widget)
        self.corner_labels = widgets.QCheckBox("Labels", self.widget)
        primary = widgets.QWidget(self.widget)
        primary_row = widgets.QHBoxLayout(primary)
        primary_row.setContentsMargins(0, 0, 0, 0)
        primary_row.setSpacing(4)
        primary_row.addWidget(self.mode, 1)
        form.addRow("Axes", primary)
        metal = InspectorSection(self.widget, widgets, "Metal-centre", gui=gui, core=core)
        corner = InspectorSection(self.widget, widgets, "Corner", gui=gui, core=core)
        form.addRow(metal.widget)
        form.addRow(corner.widget)
        for label, control in (("Length", self.length), ("Thickness", self.width),
                ("Tip width", self.head_width), ("Tip length", self.head_length),
                ("Label size", self.world_font_size)):
            metal.form.addRow(label, control)
        for label, control in (("Position", self.corner), ("Size", self.size),
                ("Label size", self.corner_font_size)):
            corner.form.addRow(label, control)
        self.head_width.setToolTip("Multiplier of the metal-centred arrow tip radius; does not change shaft thickness.")
        self.head_length.setToolTip("Tip length as a fraction of the metal-centred arrow length (0.05–0.75).")
        self.width.setToolTip("Thickness of the metal-centred arrows; corner size is controlled separately.")
        self.size.setToolTip("Fraction of the viewport occupied by the corner orientation marker (0.10–0.35).")
        self.length.setToolTip("Metal-centred axis length in angstroms; Auto preserves the existing structure-derived length.")
        metal.form.addRow(self.world_labels)
        corner.form.addRow(self.corner_labels)
        colors = InspectorSection(self.widget, widgets, "X / Y / Z", gui=gui, core=core)
        self.sections = (self.section, metal, corner, colors)
        self.group_resets = []
        for section, keys in ((metal, ("length", "width", "head_width", "head_length", "world_font_size", "world_labels")),
                (corner, ("corner", "size", "corner_font_size", "corner_labels")),
                (colors, ("colors", "visible"))):
            button = section.add_reset(widgets, lambda checked=False, keys=keys: self.reset_group(keys))
            self.group_resets.append((button, section.title))
        form.addRow(colors.widget)
        self.visibility = []
        self.color_buttons = []
        self.color_controls = []
        for index, axis in enumerate("XYZ"):
            check = widgets.QCheckBox(axis, self.widget)
            check.setToolTip(f"Show or hide the {axis} axis in both placements.")
            check.toggled.connect(lambda value, index=index: self.change_axis(index, value))
            self.visibility.append(check)
            from ui.molecule_viewer_color import ColorControl
            color = ColorControl(self.widget, widgets, gui, f"Choose {axis} axis color",
                lambda index=index: self.axes.colors[index],
                lambda: (self.axes.document_id, id(self.axes.colors)),
                lambda value, index=index: self.set_color(index, value))
            axis_row = widgets.QWidget(colors.body)
            row = widgets.QHBoxLayout(axis_row)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            row.addWidget(check)
            row.addWidget(color.widget, 1)
            colors.form.addRow(axis_row)
            self.color_controls.append(color)
            self.color_buttons.append(color.button)
        reset = widgets.QToolButton(self.widget)
        reset.setText("Reset")
        reset.setSizePolicy(widgets.QSizePolicy.Policy.Fixed, widgets.QSizePolicy.Policy.Fixed)
        self.reset_button = reset
        reset.clicked.connect(self.reset)
        self.section.header.addWidget(reset)
        for key, control, signal in (("mode", self.mode, "currentTextChanged"),
                ("corner", self.corner, "currentTextChanged"), ("length", self.length, "valueChanged"),
                ("size", self.size, "valueChanged"), ("width", self.width, "valueChanged"),
                ("head_width", self.head_width, "valueChanged"), ("head_length", self.head_length, "valueChanged"),
                ("world_font_size", self.world_font_size, "valueChanged"), ("corner_font_size", self.corner_font_size, "valueChanged"),
                ("world_labels", self.world_labels, "toggled"), ("corner_labels", self.corner_labels, "toggled")):
            getattr(control, signal).connect(lambda value, key=key: self.change(key, value))
        self.sync()

    def sync(self):
        for key, control, setter in (("mode", self.mode, "setCurrentText"), ("corner", self.corner, "setCurrentText"),
                ("length", self.length, "setValue"), ("size", self.size, "setValue"), ("width", self.width, "setValue"),
                ("head_width", self.head_width, "setValue"), ("head_length", self.head_length, "setValue"),
                ("world_font_size", self.world_font_size, "setValue"), ("corner_font_size", self.corner_font_size, "setValue"),
                ("world_labels", self.world_labels, "setChecked"), ("corner_labels", self.corner_labels, "setChecked")):
            blocked = control.blockSignals(True)
            getattr(control, setter)(getattr(self.axes, key))
            control.blockSignals(blocked)
        for color in self.color_controls:
            color.sync()
        for control, value in zip(self.visibility, self.axes.visible):
            blocked = control.blockSignals(True)
            control.setChecked(value)
            control.blockSignals(blocked)

        self.section.summary(self.axes.mode)
        self.sections[1].widget.setVisible(self.axes.mode in ("Metal centre", "Both"))
        self.sections[2].widget.setVisible(self.axes.mode in ("Screen corner", "Both"))
        self.sections[3].widget.setVisible(self.axes.mode != "Hidden")

    def apply_icons(self, assets):
        from ui.molecule_viewer_style import compact_button
        for button, title in self.group_resets:
            compact_button(assets, button, "reset", "Reset", "Reset " + title.lower() + " axes settings only.")

    def reset_group(self, keys):
        defaults = ViewportAxes(self.axes.plotter)
        for key in keys:
            setattr(self.axes, key, getattr(defaults, key))
        self.sync()
        self.changed()

    def change(self, key, value):
        setattr(self.axes, key, value)
        self.sync()
        self.changed()

    def change_axis(self, index, value):
        self.axes.visible[index] = value
        self.changed()

    def choose_color(self, index):
        self.color_controls[index].choose()

    def set_color(self, index, value):
        self.axes.colors[index] = value
        self.sync()
        self.changed()

    def reset(self):
        self.axes.reset()
        self.sync()
        self.changed()
