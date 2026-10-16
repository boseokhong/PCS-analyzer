"""Native Qt prototype controls; host snapshots remain authoritative."""

from logic.molecule_viewer_display import validate_display_change


class DisplayControls:
    def __init__(self, parent, widgets, gui, send, core=None):
        self.widgets, self.gui, self.send = widgets, gui, send
        self.values = None
        self.confirmed = None
        self.document_id = None
        self.widget = widgets.QWidget(parent)
        quick = widgets.QHBoxLayout(self.widget)
        quick.setContentsMargins(0, 0, 0, 0)
        self.panel = widgets.QWidget(parent)
        if core is not None:
            for container in (self.widget, self.panel):
                container.setAttribute(core.Qt.WidgetAttribute.WA_StyledBackground, True)
                container.setAutoFillBackground(True)
        form = widgets.QFormLayout(self.panel)
        from ui.molecule_viewer_style import inspector_form
        inspector_form(form, widgets)
        self.color = widgets.QComboBox(self.widget)
        self.color.addItems(["Element", "PCS"])
        self.labels = widgets.QToolButton(self.widget)
        self.labels.setText("Labels")
        self.labels.setCheckable(True)
        self.labels.setAccessibleName("Atom labels")
        self.labels.setToolTip("Show or hide atom labels. Shared with the main application.")
        self.atom = widgets.QLineEdit(self.panel)
        self.bond = widgets.QLineEdit(self.panel)
        if core is not None:
            for entry in (self.atom, self.bond):
                entry.setAlignment(core.Qt.AlignmentFlag.AlignRight)
        self.bond.setToolTip("Adjust which atom pairs are drawn as bonded. The cutoff uses covalent radii and tolerance; bond thickness is controlled separately.")
        self.atom.setToolTip("Change atom-marker size (0.10–10.0). Shared with the main application; coordinates are unchanged.")
        self.color.setAccessibleName("Atom color mode")
        self.color.setToolTip("Atom colors: Element uses CPK colors; PCS uses the calculated shift scale.")
        self.color_label = widgets.QLabel("Color mode:", self.widget)
        self.color_label.setBuddy(self.color)
        quick.addWidget(self.color_label)
        quick.addWidget(self.color)
        quick.addWidget(self.labels)
        form.addRow("Atom size", self.atom)
        form.addRow("Bond tolerance", self.bond)
        self.panel.setToolTip("Atom size and bond tolerance are shared with the main application. Appearance, axes and Figure settings are local to this viewer.")
        self.color.currentTextChanged.connect(lambda value: self.edit("color_mode", value))
        self.labels.toggled.connect(lambda value: self.edit("show_labels", value))
        self.atom.editingFinished.connect(lambda: self.numeric("atom_scale", self.atom))
        self.bond.editingFinished.connect(lambda: self.numeric("bond_tolerance", self.bond))
        self.widget.setVisible(False)

    def sync(self, values, document_id=None):
        same_document = document_id is not None and document_id == self.document_id
        self.document_id = document_id
        self.values = dict(values) if values is not None else None
        self.confirmed = dict(values) if values is not None else None
        self.widget.setVisible(values is not None)
        self.panel.setEnabled(values is not None)
        if values is None:
            return
        self.widget.setFont(self.gui.QFont(*values["ui_font"]))
        self.panel.setFont(self.gui.QFont(*values["ui_font"]))
        for control, setter, value in ((self.color, "setCurrentText", values["color_mode"]),
                                       (self.labels, "setChecked", values["show_labels"]),
                                       (self.atom, "setText", format(values["atom_scale"], ".15g")),
                                       (self.bond, "setText", format(values["bond_tolerance"], ".15g"))):
            if setter == "setText" and same_document and control.hasFocus() and control.isModified():
                continue
            blocked = control.blockSignals(True)
            try:
                getattr(control, setter)(value)
                if setter == "setText":
                    control.setModified(False)
            finally:
                control.blockSignals(blocked)

    def edit(self, key, value):
        if self.values is not None and validate_display_change(key, value) and value != self.values[key]:
            self.send(key, value)
            self.values[key] = value  # Pending local intent; host scene is untouched.

    def numeric(self, key, control):
        if self.values is None or not control.isModified():
            return
        try:
            value = float(control.text().strip())
            if not validate_display_change(key, value):
                raise ValueError("Value outside the supported range")
            self.edit(key, value)
            control.setModified(False)
        except ValueError:
            self.widgets.QMessageBox.warning(self.widget, "Molecular 3D",
                "Atom scale must be 0.10–10.0; bond tolerance must be −0.50–2.00.")
            self.sync(self.confirmed)
