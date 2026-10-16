"""Scoped Qt chrome styles from host-resolved colors; no independent palette."""

from pathlib import Path
import sys
import tempfile

from logic.molecule_viewer_display import validate_ui_theme


def packaged_icon_path(name):
    """Match IconManager's source/bundle/executable resource search without Tk."""
    roots = [Path(__file__).resolve().parents[1]]
    if getattr(sys, "_MEIPASS", None):
        roots.insert(0, Path(sys._MEIPASS))
    if getattr(sys, "frozen", False):
        roots.extend((Path(sys.executable).resolve().parent, Path(sys.executable).resolve().parent / "lib"))
    for root in roots:
        for size in (32, 24, 20):
            path = root / "img" / "icons" / str(size) / (name + ".png")
            if path.is_file():
                return path
    return None


class ChromeAssets:
    """Child-owned PNG glyphs from resolved theme, plus existing semantic icons."""

    def __init__(self, gui, core):
        self.gui, self.core = gui, core
        self.folder = tempfile.TemporaryDirectory(prefix="pcs-viewer-chrome-")
        self.cache, self.icons = {}, {}

    def glyphs(self, colors):
        validate_ui_theme(colors)
        result = {}
        shapes = {"check": ((14, 33), (27, 46), (50, 18)),
                  "down": ((16, 25), (32, 41), (48, 25)),
                  "up": ((16, 39), (32, 23), (48, 39)),
                  "right": ((25, 16), (41, 32), (25, 48)),
                  "close": ((18, 18), (46, 46), None, (46, 18), (18, 46))}
        for state, color in (("", colors["text"]), ("_disabled", colors["disabled_text"])):
            for name, points in shapes.items():
                key = (name, color)
                if key not in self.cache:
                    path = Path(self.folder.name) / f"{name}-{color[1:]}.png"
                    image = self.gui.QImage(64, 64, self.gui.QImage.Format.Format_ARGB32)
                    image.fill(self.core.Qt.GlobalColor.transparent)
                    painter = self.gui.QPainter(image)
                    try:
                        painter.setRenderHint(self.gui.QPainter.RenderHint.Antialiasing)
                        pen = self.gui.QPen(self.gui.QColor(color), 6,
                            self.core.Qt.PenStyle.SolidLine, self.core.Qt.PenCapStyle.RoundCap,
                            self.core.Qt.PenJoinStyle.RoundJoin)
                        painter.setPen(pen)
                        for first, second in zip(points, points[1:]):
                            if first is not None and second is not None:
                                painter.drawLine(self.core.QPointF(*first), self.core.QPointF(*second))
                    finally:
                        painter.end()
                    if not image.save(str(path), "PNG"):
                        raise OSError("Could not prepare viewer control glyphs")
                    self.cache[key] = path.as_posix()
                result[name + state] = self.cache[key]
        return result

    def set_icon(self, control, name):
        if name not in self.icons:
            path = packaged_icon_path(name)
            self.icons[name] = self.gui.QIcon(str(path)) if path else None
            if path:
                large = path.parent.parent / "512" / path.name
                if large.is_file():
                    self.icons[name].addFile(str(large))
        icon = self.icons[name]
        if icon is not None and not icon.isNull():
            control.setIcon(icon)  # Labels remain available when assets are missing.
            return True
        return False

    def close(self):
        self.folder.cleanup()


def compact_button(assets, button, icon, label, tooltip):
    """Use an icon when present, with readable fallback and keyboard name."""
    button.setAccessibleName(label)
    button.setToolTip(tooltip)
    button.setText("" if assets.set_icon(button, icon) else label)


class InspectorSection:
    """Presentation-only disclosure; collapsing never changes option values."""
    def __init__(self, parent, widgets, title, *, gui=None, core=None, expanded=False):
        self.core = core
        self.title = title
        self.widget = widgets.QWidget(parent)
        layout = widgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        header = self.header = widgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        # QPushButton honours left text alignment and has a text-based size hint.
        # A QToolButton with child labels retained its fixed 21px native width.
        self.toggle = widgets.QPushButton(self.widget)
        self.toggle.setObjectName("viewerDisclosure")
        self.toggle.setAutoDefault(False)
        self.toggle.setSizePolicy(widgets.QSizePolicy.Policy.Expanding,
                                  widgets.QSizePolicy.Policy.Fixed)
        self._summary = ""
        self._expanded = expanded
        if core is not None:
            self.toggle.setFocusPolicy(core.Qt.FocusPolicy.StrongFocus)
        header.addWidget(self.toggle, 1)
        self.toggle.setCheckable(True)
        self.toggle.setAccessibleName(title + " settings")
        self.toggle.setToolTip("Expand or collapse " + title.lower() + " settings. Values are retained.")
        self.body = widgets.QWidget(self.widget)
        self.form = widgets.QFormLayout(self.body)
        inspector_form(self.form, widgets)
        layout.addLayout(header)
        layout.addWidget(self.body)
        self.toggle.setChecked(expanded)
        self.set_expanded(expanded)
        self.toggle.toggled.connect(self.set_expanded)

    def summary(self, value):
        self._summary = str(value) if value else ""
        self._update_title()

    def _update_title(self):
        text = self.title + (" · " + self._summary if self._summary else "")
        self.toggle.setText(("▾ " if self._expanded else "▸ ") + text)
        self.toggle.setToolTip(text + "\nExpand or collapse. Values are retained.")

    def add_reset(self, widgets, callback):
        self.reset_button = widgets.QToolButton(self.widget)
        self.reset_button.setAutoRaise(True)
        self.reset_button.setSizePolicy(widgets.QSizePolicy.Policy.Fixed, widgets.QSizePolicy.Policy.Fixed)
        self.reset_button.setText("Reset")
        self.reset_button.clicked.connect(callback)
        self.header.addWidget(self.reset_button)
        return self.reset_button

    def set_expanded(self, expanded):
        self.body.setVisible(expanded)
        self.toggle.setAccessibleDescription("Expanded" if expanded else "Collapsed")
        self._expanded = bool(expanded)
        self._update_title()
        self.widget.updateGeometry()


def set_form_row_visible(form, control, visible):
    control.setVisible(visible)
    label = form.labelForField(control)
    if label is not None:
        label.setVisible(visible)


def move_form_row(source, control, target, title, *, index):
    # removeRow destroys its widgets. takeRow preserves the existing editor/signals.
    row = source.takeRow(control)
    if row.labelItem is not None:
        label = row.labelItem.widget()
        if label is not None:
            label.deleteLater()
    target.insertRow(index, title, control)


def align_form_labels(forms, widgets, metrics):
    width = metrics.horizontalAdvance("Background")
    width = max(88, min(112, int(width + 8))) if isinstance(width, (int, float)) else 88
    for form in forms:
        count = form.rowCount()
        if not isinstance(count, int):
            continue
        for row in range(count):
            item = form.itemAt(row, widgets.QFormLayout.ItemRole.LabelRole)
            label = item.widget() if item is not None else None
            if label is not None:
                label.setFixedWidth(width)
                label.setWordWrap(True)
            field_item = form.itemAt(row, widgets.QFormLayout.ItemRole.FieldRole)
            field = field_item.widget() if field_item is not None else None
            if field is not None:
                # Spin-box maximum text must not force otherwise usable rows to wrap.
                policy = field.sizePolicy()
                policy.setHorizontalPolicy(widgets.QSizePolicy.Policy.Ignored)
                field.setSizePolicy(policy)


def separator_stylesheet(colors):
    validate_ui_theme(colors)
    return f"""QMainWindow::separator {{ background: {colors['surface']}; border: none;
        border-left: 1px solid {colors['edge']}; width: 8px; height: 8px; }}
QMainWindow::separator:hover {{ background: {colors['hover']}; border-color: {colors['focus']}; }}"""


def inspector_form(form, widgets):
    """Compact inspector spacing in Qt logical pixels, using existing role metrics."""
    form.setContentsMargins(8, 6, 8, 6)
    form.setHorizontalSpacing(8)
    form.setVerticalSpacing(4)
    form.setRowWrapPolicy(widgets.QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setFieldGrowthPolicy(widgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

def chrome_stylesheet(colors, glyphs=None, ui_font=None):
    validate_ui_theme(colors)
    p = colors
    glyphs = glyphs or {}
    font_rule = ""
    if ui_font is not None:
        family = str(ui_font[0]).replace("\\", "\\\\").replace('"', '\\"')
        font_rule = f'font-family: "{family}"; font-size: {int(ui_font[1])}pt;'
    def image(name):
        return f'image: url("{glyphs[name]}");' if name in glyphs else ""
    close_icon = f'titlebar-close-icon: url("{glyphs["close"]}");' if "close" in glyphs else ""
    return f"""
QWidget {{ background: {p['surface']}; color: {p['text']}; {font_rule} }}
QLabel {{ background: transparent; color: {p['text']}; }}
QStatusBar {{ background: {p['surface']}; color: {p['text']}; }}
QToolTip {{ background: {p['field']}; color: {p['text']}; border: 1px solid {p['edge']}; padding: 6px 8px; }}
QToolBar {{ border: none; spacing: 4px; padding: 4px; }}
QToolButton, QPushButton {{ background: {p['surface']}; color: {p['text']}; border: 1px solid {p['edge']}; border-radius: 3px; padding: 4px 6px; }}
QToolButton:hover, QPushButton:hover {{ background: {p['hover']}; }}
QToolButton:pressed, QPushButton:pressed {{ background: {p['pressed']}; }}
QToolButton:checked {{ background: {p['hover']}; border-color: {p['focus']}; }}
QPushButton#viewerDisclosure {{ background: transparent; border: 1px solid transparent; border-radius: 0; padding: 3px 2px; font-weight: 600; text-align: left; }}
QPushButton#viewerDisclosure:checked {{ background: transparent; border-color: transparent; }}
QPushButton#viewerDisclosure:hover {{ background: {p['hover']}; }}
QPushButton#viewerDisclosure:focus {{ border-color: {p['focus']}; }}
QToolButton:disabled, QPushButton:disabled {{ background: {p['disabled']}; color: {p['disabled_text']}; border-color: {p['edge']}; }}
QToolButton:focus, QPushButton:focus, QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{ border-color: {p['focus']}; }}
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{ background: {p['field']}; border: 1px solid {p['edge']}; border-radius: 3px; padding: 3px 4px; }}
QLineEdit, QDoubleSpinBox, QSpinBox {{ selection-background-color: {p['hover']}; selection-color: {p['text']}; }}
QLineEdit:hover, QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{ border-color: {p['text']}; }}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{ border-color: {p['focus']}; }}
QCheckBox {{ background: transparent; color: {p['text']}; spacing: 6px; border: 1px solid transparent; border-radius: 3px; }}
QCheckBox:focus {{ border-color: {p['focus']}; }}
QWidget:disabled {{ color: {p['disabled_text']}; }}
QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {{ background: {p['disabled']}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {p['edge']}; border-radius: 3px; background: {p['field']}; }}
QCheckBox::indicator:checked {{ background: {p['hover']}; {image('check')} }}
QCheckBox::indicator:hover {{ border-color: {p['focus']}; }}
QCheckBox::indicator:disabled {{ background: {p['disabled']}; border-color: {p['edge']}; }}
QCheckBox::indicator:checked:disabled {{ {image('check_disabled')} }}
QCheckBox:disabled {{ color: {p['disabled_text']}; }}
QComboBox {{ padding-right: 24px; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right; width: 22px; border: none; background: transparent; }}
QComboBox::drop-down:hover {{ background: {p['hover']}; border-top-right-radius: 3px; border-bottom-right-radius: 3px; }}
QComboBox::down-arrow {{ width: 12px; height: 12px; {image('down')} }}
QComboBox::down-arrow:disabled {{ {image('down_disabled')} }}
QComboBox QAbstractItemView {{ background: {p['field']}; color: {p['text']}; border: 1px solid {p['edge']}; selection-background-color: {p['hover']}; selection-color: {p['text']}; outline: none; }}
QDoubleSpinBox, QSpinBox {{ padding-right: 22px; }}
QDoubleSpinBox::up-button, QSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 20px; border: none; background: transparent; }}
QDoubleSpinBox::down-button, QSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 20px; border: none; background: transparent; }}
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{ width: 11px; height: 11px; {image('up')} }}
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{ width: 11px; height: 11px; {image('down')} }}
QDoubleSpinBox::up-arrow:disabled, QSpinBox::up-arrow:disabled {{ {image('up_disabled')} }}
QDoubleSpinBox::down-arrow:disabled, QSpinBox::down-arrow:disabled {{ {image('down_disabled')} }}
QDoubleSpinBox::up-arrow:off, QSpinBox::up-arrow:off {{ {image('up_disabled')} }}
QDoubleSpinBox::down-arrow:off, QSpinBox::down-arrow:off {{ {image('down_disabled')} }}
QToolButton[popupMode="2"] {{ padding-right: 18px; }}
QToolButton::menu-indicator {{ subcontrol-position: right center; width: 10px; height: 10px; {image('down')} }}
QScrollArea {{ border: none; background: {p['surface']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; border: none; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; border: none; }}
QScrollBar::handle:vertical {{ background: {p['edge']}; min-height: 26px; border-radius: 2px; margin: 0 3px; }}
QScrollBar::handle:horizontal {{ background: {p['edge']}; min-width: 26px; border-radius: 2px; margin: 3px 0; }}
QScrollBar::handle:hover, QScrollBar::handle:pressed {{ background: {p['focus']}; }}
QScrollBar::handle:disabled {{ background: {p['disabled_text']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; border: none; background: transparent; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QDockWidget {{ background: {p['surface']}; border: none; {close_icon} }}
QDockWidget::title {{ background: {p['surface']}; color: {p['text']}; padding: 5px; border-bottom: 1px solid {p['edge']}; }}
QDockWidget::close-button {{ width: 14px; height: 14px; padding: 3px; border: none; {image('close')} }}
QDockWidget::close-button:hover {{ background: {p['hover']}; }}
QGroupBox {{ border: 1px solid {p['edge']}; border-radius: 3px; margin-top: 10px; padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding: 0 4px; background: {p['surface']}; color: {p['text']}; }}
QLabel#viewerSection {{ font-weight: 600; padding-top: 6px; background: transparent; }}
QTabWidget::pane {{ background: {p['surface']}; border: 1px solid {p['edge']}; }}
QTabBar {{ background: {p['surface']}; }}
QTabBar::tab {{ background: {p['surface']}; color: {p['text']}; border: 1px solid {p['edge']}; padding: 5px 10px; }}
QTabBar::tab:selected {{ background: {p['hover']}; border-bottom-color: {p['focus']}; }}
QTabBar::tab:hover {{ background: {p['hover']}; }}
"""


def menubar_stylesheet(colors):
    validate_ui_theme(colors)
    return f"""QMenuBar {{ background: {colors['surface']}; color: {colors['text']}; }}
QMenuBar::item {{ background: transparent; padding: 4px 8px; }}
QMenuBar::item:selected {{ background: {colors['hover']}; }}"""
