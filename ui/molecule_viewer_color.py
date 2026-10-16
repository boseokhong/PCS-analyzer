"""Child-local color swatch and RGB code input; no Tk or scientific state."""

import re


def rgb_code(value):
    value = value.strip()
    if not re.fullmatch(r"#?[0-9a-fA-F]{6}", value):
        raise ValueError("Use six hexadecimal digits, for example #808080")
    return "#" + value.lstrip("#").upper()


def annotation_color(background, *, legend=False):
    code = rgb_code(background)
    brightness = sum(int(code[i:i + 2], 16) * weight for i, weight in zip((1, 3, 5), (.2126, .7152, .0722)))
    level = (.85 if legend else 1.) if brightness < 128 else (.35 if legend else 0.)
    return (level, level, level)


class ColorControl:
    def __init__(self, parent, widgets, gui, title, current, affinity, apply):
        self.widgets, self.gui = widgets, gui
        self.title, self.current, self.affinity, self.apply = title, current, affinity, apply
        self.widget = widgets.QWidget(parent)
        layout = widgets.QHBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.button = widgets.QPushButton(self.widget)
        self.button.setAccessibleName(title + " palette")
        self.button.setToolTip("Choose a color from the palette.")
        self.edit = widgets.QLineEdit(self.widget)
        self.edit.setMaxLength(7)
        self.edit.setAccessibleName(title + " RGB code")
        self.edit.setToolTip("RGB color code: #RRGGBB. Press Enter or leave the field to apply.")
        layout.addWidget(self.button)
        layout.addWidget(self.edit, 1)
        self.button.clicked.connect(self.choose)
        self.edit.editingFinished.connect(self.commit)
        self.sync()

    def sync(self):
        code = rgb_code(self.edit_color())
        self.edit.setText(code)
        self.button.setAccessibleDescription(code)
        if self.gui is not None:
            # Icon painting preserves inherited button borders/focus/theme styles.
            pixmap = self.gui.QPixmap(24, 16)
            pixmap.fill(self.gui.QColor(code))
            self.button.setIcon(self.gui.QIcon(pixmap))
            self.button.setIconSize(pixmap.size())

    def commit(self):
        try:
            value = rgb_code(self.edit.text())
        except ValueError:
            self.widgets.QMessageBox.warning(self.widget, "Invalid color code",
                "Enter six hexadecimal digits, for example #808080.")
            self.sync()
            return
        if value != rgb_code(self.edit_color()):
            self.apply(value)
        self.sync()

    def edit_color(self):
        value = self.current()
        return {"blue": "#0000FF", "green": "#008000", "red": "#FF0000"}.get(value, value)

    def choose(self, checked=False):
        token = self.affinity()
        color = self.widgets.QColorDialog.getColor(self.gui.QColor(self.current()),
            self.widget, self.title)
        # A modal palette can process an incoming document change or reset.
        if color.isValid() and token == self.affinity():
            value = rgb_code(color.name())
            if value != rgb_code(self.edit_color()):
                self.apply(value)
        self.sync()
