"""Child-process-only platform chrome; never changes DPI or system settings."""

import logging
import sys

from logic.molecule_viewer_display import validate_ui_theme


def dark_surface(colors):
    validate_ui_theme(colors)
    rgb = [int(colors["surface"][i:i + 2], 16) for i in (1, 3, 5)]
    return sum(c * w for c, w in zip(rgb, (.2126, .7152, .0722))) < 128


def apply_native_palette(app, gui, core, colors):
    """Keep default Qt menu painting, using the host's resolved light/dark roles."""
    dark = dark_surface(colors)
    hints = app.styleHints()
    if hasattr(hints, "setColorScheme") and hasattr(core.Qt, "ColorScheme"):
        hints.setColorScheme(core.Qt.ColorScheme.Dark if dark else core.Qt.ColorScheme.Light)
    palette = app.palette()
    roles = gui.QPalette.ColorRole
    for role, key in (("Window", "surface"), ("WindowText", "text"),
                      ("Base", "field"), ("AlternateBase", "surface"),
                      ("Text", "text"), ("Button", "surface"), ("ButtonText", "text"),
                      ("Highlight", "hover"), ("HighlightedText", "text"),
                      ("ToolTipBase", "field"), ("ToolTipText", "text")):
        palette.setColor(getattr(roles, role), gui.QColor(colors[key]))
    for role in (roles.Text, roles.ButtonText, roles.WindowText):
        palette.setColor(gui.QPalette.ColorGroup.Disabled, role, gui.QColor(colors["disabled_text"]))
    app.setPalette(palette)


def apply_titlebar(window, colors):
    """Best-effort DWM attributes for this HWND only; unsupported OS keeps defaults."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes
    validate_ui_theme(colors)
    try:
        setter = ctypes.windll.dwmapi.DwmSetWindowAttribute
        setter.argtypes = (wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD)
        setter.restype = ctypes.c_long
        hwnd = int(window.winId())
        def colorref(value):
            return sum(int(value[i:i + 2], 16) << shift for i, shift in ((1, 0), (3, 8), (5, 16)))
        for attribute, value in ((20, int(dark_surface(colors))),
                                 (35, colorref(colors["surface"])), (36, colorref(colors["text"]))):
            data = wintypes.DWORD(value)
            result = setter(hwnd, attribute, ctypes.byref(data), ctypes.sizeof(data))
            if result < 0:
                logging.getLogger(__name__).debug("DWM attribute %s unsupported: %s", attribute, result)
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        logging.getLogger(__name__).warning("Viewer title-bar styling unavailable: %s", exc)
