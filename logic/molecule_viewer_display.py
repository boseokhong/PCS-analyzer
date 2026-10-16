"""Private display edits; shared numeric meaning, no scientific calculation."""

import math
import re


def validate_ui_theme(value):
    keys = {"surface", "text", "field", "edge", "hover", "pressed", "focus", "disabled", "disabled_text"}
    if (not isinstance(value, dict) or set(value) != keys
            or any(not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color)
                   for color in value.values())):
        raise ValueError("Invalid viewer UI theme")


def validate_display_change(key, value):
    if not isinstance(key, str):
        return False
    if key == "color_mode":
        return isinstance(value, str) and value in ("Element", "PCS")
    if key == "show_labels":
        return type(value) is bool
    limits = {"atom_scale": (.1, 10.), "bond_tolerance": (-.5, 2.)}
    if key in limits:
        return (type(value) in (int, float) and math.isfinite(value)
                and limits[key][0] <= value <= limits[key][1])
    return False


def validate_display_controls(value):
    if (not isinstance(value, dict)
            or set(value) != {"color_mode", "show_labels", "atom_scale", "bond_tolerance", "ui_font"}
            or any(not validate_display_change(key, value[key]) for key in value if key != "ui_font")):
        raise ValueError("Invalid display controls")
    font = value["ui_font"]
    if (not isinstance(font, list) or len(font) != 2 or not isinstance(font[0], str)
            or not 1 <= len(font[0]) <= 256 or type(font[1]) is not int or not 1 <= font[1] <= 128):
        raise ValueError("Invalid display control font")
