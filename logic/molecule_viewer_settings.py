"""Versioned, bounded visualization settings; no coordinates or GUI dependencies."""

import json
import math
import os
from pathlib import Path
import re
import tempfile

MAX_BYTES = 65536
FORMAT = "pcs-analyzer-visualization"
APPEARANCE_CHOICES = {"atom_finish": ("Default", "Matte", "Glossy"),
    "bond_finish": ("Default", "Matte", "Glossy"), "detail": ("Low", "Standard", "High"),
    "bond_mode": ("Line", "Cylinder"), "preset": ("Basic", "Figure", "Custom"),
    "anti_aliasing": ("Default", "Off", "FXAA")}
APPEARANCE_NUMBERS = {"line_width": (1, 12), "bond_radius": (.01, .5),
    "ambient": (-1, 1), "diffuse": (-1, 1), "specular": (-1, 1),
    "specular_power": (-1, 128), "light_intensity": (0, 2)}
AXIS_CHOICES = {"mode": ("Metal centre", "Screen corner", "Both", "Hidden"),
    "corner": ("Bottom left", "Bottom right", "Top left", "Top right")}
AXIS_NUMBERS = {"length": (0, 10000), "size": (.10, .35), "width": (.1, 5),
    "head_width": (.1, 3), "head_length": (.05, .75),
    "world_font_size": (0, 96), "corner_font_size": (0, 96)}
AXIS_KEYS = tuple(AXIS_CHOICES) + tuple(AXIS_NUMBERS) + ("world_labels", "corner_labels", "colors", "visible")
FIGURE_NUMBERS = {"width": (.001, 1e6), "height": (.001, 1e6), "dpi": (72, 1200),
    "margin_mm": (0, 100), "font_pt": (0, 72), "line_pt": (0, 12)}
LAYERS = ("world_axes", "corner_axes", "labels", "pcs_legend", "averaging_legend", "selection", "reference")


def _keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("Missing or unsupported visualization fields")


def _number(value, bounds):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
        raise ValueError("Visualization number outside supported range")


def _color(value, named=False):
    if not isinstance(value, str) or not (re.fullmatch(r"#[0-9a-fA-F]{6}", value) or named and value in ("blue", "green", "red")):
        raise ValueError("Invalid RGB visualization color")


def validate_settings(data):
    keys = ("format", "version", "appearance", "axes", "figure")
    _keys(data, (*keys, "pcs_colors") if isinstance(data, dict) and "pcs_colors" in data else keys)
    if data["format"] != FORMAT or type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("Unsupported visualization settings format/version")
    appearance, axes, figure = (data[key] for key in ("appearance", "axes", "figure"))
    _keys(appearance, (*APPEARANCE_CHOICES, *APPEARANCE_NUMBERS, "smooth", "bond_color", "background_color"))
    _keys(axes, AXIS_KEYS)
    _keys(figure, (*FIGURE_NUMBERS, "unit", "background", *("include_" + key for key in LAYERS)))
    for group, choices, numbers in ((appearance, APPEARANCE_CHOICES, APPEARANCE_NUMBERS),
                                   (axes, AXIS_CHOICES, AXIS_NUMBERS),
                                   (figure, {"unit": ("mm", "in", "px"), "background": ("White", "Transparent", "Viewport")}, FIGURE_NUMBERS)):
        for key, values in choices.items():
            if group[key] not in values:
                raise ValueError("Unknown visualization option: " + key)
        for key, bounds in numbers.items():
            _number(group[key], bounds)
    for group, keys in ((appearance, ("smooth",)), (axes, ("world_labels", "corner_labels")),
                        (figure, tuple("include_" + key for key in LAYERS))):
        if any(type(group[key]) is not bool for key in keys):
            raise ValueError("Visualization toggles must be boolean")
    if any(type(group[key]) is not int for group, key in ((axes, "world_font_size"), (axes, "corner_font_size"), (figure, "dpi"))):
        raise ValueError("Font pixel sizes and DPI must be integers")
    _color(appearance["bond_color"])
    _color(appearance["background_color"])
    if not isinstance(axes["colors"], list) or len(axes["colors"]) != 3:
        raise ValueError("Expected three axis colors")
    for color in axes["colors"]:
        _color(color, named=True)
    if not isinstance(axes["visible"], list) or len(axes["visible"]) != 3 or any(type(value) is not bool for value in axes["visible"]):
        raise ValueError("Expected three axis visibility toggles")
    if "pcs_colors" in data:
        from logic.molecule_viewer_pcs import validate_pcs_options
        validate_pcs_options(data["pcs_colors"])
    return data


def load_settings(path):
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("Visualization settings exceed 64 KiB")
    return validate_settings(json.loads(payload.decode("utf-8")))


def save_settings(path, data):
    validate_settings(data)
    payload = json.dumps(data, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    if len(payload) > MAX_BYTES:
        raise ValueError("Visualization settings exceed 64 KiB")
    target = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".pcs-visual-", suffix=".json", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
