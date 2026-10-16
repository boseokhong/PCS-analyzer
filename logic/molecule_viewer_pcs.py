"""Viewer-local PCS display preferences; never edits calculated PCS values."""
import math
import re

PALETTES = ("RdBu_r", "coolwarm", "PiYG_r", "viridis", "cividis")
DEFAULTS = dict(range_mode="Auto", minimum=-1., maximum=1., centre_zero=True,
                palette="RdBu_r", reverse=False, missing_color="#BFBFBF",
                show_legend=True, tick_step=0., legend_font=0)


def validate_pcs_options(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULTS):
        raise ValueError("Missing or unsupported PCS color options")
    if value["range_mode"] not in ("Auto", "Manual") or value["palette"] not in PALETTES:
        raise ValueError("Unknown PCS range or palette")
    for key in ("centre_zero", "reverse", "show_legend"):
        if type(value[key]) is not bool:
            raise ValueError("PCS display toggles must be boolean")
    for key in ("minimum", "maximum", "tick_step"):
        number = value[key]
        if type(number) not in (int, float) or not math.isfinite(number) or abs(number) > 1e12:
            raise ValueError("PCS display limits must be finite, within ±1e12 ppm")
    if value["minimum"] >= value["maximum"] or value["tick_step"] < 0:
        raise ValueError("Minimum must be less than maximum; tick spacing must be nonnegative")
    if type(value["legend_font"]) is not int or not 0 <= value["legend_font"] <= 96:
        raise ValueError("Legend font must be Auto (0) or 1–96 points")
    if not isinstance(value["missing_color"], str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", value["missing_color"]):
        raise ValueError("Missing PCS color must be #RRGGBB")
    return value


def display_limits(presentation, options):
    """Resolve only visualization limits; retain the original Auto default."""
    if options["range_mode"] == "Manual":
        return options["minimum"], options["maximum"]
    if options["centre_zero"] and presentation.get("color_scale") is not None:
        return tuple(presentation["color_scale"]["range"])
    values = [v for v in presentation.get("pcs_values", []) if v is not None]
    if not values:
        return None
    low, high = min(values), max(values)
    if options["centre_zero"]:
        high = max(abs(low), abs(high), 1e-6)
        return -high, high
    if low == high:
        padding = max(abs(low) * .01, 1e-6)
        return low - padding, high + padding
    return low, high


class PcsColors:
    def __init__(self):
        self.document_id = None
        self.generation = 0
        self.reset()

    def reset(self):
        self.values = dict(DEFAULTS)
        self.generation += 1

    def document(self, document_id):
        if document_id != self.document_id:
            self.document_id = document_id
            self.reset()

    def options(self):
        return dict(self.values)

    def restore(self, options):
        validate_pcs_options(options)
        self.values = dict(options)
        self.generation += 1
