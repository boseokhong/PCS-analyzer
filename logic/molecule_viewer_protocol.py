"""Private prototype messages. Plain values only; not a public plugin API."""

import json
import math
from pathlib import Path

PROTOCOL_VERSION = 1
MAX_SCENE_BYTES = 16 * 1024 * 1024
MAX_ATOMS = 10_000  # Prototype resource budget, not a new application limit.
MAX_EVENT_BYTES = 8192
EVENT_PREFIX = "PCS_MOLECULE_VIEWER "


def validate_scene(scene):
    """Validate the initial snapshot before importing any graphics backend."""
    if (not isinstance(scene, dict) or type(scene.get("protocol_version")) is not int
            or scene.get("protocol_version") != PROTOCOL_VERSION):
        raise ValueError("Unsupported molecular viewer protocol")
    if scene.get("units") != "angstrom" or scene.get("coordinate_frame") != "metal-centred-working-display":
        raise ValueError("Prototype requires explicit working display frame in angstrom")
    for key in ("session_id", "document_id"):
        if not isinstance(scene.get(key), str) or not 1 <= len(scene[key]) <= 256:
            raise ValueError(f"Invalid {key}")
    if "document_title" in scene and (not isinstance(scene["document_title"], str)
                                      or len(scene["document_title"]) > 512):
        raise ValueError("Invalid document_title")
    if "structure_loaded" in scene and type(scene["structure_loaded"]) is not bool:
        raise ValueError("Invalid structure-loaded status")
    if "integrated_export_bridge" in scene and type(scene["integrated_export_bridge"]) is not bool:
        raise ValueError("Invalid integrated export bridge capability")
    revision = scene.get("scene_revision")
    if type(revision) is not int or revision < 0:
        raise ValueError("Invalid scene_revision")
    if type(scene.get("scene_generation", 0)) is not int or scene.get("scene_generation", 0) < 0:
        raise ValueError("Invalid scene_generation")
    keys = ("labels", "ref_ids", "coords", "element_symbols", "colors", "radii")
    if any(not isinstance(scene.get(key), list) for key in keys):
        raise ValueError("Scene atom arrays must be lists")
    count = len(scene["ref_ids"])
    if count > MAX_ATOMS or any(len(scene[key]) != count for key in keys):
        raise ValueError("Scene atom arrays exceed prototype budget or have inconsistent lengths")
    if any(type(ref) is not int or ref <= 0 for ref in scene["ref_ids"]):
        raise ValueError("Invalid atom Ref")
    if len(set(scene["ref_ids"])) != count:
        raise ValueError("Duplicate atom Ref")
    for label, symbol, color, coord, radius in zip(
        scene["labels"], scene["element_symbols"], scene["colors"], scene["coords"], scene["radii"]
    ):
        if not isinstance(label, str) or len(label) > 256:
            raise ValueError("Invalid atom label")
        if not isinstance(symbol, str) or len(symbol) > 16:
            raise ValueError("Invalid element symbol")
        if not isinstance(color, str) or len(color) != 7 or color[0] != "#":
            raise ValueError("Prototype colors must be RGB hex strings")
        try:
            int(color[1:], 16)
        except ValueError as exc:
            raise ValueError("Invalid RGB color") from exc
        if not isinstance(coord, list) or len(coord) != 3:
            raise ValueError("Invalid atom coordinate")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in coord):
            raise ValueError("Nonfinite or nonnumeric coordinate")
        if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
            raise ValueError("Invalid display radius")
    bonds = scene.get("bonds")
    if not isinstance(bonds, list) or len(bonds) > MAX_ATOMS * 12:
        raise ValueError("Invalid or oversized prototype bond list")
    for bond in bonds:
        if not isinstance(bond, list) or len(bond) != 2:
            raise ValueError("Invalid bond pair")
        if any(type(i) is not int or not 0 <= i < count for i in bond) or bond[0] == bond[1]:
            raise ValueError("Invalid bond index")
    if "presentation" in scene:
        _validate_presentation(scene["presentation"], scene["ref_ids"])
    if "display_controls" in scene:
        from logic.molecule_viewer_display import validate_display_controls
        validate_display_controls(scene["display_controls"])
    if "ui_theme" in scene:
        from logic.molecule_viewer_display import validate_ui_theme
        validate_ui_theme(scene["ui_theme"])
    reference = scene.get("reference")
    if "reference" in scene:
        if not isinstance(reference, dict) or any(not isinstance(reference.get(key), list)
                for key in ("labels", "coords", "ref_ids", "colors")):
            raise ValueError("Invalid reference layer")
        refs = reference["ref_ids"]
        if (len(refs) + count > MAX_ATOMS
                or any(type(ref) is not int or ref <= 0 for ref in refs)
                or len(set(refs)) != len(refs)
                or any(len(reference[key]) != len(refs) for key in ("labels", "coords", "colors"))):
            raise ValueError("Invalid or oversized reference arrays")
        for label, xyz, color in zip(reference["labels"], reference["coords"], reference["colors"]):
            if not isinstance(label, str) or len(label) > 256:
                raise ValueError("Invalid reference label")
            if (not isinstance(xyz, list) or len(xyz) != 3
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in xyz)):
                raise ValueError("Invalid reference coordinate")
            if not isinstance(color, str) or len(color) != 7 or color[0] != "#":
                raise ValueError("Invalid reference color")
            try:
                int(color[1:], 16)
            except ValueError as exc:
                raise ValueError("Invalid reference color") from exc
    return scene


def _validate_presentation(value, ref_ids):
    count = len(ref_ids)
    visible_refs = set(ref_ids)
    if not isinstance(value, dict) or value.get("color_mode") not in ("Element", "PCS"):
        raise ValueError("Invalid presentation color mode")
    rgba = value.get("atom_rgba")
    if not isinstance(rgba, list) or len(rgba) != count:
        raise ValueError("Invalid atom presentation colors")
    def valid_color(color):
        return (isinstance(color, list) and len(color) == 4
                and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in color))
    if not all(valid_color(color) for color in rgba):
        raise ValueError("Invalid RGBA presentation color")
    if "pcs_values" in value:
        numbers = value["pcs_values"]
        if (not isinstance(numbers, list) or len(numbers) != count or
                any(v is not None and (type(v) not in (int, float) or not math.isfinite(v)) for v in numbers)):
            raise ValueError("Invalid PCS display values")
    if type(value.get("show_labels")) is not bool:
        raise ValueError("Invalid label visibility")
    averaged = value.get("averaged_ref_ids", [])
    if (not isinstance(averaged, list) or len(averaged) > count
            or any(type(ref) is not int or ref not in visible_refs for ref in averaged)
            or len(set(averaged)) != len(averaged)):
        raise ValueError("Invalid averaged Ref IDs")
    length = value.get("axis_length")
    if (count == 0 and length is not None) or (count and (
            type(length) not in (int, float) or not math.isfinite(length) or length <= 0)):
        raise ValueError("Invalid presentation axis length")
    if not isinstance(value.get("font_family"), str) or not 1 <= len(value["font_family"]) <= 256:
        raise ValueError("Invalid presentation font family")
    font_file = value.get("font_file")
    if (not isinstance(font_file, str) or not 1 <= len(font_file) <= 4096
            or not Path(font_file).is_absolute()):
        raise ValueError("Invalid presentation font file")
    for key in ("label_size", "axis_label_size", "tick_size"):
        if type(value.get(key)) is not int or not 1 <= value[key] <= 128:
            raise ValueError("Invalid presentation font size")
    scale = value.get("color_scale")
    if scale is not None:
        if not isinstance(scale, dict) or value["color_mode"] != "PCS" or scale.get("title") != "PCS (ppm)":
            raise ValueError("Invalid PCS scale")
        limits = scale.get("range")
        if (not isinstance(limits, list) or len(limits) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in limits)
                or limits[0] >= 0 or limits[1] != -limits[0]):
            raise ValueError("Invalid symmetric PCS range")
        table = scale.get("rgba")
        if not isinstance(table, list) or len(table) != 256 or not all(valid_color(color) for color in table):
            raise ValueError("Invalid PCS lookup table")


def read_scene(path):
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_SCENE_BYTES + 1)
    if len(data) > MAX_SCENE_BYTES:
        raise ValueError("Scene exceeds prototype message budget")
    return validate_scene(json.loads(data))


def event_for(scene, kind, **values):
    return dict(protocol_version=PROTOCOL_VERSION, session_id=scene["session_id"],
                document_id=scene["document_id"], scene_revision=scene["scene_revision"],
                scene_generation=scene.get("scene_generation", 0), type=kind, **values)


def matches_scene(event, scene):
    return all(type(event.get(key)) is type(scene.get(key, 0))
               and event.get(key) == scene.get(key, 0)
               for key in ("protocol_version", "session_id", "document_id",
                           "scene_revision", "scene_generation"))


def encode_event(event):
    result = EVENT_PREFIX + json.dumps(event, allow_nan=False, ensure_ascii=True) + "\n"
    if len(result.encode("utf-8")) > MAX_EVENT_BYTES:
        raise ValueError("Viewer event exceeds protocol budget")
    return result
