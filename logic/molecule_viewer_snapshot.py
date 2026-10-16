"""Build a limited prototype snapshot from the preserved host display pipeline."""

from logic.chem_constants import CPK_COLORS, covalent_radii
from logic.molecule_viewer_protocol import PROTOCOL_VERSION, validate_scene


def averaging_legend_entries(scene):
    reference = scene.get("reference")
    entries = ["Reference"] if reference and reference["coords"] else []
    averaged = scene.get("presentation", {}).get("averaged_ref_ids", [])
    visible_refs = set(scene["ref_ids"])
    if any(ref in visible_refs for ref in averaged):
        entries.append("Ensemble avg")
    return entries


def prototype_scene_from_view_data(data, *, document_id, scene_revision, bonds, atom_scale=1.0):
    """Reuse host coordinates/bond guides; do not calculate PCS, fits or topology.

    Reference geometry is a separate visual layer, never new observations/bonds.
    """
    if data is None:
        data = {"labels": [], "coords": [], "ref_ids": []}
    scale = float(atom_scale)
    if not 0.10 <= scale <= 10.0:
        raise ValueError("Invalid prototype atom scale")
    labels = [str(label) for label in data["labels"]]
    symbols = []
    for label in labels:
        text = label.strip()
        if text.startswith("MeH@"):
            symbol = "H"
        elif text.startswith("CF3F@"):
            symbol = "F"
        elif len(text) >= 2 and text[1].islower():
            symbol = text[:2]
        else:
            symbol = text[:1]
        symbols.append(symbol)
    scene = {
        "protocol_version": PROTOCOL_VERSION, "session_id": "unlaunched",
        "document_id": str(document_id), "scene_revision": scene_revision,
        "units": "angstrom", "coordinate_frame": "metal-centred-working-display",
        "labels": labels, "element_symbols": symbols,
        "ref_ids": [int(ref) for ref in data["ref_ids"]],
        "coords": [[float(v) for v in row] for row in data["coords"]],
        "colors": [CPK_COLORS.get(el, CPK_COLORS["default"]) for el in symbols],
        "radii": [max(0.05, covalent_radii.get(el, 1.0) * 0.25 * scale) for el in symbols],
        "bonds": [[int(i), int(j)] for i, j in bonds],
    }
    if any(key in data for key in ("reference_coords", "reference_labels", "reference_ref_ids")):
        if any(key not in data for key in ("reference_coords", "reference_labels", "reference_ref_ids")):
            raise ValueError("Reference overlays require coordinates, labels and Ref IDs")
        # Reuse the same element/color parsing without transforming coordinates.
        reference = prototype_scene_from_view_data(
            {"labels": data["reference_labels"], "coords": data["reference_coords"],
             "ref_ids": data["reference_ref_ids"]},
            document_id=document_id, scene_revision=scene_revision, bonds=[],
        )
        scene["reference"] = {key: reference[key] for key in ("labels", "coords", "ref_ids", "colors")}
    return validate_scene(scene)
