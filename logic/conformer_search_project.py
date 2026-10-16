"""Plain, versioned Conformer Search records; no GUI or optimizer imports."""
from copy import deepcopy
import math
import numpy as np

SCHEMA_VERSION = 1


def plain(value):
    """Encode numerical records as JSON values, retaining undefined metrics."""
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, (set, frozenset)):
        return plain(sorted(value))
    if isinstance(value, dict):
        return {str(k): (None if k in ("r2", "q_factor") and
                         isinstance(v, (float, np.floating)) and not math.isfinite(v)
                         else plain(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Conformer Search record contains a nonfinite number")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported Conformer Search value: {type(value).__name__}")


def parse_payload(payload):
    if payload is None:
        return None
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported Conformer Search project schema")
    record = plain(deepcopy(payload))
    inputs = record["inputs"]
    elements = inputs["elements"]
    n = len(elements)
    if not n or any(not isinstance(e, str) or not e.strip() for e in elements):
        raise ValueError("Invalid search element list")

    def coords(value):
        array = np.asarray(value, dtype=float)
        if array.shape != (n, 3) or not np.isfinite(array).all():
            raise ValueError("Search coordinates must be finite N-by-3 arrays")

    def index(value):
        if type(value) is not int or not 0 <= value < n:
            raise ValueError("Invalid search atom index")

    coords(inputs["coords"])
    record.setdefault("editor_graph", {k:deepcopy(inputs[k]) for k in
                                      ("bond_options", "manual_add", "manual_remove", "metal_index")})
    graph = record["editor_graph"]
    if not isinstance(graph["bond_options"], dict) or not isinstance(inputs["bond_options"], dict):
        raise ValueError("Invalid search bond options")
    for options in (graph["bond_options"], inputs["bond_options"]):
        if "allow_metal_carbon" in options and type(options["allow_metal_carbon"]) is not bool:
            raise ValueError("Invalid metal-carbon bond toggle")
        for key in ("scale", "max_metal_bond_distance"):
            if key in options and options[key] is not None:
                if type(options[key]) not in (int, float) or not math.isfinite(options[key]):
                    raise ValueError("Invalid numerical bond option")
        if "coord_donors" in options and (not isinstance(options["coord_donors"], list)
                or any(not isinstance(e, str) for e in options["coord_donors"])):
            raise ValueError("Invalid coordination element list")
    index(graph["metal_index"])
    for bond in graph["manual_add"] + graph["manual_remove"]:
        if len(bond) != 2 or bond[0] == bond[1]:
            raise ValueError("Invalid editor bond")
        for value in bond:
            index(value)
    index(inputs["metal_index"])
    for value in inputs["fixed"]:
        index(value)
    for value in record["ui_fixed"]:
        index(value)
    for bond in record["ui_bonds"]:
        if len(bond) != 2 or bond[0] == bond[1]:
            raise ValueError("Invalid selected search bond")
        for value in bond:
            index(value)
    for key in ("selected_bonds", "manual_add", "manual_remove"):
        for bond in inputs[key]:
            if len(bond) != 2 or bond[0] == bond[1]:
                raise ValueError("Invalid search bond")
            for value in bond:
                index(value)
    for atom, observed in inputs["pcs"]:
        index(atom)
        if not math.isfinite(float(observed)):
            raise ValueError("Invalid search PCS observation")
    candidates = record["candidates"]
    reports = record["reports"]
    if not candidates or len(reports) != len(candidates) or any(not isinstance(r, str) for r in reports):
        raise ValueError("Search candidates require matching report strings")
    for candidate in candidates:
        coords(candidate["coords_opt"])
        if candidate.get("mode") not in ("dihedral", "rigid"):
            raise ValueError("Invalid search result mode")
        if type(candidate["rmsd"]) not in (int, float) or not math.isfinite(candidate["rmsd"]) or candidate["rmsd"] < 0:
            raise ValueError("Invalid search RMSD")
        if type(candidate["n_clashes"]) is not int or candidate["n_clashes"] < 0:
            raise ValueError("Invalid search clash count")
        for key in ("fixed_displacement", "max_angular_dev_deg", "max_radial_change", "min_donor_dist", "angle_opt", "anchor_drift"):
            if key in candidate and type(candidate[key]) not in (int, float):
                raise ValueError(f"Invalid candidate numeric field: {key}")
        for row in candidate["per_atom"]:
            if len(row) != 5:
                raise ValueError("Invalid search per-atom result")
            index(row[0])
            if row[1] != elements[row[0]]:
                raise ValueError("Search per-atom element mismatch")
            if not all(type(v) in (int, float) and math.isfinite(v) for v in row[2:]):
                raise ValueError("Invalid search per-atom values")
        for key in ("r2", "q_factor"):
            if key not in candidate:
                raise ValueError(f"Missing search metric: {key}")
            if candidate[key] is not None and type(candidate[key]) not in (int, float):
                raise ValueError(f"Invalid search metric: {key}")
        bonds = candidate["selected_bonds"]
        for bond in bonds + candidate.get("locked_bonds", []):
            if len(bond) != 2 or bond[0] == bond[1]:
                raise ValueError("Invalid candidate bond")
            for value in bond:
                index(value)
        if candidate["mode"] == "dihedral":
            for key in ("angles_opt", "angles_init", "angle_deltas"):
                values = np.asarray(candidate[key], dtype=float)
                if values.shape != (len(bonds),) or not np.isfinite(values).all():
                    raise ValueError("Invalid candidate dihedral angles")
        else:
            for value in candidate["anchor_atoms"]:
                index(value)
    for key in ("selected_index", "preview_index"):
        value = record[key]
        if type(value) is not int or not (-1 if key == "preview_index" else 0) <= value < len(candidates):
            raise ValueError("Invalid selected search candidate")
    if type(record["applied"]) is not bool:
        raise ValueError("Invalid search applied flag")
    record.setdefault("subtab_index", 3)
    if type(record["subtab_index"]) is not int or not 0 <= record["subtab_index"] < 4:
        raise ValueError("Invalid search subtab")
    if not isinstance(record["context"], dict) or not isinstance(inputs["controls"], dict):
        raise ValueError("Invalid recorded search context/controls")
    if not isinstance(record["controls"], dict):
        raise ValueError("Invalid current search controls")
    for controls in (record["controls"], inputs["controls"]):
        for key, value in controls.items():
            if key in ("_anchor_mode_var", "_use_global_var"):
                if type(value) is not bool:
                    raise ValueError("Invalid search toggle")
            elif not isinstance(value, str):
                raise ValueError("Invalid search control text")
    for key in ("atoms", "refs", "pcs", "metal", "tensor", "orientation", "geometry_preview"):
        if key not in record["context"]:
            raise ValueError(f"Missing search host context: {key}")
    if len(record["context"]["atoms"]) != n or len(record["context"]["refs"]) != n:
        raise ValueError("Search context atom/Ref count mismatch")
    return record


def candidate_values(record):
    values = deepcopy(record["candidates"])
    for candidate in values:
        candidate["coords_opt"] = np.asarray(candidate["coords_opt"], dtype=float)
        for key in ("r2", "q_factor"):
            if candidate[key] is None:
                candidate[key] = float("nan")
    return values


def preview_payload(record):
    idx = record["preview_index"]
    if idx < 0:
        return None
    candidate = candidate_values(record)[idx]
    inputs = record["inputs"]
    return dict(result=candidate, elements=list(inputs["elements"]),
                coords_initial=np.asarray(inputs["coords"], dtype=float),
                coords_opt=candidate["coords_opt"].copy(), report=record["reports"][idx],
                metal_idx=inputs["metal_index"])


def context_matches(record, current):
    """Compare recorded host conditions; permit the explicitly applied geometry."""
    expected = deepcopy(record["context"])
    if record["applied"] and record["preview_index"] >= 0:
        expected["atoms"] = [[e, *p] for e, p in zip(record["inputs"]["elements"],
                                record["candidates"][record["preview_index"]]["coords_opt"])]

    def same(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7)
        return a == b
    return same(expected, current)
