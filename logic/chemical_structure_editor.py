"""Chemical-structure editing helpers for Chemical Map D5b.4.

This module keeps three concerns separate:

* ``StructureGraph`` stores chemistry/topology.
* persistent manual bond overrides store user corrections to that topology.
* ``chemical_map_2d_coordinates`` stores depiction-only 2D coordinates.

No function in this module changes the loaded XYZ coordinates.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from statistics import median
from typing import Mapping

from logic.chemical_map import (
    ENGINE_COORDGEN,
    ENGINE_GRAPH_2D,
    ENGINE_RDKIT_2D,
    LAYOUT_2D_CHEMICAL,
    ChemicalMapModel,
    build_chemical_map_model,
    layout_chemical_map,
    normalize_layout_engine,
)
from logic.ligand_grouping import LigandAnalysis
from logic.structure_graph import StructureGraph
from logic.structure_graph_state import refresh_structure_graph_in_state


BOND_SINGLE = "single"
BOND_DOUBLE = "double"
BOND_TRIPLE = "triple"
BOND_AROMATIC = "aromatic"
BOND_DELOCALIZED = "delocalized"
BOND_UNKNOWN = "unknown"

BOND_TYPES = (
    BOND_SINGLE,
    BOND_DOUBLE,
    BOND_TRIPLE,
    BOND_AROMATIC,
    BOND_DELOCALIZED,
    BOND_UNKNOWN,
)

_BOND_ORDER = {
    BOND_SINGLE: 1.0,
    BOND_DOUBLE: 2.0,
    BOND_TRIPLE: 3.0,
    # These numeric values are depiction semantics, not measured bond orders.
    BOND_AROMATIC: 1.5,
    BOND_DELOCALIZED: 1.25,
    BOND_UNKNOWN: None,
}
_NOTE_PREFIX = "chemical_map_bond_type="


@dataclass(frozen=True)
class BondEditOutcome:
    edited_pairs: tuple[tuple[int, int], ...]
    propagated: bool
    warning: str | None = None


@dataclass(frozen=True)
class LayoutCleanupOutcome:
    fragment_ids: tuple[int, ...]
    propagated: bool
    warning: str | None = None


def bond_type_to_order(bond_type: str):
    key = str(bond_type or "").strip().lower()
    if key not in _BOND_ORDER:
        raise ValueError(f"unsupported bond type: {bond_type!r}")
    return _BOND_ORDER[key]


def bond_type_note(bond_type: str) -> str:
    key = str(bond_type or "").strip().lower()
    if key not in _BOND_ORDER:
        raise ValueError(f"unsupported bond type: {bond_type!r}")
    return f"{_NOTE_PREFIX}{key}"


def bond_type_from_note(note: object, order=None) -> str:
    text = str(note or "")
    if _NOTE_PREFIX in text:
        value = text.split(_NOTE_PREFIX, 1)[1].split(";", 1)[0].strip().lower()
        if value in _BOND_ORDER:
            return value
    try:
        numeric = float(order) if order is not None else None
    except Exception:
        numeric = None
    if numeric is None:
        return BOND_UNKNOWN
    if numeric >= 2.5:
        return BOND_TRIPLE
    if numeric >= 1.75:
        return BOND_DOUBLE
    if 1.40 <= numeric < 1.75:
        return BOND_AROMATIC
    if 1.10 < numeric < 1.40:
        return BOND_DELOCALIZED
    return BOND_SINGLE


def _pair(a: int, b: int) -> tuple[int, int]:
    x, y = int(a), int(b)
    if x == y:
        raise ValueError("a bond cannot connect an atom to itself")
    return (x, y) if x < y else (y, x)


def _normalized_additions(state: dict) -> dict[tuple[int, int], dict]:
    additions: dict[tuple[int, int], dict] = {}
    for record in state.get("structure_graph_manual_add_edges", []) or []:
        if isinstance(record, dict):
            key = _pair(record["atom1_id"], record["atom2_id"])
            additions[key] = {
                "atom1_id": key[0],
                "atom2_id": key[1],
                "kind": str(record.get("kind", "covalent")),
                "order": record.get("order"),
                "note": str(record.get("note", "")),
            }
            continue
        values = tuple(record)
        if len(values) < 3:
            continue
        key = _pair(values[0], values[1])
        additions[key] = {
            "atom1_id": key[0],
            "atom2_id": key[1],
            "kind": str(values[2]),
            "order": values[3] if len(values) >= 4 else None,
            "note": str(values[4] if len(values) >= 5 else ""),
        }
    return additions


def _normalized_removals(state: dict) -> set[tuple[int, int]]:
    output = set()
    for record in state.get("structure_graph_manual_remove_edges", []) or []:
        try:
            output.add(_pair(record[0], record[1]))
        except Exception:
            continue
    return output


def _write_overrides(
    state: dict,
    additions: Mapping[tuple[int, int], dict],
    removals: set[tuple[int, int]],
) -> None:
    state["structure_graph_manual_add_edges"] = [
        dict(additions[key]) for key in sorted(additions)
    ]
    state["structure_graph_manual_remove_edges"] = [
        [a, b] for a, b in sorted(removals)
    ]


def _group_mapping_for_fragment(group, fragment_id: int) -> tuple[dict[int, int], bool]:
    """Return representative->fragment mapping and ambiguity."""
    fragment_id = int(fragment_id)
    item = next(
        (entry for entry in group.correspondences if entry.fragment_id == fragment_id),
        None,
    )
    if item is None:
        raise KeyError(f"no ligand correspondence for fragment {fragment_id}")
    return item.as_dict(), bool(item.ambiguous)


def matching_bond_pairs(
    analysis: LigandAnalysis,
    *,
    fragment_id: int,
    atom1_id: int,
    atom2_id: int,
) -> tuple[tuple[tuple[int, int], ...], str | None]:
    """Map a bond to corresponding ligand copies when mapping is unique.

    The current ligand can be any member of the group.  Correspondences are
    stored representative->member, so the current mapping is inverted first.
    """
    group = analysis.group_for_fragment_id(int(fragment_id))
    current_map, current_ambiguous = _group_mapping_for_fragment(group, fragment_id)
    if current_ambiguous:
        return ((_pair(atom1_id, atom2_id)),), (
            "Matching-ligand propagation was skipped because the current ligand "
            "has an ambiguous atom correspondence."
        )

    inverse = {member: representative for representative, member in current_map.items()}
    try:
        rep_a = inverse[int(atom1_id)]
        rep_b = inverse[int(atom2_id)]
    except KeyError:
        return ((_pair(atom1_id, atom2_id)),), (
            "The selected bond is not part of the current ligand correspondence."
        )

    output: list[tuple[int, int]] = []
    for item in group.correspondences:
        mapping = item.as_dict()
        if item.ambiguous:
            return ((_pair(atom1_id, atom2_id)),), (
                "Matching-ligand propagation was skipped because at least one "
                "ligand copy has an ambiguous atom correspondence."
            )
        if rep_a not in mapping or rep_b not in mapping:
            continue
        output.append(_pair(mapping[rep_a], mapping[rep_b]))

    unique = tuple(sorted(set(output)))
    return unique or ((_pair(atom1_id, atom2_id)),), None


def set_bond_type_in_state(
    state: dict,
    analysis: LigandAnalysis,
    *,
    fragment_id: int,
    atom1_id: int,
    atom2_id: int,
    bond_type: str,
    propagate_matching: bool = False,
) -> BondEditOutcome:
    bond_type = str(bond_type).strip().lower()
    order = bond_type_to_order(bond_type)
    pairs = (_pair(atom1_id, atom2_id),)
    warning = None
    if propagate_matching:
        pairs, warning = matching_bond_pairs(
            analysis,
            fragment_id=fragment_id,
            atom1_id=atom1_id,
            atom2_id=atom2_id,
        )

    additions = _normalized_additions(state)
    removals = _normalized_removals(state)
    for a, b in pairs:
        removals.discard((a, b))
        additions[(a, b)] = {
            "atom1_id": a,
            "atom2_id": b,
            "kind": "covalent",
            "order": order,
            "note": bond_type_note(bond_type),
        }
    _write_overrides(state, additions, removals)
    refresh_structure_graph_in_state(state)
    return BondEditOutcome(pairs, len(pairs) > 1, warning)


def remove_bond_in_state(
    state: dict,
    analysis: LigandAnalysis,
    *,
    fragment_id: int,
    atom1_id: int,
    atom2_id: int,
    propagate_matching: bool = False,
) -> BondEditOutcome:
    pairs = (_pair(atom1_id, atom2_id),)
    warning = None
    if propagate_matching:
        pairs, warning = matching_bond_pairs(
            analysis,
            fragment_id=fragment_id,
            atom1_id=atom1_id,
            atom2_id=atom2_id,
        )
    additions = _normalized_additions(state)
    removals = _normalized_removals(state)
    for pair in pairs:
        additions.pop(pair, None)
        removals.add(pair)
    _write_overrides(state, additions, removals)
    refresh_structure_graph_in_state(state)
    return BondEditOutcome(pairs, len(pairs) > 1, warning)


def reset_bond_override_in_state(
    state: dict,
    analysis: LigandAnalysis,
    *,
    fragment_id: int,
    atom1_id: int,
    atom2_id: int,
    propagate_matching: bool = False,
) -> BondEditOutcome:
    pairs = (_pair(atom1_id, atom2_id),)
    warning = None
    if propagate_matching:
        pairs, warning = matching_bond_pairs(
            analysis,
            fragment_id=fragment_id,
            atom1_id=atom1_id,
            atom2_id=atom2_id,
        )
    additions = _normalized_additions(state)
    removals = _normalized_removals(state)
    for pair in pairs:
        additions.pop(pair, None)
        removals.discard(pair)
    _write_overrides(state, additions, removals)
    refresh_structure_graph_in_state(state)
    return BondEditOutcome(pairs, len(pairs) > 1, warning)


def add_bond_in_state(
    state: dict,
    analysis: LigandAnalysis,
    *,
    fragment_id: int,
    atom1_id: int,
    atom2_id: int,
    bond_type: str = BOND_SINGLE,
    propagate_matching: bool = False,
) -> BondEditOutcome:
    return set_bond_type_in_state(
        state,
        analysis,
        fragment_id=fragment_id,
        atom1_id=atom1_id,
        atom2_id=atom2_id,
        bond_type=bond_type,
        propagate_matching=propagate_matching,
    )


def _layout_store(state: dict) -> dict:
    store = state.get("chemical_map_2d_coordinates")
    if not isinstance(store, dict):
        store = {}
        state["chemical_map_2d_coordinates"] = store
    return store


def get_2d_override(state: dict, fragment_id: int) -> dict[int, tuple[float, float]] | None:
    raw = _layout_store(state).get(str(int(fragment_id)))
    if not isinstance(raw, dict):
        return None
    result: dict[int, tuple[float, float]] = {}
    for key, value in raw.items():
        try:
            x, y = value
            result[int(key)] = (float(x), float(y))
        except Exception:
            continue
    return result or None


def get_2d_override_source(state: dict, fragment_id: int) -> str | None:
    """Return the semantic source of a stored 2D depiction override.

    D5b.4 stores both cleanup-generated layouts and true manual edits in the
    same coordinate store.  The metadata lets the UI distinguish them so a
    layout-engine change can silently discard cleanup coordinates while still
    protecting user-authored manual adjustments.
    """
    meta = state.get("chemical_map_2d_layout_meta")
    if not isinstance(meta, dict):
        return None
    item = meta.get(str(int(fragment_id)))
    if not isinstance(item, dict):
        return None
    source = str(item.get("source", "") or "").strip()
    return source or None


def is_manual_2d_override_source(source: object) -> bool:
    """Return True only for depiction changes explicitly made by the user.

    Unknown/legacy coordinate stores are handled conservatively by the UI when
    coordinates exist but metadata is absent.
    """
    text = str(source or "").strip().lower()
    return text == "manual" or text.startswith("manual-")


def set_2d_override(
    state: dict,
    fragment_id: int,
    coordinates: Mapping[int, tuple[float, float]],
    *,
    source: str = "manual",
) -> None:
    store = _layout_store(state)
    store[str(int(fragment_id))] = {
        str(int(atom_id)): [float(x), float(y)]
        for atom_id, (x, y) in coordinates.items()
    }
    meta = state.setdefault("chemical_map_2d_layout_meta", {})
    meta[str(int(fragment_id))] = {"source": str(source)}


def clear_2d_override(state: dict, fragment_id: int) -> None:
    _layout_store(state).pop(str(int(fragment_id)), None)
    meta = state.get("chemical_map_2d_layout_meta")
    if isinstance(meta, dict):
        meta.pop(str(int(fragment_id)), None)


def set_atom_2d_position(
    state: dict,
    fragment_id: int,
    base_coordinates: Mapping[int, tuple[float, float]],
    atom_id: int,
    x: float,
    y: float,
) -> None:
    coords = dict(get_2d_override(state, fragment_id) or base_coordinates)
    coords[int(atom_id)] = (float(x), float(y))
    set_2d_override(state, fragment_id, coords, source="manual")



def transform_2d_layout(
    state: dict,
    fragment_id: int,
    base_coordinates: Mapping[int, tuple[float, float]],
    *,
    rotate_degrees: float = 0.0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> None:
    """Apply a depiction-only rigid transform around the fragment centroid."""
    import math

    coords = dict(get_2d_override(state, fragment_id) or base_coordinates)
    if not coords:
        return
    cx = sum(x for x, _ in coords.values()) / len(coords)
    cy = sum(y for _, y in coords.values()) / len(coords)
    theta = math.radians(float(rotate_degrees))
    c = math.cos(theta)
    s = math.sin(theta)
    transformed = {}
    for atom_id, (x, y) in coords.items():
        dx = x - cx
        dy = y - cy
        if flip_horizontal:
            dx = -dx
        if flip_vertical:
            dy = -dy
        rx = c * dx - s * dy
        ry = s * dx + c * dy
        transformed[int(atom_id)] = (cx + rx, cy + ry)
    set_2d_override(state, fragment_id, transformed, source="manual-transform")



def _normalize_cleanup_coordinates(
    coordinates: Mapping[int, tuple[float, float]],
    *,
    margin: float = 0.14,
) -> dict[int, tuple[float, float]]:
    """Scale a depiction into a stable square without changing its angles.

    Cleanup deliberately normalizes the *heavy-atom* skeleton before explicit
    hydrogen positions are added.  Hidden H atoms therefore cannot shrink or
    distort the carbon/heteroatom backbone that the chemist actually reads.
    """
    if not coordinates:
        return {}
    ids = tuple(coordinates)
    xs = [float(coordinates[atom_id][0]) for atom_id in ids]
    ys = [float(coordinates[atom_id][1]) for atom_id in ids]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y
    span = max(span_x, span_y, 1.0e-9)
    usable = max(1.0 - 2.0 * float(margin), 0.1)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    scale = usable / span
    return {
        int(atom_id): (
            0.5 + (float(coordinates[atom_id][0]) - cx) * scale,
            0.5 + (float(coordinates[atom_id][1]) - cy) * scale,
        )
        for atom_id in ids
    }


def _angle_distance(a: float, b: float) -> float:
    diff = abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)
    return diff


def _place_explicit_hydrogens(
    model: ChemicalMapModel,
    heavy_coordinates: Mapping[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """Place explicit H atoms after the heavy-atom cleanup.

    Explicit hydrogens are important PCS targets, but they must not control the
    skeletal layout.  We therefore place them into the largest free angular
    sectors around their parent atom after CoordGen/rdDepictor has regularized
    the heavy-atom graph.  This preserves readable 120-degree iPr/tBu backbones
    while still giving every explicit H a usable 2D coordinate.
    """
    result = {int(k): (float(v[0]), float(v[1])) for k, v in heavy_coordinates.items()}
    atoms = {atom.atom_id: atom for atom in model.atoms}
    adjacency: dict[int, list[int]] = {atom_id: [] for atom_id in atoms}
    for bond in model.bonds:
        adjacency.setdefault(bond.atom1_id, []).append(bond.atom2_id)
        adjacency.setdefault(bond.atom2_id, []).append(bond.atom1_id)

    heavy_lengths: list[float] = []
    for bond in model.bonds:
        if bond.atom1_id not in result or bond.atom2_id not in result:
            continue
        x1, y1 = result[bond.atom1_id]
        x2, y2 = result[bond.atom2_id]
        length = math.hypot(x2 - x1, y2 - y1)
        if length > 1.0e-9:
            heavy_lengths.append(length)
    typical = median(heavy_lengths) if heavy_lengths else 0.16
    h_length = max(0.035, min(0.085, typical * 0.56))

    for center_id, center in atoms.items():
        if center.element == "H" or center_id not in result:
            continue
        hydrogens = [
            neighbor
            for neighbor in adjacency.get(center_id, ())
            if atoms.get(neighbor) is not None and atoms[neighbor].element == "H"
        ]
        if not hydrogens:
            continue
        cx, cy = result[center_id]
        used_angles: list[float] = []
        for neighbor in adjacency.get(center_id, ()):
            if neighbor in hydrogens or neighbor not in result:
                continue
            nx, ny = result[neighbor]
            if math.hypot(nx - cx, ny - cy) > 1.0e-9:
                used_angles.append(math.atan2(ny - cy, nx - cx))

        # Dense candidate sampling makes the placement deterministic while
        # choosing the largest empty sectors around the parent atom.
        candidate_angles = [2.0 * math.pi * i / 72.0 for i in range(72)]
        for hydrogen_id in sorted(hydrogens):
            if used_angles:
                angle = max(
                    candidate_angles,
                    key=lambda value: min(
                        _angle_distance(value, occupied) for occupied in used_angles
                    ),
                )
            else:
                angle = 2.0 * math.pi * (len(used_angles) % max(len(hydrogens), 1)) / max(
                    len(hydrogens), 1
                )
            result[int(hydrogen_id)] = (
                cx + h_length * math.cos(angle),
                cy + h_length * math.sin(angle),
            )
            used_angles.append(angle)

    # Defensive fallback for unusual H-H fragments or disconnected explicit H.
    for atom in model.atoms:
        if atom.atom_id not in result:
            result[atom.atom_id] = (0.5, 0.5)
    return result


def _chemical_cleanup_coordinates(
    model: ChemicalMapModel,
    *,
    engine: str,
) -> tuple[dict[int, tuple[float, float]], str] | None:
    """Generate a chemistry-aware heavy-atom depiction with RDKit.

    The current user-edited bond semantics are passed to RDKit for single,
    double and triple bonds.  Aromatic/delocalized edges keep their topology
    but are not re-interpreted chemically here; Chemical Map still renders
    those semantics itself.  This avoids silently changing chemistry while
    allowing bond order, ring topology and atom degree to drive conventional
    2D geometry (e.g. a Y-shaped ~120-degree isopropyl backbone).
    """
    normalized_engine = normalize_layout_engine(engine)
    if normalized_engine == ENGINE_GRAPH_2D:
        return None
    try:
        from rdkit import Chem
    except Exception:
        return None

    heavy_atoms = tuple(atom for atom in model.atoms if atom.element != "H")
    if not heavy_atoms:
        return None

    try:
        rw = Chem.RWMol()
        index_by_atom_id: dict[int, int] = {}
        for atom in heavy_atoms:
            index_by_atom_id[atom.atom_id] = rw.AddAtom(Chem.Atom(atom.element))

        for bond in model.bonds:
            if bond.atom1_id not in index_by_atom_id or bond.atom2_id not in index_by_atom_id:
                continue
            bond_type = str(getattr(bond, "bond_type", "unknown") or "unknown")
            order = getattr(bond, "order", None)
            if bond_type == BOND_TRIPLE or (order is not None and float(order) >= 2.5):
                rd_bond_type = Chem.BondType.TRIPLE
            elif bond_type == BOND_DOUBLE or (order is not None and float(order) >= 1.75):
                rd_bond_type = Chem.BondType.DOUBLE
            else:
                # Aromatic/delocalized drawing semantics remain explicit in the
                # StructureGraph renderer; the cleanup does not infer resonance.
                rd_bond_type = Chem.BondType.SINGLE
            rw.AddBond(
                index_by_atom_id[bond.atom1_id],
                index_by_atom_id[bond.atom2_id],
                rd_bond_type,
            )

        rd_mol = rw.GetMol()
        if normalized_engine == ENGINE_RDKIT_2D:
            from rdkit.Chem import rdDepictor

            rdDepictor.Compute2DCoords(
                rd_mol,
                canonOrient=True,
                clearConfs=True,
                nFlipsPerSample=3,
                nSample=80,
                sampleSeed=0xD5B44,
                permuteDeg4Nodes=True,
                bondLength=1.5,
                forceRDKit=True,
                useRingTemplates=True,
            )
            backend = "RDKit 2D chemical cleanup"
        else:
            try:
                from rdkit.Chem import rdCoordGen

                # CoordGen's defaults already enforce conventional chemical
                # drawing geometry for rings and branched heavy-atom graphs.
                rdCoordGen.AddCoords(rd_mol)
                backend = "RDKit CoordGen chemical cleanup"
            except Exception:
                from rdkit.Chem import rdDepictor

                rdDepictor.Compute2DCoords(
                    rd_mol,
                    canonOrient=True,
                    clearConfs=True,
                    nFlipsPerSample=3,
                    nSample=80,
                    sampleSeed=0xD5B44,
                    permuteDeg4Nodes=True,
                    bondLength=1.5,
                    forceRDKit=True,
                    useRingTemplates=True,
                )
                backend = "RDKit 2D chemical cleanup (CoordGen unavailable)"

        conf = rd_mol.GetConformer()
        raw_heavy = {}
        for atom_id, index in index_by_atom_id.items():
            pos = conf.GetAtomPosition(index)
            raw_heavy[int(atom_id)] = (float(pos.x), float(pos.y))
        normalized_heavy = _normalize_cleanup_coordinates(raw_heavy)
        return _place_explicit_hydrogens(model, normalized_heavy), backend
    except Exception:
        # Cleanup must remain available even for unusual ligand valence states.
        return None


def cleanup_structure_2d(
    state: dict,
    graph: StructureGraph,
    analysis: LigandAnalysis,
    *,
    group_id: int,
    fragment_id: int,
    engine: str = ENGINE_COORDGEN,
    propagate_matching: bool = True,
) -> LayoutCleanupOutcome:
    """Rebuild depiction coordinates from the current chemical graph only.

    Chemistry/topology and loaded XYZ coordinates are untouched.  When unique
    ligand correspondences exist, the same cleaned orientation is transferred
    to matching ligand copies.
    """
    model = build_chemical_map_model(
        graph,
        analysis,
        group_id=group_id,
        fragment_id=fragment_id,
        label_overrides=state.get("ref_label_overrides", {}) or {},
    )
    cleaned = _chemical_cleanup_coordinates(model, engine=engine)
    if cleaned is not None:
        base, cleanup_backend = cleaned
    else:
        layout = layout_chemical_map(
            graph,
            model,
            mode=LAYOUT_2D_CHEMICAL,
            engine=engine,
        )
        base = dict(layout.coordinates)
        cleanup_backend = layout.backend
    set_2d_override(
        state,
        fragment_id,
        base,
        source=f"cleanup:{cleanup_backend}",
    )

    if not propagate_matching:
        return LayoutCleanupOutcome((int(fragment_id),), False, None)

    group = analysis.group_for_fragment_id(fragment_id)
    current_map, current_ambiguous = _group_mapping_for_fragment(group, fragment_id)
    if current_ambiguous:
        return LayoutCleanupOutcome(
            (int(fragment_id),),
            False,
            "Matching-ligand cleanup was skipped because atom correspondence is ambiguous.",
        )
    inverse = {member: rep for rep, member in current_map.items()}

    completed = [int(fragment_id)]
    for item in group.correspondences:
        target_fragment = int(item.fragment_id)
        if target_fragment == int(fragment_id):
            continue
        if item.ambiguous:
            return LayoutCleanupOutcome(
                tuple(completed),
                len(completed) > 1,
                "Some matching ligands were not updated because their correspondence is ambiguous.",
            )
        target_map = item.as_dict()
        target_coords: dict[int, tuple[float, float]] = {}
        for current_atom, xy in base.items():
            rep = inverse.get(int(current_atom))
            if rep is None or rep not in target_map:
                continue
            target_coords[int(target_map[rep])] = tuple(xy)
        if target_coords:
            set_2d_override(
                state,
                target_fragment,
                target_coords,
                source=f"cleanup-matched:{layout.backend}",
            )
            completed.append(target_fragment)

    return LayoutCleanupOutcome(tuple(completed), len(completed) > 1, None)


def reset_2d_layout(
    state: dict,
    analysis: LigandAnalysis,
    *,
    fragment_id: int,
    include_matching: bool = False,
) -> tuple[int, ...]:
    group = analysis.group_for_fragment_id(fragment_id)
    targets = [int(fragment_id)]
    if include_matching:
        targets = [int(value) for value in group.member_fragment_ids]
    for target in targets:
        clear_2d_override(state, target)
    return tuple(targets)


def snapshot_editor_state(state: dict) -> dict:
    keys = (
        "structure_graph_manual_add_edges",
        "structure_graph_manual_remove_edges",
        "chemical_map_2d_coordinates",
        "chemical_map_2d_layout_meta",
    )
    return {key: deepcopy(state.get(key)) for key in keys}


def restore_editor_state(state: dict, snapshot: Mapping[str, object]) -> None:
    for key, value in snapshot.items():
        if value is None:
            state.pop(key, None)
        else:
            state[key] = deepcopy(value)
    refresh_structure_graph_in_state(state)


__all__ = [
    "BOND_AROMATIC",
    "BOND_DELOCALIZED",
    "BOND_DOUBLE",
    "BOND_SINGLE",
    "BOND_TRIPLE",
    "BOND_TYPES",
    "BOND_UNKNOWN",
    "BondEditOutcome",
    "LayoutCleanupOutcome",
    "add_bond_in_state",
    "bond_type_from_note",
    "bond_type_note",
    "bond_type_to_order",
    "cleanup_structure_2d",
    "clear_2d_override",
    "get_2d_override",
    "get_2d_override_source",
    "is_manual_2d_override_source",
    "matching_bond_pairs",
    "remove_bond_in_state",
    "reset_2d_layout",
    "reset_bond_override_in_state",
    "restore_editor_state",
    "set_atom_2d_position",
    "transform_2d_layout",
    "set_bond_type_in_state",
    "set_2d_override",
    "snapshot_editor_state",
]
