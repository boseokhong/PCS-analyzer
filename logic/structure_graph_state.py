"""PCS Analyzer state adapter for the D4 StructureGraph foundation."""

from __future__ import annotations

from logic.structure_graph import StructureGraphConfig, build_structure_graph


def _config_from_state(state: dict) -> StructureGraphConfig:
    raw = dict(state.get("structure_graph_config", {}) or {})
    if "coord_donors" in raw:
        raw["coord_donors"] = frozenset(raw["coord_donors"])
    return StructureGraphConfig(**raw)


def _manual_add_from_state(state: dict) -> list[tuple]:
    """Return persistent manual-edge records including depiction bond metadata.

    Legacy D4a records contained only ``(atom1, atom2, kind)``.  D5b.4 keeps
    those records compatible while optionally carrying ``order`` and ``note``.
    """
    output = []
    for record in state.get("structure_graph_manual_add_edges", []) or []:
        if isinstance(record, dict):
            output.append(
                (
                    int(record["atom1_id"]),
                    int(record["atom2_id"]),
                    str(record.get("kind", "covalent")),
                    record.get("order"),
                    str(record.get("note", "")),
                )
            )
        else:
            values = tuple(record)
            if len(values) < 3:
                continue
            atom1_id, atom2_id, kind = values[:3]
            order = values[3] if len(values) >= 4 else None
            note = values[4] if len(values) >= 5 else ""
            output.append(
                (
                    int(atom1_id),
                    int(atom2_id),
                    str(kind),
                    order,
                    str(note or ""),
                )
            )
    return output


def _manual_remove_from_state(state: dict) -> list[tuple[int, int]]:
    return [
        (int(record[0]), int(record[1]))
        for record in (state.get("structure_graph_manual_remove_edges", []) or [])
    ]


def build_structure_graph_from_state(state: dict):
    """Build StructureGraph while preserving PCS Analyzer atom/Ref IDs."""
    atom_data = state.get("atom_data_raw") or state.get("atom_data") or []
    if not atom_data:
        raise ValueError("PCS Analyzer state contains no loaded structure")

    atom_ids = state.get("atom_ids_raw")
    if atom_ids is None:
        atom_ids = list(range(1, len(atom_data) + 1))

    metal_ref_id = state.get("metal_ref_id")
    coordination_centers = None
    if metal_ref_id is not None:
        coordination_centers = [int(metal_ref_id)]

    return build_structure_graph(
        atom_data,
        atom_ids=atom_ids,
        coordination_center_ids=coordination_centers,
        config=_config_from_state(state),
        manual_add_edges=_manual_add_from_state(state),
        manual_remove_edges=_manual_remove_from_state(state),
    )


def refresh_structure_graph_in_state(state: dict):
    graph = build_structure_graph_from_state(state)
    state["structure_graph"] = graph
    state["structure_graph_summary"] = graph.summary()
    # Ligand grouping is derived from this graph.  Invalidate the transient
    # cache whenever topology is rebuilt so visibility never uses stale
    # fragment membership after bond/configuration edits.
    state.pop("ligand_analysis", None)
    state.pop("ligand_analysis_summary", None)
    state.pop("ligand_visibility_error", None)
    return graph


def set_structure_graph_config(state: dict, **changes):
    current = dict(state.get("structure_graph_config", {}) or {})
    current.update(changes)

    # Keep state serialization JSON-friendly.
    if "coord_donors" in current:
        current["coord_donors"] = sorted(set(current["coord_donors"]))

    state["structure_graph_config"] = current
    return refresh_structure_graph_in_state(state)


def add_structure_bond_override(
    state: dict,
    atom1_id: int,
    atom2_id: int,
    *,
    kind: str = "covalent",
    order=None,
    note: str = "",
):
    a, b = sorted((int(atom1_id), int(atom2_id)))

    removes = {
        tuple(sorted((int(x), int(y))))
        for x, y in state.get("structure_graph_manual_remove_edges", []) or []
    }
    removes.discard((a, b))
    state["structure_graph_manual_remove_edges"] = [
        [x, y] for x, y in sorted(removes)
    ]

    additions = {}
    for record in state.get("structure_graph_manual_add_edges", []) or []:
        if isinstance(record, dict):
            key = tuple(sorted((int(record["atom1_id"]), int(record["atom2_id"]))))
            additions[key] = dict(record)
            continue
        values = tuple(record)
        if len(values) < 3:
            continue
        x, y, existing_kind = values[:3]
        key = tuple(sorted((int(x), int(y))))
        additions[key] = {
            "atom1_id": key[0],
            "atom2_id": key[1],
            "kind": str(existing_kind),
            "order": values[3] if len(values) >= 4 else None,
            "note": str(values[4] if len(values) >= 5 else ""),
        }
    additions[(a, b)] = {
        "atom1_id": a,
        "atom2_id": b,
        "kind": str(kind),
        "order": order,
        "note": str(note or ""),
    }
    state["structure_graph_manual_add_edges"] = [
        additions[key] for key in sorted(additions)
    ]

    return refresh_structure_graph_in_state(state)


def remove_structure_bond_override(
    state: dict,
    atom1_id: int,
    atom2_id: int,
):
    a, b = sorted((int(atom1_id), int(atom2_id)))

    additions = []
    for record in state.get("structure_graph_manual_add_edges", []) or []:
        if not isinstance(record, dict):
            continue
        key = tuple(sorted((int(record["atom1_id"]), int(record["atom2_id"]))))
        if key != (a, b):
            additions.append(dict(record))
    state["structure_graph_manual_add_edges"] = additions

    removes = {
        tuple(sorted((int(x), int(y))))
        for x, y in state.get("structure_graph_manual_remove_edges", []) or []
    }
    removes.add((a, b))
    state["structure_graph_manual_remove_edges"] = [
        [x, y] for x, y in sorted(removes)
    ]

    return refresh_structure_graph_in_state(state)


def clear_structure_bond_overrides(state: dict):
    state["structure_graph_manual_add_edges"] = []
    state["structure_graph_manual_remove_edges"] = []
    return refresh_structure_graph_in_state(state)


__all__ = [
    "add_structure_bond_override",
    "build_structure_graph_from_state",
    "clear_structure_bond_overrides",
    "refresh_structure_graph_in_state",
    "remove_structure_bond_override",
    "set_structure_graph_config",
]
