"""Structure-aware molecular graph foundation for PCS Analyzer.

This module deliberately separates chemical topology from UI rendering.

Key design rules
----------------
* Existing PCS Analyzer atom/Ref IDs are preserved.
* Covalent ligand connectivity and metal coordination are different edge kinds.
* No bond orders are invented in D4a.
* Automatic distance perception is heuristic and records provenance/confidence.
* User overrides are first-class and can supersede automatic perception.
* Ligand/fragments are obtained after removing coordination/haptic edges and metals.

No new third-party package is required by this module. It uses NumPy, which is
already a PCS Analyzer dependency, plus the project's own chemical constants.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence
import math

import numpy as np

from logic.chem_constants import METAL_ELEMENTS, covalent_radii


EDGE_KINDS = {"covalent", "coordination", "haptic"}
EDGE_SOURCES = {"distance", "explicit", "user", "template"}
CONFIDENCE_LEVELS = {"high", "medium", "possible", "explicit", "manual"}

DEFAULT_COORD_DONORS = frozenset({
    "N", "O", "F", "P", "S", "Cl", "Br", "I", "As", "Se",
})


def _edge_key(atom1_id: int, atom2_id: int) -> tuple[int, int]:
    a = int(atom1_id)
    b = int(atom2_id)
    if a == b:
        raise ValueError("an edge cannot connect an atom to itself")
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class StructureAtom:
    atom_id: int
    element: str
    x: float
    y: float
    z: float

    @property
    def coord(self) -> np.ndarray:
        return np.asarray((self.x, self.y, self.z), dtype=float)


@dataclass(frozen=True)
class StructureEdge:
    atom1_id: int
    atom2_id: int
    kind: str
    distance_angstrom: float
    source: str = "distance"
    confidence: str = "medium"
    cutoff_angstrom: float | None = None
    order: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in EDGE_KINDS:
            raise ValueError(f"unsupported edge kind: {self.kind}")
        if self.source not in EDGE_SOURCES:
            raise ValueError(f"unsupported edge source: {self.source}")
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(f"unsupported edge confidence: {self.confidence}")
        if int(self.atom1_id) == int(self.atom2_id):
            raise ValueError("edge endpoints must be distinct")

    @property
    def key(self) -> tuple[int, int]:
        return _edge_key(self.atom1_id, self.atom2_id)


@dataclass(frozen=True)
class BondDiagnostic:
    atom1_id: int
    atom2_id: int
    distance_angstrom: float
    cutoff_angstrom: float | None
    reason: str


@dataclass(frozen=True)
class StructureFragment:
    fragment_id: int
    atom_ids: tuple[int, ...]
    donor_atom_ids: tuple[int, ...]
    attached_metal_ids: tuple[int, ...]
    formula: str

    @property
    def coordinating(self) -> bool:
        return bool(self.attached_metal_ids)


@dataclass(frozen=True)
class StructureGraphConfig:
    # Conservative ligand-internal connectivity. This is intentionally close
    # to the project's existing symmetry-graph default.
    covalent_scale: float = 1.10

    # Coordination contacts use a separate, more permissive envelope matching
    # the current conformer-search philosophy.
    coordination_scale: float = 1.30

    fallback_radius: float = 0.77
    minimum_distance_angstrom: float = 0.40
    max_metal_bond_distance: float | None = None

    coord_donors: frozenset[str] = DEFAULT_COORD_DONORS
    allow_metal_carbon: bool = False
    allow_metal_metal: bool = False

    def __post_init__(self) -> None:
        if self.covalent_scale <= 0:
            raise ValueError("covalent_scale must be positive")
        if self.coordination_scale <= 0:
            raise ValueError("coordination_scale must be positive")
        if self.fallback_radius <= 0:
            raise ValueError("fallback_radius must be positive")
        if self.minimum_distance_angstrom < 0:
            raise ValueError("minimum_distance_angstrom cannot be negative")


@dataclass
class StructureGraph:
    atoms: dict[int, StructureAtom]
    edges: dict[tuple[int, int], StructureEdge] = field(default_factory=dict)
    metal_atom_ids: set[int] = field(default_factory=set)
    coordination_center_ids: set[int] = field(default_factory=set)
    diagnostics: list[BondDiagnostic] = field(default_factory=list)
    config: StructureGraphConfig = field(default_factory=StructureGraphConfig)

    def atom(self, atom_id: int) -> StructureAtom:
        try:
            return self.atoms[int(atom_id)]
        except KeyError as exc:
            raise KeyError(f"unknown atom ID {atom_id}") from exc

    def edge(self, atom1_id: int, atom2_id: int) -> StructureEdge | None:
        return self.edges.get(_edge_key(atom1_id, atom2_id))

    def add_edge(self, edge: StructureEdge, *, replace: bool = True) -> None:
        self.atom(edge.atom1_id)
        self.atom(edge.atom2_id)
        key = edge.key
        if not replace and key in self.edges:
            return
        self.edges[key] = edge

    def remove_edge(self, atom1_id: int, atom2_id: int) -> StructureEdge | None:
        return self.edges.pop(_edge_key(atom1_id, atom2_id), None)

    def add_manual_edge(
        self,
        atom1_id: int,
        atom2_id: int,
        *,
        kind: str = "covalent",
        order: float | None = None,
        note: str = "",
    ) -> StructureEdge:
        atom1 = self.atom(atom1_id)
        atom2 = self.atom(atom2_id)
        distance = float(np.linalg.norm(atom1.coord - atom2.coord))
        edge = StructureEdge(
            atom1_id=int(atom1_id),
            atom2_id=int(atom2_id),
            kind=kind,
            distance_angstrom=distance,
            source="user",
            confidence="manual",
            cutoff_angstrom=None,
            order=order,
            note=note,
        )
        self.add_edge(edge, replace=True)
        return edge

    def remove_manual_edge(self, atom1_id: int, atom2_id: int) -> StructureEdge | None:
        return self.remove_edge(atom1_id, atom2_id)

    def neighbors(
        self,
        atom_id: int,
        *,
        kinds: set[str] | frozenset[str] | None = None,
    ) -> tuple[int, ...]:
        atom_id = int(atom_id)
        self.atom(atom_id)
        output = []
        for edge in self.edges.values():
            if kinds is not None and edge.kind not in kinds:
                continue
            if edge.atom1_id == atom_id:
                output.append(edge.atom2_id)
            elif edge.atom2_id == atom_id:
                output.append(edge.atom1_id)
        return tuple(sorted(output))

    def connected_components(
        self,
        *,
        include_edge_kinds: frozenset[str] = frozenset({"covalent"}),
        exclude_metal_atoms: bool = True,
    ) -> list[tuple[int, ...]]:
        """Return graph components without requiring NetworkX."""
        available = set(self.atoms)
        if exclude_metal_atoms:
            available -= set(self.metal_atom_ids)

        adjacency = {atom_id: set() for atom_id in available}
        for edge in self.edges.values():
            if edge.kind not in include_edge_kinds:
                continue
            if edge.atom1_id not in available or edge.atom2_id not in available:
                continue
            adjacency[edge.atom1_id].add(edge.atom2_id)
            adjacency[edge.atom2_id].add(edge.atom1_id)

        components: list[tuple[int, ...]] = []
        unseen = set(available)
        while unseen:
            start = min(unseen)
            stack = [start]
            component = set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                unseen.discard(current)
                stack.extend(adjacency[current] - component)
            components.append(tuple(sorted(component)))

        components.sort(key=lambda group: (min(group), len(group)))
        return components

    def fragments(self) -> list[StructureFragment]:
        """Return non-metal covalent fragments plus their metal attachments."""
        result: list[StructureFragment] = []
        for fragment_index, atom_ids in enumerate(self.connected_components(), start=1):
            atom_set = set(atom_ids)
            donors: set[int] = set()
            metals: set[int] = set()

            for edge in self.edges.values():
                if edge.kind not in {"coordination", "haptic"}:
                    continue

                if edge.atom1_id in atom_set and edge.atom2_id in self.metal_atom_ids:
                    donors.add(edge.atom1_id)
                    metals.add(edge.atom2_id)
                elif edge.atom2_id in atom_set and edge.atom1_id in self.metal_atom_ids:
                    donors.add(edge.atom2_id)
                    metals.add(edge.atom1_id)

            result.append(
                StructureFragment(
                    fragment_id=fragment_index,
                    atom_ids=tuple(atom_ids),
                    donor_atom_ids=tuple(sorted(donors)),
                    attached_metal_ids=tuple(sorted(metals)),
                    formula=_formula(self.atoms[atom_id].element for atom_id in atom_ids),
                )
            )
        return result

    def summary(self) -> dict:
        fragments = self.fragments()
        return {
            "atom_count": len(self.atoms),
            "edge_count": len(self.edges),
            "covalent_edge_count": sum(
                edge.kind == "covalent" for edge in self.edges.values()
            ),
            "coordination_edge_count": sum(
                edge.kind == "coordination" for edge in self.edges.values()
            ),
            "metal_atom_ids": sorted(self.metal_atom_ids),
            "coordination_center_ids": sorted(self.coordination_center_ids),
            "fragment_count": len(fragments),
            "coordinating_fragment_count": sum(f.coordinating for f in fragments),
            "diagnostic_count": len(self.diagnostics),
        }


def _formula(elements: Iterable[str]) -> str:
    counts = Counter(str(element) for element in elements)
    ordered = []
    for symbol in ("C", "H"):
        if symbol in counts:
            ordered.append(symbol)
    ordered.extend(sorted(symbol for symbol in counts if symbol not in {"C", "H"}))

    parts = []
    for symbol in ordered:
        count = counts[symbol]
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def _confidence_for_covalent(distance: float, radius_sum: float) -> str:
    ratio = distance / radius_sum if radius_sum > 0 else math.inf
    return "high" if ratio <= 1.03 else "medium"


def _confidence_for_coordination(distance: float, radius_sum: float) -> str:
    ratio = distance / radius_sum if radius_sum > 0 else math.inf
    if ratio <= 1.10:
        return "high"
    if ratio <= 1.20:
        return "medium"
    return "possible"


def _radius(element: str, config: StructureGraphConfig) -> float:
    return float(covalent_radii.get(element, config.fallback_radius))


def build_structure_graph(
    atom_data: Sequence[tuple[str, float, float, float]],
    *,
    atom_ids: Sequence[int] | None = None,
    coordination_center_ids: Iterable[int] | None = None,
    config: StructureGraphConfig | None = None,
    explicit_edges: Iterable[StructureEdge] = (),
    manual_add_edges: Iterable[tuple] = (),
    manual_remove_edges: Iterable[tuple[int, int]] = (),
) -> StructureGraph:
    """Create a structure graph from PCS Analyzer-style Cartesian atom tuples."""
    cfg = config or StructureGraphConfig()

    if atom_ids is None:
        ids = list(range(1, len(atom_data) + 1))
    else:
        ids = [int(value) for value in atom_ids]

    if len(ids) != len(atom_data):
        raise ValueError("atom_ids length must match atom_data length")
    if len(set(ids)) != len(ids):
        raise ValueError("atom_ids must be unique")

    atoms: dict[int, StructureAtom] = {}
    for atom_id, record in zip(ids, atom_data):
        if len(record) != 4:
            raise ValueError("each atom record must be (element, x, y, z)")
        element, x, y, z = record
        atoms[atom_id] = StructureAtom(
            atom_id=atom_id,
            element=str(element),
            x=float(x),
            y=float(y),
            z=float(z),
        )

    metal_ids = {
        atom_id
        for atom_id, atom in atoms.items()
        if atom.element in METAL_ELEMENTS
    }

    if coordination_center_ids is None:
        centers = set(metal_ids)
    else:
        requested_centers = {int(value) for value in coordination_center_ids}
        unknown = requested_centers - set(atoms)
        if unknown:
            raise ValueError(f"unknown coordination center IDs: {sorted(unknown)}")
        centers = requested_centers & metal_ids

    graph = StructureGraph(
        atoms=atoms,
        metal_atom_ids=set(metal_ids),
        coordination_center_ids=set(centers),
        config=cfg,
    )

    for edge in explicit_edges:
        explicit = StructureEdge(
            atom1_id=edge.atom1_id,
            atom2_id=edge.atom2_id,
            kind=edge.kind,
            distance_angstrom=edge.distance_angstrom,
            source="explicit",
            confidence="explicit",
            cutoff_angstrom=edge.cutoff_angstrom,
            order=edge.order,
            note=edge.note,
        )
        graph.add_edge(explicit, replace=True)

    atom_id_list = list(atoms)
    for i, atom1_id in enumerate(atom_id_list):
        atom1 = atoms[atom1_id]
        r1 = _radius(atom1.element, cfg)

        for atom2_id in atom_id_list[i + 1 :]:
            key = _edge_key(atom1_id, atom2_id)
            if key in graph.edges:
                continue

            atom2 = atoms[atom2_id]
            r2 = _radius(atom2.element, cfg)
            radius_sum = r1 + r2
            distance = float(np.linalg.norm(atom1.coord - atom2.coord))

            if distance < cfg.minimum_distance_angstrom:
                graph.diagnostics.append(
                    BondDiagnostic(
                        atom1_id,
                        atom2_id,
                        distance,
                        None,
                        "atoms are unrealistically close; no automatic edge created",
                    )
                )
                continue

            atom1_is_metal = atom1_id in metal_ids
            atom2_is_metal = atom2_id in metal_ids

            if not atom1_is_metal and not atom2_is_metal:
                cutoff = radius_sum * cfg.covalent_scale
                if distance <= cutoff:
                    graph.add_edge(
                        StructureEdge(
                            atom1_id,
                            atom2_id,
                            "covalent",
                            distance,
                            source="distance",
                            confidence=_confidence_for_covalent(distance, radius_sum),
                            cutoff_angstrom=cutoff,
                        ),
                        replace=False,
                    )
                continue

            if atom1_is_metal and atom2_is_metal:
                cutoff = radius_sum * cfg.coordination_scale
                if distance <= cutoff:
                    if cfg.allow_metal_metal and (
                        atom1_id in centers or atom2_id in centers
                    ):
                        graph.add_edge(
                            StructureEdge(
                                atom1_id,
                                atom2_id,
                                "coordination",
                                distance,
                                source="distance",
                                confidence=_confidence_for_coordination(
                                    distance, radius_sum
                                ),
                                cutoff_angstrom=cutoff,
                                note="metal-metal contact enabled by configuration",
                            )
                        )
                    else:
                        graph.diagnostics.append(
                            BondDiagnostic(
                                atom1_id,
                                atom2_id,
                                distance,
                                cutoff,
                                "metal-metal contact skipped",
                            )
                        )
                continue

            metal_id = atom1_id if atom1_is_metal else atom2_id
            partner_id = atom2_id if atom1_is_metal else atom1_id
            partner = atoms[partner_id]

            if metal_id not in centers:
                continue

            cutoff = radius_sum * cfg.coordination_scale
            if cfg.max_metal_bond_distance is not None:
                cutoff = min(cutoff, float(cfg.max_metal_bond_distance))

            if distance > cutoff:
                continue

            if partner.element == "C" and not cfg.allow_metal_carbon:
                graph.diagnostics.append(
                    BondDiagnostic(
                        atom1_id,
                        atom2_id,
                        distance,
                        cutoff,
                        "metal-carbon contact skipped; enable explicitly for organometallic use",
                    )
                )
                continue

            if (
                partner.element not in cfg.coord_donors
                and not (partner.element == "C" and cfg.allow_metal_carbon)
            ):
                graph.diagnostics.append(
                    BondDiagnostic(
                        atom1_id,
                        atom2_id,
                        distance,
                        cutoff,
                        f"metal-{partner.element} contact skipped by donor filter",
                    )
                )
                continue

            graph.add_edge(
                StructureEdge(
                    atom1_id,
                    atom2_id,
                    "coordination",
                    distance,
                    source="distance",
                    confidence=_confidence_for_coordination(distance, radius_sum),
                    cutoff_angstrom=cutoff,
                ),
                replace=False,
            )

    for atom1_id, atom2_id in manual_remove_edges:
        graph.remove_edge(atom1_id, atom2_id)

    for record in manual_add_edges:
        values = tuple(record)
        if len(values) < 3:
            raise ValueError("manual_add_edges records must contain atom1, atom2, kind")
        atom1_id, atom2_id, kind = values[:3]
        order = values[3] if len(values) >= 4 else None
        note = str(values[4]) if len(values) >= 5 and values[4] is not None else ""
        graph.add_manual_edge(
            atom1_id,
            atom2_id,
            kind=kind,
            order=order,
            note=note,
        )

    return graph


__all__ = [
    "BondDiagnostic",
    "DEFAULT_COORD_DONORS",
    "StructureAtom",
    "StructureEdge",
    "StructureFragment",
    "StructureGraph",
    "StructureGraphConfig",
    "build_structure_graph",
]
