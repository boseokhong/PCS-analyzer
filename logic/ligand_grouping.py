"""Ligand detection, coordination analysis, and equivalence grouping.

D4b consumes :mod:`logic.structure_graph` and deliberately adds no new runtime
dependency.  Equivalent ligands are established by exact element-labelled graph
isomorphism after inexpensive invariant checks; a deterministic fingerprint is
used only as a fast prefilter, never as the final equivalence decision.

Important chemistry semantics
-----------------------------
* ligand identity comes from the covalent fragment topology;
* coordination/haptic roles can optionally be included in equivalence;
* sigma-donor denticity and hapticity are reported separately;
* symmetric ligands may have multiple valid atom mappings.  One deterministic
  mapping is returned for display/alignment, but ``ambiguous=True`` prevents it
  from being mistaken for a unique NMR assignment correspondence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping

from logic.structure_graph import StructureEdge, StructureFragment, StructureGraph


@dataclass(frozen=True)
class MetalBindingSite:
    """How one ligand fragment binds one metal atom."""

    metal_id: int
    donor_atom_ids: tuple[int, ...]
    haptic_atom_ids: tuple[int, ...]

    @property
    def denticity(self) -> int:
        """Number of ordinary coordination donor atoms (kappa count)."""
        return len(self.donor_atom_ids)

    @property
    def hapticity(self) -> int:
        """Number of atoms connected through haptic edges (eta count)."""
        return len(self.haptic_atom_ids)


@dataclass(frozen=True)
class LigandInstance:
    """One coordinating covalent fragment in the loaded structure."""

    fragment_id: int
    atom_ids: tuple[int, ...]
    formula: str
    binding_sites: tuple[MetalBindingSite, ...]
    topology_fingerprint: str
    coordination_fingerprint: str

    @property
    def attached_metal_ids(self) -> tuple[int, ...]:
        return tuple(site.metal_id for site in self.binding_sites)

    @property
    def donor_atom_ids(self) -> tuple[int, ...]:
        atoms = {
            atom_id
            for site in self.binding_sites
            for atom_id in site.donor_atom_ids
        }
        return tuple(sorted(atoms))

    @property
    def haptic_atom_ids(self) -> tuple[int, ...]:
        atoms = {
            atom_id
            for site in self.binding_sites
            for atom_id in site.haptic_atom_ids
        }
        return tuple(sorted(atoms))

    @property
    def denticity(self) -> int:
        return len(self.donor_atom_ids)

    @property
    def hapticity(self) -> int:
        return len(self.haptic_atom_ids)


@dataclass(frozen=True)
class AtomCorrespondence:
    """Representative-to-member atom mapping for one ligand group member."""

    fragment_id: int
    pairs: tuple[tuple[int, int], ...]
    ambiguous: bool

    def as_dict(self) -> dict[int, int]:
        return dict(self.pairs)


@dataclass(frozen=True)
class LigandGroup:
    """A set of coordination-equivalent ligand fragments."""

    group_id: int
    representative_fragment_id: int
    member_fragment_ids: tuple[int, ...]
    formula: str
    topology_fingerprint: str
    coordination_fingerprint: str
    correspondences: tuple[AtomCorrespondence, ...]

    @property
    def multiplicity(self) -> int:
        return len(self.member_fragment_ids)

    @property
    def display_label(self) -> str:
        suffix = f" x{self.multiplicity}" if self.multiplicity > 1 else ""
        return f"{self.formula}{suffix}"


@dataclass(frozen=True)
class LigandAnalysis:
    """Complete D4b ligand analysis for one :class:`StructureGraph`."""

    ligands: tuple[LigandInstance, ...]
    groups: tuple[LigandGroup, ...]
    spectator_fragments: tuple[StructureFragment, ...]

    def ligand_by_fragment_id(self, fragment_id: int) -> LigandInstance:
        for ligand in self.ligands:
            if ligand.fragment_id == int(fragment_id):
                return ligand
        raise KeyError(f"unknown ligand fragment ID {fragment_id}")

    def group_for_fragment_id(self, fragment_id: int) -> LigandGroup:
        fragment_id = int(fragment_id)
        for group in self.groups:
            if fragment_id in group.member_fragment_ids:
                return group
        raise KeyError(f"fragment {fragment_id} is not in a ligand group")

    def summary(self) -> dict:
        return {
            "ligand_count": len(self.ligands),
            "ligand_group_count": len(self.groups),
            "spectator_fragment_count": len(self.spectator_fragments),
            "groups": [
                {
                    "group_id": group.group_id,
                    "representative_fragment_id": group.representative_fragment_id,
                    "member_fragment_ids": list(group.member_fragment_ids),
                    "formula": group.formula,
                    "multiplicity": group.multiplicity,
                    "has_ambiguous_correspondence": any(
                        item.ambiguous for item in group.correspondences
                    ),
                }
                for group in self.groups
            ],
        }


@dataclass(frozen=True)
class FragmentIsomorphism:
    """One exact graph isomorphism plus whether alternatives also exist."""

    mapping: tuple[tuple[int, int], ...]
    ambiguous: bool

    def as_dict(self) -> dict[int, int]:
        return dict(self.mapping)


def analyze_ligands(
    graph: StructureGraph,
    *,
    include_coordination_role: bool = True,
) -> LigandAnalysis:
    """Detect coordinating fragments and group equivalent ligand instances.

    Parameters
    ----------
    graph
        D4a structure graph.
    include_coordination_role
        If true (default), atoms must match both covalent topology and their
        coordination/haptic role.  This avoids silently grouping a chelating
        and a monodentate occurrence of the same covalent molecule.
    """
    fragments = tuple(graph.fragments())
    ligands = tuple(
        _ligand_instance(graph, fragment)
        for fragment in fragments
        if fragment.coordinating
    )
    spectators = tuple(fragment for fragment in fragments if not fragment.coordinating)

    mutable_groups: list[dict] = []

    for ligand in ligands:
        placed = False
        for group in mutable_groups:
            representative: LigandInstance = group["representative"]
            if ligand.coordination_fingerprint != representative.coordination_fingerprint:
                if include_coordination_role:
                    continue
            if ligand.topology_fingerprint != representative.topology_fingerprint:
                continue

            iso = find_fragment_isomorphism(
                graph,
                representative.atom_ids,
                ligand.atom_ids,
                include_coordination_role=include_coordination_role,
            )
            if iso is None:
                continue

            group["members"].append(ligand)
            group["correspondences"].append(
                AtomCorrespondence(
                    fragment_id=ligand.fragment_id,
                    pairs=iso.mapping,
                    ambiguous=iso.ambiguous,
                )
            )
            placed = True
            break

        if placed:
            continue

        identity = tuple((atom_id, atom_id) for atom_id in ligand.atom_ids)
        mutable_groups.append(
            {
                "representative": ligand,
                "members": [ligand],
                "correspondences": [
                    AtomCorrespondence(
                        fragment_id=ligand.fragment_id,
                        pairs=identity,
                        ambiguous=False,
                    )
                ],
            }
        )

    groups: list[LigandGroup] = []
    for group_id, raw in enumerate(mutable_groups, start=1):
        representative: LigandInstance = raw["representative"]
        members: list[LigandInstance] = raw["members"]
        groups.append(
            LigandGroup(
                group_id=group_id,
                representative_fragment_id=representative.fragment_id,
                member_fragment_ids=tuple(member.fragment_id for member in members),
                formula=representative.formula,
                topology_fingerprint=representative.topology_fingerprint,
                coordination_fingerprint=representative.coordination_fingerprint,
                correspondences=tuple(raw["correspondences"]),
            )
        )

    return LigandAnalysis(
        ligands=ligands,
        groups=tuple(groups),
        spectator_fragments=spectators,
    )


def find_fragment_isomorphism(
    graph: StructureGraph,
    atom_ids_a: Iterable[int],
    atom_ids_b: Iterable[int],
    *,
    include_coordination_role: bool = True,
) -> FragmentIsomorphism | None:
    """Return an exact covalent graph isomorphism, if one exists.

    Search stops after the second valid solution: this is enough to distinguish
    a unique atom correspondence from a symmetry-ambiguous one without
    enumerating a potentially large automorphism group.
    """
    a_ids = tuple(sorted(int(value) for value in atom_ids_a))
    b_ids = tuple(sorted(int(value) for value in atom_ids_b))
    if len(a_ids) != len(b_ids):
        return None
    if not a_ids:
        return FragmentIsomorphism((), False)

    a_set = set(a_ids)
    b_set = set(b_ids)
    adjacency_a, edge_labels_a = _induced_covalent_graph(graph, a_set)
    adjacency_b, edge_labels_b = _induced_covalent_graph(graph, b_set)

    if sum(len(v) for v in adjacency_a.values()) != sum(
        len(v) for v in adjacency_b.values()
    ):
        return None

    node_labels_a = {
        atom_id: _node_label(
            graph, atom_id, a_set, include_coordination_role=include_coordination_role
        )
        for atom_id in a_ids
    }
    node_labels_b = {
        atom_id: _node_label(
            graph, atom_id, b_set, include_coordination_role=include_coordination_role
        )
        for atom_id in b_ids
    }

    if Counter(node_labels_a.values()) != Counter(node_labels_b.values()):
        return None

    # Refined local invariant: label + degree + neighbour-label multiset.
    invariants_a = {
        atom_id: _local_invariant(atom_id, adjacency_a, node_labels_a)
        for atom_id in a_ids
    }
    invariants_b = {
        atom_id: _local_invariant(atom_id, adjacency_b, node_labels_b)
        for atom_id in b_ids
    }
    if Counter(invariants_a.values()) != Counter(invariants_b.values()):
        return None

    candidates = {
        atom_id: tuple(
            candidate
            for candidate in b_ids
            if invariants_b[candidate] == invariants_a[atom_id]
        )
        for atom_id in a_ids
    }
    if any(not values for values in candidates.values()):
        return None

    order = tuple(
        sorted(
            a_ids,
            key=lambda atom_id: (
                len(candidates[atom_id]),
                -len(adjacency_a[atom_id]),
                atom_id,
            ),
        )
    )

    solutions: list[dict[int, int]] = []
    mapping: dict[int, int] = {}
    used_b: set[int] = set()

    def compatible(atom_a: int, atom_b: int) -> bool:
        for mapped_a, mapped_b in mapping.items():
            linked_a = mapped_a in adjacency_a[atom_a]
            linked_b = mapped_b in adjacency_b[atom_b]
            if linked_a != linked_b:
                return False
            if linked_a:
                label_a = edge_labels_a[_pair(atom_a, mapped_a)]
                label_b = edge_labels_b[_pair(atom_b, mapped_b)]
                if label_a != label_b:
                    return False
        return True

    def search(index: int) -> None:
        if len(solutions) >= 2:
            return
        if index >= len(order):
            solutions.append(dict(mapping))
            return

        atom_a = order[index]
        for atom_b in candidates[atom_a]:
            if atom_b in used_b:
                continue
            if not compatible(atom_a, atom_b):
                continue
            mapping[atom_a] = atom_b
            used_b.add(atom_b)
            search(index + 1)
            used_b.remove(atom_b)
            del mapping[atom_a]
            if len(solutions) >= 2:
                return

    search(0)
    if not solutions:
        return None

    chosen = solutions[0]
    return FragmentIsomorphism(
        mapping=tuple((atom_id, chosen[atom_id]) for atom_id in sorted(chosen)),
        ambiguous=len(solutions) > 1,
    )


def fragment_fingerprint(
    graph: StructureGraph,
    atom_ids: Iterable[int],
    *,
    include_coordination_role: bool,
) -> str:
    """Deterministic WL-style fingerprint used only as an isomorphism prefilter."""
    ids = tuple(sorted(int(value) for value in atom_ids))
    atom_set = set(ids)
    adjacency, edge_labels = _induced_covalent_graph(graph, atom_set)

    colors = {
        atom_id: _stable_digest(
            repr(
                _node_label(
                    graph,
                    atom_id,
                    atom_set,
                    include_coordination_role=include_coordination_role,
                )
            )
        )
        for atom_id in ids
    }

    # n rounds are sufficient for stabilization on a graph of n vertices.
    for _ in range(max(len(ids), 1)):
        updated = {}
        for atom_id in ids:
            neighbours = sorted(
                (
                    edge_labels[_pair(atom_id, neighbour)],
                    colors[neighbour],
                )
                for neighbour in adjacency[atom_id]
            )
            updated[atom_id] = _stable_digest(
                colors[atom_id] + "|" + repr(neighbours)
            )
        if updated == colors:
            break
        colors = updated

    payload = (
        len(ids),
        sum(len(neighbours) for neighbours in adjacency.values()) // 2,
        sorted(colors.values()),
    )
    return _stable_digest(repr(payload))


def _ligand_instance(
    graph: StructureGraph,
    fragment: StructureFragment,
) -> LigandInstance:
    atom_set = set(fragment.atom_ids)
    by_metal: dict[int, dict[str, set[int]]] = defaultdict(
        lambda: {"coordination": set(), "haptic": set()}
    )

    for edge in graph.edges.values():
        if edge.kind not in {"coordination", "haptic"}:
            continue

        ligand_atom: int | None = None
        metal_atom: int | None = None
        if edge.atom1_id in atom_set and edge.atom2_id in graph.metal_atom_ids:
            ligand_atom = edge.atom1_id
            metal_atom = edge.atom2_id
        elif edge.atom2_id in atom_set and edge.atom1_id in graph.metal_atom_ids:
            ligand_atom = edge.atom2_id
            metal_atom = edge.atom1_id

        if ligand_atom is None or metal_atom is None:
            continue

        by_metal[metal_atom][edge.kind].add(ligand_atom)

    binding_sites = tuple(
        MetalBindingSite(
            metal_id=metal_id,
            donor_atom_ids=tuple(sorted(roles["coordination"])),
            haptic_atom_ids=tuple(sorted(roles["haptic"])),
        )
        for metal_id, roles in sorted(by_metal.items())
    )

    return LigandInstance(
        fragment_id=fragment.fragment_id,
        atom_ids=fragment.atom_ids,
        formula=fragment.formula,
        binding_sites=binding_sites,
        topology_fingerprint=fragment_fingerprint(
            graph,
            fragment.atom_ids,
            include_coordination_role=False,
        ),
        coordination_fingerprint=fragment_fingerprint(
            graph,
            fragment.atom_ids,
            include_coordination_role=True,
        ),
    )


def _node_label(
    graph: StructureGraph,
    atom_id: int,
    fragment_atom_ids: set[int],
    *,
    include_coordination_role: bool,
) -> tuple:
    atom = graph.atom(atom_id)
    if not include_coordination_role:
        return (atom.element,)

    coordination_count = 0
    haptic_count = 0
    for edge in graph.edges.values():
        if atom_id not in {edge.atom1_id, edge.atom2_id}:
            continue
        other = edge.atom2_id if edge.atom1_id == atom_id else edge.atom1_id
        if other in fragment_atom_ids:
            continue
        if other not in graph.metal_atom_ids:
            continue
        if edge.kind == "coordination":
            coordination_count += 1
        elif edge.kind == "haptic":
            haptic_count += 1

    return (atom.element, coordination_count, haptic_count)


def _local_invariant(
    atom_id: int,
    adjacency: Mapping[int, set[int]],
    labels: Mapping[int, tuple],
) -> tuple:
    return (
        labels[atom_id],
        len(adjacency[atom_id]),
        tuple(sorted(repr(labels[neighbor]) for neighbor in adjacency[atom_id])),
    )


def _induced_covalent_graph(
    graph: StructureGraph,
    atom_ids: set[int],
) -> tuple[dict[int, set[int]], dict[tuple[int, int], tuple]]:
    adjacency = {atom_id: set() for atom_id in atom_ids}
    edge_labels: dict[tuple[int, int], tuple] = {}

    for edge in graph.edges.values():
        if edge.kind != "covalent":
            continue
        if edge.atom1_id not in atom_ids or edge.atom2_id not in atom_ids:
            continue
        adjacency[edge.atom1_id].add(edge.atom2_id)
        adjacency[edge.atom2_id].add(edge.atom1_id)

        # D4a intentionally does not infer bond order. If explicit/template
        # order exists, preserve it; otherwise all unknown covalent edges are
        # topologically equivalent.
        if edge.order is None:
            edge_label = ("unknown",)
        else:
            try:
                edge_label = ("order", float(edge.order))
            except (TypeError, ValueError):
                edge_label = ("symbolic", str(edge.order))
        edge_labels[_pair(edge.atom1_id, edge.atom2_id)] = edge_label

    return adjacency, edge_labels


def _pair(atom1_id: int, atom2_id: int) -> tuple[int, int]:
    return (
        (atom1_id, atom2_id)
        if atom1_id < atom2_id
        else (atom2_id, atom1_id)
    )


def _stable_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


__all__ = [
    "AtomCorrespondence",
    "FragmentIsomorphism",
    "LigandAnalysis",
    "LigandGroup",
    "LigandInstance",
    "MetalBindingSite",
    "analyze_ligands",
    "find_fragment_isomorphism",
    "fragment_fingerprint",
]
