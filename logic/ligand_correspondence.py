"""Conservative atom correspondence across equivalent ligand instances.

D5a builds on :mod:`logic.ligand_grouping` without changing the D4b grouping
semantics.  The key difference from ``LigandGroup.correspondences`` is that
this module does *not* force a single atom-to-atom mapping when ligand
symmetry makes several mappings chemically indistinguishable.

Instead, atoms are classified by a rooted, element/bond/coordination-aware
WL-style signature.  For an exact ligand group, atoms in the same signature
bucket are candidate matching positions.  A bucket containing more than one
atom in a ligand is explicitly marked ambiguous.  This is intentionally
conservative: ambiguity is surfaced to the UI rather than silently resolved by
atom numbering or 3D orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping

from logic.ligand_grouping import LigandAnalysis
from logic.structure_graph import StructureGraph


@dataclass(frozen=True)
class FragmentAtomCandidates:
    """Candidate atoms for one ligand fragment in a correspondence bucket."""

    fragment_id: int
    atom_ids: tuple[int, ...]


@dataclass(frozen=True)
class AtomMatchGroup:
    """One chemically matching atom-position bucket across a ligand group."""

    ligand_group_id: int
    match_key: str
    candidates: tuple[FragmentAtomCandidates, ...]

    @property
    def ambiguous(self) -> bool:
        return any(len(item.atom_ids) != 1 for item in self.candidates)

    @property
    def complete(self) -> bool:
        return bool(self.candidates) and all(item.atom_ids for item in self.candidates)

    def candidates_for_fragment(self, fragment_id: int) -> tuple[int, ...]:
        fragment_id = int(fragment_id)
        for item in self.candidates:
            if item.fragment_id == fragment_id:
                return item.atom_ids
        return ()


@dataclass(frozen=True)
class AtomMatchResult:
    """Matching candidates for a source atom across equivalent ligands."""

    source_atom_id: int
    source_fragment_id: int
    ligand_group_id: int
    match_key: str
    candidates: tuple[FragmentAtomCandidates, ...]
    status: str

    @property
    def ambiguous(self) -> bool:
        return self.status == "ambiguous"

    @property
    def matching_atom_ids(self) -> tuple[int, ...]:
        return tuple(
            atom_id
            for item in self.candidates
            for atom_id in item.atom_ids
            if atom_id != self.source_atom_id
        )


@dataclass(frozen=True)
class LigandCorrespondenceAnalysis:
    """Structure-aware candidate correspondences for all ligand groups."""

    atom_groups: tuple[AtomMatchGroup, ...]
    atom_to_group_index: Mapping[int, int]
    atom_to_fragment_id: Mapping[int, int]
    fragment_to_ligand_group_id: Mapping[int, int]

    def group_for_atom(self, atom_id: int) -> AtomMatchGroup:
        atom_id = int(atom_id)
        try:
            index = self.atom_to_group_index[atom_id]
        except KeyError as exc:
            raise KeyError(f"atom {atom_id} is not in a coordinating ligand") from exc
        return self.atom_groups[index]

    def matching_atoms(self, atom_id: int) -> AtomMatchResult:
        atom_id = int(atom_id)
        group = self.group_for_atom(atom_id)
        source_fragment_id = self.atom_to_fragment_id[atom_id]

        if len(group.candidates) <= 1:
            status = "single_ligand"
        elif not group.complete:
            status = "incomplete"
        elif group.ambiguous:
            status = "ambiguous"
        else:
            status = "unique"

        return AtomMatchResult(
            source_atom_id=atom_id,
            source_fragment_id=source_fragment_id,
            ligand_group_id=group.ligand_group_id,
            match_key=group.match_key,
            candidates=group.candidates,
            status=status,
        )

    def rooted_signature(self, atom_id: int) -> str:
        """Return the stable correspondence key for one ligand atom."""
        return self.group_for_atom(atom_id).match_key


def analyze_ligand_correspondence(
    graph: StructureGraph,
    ligand_analysis: LigandAnalysis,
) -> LigandCorrespondenceAnalysis:
    """Build conservative atom correspondence buckets for ligand groups.

    The input ligand groups were already proven isomorphic by D4b.  D5a now
    classifies rooted positions inside each group.  Symmetry-related positions
    deliberately remain in the same bucket, e.g. the two N atoms of a symmetric
    N-C-N ligand or the three H atoms of a methyl group.
    """

    atom_groups: list[AtomMatchGroup] = []
    atom_to_group_index: dict[int, int] = {}
    atom_to_fragment_id: dict[int, int] = {}
    fragment_to_group_id: dict[int, int] = {}

    for ligand_group in ligand_analysis.groups:
        signatures_by_fragment: dict[int, dict[str, list[int]]] = {}

        for fragment_id in ligand_group.member_fragment_ids:
            ligand = ligand_analysis.ligand_by_fragment_id(fragment_id)
            fragment_to_group_id[fragment_id] = ligand_group.group_id
            per_signature: dict[str, list[int]] = {}

            for atom_id in ligand.atom_ids:
                signature = rooted_atom_signature(
                    graph,
                    ligand.atom_ids,
                    atom_id,
                    include_coordination_role=True,
                )
                per_signature.setdefault(signature, []).append(atom_id)
                atom_to_fragment_id[atom_id] = fragment_id

            signatures_by_fragment[fragment_id] = per_signature

        all_keys = sorted(
            {
                key
                for per_fragment in signatures_by_fragment.values()
                for key in per_fragment
            }
        )

        for key in all_keys:
            candidates = tuple(
                FragmentAtomCandidates(
                    fragment_id=fragment_id,
                    atom_ids=tuple(sorted(signatures_by_fragment[fragment_id].get(key, ()))),
                )
                for fragment_id in ligand_group.member_fragment_ids
            )
            group_index = len(atom_groups)
            atom_groups.append(
                AtomMatchGroup(
                    ligand_group_id=ligand_group.group_id,
                    match_key=key,
                    candidates=candidates,
                )
            )
            for item in candidates:
                for atom_id in item.atom_ids:
                    atom_to_group_index[atom_id] = group_index

    return LigandCorrespondenceAnalysis(
        atom_groups=tuple(atom_groups),
        atom_to_group_index=dict(atom_to_group_index),
        atom_to_fragment_id=dict(atom_to_fragment_id),
        fragment_to_ligand_group_id=dict(fragment_to_group_id),
    )


def rooted_atom_signature(
    graph: StructureGraph,
    fragment_atom_ids: Iterable[int],
    root_atom_id: int,
    *,
    include_coordination_role: bool = True,
) -> str:
    """Return an isomorphism-invariant rooted WL signature for one atom.

    The signature is used for *candidate* correspondence, not as proof that two
    atoms are experimentally NMR-equivalent.  WL equivalence can be coarser
    than a full automorphism orbit, which only makes this resolver more
    conservative (ambiguous) rather than forcing an unsafe unique mapping.
    """

    ids = tuple(sorted(int(value) for value in fragment_atom_ids))
    atom_set = set(ids)
    root_atom_id = int(root_atom_id)
    if root_atom_id not in atom_set:
        raise ValueError(f"root atom {root_atom_id} is not in the fragment")

    adjacency, edge_labels = _induced_covalent_graph(graph, atom_set)
    base_labels = {
        atom_id: _node_label(
            graph,
            atom_id,
            atom_set,
            include_coordination_role=include_coordination_role,
        )
        for atom_id in ids
    }

    colors = {
        atom_id: _stable_digest(
            repr((base_labels[atom_id], atom_id == root_atom_id))
        )
        for atom_id in ids
    }

    for _ in range(max(len(ids), 1)):
        updated: dict[int, str] = {}
        for atom_id in ids:
            neighbours = sorted(
                (
                    edge_labels[_pair(atom_id, neighbour)],
                    colors[neighbour],
                )
                for neighbour in adjacency[atom_id]
            )
            updated[atom_id] = _stable_digest(
                repr(
                    (
                        base_labels[atom_id],
                        atom_id == root_atom_id,
                        colors[atom_id],
                        neighbours,
                    )
                )
            )
        if updated == colors:
            break
        colors = updated

    # Include the whole colored graph multiset so the key is self-contained.
    payload = (
        colors[root_atom_id],
        tuple(sorted(colors.values())),
        len(ids),
        sum(len(values) for values in adjacency.values()) // 2,
    )
    return _stable_digest(repr(payload))


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
        if other in fragment_atom_ids or other not in graph.metal_atom_ids:
            continue
        if edge.kind == "coordination":
            coordination_count += 1
        elif edge.kind == "haptic":
            haptic_count += 1
    return (atom.element, coordination_count, haptic_count)


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
        if edge.order is None:
            label = ("unknown",)
        else:
            try:
                label = ("order", float(edge.order))
            except (TypeError, ValueError):
                label = ("symbolic", str(edge.order))
        edge_labels[_pair(edge.atom1_id, edge.atom2_id)] = label

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
    "AtomMatchGroup",
    "AtomMatchResult",
    "FragmentAtomCandidates",
    "LigandCorrespondenceAnalysis",
    "analyze_ligand_correspondence",
    "rooted_atom_signature",
]
