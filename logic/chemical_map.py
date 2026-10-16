"""Chemical-map view model and 2D layout helpers.

D5b keeps :class:`logic.structure_graph.StructureGraph` as the source of truth
and offers two explicit layout modes:

``2d_chemical``
    A topology-driven chemical depiction.  RDKit/CoordGen is used when it is
    available, otherwise a deterministic graph-distance (classical MDS)
    fallback is used.  This mode is independent of the loaded 3D geometry and
    is the default Chemical Map view.

``geometry_projection``
    The original PCA projection of the loaded Cartesian geometry.  It is kept
    as an alternate view because it is useful for recognizing the actual 3D
    ligand conformation even when projected atoms overlap.

RDKit remains OPTIONAL.  Unknown bond orders are not invented: they are drawn
or laid out as single bonds when no numeric order is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Iterable, Mapping

import numpy as np

from logic.ligand_grouping import LigandAnalysis
from logic.structure_graph import StructureGraph


LAYOUT_2D_CHEMICAL = "2d_chemical"
LAYOUT_GEOMETRY_PROJECTION = "geometry_projection"

ENGINE_COORDGEN = "coordgen"
ENGINE_RDKIT_2D = "rdkit_2d"
ENGINE_GRAPH_2D = "graph_2d"


@dataclass(frozen=True)
class ChemicalMapAtom:
    atom_id: int
    element: str
    display_label: str
    is_donor: bool = False
    is_haptic: bool = False
    attached_hydrogen_count: int = 0


@dataclass(frozen=True)
class ChemicalMapBond:
    atom1_id: int
    atom2_id: int
    order: float | None
    bond_type: str = "unknown"


@dataclass(frozen=True)
class ChemicalMapModel:
    group_id: int
    fragment_id: int
    representative_fragment_id: int
    formula: str
    multiplicity: int
    atoms: tuple[ChemicalMapAtom, ...]
    bonds: tuple[ChemicalMapBond, ...]
    ambiguous_correspondence: bool
    attached_metal_ids: tuple[int, ...]
    denticity: int
    hapticity: int

    @property
    def atom_ids(self) -> tuple[int, ...]:
        return tuple(atom.atom_id for atom in self.atoms)


@dataclass(frozen=True)
class ChemicalMapLayout:
    coordinates: Mapping[int, tuple[float, float]]
    backend: str
    mode: str = LAYOUT_2D_CHEMICAL


def build_chemical_map_model(
    graph: StructureGraph,
    analysis: LigandAnalysis,
    *,
    group_id: int,
    fragment_id: int | None = None,
    label_overrides: Mapping[int, str] | None = None,
) -> ChemicalMapModel:
    """Build one physical ligand instance for Chemical Map rendering."""
    group = next(
        (group for group in analysis.groups if group.group_id == int(group_id)),
        None,
    )
    if group is None:
        raise KeyError(f"unknown ligand group ID {group_id}")

    if fragment_id is None:
        fragment_id = group.representative_fragment_id
    fragment_id = int(fragment_id)
    if fragment_id not in group.member_fragment_ids:
        raise ValueError(
            f"fragment {fragment_id} does not belong to ligand group {group_id}"
        )

    ligand = analysis.ligand_by_fragment_id(fragment_id)
    labels = dict(label_overrides or {})
    atom_set = set(ligand.atom_ids)

    covalent_neighbors: dict[int, list[int]] = {atom_id: [] for atom_id in atom_set}
    for edge in graph.edges.values():
        if (
            edge.kind == "covalent"
            and edge.atom1_id in atom_set
            and edge.atom2_id in atom_set
        ):
            covalent_neighbors[edge.atom1_id].append(edge.atom2_id)
            covalent_neighbors[edge.atom2_id].append(edge.atom1_id)

    atoms = tuple(
        ChemicalMapAtom(
            atom_id=atom_id,
            element=graph.atom(atom_id).element,
            display_label=str(labels.get(atom_id, graph.atom(atom_id).element)),
            is_donor=atom_id in ligand.donor_atom_ids,
            is_haptic=atom_id in ligand.haptic_atom_ids,
            attached_hydrogen_count=sum(
                1
                for neighbor in covalent_neighbors.get(atom_id, ())
                if graph.atom(neighbor).element == "H"
            ),
        )
        for atom_id in ligand.atom_ids
    )

    bonds = tuple(
        ChemicalMapBond(
            atom1_id=edge.atom1_id,
            atom2_id=edge.atom2_id,
            order=_numeric_bond_order(edge.order),
            bond_type=_bond_type_from_edge(edge.order, edge.note),
        )
        for edge in sorted(graph.edges.values(), key=lambda e: e.key)
        if edge.kind == "covalent"
        and edge.atom1_id in atom_set
        and edge.atom2_id in atom_set
    )

    correspondence = next(
        (
            item
            for item in group.correspondences
            if item.fragment_id == fragment_id
        ),
        None,
    )
    ambiguous = bool(correspondence.ambiguous) if correspondence is not None else False

    return ChemicalMapModel(
        group_id=group.group_id,
        fragment_id=fragment_id,
        representative_fragment_id=group.representative_fragment_id,
        formula=group.formula,
        multiplicity=group.multiplicity,
        atoms=atoms,
        bonds=bonds,
        ambiguous_correspondence=ambiguous,
        attached_metal_ids=ligand.attached_metal_ids,
        denticity=ligand.denticity,
        hapticity=ligand.hapticity,
    )


def layout_chemical_map(
    graph: StructureGraph,
    model: ChemicalMapModel,
    *,
    mode: str | None = None,
    engine: str | None = None,
    prefer_rdkit: bool | None = None,
) -> ChemicalMapLayout:
    """Return centered/scaled 2D coordinates for a Chemical Map model.

    Parameters
    ----------
    mode:
        ``"2d_chemical"`` (default) or ``"geometry_projection"``.
    engine:
        2D layout backend: ``"coordgen"`` (default), ``"rdkit_2d"`` or
        ``"graph_2d"``.  The first two use RDKit when available; graph mode is
        deterministic and dependency-free.
    prefer_rdkit:
        Compatibility argument retained for D4c/D5b callers/tests.  If
        ``mode`` is omitted and ``prefer_rdkit`` is explicitly ``False``, the
        legacy geometry-projection behavior is selected.  In explicit
        2D-chemical mode, ``False`` forces the graph fallback.
    """
    if mode is None:
        mode = (
            LAYOUT_GEOMETRY_PROJECTION
            if prefer_rdkit is False
            else LAYOUT_2D_CHEMICAL
        )
    mode = normalize_layout_mode(mode)

    if mode == LAYOUT_GEOMETRY_PROJECTION:
        coords = _pca_projection(graph, model.atom_ids)
        return ChemicalMapLayout(
            coordinates=_normalize_coordinates(coords),
            backend="3D geometry projection",
            mode=mode,
        )

    if prefer_rdkit is False:
        engine = ENGINE_GRAPH_2D
    else:
        engine = normalize_layout_engine(engine or ENGINE_COORDGEN)

    if engine != ENGINE_GRAPH_2D:
        result = _try_rdkit_layout(model, engine=engine)
        if result is not None:
            coords, backend = result
            return ChemicalMapLayout(
                coordinates=_normalize_coordinates(coords),
                backend=backend,
                mode=mode,
            )

    coords = _graph_distance_layout(model)
    return ChemicalMapLayout(
        coordinates=_normalize_coordinates(coords),
        backend="Graph 2D",
        mode=mode,
    )


def normalize_layout_mode(value: str) -> str:
    """Normalize UI/API aliases to one of the two supported layout modes."""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "2d": LAYOUT_2D_CHEMICAL,
        "chemical": LAYOUT_2D_CHEMICAL,
        "2d_chemical": LAYOUT_2D_CHEMICAL,
        "2dchemical": LAYOUT_2D_CHEMICAL,
        "geometry": LAYOUT_GEOMETRY_PROJECTION,
        "projection": LAYOUT_GEOMETRY_PROJECTION,
        "geometry_projection": LAYOUT_GEOMETRY_PROJECTION,
        "3d_geometry_projection": LAYOUT_GEOMETRY_PROJECTION,
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(f"unsupported Chemical Map layout mode: {value!r}") from exc




def normalize_layout_engine(value: str) -> str:
    """Normalize 2D backend aliases used by the Chemical Map UI/API."""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "coordgen": ENGINE_COORDGEN,
        "rdkit_coordgen": ENGINE_COORDGEN,
        "chemical": ENGINE_COORDGEN,
        "rdkit": ENGINE_RDKIT_2D,
        "rdkit_2d": ENGINE_RDKIT_2D,
        "rddepictor": ENGINE_RDKIT_2D,
        "depictor": ENGINE_RDKIT_2D,
        "graph": ENGINE_GRAPH_2D,
        "graph_2d": ENGINE_GRAPH_2D,
        "fallback": ENGINE_GRAPH_2D,
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(f"unsupported Chemical Map 2D engine: {value!r}") from exc

def find_model_for_atom(
    analysis: LigandAnalysis,
    atom_id: int,
) -> tuple[int, int] | None:
    """Return ``(group_id, fragment_id)`` containing an atom, if coordinating."""
    atom_id = int(atom_id)
    for ligand in analysis.ligands:
        if atom_id not in ligand.atom_ids:
            continue
        group = analysis.group_for_fragment_id(ligand.fragment_id)
        return group.group_id, ligand.fragment_id
    return None



def _bond_type_from_edge(order, note) -> str:
    text = str(note or "")
    marker = "chemical_map_bond_type="
    if marker in text:
        value = text.split(marker, 1)[1].split(";", 1)[0].strip().lower()
        if value in {"single", "double", "triple", "aromatic", "delocalized", "unknown"}:
            return value
    numeric = _numeric_bond_order(order)
    if numeric is None:
        return "unknown"
    if numeric >= 2.5:
        return "triple"
    if numeric >= 1.75:
        return "double"
    if 1.40 <= numeric < 1.75:
        return "aromatic"
    if 1.10 < numeric < 1.40:
        return "delocalized"
    return "single"

def _numeric_bond_order(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def _try_rdkit_layout(
    model: ChemicalMapModel,
    *,
    engine: str = ENGINE_COORDGEN,
) -> tuple[dict[int, tuple[float, float]], str] | None:
    try:
        from rdkit import Chem
    except Exception:
        return None

    try:
        mol = Chem.RWMol()
        index_by_atom_id: dict[int, int] = {}

        for atom in model.atoms:
            index_by_atom_id[atom.atom_id] = mol.AddAtom(Chem.Atom(atom.element))

        for bond in model.bonds:
            order = bond.order
            if bond.bond_type == "triple" or (order is not None and order >= 2.5):
                rd_bond_type = Chem.BondType.TRIPLE
            elif bond.bond_type == "double" or (
                order is not None and order >= 1.75
            ):
                rd_bond_type = Chem.BondType.DOUBLE
            else:
                # Aromatic/delocalized semantics are intentionally kept out of
                # RDKit sanitization here.  CoordGen only needs connectivity;
                # Chemical Map renders those styles itself.
                rd_bond_type = Chem.BondType.SINGLE

            mol.AddBond(
                index_by_atom_id[bond.atom1_id],
                index_by_atom_id[bond.atom2_id],
                rd_bond_type,
            )

        rd_mol = mol.GetMol()
        engine = normalize_layout_engine(engine)
        if engine == ENGINE_RDKIT_2D:
            from rdkit.Chem import rdDepictor

            rdDepictor.Compute2DCoords(
                rd_mol,
                canonOrient=True,
                clearConfs=True,
            )
            backend = "RDKit 2D"
        else:
            try:
                # CoordGen tends to produce a conventional chemical-diagram
                # layout for rings and branched ligand fragments.
                from rdkit.Chem import rdCoordGen

                rdCoordGen.AddCoords(rd_mol)
                backend = "RDKit CoordGen"
            except Exception:
                from rdkit.Chem import rdDepictor

                rdDepictor.Compute2DCoords(
                    rd_mol,
                    canonOrient=True,
                    clearConfs=True,
                )
                backend = "RDKit 2D (CoordGen unavailable)"

        conf = rd_mol.GetConformer()
        result = {}
        for atom_id, index in index_by_atom_id.items():
            pos = conf.GetAtomPosition(index)
            result[atom_id] = (float(pos.x), float(pos.y))
        return result, backend
    except Exception:
        # A depiction failure must never prevent Chemical Map from opening.
        return None


def _graph_distance_layout(
    model: ChemicalMapModel,
) -> dict[int, tuple[float, float]]:
    """Deterministic topology-only 2D fallback using graph-distance MDS.

    The fallback intentionally does not look at Cartesian coordinates.  It is
    therefore a real alternate representation rather than another projection
    of the same 3D geometry.  Classical MDS of shortest-path distances gives
    sensible layouts for chains, rings and branched groups and avoids the atom
    pile-ups that motivated the D5b Chemical Map redesign.
    """
    ids = tuple(model.atom_ids)
    n = len(ids)
    if n == 0:
        return {}
    if n == 1:
        return {ids[0]: (0.0, 0.0)}

    index = {atom_id: i for i, atom_id in enumerate(ids)}
    adjacency: list[list[int]] = [[] for _ in ids]
    for bond in model.bonds:
        if bond.atom1_id not in index or bond.atom2_id not in index:
            continue
        i = index[bond.atom1_id]
        j = index[bond.atom2_id]
        adjacency[i].append(j)
        adjacency[j].append(i)

    distances = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(distances, 0.0)
    for start in range(n):
        queue = deque([start])
        while queue:
            current = queue.popleft()
            base = distances[start, current]
            for neighbor in adjacency[current]:
                if np.isfinite(distances[start, neighbor]):
                    continue
                distances[start, neighbor] = base + 1.0
                queue.append(neighbor)

    # Ligand fragments should be connected.  Be defensive for hand-edited
    # connectivity by assigning separated components a finite large distance.
    finite = distances[np.isfinite(distances)]
    max_finite = float(np.max(finite)) if finite.size else 1.0
    distances[~np.isfinite(distances)] = max_finite + 2.0

    squared = distances ** 2
    centering = np.eye(n) - np.ones((n, n), dtype=float) / n
    gram = -0.5 * centering @ squared @ centering

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        positive = [idx for idx in order if eigenvalues[idx] > 1e-10][:2]
        coords = np.zeros((n, 2), dtype=float)
        for axis, eig_idx in enumerate(positive):
            coords[:, axis] = (
                eigenvectors[:, eig_idx] * np.sqrt(eigenvalues[eig_idx])
            )
    except np.linalg.LinAlgError:
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        coords = np.column_stack((np.cos(angles), np.sin(angles)))

    # Canonical deterministic orientation: largest-spread axis is horizontal;
    # sign is fixed by the lowest atom ID so switching ligand instances does
    # not randomly mirror the display between redraws.
    if np.ptp(coords[:, 1]) > np.ptp(coords[:, 0]):
        coords = coords[:, [1, 0]]
    anchor = int(np.argmin(np.asarray(ids, dtype=int)))
    if coords[anchor, 0] > 0:
        coords[:, 0] *= -1.0
    if abs(coords[anchor, 0]) <= 1e-12 and coords[anchor, 1] > 0:
        coords[:, 1] *= -1.0

    return {
        atom_id: (float(coords[i, 0]), float(coords[i, 1]))
        for i, atom_id in enumerate(ids)
    }


def _pca_projection(
    graph: StructureGraph,
    atom_ids: Iterable[int],
) -> dict[int, tuple[float, float]]:
    ids = tuple(int(atom_id) for atom_id in atom_ids)
    if not ids:
        return {}
    if len(ids) == 1:
        return {ids[0]: (0.0, 0.0)}

    coords = np.asarray(
        [
            (
                graph.atom(atom_id).x,
                graph.atom(atom_id).y,
                graph.atom(atom_id).z,
            )
            for atom_id in ids
        ],
        dtype=float,
    )
    centered = coords - np.mean(coords, axis=0, keepdims=True)

    if len(ids) == 2:
        delta = centered[1] - centered[0]
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-12:
            projected = np.asarray([[-0.5, 0.0], [0.5, 0.0]])
        else:
            direction = delta / norm
            projected = np.column_stack(
                (centered @ direction, np.zeros(len(ids), dtype=float))
            )
    else:
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            basis = vh[:2].T
            projected = centered @ basis
        except np.linalg.LinAlgError:
            projected = centered[:, :2]

    return {
        atom_id: (float(projected[i, 0]), float(projected[i, 1]))
        for i, atom_id in enumerate(ids)
    }


def _normalize_coordinates(
    coordinates: Mapping[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    if not coordinates:
        return {}

    ids = list(coordinates)
    arr = np.asarray([coordinates[atom_id] for atom_id in ids], dtype=float)
    arr = arr - np.mean(arr, axis=0, keepdims=True)

    span_x = float(np.ptp(arr[:, 0])) if len(arr) else 0.0
    span_y = float(np.ptp(arr[:, 1])) if len(arr) else 0.0
    span = max(span_x, span_y, 1e-9)
    arr = arr / span

    return {
        atom_id: (float(arr[i, 0]), float(arr[i, 1]))
        for i, atom_id in enumerate(ids)
    }


__all__ = [
    "LAYOUT_2D_CHEMICAL",
    "LAYOUT_GEOMETRY_PROJECTION",
    "ChemicalMapAtom",
    "ChemicalMapBond",
    "ChemicalMapLayout",
    "ChemicalMapModel",
    "build_chemical_map_model",
    "find_model_for_atom",
    "layout_chemical_map",
    "normalize_layout_mode",
]
