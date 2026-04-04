# logic/func_group_collapse.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from logic.chem_constants import covalent_radii

# Atom tuple convention used across project:
# Atom = Tuple[str, float, float, float]  # (element, x, y, z)


@dataclass(frozen=True)
class AX3Group:
    """
    Generic AX3 group (C3-like local symmetry):
      center atom A bonded to exactly three ligand atoms X (same element).
    """
    a_idx: int
    x_elem: str
    x_idx: Tuple[int, int, int]


@dataclass(frozen=True)
class CollapseRecord:
    """
    Metadata record describing a collapse operation.

    - group_type: e.g. "methyl", "cf3", "ax3"
    - pseudo_index: index in the *returned* atom list (after collapse)
    - center_index_original: center atom index (A) in the *input* atom list
    - member_indices_original: original atom indices that were collapsed (the 3 X indices)
    - label: human-readable label
    """
    group_type: str
    pseudo_index: int
    center_index_original: int
    member_indices_original: Tuple[int, ...]
    label: str


def _coords(atom_data: Sequence[Tuple[str, float, float, float]]) -> np.ndarray:
    """Return Nx3 coordinate array."""
    return np.array([[x, y, z] for _, x, y, z in atom_data], dtype=float)


def build_bond_graph(
    atom_data: Sequence[Tuple[str, float, float, float]],
    *,
    scale: float = 1.10,
    fallback_radius: float = 0.77,
) -> List[Set[int]]:
    """
    Build adjacency list by covalent radii distance rule:
      d(i,j) <= (r_cov(i) + r_cov(j)) * scale

    Notes:
      - O(N^2) distance check (fine for typical molecules).
      - Uses covalent_radii dict from logic/chem_constants.py.
    """
    n = len(atom_data)
    if n == 0:
        return []

    elements = [el for el, *_ in atom_data]
    radii = np.array([covalent_radii.get(el, fallback_radius) for el in elements], dtype=float)
    xyz = _coords(atom_data)

    neigh: List[Set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dij = float(np.linalg.norm(xyz[i] - xyz[j]))
            cutoff = (radii[i] + radii[j]) * scale
            if dij <= cutoff:
                neigh[i].add(j)
                neigh[j].add(i)
    return neigh


def find_ax3_groups(
    atom_data: Sequence[Tuple[str, float, float, float]],
    neigh: Sequence[Set[int]],
    *,
    center_elem: str,
    ligand_elem: str,
    require_center_nonligand_count: Optional[int] = None,
) -> List[AX3Group]:
    """
    Detect AX3 groups:
      - center atom element == center_elem
      - exactly three neighbors of element == ligand_elem

    Optional filter:
      require_center_nonligand_count:
        - If set (e.g. 1 for CH3 carbon), require that the center atom has exactly
          that many non-ligand neighbors.
        - Set None to disable (more permissive).
    """
    elements = [el for el, *_ in atom_data]
    out: List[AX3Group] = []

    for a_idx, el in enumerate(elements):
        if el != center_elem:
            continue

        x_nbrs = [j for j in neigh[a_idx] if elements[j] == ligand_elem]
        if len(x_nbrs) != 3:
            continue

        if require_center_nonligand_count is not None:
            non_x = [j for j in neigh[a_idx] if elements[j] != ligand_elem]
            if len(non_x) != require_center_nonligand_count:
                continue

        out.append(AX3Group(a_idx=a_idx, x_elem=ligand_elem, x_idx=(x_nbrs[0], x_nbrs[1], x_nbrs[2])))

    return out


def _ax3_pseudo_position_centroid(
    atom_data: Sequence[Tuple[str, float, float, float]],
    g: AX3Group,
) -> np.ndarray:
    """Centroid of the three ligand atoms (simple & robust)."""
    xs = np.array([[atom_data[i][1], atom_data[i][2], atom_data[i][3]] for i in g.x_idx], dtype=float)
    return xs.mean(axis=0)


def collapse_ax3_groups(
    atom_data: Sequence[Tuple[str, float, float, float]],
    *,
    center_elem: str,
    ligand_elem: str,
    mode: str = "mask",
    pseudo_element: Optional[str] = None,
    label_prefix: str = "AX3",
    group_type: str = "ax3",
    require_center_nonligand_count: Optional[int] = None,
    bond_scale: float = 1.03,
) -> Tuple[List[Tuple[str, float, float, float]], List[CollapseRecord], Set[int]]:
    """
    Collapse AX3 ligands (3 X) into one pseudo atom per detected group.

    mode:
      - "mask": keep original atoms, append pseudo atoms, and return 'masked' original indices
      - "drop": remove the 3 X atoms from returned list, append pseudo atoms

    pseudo_element:
      - if None, defaults to ligand_elem (e.g. H->H, F->F)

    Returns:
      (new_atom_data, records, masked_original_indices)
    """
    if mode not in ("mask", "drop"):
        raise ValueError("mode must be 'mask' or 'drop'")

    if pseudo_element is None:
        pseudo_element = ligand_elem

    neigh = build_bond_graph(atom_data, scale=bond_scale)
    groups = find_ax3_groups(
        atom_data,
        neigh,
        center_elem=center_elem,
        ligand_elem=ligand_elem,
        require_center_nonligand_count=require_center_nonligand_count,
    )

    masked: Set[int] = set()
    records: List[CollapseRecord] = []

    # Avoid overlaps (rare, but safe)
    usable: List[AX3Group] = []
    for g in groups:
        if any(i in masked for i in g.x_idx):
            continue
        usable.append(g)
        masked.update(g.x_idx)

    # Build base list (mask vs drop)
    if mode == "mask":
        new_atom_data: List[Tuple[str, float, float, float]] = list(atom_data)
    else:
        new_atom_data = [a for i, a in enumerate(atom_data) if i not in masked]

    # Append pseudo atoms + records
    for g in usable:
        pos = _ax3_pseudo_position_centroid(atom_data, g)
        label = f"{label_prefix}@{center_elem}{g.a_idx + 1}"
        pseudo_tuple = (pseudo_element, float(pos[0]), float(pos[1]), float(pos[2]))

        pseudo_index = len(new_atom_data)
        new_atom_data.append(pseudo_tuple)

        records.append(
            CollapseRecord(
                group_type=group_type,
                pseudo_index=pseudo_index,
                center_index_original=g.a_idx,
                member_indices_original=g.x_idx,
                label=label,
            )
        )

    return new_atom_data, records, masked


# -----------------------------
# Convenience wrappers
# -----------------------------
def collapse_methyl_groups(
    atom_data: Sequence[Tuple[str, float, float, float]],
    *,
    mode: str = "mask",
    pseudo_element: str = "H",
    pseudo_label_prefix: str = "MeH",
    require_carbon_substituent_count: Optional[int] = 1,
    bond_scale: float = 1.03,
) -> Tuple[List[Tuple[str, float, float, float]], List[CollapseRecord], Set[int]]:
    """
    Backward-compatible methyl collapse:
      CH3 = C(H)3 with (optional) exactly one non-H neighbor.
    """
    # For CH3: center=C, ligand=H.
    # "non-ligand neighbors" means non-H neighbors -> classic CH3- has exactly 1.
    require_center_nonligand = require_carbon_substituent_count  # kept name for compatibility

    return collapse_ax3_groups(
        atom_data,
        center_elem="C",
        ligand_elem="H",
        mode=mode,
        pseudo_element=pseudo_element,
        label_prefix=pseudo_label_prefix,
        group_type="methyl",
        require_center_nonligand_count=require_center_nonligand,
        bond_scale=bond_scale,
    )


def collapse_cf3_groups(
    atom_data: Sequence[Tuple[str, float, float, float]],
    *,
    mode: str = "mask",
    pseudo_element: str = "F",
    pseudo_label_prefix: str = "CF3F",
    # CF3 carbon typically has exactly one non-F neighbor (the substituent carbon)
    require_carbon_substituent_count: Optional[int] = 1,
    bond_scale: float = 1.03,
) -> Tuple[List[Tuple[str, float, float, float]], List[CollapseRecord], Set[int]]:
    """
    Collapse CF3 fluorines: C(F)3 -> one pseudo-F.
    """
    require_center_nonligand = require_carbon_substituent_count

    return collapse_ax3_groups(
        atom_data,
        center_elem="C",
        ligand_elem="F",
        mode=mode,
        pseudo_element=pseudo_element,
        label_prefix=pseudo_label_prefix,
        group_type="cf3",
        require_center_nonligand_count=require_center_nonligand,
        bond_scale=bond_scale,
    )


# -----------------------------
# tert-Butyl detector (reporting / validation)
# -----------------------------
def find_tert_butyl_centers(
    atom_data: Sequence[Tuple[str, float, float, float]],
    neigh: Sequence[Set[int]],
    methyl_center_indices: Set[int],
    *,
    require_center_carbon_degree: Optional[int] = 4,
) -> List[int]:
    """
    Find tert-butyl centers (quaternary carbon attached to three methyl carbons).

    Criteria:
      - center atom is carbon
      - among its carbon neighbors, at least 3 are methyl carbons (from methyl detector)
      - optionally require exact carbon degree (typical tBu center has 4 non-H neighbors; in pure graph,
        its carbon neighbors count could be 4 depending on representation; we use a conservative filter)

    Args:
      methyl_center_indices: set of carbon indices that are methyl carbons (CH3 centers)
      require_center_carbon_degree:
        - If 4, require center carbon has exactly 4 carbon neighbors (typical tBu attached to R via carbon).
        - If None, do not enforce degree, only require >=3 methyl-carbon neighbors.

    Returns:
      List of indices (in original atom_data) for tBu center carbons.
    """
    elements = [el for el, *_ in atom_data]
    centers: List[int] = []

    for c0, el in enumerate(elements):
        if el != "C":
            continue

        carbon_nbrs = [j for j in neigh[c0] if elements[j] == "C"]

        if require_center_carbon_degree is not None:
            if len(carbon_nbrs) != require_center_carbon_degree:
                continue

        methyl_carbon_nbrs = [j for j in carbon_nbrs if j in methyl_center_indices]
        if len(methyl_carbon_nbrs) >= 3:
            centers.append(c0)

    return centers