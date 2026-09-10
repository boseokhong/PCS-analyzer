# logic/torsional_ensemble.py
# PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE2

"""Planar-ring detection and torsional ensemble sampling for PCS Analyzer.

Phase 2 provides geometry/detection infrastructure and configuration helpers.
It deliberately does not modify the PCS calculation pipeline yet; integration
with symmetry-aware G averaging is performed by the following patch phase.

Design principles
-----------------
- Auto detection operates on a ligand-only covalent graph. Metal atoms and
  metal-ligand coordination edges are excluded from cycle detection.
- A planar cyclic/fused fragment is only promoted to an auto rotor when a
  non-metal external attachment bond can define a disconnected rotating side.
- Ring names are descriptive UI labels only. They are never used in PCS maths.
- Torsional samples preserve atom identities; no pseudo atoms are created.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import Counter, deque
from typing import Iterable, Sequence

import numpy as np

from logic.chem_constants import covalent_radii, METAL_ELEMENTS


DEFAULT_BOND_SCALE = 1.20
DEFAULT_METAL_BOND_SCALE = 1.25
DEFAULT_PLANARITY_THRESHOLD = 0.10  # Angstrom RMS distance from best-fit plane
DEFAULT_MIN_RING_SIZE = 3
DEFAULT_MAX_RING_SIZE = 8
DEFAULT_SAMPLE_COUNT = 24

_COORD_DONORS = {"N", "O", "F", "P", "S", "Cl", "Br", "I", "As", "Se"}


# Composition-only UI descriptors. Positional isomers intentionally share labels.
RING_CLASSIFICATION = {
    # 3-membered
    (3, (("C", 3),)): "Cyclopropenyl-like",
    (3, (("C", 2), ("N", 1))): "Azirine-like",
    (3, (("C", 2), ("O", 1))): "Oxirene-like",
    (3, (("C", 2), ("S", 1))): "Thiirene-like",
    (3, (("C", 1), ("N", 2))): "Diazirine-like",

    # 4-membered
    (4, (("C", 4),)): "Cyclobutadienyl-like",
    (4, (("C", 3), ("N", 1))): "Azete-like",
    (4, (("C", 3), ("O", 1))): "Oxete-like",
    (4, (("C", 3), ("S", 1))): "Thiete-like",
    (4, (("C", 2), ("N", 2))): "Diazete-like",

    # 5-membered
    (5, (("C", 4), ("N", 1))): "Pyrrolyl-like",
    (5, (("C", 4), ("O", 1))): "Furyl-like",
    (5, (("C", 4), ("S", 1))): "Thienyl-like",
    (5, (("C", 3), ("N", 2))): "Diazolyl-like",
    (5, (("C", 3), ("N", 1), ("O", 1))): "Oxazolyl-like",
    (5, (("C", 3), ("N", 1), ("S", 1))): "Thiazolyl-like",
    (5, (("C", 3), ("O", 2))): "Dioxolyl-like",
    (5, (("C", 3), ("O", 1), ("S", 1))): "Oxathiolyl-like",
    (5, (("C", 3), ("S", 2))): "Dithiolyl-like",
    (5, (("C", 2), ("N", 3))): "Triazolyl-like",
    (5, (("C", 2), ("N", 2), ("O", 1))): "Oxadiazolyl-like",
    (5, (("C", 2), ("N", 2), ("S", 1))): "Thiadiazolyl-like",
    (5, (("C", 1), ("N", 4))): "Tetrazolyl-like",

    # 6-membered
    (6, (("C", 6),)): "Phenyl-like",
    (6, (("C", 5), ("N", 1))): "Pyridyl-like",
    (6, (("C", 4), ("N", 2))): "Diazine-like",
    (6, (("C", 3), ("N", 3))): "Triazine-like",
    (6, (("C", 2), ("N", 4))): "Tetrazine-like",
    (6, (("C", 5), ("O", 1))): "Pyran-like",
    (6, (("C", 5), ("S", 1))): "Thiopyran-like",
    (6, (("C", 4), ("N", 1), ("O", 1))): "Oxazine-like",
    (6, (("C", 4), ("N", 1), ("S", 1))): "Thiazine-like",
    (6, (("C", 4), ("O", 2))): "Dioxine-like",
    (6, (("C", 4), ("O", 1), ("S", 1))): "Oxathiin-like",
    (6, (("C", 4), ("S", 2))): "Dithiine-like",
}


@dataclass
class TorsionalGroup:
    group_id: str
    label: str
    axis_atoms: tuple[int, int]
    rotating_atoms: tuple[int, ...]
    ring_atoms: tuple[int, ...] = ()
    ring_cycles: tuple[tuple[int, ...], ...] = ()
    source: str = "auto"
    enabled: bool = True
    range_mode: str = "full"          # full | restricted
    range_start_deg: float = 0.0
    range_end_deg: float = 360.0
    n_samples: int = DEFAULT_SAMPLE_COUNT
    weight_mode: str = "uniform"
    planarity_rms: float = 0.0

    def to_dict(self) -> dict:
        out = asdict(self)
        out["axis_atoms"] = tuple(int(x) for x in self.axis_atoms)
        out["rotating_atoms"] = tuple(int(x) for x in self.rotating_atoms)
        out["ring_atoms"] = tuple(int(x) for x in self.ring_atoms)
        out["ring_cycles"] = tuple(tuple(int(x) for x in cyc) for cyc in self.ring_cycles)
        return out


def _as_group_dict(group) -> dict:
    if isinstance(group, TorsionalGroup):
        return group.to_dict()
    return dict(group)


def group_signature(group) -> tuple:
    """Stable signature used to preserve UI settings after re-detection."""
    g = _as_group_dict(group)
    axis = tuple(int(x) for x in g.get("axis_atoms", ()))
    ring = tuple(sorted(int(x) for x in g.get("ring_atoms", ())))
    source = str(g.get("source", "auto"))
    if source == "manual":
        rotating = tuple(sorted(int(x) for x in g.get("rotating_atoms", ())))
        return source, axis, rotating
    return source, axis, ring


def _composition_key(elements: Iterable[str]) -> tuple[tuple[str, int], ...]:
    counts = Counter(str(el) for el in elements)
    return tuple(sorted(((el, int(n)) for el, n in counts.items()), key=lambda x: x[0]))


def composition_formula(elements: Iterable[str]) -> str:
    """Return a compact Hill-like formula for UI fallback labels."""
    counts = Counter(str(el) for el in elements)
    ordered = []
    if "C" in counts:
        ordered.append("C")
    if "H" in counts:
        ordered.append("H")
    ordered.extend(sorted(k for k in counts if k not in {"C", "H"}))
    parts = []
    for el in ordered:
        n = counts[el]
        parts.append(el if n == 1 else f"{el}{n}")
    return "".join(parts) or "ring"


def classify_planar_ring(elements: Sequence[str]) -> str:
    n = len(elements)
    key = (n, _composition_key(elements))
    known = RING_CLASSIFICATION.get(key)
    if known:
        return known
    formula = composition_formula(elements)
    return f"Planar {n}-member ring ({formula})"


def classify_fused_system(elements: Sequence[str], cycle_sizes: Sequence[int]) -> str:
    """Broad composition-based UI descriptor for a fused planar system."""
    comp = Counter(str(el) for el in elements)
    sizes = tuple(sorted(int(x) for x in cycle_sizes))
    n_atoms = len(elements)

    if sizes == (6, 6):
        if comp == Counter({"C": 10}):
            return "Naphthalene-like fused system"
        if comp == Counter({"C": 9, "N": 1}):
            return "Quinoline-like fused system"
        if comp == Counter({"C": 8, "N": 2}):
            return "Benzodiazine-like fused system"

    if sizes == (5, 6):
        if comp == Counter({"C": 8, "N": 1}):
            return "Indole-like fused system"
        if comp == Counter({"C": 8, "O": 1}):
            return "Benzofuran-like fused system"
        if comp == Counter({"C": 8, "S": 1}):
            return "Benzothiophene-like fused system"
        if comp == Counter({"C": 7, "N": 2}):
            return "Benzodiazole-like fused system"
        if comp == Counter({"C": 7, "N": 1, "O": 1}):
            return "Benzoxazole-like fused system"
        if comp == Counter({"C": 7, "N": 1, "S": 1}):
            return "Benzothiazole-like fused system"
        if comp == Counter({"C": 5, "N": 4}):
            return "Purine-like fused system"

    formula = composition_formula(elements)
    return f"Fused planar system ({len(cycle_sizes)} rings, {formula})"


def build_graph(
    atom_data: Sequence[tuple[str, float, float, float]],
    *,
    bond_scale: float = DEFAULT_BOND_SCALE,
    include_metals: bool = False,
    metal_bond_scale: float = DEFAULT_METAL_BOND_SCALE,
) -> list[set[int]]:
    """Build a conservative molecular adjacency list from covalent radii.

    ``include_metals=False`` is used for ring detection and completely removes
    metal vertices/coordination edges from the cycle graph.
    """
    n = len(atom_data)
    adj = [set() for _ in range(n)]
    if n < 2:
        return adj

    elems = [str(a[0]) for a in atom_data]
    xyz = np.asarray([[a[1], a[2], a[3]] for a in atom_data], dtype=float)

    for i in range(n):
        ei = elems[i]
        for j in range(i + 1, n):
            ej = elems[j]
            mi = ei in METAL_ELEMENTS
            mj = ej in METAL_ELEMENTS

            if (mi or mj) and not include_metals:
                continue
            if mi and mj:
                continue

            ri = float(covalent_radii.get(ei, covalent_radii.get("default", 0.80)))
            rj = float(covalent_radii.get(ej, covalent_radii.get("default", 0.80)))
            scale = float(bond_scale)

            if mi or mj:
                partner = ej if mi else ei
                if partner not in _COORD_DONORS:
                    continue
                scale = float(metal_bond_scale)

            if float(np.linalg.norm(xyz[i] - xyz[j])) <= (ri + rj) * scale:
                adj[i].add(j)
                adj[j].add(i)

    return adj


def _canonical_cycle(cycle: Sequence[int]) -> tuple[int, ...]:
    cyc = tuple(int(x) for x in cycle)
    n = len(cyc)
    variants = []
    for seq in (cyc, tuple(reversed(cyc))):
        for k in range(n):
            variants.append(seq[k:] + seq[:k])
    return min(variants)


def find_simple_cycles(
    adj: Sequence[set[int]],
    *,
    min_size: int = DEFAULT_MIN_RING_SIZE,
    max_size: int = DEFAULT_MAX_RING_SIZE,
) -> list[tuple[int, ...]]:
    """Enumerate unique simple cycles up to ``max_size`` in a sparse graph."""
    n = len(adj)
    found: set[tuple[int, ...]] = set()

    for start in range(n):
        if not adj[start]:
            continue

        def dfs(current: int, path: list[int], visited: set[int]) -> None:
            if len(path) > max_size:
                return
            for nb in adj[current]:
                # Enforce start as the smallest index in a cycle to reduce work.
                if nb < start:
                    continue
                if nb == start:
                    if len(path) >= min_size:
                        found.add(_canonical_cycle(path))
                    continue
                if nb in visited or len(path) >= max_size:
                    continue
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.remove(nb)

        dfs(start, [start], {start})

    return sorted(found, key=lambda c: (len(c), c))


def fit_planarity(coords: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return (rms_distance, max_distance, plane_normal)."""
    pts = np.asarray(coords, dtype=float)
    if len(pts) < 3:
        return float("inf"), float("inf"), np.zeros(3)

    center = pts.mean(axis=0)
    X = pts - center
    try:
        _, s, vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("inf"), float("inf"), np.zeros(3)

    # Reject nearly collinear 3-member candidates.
    if len(pts) == 3 and (len(s) < 2 or float(s[1]) < 1e-6):
        return float("inf"), float("inf"), np.zeros(3)

    normal = np.asarray(vt[-1], dtype=float)
    distances = np.abs(X @ normal)
    rms = float(np.sqrt(np.mean(distances ** 2)))
    max_dev = float(np.max(distances))
    return rms, max_dev, normal


def _merge_fused_cycles(
    planar_cycles: Sequence[tuple[int, ...]],
    xyz: np.ndarray,
    *,
    threshold: float,
) -> list[tuple[tuple[int, ...], ...]]:
    """Merge edge-sharing planar cycles when their combined atoms remain planar."""
    if not planar_cycles:
        return []

    n = len(planar_cycles)
    nbr = [set() for _ in range(n)]
    for i in range(n):
        si = set(planar_cycles[i])
        for j in range(i + 1, n):
            sj = set(planar_cycles[j])
            if len(si & sj) < 2:
                continue
            atoms = sorted(si | sj)
            rms, _, _ = fit_planarity(xyz[atoms])
            if rms <= float(threshold):
                nbr[i].add(j)
                nbr[j].add(i)

    seen = set()
    systems = []
    for i in range(n):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        idxs = []
        while stack:
            k = stack.pop()
            idxs.append(k)
            for q in nbr[k]:
                if q not in seen:
                    seen.add(q)
                    stack.append(q)
        systems.append(tuple(planar_cycles[k] for k in sorted(idxs)))
    return systems


def _component_without_edge(adj: Sequence[set[int]], start: int, a: int, b: int) -> set[int]:
    seen = {int(start)}
    q = deque([int(start)])
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if (cur == a and nb == b) or (cur == b and nb == a):
                continue
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    return seen


def _shortest_distance_to_metal(adj: Sequence[set[int]], elements: Sequence[str], start: int) -> int | None:
    if elements[start] in METAL_ELEMENTS:
        return 0
    q = deque([(start, 0)])
    seen = {start}
    while q:
        cur, d = q.popleft()
        for nb in adj[cur]:
            if nb in seen:
                continue
            if elements[nb] in METAL_ELEMENTS:
                return d + 1
            seen.add(nb)
            q.append((nb, d + 1))
    return None


def _choose_attachment_axis(
    system_atoms: set[int],
    ligand_adj: Sequence[set[int]],
    full_adj: Sequence[set[int]],
    elements: Sequence[str],
) -> tuple[tuple[int, int], set[int]] | None:
    """Choose the most plausible scaffold attachment bond for one planar system.

    Returns ``((anchor_index, ring_index), rotating_component_indices)``.
    Metal-containing axes are rejected in auto mode.
    """
    candidates = []
    for ring_idx in sorted(system_atoms):
        for ext in sorted(ligand_adj[ring_idx]):
            if ext in system_atoms:
                continue
            if elements[ext] == "H":
                continue
            if elements[ring_idx] in METAL_ELEMENTS or elements[ext] in METAL_ELEMENTS:
                continue

            ring_side = _component_without_edge(ligand_adj, ring_idx, ring_idx, ext)
            ext_side = _component_without_edge(ligand_adj, ext, ring_idx, ext)
            if ext in ring_side or ring_idx in ext_side:
                # The edge participates in another path/cycle and is not a clean torsional cut.
                continue
            if not system_atoms.issubset(ring_side):
                continue

            metal_dist = _shortest_distance_to_metal(full_adj, elements, ext)
            metal_rank = 0 if metal_dist is not None else 1
            dist_rank = metal_dist if metal_dist is not None else 10**6
            # Prefer a metal-connected anchor, then the nearest metal path, then
            # a larger anchor-side scaffold over a tiny substituent.
            score = (metal_rank, dist_rank, -len(ext_side), ext, ring_idx)
            candidates.append((score, (ext, ring_idx), ring_side))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _, axis, rotating = candidates[0]
    return axis, rotating


def detect_planar_rotors(
    atom_data: Sequence[tuple[str, float, float, float]],
    ref_ids: Sequence[int] | None = None,
    *,
    planarity_threshold: float = DEFAULT_PLANARITY_THRESHOLD,
    bond_scale: float = DEFAULT_BOND_SCALE,
    min_ring_size: int = DEFAULT_MIN_RING_SIZE,
    max_ring_size: int = DEFAULT_MAX_RING_SIZE,
) -> list[TorsionalGroup]:
    """Detect ligand-only planar cyclic/fused rotational groups."""
    if not atom_data:
        return []

    n = len(atom_data)
    refs = list(ref_ids) if ref_ids is not None else list(range(1, n + 1))
    if len(refs) != n:
        refs = list(range(1, n + 1))

    elements = [str(a[0]) for a in atom_data]
    xyz = np.asarray([[a[1], a[2], a[3]] for a in atom_data], dtype=float)

    ligand_adj = build_graph(atom_data, bond_scale=bond_scale, include_metals=False)
    full_adj = build_graph(atom_data, bond_scale=bond_scale, include_metals=True)

    cycles = find_simple_cycles(ligand_adj, min_size=min_ring_size, max_size=max_ring_size)
    planar = []
    for cyc in cycles:
        if any(elements[i] in METAL_ELEMENTS for i in cyc):
            continue
        rms, _, _ = fit_planarity(xyz[list(cyc)])
        if rms <= float(planarity_threshold):
            planar.append(cyc)

    systems = _merge_fused_cycles(planar, xyz, threshold=float(planarity_threshold))
    groups: list[TorsionalGroup] = []

    for system_cycles in systems:
        system_atoms = set().union(*(set(c) for c in system_cycles))
        rms, _, _ = fit_planarity(xyz[sorted(system_atoms)])
        if rms > float(planarity_threshold):
            continue

        axis_info = _choose_attachment_axis(system_atoms, ligand_adj, full_adj, elements)
        if axis_info is None:
            continue
        (anchor_idx, ring_idx), rotating_idx = axis_info

        cycle_refs = tuple(tuple(int(refs[i]) for i in cyc) for cyc in system_cycles)
        ring_refs = tuple(sorted(int(refs[i]) for i in system_atoms))
        rotating_refs = tuple(sorted(int(refs[i]) for i in rotating_idx if elements[i] not in METAL_ELEMENTS))
        axis_refs = (int(refs[anchor_idx]), int(refs[ring_idx]))

        if len(system_cycles) == 1:
            cyc = system_cycles[0]
            label = classify_planar_ring([elements[i] for i in cyc])
        else:
            label = classify_fused_system(
                [elements[i] for i in sorted(system_atoms)],
                [len(cyc) for cyc in system_cycles],
            )

        groups.append(
            TorsionalGroup(
                group_id="",
                label=label,
                axis_atoms=axis_refs,
                rotating_atoms=rotating_refs,
                ring_atoms=ring_refs,
                ring_cycles=cycle_refs,
                source="auto",
                enabled=True,
                planarity_rms=float(rms),
            )
        )

    groups.sort(key=lambda g: (g.axis_atoms, g.ring_atoms, g.label))
    for i, group in enumerate(groups, start=1):
        group.group_id = f"rot{i}"
    return groups


def merge_group_settings(detected: Sequence, existing: Sequence | None) -> list[dict]:
    """Preserve user settings for re-detected groups with the same signature."""
    old = {group_signature(g): _as_group_dict(g) for g in (existing or [])}
    out = []
    for item in detected:
        g = _as_group_dict(item)
        prev = old.get(group_signature(g))
        if prev:
            for key in (
                "enabled", "range_mode", "range_start_deg", "range_end_deg",
                "n_samples", "weight_mode",
            ):
                if key in prev:
                    g[key] = prev[key]
        out.append(g)

    # Manual groups do not come from auto detection; keep them.
    for item in existing or []:
        g = _as_group_dict(item)
        if str(g.get("source", "auto")) == "manual":
            out.append(g)

    # Re-number display IDs without changing signatures.
    for i, g in enumerate(out, start=1):
        g["group_id"] = f"rot{i}"
    return out


def build_manual_torsion_group(
    atom_data: Sequence[tuple[str, float, float, float]],
    ref_ids: Sequence[int],
    axis_a_ref: int,
    axis_b_ref: int,
    *,
    moving_side_ref: int | None = None,
) -> TorsionalGroup:
    """Create a manual torsional group from an existing graph bond.

    Metal-containing axes are allowed here by design because manual mode is an
    explicit expert override. ``moving_side_ref`` selects which disconnected
    side of the cut bond rotates; when omitted, the side without a metal is
    preferred.
    """
    refs = list(int(x) for x in ref_ids)
    if len(refs) != len(atom_data):
        raise ValueError("Ref-ID count does not match atom count.")
    id2idx = {rid: i for i, rid in enumerate(refs)}
    if axis_a_ref not in id2idx or axis_b_ref not in id2idx:
        raise ValueError("Manual axis Ref IDs were not found in the current structure.")

    a = id2idx[int(axis_a_ref)]
    b = id2idx[int(axis_b_ref)]
    adj = build_graph(atom_data, include_metals=True)
    if b not in adj[a]:
        raise ValueError("The selected manual axis is not a detected bond.")

    side_a = _component_without_edge(adj, a, a, b)
    side_b = _component_without_edge(adj, b, a, b)
    if b in side_a or a in side_b:
        raise ValueError("The selected bond does not define two separable torsional fragments.")

    elems = [str(x[0]) for x in atom_data]
    if moving_side_ref is not None:
        if int(moving_side_ref) not in id2idx:
            raise ValueError("Moving-side Ref ID was not found.")
        m = id2idx[int(moving_side_ref)]
        if m in side_a:
            rotating = side_a
        elif m in side_b:
            rotating = side_b
        else:
            raise ValueError("Moving-side Ref ID is not connected to either axis fragment.")
    else:
        a_has_metal = any(elems[i] in METAL_ELEMENTS for i in side_a)
        b_has_metal = any(elems[i] in METAL_ELEMENTS for i in side_b)
        if a_has_metal and not b_has_metal:
            rotating = side_b
        elif b_has_metal and not a_has_metal:
            rotating = side_a
        else:
            rotating = side_b

    rotating_refs = tuple(sorted(refs[i] for i in rotating))
    return TorsionalGroup(
        group_id="manual",
        label="Manual torsion",
        axis_atoms=(int(axis_a_ref), int(axis_b_ref)),
        rotating_atoms=rotating_refs,
        source="manual",
        enabled=True,
    )


def sampling_angles(group) -> np.ndarray:
    g = _as_group_dict(group)
    n = max(1, int(g.get("n_samples", DEFAULT_SAMPLE_COUNT)))
    mode = str(g.get("range_mode", "full")).lower()
    if mode == "restricted":
        start = float(g.get("range_start_deg", 0.0))
        end = float(g.get("range_end_deg", 360.0))
        if n == 1:
            return np.asarray([start], dtype=float)
        return np.linspace(start, end, n, endpoint=True, dtype=float)
    return np.linspace(0.0, 360.0, n, endpoint=False, dtype=float)


def uniform_weights(group) -> np.ndarray:
    angles = sampling_angles(group)
    return np.full(len(angles), 1.0 / max(1, len(angles)), dtype=float)


def _rotate_points_about_axis(points: np.ndarray, p0: np.ndarray, p1: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(p1, float) - np.asarray(p0, float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("Torsion axis has zero length.")
    u = axis / norm
    th = np.deg2rad(float(angle_deg))
    K = np.array(
        [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]],
        dtype=float,
    )
    R = np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)
    q = np.asarray(points, float) - np.asarray(p0, float)
    return q @ R.T + np.asarray(p0, float)


def sample_group_coordinates(coords: np.ndarray, ref_ids: Sequence[int], group) -> np.ndarray:
    """Return sampled full-molecule coordinates with shape (S, N, 3)."""
    xyz = np.asarray(coords, dtype=float)
    refs = list(int(x) for x in ref_ids)
    if xyz.shape != (len(refs), 3):
        raise ValueError("Coordinate/Ref-ID shape mismatch.")
    g = _as_group_dict(group)
    id2idx = {rid: i for i, rid in enumerate(refs)}
    a_ref, b_ref = (int(x) for x in g.get("axis_atoms", ()))
    if a_ref not in id2idx or b_ref not in id2idx:
        raise ValueError("Torsion axis atoms are missing from coordinates.")
    moving_idx = [id2idx[int(r)] for r in g.get("rotating_atoms", ()) if int(r) in id2idx]
    if not moving_idx:
        raise ValueError("Torsional group has no rotating atoms.")

    angles = sampling_angles(g)
    out = np.repeat(xyz[None, :, :], len(angles), axis=0)
    p0 = xyz[id2idx[a_ref]]
    p1 = xyz[id2idx[b_ref]]
    for k, angle in enumerate(angles):
        out[k, moving_idx, :] = _rotate_points_about_axis(xyz[moving_idx], p0, p1, float(angle))
    return out


def average_group_coordinates(coords: np.ndarray, ref_ids: Sequence[int], group) -> np.ndarray:
    samples = sample_group_coordinates(coords, ref_ids, group)
    weights = uniform_weights(group)
    return np.tensordot(weights, samples, axes=(0, 0))


def validate_independent_groups(groups: Sequence) -> list[tuple[str, str, tuple[int, ...]]]:
    """Return overlaps between enabled rotating fragments."""
    enabled = [_as_group_dict(g) for g in groups if bool(_as_group_dict(g).get("enabled", True))]
    overlaps = []
    for i in range(len(enabled)):
        ai = set(int(x) for x in enabled[i].get("rotating_atoms", ()))
        for j in range(i + 1, len(enabled)):
            aj = set(int(x) for x in enabled[j].get("rotating_atoms", ()))
            common = tuple(sorted(ai & aj))
            if common:
                overlaps.append((str(enabled[i].get("group_id", i + 1)), str(enabled[j].get("group_id", j + 1)), common))
    return overlaps


# PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE3

def enabled_groups(groups: Sequence | None) -> list[dict]:
    """Return enabled torsional groups as plain dictionaries."""
    return [
        _as_group_dict(g)
        for g in (groups or [])
        if bool(_as_group_dict(g).get("enabled", True))
    ]


def average_coordinates_independent(
    coords: np.ndarray,
    ref_ids: Sequence[int],
    groups: Sequence | None,
) -> np.ndarray:
    """Return representative coordinates for independent torsional groups.

    Each enabled group is sampled from the same reference structure.  Only the
    coordinates belonging to that group's rotating fragment are replaced by
    their weighted ensemble mean.  Enabled groups must therefore have disjoint
    rotating fragments; ``validate_independent_groups`` can be used before this
    function.
    """
    xyz = np.asarray(coords, dtype=float)
    refs = [int(x) for x in ref_ids]
    if xyz.shape != (len(refs), 3):
        raise ValueError("Coordinate/Ref-ID shape mismatch.")

    active = enabled_groups(groups)
    if not active:
        return xyz.copy()

    overlaps = validate_independent_groups(active)
    if overlaps:
        a, b, common = overlaps[0]
        raise ValueError(
            f"Torsional groups {a} and {b} overlap at Ref IDs {common}."
        )

    id2idx = {rid: i for i, rid in enumerate(refs)}
    out = xyz.copy()
    for group in active:
        samples = sample_group_coordinates(xyz, refs, group)
        weights = uniform_weights(group)
        avg = np.tensordot(weights, samples, axes=(0, 0))
        for rid in group.get("rotating_atoms", ()):
            rid = int(rid)
            idx = id2idx.get(rid)
            if idx is not None:
                out[idx] = avg[idx]
    return out


def ensemble_geometry_factors_independent(
    coords: np.ndarray,
    ref_ids: Sequence[int],
    groups: Sequence | None,
    metal=(0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return torsion-ensemble averaged Gax and Grh for all Ref IDs.

    The base factors are calculated at the reference coordinates.  For each
    enabled independent torsion, factors for atoms in its rotating fragment are
    replaced by the weighted average over sampled conformers.  This avoids the
    incorrect shortcut ``G(<r>)``.
    """
    xyz = np.asarray(coords, dtype=float)
    refs = [int(x) for x in ref_ids]
    if xyz.shape != (len(refs), 3):
        raise ValueError("Coordinate/Ref-ID shape mismatch.")
    metal = np.asarray(metal, dtype=float)

    def _geom(arr):
        vec = np.asarray(arr, dtype=float) - metal
        r = np.linalg.norm(vec, axis=-1)
        r_safe = np.where(r == 0.0, np.inf, r)
        x = vec[..., 0]
        y = vec[..., 1]
        z = vec[..., 2]
        theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
        phi = np.arctan2(y, x)
        gax = (3.0 * np.cos(theta) ** 2 - 1.0) / (r_safe ** 3)
        grh = 1.5 * np.sin(theta) ** 2 * np.cos(2.0 * phi) / (r_safe ** 3)
        return gax, grh

    gax, grh = _geom(xyz)
    gax = np.asarray(gax, dtype=float).copy()
    grh = np.asarray(grh, dtype=float).copy()

    active = enabled_groups(groups)
    if not active:
        return gax, grh

    overlaps = validate_independent_groups(active)
    if overlaps:
        a, b, common = overlaps[0]
        raise ValueError(
            f"Torsional groups {a} and {b} overlap at Ref IDs {common}."
        )

    id2idx = {rid: i for i, rid in enumerate(refs)}
    for group in active:
        samples = sample_group_coordinates(xyz, refs, group)
        weights = uniform_weights(group)
        sgax, sgrh = _geom(samples)
        avg_gax = np.tensordot(weights, sgax, axes=(0, 0))
        avg_grh = np.tensordot(weights, sgrh, axes=(0, 0))
        for rid in group.get("rotating_atoms", ()):
            rid = int(rid)
            idx = id2idx.get(rid)
            if idx is not None:
                gax[idx] = avg_gax[idx]
                grh[idx] = avg_grh[idx]

    return gax, grh
