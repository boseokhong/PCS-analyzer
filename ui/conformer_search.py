# ui/conformer_search.py

"""
=========================
Conformational search by dihedral-angle optimization to fit experimental PCS data.

Problem statement
-----------------
Given a molecular structure (XYZ) with a paramagnetic metal centre, find the set
of dihedral angles around rotatable aliphatic bonds that minimises the RMSD
between predicted and experimental PCS values.

Key constraint: user-defined *fixed atoms* must not change their Cartesian
position during the search.  Any rotatable bond whose downstream subtree
contains a fixed atom is automatically excluded — the bond is shown as
'locked' in the UI and cannot be selected for rotation.

This allows the user to, e.g., pin the donor atoms of a tetrapodal ligand
(N in z-direction, known from the crystal structure) while freely optimising
the conformations of the pendant arms.

Workflow
--------
1.  Load XYZ → build connectivity graph.
2.  Detect all rotatable aliphatic bonds.
3.  Mark fixed atoms (atoms whose position must not change).
4.  For each rotatable bond: if its rotating subtree contains a fixed atom,
    the bond is 'locked' and excluded from the parameter vector.
5.  Load experimental PCS values (CSV: Ref, δ_exp).
6.  Two-stage optimization:
        Stage 1 – Differential Evolution (global)
        Stage 2 – L-BFGS-B (local refinement)
    Clash penalty keeps the structure physically valid.
7.  Save best structure as XYZ; show per-atom PCS report.

Usage
-----
GUI:  python -m utils.conformer_search          (or via PCS Suite button)
CLI:  python -m utils.conformer_search --help
"""

from __future__ import annotations

import os
import sys
import math
import itertools
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist

from logic.chem_constants import VDW_RADII, covalent_radii, METAL_ELEMENTS

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL / CHEMICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Fraction of vdW-sum below which two atoms are considered clashing
CLASH_FACTOR   = 0.65
CLASH_PENALTY_K = 500.0   # ppm-equivalent penalty per clashing pair

# ─────────────────────────────────────────────────────────────────────────────
# MOLECULAR GRAPH
# ─────────────────────────────────────────────────────────────────────────────

class Molecule:
    """
    Lightweight molecular graph built from element symbols and Cartesian coords.

    Attributes
    ----------
    n_atoms  : int
    elements : list[str]
    coords   : np.ndarray  (n_atoms, 3)
    bonds    : list[(i,j)] with i < j
    adj      : list[list[int]]  adjacency list
    """

    def __init__(self, elements: List[str], coords: np.ndarray):
        self.n_atoms  = len(elements)
        self.elements = elements
        self.coords   = coords.copy()
        self.bonds:   List[Tuple[int, int]] = []
        self.adj:     List[List[int]]       = [[] for _ in range(self.n_atoms)]
        self._build_bonds()

    def _build_bonds(self):
        D = cdist(self.coords, self.coords)
        for i in range(self.n_atoms):
            ri = covalent_radii.get(self.elements[i], covalent_radii["default"])
            for j in range(i + 1, self.n_atoms):
                rj = covalent_radii.get(self.elements[j], covalent_radii["default"])
                if D[i, j] < (ri + rj) * 1.30:
                    self.bonds.append((i, j))
                    self.adj[i].append(j)
                    self.adj[j].append(i)

    def is_in_ring(self, i: int, j: int,
                    metal_indices: Optional[Set[int]] = None) -> bool:
        """
        True if bond i–j is part of a ring.

        Metal centres are excluded from traversal: they create pseudo-rings
        via coordination bonds, not covalent rings.  Pass metal_indices to
        ensure metal-mediated paths are ignored.

        Bug fix vs naive implementation: visited starts at {j} (not {i})
        so that a path arriving back at i is correctly detected.
        """
        block   = set(metal_indices) if metal_indices else set()
        stack   = [n for n in self.adj[j] if n != i and n not in block]
        visited = {j} | block
        while stack:
            cur = stack.pop()
            if cur == i:
                return True
            if cur not in visited:
                visited.add(cur)
                stack.extend(n for n in self.adj[cur] if n not in visited)
        return False

    def subtree_of(self, anchor: int, exclude: int) -> List[int]:
        """All atoms reachable from *anchor* without crossing anchor←exclude."""
        visited = {exclude}
        stack   = [anchor]
        result  = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            result.append(cur)
            stack.extend(n for n in self.adj[cur] if n not in visited)
        return result

    def subtree_with_barriers(self, anchor: int, exclude: int,
                               barriers: Set[int]) -> List[int]:
        """
        Atoms reachable from *anchor* (not crossing *exclude*), treating
        *barriers* as hard stops: a barrier atom is included in the result
        but traversal does not continue beyond it.

        This correctly limits the rotating subtree when fixed atoms or metal
        centres act as anchors rather than as free chain atoms.
        """
        visited = {exclude}
        stack   = [anchor]
        result  = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            result.append(cur)
            # Stop traversal at barrier atoms (but include the barrier itself)
            if cur in barriers and cur != anchor:
                continue
            stack.extend(n for n in self.adj[cur] if n not in visited)
        return result

    def find_rotatable_bonds(self,
                              metal_idx: Optional[int] = None,
                              ) -> List[Tuple[int, int]]:
        """
        Return all bonds that are candidates for dihedral rotation.

        A bond is rotatable when:
        - Connects two non-hydrogen heavy atoms.
        - Not part of a ring (metal-mediated paths excluded from ring check).
        - At least one neighbour on each side.
        - Neither atom is the metal centre.

        Note: fixed_atoms are NOT filtered here — that is done separately
        in classify_bonds_by_fixed() so the GUI can show locked bonds.
        """
        # Collect all metal atom indices for ring-detection exclusion
        metal_indices = {i for i, e in enumerate(self.elements) if e in METAL_ELEMENTS}

        rotatable = []
        for i, j in self.bonds:
            ei, ej = self.elements[i], self.elements[j]
            if ei == "H" or ej == "H":
                continue
            if metal_idx is not None and (i == metal_idx or j == metal_idx):
                continue
            if self.is_in_ring(i, j, metal_indices=metal_indices):
                continue
            if not [n for n in self.adj[i] if n != j]:
                continue
            if not [n for n in self.adj[j] if n != i]:
                continue
            rotatable.append((i, j))
        return rotatable

# ─────────────────────────────────────────────────────────────────────────────
# FIXED-ATOM CONSTRAINT
# ─────────────────────────────────────────────────────────────────────────────

def classify_bonds_by_fixed(mol: Molecule,
                             rotatable_candidates: List[Tuple[int, int]],
                             fixed_atoms: Set[int],
                             metal_idx: Optional[int] = None,
                             anchor_mode: bool = True
                             ) -> Tuple[List[Tuple[int, int]],
                                        List[Tuple[int, int]]]:
    """
    Split candidate rotatable bonds into *free* and *locked* sets.

    Rotating side (arm side)
    ------------------------
    The arm side is the subtree that does NOT contain the metal centre.
    If neither or both sides contain the metal, the smaller subtree is used.

    Two locking modes
    -----------------
    anchor_mode=True  (default, recommended for coordination chemistry)
        A bond (i→j) is LOCKED only if atoms *deeper than j* in the rotating
        subtree are fixed.  The directly-bonded arm atom j itself is treated
        as the *new pivot* for the arm beyond it: even if j is marked as
        fixed, rotating around (i→j) keeps j's position constrained via the
        fixed-atom penalty in the optimiser — it will not drift far.

        This matches the chemist's mental model for a tetrapodal ligand:
        "Fix Ca as my coordination carbon (anchor). Ca–Cb bond can still
        rotate, with Ca as the dihedral pivot. Cb is the start of the
        freely rotating arm."

    anchor_mode=False  (strict / physically exact)
        A bond is LOCKED if ANY fixed atom appears anywhere in the rotating
        subtree, including j itself.  Use this when atoms must be truly
        immovable (e.g., for symmetry-constrained heavy atoms).

    Parameters
    ----------
    mol                 : Molecule
    rotatable_candidates: bonds to classify
    fixed_atoms         : 0-based atom indices that should not move
    metal_idx           : 0-based index of the metal centre
    anchor_mode         : see above (default True)

    Returns
    -------
    free_bonds   : bonds free to optimise
    locked_bonds : bonds excluded from optimization
    """
    if not fixed_atoms:
        return list(rotatable_candidates), []

    # Metal centres are always barriers: they form coordination bonds, not
    # covalent chains, so subtree traversal must stop at them.
    metal_indices = {i for i, e in enumerate(mol.elements) if e in METAL_ELEMENTS}
    # Full barrier set: user-fixed atoms + metal centres
    barriers_full = set(fixed_atoms) | metal_indices

    free, locked = [], []
    for i, j in rotatable_candidates:
        # Use barrier-aware subtree traversal: stops at fixed atoms and metals.
        # This prevents traversal from "leaking" through coordination bonds to
        # the other side of the metal, which would make every subtree contain
        # fixed atoms and lock every bond.
        subtree_j = set(mol.subtree_with_barriers(j, exclude=i, barriers=barriers_full))
        subtree_i = set(mol.subtree_with_barriers(i, exclude=j, barriers=barriers_full))

        # Arm side = the side that does NOT contain a metal centre
        metal_in_j = bool(metal_indices & subtree_j)
        metal_in_i = bool(metal_indices & subtree_i)
        if metal_in_i and not metal_in_j:
            rotating = subtree_j; arm_anchor = j
        elif metal_in_j and not metal_in_i:
            rotating = subtree_i; arm_anchor = i
        else:
            if len(subtree_j) <= len(subtree_i):
                rotating = subtree_j; arm_anchor = j
            else:
                rotating = subtree_i; arm_anchor = i

        # Check for user-fixed atoms (NOT metals — those are already barriers)
        # in the rotating subtree.
        user_fixed = fixed_atoms - metal_indices
        if anchor_mode:
            # arm_anchor acts as the new pivot: only atoms DEEPER than it lock the bond
            check_set = rotating - {arm_anchor}
        else:
            check_set = rotating

        if check_set & user_fixed:
            locked.append((i, j))
        else:
            free.append((i, j))

    return free, locked

# ─────────────────────────────────────────────────────────────────────────────
# DIHEDRAL ROTATION
# ─────────────────────────────────────────────────────────────────────────────

def dihedral_angle(p0: np.ndarray, p1: np.ndarray,
                   p2: np.ndarray, p3: np.ndarray) -> float:
    """Dihedral angle p0–p1–p2–p3 in degrees (IUPAC sign convention)."""
    b1 = p1 - p0; b2 = p2 - p1; b3 = p3 - p2
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    n1n = np.linalg.norm(n1); n2n = np.linalg.norm(n2)
    if n1n < 1e-9 or n2n < 1e-9:
        return 0.0
    n1 /= n1n; n2 /= n2n
    m1  = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))
    return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))


def rotate_dihedral(coords: np.ndarray,
                    mol:    Molecule,
                    bond:   Tuple[int, int],
                    angle_deg: float,
                    metal_idx: Optional[int] = None) -> np.ndarray:
    """
    Set the dihedral around *bond* (i, j) to *angle_deg*.

    The rotating side is determined as follows:
    - If *metal_idx* is given: rotate the side that does NOT contain
      the metal centre (the "arm" side).  This matches the physics of
      coordination-complex arm rotation.
    - Otherwise: rotate the smaller subtree.

    Bond lengths, bond angles, and ring geometries are preserved exactly.
    Returns a new coordinate array.
    """
    i, j = bond
    coords = coords.copy()

    axis = coords[j] - coords[i]
    alen = np.linalg.norm(axis)
    if alen < 1e-9:
        return coords
    axis /= alen

    nb_i = [n for n in mol.adj[i] if n != j]
    nb_j = [n for n in mol.adj[j] if n != i]
    if not nb_i or not nb_j:
        return coords

    # Use barrier-aware subtrees so rotation never moves across metal bonds
    _metal_set2 = {k for k, e in enumerate(mol.elements) if e in METAL_ELEMENTS}
    _barriers2  = _metal_set2 | ({metal_idx} if metal_idx is not None else set())
    subtree_j = mol.subtree_with_barriers(j, i, _barriers2)
    subtree_i = mol.subtree_with_barriers(i, j, _barriers2)

    # Determine rotating side
    if metal_idx is not None:
        metal_in_j = metal_idx in subtree_j
        metal_in_i = metal_idx in subtree_i
        if metal_in_i and not metal_in_j:
            # j-side is the arm → rotate j-side (standard orientation i→j)
            rotate_idx = subtree_j
            ref_fixed  = coords[nb_i[0]]
            ref_moving = coords[nb_j[0]]
            pivot      = coords[i]
            # axis already points i→j
        elif metal_in_j and not metal_in_i:
            # i-side is the arm → rotate i-side, flip axis to j→i
            rotate_idx = subtree_i
            ref_fixed  = coords[nb_j[0]]
            ref_moving = coords[nb_i[0]]
            pivot      = coords[j]
            axis       = -axis
        else:
            # Metal on neither or both → fall back to smaller-subtree
            if len(subtree_j) <= len(subtree_i):
                rotate_idx = subtree_j
                ref_fixed  = coords[nb_i[0]]
                ref_moving = coords[nb_j[0]]
                pivot      = coords[i]
            else:
                rotate_idx = subtree_i
                ref_fixed  = coords[nb_j[0]]
                ref_moving = coords[nb_i[0]]
                pivot      = coords[j]
                axis       = -axis
    else:
        # No metal hint → smaller subtree rotates
        if len(subtree_j) <= len(subtree_i):
            rotate_idx = subtree_j
            ref_fixed  = coords[nb_i[0]]
            ref_moving = coords[nb_j[0]]
            pivot      = coords[i]
        else:
            rotate_idx = subtree_i
            ref_fixed  = coords[nb_j[0]]
            ref_moving = coords[nb_i[0]]
            pivot      = coords[j]
            axis       = -axis

    # Measure current dihedral and compute rotation delta
    current = dihedral_angle(ref_fixed, pivot, pivot + axis, ref_moving)
    delta   = angle_deg - current
    th      = math.radians(delta)
    cos_th  = math.cos(th)
    sin_th  = math.sin(th)

    for k in rotate_idx:
        v         = coords[k] - pivot
        coords[k] = (pivot
                     + cos_th * v
                     + sin_th * np.cross(axis, v)
                     + (1 - cos_th) * np.dot(axis, v) * axis)
    return coords

# ─────────────────────────────────────────────────────────────────────────────
# PCS CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_pcs(coords: np.ndarray,
             metal:  np.ndarray,
             dchi_ax: float,
             dchi_rh: float,
             euler_zyz: Tuple[float, float, float]) -> np.ndarray:
    """
    PCS [ppm] for all rows in *coords* given tensor parameters.
    δ_PCS = (Δχ_ax·G_ax + Δχ_rh·G_rh) × 1e4 / (12π)
    Δχ in 10⁻³² m³.  euler_zyz in radians (ZYZ convention).
    """
    from scipy.spatial.transform import Rotation as _Rot
    rot_mat = _Rot.from_euler("ZYZ", euler_zyz).as_matrix()
    delta   = coords - metal
    dt      = (rot_mat @ delta.T).T
    r       = np.linalg.norm(dt, axis=1)
    r_safe  = np.where(r > 1e-9, r, 1e-9)
    cos2    = (dt[:, 2] / r_safe) ** 2
    sin2    = 1.0 - cos2
    phi     = np.arctan2(dt[:, 1], dt[:, 0])
    Gax     = (3 * cos2 - 1) / r_safe ** 3
    Grh     = 1.5 * sin2 * np.cos(2 * phi) / r_safe ** 3
    return (dchi_ax * Gax + dchi_rh * Grh) * 1e4 / (12.0 * np.pi)

# ─────────────────────────────────────────────────────────────────────────────
# CLASH DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _precompute_bonded_13(mol: Molecule) -> set:
    """Precompute set of (i,j) pairs that are bonded or share a common neighbour (1-3)."""
    pairs = set()
    for i, j in mol.bonds:
        pairs.add((min(i,j), max(i,j)))
    for k in range(mol.n_atoms):
        for a, b in itertools.combinations(mol.adj[k], 2):
            pairs.add((min(a,b), max(a,b)))
    return pairs


def count_clashes(coords: np.ndarray, elements: List[str],
                  bonded_pairs: set) -> int:
    """Number of non-bonded atom pairs closer than CLASH_FACTOR × vdW sum (vectorised)."""
    n = len(elements)
    radii = np.array([VDW_RADII.get(e, VDW_RADII["default"]) for e in elements])
    D     = cdist(coords, coords)
    R_sum = radii[:, None] + radii[None, :]
    clash = D < CLASH_FACTOR * R_sum

    # Upper triangle, skip 1-2
    mask = np.zeros((n, n), dtype=bool)
    idx_i, idx_j = np.triu_indices(n, k=2)
    mask[idx_i, idx_j] = True
    if bonded_pairs:
        bi = np.array([p[0] for p in bonded_pairs], dtype=int)
        bj = np.array([p[1] for p in bonded_pairs], dtype=int)
        mask[bi, bj] = False
        mask[bj, bi] = False
    return int(np.sum(clash & mask))


def clash_penalty(coords: np.ndarray, elements: List[str],
                  bonded_pairs: set) -> float:
    """
    Smooth quadratic clash penalty (ppm² units) – fully vectorised with NumPy.
    ~100× faster than the pure-Python loop for molecules with ≥20 atoms.
    """
    n = len(elements)
    radii = np.array([VDW_RADII.get(e, VDW_RADII["default"]) for e in elements])
    D = cdist(coords, coords)

    # Vectorised threshold matrix and overlap
    R_sum = radii[:, None] + radii[None, :]          # (N, N) sum of vdW radii
    thr   = CLASH_FACTOR * R_sum                      # (N, N) clash threshold
    overlap = np.maximum(0.0, thr - D)               # positive where clashing

    # Upper triangle, skip i==j and i==j+1 (too close anyway)
    mask = np.zeros((n, n), dtype=bool)
    idx_i, idx_j = np.triu_indices(n, k=2)
    mask[idx_i, idx_j] = True

    # Remove bonded and 1-3 pairs
    if bonded_pairs:
        bi = np.array([p[0] for p in bonded_pairs], dtype=int)
        bj = np.array([p[1] for p in bonded_pairs], dtype=int)
        mask[bi, bj] = False
        mask[bj, bi] = False

    penalty = float(np.sum((CLASH_PENALTY_K * overlap[mask]) ** 2))
    return penalty


# ─────────────────────────────────────────────────────────────────────────────
# FIXED-ATOM PENALTY (soft constraint — used as a backup sanity check)
# ─────────────────────────────────────────────────────────────────────────────

FIXED_PENALTY_K = 1e6   # very large ppm² per Å² displacement


def fixed_atom_penalty(coords: np.ndarray,
                        ref_coords: np.ndarray,
                        fixed_indices: List[int]) -> float:
    """
    Extra penalty if any fixed atom has moved from its reference position.

    This is a soft backup: the bond-classification filter (classify_bonds_by_fixed)
    is the *primary* constraint and should prevent fixed atoms from moving.
    This penalty catches edge cases where floating-point drift occurs.
    """
    if not fixed_indices:
        return 0.0
    delta = coords[fixed_indices] - ref_coords[fixed_indices]
    return FIXED_PENALTY_K * float(np.sum(delta ** 2))

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────

class ConformerOptimiser:
    """
    Optimise dihedral angles of free bonds to minimise PCS RMSD + clash penalty.

    Fixed atoms are guaranteed not to move because any bond whose rotating
    subtree contains a fixed atom has already been removed from *rotatable*
    by classify_bonds_by_fixed() before this class is instantiated.
    A soft fixed-atom penalty is applied as a numerical safety net.

    Parameters
    ----------
    mol           : Molecule
    rotatable     : bonds that are free to rotate (no fixed atom in subtree)
    obs_atoms     : 0-based atom indices with experimental PCS
    obs_pcs       : experimental PCS values [ppm]
    metal         : metal position [Å]
    dchi_ax/rh    : Δχ values [×10⁻³² m³]
    euler_zyz     : tensor orientation (α,β,γ) [rad], ZYZ convention
    metal_idx     : atom index of the metal centre
    fixed_atoms   : set of atom indices whose positions must not change
    progress_cb   : optional callable(str) for status messages
    """

    def __init__(self,
                 mol:          Molecule,
                 rotatable:    List[Tuple[int, int]],
                 obs_atoms:    List[int],
                 obs_pcs:      np.ndarray,
                 metal:        np.ndarray,
                 dchi_ax:      float,
                 dchi_rh:      float,
                 euler_zyz:    Tuple[float, float, float],
                 metal_idx:    Optional[int] = None,
                 fixed_atoms:  Optional[Set[int]] = None,
                 progress_cb   = None):

        self.mol         = mol
        self.rotatable   = list(rotatable)
        self.obs_atoms   = np.array(obs_atoms, int)
        self.obs_pcs     = np.asarray(obs_pcs, float)
        self.metal       = np.asarray(metal, float)
        self.dchi_ax     = float(dchi_ax)
        self.dchi_rh     = float(dchi_rh)
        self.euler_zyz   = tuple(euler_zyz)
        self.metal_idx   = metal_idx
        self.fixed_atoms = list(fixed_atoms) if fixed_atoms else []
        self.progress_cb = progress_cb

        self._bonded13   = _precompute_bonded_13(mol)
        self._ref_coords = mol.coords.copy()

        # ── Pre-cache per-bond rotation data ──────────────────────────────
        # rotate_dihedral calls subtree_with_barriers (graph traversal) on
        # every cost evaluation. With 16 bonds × 100k DE calls that is the
        # dominant cost. Cache it once here.
        _metal_set_c = {k for k, e in enumerate(mol.elements) if e in METAL_ELEMENTS}
        _barriers_c  = _metal_set_c | (set(fixed_atoms) if fixed_atoms else set())

        self._rot_cache = []
        for bi, bj in self.rotatable:
            stj = mol.subtree_with_barriers(bj, bi, _barriers_c)
            sti = mol.subtree_with_barriers(bi, bj, _barriers_c)
            metal_in_j = bool(_metal_set_c & set(stj))
            metal_in_i = bool(_metal_set_c & set(sti))
            if metal_in_i and not metal_in_j:
                rot_idx = np.array(stj, int); pivot_idx = bi; axis_sign = 1
            elif metal_in_j and not metal_in_i:
                rot_idx = np.array(sti, int); pivot_idx = bj; axis_sign = -1
            else:
                if len(stj) <= len(sti):
                    rot_idx = np.array(stj, int); pivot_idx = bi; axis_sign = 1
                else:
                    rot_idx = np.array(sti, int); pivot_idx = bj; axis_sign = -1
            nb_i = [n for n in mol.adj[bi] if n != bj]
            nb_j = [n for n in mol.adj[bj] if n != bi]
            ref_fixed  = nb_i[0] if axis_sign ==  1 else nb_j[0]
            ref_moving = nb_j[0] if axis_sign ==  1 else nb_i[0]
            self._rot_cache.append(
                (pivot_idx, axis_sign, rot_idx, ref_fixed, ref_moving, bi, bj))

        self._x0 = self._read_current_angles(mol.coords)

    def _read_current_angles(self, coords: np.ndarray) -> np.ndarray:
        angles = []
        for i, j in self.rotatable:
            nb_i = [n for n in self.mol.adj[i] if n != j]
            nb_j = [n for n in self.mol.adj[j] if n != i]
            if nb_i and nb_j:
                ang = dihedral_angle(coords[nb_i[0]], coords[i],
                                     coords[j], coords[nb_j[0]])
            else:
                ang = 0.0
            angles.append(ang)
        return np.array(angles, float)

    def _apply_angles(self, angles: np.ndarray) -> np.ndarray:
        """Apply dihedral angles using pre-cached subtree data (fast path)."""
        coords = self.mol.coords.copy()
        for angle, (pivot_idx, axis_sign, rot_idx,
                    ref_fixed, ref_moving, bi, bj) in zip(angles, self._rot_cache):
            pivot    = coords[pivot_idx]
            raw_axis = coords[bj] - coords[bi]
            al       = float(np.linalg.norm(raw_axis))
            if al < 1e-9:
                continue
            axis = (raw_axis / al) * axis_sign

            current = dihedral_angle(coords[ref_fixed], pivot,
                                     pivot + axis, coords[ref_moving])
            delta  = angle - current
            th     = math.radians(delta)
            cos_th = math.cos(th)
            sin_th = math.sin(th)

            vecs   = coords[rot_idx] - pivot          # (M, 3)
            dot_v  = (vecs * axis).sum(axis=1, keepdims=True)
            cross_v = np.cross(axis, vecs)            # (M, 3)
            coords[rot_idx] = (pivot
                               + cos_th * vecs
                               + sin_th * cross_v
                               + (1 - cos_th) * dot_v * axis)
        return coords

    def _cost(self, x: np.ndarray) -> float:
        coords   = self._apply_angles(x)
        pcs_pred = calc_pcs(coords[self.obs_atoms], self.metal,
                            self.dchi_ax, self.dchi_rh, self.euler_zyz)
        resid    = pcs_pred - self.obs_pcs
        cost     = float(np.dot(resid, resid))
        cost    += clash_penalty(coords, self.mol.elements, self._bonded13)
        cost    += fixed_atom_penalty(coords, self._ref_coords, self.fixed_atoms)
        return cost

    def run(self,
            use_global: bool = True,
            de_maxiter: int = 600,
            de_popsize: int = 12,
            selected_bonds=None,
            locked_bonds=None) -> dict:
        """
        Run two-stage optimization and return a result dict.

        Result keys
        -----------
        coords_opt   : np.ndarray (n_atoms, 3) – optimised coordinates
        angles_opt   : dihedral angles at optimum [deg]
        angles_init  : dihedral angles of input structure [deg]
        angle_deltas : angles_opt − angles_init (wrapped to [−180, 180])
        rmsd         : RMSD of PCS fit [ppm]
        r2           : R² of PCS fit
        q_factor     : Q-factor of PCS fit
        per_atom     : list of (atom_idx, element, pcs_exp, pcs_pred, residual)
        n_clashes    : steric clashes in optimised structure
        fixed_displacement : max Cartesian displacement of any fixed atom [Å]
        """
        n_bonds = len(self.rotatable)
        if n_bonds == 0:
            raise RuntimeError(
                "No free rotatable bonds.\n\n"
                "All candidate bonds are locked because a fixed atom appears\n"
                "in their rotating subtree.  Either unfix some atoms or add\n"
                "more rotatable bonds.")

        bounds = [(-180.0, 180.0)] * n_bonds
        x0     = self._x0.copy()

        if self.progress_cb:
            self.progress_cb(
                f"Optimising {n_bonds} dihedral(s) "
                f"[{len(self.fixed_atoms)} atom(s) fixed, "
                f"{len(self.obs_atoms)} PCS obs.]…")

        # Stage 1: Differential Evolution
        if use_global:
            if self.progress_cb:
                self.progress_cb(
                    f"Stage 1 – Differential Evolution "
                    f"({de_maxiter} iter × pop {de_popsize})…")
            de_result = differential_evolution(
                self._cost, bounds,
                seed=42, maxiter=de_maxiter, popsize=de_popsize,
                tol=1e-8, mutation=(0.5, 1.5), recombination=0.7,
                updating="deferred", workers=1, polish=False, x0=x0,
            )
            x0 = de_result.x
            if self.progress_cb:
                self.progress_cb(
                    f"  DE done  cost={de_result.fun:.4f}  "
                    f"success={de_result.success}")

        # Stage 2: L-BFGS-B local refinement
        if self.progress_cb:
            self.progress_cb("Stage 2 – L-BFGS-B local refinement…")

        N_CANDIDATES = 5 # candidates number
        # Collect top-N starting points from DE population.
        # de_result.population / de_result.population_energies are available
        # when updating="deferred"; fall back to [x0] if not present.
        pop = getattr(de_result, "population", None) if use_global else None
        engs = getattr(de_result, "population_energies", None) if use_global else None
        if pop is not None and engs is not None:
            order = np.argsort(engs)[:N_CANDIDATES]
            starts = [pop[i] for i in order]
        else:
            starts = [x0]

        # Stage 2: local-refine each starting point independently.
        candidates_raw = []
        for s in starts:
            r = minimize(self._cost, s, method="L-BFGS-B", bounds=bounds,
                         options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10})
            candidates_raw.append((r.fun, r.x))

        # Sort by final cost, deduplicate near-identical solutions.
        candidates_raw.sort(key=lambda t: t[0])
        candidates_raw = _deduplicate_candidates(candidates_raw, tol_deg=5.0)
        candidates_raw = candidates_raw[:N_CANDIDATES]

        self.candidates = [self._make_result(x, selected_bonds, locked_bonds)
                           for _, x in candidates_raw]
        # Primary return = best candidate (backward-compatible)
        return self.candidates[0]

    def _make_result(self, x_opt, selected_bonds=None, locked_bonds=None) -> dict:
        """Build a full result dict for one candidate angle vector."""
        coords_opt = self._apply_angles(x_opt)
        pcs_pred   = calc_pcs(coords_opt[self.obs_atoms], self.metal,
                              self.dchi_ax, self.dchi_rh, self.euler_zyz)
        resid_v    = pcs_pred - self.obs_pcs
        rmsd       = float(np.sqrt(np.mean(resid_v ** 2)))
        ss_res     = float(np.sum(resid_v ** 2))
        ss_tot     = float(np.sum((self.obs_pcs - self.obs_pcs.mean()) ** 2))
        r2         = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else float("nan")
        q_factor   = (float(np.sqrt(ss_res / np.sum(self.obs_pcs ** 2)))
                      if np.any(self.obs_pcs != 0) else float("nan"))
        angle_deltas = []
        for d in (x_opt - self._x0):
            while d >  180: d -= 360
            while d < -180: d += 360
            angle_deltas.append(d)
        per_atom = [(int(self.obs_atoms[k]),
                     self.mol.elements[int(self.obs_atoms[k])],
                     float(self.obs_pcs[k]), float(pcs_pred[k]), float(resid_v[k]))
                    for k in range(len(self.obs_atoms))]
        if self.fixed_atoms:
            max_fd = float(np.linalg.norm(
                coords_opt[self.fixed_atoms] - self._ref_coords[self.fixed_atoms],
                axis=1).max())
        else:
            max_fd = 0.0
        return dict(
            coords_opt        = coords_opt,
            angles_opt        = x_opt,
            angles_init       = self._x0,
            angle_deltas      = np.array(angle_deltas),
            rmsd              = rmsd,
            r2                = r2,
            q_factor          = q_factor,
            per_atom          = per_atom,
            n_clashes         = count_clashes(coords_opt, self.mol.elements, self._bonded13),
            fixed_displacement= max_fd,
            selected_bonds    = selected_bonds or [],
            locked_bonds      = locked_bonds or [],
        )


# ─────────────────────────────────────────────────────────────────────────────
# FILE I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_xyz(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    elements, coords = [], []
    for line in lines[2:2+n]:
        p = line.split()
        if len(p) >= 4:
            elements.append(p[0])
            coords.append([float(p[1]), float(p[2]), float(p[3])])
    if len(elements) != n:
        raise ValueError(f"XYZ atom count mismatch: expected {n}, got {len(elements)}")
    return elements, np.array(coords, float)


def save_xyz(path: str, elements: List[str], coords: np.ndarray,
             comment: str = "Optimised conformer – PCS Suite"):
    with open(path, "w") as f:
        f.write(f"{len(elements)}\n{comment}\n")
        for el, (x, y, z) in zip(elements, coords):
            f.write(f"{el:<4s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")


def load_pcs_csv(path: str, elements: List[str],
                 zero_indexed: bool = False) -> Tuple[List[int], List[float]]:
    """
    Load experimental PCS from a CSV file (columns: Ref, [Element,] δ_exp).
    Returns (0-based atom indices, pcs values).
    """
    import csv
    atom_indices, pcs_values = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        try:
            dialect = csv.Sniffer().sniff(f.read(2048), delimiters=",\t; ")
            f.seek(0)
        except csv.Error:
            f.seek(0)
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        for row in reader:
            row = [c.strip() for c in row if c.strip()]
            if not row or row[0].lower() in ("ref","id","#","atom"):
                continue
            try:
                ref = int(row[0])
                val = float(row[2]) if len(row) >= 3 else float(row[1])
            except (ValueError, IndexError):
                continue
            idx = ref if zero_indexed else ref - 1
            if 0 <= idx < len(elements):
                atom_indices.append(idx)
                pcs_values.append(val)
    return atom_indices, pcs_values

def _deduplicate_candidates(
        candidates: list,          # [(cost, x_vec), ...]
        tol_deg: float = 5.0,
) -> list:
    """Remove angle vectors within tol_deg (L-inf) of an already-kept solution."""
    kept = []
    for cost, x in candidates:
        if all(np.max(np.abs(x - kx)) > tol_deg for _, kx in kept):
            kept.append((cost, x))
    return kept

# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def format_report(result: dict, rotatable_bonds, mol, locked_bonds=None,
                  rank: int = 1, n_total: int = 1) -> str:
    rank_str = f"  [Candidate #{rank} / {n_total}]" if n_total > 1 else ""
    lines = [
        "═" * 64,
        f"  CONFORMATIONAL SEARCH – PCS FIT REPORT{rank_str}",
        "═" * 64, "",
        f"  Free rotatable bonds     : {len(rotatable_bonds)}",
        f"  Locked bonds (fixed-atom): {len(locked_bonds) if locked_bonds else 0}",
        f"  Atoms with δ_exp         : {len(result['per_atom'])}",
        f"  Steric clashes (final)   : {result['n_clashes']}",
    ]
    fd = result.get('fixed_displacement', 0.0)
    lines.append(f"  Max fixed-atom drift     : {fd:.4f} Å  "
                 f"{'✓ OK' if fd < 0.01 else '⚠ WARNING'}")
    lines += [
        "", "  FIT QUALITY", "  " + "─" * 44,
        f"  RMSD     = {result['rmsd']:.4f} ppm",
        (f"  R²       = {result['r2']:.4f}"
         if not math.isnan(result['r2']) else "  R²       = —"),
        (f"  Q-factor = {result['q_factor']:.4f}"
         if not math.isnan(result['q_factor']) else "  Q-factor = —"),
        "", "  DIHEDRAL ANGLES  [degrees]", "  " + "─" * 44,
        f"  {'Bond':<12} {'Atom-i':<6} {'Atom-j':<6} "
        f"{'Initial':>9} {'Optimised':>10} {'Δ':>8}",
    ]
    for k, (i, j) in enumerate(rotatable_bonds):
        ang_i = result['angles_init'][k]
        ang_o = result['angles_opt'][k]
        d     = result['angle_deltas'][k]
        lines.append(f"  {i+1}–{j+1:<10} "
                     f"{mol.elements[i]:<6} {mol.elements[j]:<6} "
                     f"{ang_i:>9.2f} {ang_o:>10.2f} {d:>8.2f}")
    lines += [
        "", "  PER-ATOM PCS FIT  [ppm]", "  " + "─" * 44,
        f"  {'Ref':>6} {'Atom':<4} {'δ_exp':>9} {'δ_pred':>9} {'Resid':>9}",
    ]
    for idx, el, exp, pred, res in result['per_atom']:
        lines.append(f"  {idx+1:>6} {el:<4} {exp:>+9.3f} {pred:>+9.3f} {res:>+9.3f}")
    lines += ["", "═" * 64]
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class ConformerSearchGUI:
    """
    Four-tab GUI:
      ①  Setup           – files, metal, tensor, optimization options
      ②  Fixed atoms      – checkboxes to pin atoms in place
      ③  Rotatable bonds  – per-bond enable/disable with lock indicator
      ④  Results          – report, export, correlation plot
    """

    def __init__(
        self,
        master: Optional[tk.Misc] = None,
        *,
        on_preview_result=None,
        initial_data: Optional[dict] = None,
        embed_in: Optional[tk.Misc] = None,
    ):
        if embed_in is not None:
            self.root = embed_in
            self._standalone = False
            self._embedded = True
        elif master is None:
            self.root = tk.Tk()
            self.root.title("Conformational Search – PCS Dihedral Optimiser")
            self.root.geometry("960x780")
            self._standalone = True
            self._embedded = False
        else:
            self.root = tk.Toplevel(master)
            self.root.title("Conformational Search – PCS Dihedral Optimiser")
            self.root.geometry("960x780")
            self._standalone = False
            self._embedded = False

        self.on_preview_result = on_preview_result
        self.initial_data = initial_data or {}

        self.mol = None
        self.rotatable_all = []
        self.bond_vars = []
        self.bond_locked = []
        self.atom_fix_vars = []
        self.result = None
        self.metal_idx = None
        self._is_running = False
        self._progress_mode = tk.StringVar(value="idle")
        self._opt_level_var = tk.StringVar(value="Balanced")
        self._run_btn = None
        self._progress_bar = None
        self._setup_summary_var = tk.StringVar(value="Search setup: 0 free bonds | 0 fixed atoms | 0 PCS points")
        self._applying_preset = False

        self._build_ui()
        self._prefill_from_initial_data()
        if getattr(self, '_embedded', False) and self.initial_data.get("xyz_path"):
            self.root.after(100, self._auto_detect_if_ready)

    def _auto_detect_if_ready(self):
        """embed mode autodetect - bond detection"""
        xyz_path = self._xyz_var.get().strip()
        if not xyz_path or not os.path.isfile(xyz_path):
            return
        try:
            elements, coords = load_xyz(xyz_path)
            self.mol = Molecule(elements, coords)
            self._status_var.set(
                f"Loaded {self.mol.n_atoms} atoms, {len(self.mol.bonds)} bonds.")
        except Exception:
            return
        self._detect_all()

    def _prefill_from_initial_data(self):
        """
        Fill the UI from externally supplied initial_data.
        Expected keys (all optional):
            xyz_path, pcs_path,
            metal_idx_1b,
            metal_xyz=(x,y,z),
            dchi_ax, dchi_rh,
            euler_deg=(alpha,beta,gamma)
        """
        data = self.initial_data or {}
        try:
            xyz_path = data.get("xyz_path")
            if xyz_path:
                self._xyz_var.set(str(xyz_path))

            pcs_path = data.get("pcs_path")
            if pcs_path:
                self._pcs_var.set(str(pcs_path))

            metal_idx_1b = data.get("metal_idx_1b")
            if metal_idx_1b is not None:
                self._metal_idx_var.set(str(int(metal_idx_1b)))

            metal_xyz = data.get("metal_xyz")
            if metal_xyz is not None and len(metal_xyz) == 3:
                self._mx_var.set(f"{float(metal_xyz[0]):g}")
                self._my_var.set(f"{float(metal_xyz[1]):g}")
                self._mz_var.set(f"{float(metal_xyz[2]):g}")

            if "dchi_ax" in data:
                self._dchi_ax_var.set(f"{float(data.get('dchi_ax', 0.0)):g}")
            if "dchi_rh" in data:
                self._dchi_rh_var.set(f"{float(data.get('dchi_rh', 0.0)):g}")

            euler_deg = data.get("euler_deg")
            if euler_deg is not None and len(euler_deg) == 3:
                self._ea_var.set(f"{float(euler_deg[0]):g}")
                self._eb_var.set(f"{float(euler_deg[1]):g}")
                self._eg_var.set(f"{float(euler_deg[2]):g}")
            self._refresh_imported_summary()
            self._refresh_search_setup_summary()

        except Exception:
            # Keep GUI robust even if initial_data is incomplete or malformed.
            pass

    # Summary helper
    def _get_imported_summary_items(self):
        pcs_points = "N/A"
        try:
            pcs_path = self._pcs_var.get().strip()
            if pcs_path and os.path.isfile(pcs_path):
                import csv
                with open(pcs_path, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                pcs_points = str(len(rows))
        except Exception:
            pcs_points = "N/A"

        try:
            metal_ref = self._metal_idx_var.get().strip() or "N/A"
        except Exception:
            metal_ref = "N/A"

        try:
            mx = float(self._mx_var.get())
            my = float(self._my_var.get())
            mz = float(self._mz_var.get())
            metal_xyz_text = f"({mx:.3f}, {my:.3f}, {mz:.3f}) Å"
        except Exception:
            metal_xyz_text = "N/A"

        try:
            dchi_ax = float(self._dchi_ax_var.get())
            dchi_rh = float(self._dchi_rh_var.get())
            tensor_text = f"Δχ_ax = {dchi_ax:.2f}, Δχ_rh = {dchi_rh:.2f}"
        except Exception:
            tensor_text = "N/A"

        try:
            ea = float(self._ea_var.get())
            eb = float(self._eb_var.get())
            eg = float(self._eg_var.get())
            euler_text = f"({ea:.1f}, {eb:.1f}, {eg:.1f})°"
        except Exception:
            euler_text = "N/A"

        return [
            ("Structure", "current structure from main window"),
            ("PCS data", "assigned δ_exp values from main table"),
            ("Metal atom", f"Ref {metal_ref}"),
            ("Metal position", f"{metal_xyz_text}  (from original XYZ coordinates)"),
            ("Tensor", tensor_text),
            ("Orientation", f"Euler angle = {euler_text}"),
            ("PCS points", pcs_points),
        ]

    def _refresh_imported_summary(self, *_args):
        """
        Refresh the Imported from main summary label, if present.
        """
        try:
            if not hasattr(self, "_imported_value_labels"):
                return

            items = self._get_imported_summary_items()
            for i, (_, value) in enumerate(items):
                if i < len(self._imported_value_labels):
                    self._imported_value_labels[i].configure(text=value)
        except Exception:
            pass

    # optimization preset
    def _apply_optimization_preset(self, *_args):
        preset = self._opt_level_var.get().strip().lower()

        presets = {
            "quick": (100, 8),
            "balanced": (300, 12),
            "thorough": (800, 18),
            "custom": None,
        }

        values = presets.get(preset)
        if values is None:
            return

        maxiter, popsize = values
        try:
            self._applying_preset = True
            self._de_maxiter_var.set(str(maxiter))
            self._de_popsize_var.set(str(popsize))
        finally:
            self._applying_preset = False

    def _mark_custom_optimization(self, *_args):
        try:
            if getattr(self, "_applying_preset", False):
                return
            current = self._opt_level_var.get().strip()
            if current != "Custom":
                self._opt_level_var.set("Custom")
        except Exception:
            pass

    def _set_running_state(self, running: bool, text: str = ""):
        self._is_running = running

        try:
            if self._run_btn is not None:
                self._run_btn.configure(state=("disabled" if running else "normal"))
        except Exception:
            pass

        try:
            if self._progress_bar is not None:
                if running:
                    self._progress_bar.start(10)
                else:
                    self._progress_bar.stop()
        except Exception:
            pass

    def _build_search_setup_text(self) -> str:
        try:
            fixed_set = self._get_fixed_set()
            fixed_count = len(fixed_set) - (1 if self.metal_idx is not None else 0)
        except Exception:
            fixed_count = 0

        try:
            free_selected = sum(
                1 for v, locked in zip(self.bond_vars, self.bond_locked)
                if v.get() and not locked
            )
        except Exception:
            free_selected = 0

        pcs_points = 0
        try:
            pcs_path = self._pcs_var.get().strip()
            if pcs_path and os.path.isfile(pcs_path):
                import csv
                with open(pcs_path, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    pcs_points = len(list(reader))
        except Exception:
            pcs_points = 0

        return f"Search setup: {free_selected} free bonds | {fixed_count} fixed atoms | {pcs_points} PCS points"

    def _refresh_search_setup_summary(self, *_args):
        try:
            self._setup_summary_var.set(self._build_search_setup_text())
        except Exception:
            pass

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._nb = ttk.Notebook(self.root)
        self._nb.pack(fill="both", expand=True, padx=6, pady=6)

        if getattr(self, '_embedded', False):
            # embedded: Setup 탭은 "Run & Options"로 축소해서 마지막에
            t1 = ttk.Frame(self._nb); self._nb.add(t1, text="①  Fixed atoms")
            t2 = ttk.Frame(self._nb); self._nb.add(t2, text="②  Rotatable bonds")
            t3 = ttk.Frame(self._nb); self._nb.add(t3, text="③  Run & Options")
            t4 = ttk.Frame(self._nb); self._nb.add(t4, text="④  Results")

            self._build_fixed_tab(t1)
            self._build_bonds_tab(t2)
            self._build_setup_tab(t3)
            self._build_results_tab(t4)
        else:
            # standalone / Toplevel: 기존 순서 유지
            t1 = ttk.Frame(self._nb); self._nb.add(t1, text="①  Setup")
            t2 = ttk.Frame(self._nb); self._nb.add(t2, text="②  Fixed atoms")
            t3 = ttk.Frame(self._nb); self._nb.add(t3, text="③  Rotatable bonds")
            t4 = ttk.Frame(self._nb); self._nb.add(t4, text="④  Results")

            self._build_setup_tab(t1)
            self._build_fixed_tab(t2)
            self._build_bonds_tab(t3)
            self._build_results_tab(t4)

    # ─── Tab 3: Setup ────────────────────────────────────────────────────────
    def _build_setup_tab(self, p):
        pad = dict(padx=8, pady=4)
        embedded = getattr(self, '_embedded', False)

        if not embedded:
            # ── standalone/Toplevel:  ──────────────────────────────
            sf = ttk.LabelFrame(p, text="Molecular structure (XYZ)", padding=6)
            sf.pack(fill="x", **pad)
            row = ttk.Frame(sf);
            row.pack(fill="x")
            self._xyz_var = tk.StringVar()
            ttk.Entry(row, textvariable=self._xyz_var, width=54).pack(
                side="left", fill="x", expand=True)
            ttk.Button(row, text="Browse…", command=self._load_xyz
                       ).pack(side="left", padx=(4, 0))
            mr = ttk.Frame(sf);
            mr.pack(fill="x", pady=(4, 0))
            ttk.Label(mr, text="Metal atom index (1-based):").pack(side="left")
            self._metal_idx_var = tk.StringVar(value="1")
            self._metal_idx_var.trace_add("write", self._refresh_imported_summary)
            ttk.Entry(mr, textvariable=self._metal_idx_var, width=6
                      ).pack(side="left", padx=6)
            ttk.Label(mr, text="← atom number in XYZ file").pack(side="left")

            pf = ttk.LabelFrame(p, text="Experimental PCS  (CSV: Ref, δ_exp  [ppm])",
                                padding=6)
            pf.pack(fill="x", **pad)
            row2 = ttk.Frame(pf);
            row2.pack(fill="x")
            self._pcs_var = tk.StringVar()
            ttk.Entry(row2, textvariable=self._pcs_var, width=54).pack(
                side="left", fill="x", expand=True)
            ttk.Button(row2, text="Browse…", command=self._load_pcs
                       ).pack(side="left", padx=(4, 0))

            tf = ttk.LabelFrame(p, text="Tensor parameters  (×10⁻³² m³ / degrees)",
                                padding=6)
            tf.pack(fill="x", **pad)
            for row_items in [
                [("Δχ_ax", "_dchi_ax_var", "-2.00"), ("Δχ_rh", "_dchi_rh_var", "0.00")],
                [("Metal x [Å]", "_mx_var", "0.00"), ("y", "_my_var", "0.00"), ("z", "_mz_var", "0.00")],
                [("Euler ZYZ α [°]", "_ea_var", "0.00"), ("β", "_eb_var", "0.00"), ("γ", "_eg_var", "0.00")],
            ]:
                r = ttk.Frame(tf);
                r.pack(fill="x", pady=2)
                for lbl, attr, default in row_items:
                    setattr(self, attr, tk.StringVar(value=default))
                    getattr(self, attr).trace_add("write", self._refresh_imported_summary)
                    ttk.Label(r, text=lbl + ":").pack(side="left", padx=(0, 2))
                    ttk.Entry(r, textvariable=getattr(self, attr),
                              width=8).pack(side="left", padx=(0, 10))

        else:
            # ── embedded:  ──────────────────────────────
            self._xyz_var = tk.StringVar()
            self._xyz_var.trace_add("write", self._refresh_imported_summary)
            self._pcs_var = tk.StringVar()
            self._pcs_var.trace_add("write", self._refresh_imported_summary)
            self._metal_idx_var = tk.StringVar(value="1")
            self._metal_idx_var.trace_add("write", self._refresh_imported_summary)
            for attr, default in [
                ("_dchi_ax_var", "-2.00"), ("_dchi_rh_var", "0.00"),
                ("_mx_var", "0.00"), ("_my_var", "0.00"), ("_mz_var", "0.00"),
                ("_ea_var", "0.00"), ("_eb_var", "0.00"), ("_eg_var", "0.00"),
            ]:
                setattr(self, attr, tk.StringVar(value=default))
                getattr(self, attr).trace_add("write", self._refresh_imported_summary)

        # ── Summary imported-from-main summary ─────────────────────
        inf = ttk.LabelFrame(
            p,
            text="Starting values imported from the main window",
            padding=10
        )
        inf.pack(fill="x", **pad)

        ttk.Label(
            inf,
            text="These values are used as the starting point for the current search.",
            foreground="#555555",
        ).pack(anchor="w", pady=(0, 4))

        self._imported_value_labels = []

        grid = ttk.Frame(inf)
        grid.pack(fill="x")

        items = [
            ("Structure", "No imported values available yet."),
            ("PCS data", "—"),
            ("Metal atom", "—"),
            ("Metal position", "—"),
            ("Tensor", "—"),
            ("Orientation", "—"),
            ("PCS points", "—"),
        ]

        for r, (key, value) in enumerate(items):
            ttk.Label(
                grid,
                text=key,
                font=("TkDefaultFont", 9, "bold"),
                width=14,
                anchor="w",
            ).grid(row=r, column=0, sticky="nw", padx=(0, 10), pady=2)

            val_lbl = ttk.Label(
                grid,
                text=value,
                anchor="w",
                justify="left",
                foreground="#333333",
            )
            val_lbl.grid(row=r, column=1, sticky="w", pady=2)
            self._imported_value_labels.append(val_lbl)

        grid.columnconfigure(1, weight=1)

        # ── Optimization options:  ───────────────────────────────
        of = ttk.LabelFrame(p, text="Optimization options", padding=10)
        of.pack(fill="x", **pad)

        r1 = ttk.Frame(of)
        r1.pack(fill="x")

        self._use_global_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            r1,
            text="Use global search (Differential Evolution)",
            variable=self._use_global_var
        ).pack(side="left")

        r_preset = ttk.Frame(of)
        r_preset.pack(fill="x", pady=(8, 0))

        ttk.Label(
            r_preset,
            text="Optimization level:",
            font=("TkDefaultFont", 9, "bold"),
        ).pack(side="left")

        preset_box = ttk.Combobox(
            r_preset,
            textvariable=self._opt_level_var,
            values=["Quick", "Balanced", "Thorough", "Custom"],
            state="readonly",
            width=12,
        )
        preset_box.pack(side="left", padx=6)
        preset_box.bind("<<ComboboxSelected>>", self._apply_optimization_preset)

        ttk.Label(
            r_preset,
            text="Quick = faster / Balanced = recommended / Thorough = slower but more robust",
            foreground="#555555",
        ).pack(side="left", padx=(10, 0))

        r2 = ttk.Frame(of)
        r2.pack(fill="x", pady=(8, 0))

        ttk.Label(
            r2,
            text="Custom values:",
            font=("TkDefaultFont", 9, "bold"),
        ).pack(side="left", padx=(0, 8))

        ttk.Label(r2, text="DE iterations").pack(side="left")
        self._de_maxiter_var = tk.StringVar(value="300")
        self._de_maxiter_var.trace_add("write", self._mark_custom_optimization)
        ttk.Entry(r2, textvariable=self._de_maxiter_var, width=6).pack(side="left", padx=(6, 12))

        ttk.Label(r2, text="Population size").pack(side="left")
        self._de_popsize_var = tk.StringVar(value="12")
        self._de_popsize_var.trace_add("write", self._mark_custom_optimization)
        ttk.Entry(r2, textvariable=self._de_popsize_var, width=5).pack(side="left", padx=(6, 0))

        ttk.Label(
            of,
            text="Changing the values manually switches the level to Custom.",
            foreground="#777777",
        ).pack(anchor="w", pady=(6, 0))

        # ── Action row ─────────────────────────────────────────────────────────
        af = ttk.Frame(p)
        af.pack(fill="x", **pad)

        if not embedded:
            ttk.Button(
                af,
                text="▶ Detect bonds",
                command=self._detect_all
            ).pack(side="left", padx=(0, 10))

        self._progress_bar = ttk.Progressbar(
            af,
            mode="indeterminate",
            length=100,
        )
        self._progress_bar.pack(side="left", padx=(0, 8))

        self._run_btn = ttk.Button(
            af,
            text="▶▶ Run conformer search",
            command=self._run_search
        )
        self._run_btn.pack(side="left", padx=(0, 10))

        self._status_var = tk.StringVar(
            value="Run after 'Sync structure & PCS'." if embedded else "Load an XYZ file to start."
        )

        ttk.Label(
            af,
            textvariable=self._status_var,
            foreground="#666666",
        ).pack(side="left", fill="x", expand=True)

    # ─── Tab 1: Fixed atoms ──────────────────────────────────────────────────

    def _build_fixed_tab(self, p):
        # Header explanation
        hdr = ttk.Frame(p)
        hdr.pack(fill="x", padx=10, pady=(10, 6))

        ttk.Label(
            hdr,
            text="Fix atoms that should stay unchanged",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            hdr,
            text=(
                "Selected atoms remain fixed during the conformer search. "
                "Bonds connected to fixed regions are locked automatically, "
                "so the search focuses on flexible parts of the structure."
            ),
            justify="left",
            foreground="#444444",
            wraplength=900,
        ).pack(anchor="w", pady=(4, 0))

        tip = tk.Frame(hdr, bg="#eef4ff", bd=0, highlightthickness=1, highlightbackground="#d7e3ff")
        tip.pack(fill="x", pady=(8, 0))

        tk.Label(
            tip,
            text=(
                "Typical use\n"
                "• Fix donor atoms (N, O, P, S...) and rigid core atoms\n"
                "• Leave pendant arms free for conformational optimization"
            ),
            bg="#eef4ff",
            fg="#284b7a",
            justify="left",
            anchor="w",
            font=("TkDefaultFont", 9),
            padx=8,
            pady=6,
        ).pack(anchor="w", fill="x")

        # Filter / bulk controls
        ctrl = ttk.Frame(p); ctrl.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(ctrl, text="Filter element:").pack(side="left")
        self._fix_filter_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self._fix_filter_var,
                  width=5).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Apply filter",
                   command=self._apply_fixed_filter).pack(side="left", padx=2)
        ttk.Separator(ctrl, orient="vertical").pack(
            side="left", fill="y", padx=8, pady=2)
        ttk.Button(ctrl, text="\u2714 Fix all shown",
                   command=lambda: self._set_fixed_all(True)).pack(side="left")
        ttk.Button(ctrl, text="\u2718 Unfix all",
                   command=lambda: self._set_fixed_all(False)).pack(side="left", padx=4)
        ttk.Button(ctrl, text="\u2195 Invert",
                   command=self._invert_fixed).pack(side="left")
        ttk.Separator(ctrl, orient="vertical").pack(
            side="left", fill="y", padx=8, pady=2)
        # THE KEY BUTTON — apply selection and jump to Tab ③
        ttk.Button(ctrl,
                   text="\u27f3  Apply & show locked bonds",
                   command=self._refresh_bond_locks,
                   ).pack(side="left", padx=6)

        # Scrollable atom list
        cf = ttk.Frame(p); cf.pack(fill="both", expand=True, padx=8, pady=(0, 6))
        self._fix_canvas = tk.Canvas(cf, bg="#f5f6fa", highlightthickness=0)
        vsb = ttk.Scrollbar(cf, orient="vertical",
                            command=self._fix_canvas.yview)
        self._fix_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._fix_canvas.pack(side="left", fill="both", expand=True)
        self._fix_inner = ttk.Frame(self._fix_canvas)
        self._fix_canvas.create_window((0, 0), window=self._fix_inner, anchor="nw")
        self._fix_inner.bind("<Configure>",
                             lambda e: self._fix_canvas.configure(
                                 scrollregion=self._fix_canvas.bbox("all")))
        self._fix_rows: list = []
        self._visible_set: set = set()


    # ─── Tab 2: Rotatable bonds ──────────────────────────────────────────────

    def _build_bonds_tab(self, p):
        hdr = ttk.Frame(p)
        hdr.pack(fill="x", padx=10, pady=(10, 6))

        ttk.Label(
            hdr,
            text="Choose which bonds are allowed to rotate",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            hdr,
            text=(
                "Free bonds can be included in the conformer search, while locked bonds "
                "are excluded automatically."
            ),
            justify="left",
            foreground="#444444",
            wraplength=900,
        ).pack(anchor="w", pady=(4, 0))

        legend = tk.Frame(hdr, bg="#f5f7fb", bd=0, highlightthickness=1, highlightbackground="#e2e7f0")
        legend.pack(fill="x", pady=(8, 0))

        tk.Label(
            legend,
            text=(
                "🔓 Free to rotate    🔒 Locked and unavailable\n\n"
                "e.g. Bond 2–4: N(2)–C(4) (+0.0°) [92 atoms] " 
                "→ angle = current dihedral, atoms = rotating side size\n"
                "Anchor mode (recommended)\n"
                "Fixed atoms act as anchors, so nearby flexible bonds can still rotate. "
                "This works well for ligands with rigid donor regions and flexible side arms."
            ),
            bg="#eef4ff",
            fg="#284b7a",
            justify="left",
            anchor="w",
            font=("TkDefaultFont", 9),
            padx=8,
            pady=6,
        ).pack(anchor="w", fill="x")

        # Mode selector
        mode_row = ttk.Frame(p); mode_row.pack(fill="x", padx=8, pady=(0, 4))
        self._anchor_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mode_row,
                        text="Anchor mode  (recommended for tetrapodal/tripodal ligands)",
                        variable=self._anchor_mode_var,
                        command=self._refresh_bond_locks).pack(side="left")

        btn = ttk.Frame(p); btn.pack(fill="x", padx=8)
        ttk.Button(btn, text="\u2714 Select all free",
                   command=lambda: [v.set(True)
                                    for v, lk in zip(self.bond_vars,
                                                     self.bond_locked)
                                    if not lk]).pack(side="left")
        ttk.Button(btn, text="\u2718 Select none",
                   command=lambda: [v.set(False)
                                    for v in self.bond_vars]).pack(side="left", padx=6)
        ttk.Button(btn, text="\u2195 Invert",
                   command=lambda: [v.set(not v.get())
                                    for v, lk in zip(self.bond_vars,
                                                     self.bond_locked)
                                    if not lk]).pack(side="left")

        cf = ttk.Frame(p); cf.pack(fill="both", expand=True, padx=8, pady=6)
        self._bond_canvas = tk.Canvas(cf, bg="#f5f6fa", highlightthickness=0)
        vsb = ttk.Scrollbar(cf, orient="vertical",
                            command=self._bond_canvas.yview)
        self._bond_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._bond_canvas.pack(side="left", fill="both", expand=True)
        self._bond_inner = ttk.Frame(self._bond_canvas)
        self._bond_canvas.create_window((0, 0), window=self._bond_inner, anchor="nw")
        self._bond_inner.bind("<Configure>",
                              lambda e: self._bond_canvas.configure(
                                  scrollregion=self._bond_canvas.bbox("all")))

    # ─── Tab ④: Results ──────────────────────────────────────────────────────

    def _build_results_tab(self, p):
        # -- Candidate selector frame (initially empty; populated after run) --
        self._candidate_selector_frame = ttk.Frame(p)
        self._candidate_selector_frame.pack(fill="x", padx=8, pady=(6, 2))

        # -- Action buttons row --
        btn_row = ttk.Frame(p)
        btn_row.pack(fill="x", padx=8, pady=4)
        ttk.Button(btn_row, text="View structure",
                   command=self._show_structure_plot).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="Select candidate…",
                   command=self._apply_selected).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="💾 Save XYZ…",
                   command=self._save_xyz).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="💾 Save report…",
                   command=self._save_report).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="📊 Correlation plot",
                   command=self._show_plot).pack(side="left")

        self._result_box = scrolledtext.ScrolledText(
            p, font=("Courier", 9), state="disabled")
        self._result_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ── Actions ───────────────────────────────────────────────────────────────

    def _load_xyz(self):
        path = filedialog.askopenfilename(
            title="Open XYZ", filetypes=[("XYZ","*.xyz"),("All","*.*")])
        if not path:
            return
        try:
            elements, coords = load_xyz(path)
            self.mol = Molecule(elements, coords)
            self._xyz_var.set(path)
            self._status_var.set(
                f"Loaded {self.mol.n_atoms} atoms, {len(self.mol.bonds)} bonds.")
        except Exception as e:
            messagebox.showerror("XYZ load error", str(e))

    def _load_pcs(self):
        path = filedialog.askopenfilename(
            title="Open PCS CSV", filetypes=[("CSV","*.csv"),("All","*.*")])
        if path:
            self._pcs_var.set(path)

    def _get_metal_idx(self) -> Optional[int]:
        try:
            idx = int(self._metal_idx_var.get()) - 1
            if not (0 <= idx < self.mol.n_atoms):
                raise ValueError
            return idx
        except Exception:
            messagebox.showerror("Metal index",
                                 f"Enter a number between 1 and {self.mol.n_atoms}.")
            return None

    def _detect_all(self):
        """Detect rotatable bonds and populate the atom list in Tab 1."""
        if self.mol is None:
            messagebox.showwarning("No structure", "Load an XYZ file first.")
            return
        metal_idx = self._get_metal_idx()
        if metal_idx is None:
            return
        self.metal_idx = metal_idx

        self.rotatable_all = self.mol.find_rotatable_bonds(self.metal_idx)

        # Build atom list for fixed-atom tab
        self._populate_atom_list()
        # Build bond list
        self._refresh_bond_locks()

        self._status_var.set(
            f"{len(self.rotatable_all)} rotatable bonds detected. "
            f"Mark fixed atoms in Tab ①, then run.")
        self._nb.select(0 if getattr(self, '_embedded', False) else 1)

    def _populate_atom_list(self):
        """Fill Tab ② with one row per heavy atom."""
        for w in self._fix_inner.winfo_children():
            w.destroy()
        self.atom_fix_vars.clear()
        self._fix_rows.clear()
        self._visible_set = set()   # Bug-fix: track visibility ourselves

        header = ttk.Frame(self._fix_inner)
        header.pack(fill="x", padx=4, pady=(2, 0))
        for txt, w in [("Fix?", 5), ("Ref", 7), ("Atom", 5),
                        ("x [Å]", 9), ("y [Å]", 9), ("z [Å]", 9),
                        ("Bonds to", 20)]:
            ttk.Label(header, text=txt, font=("default", 8, "bold"),
                      width=w, anchor="center").pack(side="left")
        ttk.Separator(self._fix_inner, orient="horizontal").pack(fill="x", pady=2)

        for idx in range(self.mol.n_atoms):
            if idx == self.metal_idx:
                continue   # metal is always implicitly fixed
            el  = self.mol.elements[idx]
            xyz = self.mol.coords[idx]
            nbs = [f"{n+1}({self.mol.elements[n]})" for n in self.mol.adj[idx]]

            v = tk.BooleanVar(value=False)
            self.atom_fix_vars.append(v)

            row = ttk.Frame(self._fix_inner)
            row.pack(fill="x", padx=4, pady=1)

            # No automatic command on checkbox — user controls when to apply
            ttk.Checkbutton(row, variable=v).pack(side="left")
            ttk.Label(row, text=str(idx+1), width=7, anchor="e").pack(side="left")
            ttk.Label(row, text=el, width=5, anchor="center",
                      font=("default", 9, "bold")).pack(side="left")
            for val in xyz:
                ttk.Label(row, text=f"{val:+8.3f}", width=9,
                          anchor="e", font=("Courier", 8)).pack(side="left")
            ttk.Label(row, text=", ".join(nbs[:8]) + ("…" if len(nbs) > 8 else ""),
                      font=("default", 8), foreground="#555"
                      ).pack(side="left", padx=(4, 0))

            self._fix_rows.append((v, el, idx, row))
            self._visible_set.add(idx)   # all visible initially
            self._refresh_search_setup_summary()

    def _apply_fixed_filter(self):
        """Show/hide atom rows by element symbol — updates _visible_set."""
        filt = self._fix_filter_var.get().strip().capitalize()
        self._visible_set = set()
        for v, el, idx, row in self._fix_rows:
            if filt and el != filt:
                row.pack_forget()
            else:
                row.pack(fill="x", padx=4, pady=1)
                self._visible_set.add(idx)
        self._update_fixed_summary()

    def _set_fixed_all(self, value: bool):
        """Set all *visible* atoms' fixed-state to value."""
        for v, el, idx, row in self._fix_rows:
            if idx in self._visible_set:
                v.set(value)
        self._update_fixed_summary()

    def _invert_fixed(self):
        """Invert fixed-state for all *visible* atoms."""
        for v, el, idx, row in self._fix_rows:
            if idx in self._visible_set:
                v.set(not v.get())
        self._update_fixed_summary()

    def _update_fixed_summary(self):
        """Update the summary label in Tab ② (not the bond list yet)."""
        n = sum(1 for v, el, idx, row in self._fix_rows if v.get())
        if hasattr(self, '_fix_summary_var'):
            self._fix_summary_var.set(
                f"{n} atom(s) marked as fixed  "
                f"(click '⟳ Apply & show locked bonds')")

    def _get_fixed_set(self) -> Set[int]:
        """Return set of currently fixed atom indices (0-based)."""
        fixed = set()
        if self.metal_idx is not None:
            fixed.add(self.metal_idx)
        for v, el, idx, row in self._fix_rows:
            if v.get():
                fixed.add(idx)
        return fixed

    def _refresh_bond_locks(self):
        """
        Recompute which bonds are locked given the current fixed-atom set,
        rebuild the bond list in Tab ③, and switch focus there so the user
        sees the result immediately.
        """
        if not self.rotatable_all:
            return
        fixed_set   = self._get_fixed_set()
        use_anchor  = getattr(self, '_anchor_mode_var', None)
        use_anchor  = use_anchor.get() if use_anchor is not None else True
        free, locked_list = classify_bonds_by_fixed(
            self.mol, self.rotatable_all, fixed_set,
            metal_idx=self.metal_idx, anchor_mode=use_anchor)
        locked_set = set(map(tuple, locked_list))

        # Rebuild bond checkbox list
        for w in self._bond_inner.winfo_children():
            w.destroy()
        self.bond_vars.clear()
        self.bond_locked.clear()

        n_free   = sum(1 for b in self.rotatable_all if tuple(b) not in locked_set)
        n_locked = len(self.rotatable_all) - n_free
        ttk.Label(self._bond_inner,
                  text=(f"{len(self.rotatable_all)} rotatable bonds  "
                        f"({n_free} free 🔓,  {n_locked} locked 🔒)"),
                  font=("default", 9, "bold")
                  ).pack(anchor="w", padx=6, pady=(4, 2))

        for i, j in self.rotatable_all:
            ei = self.mol.elements[i]
            ej = self.mol.elements[j]
            is_locked = (i, j) in locked_set

            nb_i = [n for n in self.mol.adj[i] if n != j]
            nb_j = [n for n in self.mol.adj[j] if n != i]
            if nb_i and nb_j:
                ang = dihedral_angle(
                    self.mol.coords[nb_i[0]], self.mol.coords[i],
                    self.mol.coords[j],        self.mol.coords[nb_j[0]])
                ang_txt = f"  ({ang:+.1f}°)"
            else:
                ang_txt = ""

            # Count fixed atoms in the rotating subtree
            subtree = set(self.mol.subtree_of(j, i))
            fixed_in_sub = subtree & fixed_set
            lock_reason  = ""
            if is_locked and fixed_in_sub:
                lock_reason = (f"  🔒 locked — fixed atom(s): "
                               f"{', '.join(str(k+1)+'('+self.mol.elements[k]+')' for k in sorted(fixed_in_sub))}")

            v = tk.BooleanVar(value=(not is_locked))
            self.bond_vars.append(v)
            self.bond_locked.append(is_locked)

            row = ttk.Frame(self._bond_inner)
            row.pack(fill="x", padx=6, pady=1)

            state = "disabled" if is_locked else "normal"
            icon  = "🔒" if is_locked else "🔓"
            label = (f"{icon}  Bond {i+1}–{j+1}:  "
                     f"{ei}({i+1}) – {ej}({j+1})"
                     f"{ang_txt}  "
                     f"[{len(self.mol.subtree_of(j,i))} atoms]"
                     f"{lock_reason}")
            cb = ttk.Checkbutton(row, text=label, variable=v, state=state)
            cb.pack(anchor="w")
            if is_locked:
                # Grey-out locked rows visually
                try:
                    cb.configure(style="Locked.TCheckbutton")
                except Exception:
                    pass

        fixed_count = len(fixed_set) - (1 if self.metal_idx is not None else 0)
        mode_str = 'anchor' if use_anchor else 'strict'
        self._status_var.set(
            f"{n_free} free bonds, {n_locked} locked  "
            f"({fixed_count} user-fixed atom(s), {mode_str} mode).")
        if n_locked == len(self.rotatable_all) and n_locked > 0 and not use_anchor:
            # All locked in strict mode — suggest switching
            ttk.Label(self._bond_inner,
                      text="⚠ All bonds locked in strict mode.\n"
                           "Try enabling 'Anchor mode' (checkbox above) — "
                           "fixed atoms then act as pivots instead of blockers.",
                      foreground='#c0392b', font=('default',9,'bold'),
                      wraplength=700, justify='left').pack(anchor='w', padx=8, pady=4)
        self._refresh_search_setup_summary()
        self._nb.select(1 if getattr(self, '_embedded', False) else 2)

    def _run_search(self):
        if self.mol is None:
            messagebox.showwarning("No structure", "Load an XYZ file first.")
            return
        if not self.rotatable_all:
            messagebox.showwarning("No bonds",
                                   "Click 'Detect bonds & fill atom list' first.")
            return

        fixed_set    = self._get_fixed_set()
        use_anchor2 = getattr(self, '_anchor_mode_var', None)
        use_anchor2 = use_anchor2.get() if use_anchor2 is not None else True
        _, locked_bonds = classify_bonds_by_fixed(
            self.mol, self.rotatable_all, fixed_set,
            metal_idx=self.metal_idx, anchor_mode=use_anchor2)
        # Only use bonds that are both free and user-selected
        selected_bonds = [b for b, v, lk in zip(self.rotatable_all,
                                                  self.bond_vars,
                                                  self.bond_locked)
                          if v.get() and not lk]
        if not selected_bonds:
            messagebox.showwarning("No free bonds selected",
                                   "Select at least one free (🔓) rotatable bond.")
            return

        # Parse tensor parameters
        try:
            dchi_ax = float(self._dchi_ax_var.get())
            dchi_rh = float(self._dchi_rh_var.get())
            mx, my, mz = (float(self._mx_var.get()),
                          float(self._my_var.get()),
                          float(self._mz_var.get()))
            ea, eb, eg = (math.radians(float(self._ea_var.get())),
                          math.radians(float(self._eb_var.get())),
                          math.radians(float(self._eg_var.get())))
            metal = np.array([mx, my, mz])
            euler = (ea, eb, eg)
        except ValueError as e:
            messagebox.showerror("Tensor parameters", f"Invalid value: {e}")
            return

        # Load PCS
        pcs_path = self._pcs_var.get().strip()
        if not pcs_path or not os.path.isfile(pcs_path):
            messagebox.showerror("PCS data",
                                 "Select a valid PCS CSV file in Tab ①.")
            return
        try:
            atom_indices, pcs_values = load_pcs_csv(pcs_path, self.mol.elements)
        except Exception as e:
            messagebox.showerror("PCS CSV", str(e)); return
        if not atom_indices:
            messagebox.showerror("PCS data",
                                 "No valid data found.\n"
                                 "Expected columns: Ref (1-based), δ_exp [ppm]")
            return

        try:
            de_maxiter = int(self._de_maxiter_var.get())
            de_popsize = int(self._de_popsize_var.get())
        except ValueError:
            de_maxiter, de_popsize = 600, 12

        use_global = self._use_global_var.get()
        self._status_var.set("Running optimization…")
        self._set_running_state(True, "Preparing optimization...")
        self.root.update_idletasks()

        def _progress(msg):
            self._status_var.set(msg)
            self.root.update_idletasks()

        import threading
        def _run():
            try:
                opt = ConformerOptimiser(
                    mol         = self.mol,
                    rotatable   = selected_bonds,
                    obs_atoms   = atom_indices,
                    obs_pcs     = np.array(pcs_values, float),
                    metal       = metal,
                    dchi_ax     = dchi_ax,
                    dchi_rh     = dchi_rh,
                    euler_zyz   = euler,
                    metal_idx   = self.metal_idx,
                    fixed_atoms = fixed_set,
                    progress_cb = _progress,
                )
                opt.run(
                    use_global=use_global,
                    de_maxiter=de_maxiter,
                    de_popsize=de_popsize,
                    selected_bonds=selected_bonds,
                    locked_bonds=locked_bonds,
                )
                # opt.candidates is populated by run(); each dict already has selected/locked.
                candidates = opt.candidates  # list[dict], sorted by RMSD
                self.root.after(0, lambda c=candidates: (
                    self._set_running_state(False, "Completed."),
                    self._show_results(c),
                ))
            except Exception:
                import traceback
                tb = traceback.format_exc()
                self.root.after(0, lambda t=tb: (
                    self._set_running_state(False, "Error."),
                    messagebox.showerror("Optimization error", t[:900]),
                    self._status_var.set("Error.")
                ))

        threading.Thread(target=_run, daemon=True).start()

    def _show_results(self, candidates: list):
        """Store all candidates; populate the selector widget; show best report."""
        self.candidates = candidates
        self.result = candidates[0]  # backward-compat: best result

        # -- Rebuild candidate selector radio buttons --
        self._build_candidate_selector()

        # -- Show report for best candidate --
        self._display_candidate_report(0)

        fd = candidates[0].get('fixed_displacement', 0.0)
        self._status_var.set(
            f"Done.  {len(candidates)} candidates  "
            f"Best RMSD={candidates[0]['rmsd']:.4f} ppm  R²={candidates[0]['r2']:.4f}"
            + (f"  ⚠ {candidates[0]['n_clashes']} clashes" if candidates[0]['n_clashes'] else "")
            + (f"  ⚠ fixed drift {fd:.3f} Å" if fd > 0.01 else "")
        )
        self._nb.select(3)

        if callable(self.on_preview_result):
            payload = {
                "result": candidates[0],
                "elements": list(self.mol.elements),
                "coords_initial": self.mol.coords.copy(),
                "coords_opt": candidates[0]["coords_opt"].copy(),
                "report": format_report(candidates[0],
                                        candidates[0]['selected_bonds'],
                                        self.mol,
                                        candidates[0].get('locked_bonds'),
                                        rank=1, n_total=len(candidates)),
                "metal_idx": self.metal_idx,
            }
            try:
                self.on_preview_result(payload)
            except Exception as e:
                messagebox.showwarning("Preview callback warning",
                                       f"Result computed, but preview callback failed:\n{e}")

    def _build_candidate_selector(self):
        """(Re)build radio-button row in the results tab."""
        frame = self._candidate_selector_frame
        style = ttk.Style()
        style.configure("Candidate.TRadiobutton", font=("Segoe UI", 8))
        for w in frame.winfo_children():
            w.destroy()
        self._candidate_var = tk.IntVar(value=0)
        for i, cand in enumerate(self.candidates):
            clash_warn = " ⚠" if cand['n_clashes'] else ""
            lbl = (f"#{i + 1}  RMSD {cand['rmsd']:.4f}"
                   + (f"  R² {cand['r2']:.4f}" if not math.isnan(cand['r2']) else "")
                   + clash_warn)
            rb = ttk.Radiobutton(
                frame,
                text=lbl,
                style="Candidate.TRadiobutton",
                variable=self._candidate_var,
                value=i,
                command=lambda idx=i: self._display_candidate_report(idx),
            )
            rb.grid(row=0, column=i, padx=(0, 12), sticky="w")

    def _apply_selected(self):
        """
        Open a dialog to select one candidate and send it to the host application
        as a preview payload.  The host's '✅ Apply Preview' button then commits
        the coordinates to the main structure.  If no host callback is registered
        (standalone mode), apply the coordinates immediately.
        """
        if not getattr(self, 'candidates', None):
            messagebox.showwarning("No result", "Run the search first.")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Select candidate to apply")
        dlg.resizable(False, False)

        if callable(self.on_preview_result):
            hint = ("Select a candidate.  The result will be sent to the main window as a preview.\n"
                    "Click '✅ Apply Preview' in the main toolbar to commit the structure.")
        else:
            hint = "Select a candidate to apply directly."
        ttk.Label(dlg, text=hint, wraplength=360).pack(padx=16, pady=(14, 8), anchor="w")

        sel_var = tk.IntVar(value=0)
        for i, cand in enumerate(self.candidates):
            r2_str = f"  R²={cand['r2']:.4f}" if not math.isnan(cand['r2']) else ""
            clash_str = "  ⚠ clash" if cand['n_clashes'] else ""
            best_str = "  ★ best" if i == 0 else ""
            lbl = f"#{i + 1}  RMSD {cand['rmsd']:.4f} ppm{r2_str}{clash_str}{best_str}"
            ttk.Radiobutton(dlg, text=lbl, variable=sel_var, value=i).pack(
                anchor="w", padx=24, pady=2)

        def _do_apply():
            idx = sel_var.get()
            cand = self.candidates[idx]
            self.result = cand  # keep internal reference in sync

            payload = {
                "result": cand,
                "elements": list(self.mol.elements),
                "coords_initial": self.mol.coords.copy(),
                "coords_opt": cand["coords_opt"].copy(),
                "report": format_report(
                    cand, cand['selected_bonds'], self.mol,
                    cand.get('locked_bonds'),
                    rank=idx + 1, n_total=len(self.candidates)),
                "metal_idx": self.metal_idx,
            }
            dlg.destroy()

            if callable(self.on_preview_result):
                # Embedded mode: push payload to host; host's Apply button commits it.
                try:
                    self.on_preview_result(payload)
                except Exception as e:
                    messagebox.showwarning("Preview send warning", str(e))
            else:
                # Standalone mode: no host to apply to — just confirm to user.
                messagebox.showinfo(
                    "Candidate selected",
                    f"Candidate #{idx + 1} set as active result.\n"
                    f"Use 'Save XYZ…' to export the structure."
                )

        btn_row = ttk.Frame(dlg)
        btn_row.pack(padx=16, pady=12, anchor="e")
        ttk.Button(btn_row, text="Cancel", command=dlg.destroy).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Select candidate", command=_do_apply).pack(side="left")

    def _show_structure_plot(self):
        """Open an interactive 2D polar structure viewer for conformer candidates."""
        if not getattr(self, "candidates", None):
            messagebox.showwarning("No result", "Run the search first.")
            return

        import matplotlib
        matplotlib.use("TkAgg")
        import numpy as np
        import tkinter as tk
        from tkinter import ttk, filedialog
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.lines import Line2D

        try:
            from logic.chem_constants import CPK_COLORS
            def cpk(el):
                return CPK_COLORS.get(el, CPK_COLORS.get("default", "#aaaaaa"))
        except Exception:
            _CPK = {
                "H": "#ffffff", "C": "#444444", "N": "#3050f8", "O": "#ff0d0d",
                "F": "#90e050", "S": "#ffff30", "default": "#aaaaaa"
            }

            def cpk(el):
                return _CPK.get(el, _CPK["default"])

        # original + candidates
        CAND_COLORS = ["#A0A7B4", "#1a8c5e", "#3b7dd8", "#c47a1a", "#9b4fc4", "#c44f4f"]

        win = tk.Toplevel(self.root)
        win.title("Conformer Candidates — 2D Polar Structure View")
        win.geometry("1120x760")
        win.minsize(980, 680)

        # =========================
        # Main layout
        # =========================
        outer = ttk.Frame(win, padding=10)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 8))

        ttk.Label(
            header,
            text="Conformer Candidates",
            font=("TkDefaultFont", 12, "bold")
        ).pack(side="left")

        ttk.Label(
            header,
            text="2D polar view of original structure and optimized candidates",
            foreground="#666666"
        ).pack(side="left", padx=(10, 0))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(body, width=280)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        # =========================
        # Figure
        # =========================
        fig = plt.Figure(figsize=(7.4, 6.4), dpi=120, facecolor="#F7F8FA")
        ax = fig.add_subplot(1, 1, 1, projection="polar")
        fig.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.08)

        canvas = FigureCanvasTkAgg(fig, master=left)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, left)
        toolbar.update()

        metal_pos = np.array([
            float(self._mx_var.get()),
            float(self._my_var.get()),
            float(self._mz_var.get()),
        ], dtype=float)

        # =========================
        # UI state
        # =========================
        v_original = tk.BooleanVar(value=True)
        v_best = tk.BooleanVar(value=True)
        v_2 = tk.BooleanVar(value=True)
        v_3 = tk.BooleanVar(value=False)
        v_4 = tk.BooleanVar(value=False)
        v_5 = tk.BooleanVar(value=False)

        v_show_h = tk.BooleanVar(value=True)
        v_show_bonds = tk.BooleanVar(value=True)
        v_show_atoms = tk.BooleanVar(value=False)

        proj_var = tk.StringVar(value="0-180")

        summary_var = tk.StringVar(value="")

        # =========================
        # Helpers
        # =========================
        def _polar_coords(coords):
            rel = np.asarray(coords, float) - metal_pos
            r = np.linalg.norm(rel, axis=1)
            theta = np.where(
                r > 1e-9,
                np.arccos(np.clip(rel[:, 2] / (r + 1e-15), -1.0, 1.0)),
                0.0
            )
            return theta, r

        def _candidate_visible(idx):
            if idx == -1:
                return v_original.get()
            if idx == 0:
                return v_best.get()
            if idx == 1:
                return v_2.get()
            if idx == 2:
                return v_3.get()
            if idx == 3:
                return v_4.get()
            if idx == 4:
                return v_5.get()
            return False

        def _project_theta(theta):
            if proj_var.get() == "0-90":
                return np.where(theta > np.pi / 2, np.pi - theta, theta), np.pi / 2
            return theta, np.pi

        def _draw_structure(coords, elements, bonds, color, alpha, lw, zorder, ms, edge_alpha=1.0):
            theta, r = _polar_coords(coords)
            theta_plot, theta_max = _project_theta(theta)

            if v_show_bonds.get():
                for bi, bj in bonds:
                    e1 = elements[bi]
                    e2 = elements[bj]
                    if (not v_show_h.get()) and (e1 == "H" or e2 == "H"):
                        continue
                    ax.plot(
                        [theta_plot[bi], theta_plot[bj]],
                        [r[bi], r[bj]],
                        color=color,
                        alpha=alpha * 0.55,
                        lw=lw,
                        solid_capstyle="round",
                        zorder=zorder,
                    )

            if v_show_atoms.get():
                for k, el in enumerate(elements):
                    if (not v_show_h.get()) and el == "H":
                        continue
                    ax.scatter(
                        theta_plot[k], r[k],
                        color=cpk(el),
                        edgecolors=color,
                        linewidths=max(0.6, lw),
                        s=ms,
                        zorder=zorder + 1,
                        alpha=alpha * edge_alpha,
                    )

            return theta_max

        def _legend_handles():
            handles = []

            if v_original.get():
                handles.append(
                    Line2D([0], [0], color=CAND_COLORS[0], lw=1.8, label="original")
                )

            labels = []
            if v_best.get() and len(self.candidates) >= 1:
                labels.append((0, "#1 best"))
            if v_2.get() and len(self.candidates) >= 2:
                labels.append((1, "#2"))
            if v_3.get() and len(self.candidates) >= 3:
                labels.append((2, "#3"))
            if v_4.get() and len(self.candidates) >= 4:
                labels.append((3, "#4"))
            if v_5.get() and len(self.candidates) >= 5:
                labels.append((4, "#5"))

            for idx, label in labels:
                col = CAND_COLORS[min(idx + 1, len(CAND_COLORS) - 1)]
                handles.append(Line2D([0], [0], color=col, lw=2.0, label=label))

            return handles

        def _build_summary_text():
            lines = []
            for i, cand in enumerate(self.candidates[:5], start=1):
                rmsd = cand.get("rmsd", float("nan"))
                tag = "best" if i == 1 else ""
                lines.append(f"#{i:<1}  RMSD {rmsd:.4f} {tag}".rstrip())
            return "\n".join(lines)

        def _configure_axes(r_max):
            ax.set_facecolor("white")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)

            if proj_var.get() == "0-90":
                ax.set_thetamax(90)
                ax.set_xticks([0, np.pi / 6, np.pi / 3, np.pi / 2])
                ax.set_xticklabels(["0°", "30°", "60°", "90°"], fontsize=9)
            else:
                ax.set_thetamax(180)
                ax.set_xticks([0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6, np.pi])
                ax.set_xticklabels(["0°", "30°", "60°", "90°", "120°", "150°", "180°"], fontsize=9)

            ax.set_ylim(0, r_max)

            # cleaner radial ticks
            rticks = np.linspace(0, r_max, 5)
            ax.set_yticks(rticks[1:])
            ax.set_yticklabels([f"{v:.1f}" for v in rticks[1:]], fontsize=8, color="#666666")

            ax.grid(True, alpha=0.22, lw=0.8, color="#7F8C99")
            ax.spines["polar"].set_color("#D5DAE0")
            ax.spines["polar"].set_linewidth(1.0)

        def _redraw(*_):
            ax.clear()

            # original
            if _candidate_visible(-1):
                _draw_structure(
                    self.mol.coords, self.mol.elements, self.mol.bonds,
                    color=CAND_COLORS[0], alpha=0.55, lw=0.8, zorder=2, ms=22
                )

            # candidates
            for i, cand in enumerate(self.candidates[:5]):
                if not _candidate_visible(i):
                    continue

                color = CAND_COLORS[min(i + 1, len(CAND_COLORS) - 1)]

                alpha = max(0.35, 1.0 - i * 0.12)
                lw = 0.8
                ms = max(14, 28 - i * 2)
                zorder = 10 - i

                _draw_structure(
                    cand["coords_opt"], self.mol.elements, self.mol.bonds,
                    color=color, alpha=alpha, lw=lw, zorder=zorder, ms=ms
                )

            # metal marker
            ax.scatter(
                [0.0], [0.0],
                s=120,
                color="#3D4652",
                edgecolors="white",
                linewidths=1.1,
                zorder=50
            )

            all_rmax = [np.linalg.norm(self.mol.coords - metal_pos, axis=1).max()]
            all_rmax += [np.linalg.norm(c["coords_opt"] - metal_pos, axis=1).max() for c in self.candidates[:5]]
            r_max = max(all_rmax) * 1.08

            _configure_axes(r_max)

            handles = _legend_handles()
            if handles:
                leg = ax.legend(
                    handles=handles,
                    fontsize=8,
                    loc="upper right",
                    frameon=True,
                    fancybox=True,
                    framealpha=0.92,
                    borderpad=0.6,
                )
                leg.get_frame().set_edgecolor("#D9DEE5")
                leg.get_frame().set_linewidth(0.8)

            ax.set_title("Polar structure comparison", fontsize=11, pad=16)
            summary_var.set(_build_summary_text())

            canvas.draw_idle()

        def _save_plot():
            path = filedialog.asksaveasfilename(
                parent=win,
                title="Save structure plot",
                defaultextension=".png",
                filetypes=[
                    ("PNG image", "*.png"),
                    ("PDF file", "*.pdf"),
                    ("SVG file", "*.svg"),
                    ("All files", "*.*"),
                ],
            )
            if not path:
                return
            try:
                fig.savefig(path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
                messagebox.showinfo("Saved", f"Saved:\n{path}", parent=win)
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc), parent=win)

        # =========================
        # Right panel
        # =========================
        box_candidates = ttk.LabelFrame(right, text="Candidates", padding=8)
        box_candidates.pack(fill="x", pady=(0, 8))

        ttk.Checkbutton(box_candidates, text="Original", variable=v_original, command=_redraw).pack(anchor="w")
        ttk.Checkbutton(box_candidates, text="#1 best", variable=v_best, command=_redraw).pack(anchor="w")
        ttk.Checkbutton(box_candidates, text="#2", variable=v_2, command=_redraw).pack(anchor="w")
        ttk.Checkbutton(box_candidates, text="#3", variable=v_3, command=_redraw).pack(anchor="w")
        ttk.Checkbutton(box_candidates, text="#4", variable=v_4, command=_redraw).pack(anchor="w")
        ttk.Checkbutton(box_candidates, text="#5", variable=v_5, command=_redraw).pack(anchor="w")

        box_projection = ttk.LabelFrame(right, text="Projection", padding=8)
        box_projection.pack(fill="x", pady=(0, 8))

        ttk.Radiobutton(box_projection, text="0–180°", value="0-180", variable=proj_var, command=_redraw).pack(
            anchor="w")
        ttk.Radiobutton(box_projection, text="0–90°", value="0-90", variable=proj_var, command=_redraw).pack(anchor="w")

        box_display = ttk.LabelFrame(right, text="Display", padding=8)
        box_display.pack(fill="x", pady=(0, 8))

        ttk.Checkbutton(box_display, text="Show H", variable=v_show_h, command=_redraw).pack(anchor="w")
        ttk.Checkbutton(box_display, text="Show bonds", variable=v_show_bonds, command=_redraw).pack(anchor="w")
        ttk.Checkbutton(box_display, text="Show atom markers", variable=v_show_atoms, command=_redraw).pack(anchor="w")

        box_info = ttk.LabelFrame(right, text="RMSD summary", padding=8)
        box_info.pack(fill="x", pady=(0, 8))

        ttk.Label(
            box_info,
            textvariable=summary_var,
            justify="left",
            foreground="#444444"
        ).pack(anchor="w")

        ttk.Button(right, text="Save figure…", command=_save_plot).pack(fill="x", pady=(4, 0))

        _redraw()
        
    def _display_candidate_report(self, idx: int):
        """Render the report for candidate idx in the result text box."""
        cand = self.candidates[idx]
        self.result = cand

        report = format_report(
            cand, cand['selected_bonds'], self.mol,
            cand.get('locked_bonds'),
            rank=idx + 1, n_total=len(self.candidates),
        )
        self._result_box.config(state="normal")
        self._result_box.delete("1.0", "end")
        self._result_box.insert("end", report)
        self._result_box.config(state="disabled")

        fd = cand.get('fixed_displacement', 0.0)
        self._status_var.set(
            f"Showing candidate #{idx + 1}/{len(self.candidates)}  "
            f"RMSD={cand['rmsd']:.4f} ppm"
            + (f"  R²={cand['r2']:.4f}" if not math.isnan(cand['r2']) else "")
            + (f"  ⚠ {cand['n_clashes']} clashes" if cand['n_clashes'] else "")
            + (f"  ⚠ fixed drift {fd:.3f} Å" if fd > 0.01 else "")
        )

    def _save_xyz(self):
        if not getattr(self, 'candidates', None):
            messagebox.showwarning("No result", "Run the search first.")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Save optimised XYZ")
        dlg.resizable(False, False)
        ttk.Label(dlg, text="Select candidates to include in the multi-model XYZ file:",
                  wraplength=380).pack(padx=16, pady=(14, 6), anchor="w")

        chk_vars = []
        for i, cand in enumerate(self.candidates):
            var = tk.BooleanVar(value=(i == 0))
            clash_str = "  ⚠ clash" if cand['n_clashes'] else ""
            r2_str = f"  R²={cand['r2']:.4f}" if not math.isnan(cand['r2']) else ""
            lbl = f"#{i + 1}  RMSD {cand['rmsd']:.4f} ppm{r2_str}{clash_str}"
            ttk.Checkbutton(dlg, text=lbl, variable=var).pack(
                anchor="w", padx=24, pady=2)
            chk_vars.append(var)

        info_var = tk.StringVar(value="")
        ttk.Label(dlg, textvariable=info_var,
                  foreground="gray").pack(padx=16, pady=(4, 0), anchor="w")

        def _update_info(*_):
            n = sum(v.get() for v in chk_vars)
            info_var.set(f"{n} selected → {n}-frame XYZ")

        for v in chk_vars:
            v.trace_add("write", _update_info)
        _update_info()

        def _do_save():
            chosen = [self.candidates[i]
                      for i, v in enumerate(chk_vars) if v.get()]
            if not chosen:
                messagebox.showwarning("Nothing selected",
                                       "Check at least one candidate.", parent=dlg)
                return
            path = filedialog.asksaveasfilename(
                parent=dlg, title="Save multi-model XYZ",
                defaultextension=".xyz",
                filetypes=[("XYZ", "*.xyz"), ("All", "*.*")])
            if not path:
                return
            with open(path, "w") as f:
                for rank, cand in enumerate(chosen, 1):
                    comment = (f"Candidate #{rank} | "
                               f"RMSD={cand['rmsd']:.4f} ppm | "
                               f"R²={cand['r2']:.4f}")
                    f.write(f"{self.mol.n_atoms}\n{comment}\n")
                    for el, (x, y, z) in zip(self.mol.elements, cand['coords_opt']):
                        f.write(f"{el:<4s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")
            dlg.destroy()
            messagebox.showinfo("Saved", f"Saved {len(chosen)}-frame XYZ:\n{path}")

        btn_row = ttk.Frame(dlg)
        btn_row.pack(padx=16, pady=12, anchor="e")
        ttk.Button(btn_row, text="Cancel", command=dlg.destroy).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Save file…", command=_do_save).pack(side="left")

    def _save_report(self):
        if self.result is None:
            messagebox.showwarning("No result", "Run the search first."); return
        path = filedialog.asksaveasfilename(
            title="Save report", defaultextension=".txt",
            filetypes=[("Text","*.txt"),("All","*.*")])
        if not path: return
        rpt = format_report(self.result, self.result['selected_bonds'],
                            self.mol, self.result.get('locked_bonds'))
        with open(path, "w", encoding="utf-8") as f:
            f.write(rpt)
        messagebox.showinfo("Saved", f"Saved:\n{path}")

    def _show_plot(self):
        if self.result is None:
            messagebox.showwarning("No result", "Run the search first.")
            return

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        per = self.result['per_atom']
        exp = np.array([p[2] for p in per])
        pred = np.array([p[3] for p in per])
        res = np.array([p[4] for p in per])
        lbl = [f"{p[0] + 1}{p[1]}" for p in per]

        fig = plt.Figure(figsize=(10, 4.5), dpi=120)
        gs = GridSpec(1, 2, figure=fig, wspace=0.42)

        ax1 = fig.add_subplot(gs[0, 0])
        vl = min(exp.min(), pred.min()) * 1.12 - 0.05
        vh = max(exp.max(), pred.max()) * 1.12 + 0.05
        ax1.plot([vl, vh], [vl, vh], "k--", lw=0.8, alpha=0.35)
        vabs = max(abs(res)) if len(res) else 1.0
        sc = ax1.scatter(exp, pred, c=res, cmap="RdBu_r",
                         vmin=-vabs, vmax=vabs, s=60,
                         edgecolors="none", alpha=0.85, zorder=3)
        for i, t in enumerate(lbl):
            ax1.annotate(t, (exp[i], pred[i]),
                         xytext=(3, 3), textcoords="offset points",
                         fontsize=5, color="gray")
        fig.colorbar(sc, ax=ax1, label="Residual [ppm]", fraction=0.04)
        ax1.set_xlabel("δ_exp [ppm]")
        ax1.set_ylabel("δ_pred [ppm]")
        r2s = f"{self.result['r2']:.4f}" if not math.isnan(self.result['r2']) else "—"
        ax1.set_title(f"Correlation  R²={r2s}  RMSD={self.result['rmsd']:.4f} ppm", fontsize=9)
        ax1.set_aspect("equal", adjustable="box")

        ax2 = fig.add_subplot(gs[0, 1])
        colours = ["#2ca02c" if abs(r) <= 0.05 else
                   "#ff7f0e" if abs(r) <= 0.15 else "#d62728" for r in res]
        ax2.bar(range(len(res)), res, color=colours, alpha=0.82, edgecolor="none")
        ax2.axhline(0, color="gray", lw=0.8, ls="--")
        ax2.set_xticks(range(len(res)))
        ax2.set_xticklabels(lbl, rotation=45, ha="right", fontsize=6)
        ax2.set_ylabel("pred − exp [ppm]")
        ax2.set_title("Residuals", fontsize=9)

        fig.suptitle("Conformer search – PCS fit", fontsize=10)
        fig.tight_layout()

        win = tk.Toplevel(self.root)
        win.title("Correlation plot")
        win.geometry("980x520")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()

        canvas.draw_idle()

    def run(self):
        if self._standalone:
            self.root.mainloop()

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_conformer_search_gui(master=None, embed_in=None, **kwargs):
    app = ConformerSearchGUI(master, embed_in=embed_in, **kwargs)
    if master is None and embed_in is None:
        app.run()
    return app

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Conformational search to fit PCS data.")
    parser.add_argument("--xyz",        required=True)
    parser.add_argument("--pcs",        required=True)
    parser.add_argument("--metal",      type=int,   default=1)
    parser.add_argument("--fix",        type=int,   nargs="*", default=[],
                        help="1-based atom indices to fix (space-separated)")
    parser.add_argument("--dchi-ax",    type=float, default=-2.0)
    parser.add_argument("--dchi-rh",    type=float, default=0.0)
    parser.add_argument("--euler-zyz",  type=float, nargs=3, default=[0.,0.,0.],
                        metavar=("A","B","G"))
    parser.add_argument("--metal-xyz",  type=float, nargs=3, default=[0.,0.,0.],
                        metavar=("X","Y","Z"))
    parser.add_argument("--no-global",  action="store_true")
    parser.add_argument("--de-maxiter", type=int,   default=600)
    parser.add_argument("--de-popsize", type=int,   default=12)
    parser.add_argument("--out",        default="optimised.xyz")
    args = parser.parse_args()

    print(f"Loading: {args.xyz}")
    elements, coords = load_xyz(args.xyz)
    mol = Molecule(elements, coords)
    metal_idx  = args.metal - 1
    fixed_set  = {args.metal - 1} | {i - 1 for i in (args.fix or [])}
    print(f"  {mol.n_atoms} atoms | metal: {args.metal}({elements[metal_idx]})"
          f" | fixed: {sorted(fixed_set)}")

    candidates = mol.find_rotatable_bonds(metal_idx)
    free, locked = classify_bonds_by_fixed(mol, candidates, fixed_set,
                                            metal_idx=metal_idx, anchor_mode=True)
    print(f"  {len(candidates)} rotatable bonds: {len(free)} free, {len(locked)} locked")

    print(f"\nLoading PCS: {args.pcs}")
    atom_idx, pcs_vals = load_pcs_csv(args.pcs, elements)
    print(f"  {len(atom_idx)} observations")

    opt = ConformerOptimiser(
        mol=mol, rotatable=free,
        obs_atoms=atom_idx, obs_pcs=np.array(pcs_vals),
        metal=np.array(args.metal_xyz),
        dchi_ax=args.dchi_ax, dchi_rh=args.dchi_rh,
        euler_zyz=tuple(math.radians(a) for a in args.euler_zyz),
        metal_idx=metal_idx, fixed_atoms=fixed_set,
        progress_cb=print)
    result = opt.run(not args.no_global, args.de_maxiter, args.de_popsize)
    print(format_report(result, free, mol, locked))
    save_xyz(args.out, elements, result['coords_opt'])
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_main()
    else:
        run_conformer_search_gui()
