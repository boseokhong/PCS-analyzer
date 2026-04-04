# tools/logic_structure_helpers.py
"""
Utility functions for molecular structure rendering.

Provides CPK colour lookup and covalent-radius-based bond detection.
Intended to be used by PyVista viewer code; no GUI imports here.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from logic.chem_constants import CPK_COLORS, covalent_radii


# ---------------------------------------------------------------------------
# Colour / radius helpers
# ---------------------------------------------------------------------------

def get_cpk_color(atom_label: str) -> str:
    """
    Return the CPK colour string for the given element label.

    Tries two-character match first (e.g. "Fe", "Cu"), then one-character.
    Falls back to the default colour if no match is found.
    """
    if not atom_label:
        return CPK_COLORS["default"]
    if len(atom_label) >= 2 and atom_label[:2] in CPK_COLORS:
        return CPK_COLORS[atom_label[:2]]
    return CPK_COLORS.get(atom_label[0], CPK_COLORS["default"])


def radius_for_element(el: str) -> float:
    """
    Return a display radius for the given element.

    Scales the covalent radius by 0.35 to produce visually reasonable
    sphere sizes in CPK/ball-and-stick representations.
    """
    return float(covalent_radii.get(el, 0.80)) * 0.35


# ---------------------------------------------------------------------------
# Bond detection
# ---------------------------------------------------------------------------

def calculate_bonds(
    atom_coords: np.ndarray,
    atom_elements: list[str],
    scale: float = 1.05,
) -> list[tuple[int, int]]:
    """
    Detect covalent bonds by distance thresholding.

    Two atoms i and j are considered bonded if their distance is within
    ``scale`` × (cov_radius_i + cov_radius_j).

    Parameters
    ----------
    atom_coords :
        Array of shape (N, 3) with atomic coordinates.
    atom_elements :
        List of N element symbols.
    scale :
        Tolerance factor on top of the sum of covalent radii.
        Default 1.10 (10 % tolerance).

    Returns
    -------
    List of (i, j) index pairs with i < j.
    """
    n = len(atom_coords)
    if n < 2:
        return []

    distances = squareform(pdist(atom_coords))
    bonds: list[tuple[int, int]] = []

    for i in range(n):
        ri = covalent_radii.get(atom_elements[i], 0.0)
        for j in range(i + 1, n):
            rj = covalent_radii.get(atom_elements[j], 0.0)
            threshold = (ri + rj) * scale
            if threshold > 0 and distances[i, j] <= threshold:
                bonds.append((i, j))

    return bonds