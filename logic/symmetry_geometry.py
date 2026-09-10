# logic/symmetry_geometry.py

"""Geometry-factor helpers for symmetry-averaged pseudo atoms.

Pseudo-atom Cartesian coordinates are retained as representative positions for
visualisation.  PCS geometry factors, however, are averaged over the original
symmetry-equivalent member atoms because G is nonlinear in position.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def geom_factors_ax_rh(coords, metal=(0.0, 0.0, 0.0)):
    """Return r, theta, phi, Gax and Grh for coordinates in the tensor frame."""
    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, empty
    coords = np.atleast_2d(coords)
    metal = np.asarray(metal, dtype=float)

    vecs = coords - metal
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    r = np.linalg.norm(vecs, axis=1)
    r_safe = np.where(r == 0.0, np.inf, r)

    theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    phi = np.arctan2(y, x)
    Gax = (3.0 * np.cos(theta) ** 2 - 1.0) / (r_safe ** 3)
    Grh = 1.5 * np.sin(theta) ** 2 * np.cos(2.0 * phi) / (r_safe ** 3)
    return r, theta, phi, Gax, Grh


def effective_geometry_factors(
    ref_ids: Sequence[int],
    effective_coords,
    metal=(0.0, 0.0, 0.0),
    *,
    pseudo_members: Mapping[int, Sequence[int]] | None = None,
    raw_coords_by_id: Mapping[int, Sequence[float]] | None = None,
):
    """Return geometry factors for effective atoms.

    For ordinary atoms, factors are calculated from ``effective_coords``.
    For a pseudo atom with member metadata, Gax and Grh are replaced by the
    arithmetic mean of the factors calculated at the original member
    positions.  r/theta/phi remain those of the pseudo coordinate because they
    describe its representative display position rather than an averaged PCS
    geometry.
    """
    ref_ids = list(ref_ids)
    coords = np.asarray(effective_coords, dtype=float)
    r, theta, phi, Gax, Grh = geom_factors_ax_rh(coords, metal)

    if not pseudo_members or not raw_coords_by_id:
        return r, theta, phi, Gax, Grh

    Gax = np.array(Gax, dtype=float, copy=True)
    Grh = np.array(Grh, dtype=float, copy=True)

    for i, rid in enumerate(ref_ids):
        members = pseudo_members.get(rid)
        if not members:
            continue

        member_coords = []
        for member_id in members:
            coord = raw_coords_by_id.get(member_id)
            if coord is None:
                member_coords = []
                break
            member_coords.append(coord)

        if not member_coords:
            continue

        _, _, _, member_gax, member_grh = geom_factors_ax_rh(
            np.asarray(member_coords, dtype=float), metal
        )
        Gax[i] = float(np.mean(member_gax))
        Grh[i] = float(np.mean(member_grh))

    return r, theta, phi, Gax, Grh
