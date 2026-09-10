# logic/symmetry_geometry.py
# PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE3

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
    torsion_groups: Sequence | None = None,
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

    # PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE3
    # Build raw-atom G maps once.  When torsional averaging is active, factors
    # are averaged over sampled conformers in the *current tensor frame*.
    raw_gax_by_id = {}
    raw_grh_by_id = {}
    if raw_coords_by_id:
        raw_ids = [int(rid) for rid in raw_coords_by_id.keys()]
        raw_coords = np.asarray([raw_coords_by_id[rid] for rid in raw_coords_by_id.keys()], dtype=float)
        if torsion_groups:
            from logic.torsional_ensemble import ensemble_geometry_factors_independent
            raw_gax, raw_grh = ensemble_geometry_factors_independent(
                raw_coords, raw_ids, torsion_groups, metal=metal
            )
        else:
            _, _, _, raw_gax, raw_grh = geom_factors_ax_rh(raw_coords, metal)
        raw_gax_by_id = {rid: float(v) for rid, v in zip(raw_ids, raw_gax)}
        raw_grh_by_id = {rid: float(v) for rid, v in zip(raw_ids, raw_grh)}

    Gax = np.array(Gax, dtype=float, copy=True)
    Grh = np.array(Grh, dtype=float, copy=True)

    for i, rid in enumerate(ref_ids):
        members = (pseudo_members or {}).get(rid)
        if members:
            vals_ax = [raw_gax_by_id[m] for m in members if m in raw_gax_by_id]
            vals_rh = [raw_grh_by_id[m] for m in members if m in raw_grh_by_id]
            if len(vals_ax) == len(members) and vals_ax:
                Gax[i] = float(np.mean(vals_ax))
                Grh[i] = float(np.mean(vals_rh))
        elif rid in raw_gax_by_id and torsion_groups:
            # Ordinary atom affected by a torsional ensemble.
            Gax[i] = raw_gax_by_id[rid]
            Grh[i] = raw_grh_by_id[rid]

    return r, theta, phi, Gax, Grh
