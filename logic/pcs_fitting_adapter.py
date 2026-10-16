"""Narrow compatibility adapter from fitting geometry arrays to the PCS engine."""

from __future__ import annotations

import numpy as np

from logic.pcs_forward_api import (
    PCSAtom, PCSPredictionRequest, PCSStructureTarget, PCSTensor,
)
from logic.pcs_forward_model import PCSForwardEngine


_PCS_FITTING_ENGINE = PCSForwardEngine()


def predict_pcs_from_factors(Gax, Grh, dchi_ax, dchi_rh=0.0):
    """Return legacy-shaped PCS values using caller-computed geometry factors.

    Fitting code retains ownership of its established coordinate rotations and
    effective-geometry averaging. The canonical engine owns tensor enable
    semantics, PCS scaling, and numerical evaluation.
    """
    gax_values, grh_values = np.broadcast_arrays(
        np.asarray(Gax, dtype=float),
        np.asarray(Grh, dtype=float),
    )
    shape = gax_values.shape
    flat_gax = gax_values.reshape(-1)
    flat_grh = grh_values.reshape(-1)
    atoms = tuple(
        PCSAtom(ref_id=index, position=(0.0, 0.0, 0.0))
        for index in range(len(flat_gax))
    )
    request = PCSPredictionRequest(
        tensor=PCSTensor(
            dchi_ax=float(dchi_ax),
            dchi_rh=float(dchi_rh),
            rhombic_enabled=True,
        ),
        atoms=atoms,
        request_id="fitting-compatibility",
    )
    prediction = _PCS_FITTING_ENGINE.predict_with_effective_geometry(
        request,
        {
            index: (float(flat_gax[index]), float(flat_grh[index]))
            for index in range(len(flat_gax))
        },
    )
    values = np.asarray(
        [item.pcs_ppm for item in prediction.atom_predictions],
        dtype=float,
    ).reshape(shape)
    return float(values) if values.ndim == 0 else values


def predict_structured_pcs_from_factors(
    ref_ids, Gax, Grh, dchi_ax, dchi_rh, targets
):
    """Return one engine-reduced prediction per observable target."""
    ids = tuple(int(value) for value in ref_ids)
    gax = np.asarray(Gax, dtype=float).reshape(-1)
    grh = np.asarray(Grh, dtype=float).reshape(-1)
    if len(ids) != len(gax) or len(ids) != len(grh):
        raise ValueError("structured fitting geometry arrays are inconsistent")
    target_values = tuple(targets)
    if any(not isinstance(item, PCSStructureTarget) for item in target_values):
        raise TypeError("targets must contain PCSStructureTarget values")
    request = PCSPredictionRequest(
        tensor=PCSTensor(
            dchi_ax=float(dchi_ax), dchi_rh=float(dchi_rh), rhombic_enabled=True,
        ),
        atoms=tuple(PCSAtom(ref_id=ref_id, position=(0.0, 0.0, 0.0)) for ref_id in ids),
        targets=target_values,
        request_id="fitting-structured",
    )
    result = _PCS_FITTING_ENGINE.predict_with_effective_geometry(
        request,
        {ref_id: (float(gax[i]), float(grh[i])) for i, ref_id in enumerate(ids)},
    )
    return np.asarray([item.pcs_ppm for item in result.target_predictions], dtype=float)


__all__ = ["predict_pcs_from_factors", "predict_structured_pcs_from_factors"]
