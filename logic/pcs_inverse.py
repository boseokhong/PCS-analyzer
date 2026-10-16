"""UI-independent inverse geometry helpers for axial PCS contours."""

from __future__ import annotations

import numpy as np


def axial_pcs_radius(pcs_value, theta, dchi_ax):
    """Return axial PCS contour radii in Å using the legacy inverse equation."""
    denominator = 1.0e4 * dchi_ax * (3.0 * np.cos(theta) ** 2 - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = denominator / (12.0 * np.pi * pcs_value)
    fraction = np.where(fraction > 0.0, fraction, np.nan)
    radius = np.cbrt(fraction)
    radius[~np.isfinite(radius)] = np.nan
    return radius


__all__ = ["axial_pcs_radius"]
