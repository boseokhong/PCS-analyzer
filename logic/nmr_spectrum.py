# logic/nmr_spectrum.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any, Tuple

import numpy as np

Lineshape = Literal["lorentzian", "gaussian"]

def _lorentzian(x: np.ndarray, x0: float, fwhm: float) -> np.ndarray:
    # L(x) = (1/pi) * (0.5*gamma)/((x-x0)^2 + (0.5*gamma)^2)
    g = max(float(fwhm), 1e-12)
    return (0.5 * g) / ((x - x0) ** 2 + (0.5 * g) ** 2)

def _gaussian(x: np.ndarray, x0: float, fwhm: float) -> np.ndarray:
    # Convert FWHM -> sigma: FWHM = 2*sqrt(2*ln2)*sigma
    f = max(float(fwhm), 1e-12)
    sigma = f / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)

def simulate_envelope(
    shifts_ppm: np.ndarray,
    intensities: np.ndarray,
    x_range: Tuple[float, float],
    npts: int = 8000,
    fwhm: float = 0.2,
    kind: Lineshape = "lorentzian",
) -> Tuple[np.ndarray, np.ndarray]:
    shifts_ppm = np.asarray(shifts_ppm, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    x_min, x_max = float(x_range[0]), float(x_range[1])
    x = np.linspace(x_min, x_max, int(npts))
    y = np.zeros_like(x)

    if kind == "lorentzian":
        fn = _lorentzian
    elif kind == "gaussian":
        fn = _gaussian
    else:
        raise ValueError(f"Unknown lineshape kind: {kind}")

    for d, I in zip(shifts_ppm, intensities):
        y += I * fn(x, float(d), float(fwhm))

    return x, y

def make_payload(
    shifts_ppm: np.ndarray,
    intensities: Optional[np.ndarray] = None,
    *,
    show_envelope: bool = False,
    x_range: Optional[Tuple[float, float]] = None,
    npts: int = 8000,
    fwhm: float = 0.2,
    kind: Lineshape = "lorentzian",
) -> Dict[str, Any]:
    shifts_ppm = np.asarray(shifts_ppm, dtype=float)
    if intensities is None:
        intensities = np.ones_like(shifts_ppm, dtype=float)
    else:
        intensities = np.asarray(intensities, dtype=float)

    if x_range is None:
        # auto range with padding
        lo = float(np.min(shifts_ppm)) - 5.0
        hi = float(np.max(shifts_ppm)) + 5.0
        x_range = (lo, hi)

    envelope = None
    if show_envelope:
        x, y = simulate_envelope(
            shifts_ppm=shifts_ppm,
            intensities=intensities,
            x_range=x_range,
            npts=npts,
            fwhm=fwhm,
            kind=kind,
        )
        envelope = {"x": x, "y": y}

    return {
        "sticks": {"x": shifts_ppm, "h": intensities},
        "envelope": envelope,
        "meta": {
            "invert_xaxis": True,
            "x_range": x_range,
            "lineshape": kind,
            "fwhm": fwhm,
        },
    }