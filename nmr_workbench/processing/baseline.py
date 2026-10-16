"""Baseline estimation services for 1D NMR spectra.

The Bernstein implementation intentionally normalizes the spectral coordinate to
[0, 1] before constructing the Bernstein basis.  It fits a robust lower-envelope
representation of the spectrum so that positive NMR peaks do not dominate the
baseline estimate.

This module keeps the public API expected by SpectrumProcessor:
    BaselineCorrectionService.fit(...)
"""

from __future__ import annotations

from math import comb
from typing import Sequence

import numpy as np

try:
    from pybaselines import Baseline as _PyBaseline
except Exception:  # pragma: no cover - handled explicitly at runtime
    _PyBaseline = None


class BaselineError(RuntimeError):
    """Raised when baseline estimation fails or receives invalid input."""


class _PyBaselinesWhittakerBackend:
    """Adapt pybaselines to the small backend contract used by this module."""

    def __init__(self) -> None:
        if _PyBaseline is None:
            raise BaselineError(
                "pybaselines is required for Whittaker/AsLS/arPLS methods"
            )
        self._fitter = _PyBaseline()

    def fit_asls(
        self,
        axis,
        values,
        *,
        lam: float,
        p: float,
        diff_order: int,
        weights,
    ) -> np.ndarray:
        baseline, _ = self._fitter.asls(
            np.asarray(values, dtype=float),
            lam=float(lam),
            p=float(p),
            diff_order=int(diff_order),
            weights=weights,
        )
        return np.asarray(baseline, dtype=float)

    def fit_arpls(
        self,
        axis,
        values,
        *,
        lam: float,
        diff_order: int,
        weights,
    ) -> np.ndarray:
        baseline, _ = self._fitter.arpls(
            np.asarray(values, dtype=float),
            lam=float(lam),
            diff_order=int(diff_order),
            weights=weights,
        )
        return np.asarray(baseline, dtype=float)


def _as_finite_1d(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise BaselineError(f"{name} must be one-dimensional")
    if array.size < 2:
        raise BaselineError(f"{name} must contain at least two points")
    if not np.all(np.isfinite(array)):
        raise BaselineError(f"{name} contains non-finite values")
    return array


def _region_mask(
    axis: np.ndarray,
    regions: Sequence[tuple[float, float]] | None,
) -> np.ndarray:
    if regions is None:
        return np.ones(axis.size, dtype=bool)

    mask = np.zeros(axis.size, dtype=bool)
    for pair in regions:
        if len(pair) != 2:
            raise BaselineError("each baseline region must contain two coordinates")
        left, right = map(float, pair)
        lo, hi = sorted((left, right))
        mask |= (axis >= lo) & (axis <= hi)

    if np.count_nonzero(mask) < 2:
        raise BaselineError("baseline regions contain too few points")
    return mask


def _normalize_axis(axis: np.ndarray) -> np.ndarray:
    lo = float(np.min(axis))
    hi = float(np.max(axis))
    span = hi - lo
    if not np.isfinite(span) or span <= 0.0:
        raise BaselineError("baseline axis must span a finite non-zero range")
    return (axis - lo) / span


def _bernstein_matrix(x01: np.ndarray, order: int) -> np.ndarray:
    """Return an N x (order + 1) Bernstein design matrix.

    The number of columns depends only on polynomial order.  This avoids the
    accidental N x N allocations that can occur when basis order is confused
    with spectrum length.
    """
    x = np.asarray(x01, dtype=float)
    n = int(order)
    columns = [
        float(comb(n, k)) * np.power(x, k) * np.power(1.0 - x, n - k)
        for k in range(n + 1)
    ]
    return np.column_stack(columns)


def _bin_lower_envelope(
    x01: np.ndarray,
    y: np.ndarray,
    *,
    max_bins: int = 512,
    quantile: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    """Compress dense spectra to lower-envelope support points.

    Positive NMR resonances are sparse/high relative to the baseline.  A low
    within-bin quantile supplies stable baseline support without requiring an
    N x N matrix or treating peak tops as baseline observations.
    """
    if x01.size <= 256:
        return x01, y

    bins = min(max_bins, max(32, x01.size // 128))
    edges = np.linspace(0.0, 1.0, bins + 1)

    xs = []
    ys = []
    for i in range(bins):
        if i == bins - 1:
            take = (x01 >= edges[i]) & (x01 <= edges[i + 1])
        else:
            take = (x01 >= edges[i]) & (x01 < edges[i + 1])
        if not np.any(take):
            continue
        xb = x01[take]
        yb = y[take]
        xs.append(float(np.median(xb)))
        ys.append(float(np.quantile(yb, quantile)))

    if len(xs) < 2:
        return x01, y
    return np.asarray(xs), np.asarray(ys)


def _fit_bernstein(
    axis: np.ndarray,
    signal: np.ndarray,
    *,
    order: int,
    regions: Sequence[tuple[float, float]] | None,
) -> np.ndarray:
    if isinstance(order, bool):
        raise BaselineError("Bernstein order must be an integer")
    try:
        order = int(order)
    except (TypeError, ValueError) as exc:
        raise BaselineError("Bernstein order must be an integer") from exc
    if order < 0 or order > 64:
        raise BaselineError("Bernstein order must be between 0 and 64")

    x_all = _normalize_axis(axis)
    mask = _region_mask(axis, regions)
    x_fit = x_all[mask]
    y_fit = signal[mask]

    # When explicit baseline regions are provided, respect them directly.
    # Otherwise obtain robust lower-envelope support points first.
    if regions is None:
        x_fit, y_fit = _bin_lower_envelope(x_fit, y_fit)

    if x_fit.size < order + 1:
        raise BaselineError(
            f"Bernstein order {order} requires at least {order + 1} fit points"
        )

    design_fit = _bernstein_matrix(x_fit, order)

    # Robust IRLS: residuals above the fitted baseline are likely NMR peaks and
    # are down-weighted strongly; negative residuals keep full influence.
    weights = np.ones(y_fit.size, dtype=float)
    coeff = np.zeros(order + 1, dtype=float)

    scale_floor = max(
        np.finfo(float).eps,
        1e-12 * max(float(np.max(np.abs(y_fit))), 1.0),
    )

    for _ in range(12):
        sqrt_w = np.sqrt(np.clip(weights, 1e-6, 1.0))
        aw = design_fit * sqrt_w[:, None]
        bw = y_fit * sqrt_w
        coeff_new, *_ = np.linalg.lstsq(aw, bw, rcond=None)

        fitted = design_fit @ coeff_new
        residual = y_fit - fitted
        negative = residual[residual < 0.0]
        if negative.size:
            scale = 1.4826 * np.median(np.abs(negative - np.median(negative)))
        else:
            scale = np.median(np.abs(residual - np.median(residual)))
        scale = max(float(scale), scale_floor)

        # Keep points on/below the current baseline influential and smoothly
        # suppress positive peak excursions.
        new_weights = np.ones_like(weights)
        positive = residual > 0.0
        z = residual[positive] / (3.0 * scale)
        new_weights[positive] = 1.0 / (1.0 + z * z)

        if np.allclose(coeff_new, coeff, rtol=1e-7, atol=1e-12):
            coeff = coeff_new
            break
        coeff = coeff_new
        weights = new_weights

    # Evaluation matrix is only N x (order + 1), not N x N.
    baseline = _bernstein_matrix(x_all, order) @ coeff
    if baseline.shape != signal.shape or not np.all(np.isfinite(baseline)):
        raise BaselineError("Bernstein baseline fit produced an invalid result")
    return baseline


class BaselineCorrectionService:
    """Baseline estimator used by the deterministic processing layer.

    ``backend`` is an optional small Whittaker backend adapter. Keeping this
    injection point preserves deterministic adapter testing and allows future
    backend substitution without coupling SpectrumProcessor to pybaselines.
    """

    def __init__(self, backend=None) -> None:
        self.backend = backend

    def fit(
        self,
        axis,
        signal,
        *,
        method: str,
        order: int = 8,
        lam: float = 1.0e6,
        p: float = 0.001,
        diff_order: int = 2,
        regions: Sequence[tuple[float, float]] | None = None,
    ) -> np.ndarray:
        x = _as_finite_1d(axis, "axis")
        y = _as_finite_1d(signal, "signal")
        if x.shape != y.shape:
            raise BaselineError("axis and signal must have the same shape")

        normalized = str(method or "").strip().lower()

        if normalized == "bernstein":
            return _fit_bernstein(
                x,
                y,
                order=order,
                regions=regions,
            )

        if normalized in {"whittaker", "asls", "arpls"}:
            try:
                lam = float(lam)
                diff_order = int(diff_order)
            except (TypeError, ValueError) as exc:
                raise BaselineError("invalid baseline parameters") from exc
            if not np.isfinite(lam) or lam <= 0.0:
                raise BaselineError("lambda must be finite and positive")

            # Preserve the original small backend contract:
            #   fit_asls(axis, values, ..., weights)
            #   fit_arpls(axis, values, ..., weights)
            #
            # Explicit baseline regions become fixed starting weights. With no
            # regions, ``None`` is passed exactly as expected by the adapter
            # tests and by pybaselines.
            weights = None
            if regions is not None:
                weights = _region_mask(x, regions).astype(float)

            fitter = (
                self.backend
                if self.backend is not None
                else _PyBaselinesWhittakerBackend()
            )

            try:
                if normalized == "arpls":
                    baseline = fitter.fit_arpls(
                        x,
                        y,
                        lam=lam,
                        diff_order=diff_order,
                        weights=weights,
                    )
                else:
                    p = float(p)
                    if not np.isfinite(p) or not 0.0 < p < 1.0:
                        raise BaselineError("p must lie strictly between 0 and 1")
                    baseline = fitter.fit_asls(
                        x,
                        y,
                        lam=lam,
                        p=p,
                        diff_order=diff_order,
                        weights=weights,
                    )
            except BaselineError:
                raise
            except Exception as exc:
                raise BaselineError(str(exc)) from exc

            baseline = np.asarray(baseline, dtype=float)
            if baseline.shape != y.shape or not np.all(np.isfinite(baseline)):
                raise BaselineError("baseline fit produced an invalid result")
            return baseline

        raise BaselineError(f"unsupported baseline method: {method}")
