"""Automatic phase estimation behind a small nmrglue adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class AutoPhaseError(RuntimeError):
    """Raised when automatic phase estimation cannot produce valid phases."""


@dataclass(frozen=True)
class AutoPhaseEstimate:
    """Phase values expressed in NMR Workbench's pivot convention."""

    p0_deg: float
    p1_deg: float
    pivot_index: float
    algorithm: str
    diagnostics: tuple[str, ...] = ()


class _AutoPhaseBackend(Protocol):
    def estimate(
        self, values: NDArray[np.complexfloating], *, algorithm: str
    ) -> tuple[float, float]: ...


class _NmrglueAutoPhaseBackend:
    def estimate(
        self, values: NDArray[np.complexfloating], *, algorithm: str
    ) -> tuple[float, float]:
        try:
            autophase = importlib.import_module("nmrglue").proc_autophase
        except ImportError as exc:
            raise AutoPhaseError("Auto Phase requires the 'nmrglue' dependency") from exc
        from scipy.optimize import minimize
        # Retain nmrglue's scientific objectives, but avoid a single zero start.
        data = np.asarray(values, dtype=complex)
        scale = float(np.max(np.abs(data)))
        if not np.isfinite(scale) or scale <= 0:
            raise AutoPhaseError("Auto Phase requires a finite nonzero signal")
        data = data / scale
        def scorer(samples):
            def score(phases):
                try:
                    if algorithm == "acme":
                        value = autophase._ps_acme_score(phases, samples)
                    else:
                        ramp = phases[0] + phases[1] * np.arange(samples.size) / samples.size
                        index = int(np.argmax(np.real(samples * np.exp(1j * np.deg2rad(ramp)))))
                        width = min(100, index, samples.size - index)
                        if width < 1:
                            return np.inf
                        value = autophase._ps_peak_minima_score(phases, samples, width)
                    return float(value) if np.isfinite(value) else np.inf
                except (ValueError, FloatingPointError):
                    return np.inf
            return score
        def optimize(score, p0, p1):
            return minimize(score, (p0, p1), method="Nelder-Mead",
                options={"maxiter": 1200, "maxfev": 2400, "xatol": 1e-5, "fatol": 1e-9,
                         "initial_simplex": np.array([[p0, p1], [p0 + 10., p1], [p0, p1 + 20.]])})
        def valid(fit):
            return fit.success and np.all(np.isfinite(fit.x)) and np.isfinite(fit.fun)
        try:
            # Cheap searches seed both P0 and P1. Final ranking/refinement always
            # uses the original full-resolution data, not a thinned spectrum.
            stride = max(1, int(np.ceil(data.size / 32768)))
            coarse = data[::stride]
            strongest = int(np.argmax(np.abs(coarse)))
            seed = -float(np.angle(coarse[strongest], deg=True))
            score = scorer(coarse)
            starts = [(seed - p1 * strongest / coarse.size, p1)
                      for p1 in (-1440., -720., 0., 720., 1440.)]
            starts.extend([(0., 0.), (seed - 90., 0.), (seed + 90., 0.)])
            seeds = []
            for p0, p1 in starts:
                fit = optimize(score, p0, p1)
                if valid(fit):
                    # A coarse j coordinate represents original stride*j.
                    phase = fit.x.copy()
                    phase[1] *= data.size / (coarse.size * stride)
                    if not any(np.allclose(phase, old, atol=.05, rtol=0.) for old in seeds):
                        seeds.append(phase)
            full_score = scorer(data)
            seeds.sort(key=full_score)
            candidates = [optimize(full_score, *seed) for seed in seeds[:2]]
            candidates = [fit for fit in candidates if valid(fit)]
            if not candidates:
                raise AutoPhaseError("Auto Phase did not converge to a finite solution; review manually")
            phases = min(candidates, key=lambda fit: fit.fun).x
            phases[0] = (phases[0] + 180.) % 360. - 180.
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise AutoPhaseError(f"nmrglue Auto Phase failed: {exc}") from exc
        return float(phases[0]), float(phases[1])




class AutoPhaseService:
    """Estimate ACME or peak-minima phases and convert their convention."""

    def __init__(
        self,
        backend: _AutoPhaseBackend | None = None,
        *,
        max_estimation_points: int | None = None,
    ) -> None:
        self._backend = backend or _NmrglueAutoPhaseBackend()
        self.max_estimation_points = None if max_estimation_points is None else int(max_estimation_points)
        if self.max_estimation_points is not None and self.max_estimation_points < 2:
            raise ValueError("max_estimation_points must be at least 2")

    def estimate(
        self,
        values: NDArray[np.number],
        *,
        algorithm: str = "acme",
        pivot_index: float,
    ) -> AutoPhaseEstimate:
        data = np.asarray(values)
        if data.ndim != 1 or data.size < 2 or not np.iscomplexobj(data):
            raise AutoPhaseError(
                "Auto Phase requires a one-dimensional complex spectrum"
            )
        if not np.all(np.isfinite(data)):
            raise AutoPhaseError("Auto Phase input must contain only finite values")
        normalized = algorithm.strip().lower().replace("-", "_")
        if normalized not in {"acme", "peak_minima"}:
            raise AutoPhaseError("Auto Phase algorithm must be ACME or peak-minima")
        pivot = float(pivot_index)
        if not np.isfinite(pivot) or not 0 <= pivot <= data.size - 1:
            raise AutoPhaseError("Auto Phase pivot lies outside the spectrum")

        if self.max_estimation_points is not None and data.size > self.max_estimation_points:
            indices = np.linspace(
                0, data.size - 1, self.max_estimation_points, dtype=np.int64
            )
            estimation_data = data[indices]
        else:
            estimation_data = data
        try:
            nmrglue_p0, nmrglue_p1 = self._backend.estimate(
                np.asarray(estimation_data, dtype=complex), algorithm=normalized
            )
        except AutoPhaseError:
            raise
        except Exception as exc:
            raise AutoPhaseError(f"Auto Phase backend failed: {exc}") from exc
        if not np.isfinite(nmrglue_p0) or not np.isfinite(nmrglue_p1):
            raise AutoPhaseError("Auto Phase returned non-finite phase values")

        # nmrglue applies p0 + p1*j/M. Workbench applies phase at an explicit
        # pivot plus a total P1 change over the full first-to-last axis span.
        point_count = estimation_data.size
        total_p1 = nmrglue_p1 * (point_count - 1) / point_count
        pivot_fraction = pivot / (data.size - 1)
        pivot_p0 = (nmrglue_p0 + total_p1 * pivot_fraction + 180.) % 360. - 180.
        diagnostics = []
        if normalized == "peak_minima":
            diagnostics.append("Local peak-minima objective: review other resonances")
        if abs(total_p1) > 360.:
            diagnostics.append("Large first-order phase (>360 degrees): review full spectral range")
        if estimation_data.size != data.size:
            diagnostics.append("Explicit estimation point limit: narrow peaks may be undersampled")
        return AutoPhaseEstimate(pivot_p0, total_p1, pivot, normalized, tuple(diagnostics))


def rebase_phase_pivot(p0_deg, p1_deg, old_index, new_index, point_count):
    """Change pivot coordinates without changing the phase at any data point."""
    values = np.asarray([p0_deg, p1_deg, old_index, new_index], dtype=float)
    if not np.all(np.isfinite(values)) or point_count < 2:
        raise AutoPhaseError("Phase pivot conversion requires finite parameters and at least two points")
    if not (0 <= old_index <= point_count - 1 and 0 <= new_index <= point_count - 1):
        raise AutoPhaseError("Phase pivot lies outside the spectrum")
    return float(p0_deg + p1_deg * (new_index - old_index) / (point_count - 1))
