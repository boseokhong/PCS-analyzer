"""Immutable VT observations and ordinary least squares, without UI or plotting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Iterable

import numpy as np

from nmr_workbench.core import NMRSeries, PeakTrack, TrackProvenance


@dataclass(frozen=True)
class VTPointIdentity:
    series_id: str
    track_id: str
    member_id: str
    dataset_id: str
    peak_id: str | None


@dataclass(frozen=True)
class VTDiagnostic:
    code: str
    message: str
    identities: tuple[VTPointIdentity, ...] = ()


@dataclass(frozen=True)
class VTObservation:
    identity: VTPointIdentity
    acquisition_order: int
    member_label: str | None
    member_included: bool
    point_state: str
    point_reason: str | None
    provenance: TrackProvenance | None
    temperature_k: float | None
    temperature_source: str | None
    temperature_status: str
    chemical_shift_ppm: float | None
    intensity: float | None
    linewidth_hz: float | None
    shift_uncertainty_ppm: float | None
    uncertainty_observable_id: str | None
    dataset_revision: int | None
    linked_dataset_revision: int | None
    peak_fingerprint: str | None
    history_fingerprint: str | None
    link_valid: bool
    diagnostics: tuple[VTDiagnostic, ...]


@dataclass(frozen=True)
class VTTrackData:
    series_id: str
    track_id: str
    track_label: str | None
    track_provenance: TrackProvenance
    observations: tuple[VTObservation, ...]
    diagnostics: tuple[VTDiagnostic, ...]

    def to_dict(self):
        return {"schema_version": 1, **asdict(self)}


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def extract_vt_track(series: NMRSeries, track: PeakTrack) -> VTTrackData:
    """Read live peak values and validity without updating/reconfirming the track.

    Every member (including unreviewed members) and orphan point is represented.
    Missing/invalid numeric values become null plus diagnostics for safe export.
    """
    if series.id != track.series_id:
        raise ValueError("track belongs to another series")
    issues = track.validate(series)  # pure; do not call update_stale/to_dict on series
    rows = []
    members = {m.id: m for m in series.members}
    entries = [(m, track.get_point(m.id)) for m in series.members]
    entries.extend((None, p) for p in track.points if p.member_id not in members)
    for member, point in entries:
        identity = VTPointIdentity(series.id, track.id, member.id if member else point.member_id,
            point.dataset_id if point else member.dataset_id, point.peak_id if point else None)
        diagnostics = []

        def diagnostic(code, message):
            diagnostics.append(VTDiagnostic(code, message, (identity,)))

        state = point.state if point else "unreviewed"
        if state != "confirmed":
            diagnostic(state, f"Track point is {state}; excluded from numerical fitting")
        broken = [i.code for i in issues if i.member_id == identity.member_id and i.code != "member_excluded"]
        for code in broken:
            if code != state:
                diagnostic(code, f"Linked observation failed validation: {code}")
        if member is None:
            temperature, source, temperature_status = None, None, "missing"
            dataset, included = None, False
        else:
            resolved = member.resolve_temperature(series_override_k=series.temperature_override_k)
            temperature, source, temperature_status = resolved.temperature_k, resolved.source, resolved.status
            dataset, included = member.dataset, member.included
        if not included:
            diagnostic("excluded_member", "Member is excluded or no longer belongs to the series")
        if temperature_status != "valid":
            diagnostic(f"{temperature_status}_temperature", "Effective temperature must be finite positive Kelvin")
        peaks = [p for p in dataset.peaks if p.id == identity.peak_id] if dataset and dataset.id == identity.dataset_id else []
        peak = peaks[0] if len(peaks) == 1 else None
        shift = intensity = linewidth = uncertainty = None
        observable_id = None
        if peak:
            shift = _finite(peak.position_ppm)
            if shift is None:
                diagnostic("nonfinite_shift", "Observed chemical shift is not a finite real ppm value")
            for name, value in (("intensity", peak.height), ("linewidth", peak.linewidth_hz)):
                number = _finite(value)
                if value is not None and (number is None or (name == "linewidth" and number <= 0)):
                    diagnostic(f"invalid_{name}", f"{name} is invalid/nonfinite")
                    number = None
                if name == "intensity":
                    intensity = number
                else:
                    linewidth = number
            linked = [o for o in dataset.observables if o.source_peak_id == peak.id]
            if len(linked) > 1:
                diagnostic("ambiguous_uncertainty", "Multiple observables reference this peak; no uncertainty selected")
            elif linked and linked[0].uncertainty_ppm is not None:
                observable = linked[0]
                if observable.dataset_id != dataset.id or shift is None or _finite(observable.chemical_shift_ppm) != shift:
                    diagnostic("uncertainty_out_of_sync", "Observable identity/shift does not agree with the source peak")
                else:
                    uncertainty = _finite(observable.uncertainty_ppm)
                    if uncertainty is None or uncertainty < 0:
                        uncertainty = None
                        diagnostic("invalid_uncertainty", "Shift uncertainty must be finite nonnegative ppm")
                    else:
                        observable_id = observable.id
        rows.append(VTObservation(identity, member.acquisition_order if member else point.acquisition_order,
            member.label if member else None, included, state, point.reason if point else None,
            point.provenance if point else None, temperature, source, temperature_status,
            shift, intensity, linewidth, uncertainty, observable_id,
            dataset.spectral_revision if dataset and dataset.id == identity.dataset_id else None,
            point.dataset_revision if point else None,
            point.peak_fingerprint if point else None, point.history_fingerprint if point else None,
            state == "confirmed" and not broken and peak is not None, tuple(diagnostics)))
    diagnostics = [d for row in rows for d in row.diagnostics]
    groups = {}
    for row in rows:
        if row.temperature_k is not None:
            groups.setdefault(row.temperature_k, []).append(row.identity)
    for temperature, identities in groups.items():
        if len(identities) > 1:
            diagnostics.append(VTDiagnostic("duplicate_temperature", f"Repeated effective temperature {temperature:g} K; observations retained", tuple(identities)))
    return VTTrackData(series.id, track.id, track.label, track.provenance, tuple(rows), tuple(diagnostics))


@dataclass(frozen=True)
class VTFitDomain:
    name: str
    x_quantity: str
    x_unit: str
    y_quantity: str
    y_unit: str
    weighting: str = "unweighted"


_DOMAINS = {
    "delta_vs_T": VTFitDomain("delta_vs_T", "temperature", "K", "observed_chemical_shift", "ppm"),
    "delta_vs_inverse_T": VTFitDomain("delta_vs_inverse_T", "inverse_temperature", "K^-1", "observed_chemical_shift", "ppm"),
    "deltaT_vs_inverse_T": VTFitDomain("deltaT_vs_inverse_T", "inverse_temperature", "K^-1", "observed_chemical_shift_times_temperature", "ppm K"),
    "intensity_vs_T": VTFitDomain("intensity_vs_T", "temperature", "K", "peak_height", "a.u."),
    "linewidth_vs_T": VTFitDomain("linewidth_vs_T", "temperature", "K", "linewidth", "Hz"),
}


@dataclass(frozen=True)
class VTCoordinate:
    observation: VTObservation
    x: float | None
    y: float | None
    y_uncertainty: float | None
    included: bool
    exclusion_reasons: tuple[str, ...]


@dataclass(frozen=True)
class VTCoordinates:
    data: VTTrackData
    domain: VTFitDomain
    points: tuple[VTCoordinate, ...]
    temperature_range_k: tuple[float, float] | None
    diagnostics: tuple[VTDiagnostic, ...]

    def to_dict(self):
        return {"schema_version": 1, **asdict(self)}


def prepare_vt_coordinates(data: VTTrackData, *, domain="delta_vs_T",
        excluded_member_ids: Iterable[str] = (), temperature_range_k=None) -> VTCoordinates:
    """Analysis-local exclusions/range never override invalid correspondence."""
    if domain not in _DOMAINS:
        raise ValueError("unknown VT coordinate domain")
    fit_domain = _DOMAINS[domain]
    excluded = frozenset(excluded_member_ids)
    unknown = excluded - {o.identity.member_id for o in data.observations}
    if unknown:
        raise ValueError("unknown analysis exclusion member IDs")
    if temperature_range_k is not None:
        if len(temperature_range_k) != 2:
            raise ValueError("temperature range must contain lower/upper Kelvin bounds")
        low, high = (_finite(v) for v in temperature_range_k)
        if low is None or high is None or low <= 0 or high < low:
            raise ValueError("invalid temperature range in Kelvin")
        temperature_range_k = (low, high)
    diagnostics = list(data.diagnostics)
    points = []
    for row in data.observations:
        reasons = []
        if not row.link_valid:
            reasons.append("invalid_track_point")
        if not row.member_included:
            reasons.append("excluded_member")
        temperature = row.temperature_k
        if temperature is None or _finite(temperature) is None or temperature <= 0:
            reasons.append("invalid_temperature")
            temperature = None
        if row.identity.member_id in excluded:
            reasons.append("analysis_excluded")
        if temperature is not None and temperature_range_k and not temperature_range_k[0] <= temperature <= temperature_range_k[1]:
            reasons.append("outside_temperature_range")
        value = row.intensity if domain == "intensity_vs_T" else row.linewidth_hz if domain == "linewidth_vs_T" else row.chemical_shift_ppm
        if value is None or _finite(value) is None:
            reasons.append("unavailable_observable")
            diagnostics.append(VTDiagnostic("unavailable_observable", f"Requested {fit_domain.y_quantity} is unavailable/nonfinite", (row.identity,)))
            value = None
        x = temperature if fit_domain.x_quantity == "temperature" else (1 / temperature if temperature is not None else None)
        y = value
        sigma = row.shift_uncertainty_ppm if domain.startswith("delta") else None
        if domain == "deltaT_vs_inverse_T":
            y = value * temperature if value is not None and temperature is not None else None
            sigma = sigma * temperature if sigma is not None and temperature is not None else None
        if (x is not None and not math.isfinite(x)) or (y is not None and not math.isfinite(y)):
            reasons.append("nonfinite_coordinate")
            diagnostics.append(VTDiagnostic("nonfinite_coordinate", "Coordinate transformation overflowed", (row.identity,)))
            x = x if x is None or math.isfinite(x) else None
            y = y if y is None or math.isfinite(y) else None
        if sigma is not None and not math.isfinite(sigma):
            sigma = None
            diagnostics.append(VTDiagnostic("invalid_uncertainty", "Transformed uncertainty overflowed", (row.identity,)))
        points.append(VTCoordinate(row, x, y, sigma, not reasons, tuple(reasons)))
    return VTCoordinates(data, fit_domain, tuple(points), temperature_range_k, tuple(diagnostics))


@dataclass(frozen=True)
class VTLinearFitResult:
    status: str
    domain: VTFitDomain
    coordinates: VTCoordinates
    slope: float | None
    intercept: float | None
    fitted_values: tuple[float, ...]
    residuals: tuple[float, ...]
    r_squared: float | None
    point_count: int
    included_identities: tuple[VTPointIdentity, ...]
    excluded_identities: tuple[VTPointIdentity, ...]
    x_range: tuple[float, float] | None
    diagnostics: tuple[VTDiagnostic, ...]

    def to_dict(self):
        return {"schema_version": 1, **asdict(self)}


def fit_vt_linear(coordinates: VTCoordinates) -> VTLinearFitResult:
    """Fit y = slope*x + intercept; residuals are observed minus fitted.

    Results align with included_identities. Repeated x values are never collapsed.
    Constant response has undefined R² (null); degenerate x cannot be fitted.
    """
    included = tuple(p for p in coordinates.points if p.included)
    identities = tuple(p.observation.identity for p in included)
    excluded = tuple(p.observation.identity for p in coordinates.points if not p.included)
    diagnostics = list(coordinates.diagnostics)
    slope = intercept = r_squared = None
    fitted = residuals = ()
    x_range = None
    status = "invalid"
    if len(included) < 2:
        diagnostics.append(VTDiagnostic("insufficient_points", "Linear fitting requires at least two valid observations", identities))
    else:
        x = np.asarray([p.x for p in included], dtype=float)
        y = np.asarray([p.y for p in included], dtype=float)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            diagnostics.append(VTDiagnostic("nonfinite_fit_data", "Fit inputs must be finite", identities))
        elif np.ptp(x) == 0:
            diagnostics.append(VTDiagnostic("degenerate_x", "Linear fitting requires distinct temperature coordinates", identities))
        else:
            # Center and scale x to avoid conditioning problems for inverse T.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                x_mean, y_mean = float(x.mean()), float(y.mean())
                scale = float(np.max(np.abs(x - x_mean)))
                normalized = (x - x_mean) / scale
                coefficient = float(np.dot(normalized, y - y_mean) / np.dot(normalized, normalized))
                slope = coefficient / scale
                intercept = y_mean - slope * x_mean
                predicted = y_mean + coefficient * normalized
                residual = y - predicted
                total = float(np.dot(y - y_mean, y - y_mean))
                error = float(np.dot(residual, residual))
            if not all(math.isfinite(v) for v in (slope, intercept, total, error)) or not np.all(np.isfinite(predicted)):
                slope = intercept = None
                diagnostics.append(VTDiagnostic("numerical_failure", "Linear regression overflowed; rescale inputs", identities))
            else:
                status = "ok"
                fitted, residuals = tuple(map(float, predicted)), tuple(map(float, residual))
                x_range = (float(x.min()), float(x.max()))
                if total == 0:
                    diagnostics.append(VTDiagnostic("constant_response", "R² is undefined for a constant response", identities))
                else:
                    r_squared = 1 - error / total
                if len(included) == 2:
                    diagnostics.append(VTDiagnostic("two_point_fit", "Two points leave no residual degrees of freedom", identities))
    if any(p.y_uncertainty is not None for p in included):
        diagnostics.append(VTDiagnostic("uncertainty_not_weighted", "Observed uncertainties are preserved; regression is unweighted", identities))
    return VTLinearFitResult(status, coordinates.domain, coordinates, slope, intercept,
        fitted, residuals, r_squared, len(included), identities, excluded, x_range, tuple(diagnostics))
