"""Pure display preparation of independent native spectra for a VT series."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from nmr_workbench.core import NMRSeries
from .view_model import dataset_display_name


@dataclass(frozen=True)
class SeriesTrace:
    series_id: str
    member_id: str
    dataset_id: str
    axis_ppm: np.ndarray
    display_values: np.ndarray
    offset: float
    label: str


@dataclass(frozen=True)
class PreparedSeriesTraces:
    traces: tuple[SeriesTrace, ...]
    unavailable: tuple[str, ...]


def prepare_series_traces(
    series: NMRSeries, *, order: str = "Acquisition", spacing: float = 1.2,
    normalize: bool = True,
) -> PreparedSeriesTraces:
    """First member is highest. Offset spacing is a multiple of display range.

    Axis arrays are borrowed, never modified. Every displayed intensity array
    is separate from dataset data. No replay, interpolation, resampling or
    mutation of dataset conditions/history takes place here.
    """
    if isinstance(spacing, bool) or not math.isfinite(spacing) or spacing < 0:
        raise ValueError("Spacing must be a finite non-negative number.")
    if order == "Acquisition":
        members = tuple(m for m in series.members if m.included)
    elif order in {"Temperature ↑", "Temperature ↓"}:
        members = series.temperature_sorted_members(
            descending=order == "Temperature ↓", included_only=True)
    else:
        raise ValueError("Unknown trace order.")
    available, unavailable = [], []
    for member in members:
        spectrum = member.dataset.processed_spectrum
        name = member.label or dataset_display_name(member.dataset)
        if spectrum is None or spectrum.domain != "frequency" or spectrum.axis_unit.lower() != "ppm":
            unavailable.append(f"{name}: no committed ppm spectrum")
            continue
        values = np.real(spectrum.intensities).astype(float, copy=True)
        finite = np.isfinite(spectrum.axis) & np.isfinite(values)
        if not finite.any():
            unavailable.append(f"{name}: no finite ppm/intensity points")
            continue
        if normalize:
            amplitude = float(np.max(np.abs(values[finite])))
            if amplitude > 0:
                values /= amplitude
        temperature = member.resolve_temperature(series_override_k=series.temperature_override_k)
        label = f"{name} · {temperature.temperature_k:g} K" if temperature.temperature_k is not None else f"{name} · temperature {temperature.status}"
        available.append((member, spectrum.axis, values, finite, label))
    # One scalar range determines layout, never a common spectrum/grid.
    if available:
        low = min(float(np.min(values[finite])) for _, _, values, finite, _ in available)
        high = max(float(np.max(values[finite])) for _, _, values, finite, _ in available)
        height = high - low or max(abs(low), abs(high)) or 1.0
    else:
        height = 1.0
    traces = []
    for index, (member, axis, values, finite, label) in enumerate(available):
        offset = (len(available) - index - 1) * spacing * height
        values += offset  # This array is already a display-only copy.
        traces.append(SeriesTrace(series.id, member.id, member.dataset_id,
                                  axis, values, offset, label))
    return PreparedSeriesTraces(tuple(traces), tuple(unavailable))
