"""Experiment metadata that is independent of any NMR vendor format."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real


@dataclass
class ExperimentalConditions:
    """Physical and acquisition conditions for one NMR experiment.

    Temperatures use Kelvin. ``temperature_k`` records the nominal acquisition
    temperature, while ``calibrated_temperature_k`` is reserved for a measured
    or externally calibrated value. Missing values remain ``None``.
    """

    nucleus: str | None = None
    solvent: str | None = None
    frequency_mhz: float | None = None
    magnetic_field_t: float | None = None
    temperature_k: float | None = None
    calibrated_temperature_k: float | None = None
    reference_compound: str | None = None
    reference_ppm: float | None = None
    concentration_mol_l: float | None = None
    pressure_pa: float | None = None
    ph: float | None = None
    acquisition_date: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("temperature_k", "calibrated_temperature_k"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive Kelvin")
        if self.frequency_mhz is not None and self.frequency_mhz <= 0:
            raise ValueError("frequency_mhz must be positive")


@dataclass(frozen=True)
class TemperatureResolution:
    """Effective Kelvin value and its source, or a missing/invalid diagnostic."""

    temperature_k: float | None
    source: str | None
    status: str


def resolve_temperature(
    conditions: ExperimentalConditions | None,
    *,
    override_k: float | None = None,
    series_override_k: float | None = None,
) -> TemperatureResolution:
    """Resolve member override > series override > calibrated > nominal.

    Invalid higher-priority values do not silently fall back. This function
    never changes acquisition conditions or vendor metadata.
    """
    candidates = (
        ("member_override", override_k),
        ("series_override", series_override_k),
        ("calibrated", getattr(conditions, "calibrated_temperature_k", None)),
        ("nominal", getattr(conditions, "temperature_k", None)),
    )
    for source, value in candidates:
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            return TemperatureResolution(None, source, "invalid")
        return TemperatureResolution(float(value), source, "valid")
    return TemperatureResolution(None, None, "missing")

