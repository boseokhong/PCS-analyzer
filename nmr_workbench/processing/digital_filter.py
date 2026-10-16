"""Isolated Bruker digital-filter correction backed by nmrglue."""

from __future__ import annotations

import importlib
from typing import Any, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from nmr_workbench.core import Spectrum1D


class DigitalFilterError(RuntimeError):
    """Raised when Bruker digital-filter correction cannot be applied."""


class _DigitalFilterBackend(Protocol):
    def correct(
        self,
        data: NDArray[np.number],
        *,
        decimation: int,
        firmware_version: int,
        group_delay: float,
        truncate: bool,
    ) -> NDArray[np.number]: ...


class _NmrglueDigitalFilterBackend:
    def correct(
        self,
        data: NDArray[np.number],
        *,
        decimation: int,
        firmware_version: int,
        group_delay: float,
        truncate: bool,
    ) -> NDArray[np.number]:
        try:
            bruker = importlib.import_module("nmrglue").bruker
        except ImportError as exc:
            raise DigitalFilterError(
                "Bruker digital-filter correction requires 'nmrglue'"
            ) from exc
        return np.asarray(
            bruker.rm_dig_filter(
                data,
                decim=decimation,
                dspfvs=firmware_version,
                grpdly=group_delay,
                truncate_grpdly=truncate,
            )
        )


class BrukerDigitalFilter:
    """Validate and apply nmrglue's established Bruker correction routine."""

    def __init__(self, backend: _DigitalFilterBackend | None = None) -> None:
        self._backend = backend or _NmrglueDigitalFilterBackend()

    def correct(
        self,
        spectrum: Spectrum1D,
        acquisition_parameters: Mapping[str, Any],
        *,
        vendor: str | None,
        truncate: bool = True,
    ) -> Spectrum1D:
        """Return a corrected time-domain spectrum without mutating the input."""

        if spectrum.domain != "time":
            raise DigitalFilterError("digital-filter correction requires time-domain data")
        if spectrum.state != "raw":
            raise DigitalFilterError("digital-filter correction requires raw acquisition data")
        if (vendor or "").strip().lower() != "bruker":
            raise DigitalFilterError("digital-filter correction is only defined for Bruker data")
        decimation = _required_positive_int(acquisition_parameters, "digital_filter_decimation")
        firmware = _required_positive_int(
            acquisition_parameters, "digital_filter_firmware_version"
        )
        try:
            group_delay = float(acquisition_parameters.get("digital_filter_group_delay", 0.0))
        except (TypeError, ValueError) as exc:
            raise DigitalFilterError(
                "digital_filter_group_delay must be numeric"
            ) from exc
        if not np.isfinite(group_delay) or group_delay < 0:
            raise DigitalFilterError("digital_filter_group_delay must be finite and non-negative")

        try:
            corrected = np.asarray(
                self._backend.correct(
                    np.array(spectrum.intensities, copy=True),
                    decimation=decimation,
                    firmware_version=firmware,
                    group_delay=group_delay,
                    truncate=bool(truncate),
                )
            ).squeeze()
        except DigitalFilterError:
            raise
        except (TypeError, ValueError, RuntimeError, OSError) as exc:
            raise DigitalFilterError(f"nmrglue digital-filter correction failed: {exc}") from exc
        if corrected.ndim != 1:
            raise DigitalFilterError("digital-filter backend returned non-1D data")
        if corrected.size == 0:
            raise DigitalFilterError("digital-filter backend returned an empty FID")

        dwell = _dwell_time(spectrum, acquisition_parameters)
        if dwell is None:
            axis = np.arange(corrected.size, dtype=float)
            unit = "point"
        else:
            axis = np.arange(corrected.size, dtype=float) * dwell
            unit = "s"
        return Spectrum1D(axis, corrected.copy(), unit, "time", spectrum.state)


def _required_positive_int(parameters: Mapping[str, Any], key: str) -> int:
    try:
        value = int(parameters[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise DigitalFilterError(f"missing or invalid acquisition parameter: {key}") from exc
    if value <= 0:
        raise DigitalFilterError(f"{key} must be positive")
    return value


def _dwell_time(
    spectrum: Spectrum1D, acquisition_parameters: Mapping[str, Any]
) -> float | None:
    value = acquisition_parameters.get("dwell_time_s")
    try:
        dwell = float(value)
        if np.isfinite(dwell) and dwell > 0:
            return dwell
    except (TypeError, ValueError):
        pass
    if spectrum.axis_unit == "s" and spectrum.point_count > 1:
        differences = np.diff(spectrum.axis)
        if np.allclose(differences, differences[0]) and differences[0] > 0:
            return float(differences[0])
    return None
