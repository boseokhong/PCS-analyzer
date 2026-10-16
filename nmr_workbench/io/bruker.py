"""Bruker 1D adapter backed by nmrglue.

No nmrglue dictionary or object crosses the public reader boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from nmr_workbench.core import ExperimentalConditions, NMRDataset, Spectrum1D


class BrukerReadError(RuntimeError):
    """Raised when a path cannot be converted into a Bruker 1D dataset."""


class _BrukerBackend(Protocol):
    def read_raw(self, path: Path) -> tuple[Mapping[str, Any], np.ndarray]: ...

    def read_processed(self, path: Path) -> tuple[Mapping[str, Any], np.ndarray]: ...

    def ppm_axis(self, metadata: Mapping[str, Any], data: np.ndarray) -> np.ndarray: ...


@dataclass
class _NmrglueBackend:
    """Lazy nmrglue wrapper kept private to the vendor adapter."""

    def _bruker(self) -> Any:
        try:
            return importlib.import_module("nmrglue").bruker
        except ImportError as exc:
            raise BrukerReadError(
                "Bruker support requires the 'nmrglue' runtime dependency"
            ) from exc

    def read_raw(self, path: Path) -> tuple[Mapping[str, Any], np.ndarray]:
        return self._bruker().read(str(path))

    def read_processed(self, path: Path) -> tuple[Mapping[str, Any], np.ndarray]:
        bruker = self._bruker()
        metadata, data = bruker.read_pdata(str(path))
        experiment_dir = _experiment_directory(path)
        try:
            acqus = bruker.read_acqus_file(str(experiment_dir))
        except (OSError, IOError, ValueError):
            acqus = {}
        merged = dict(metadata)
        for key, value in acqus.items():
            merged.setdefault(key, value)
        return merged, data

    def ppm_axis(self, metadata: Mapping[str, Any], data: np.ndarray) -> np.ndarray:
        ng = importlib.import_module("nmrglue")
        udic = ng.bruker.guess_udic(metadata, data)
        return np.asarray(ng.fileiobase.uc_from_udic(udic).ppm_scale(), dtype=float)


class Bruker1DReader:
    """Read Bruker raw FIDs or processed 1D spectra into ``NMRDataset``."""

    def __init__(self, backend: _BrukerBackend | None = None) -> None:
        self._backend = backend or _NmrglueBackend()

    def read(self, path: str | Path, *, source: str = "auto") -> NMRDataset:
        """Load a Bruker experiment.

        ``source`` may be ``"raw"``, ``"processed"``, or ``"auto"``. In auto
        mode, a directory containing ``fid`` is treated as raw; a directory
        containing ``1r`` is treated as processed.
        """

        dataset_path = Path(path).expanduser()
        if not dataset_path.is_dir():
            raise BrukerReadError(f"Bruker dataset directory not found: {dataset_path}")
        resolved_source = self._resolve_source(dataset_path, source)
        try:
            if resolved_source == "raw":
                metadata, values = self._backend.read_raw(dataset_path)
                spectrum = self._raw_spectrum(metadata, values)
                raw_spectrum, processed_spectrum = spectrum, None
            else:
                metadata, values = self._backend.read_processed(dataset_path)
                spectrum = self._processed_spectrum(metadata, values)
                raw_spectrum, processed_spectrum = None, spectrum
        except BrukerReadError:
            raise
        except Exception as exc:
            raise BrukerReadError(f"Failed to read Bruker {resolved_source} data: {exc}") from exc

        conditions, acquisition_parameters = map_bruker_metadata(metadata)
        return NMRDataset(
            conditions=conditions,
            raw_spectrum=raw_spectrum,
            processed_spectrum=processed_spectrum,
            acquisition_parameters=acquisition_parameters,
            source_path=dataset_path.resolve(),
            vendor="Bruker",
        )

    @staticmethod
    def _resolve_source(path: Path, source: str) -> str:
        source = source.lower().strip()
        if source not in {"auto", "raw", "processed"}:
            raise ValueError("source must be 'auto', 'raw', or 'processed'")
        if source != "auto":
            return source
        if (path / "fid").is_file():
            return "raw"
        if (path / "1r").is_file():
            return "processed"
        raise BrukerReadError(
            f"Could not identify Bruker 1D data in {path}; expected 'fid' or '1r'"
        )

    def _raw_spectrum(self, metadata: Mapping[str, Any], values: np.ndarray) -> Spectrum1D:
        data = _as_1d(values)
        acqus = _section(metadata, "acqus")
        spectral_width_hz = _as_float(_value(acqus, "SW_h"))
        if spectral_width_hz is not None and spectral_width_hz > 0:
            axis = np.arange(data.size, dtype=float) / spectral_width_hz
            axis_unit = "s"
        else:
            axis = np.arange(data.size, dtype=float)
            axis_unit = "point"
        return Spectrum1D(axis, data, axis_unit, "time", "raw")

    def _processed_spectrum(
        self, metadata: Mapping[str, Any], values: np.ndarray
    ) -> Spectrum1D:
        data = _as_1d(values)
        try:
            axis = np.asarray(self._backend.ppm_axis(metadata, data), dtype=float)
        except Exception:
            axis = _fallback_ppm_axis(metadata, data.size)
        if axis.shape != data.shape:
            raise BrukerReadError("processed ppm axis length does not match spectrum")
        return Spectrum1D(axis, data, "ppm", "frequency", "processed")


def map_bruker_metadata(
    metadata: Mapping[str, Any],
) -> tuple[ExperimentalConditions, dict[str, Any]]:
    """Map reliable Bruker values to vendor-neutral domain metadata."""

    acqus = _section(metadata, "acqus")
    procs = _section(metadata, "procs")
    nucleus = _clean_text(_value(acqus, "NUC1"))
    if nucleus is not None:
        nucleus = nucleus.replace("<", "").replace(">", "").replace(" ", "")
    conditions = ExperimentalConditions(
        nucleus=nucleus,
        solvent=_clean_text(_value(acqus, "SOLVENT")),
        frequency_mhz=_as_float(_first(acqus, "SFO1", "BF1")),
        magnetic_field_t=_positive_float(_value(acqus, "B0")),
        temperature_k=_positive_float(_value(acqus, "TE")),
        acquisition_date=_bruker_datetime(_value(acqus, "DATE")),
    )

    spectral_width_hz = _positive_float(_value(acqus, "SW_h"))
    frequency_mhz = conditions.frequency_mhz
    transmitter_offset_hz = _as_float(_value(acqus, "O1"))

    acquisition_parameters = _drop_unknown(
        {
            "spectral_width_hz": spectral_width_hz,
            "dwell_time_s": 1.0 / spectral_width_hz if spectral_width_hz else None,
            "transmitter_offset_hz": transmitter_offset_hz,
            "ppm_reference_frequency_mhz": frequency_mhz,
            "carrier_ppm": (
                transmitter_offset_hz / frequency_mhz
                if transmitter_offset_hz is not None and frequency_mhz
                else None
            ),
            "time_domain_points": _as_int(_value(acqus, "TD")),
            "acquisition_time_s": _as_float(_value(acqus, "AQ")),
            "relaxation_delay_s": _array_index_float(_value(acqus, "D"), 1),
            "pulse_program": _clean_text(_value(acqus, "PULPROG")),
            "scan_count": _as_int(_value(acqus, "NS")),
            "receiver_gain": _as_float(_value(acqus, "RG")),
            "digital_filter_group_delay": _as_float(_value(acqus, "GRPDLY")),
            "digital_filter_decimation": _as_int(_value(acqus, "DECIM")),
            "digital_filter_firmware_version": _as_int(_value(acqus, "DSPFVS")),
            "processed_points": _as_int(_value(procs, "SI")),
            "processed_spectral_width_hz": _as_float(_value(procs, "SW_p")),
            "processed_frequency_mhz": _as_float(_value(procs, "SF")),
            "processed_offset_ppm": _as_float(_value(procs, "OFFSET")),
        }
    )
    return conditions, acquisition_parameters


def _as_1d(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values).squeeze()
    if data.ndim != 1:
        raise BrukerReadError(f"only 1D data are supported; received shape {data.shape}")
    return data.copy()


def _section(metadata: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for key, value in metadata.items():
        if str(key).lower() == name.lower() and isinstance(value, Mapping):
            return value
    return {}


def _value(section: Mapping[str, Any], name: str) -> Any:
    wanted = name.upper().lstrip("$")
    for key, value in section.items():
        if str(key).upper().lstrip("$") == wanted:
            return value
    return None


def _first(section: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = _value(section, name)
        if value is not None:
            return value
    return None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("<>").strip()
    return text or None


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _positive_float(value: Any) -> float | None:
    number = _as_float(value)
    return number if number is not None and number > 0 else None


def _as_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _array_index_float(value: Any, index: int) -> float | None:
    if isinstance(value, (list, tuple, np.ndarray)):
        return _as_float(value[index]) if len(value) > index else None
    return None


def _bruker_datetime(value: Any) -> datetime | None:
    timestamp = _as_float(value)
    if timestamp is None or timestamp <= 0:
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _drop_unknown(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _fallback_ppm_axis(metadata: Mapping[str, Any], size: int) -> np.ndarray:
    procs = _section(metadata, "procs")
    offset = _as_float(_value(procs, "OFFSET"))
    width_hz = _as_float(_value(procs, "SW_p"))
    frequency = _as_float(_value(procs, "SF"))
    if offset is None or width_hz is None or not frequency:
        raise BrukerReadError("processed data do not contain enough metadata for a ppm axis")
    return offset - np.arange(size, dtype=float) * (width_hz / frequency / size)


def _experiment_directory(processed_path: Path) -> Path:
    if processed_path.parent.name.lower() == "pdata":
        return processed_path.parent.parent
    return processed_path

