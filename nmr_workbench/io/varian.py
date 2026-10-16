"""Agilent/Varian VNMR 1D raw-data adapter backed by nmrglue."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import importlib
from pathlib import Path
import re
from typing import Any, Mapping, Protocol

import numpy as np

from nmr_workbench.core import ExperimentalConditions, NMRDataset, Spectrum1D


class VarianReadError(RuntimeError):
    """Raised when a path cannot be converted into a Varian 1D dataset."""


class _VarianBackend(Protocol):
    def read_raw(self, path: Path) -> tuple[Mapping[str, Any], np.ndarray]: ...


@dataclass
class _NmrglueBackend:
    """Keep nmrglue and its dictionaries behind the adapter boundary."""

    def read_raw(self, path: Path) -> tuple[Mapping[str, Any], np.ndarray]:
        try:
            varian = importlib.import_module("nmrglue").varian
        except ImportError as exc:
            raise VarianReadError(
                "Varian support requires the 'nmrglue' runtime dependency"
            ) from exc
        return varian.read(str(path))


class Varian1DReader:
    """Read an Agilent/Varian VNMR raw ``.fid`` directory."""

    def __init__(self, backend: _VarianBackend | None = None) -> None:
        self._backend = backend or _NmrglueBackend()

    def read(self, path: str | Path, *, source: str = "raw") -> NMRDataset:
        dataset_path = Path(path).expanduser()
        if not dataset_path.is_dir():
            raise VarianReadError(f"Varian dataset directory not found: {dataset_path}")
        if source.lower().strip() not in {"auto", "raw"}:
            raise VarianReadError("Varian reader supports raw 1D data only")
        if not (dataset_path / "fid").is_file() or not (dataset_path / "procpar").is_file():
            raise VarianReadError(
                f"Malformed Varian dataset in {dataset_path}; expected 'fid' and 'procpar'"
            )
        try:
            metadata, values = self._backend.read_raw(dataset_path)
            data = np.asarray(values)
            if data.ndim != 1:
                raise VarianReadError(
                    f"only Varian 1D data are supported; received shape {data.shape}"
                )
            data = data.copy()
            procpar = _procpar_section(metadata)
            acqdim = _as_int(_procpar_value(procpar, "acqdim"))
            if acqdim is not None and acqdim != 1:
                raise VarianReadError(
                    f"only Varian 1D data are supported; procpar acqdim is {acqdim}"
                )
            conditions, acquisition_parameters, vendor_metadata = map_varian_metadata(
                procpar, metadata
            )
            acquisition_parameters["time_domain_points"] = int(data.size)
            sw = acquisition_parameters.get("spectral_width_hz")
            if sw is not None and float(sw) > 0:
                axis = np.arange(data.size, dtype=float) / float(sw)
                axis_unit = "s"
            else:
                axis = np.arange(data.size, dtype=float)
                axis_unit = "point"
        except VarianReadError:
            raise
        except Exception as exc:
            raise VarianReadError(f"Failed to read Varian raw data: {exc}") from exc

        return NMRDataset(
            conditions=conditions,
            raw_spectrum=Spectrum1D(axis, data, axis_unit, "time", "raw"),
            acquisition_parameters=acquisition_parameters,
            vendor_metadata=vendor_metadata,
            source_path=dataset_path.resolve(),
            vendor="Varian/Agilent",
        )


def map_varian_metadata(
    procpar: Mapping[str, Any], file_header: Mapping[str, Any] | None = None
) -> tuple[ExperimentalConditions, dict[str, Any], dict[str, Any]]:
    """Map VNMR procpar values without exposing nmrglue's raw dictionary.

    VNMR ``temp`` is degrees Celsius.  Absolute ppm referencing follows VNMR's
    reference-line convention: the right-edge ppm value is ``(rfp-rfl)/reffrq``.
    With the processor's centered ascending frequency offsets, carrier ppm is
    therefore ``(sw/2 + rfp - rfl) / reffrq``.  ``tof`` is retained as the
    transmitter offset but is not used to construct the displayed ppm axis.
    """

    get = lambda name: _procpar_value(procpar, name)
    spectral_width_hz = _positive_float(get("sw"))
    observed_frequency_mhz = _positive_float(get("sfrq"))
    reference_frequency_mhz = _positive_float(get("reffrq"))
    rfl_hz = _as_float(get("rfl"))
    rfp_hz = _as_float(get("rfp"))
    carrier_ppm = None
    if (
        spectral_width_hz is not None
        and reference_frequency_mhz is not None
        and rfl_hz is not None
        and rfp_hz is not None
    ):
        carrier_ppm = (
            spectral_width_hz / 2.0 + rfp_hz - rfl_hz
        ) / reference_frequency_mhz

    temp_c = _as_float(get("temp"))
    conditions = ExperimentalConditions(
        nucleus=_normalize_nucleus(_clean_text(get("tn"))),
        solvent=_clean_text(get("solvent")),
        frequency_mhz=observed_frequency_mhz,
        temperature_k=temp_c + 273.15 if temp_c is not None else None,
        acquisition_date=_varian_datetime(get("time_run"), get("date")),
    )
    header = file_header or {}
    complex_points = _as_int(header.get("np"))
    if complex_points is not None:
        complex_points //= 2
    acquisition_parameters = _drop_unknown(
        {
            "spectral_width_hz": spectral_width_hz,
            "dwell_time_s": 1.0 / spectral_width_hz if spectral_width_hz else None,
            "time_domain_points": complex_points,
            "vendor_real_point_count": _as_int(get("np")),
            "acquisition_time_s": _as_float(get("at")),
            "observed_nucleus": _normalize_nucleus(_clean_text(get("tn"))),
            "decoupler_nucleus": _normalize_nucleus(_clean_text(get("dn"))),
            "decoupler_frequency_mhz": _positive_float(get("dfrq")),
            "transmitter_offset_hz": _as_float(get("tof")),
            "reference_line_position_hz": rfl_hz,
            "reference_peak_frequency_hz": rfp_hz,
            "ppm_reference_frequency_mhz": reference_frequency_mhz,
            "carrier_ppm": carrier_ppm,
            "scan_count": _as_int(get("nt")),
            "steady_state_scan_count": _as_int(get("ss")),
            "receiver_gain": _as_float(get("gain")),
            "pulse_program": _clean_text(get("seqfil")),
            "oversampling_factor": _as_float(get("oversamp")),
            "receiver_filter_bandwidth_hz": _as_float(get("filter")),
            # nmrglue preserves the VNMR complex channel convention. A
            # positive-exponent DFT maps its decays onto positive physical
            # frequency offsets; the processor records this choice in history.
            "fourier_transform_exponent_sign": "positive",
        }
    )

    preserved_names = (
        "tn", "dn", "sfrq", "reffrq", "dfrq", "sw", "np", "tof", "rfl", "rfp",
        "solvent", "temp", "nt", "ss", "gain", "seqfil", "date", "time_run",
        "oversamp", "filter", "at", "arraydim", "acqdim", "fn", "samplename", "file",
    )
    preserved = {
        name: _json_value(get(name))
        for name in preserved_names
        if get(name) is not None
    }
    vendor_metadata = {
        "format": "Agilent/Varian VNMR",
        "procpar": preserved,
    }
    if temp_c is not None:
        vendor_metadata["temperature"] = {"value": temp_c, "unit": "degC"}
    return conditions, acquisition_parameters, vendor_metadata


def _procpar_section(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    section = metadata.get("procpar")
    if not isinstance(section, Mapping):
        raise VarianReadError("Varian metadata does not contain a procpar mapping")
    return section


def _procpar_value(procpar: Mapping[str, Any], name: str) -> Any:
    entry = procpar.get(name)
    if isinstance(entry, Mapping):
        values = entry.get("values")
        if isinstance(values, (list, tuple)):
            if len(values) == 1:
                return values[0]
            return list(values)
    return entry


def _normalize_nucleus(value: str | None) -> str | None:
    if value is None:
        return None
    compact = value.replace("^", "").replace(" ", "")
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", compact)
    if match:
        return f"{match.group(2)}{match.group(1).capitalize()}"
    match = re.fullmatch(r"(\d+)([A-Za-z]+)", compact)
    if match:
        return f"{match.group(1)}{match.group(2).capitalize()}"
    return compact


def _varian_datetime(time_run: Any, date: Any) -> datetime | None:
    text = _clean_text(time_run)
    if text:
        try:
            return datetime.strptime(text, "%Y%m%dT%H%M%S")
        except ValueError:
            pass
    text = _clean_text(date)
    if text:
        try:
            return datetime.strptime(text, "%b %d %Y")
        except ValueError:
            pass
    return None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("<>").strip()
    return text or None


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _positive_float(value: Any) -> float | None:
    number = _as_float(value)
    return number if number is not None and number > 0 else None


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _drop_unknown(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _json_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
