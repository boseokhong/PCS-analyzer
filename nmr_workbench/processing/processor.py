"""Public deterministic processing API and history replay engine."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from nmr_workbench.core import NMRDataset, ProcessingHistory, ProcessingStep, Spectrum1D

from .baseline import BaselineCorrectionService, BaselineError
from .digital_filter import BrukerDigitalFilter, DigitalFilterError
from .phasing import AutoPhaseError, AutoPhaseEstimate, AutoPhaseService
from .result import ProcessingResult


class ProcessingError(RuntimeError):
    """Raised for invalid or incompatible spectrum-processing operations."""


class SpectrumProcessor:
    """Process 1D spectra without mutating raw input arrays.

    Frequency and ppm axes are ascending from negative to positive offset.
    Bruker/default input uses the negative-exponent DFT
    ``fftshift(fft(fid))``. VNMR complex input uses the positive-exponent DFT
    ``fftshift(ifft(fid)) * N``. The chosen transform is recorded explicitly so
    an older ``numpy_fft`` recipe retains its original meaning. Manual phase is
    ``p0 + p1 * (i-pivot)/(N-1)`` in degrees, so P1 is the phase change across
    one full spectral width.
    """

    DEFAULT_MAX_POINTS = 16_777_216

    def __init__(
        self,
        *,
        acquisition_parameters: Mapping[str, Any] | None = None,
        frequency_mhz: float | None = None,
        vendor: str | None = None,
        max_points: int = DEFAULT_MAX_POINTS,
        digital_filter: BrukerDigitalFilter | None = None,
        baseline_service: BaselineCorrectionService | None = None,
        auto_phase_service: AutoPhaseService | None = None,
    ) -> None:
        self.acquisition_parameters = dict(acquisition_parameters or {})
        self.frequency_mhz = float(frequency_mhz) if frequency_mhz is not None else None
        self.vendor = vendor
        self.max_points = int(max_points)
        if self.max_points < 1:
            raise ValueError("max_points must be positive")
        self.digital_filter_service = digital_filter or BrukerDigitalFilter()
        self.baseline_service = baseline_service or BaselineCorrectionService()
        self.auto_phase_service = auto_phase_service or AutoPhaseService()

    @classmethod
    def from_dataset(cls, dataset: NMRDataset, **kwargs: Any) -> "SpectrumProcessor":
        """Build a processor using a dataset's normalized acquisition context."""

        return cls(
            acquisition_parameters=dataset.acquisition_parameters,
            frequency_mhz=dataset.conditions.frequency_mhz,
            vendor=dataset.vendor,
            **kwargs,
        )

    def correct_digital_filter(
        self, source: Spectrum1D | ProcessingResult, *, truncate: bool = True
    ) -> ProcessingResult:
        spectrum, history = _unpack(source)
        try:
            corrected = self.digital_filter_service.correct(
                spectrum,
                self.acquisition_parameters,
                vendor=self.vendor,
                truncate=truncate,
            )
        except DigitalFilterError as exc:
            raise ProcessingError(str(exc)) from exc
        step = ProcessingStep(
            "digital_filter",
            "bruker_nmrglue",
            {
                "truncate": bool(truncate),
                "decimation": int(self.acquisition_parameters["digital_filter_decimation"]),
                "firmware_version": int(
                    self.acquisition_parameters["digital_filter_firmware_version"]
                ),
                "group_delay": float(
                    self.acquisition_parameters.get("digital_filter_group_delay", 0.0)
                ),
            },
        )
        return _with_step(corrected, history, step)

    def apodize(
        self,
        source: Spectrum1D | ProcessingResult,
        *,
        method: str,
        lb_hz: float | None = None,
        gb_hz: float | None = None,
    ) -> ProcessingResult:
        """Apply exponential or Gaussian time-domain apodization.

        Exponential uses ``exp(-pi * lb_hz * t)``. Gaussian uses
        ``exp(-(pi * gb_hz * t)^2 / (4 ln 2))``; ``gb_hz`` is the FWHM of the
        corresponding Gaussian broadening in the frequency domain.
        """

        spectrum, history = _unpack(source)
        _require_domain(spectrum, "time", "apodization")
        elapsed = _elapsed_seconds(spectrum, self.acquisition_parameters)
        normalized = method.strip().lower()
        if normalized == "exponential":
            width = _positive_finite(lb_hz, "lb_hz")
            window = np.exp(-np.pi * width * elapsed)
            parameters = {"lb_hz": width}
        elif normalized == "gaussian":
            width = _positive_finite(gb_hz, "gb_hz")
            window = np.exp(-((np.pi * width * elapsed) ** 2) / (4.0 * np.log(2.0)))
            parameters = {"gb_hz": width}
        else:
            raise ProcessingError(f"unsupported apodization method: {method}")
        output = _new_spectrum(spectrum, spectrum.axis, spectrum.intensities * window)
        return _with_step(output, history, ProcessingStep("apodization", normalized, parameters))

    def zero_fill(
        self, source: Spectrum1D | ProcessingResult, *, target_points: int
    ) -> ProcessingResult:
        """Pad a time-domain FID to an explicit 1D target length."""

        spectrum, history = _unpack(source)
        _require_domain(spectrum, "time", "zero filling")
        if isinstance(target_points, bool) or not isinstance(target_points, (int, np.integer)):
            raise ProcessingError("target_points must be an integer")
        target = int(target_points)
        if target < spectrum.point_count:
            raise ProcessingError("target_points cannot be smaller than the current spectrum")
        if target > self.max_points:
            raise ProcessingError(
                f"target_points exceeds the configured safety limit ({self.max_points})"
            )
        values = np.zeros(target, dtype=spectrum.intensities.dtype)
        values[: spectrum.point_count] = spectrum.intensities
        axis = _extended_time_axis(spectrum, target, self.acquisition_parameters)
        output = _new_spectrum(spectrum, axis, values)
        return _with_step(
            output,
            history,
            ProcessingStep("zero_fill", parameters={"target_points": target}),
        )

    def fourier_transform(
        self,
        source: Spectrum1D | ProcessingResult,
        *,
        exponent_sign: str | None = None,
        history_step: ProcessingStep | None = None,
    ) -> ProcessingResult:
        """Apply the acquisition's complex DFT onto ascending frequency bins."""

        spectrum, history = _unpack(source)
        _require_domain(spectrum, "time", "Fourier transformation")
        dwell = _dwell_seconds(spectrum, self.acquisition_parameters)
        sign = (
            exponent_sign
            or self.acquisition_parameters.get("fourier_transform_exponent_sign")
            or "negative"
        )
        if sign == "negative":
            transformed = np.fft.fftshift(np.fft.fft(spectrum.intensities))
            method = "numpy_fft"
        elif sign == "positive":
            transformed = (
                np.fft.fftshift(np.fft.ifft(spectrum.intensities))
                * spectrum.point_count
            )
            method = "numpy_fft_positive"
        else:
            raise ProcessingError(
                "fourier_transform_exponent_sign must be 'negative' or 'positive'"
            )
        frequencies_hz = np.fft.fftshift(np.fft.fftfreq(spectrum.point_count, d=dwell))
        carrier_ppm = self.acquisition_parameters.get("carrier_ppm")
        reference_mhz = self.acquisition_parameters.get("ppm_reference_frequency_mhz")
        if carrier_ppm is None and (self.vendor or "").strip().lower() in {"", "bruker"}:
            transmitter_hz = self.acquisition_parameters.get("transmitter_offset_hz")
            if self.frequency_mhz is not None and transmitter_hz is not None:
                carrier_ppm = float(transmitter_hz) / self.frequency_mhz
                reference_mhz = self.frequency_mhz
        if carrier_ppm is not None and reference_mhz is not None:
            reference_mhz = float(reference_mhz)
            if reference_mhz <= 0:
                raise ProcessingError("ppm_reference_frequency_mhz must be positive")
            axis = float(carrier_ppm) + frequencies_hz / reference_mhz
            axis_unit = "ppm"
        else:
            axis = frequencies_hz
            axis_unit = "Hz"
        output = Spectrum1D(axis, transformed, axis_unit, "frequency", "processed")
        step = history_step or ProcessingStep(
            "fourier_transform",
            method,
            {
                "direction": "forward",
                "exponent_sign": sign,
                "shift": "fftshift",
                "ordering": "ascending",
            },
        )
        return _with_step(output, history, step)

    def phase(
        self,
        source: Spectrum1D | ProcessingResult,
        *,
        p0_deg: float = 0.0,
        p1_deg: float = 0.0,
        pivot_ppm: float | None = None,
        pivot_index: float | None = None,
    ) -> ProcessingResult:
        """Apply zero- and first-order phase in degrees around a chosen pivot."""

        spectrum, history = _unpack(source)
        _require_domain(spectrum, "frequency", "phase correction")
        p0 = _finite(p0_deg, "p0_deg")
        p1 = _finite(p1_deg, "p1_deg")
        if pivot_ppm is not None and pivot_index is not None:
            raise ProcessingError("specify either pivot_ppm or pivot_index, not both")
        if pivot_ppm is not None:
            if spectrum.axis_unit != "ppm":
                raise ProcessingError("pivot_ppm requires a ppm axis")
            pivot = ppm_to_fractional_index(spectrum, float(pivot_ppm))
            pivot_parameters: dict[str, float | str] = {
                "pivot_kind": "ppm",
                "pivot_ppm": float(pivot_ppm),
            }
        elif pivot_index is not None:
            pivot = float(pivot_index)
            if not np.isfinite(pivot) or not 0 <= pivot <= spectrum.point_count - 1:
                raise ProcessingError("pivot_index lies outside the spectrum")
            pivot_parameters = {"pivot_kind": "index", "pivot_index": pivot}
        else:
            pivot = (spectrum.point_count - 1) / 2.0
            pivot_parameters = {"pivot_kind": "index", "pivot_index": pivot}
        denominator = max(spectrum.point_count - 1, 1)
        phase_degrees = p0 + p1 * (np.arange(spectrum.point_count) - pivot) / denominator
        corrected = spectrum.intensities * np.exp(1j * np.deg2rad(phase_degrees))
        output = _new_spectrum(spectrum, spectrum.axis, corrected)
        parameters = {"p0_deg": p0, "p1_deg": p1, **pivot_parameters}
        return _with_step(output, history, ProcessingStep("phase", "linear", parameters))

    def auto_phase(
        self,
        source: Spectrum1D | ProcessingResult,
        *,
        algorithm: str = "acme",
        pivot_ppm: float | None = None,
        pivot_index: float | None = None,
    ) -> AutoPhaseEstimate:
        """Estimate preview-compatible P0/P1 values without changing history."""

        spectrum, _ = _unpack(source)
        _require_domain(spectrum, "frequency", "automatic phase correction")
        if not np.iscomplexobj(spectrum.intensities):
            raise ProcessingError("Auto Phase requires a complex spectrum")
        if pivot_ppm is not None and pivot_index is not None:
            raise ProcessingError("specify either pivot_ppm or pivot_index, not both")
        if pivot_ppm is not None:
            pivot = ppm_to_fractional_index(spectrum, float(pivot_ppm))
        elif pivot_index is not None:
            pivot = float(pivot_index)
        else:
            pivot = (spectrum.point_count - 1) / 2.0
        try:
            return self.auto_phase_service.estimate(
                spectrum.intensities, algorithm=algorithm, pivot_index=pivot
            )
        except AutoPhaseError as exc:
            raise ProcessingError(str(exc)) from exc

    def baseline_correct(
        self,
        source: Spectrum1D | ProcessingResult,
        *,
        method: str,
        order: int = 8,
        lam: float = 1.0e6,
        p: float = 0.001,
        diff_order: int = 2,
        regions: Sequence[tuple[float, float]] | None = None,
    ) -> ProcessingResult:
        """Fit and subtract a baseline from the real spectrum component."""

        spectrum, history = _unpack(source)
        _require_domain(spectrum, "frequency", "baseline correction")
        normalized = method.strip().lower()
        try:
            stable_regions = (
                None if regions is None else tuple(tuple(map(float, item)) for item in regions)
            )
        except (TypeError, ValueError) as exc:
            raise ProcessingError("baseline regions must contain numeric pairs") from exc
        try:
            baseline = self.baseline_service.fit(
                spectrum.axis,
                np.real(spectrum.intensities),
                method=normalized,
                order=order,
                lam=lam,
                p=p,
                diff_order=diff_order,
                regions=stable_regions,
            )
        except BaselineError as exc:
            raise ProcessingError(str(exc)) from exc
        corrected = np.real(spectrum.intensities) - baseline
        if np.iscomplexobj(spectrum.intensities):
            corrected = corrected + 1j * np.imag(spectrum.intensities)
        output = _new_spectrum(spectrum, spectrum.axis, corrected)
        if normalized in {"whittaker", "asls", "arpls"}:
            parameters: dict[str, Any] = {
                "lam": float(lam),
                "diff_order": int(diff_order),
                "regions": stable_regions,
            }
            if normalized == "arpls":
                recorded_method = "arpls"
            else:
                parameters["p"] = float(p)
                recorded_method = "whittaker_asls"
        else:
            parameters = {"order": int(order), "regions": stable_regions}
            recorded_method = "bernstein"
        return _with_step(
            output,
            history,
            ProcessingStep("baseline", recorded_method, parameters),
            baseline=baseline,
        )

    def reference(
        self,
        source: Spectrum1D | ProcessingResult,
        *,
        observed_ppm: float,
        target_ppm: float,
        source_type: str | None = None,
        preset_id: str | None = None,
        resonance_id: str | None = None,
        source_id: str | None = None,
    ) -> ProcessingResult:
        """Translate a ppm axis so ``observed_ppm`` becomes ``target_ppm``.

        The resolved numeric target is always serialized. Optional provenance
        fields describe how that target was chosen, but replay never depends on
        the external preset database.
        """

        spectrum, history = _unpack(source)
        _require_domain(spectrum, "frequency", "chemical-shift referencing")
        if spectrum.axis_unit != "ppm":
            raise ProcessingError("chemical-shift referencing requires a ppm axis")

        observed = _finite(observed_ppm, "observed_ppm")
        target = _finite(target_ppm, "target_ppm")
        delta = target - observed

        output = _new_spectrum(
            spectrum,
            spectrum.axis + delta,
            spectrum.intensities,
        )

        parameters: dict[str, Any] = {
            "observed_ppm": observed,
            "target_ppm": target,
            # Keep the original key for recipe/backward compatibility and add
            # the clearer public name for new provenance-aware recipes.
            "delta_ppm": delta,
            "offset_ppm": delta,
        }

        optional_text = {
            "source_type": source_type,
            "preset_id": preset_id,
            "resonance_id": resonance_id,
            "source_id": source_id,
        }
        for key, value in optional_text.items():
            if value is not None:
                text = str(value).strip()
                if text:
                    parameters[key] = text

        step = ProcessingStep(
            "reference",
            "axis_translation",
            parameters,
        )
        return _with_step(output, history, step)

    def apply_step(
        self, source: Spectrum1D | ProcessingResult, step: ProcessingStep
    ) -> ProcessingResult:
        """Apply one serialized processing step through the public operations."""

        parameters = dict(step.parameters)
        operation = step.operation
        if operation == "digital_filter":
            parameters.pop("decimation", None)
            parameters.pop("firmware_version", None)
            parameters.pop("group_delay", None)
            return self.correct_digital_filter(source, truncate=parameters.get("truncate", True))
        if operation == "apodization":
            return self.apodize(source, method=step.method or "", **parameters)
        if operation == "zero_fill":
            return self.zero_fill(source, **parameters)
        if operation == "fourier_transform":
            if step.method == "numpy_fft":
                return self.fourier_transform(
                    source, exponent_sign="negative", history_step=step
                )
            if step.method == "numpy_fft_positive":
                return self.fourier_transform(
                    source, exponent_sign="positive", history_step=step
                )
            raise ProcessingError(
                f"unsupported Fourier-transform method: {step.method}"
            )
        if operation == "phase":
            parameters.pop("pivot_kind", None)
            return self.phase(source, **parameters)
        if operation == "baseline":
            method = "whittaker" if step.method == "whittaker_asls" else (step.method or "")
            return self.baseline_correct(source, method=method, **parameters)
        if operation == "reference":
            # Derived values are kept for auditability but replay is defined by
            # the serialized observed/target coordinates.
            parameters.pop("delta_ppm", None)
            parameters.pop("offset_ppm", None)
            return self.reference(source, **parameters)
        raise ProcessingError(f"unsupported processing operation: {operation}")

    def replay(
        self, raw_spectrum: Spectrum1D, history: ProcessingHistory
    ) -> ProcessingResult:
        """Rebuild a result from raw data and an ordered processing recipe."""

        if raw_spectrum.domain != "time" or raw_spectrum.state != "raw":
            raise ProcessingError("history replay requires a raw time-domain spectrum")
        result = ProcessingResult(_copy_spectrum(raw_spectrum))
        for step in history.steps:
            result = self.apply_step(result, step)
        return result

    def rebuild_dataset(
        self, dataset: NMRDataset, history: ProcessingHistory
    ) -> NMRDataset:
        """Return a new dataset rebuilt from its in-memory raw spectrum."""

        if dataset.raw_spectrum is None:
            raise ProcessingError("dataset does not contain a raw spectrum")
        result = self.replay(dataset.raw_spectrum, history)
        if result.spectrum.domain != "frequency":
            raise ProcessingError("dataset processing history must end in frequency-domain data")
        return replace(
            dataset,
            processed_spectrum=result.spectrum,
            processing_history=ProcessingHistory(result.history.steps),
        )

    @staticmethod
    def reset_dataset(dataset: NMRDataset) -> NMRDataset:
        """Return a dataset with processed state/history cleared and raw data retained."""

        return replace(
            dataset,
            processed_spectrum=None,
            processing_history=ProcessingHistory(),
        )


def ppm_to_fractional_index(spectrum: Spectrum1D, pivot_ppm: float) -> float:
    """Convert a ppm coordinate to a fractional array index."""

    if spectrum.axis_unit != "ppm" or spectrum.domain != "frequency":
        raise ProcessingError("ppm conversion requires a frequency-domain ppm axis")
    axis = np.asarray(spectrum.axis, dtype=float)
    if axis.size == 0 or not np.all(np.isfinite(axis)):
        raise ProcessingError("spectrum has no valid ppm axis")
    differences = np.diff(axis)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ProcessingError("ppm axis must be strictly monotonic")
    value = _finite(pivot_ppm, "pivot_ppm")
    if not min(axis[0], axis[-1]) <= value <= max(axis[0], axis[-1]):
        raise ProcessingError("pivot_ppm lies outside the spectrum")
    indices = np.arange(axis.size, dtype=float)
    if differences[0] < 0:
        return float(np.interp(value, axis[::-1], indices[::-1]))
    return float(np.interp(value, axis, indices))


def _unpack(
    source: Spectrum1D | ProcessingResult,
) -> tuple[Spectrum1D, ProcessingHistory]:
    if isinstance(source, ProcessingResult):
        return source.spectrum, ProcessingHistory(source.history.steps)
    if isinstance(source, Spectrum1D):
        return source, ProcessingHistory()
    raise TypeError("source must be Spectrum1D or ProcessingResult")


def _with_step(
    spectrum: Spectrum1D,
    history: ProcessingHistory,
    step: ProcessingStep,
    *,
    baseline: np.ndarray | None = None,
) -> ProcessingResult:
    updated = ProcessingHistory(history.steps)
    updated.append(step)
    return ProcessingResult(spectrum, updated, baseline=baseline)


def _new_spectrum(
    source: Spectrum1D, axis: np.ndarray, intensities: np.ndarray
) -> Spectrum1D:
    return Spectrum1D(
        np.array(axis, dtype=float, copy=True),
        np.array(intensities, copy=True),
        source.axis_unit,
        source.domain,
        source.state,
    )


def _copy_spectrum(spectrum: Spectrum1D) -> Spectrum1D:
    return _new_spectrum(spectrum, spectrum.axis, spectrum.intensities)


def _require_domain(spectrum: Spectrum1D, domain: str, operation: str) -> None:
    if spectrum.domain != domain:
        raise ProcessingError(f"{operation} requires {domain}-domain data")


def _finite(value: float | None, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProcessingError(f"{name} must be numeric") from exc
    if not np.isfinite(number):
        raise ProcessingError(f"{name} must be finite")
    return number


def _positive_finite(value: float | None, name: str) -> float:
    number = _finite(value, name)
    if number < 0:
        raise ProcessingError(f"{name} must be non-negative")
    return number


def _dwell_seconds(
    spectrum: Spectrum1D, acquisition_parameters: Mapping[str, Any]
) -> float:
    if spectrum.axis_unit == "s" and spectrum.point_count > 1:
        differences = np.diff(spectrum.axis)
        if np.allclose(differences, differences[0]) and differences[0] > 0:
            return float(differences[0])
        raise ProcessingError("time axis must be uniformly increasing")
    try:
        dwell = float(acquisition_parameters["dwell_time_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProcessingError("a positive dwell_time_s is required") from exc
    if not np.isfinite(dwell) or dwell <= 0:
        raise ProcessingError("dwell_time_s must be finite and positive")
    return dwell


def _elapsed_seconds(
    spectrum: Spectrum1D, acquisition_parameters: Mapping[str, Any]
) -> np.ndarray:
    dwell = _dwell_seconds(spectrum, acquisition_parameters)
    return np.arange(spectrum.point_count, dtype=float) * dwell


def _extended_time_axis(
    spectrum: Spectrum1D,
    target: int,
    acquisition_parameters: Mapping[str, Any],
) -> np.ndarray:
    dwell = _dwell_seconds(spectrum, acquisition_parameters)
    start = float(spectrum.axis[0]) if spectrum.axis_unit == "s" else 0.0
    return start + np.arange(target, dtype=float) * dwell
