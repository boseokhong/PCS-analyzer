"""Non-destructive processing-session controller for NMR Workbench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nmr_workbench.core import NMRDataset, ProcessingHistory, ProcessingStep, Spectrum1D
from nmr_workbench.processing import (
    AutoPhaseEstimate,
    ProcessingError,
    ProcessingResult,
    SpectrumProcessor,
)


class ProcessingSessionError(ValueError):
    """A user-correctable processing-session request was invalid."""


@dataclass(frozen=True)
class PreviewState:
    operation: str
    result: ProcessingResult


class ProcessingSession:
    """Separate immutable raw input, committed recipe state, and one preview.

    All numerical work is delegated to ``SpectrumProcessor``. A preview is
    always recomputed from the committed spectrum/history, so repeated future
    slider updates replace one temporary result instead of growing history.
    """

    def __init__(self, *, max_points: int = SpectrumProcessor.DEFAULT_MAX_POINTS) -> None:
        self.max_points = int(max_points)
        self.dataset: NMRDataset | None = None
        self.processor: SpectrumProcessor | None = None
        self.committed_spectrum: Spectrum1D | None = None
        self.preview: PreviewState | None = None

    def select_dataset(self, dataset: NMRDataset) -> Spectrum1D | None:
        """Select a dataset, discarding temporary preview and rebuilding history."""

        self.dataset = dataset
        self.processor = SpectrumProcessor.from_dataset(dataset, max_points=self.max_points)
        self.preview = None
        history = dataset.processing_history
        try:
            if dataset.restored_committed_spectrum is not None:
                # Session recovery uses the saved committed bytes, not a fresh
                # numerical replay which could differ across dependency versions.
                self.committed_spectrum = dataset.restored_committed_spectrum
            elif dataset.raw_spectrum is not None and history.steps:
                self.committed_spectrum = self.processor.replay(
                    dataset.raw_spectrum, history
                ).spectrum
                dataset.processed_spectrum = (
                    self.committed_spectrum
                    if self.committed_spectrum.domain == "frequency"
                    else None
                )
            elif dataset.processed_spectrum is not None:
                self.committed_spectrum = dataset.processed_spectrum
            else:
                self.committed_spectrum = dataset.raw_spectrum
        except ProcessingError as exc:
            raise ProcessingSessionError(
                f"Could not rebuild committed processing state: {exc}"
            ) from exc
        return self.committed_spectrum

    @property
    def preview_spectrum(self) -> Spectrum1D | None:
        return self.preview.result.spectrum if self.preview is not None else None

    @property
    def preview_baseline(self) -> np.ndarray | None:
        if self.preview is None or self.preview.operation != "baseline":
            return None
        return self.preview.result.baseline

    @property
    def displayed_spectrum(self) -> Spectrum1D | None:
        return self.preview_spectrum or self.committed_spectrum

    @property
    def has_preview(self) -> bool:
        return self.preview is not None

    @property
    def can_process_time_domain(self) -> bool:
        return (
            self.committed_spectrum is not None
            and self.committed_spectrum.domain == "time"
        )

    @property
    def can_phase(self) -> bool:
        spectrum = self.committed_spectrum
        return (
            spectrum is not None
            and spectrum.domain == "frequency"
            and np.iscomplexobj(spectrum.intensities)
        )

    @property
    def can_baseline(self) -> bool:
        spectrum = self.committed_spectrum
        return spectrum is not None and spectrum.domain == "frequency"

    def initialize_default_spectrum(self) -> Spectrum1D:
        """Commit conservative vendor-required preprocessing and Fourier transform."""

        dataset = self._dataset()
        if self.committed_spectrum is not None and self.committed_spectrum.domain == "frequency":
            return self.committed_spectrum
        if dataset.raw_spectrum is None:
            raise ProcessingSessionError("The selected dataset has no raw FID.")
        if dataset.processing_history.steps:
            raise ProcessingSessionError(
                "Default initialization requires an unprocessed raw dataset."
            )
        source = ProcessingResult(dataset.raw_spectrum)
        parameters = dataset.acquisition_parameters
        vendor = (dataset.vendor or "").strip().lower()
        if vendor == "bruker" and all(
            parameters.get(name) is not None
            for name in (
                "digital_filter_decimation",
                "digital_filter_firmware_version",
            )
        ):
            try:
                source = self._processor().correct_digital_filter(source)
            except ProcessingError as exc:
                raise ProcessingSessionError(str(exc)) from exc
        try:
            result = self._processor().fourier_transform(source)
        except ProcessingError as exc:
            raise ProcessingSessionError(str(exc)) from exc
        self.committed_spectrum = result.spectrum
        dataset.processing_history = ProcessingHistory(result.history.steps)
        dataset.processed_spectrum = result.spectrum
        dataset.mark_spectrum_changed()
        self.preview = None
        return result.spectrum

    def start_reprocessing_from_raw(self) -> Spectrum1D:
        """Discard committed processing while preserving and selecting the raw FID."""

        dataset = self._dataset()
        if dataset.raw_spectrum is None:
            raise ProcessingSessionError("The selected dataset has no raw FID.")
        dataset.processing_history = ProcessingHistory()
        dataset.processed_spectrum = None
        dataset.mark_spectrum_changed()
        self.committed_spectrum = dataset.raw_spectrum
        self.preview = None
        return self.committed_spectrum

    def preview_apodization(
        self,
        *,
        method: str,
        lb_hz: Any = None,
        gb_hz: Any = None,
    ) -> Spectrum1D:
        source = self._committed_result("apodization", required_domain="time")
        normalized = method.strip().lower()
        try:
            if normalized == "exponential":
                result = self._processor().apodize(
                    source, method=normalized, lb_hz=_number(lb_hz, "LB")
                )
            elif normalized == "gaussian":
                result = self._processor().apodize(
                    source, method=normalized, gb_hz=_number(gb_hz, "GB")
                )
            else:
                raise ProcessingSessionError(
                    "Apodization method must be Exponential or Gaussian."
                )
        except ProcessingError as exc:
            raise ProcessingSessionError(str(exc)) from exc
        self.preview = PreviewState("apodization", result)
        return result.spectrum

    def preview_zero_fill(self, target_points: Any) -> Spectrum1D:
        source = self._committed_result("zero filling", required_domain="time")
        target = _integer(target_points, "Target points")
        current = source.spectrum.point_count
        if target < current:
            raise ProcessingSessionError(
                f"Target points ({target:,}) cannot be smaller than the current "
                f"spectrum ({current:,})."
            )
        if target > self.max_points:
            raise ProcessingSessionError(
                f"Target points exceeds the safety limit ({self.max_points:,})."
            )
        try:
            result = self._processor().zero_fill(source, target_points=target)
        except ProcessingError as exc:
            raise ProcessingSessionError(str(exc)) from exc
        self.preview = PreviewState("zero_fill", result)
        return result.spectrum

    def _preview_time_domain_reprocess_to_spectrum(
        self,
        replacement_step: ProcessingStep,
    ) -> Spectrum1D:
        """Rebuild the frequency-domain spectrum from raw FID with one updated
        time-domain step.

        This lets Apodization and Zero Filling remain usable while the user is
        viewing the Spectrum.  Existing downstream frequency-domain processing
        steps (phase, baseline, reference) are replayed after the regenerated FT
        so the user does not have to switch manually to Raw FID.
        """

        dataset = self._dataset()
        if dataset.raw_spectrum is None:
            raise ProcessingSessionError("The selected dataset has no raw FID.")

        if replacement_step.operation not in {"apodization", "zero_fill"}:
            raise ProcessingSessionError(
                "Only apodization or zero filling can be replaced this way."
            )

        steps = list(dataset.processing_history.steps)
        ft_index = next(
            (i for i, step in enumerate(steps) if step.operation == "fourier_transform"),
            None,
        )

        if ft_index is None:
            # No committed FT recipe yet. Build a minimal one from the raw FID.
            pre_ft = []
            ft_step = None
            post_ft = []
        else:
            pre_ft = steps[:ft_index]
            ft_step = steps[ft_index]
            post_ft = steps[ft_index + 1 :]

        # Preserve vendor-required and any future time-domain operations, while
        # replacing only the selected canonical preparation step.
        digital = [s for s in pre_ft if s.operation == "digital_filter"]
        apod = [s for s in pre_ft if s.operation == "apodization"]
        zero = [s for s in pre_ft if s.operation == "zero_fill"]
        other = [
            s
            for s in pre_ft
            if s.operation not in {"digital_filter", "apodization", "zero_fill"}
        ]

        if replacement_step.operation == "apodization":
            apod = [replacement_step]
        else:
            zero = [replacement_step]

        ordered_pre_ft = [*digital, *apod, *zero, *other]

        result = ProcessingResult(dataset.raw_spectrum)
        try:
            for step in ordered_pre_ft:
                result = self._processor().apply_step(result, step)

            if ft_step is None:
                result = self._processor().fourier_transform(result)
            else:
                result = self._processor().apply_step(result, ft_step)

            # Re-run downstream corrections against the regenerated spectrum.
            for step in post_ft:
                result = self._processor().apply_step(result, step)
        except ProcessingError as exc:
            raise ProcessingSessionError(str(exc)) from exc

        self.preview = PreviewState(replacement_step.operation, result)
        return result.spectrum

    def preview_apodization_from_spectrum(
        self,
        *,
        method: str,
        lb_hz: Any = None,
        gb_hz: Any = None,
    ) -> Spectrum1D:
        normalized = method.strip().lower()
        if normalized == "exponential":
            step = ProcessingStep(
                "apodization",
                "exponential",
                {"lb_hz": _number(lb_hz, "LB")},
            )
        elif normalized == "gaussian":
            step = ProcessingStep(
                "apodization",
                "gaussian",
                {"gb_hz": _number(gb_hz, "GB")},
            )
        else:
            raise ProcessingSessionError(
                "Apodization method must be Exponential or Gaussian."
            )
        return self._preview_time_domain_reprocess_to_spectrum(step)

    def preview_zero_fill_from_spectrum(self, target_points: Any) -> Spectrum1D:
        target = _integer(target_points, "Target points")
        if target > self.max_points:
            raise ProcessingSessionError(
                f"Target points exceeds the safety limit ({self.max_points:,})."
            )
        step = ProcessingStep(
            "zero_fill",
            parameters={"target_points": target},
        )
        return self._preview_time_domain_reprocess_to_spectrum(step)

    def preview_phase(
        self,
        *,
        p0_deg: Any,
        p1_deg: Any,
        pivot_ppm: Any = None,
        pivot_index: Any = None,
    ) -> Spectrum1D:
        self._require_compatible_preview("phase")
        source = self._committed_result("phase correction", required_domain="frequency")
        if not np.iscomplexobj(source.spectrum.intensities):
            raise ProcessingSessionError(
                "Phase correction requires a complex frequency-domain spectrum."
            )
        p0 = _number(p0_deg, "P0")
        p1 = _number(p1_deg, "P1")
        try:
            if source.spectrum.axis_unit.lower() == "ppm":
                if pivot_ppm is None:
                    raise ProcessingSessionError("Pivot must be specified in ppm.")
                result = self._processor().phase(
                    source,
                    p0_deg=p0,
                    p1_deg=p1,
                    pivot_ppm=_number(pivot_ppm, "Pivot"),
                )
            else:
                if pivot_index is None:
                    raise ProcessingSessionError(
                        "Pivot must be specified as a spectrum index."
                    )
                result = self._processor().phase(
                    source,
                    p0_deg=p0,
                    p1_deg=p1,
                    pivot_index=_number(pivot_index, "Pivot index"),
                )
        except ProcessingError as exc:
            raise ProcessingSessionError(str(exc)) from exc
        self.preview = PreviewState("phase", result)
        return result.spectrum

    def preview_auto_phase(
        self,
        *,
        algorithm: str,
        pivot_ppm: Any = None,
        pivot_index: Any = None,
    ) -> AutoPhaseEstimate:
        """Estimate phase parameters and install one ordinary phase preview."""

        self._require_compatible_preview("phase")
        source = self._committed_result(
            "automatic phase correction", required_domain="frequency"
        )
        spectrum = source.spectrum
        if not np.iscomplexobj(spectrum.intensities):
            raise ProcessingSessionError("Auto Phase requires a complex spectrum.")
        try:
            if spectrum.axis_unit.lower() == "ppm":
                pivot_value = _number(pivot_ppm, "Pivot")
                estimate = self._processor().auto_phase(
                    source, algorithm=algorithm, pivot_ppm=pivot_value
                )
                result = self._processor().phase(
                    source,
                    p0_deg=estimate.p0_deg,
                    p1_deg=estimate.p1_deg,
                    pivot_ppm=pivot_value,
                )
            else:
                pivot_value = _number(pivot_index, "Pivot index")
                estimate = self._processor().auto_phase(
                    source, algorithm=algorithm, pivot_index=pivot_value
                )
                result = self._processor().phase(
                    source,
                    p0_deg=estimate.p0_deg,
                    p1_deg=estimate.p1_deg,
                    pivot_index=pivot_value,
                )
        except ProcessingError as exc:
            raise ProcessingSessionError(str(exc)) from exc
        self.preview = PreviewState("phase", result)
        return estimate

    def preview_baseline_correction(
        self,
        *,
        method: str,
        order: Any = 8,
        lam: Any = 1.0e6,
        p: Any = 0.001,
    ) -> Spectrum1D:
        """Replace the temporary whole-spectrum baseline preview."""

        self._require_compatible_preview("baseline")
        source = self._committed_result(
            "baseline correction", required_domain="frequency"
        )
        normalized = method.strip().lower()
        try:
            if normalized == "bernstein":
                baseline_order = _integer(order, "Bernstein order")
                if not 0 <= baseline_order <= 64:
                    raise ProcessingSessionError(
                        "Bernstein order must be between 0 and 64."
                    )
                result = self._processor().baseline_correct(
                    source, method="bernstein", order=baseline_order
                )
            elif normalized in {"whittaker", "asls", "arpls"}:
                lambda_value = _number(lam, "Lambda")
                if not np.isfinite(lambda_value) or lambda_value <= 0:
                    raise ProcessingSessionError("Lambda must be finite and positive.")
                if normalized == "arpls":
                    result = self._processor().baseline_correct(
                        source,
                        method="arpls",
                        lam=lambda_value,
                        diff_order=2,
                    )
                else:
                    asymmetry = _number(p, "p")
                    if not np.isfinite(asymmetry) or not 0 < asymmetry < 1:
                        raise ProcessingSessionError(
                            "p must lie strictly between 0 and 1."
                        )
                    result = self._processor().baseline_correct(
                        source,
                        method="whittaker",
                        lam=lambda_value,
                        p=asymmetry,
                        diff_order=2,
                    )
            else:
                raise ProcessingSessionError(
                    "Baseline method must be Bernstein, Whittaker / AsLS, or arPLS."
                )
        except ProcessingError as exc:
            raise ProcessingSessionError(str(exc)) from exc
        self.preview = PreviewState("baseline", result)
        return result.spectrum

    def preview_auto_baseline(self) -> Spectrum1D:
        """Preview the low-tuning arPLS baseline policy without committing it."""

        return self.preview_baseline_correction(method="arpls", lam=1.0e5)

    def preview_reference(
        self,
        *,
        observed_ppm: Any,
        target_ppm: Any,
        source_type: str | None = None,
        preset_id: str | None = None,
        resonance_id: str | None = None,
        source_id: str | None = None,
    ) -> Spectrum1D:
        """Preview chemical-shift referencing without committing history.

        The ppm axis is translated by ``target_ppm - observed_ppm`` while
        intensities remain unchanged.  Reference-source provenance is passed to
        the processing layer and recorded in the eventual ProcessingStep.
        """

        self._require_compatible_preview("reference")
        source = self._committed_result(
            "chemical-shift referencing",
            required_domain="frequency",
        )
        if source.spectrum.axis_unit.lower() != "ppm":
            raise ProcessingSessionError(
                "Chemical-shift referencing requires a ppm-domain spectrum."
            )

        observed = _number(observed_ppm, "Observed ppm")
        target = _number(target_ppm, "Target ppm")
        if not np.isfinite(observed) or not np.isfinite(target):
            raise ProcessingSessionError(
                "Observed and target ppm values must be finite."
            )

        try:
            result = self._processor().reference(
                source,
                observed_ppm=observed,
                target_ppm=target,
                source_type=source_type,
                preset_id=preset_id,
                resonance_id=resonance_id,
                source_id=source_id,
            )
        except ProcessingError as exc:
            raise ProcessingSessionError(str(exc)) from exc

        self.preview = PreviewState("reference", result)
        return result.spectrum

    def biggest_peak_pivot(self) -> float:
        """Return the committed spectrum's maximum-magnitude coordinate."""

        spectrum = self.committed_spectrum
        if spectrum is None or spectrum.domain != "frequency":
            raise ProcessingSessionError(
                "Biggest Peak requires a committed frequency-domain spectrum."
            )
        if not np.iscomplexobj(spectrum.intensities):
            raise ProcessingSessionError(
                "Biggest Peak for phasing requires complex spectrum data."
            )
        magnitude = np.abs(spectrum.intensities)
        finite = np.isfinite(magnitude)
        if not finite.any():
            raise ProcessingSessionError("The committed spectrum has no finite points.")
        index = int(np.argmax(np.where(finite, magnitude, -np.inf)))
        if spectrum.axis_unit.lower() == "ppm":
            return float(spectrum.axis[index])
        return float(index)

    def apply_preview(self) -> Spectrum1D:
        if self.preview is None:
            raise ProcessingSessionError("There is no processing preview to apply.")
        dataset = self._dataset()
        result = self.preview.result
        self.committed_spectrum = result.spectrum
        dataset.processing_history = ProcessingHistory(result.history.steps)
        dataset.processed_spectrum = (
            result.spectrum if result.spectrum.domain == "frequency" else None
        )
        dataset.mark_spectrum_changed()
        self.preview = None
        return self.committed_spectrum

    def cancel_preview(self) -> Spectrum1D | None:
        self.preview = None
        return self.committed_spectrum

    def apply_fourier_transform(self) -> Spectrum1D:
        if self.has_preview:
            raise ProcessingSessionError(
                "Apply or cancel the current preview before Fourier transformation."
            )
        source = self._committed_result("Fourier transformation", required_domain="time")
        try:
            result = self._processor().fourier_transform(source)
        except ProcessingError as exc:
            raise ProcessingSessionError(str(exc)) from exc
        dataset = self._dataset()
        self.committed_spectrum = result.spectrum
        dataset.processing_history = ProcessingHistory(result.history.steps)
        dataset.processed_spectrum = result.spectrum
        dataset.mark_spectrum_changed()
        return result.spectrum

    def rebuild_committed(self) -> Spectrum1D:
        dataset = self._dataset()
        if dataset.raw_spectrum is None:
            raise ProcessingSessionError("The selected dataset has no raw spectrum to replay.")
        try:
            result = self._processor().replay(
                dataset.raw_spectrum, dataset.processing_history
            )
        except ProcessingError as exc:
            raise ProcessingSessionError(str(exc)) from exc
        self.committed_spectrum = result.spectrum
        dataset.processed_spectrum = (
            result.spectrum if result.spectrum.domain == "frequency" else None
        )
        dataset.mark_spectrum_changed()
        self.preview = None
        return result.spectrum

    def _committed_result(
        self, operation: str, *, required_domain: str
    ) -> ProcessingResult:
        spectrum = self.committed_spectrum
        if spectrum is None:
            raise ProcessingSessionError("No spectrum is selected.")
        if spectrum.domain != required_domain:
            raise ProcessingSessionError(
                f"{operation.capitalize()} requires {required_domain}-domain data."
            )
        history = ProcessingHistory(self._dataset().processing_history.steps)
        return ProcessingResult(spectrum, history)

    def _dataset(self) -> NMRDataset:
        if self.dataset is None:
            raise ProcessingSessionError("No NMR dataset is selected.")
        return self.dataset

    def _processor(self) -> SpectrumProcessor:
        if self.processor is None:
            raise ProcessingSessionError("No NMR dataset is selected.")
        return self.processor

    def _require_compatible_preview(self, operation: str) -> None:
        if self.preview is not None and self.preview.operation != operation:
            raise ProcessingSessionError(
                f"Apply or cancel the active {self.preview.operation.replace('_', ' ')} "
                f"preview before starting {operation.replace('_', ' ')}."
            )


def suggested_zero_fill_sizes(current_points: int, *, count: int = 4) -> tuple[int, ...]:
    if current_points < 1:
        return ()
    size = 1 << (int(current_points) - 1).bit_length()
    return tuple(size << offset for offset in range(max(int(count), 1)))


def _number(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProcessingSessionError(f"{label} must be numeric.") from exc


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ProcessingSessionError(f"{label} must be an integer.")
    try:
        text = str(value).strip()
        if not text or any(character in text.lower() for character in (".", "e")):
            raise ValueError
        return int(text)
    except (TypeError, ValueError) as exc:
        raise ProcessingSessionError(f"{label} must be an integer.") from exc
