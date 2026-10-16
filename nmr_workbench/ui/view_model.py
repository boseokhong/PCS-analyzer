"""Pure presentation state and formatting helpers for NMR Workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from nmr_workbench.core import NMRDataset, NMRSeries, ProcessingHistory, ProcessingStep, Spectrum1D
from nmr_workbench.io import load_nmr_dataset
from nmr_workbench.core.batch_recipe import BatchRecipe

SpectrumChoice = Literal["auto", "raw", "processed"]


@dataclass
class WorkbenchState:
    """Loaded experiments and current selection, independent of Tk widgets."""

    datasets: list[NMRDataset] = field(default_factory=list)
    selected_dataset_id: str | None = None
    series: list[NMRSeries] = field(default_factory=list)
    selected_series_id: str | None = None
    selected_member_id: str | None = None
    selected_track_id: str | None = None
    selected_peak_id: str | None = None
    selected_observable_id: str | None = None
    batch_presets: list[BatchRecipe] = field(default_factory=list)

    def update_tracks(self):
        """Refresh correspondence validity without touching peaks/observables."""
        loaded = {d.id: d for d in self.datasets}
        for series in self.series:
            for member in series.members:
                if loaded.get(member.dataset_id) is not member.dataset:
                    series.invalidate_tracks(dataset_id=member.dataset_id, reason="dataset_unloaded")
        return tuple(issue for series in self.series for issue in series.update_tracks())

    def add_dataset(self, dataset: NMRDataset) -> str:
        if not isinstance(dataset, NMRDataset):
            raise TypeError("dataset must be an NMRDataset")
        if not any(item.id == dataset.id for item in self.datasets):
            self.datasets.append(dataset)
        self.selected_dataset_id = dataset.id
        self._sync_series_member()
        return dataset.id

    def select_dataset(self, dataset_id: str) -> NMRDataset:
        for dataset in self.datasets:
            if dataset.id == dataset_id:
                self.selected_dataset_id = dataset_id
                self._sync_series_member()
                return dataset
        raise KeyError(f"unknown NMR dataset id: {dataset_id}")

    def _sync_series_member(self) -> None:
        series = next((item for item in self.series if item.id == self.selected_series_id), None)
        if series is None or not any(t.id == self.selected_track_id for t in series.tracks):
            self.selected_track_id = None
        dataset = self.current_dataset
        if dataset is None or not any(o.id == self.selected_observable_id for o in dataset.observables):
            self.selected_observable_id = None
        if dataset is None or not any(p.id == self.selected_peak_id for p in dataset.peaks):
            self.selected_peak_id = None
        self.selected_member_id = next((
            member.id for member in series.members
            if member.dataset_id == self.selected_dataset_id
        ), None) if series else None

    @property
    def current_dataset(self) -> NMRDataset | None:
        return next(
            (item for item in self.datasets if item.id == self.selected_dataset_id),
            None,
        )


def load_dataset_into_state(
    state: WorkbenchState,
    path: str | Path,
    *,
    loader: Callable[[str | Path], NMRDataset] = load_nmr_dataset,
) -> NMRDataset:
    """Load through the vendor-neutral API and add the result to UI state."""

    dataset = loader(path)
    state.add_dataset(dataset)
    return dataset


def choose_spectrum(dataset: NMRDataset, choice: SpectrumChoice = "auto") -> Spectrum1D | None:
    if choice == "raw":
        return dataset.raw_spectrum
    if choice == "processed":
        return dataset.processed_spectrum
    if choice != "auto":
        raise ValueError("choice must be 'auto', 'raw', or 'processed'")
    return dataset.processed_spectrum or dataset.raw_spectrum


def available_spectrum_choices(dataset: NMRDataset) -> tuple[str, ...]:
    values: list[str] = []
    if dataset.processed_spectrum is not None:
        values.append("Spectrum")
    if dataset.raw_spectrum is not None:
        values.append("Raw FID")
    return tuple(values)


def should_invert_display_axis(spectrum: Spectrum1D) -> bool:
    """Return the display-only NMR convention without mutating the axis."""

    return spectrum.domain == "frequency" and spectrum.axis_unit.lower() == "ppm"


def spectrum_state_label(spectrum: Spectrum1D | None) -> str:
    if spectrum is None:
        return "No spectrum"
    if spectrum.domain == "time":
        return f"Raw time-domain FID · {spectrum.point_count:,} points · {spectrum.axis_unit}"
    if spectrum.axis_unit.lower() == "ppm":
        return f"Processed ppm-domain spectrum · {spectrum.point_count:,} points"
    return (
        f"Processed frequency-domain spectrum · {spectrum.point_count:,} points · "
        f"{spectrum.axis_unit}"
    )


def processing_display_state_label(
    spectrum: Spectrum1D | None, *, preview: bool, committed: bool
) -> str:
    """Describe controller display state without calling working data raw."""

    if spectrum is None:
        return "No spectrum"
    if preview:
        domain = "time-domain" if spectrum.domain == "time" else "frequency-domain"
        return (
            f"Preview — not applied · {domain} · {spectrum.point_count:,} points · "
            f"{spectrum.axis_unit}"
        )
    if committed:
        domain = (
            "ppm-domain"
            if spectrum.domain == "frequency" and spectrum.axis_unit.lower() == "ppm"
            else f"{spectrum.domain}-domain"
        )
        return f"Committed · {domain} spectrum · {spectrum.point_count:,} points"
    return spectrum_state_label(spectrum)


def dataset_display_name(dataset: NMRDataset) -> str:
    display_name = str(
        (getattr(dataset, "vendor_metadata", {}) or {}).get("display_name", "")
    ).strip()
    if display_name:
        return display_name
    if dataset.source_path is None:
        return f"NMR experiment {dataset.id[:8]}"
    name = Path(dataset.source_path).name
    return name[:-4] if name.lower().endswith(".fid") else name


def dataset_summary(dataset: NMRDataset) -> str:
    from nmr_workbench.core import resolve_temperature
    conditions = dataset.conditions
    values = [conditions.nucleus or "unknown nucleus"]
    if conditions.frequency_mhz is not None:
        values.append(f"{conditions.frequency_mhz:.1f} MHz")
    nominal = resolve_temperature(None, override_k=conditions.temperature_k)
    if nominal.status == "valid":
        values.append(f"{nominal.temperature_k:.0f} K")
    return "   ".join(values)


def format_metadata_rows(dataset: NMRDataset) -> list[tuple[str, str]]:
    """Return concise canonical metadata; vendor dictionaries stay hidden."""

    c = dataset.conditions
    a = dataset.acquisition_parameters
    rows: list[tuple[str, str | None]] = [
        ("Dataset", dataset_display_name(dataset)),
        ("Nucleus", c.nucleus),
        ("Vendor", dataset.vendor),
        ("Operating frequency", _number(c.frequency_mhz, "MHz", 6)),
        ("Solvent", c.solvent),
        ("Nominal temperature", _number(c.temperature_k, "K", 6)),
        ("Calibrated temperature", _number(c.calibrated_temperature_k, "K", 6)),
        ("Spectral width", _number(a.get("spectral_width_hz"), "Hz", 6)),
        ("Acquisition time", _number(a.get("acquisition_time_s"), "s", 6)),
        ("Scans", _integer(a.get("scan_count"))),
        ("Receiver gain", _number(a.get("receiver_gain"), None, 6)),
        ("Pulse sequence", _text(a.get("pulse_program"))),
        (
            "Acquisition date",
            c.acquisition_date.isoformat(sep=" ") if c.acquisition_date else None,
        ),
    ]
    return [(label, value or "—") for label, value in rows]


def format_processing_history(history: ProcessingHistory) -> list[str]:
    if not history.steps:
        return ["No processing steps have been applied."]
    return [f"{i}.  {format_processing_step(step)}" for i, step in enumerate(history.steps, 1)]


def format_processing_step(step: ProcessingStep) -> str:
    p = step.parameters
    if step.operation == "apodization":
        if step.method == "exponential":
            return f"Apodization — exponential, LB {_fmt(p.get('lb_hz'))} Hz"
        if step.method == "gaussian":
            return f"Apodization — Gaussian, GB {_fmt(p.get('gb_hz'))} Hz"
        return f"Apodization — {step.method or 'unspecified'}"
    if step.operation == "zero_fill":
        points = p.get("target_points")
        return f"Zero fill — {int(points):,} points" if points is not None else "Zero fill"
    if step.operation == "fourier_transform":
        return "Fourier transform"
    if step.operation == "phase":
        if p.get("pivot_kind") == "ppm" and p.get("pivot_ppm") is not None:
            pivot = f", pivot {_fmt(p.get('pivot_ppm'))} ppm"
        elif p.get("pivot_kind") == "index" and p.get("pivot_index") is not None:
            pivot = f", pivot index {_fmt(p.get('pivot_index'))}"
        else:
            pivot = ""
        return (
            f"Phase — P0 {_fmt(p.get('p0_deg'))}°, "
            f"P1 {_fmt(p.get('p1_deg'))}°{pivot}"
        )
    if step.operation == "baseline":
        method = {
            "whittaker_asls": "AsLS",
            "arpls": "Auto arPLS",
        }.get(step.method, "Bernstein")
        if "order" in p:
            detail = f", order {p['order']}"
        elif step.method == "arpls":
            detail = f", λ {_fmt(p.get('lam'))}"
        else:
            detail = f", λ {_fmt(p.get('lam'))}, p {_fmt(p.get('p'))}"
        return f"Baseline — {method}{detail}"
    if step.operation == "reference":
        observed = p.get("observed_ppm")
        target = p.get("target_ppm")
        offset = p.get("offset_ppm", p.get("delta_ppm"))

        if observed is None or target is None:
            return "Reference"

        detail = f"Reference — {_fmt(observed)} → {_fmt(target)} ppm"
        if offset is not None:
            try:
                detail += f" (Δ {_signed_fmt(offset)} ppm)"
            except (TypeError, ValueError):
                pass

        source_type = p.get("source_type")
        preset_id = p.get("preset_id")
        resonance_id = p.get("resonance_id")
        source_id = p.get("source_id")

        provenance: list[str] = []
        if source_type == "residual_solvent":
            provenance.append("residual solvent")
        elif source_type == "reference_standard":
            provenance.append("reference standard")
        elif source_type == "custom":
            provenance.append("custom")

        if preset_id:
            provenance.append(str(preset_id))
        if resonance_id:
            provenance.append(str(resonance_id))
        if source_id:
            provenance.append(f"source {source_id}")

        if provenance:
            detail += " · " + " · ".join(provenance)

        return detail
    if step.operation == "digital_filter":
        return "Digital filter correction — Bruker"
    words = step.operation.replace("_", " ").strip().capitalize()
    return f"{words} — {step.method}" if step.method else words


def _number(value: object, unit: str | None, precision: int) -> str | None:
    try:
        text = f"{float(value):.{precision}g}"
    except (TypeError, ValueError):
        return None
    return f"{text} {unit}" if unit else text


def _integer(value: object) -> str | None:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return None


def _text(value: object) -> str | None:
    if value is None:
        return None
    return str(value).strip() or None


def _fmt(value: object) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return "?"


def _signed_fmt(value: object) -> str:
    return f"{float(value):+.6g}"
