"""Dataset-owned observed shifts, independent of spectra and widgets.

This is a projection/edit boundary, not a second observation database. Peak
measurements remain authoritative for linked observables; independent records
can be edited without manufacturing spectral peaks.
"""
from dataclasses import dataclass
import math

from ..core import NMRDataset, NMRObservable
from ..core.metadata import ExperimentalConditions, resolve_temperature
from .observable_sync import sync_peak_observable
from .shift_analysis import normalize_nucleus


def finite_number(value, name, *, optional=False, positive=False):
    if optional and (value is None or value == ""):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def read_only(dataset):
    return bool(dataset.vendor_metadata.get("read_only_shift_bridge"))


@dataclass(frozen=True)
class NMRDataRow:
    dataset_id: str
    observable_id: str | None
    peak_id: str | None
    assignment: str | None
    nucleus: str
    shift_ppm: float
    uncertainty_ppm: float | None
    temperature_k: float | None
    included: bool
    source: str
    diagnostics: tuple[str, ...]
    intensity: float | None = None
    peak_source: str | None = None


def data_rows(datasets, *, include_unassigned_peaks=False):
    """Inspect all records, including excluded, unassigned and broken links."""
    rows = []
    for dataset in datasets:
        counts = {}
        for item in dataset.observables:
            key = (item.assignment or "").strip().casefold()
            if key and item.included:
                counts[key] = counts.get(key, 0) + 1
        temperature = resolve_temperature(dataset.conditions)
        peaks = {p.id: p for p in dataset.peaks}
        for item in dataset.observables:
            issues = []
            assignment = (item.assignment or "").strip()
            if not assignment:
                issues.append("Unassigned")
            elif counts.get(assignment.casefold(), 0) > 1:
                issues.append("Duplicate assignment")
            try:
                finite_number(item.chemical_shift_ppm, "Shift")
            except ValueError:
                issues.append("Invalid shift")
            try:
                finite_number(item.uncertainty_ppm, "Uncertainty", optional=True, positive=True)
            except ValueError:
                issues.append("Invalid uncertainty")
            if normalize_nucleus(item.nucleus) != normalize_nucleus(dataset.conditions.nucleus):
                issues.append("Nucleus mismatch")
            if item.source_peak_id:
                peak = peaks.get(item.source_peak_id)
                if peak is None:
                    issues.append("Missing source peak")
                else:
                    try:
                        if not math.isclose(finite_number(peak.position_ppm, "Peak shift"),
                            finite_number(item.chemical_shift_ppm, "Observed shift"), rel_tol=0, abs_tol=1e-9):
                            issues.append("Source shift changed")
                    except ValueError:
                        issues.append("Invalid source measurement")
            if temperature.status != "valid":
                issues.append(f"Temperature {temperature.status}")
            if not item.included:
                issues.append("Excluded from shift comparison")
            source = ("PCS project" if read_only(dataset) else "Spectrum" if item.source_peak_id
                      else item.provenance.get("origin", "Manual"))
            rows.append(NMRDataRow(dataset.id, item.id, item.source_peak_id, item.assignment,
                item.nucleus, item.chemical_shift_ppm, item.uncertainty_ppm,
                temperature.temperature_k, item.included, str(source), tuple(issues),
                peaks[item.source_peak_id].height if item.source_peak_id in peaks else None,
                peaks[item.source_peak_id].source if item.source_peak_id in peaks else None))
        if include_unassigned_peaks:
            linked = {o.source_peak_id for o in dataset.observables}
            for peak in dataset.peaks:
                if peak.id not in linked:
                    rows.append(NMRDataRow(dataset.id, None, peak.id, None,
                        dataset.conditions.nucleus or "", peak.position_ppm, None,
                        temperature.temperature_k, False, "Spectrum",
                        ("Unassigned — enter an assignment before analysis",), peak.height, peak.source))
    return tuple(rows)


def new_data_dataset(name, nucleus, role="unspecified", temperature_k=None):
    name = str(name).strip()
    nucleus = normalize_nucleus(nucleus)
    if not name or not nucleus:
        raise ValueError("Experiment name and nucleus are required")
    temperature = finite_number(temperature_k, "Temperature (K)", optional=True, positive=True)
    return NMRDataset(conditions=ExperimentalConditions(nucleus=nucleus, temperature_k=temperature),
        experimental_role=role, vendor_metadata={"display_name": name},
        source_provenance={"origin": "manual_observations"})


def edit_observable(dataset, *, observable_id=None, assignment=None, shift_ppm=None,
                    uncertainty_ppm=None, included=True, provenance=None):
    """Validate first; edit a record in place, preserving its stable identity."""
    if read_only(dataset):
        raise ValueError("PCS project observations are read-only. Use the existing project import or copy values into a new experiment.")
    if type(included) is not bool:
        raise ValueError("Use must be a boolean")
    nucleus = normalize_nucleus(dataset.conditions.nucleus)
    if not nucleus:
        raise ValueError("Set the experiment nucleus before adding observations")
    sigma = finite_number(uncertainty_ppm, "Uncertainty (ppm)", optional=True, positive=True)
    label = str(assignment or "").strip() or None
    current = next((o for o in dataset.observables if o.id == observable_id), None)
    if observable_id is not None and current is None:
        raise ValueError("The observation no longer exists")
    if current and current.source_peak_id:
        if not label:
            raise ValueError("Clear a spectrum assignment in the Assignment cell; this detail editor preserves its linked observation")
        peak = next((p for p in dataset.peaks if p.id == current.source_peak_id), None)
        if peak is None:
            raise ValueError("Source peak is missing; review its link in NMR Data")
        # Shift cannot be manually overwritten through this common data table.
        if shift_ppm is not None and finite_number(shift_ppm, "Shift") != float(peak.position_ppm):
            raise ValueError("A spectrum-linked shift must match its source peak")
        current = sync_peak_observable(dataset, peak_id=peak.id, assignment=label)
    else:
        shift = finite_number(shift_ppm, "Shift (ppm)")
        if current is None:
            current = NMRObservable(nucleus=nucleus, chemical_shift_ppm=shift,
                dataset_id=dataset.id, provenance=dict(provenance or {"origin": "Manual"}))
            dataset.observables.append(current)
        current.chemical_shift_ppm = shift
        current.assignment = label
    current.uncertainty_ppm = sigma
    current.included = included
    return current


def set_observable_use(dataset, observable_id, included):
    if read_only(dataset):
        raise ValueError("PCS project observations are read-only")
    if type(included) is not bool:
        raise ValueError("Use must be a boolean")
    item = next((o for o in dataset.observables if o.id == observable_id), None)
    if item is None:
        raise ValueError("The observation no longer exists")
    item.included = included


def remove_observable(dataset, observable_id):
    if read_only(dataset):
        raise ValueError("PCS project observations are read-only")
    item = next((o for o in dataset.observables if o.id == observable_id), None)
    if item is None:
        raise ValueError("The observation no longer exists")
    if item.source_peak_id:
        raise ValueError("Remove a spectrum-linked observation by clearing its assignment in Peaks")
    dataset.observables.remove(item)
