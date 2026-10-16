"""Reviewed assignment-table exchange using stable IDs, never ppm matching."""
import csv
from collections import Counter
from copy import copy, deepcopy
from dataclasses import dataclass
import math

from nmr_workbench.analysis.observable_sync import sync_peak_observable
from .exchange_files import atomic_text_export


FIELDS = ("schema_version", "dataset_id", "dataset_name", "peak_id", "observable_id",
          "spectral_revision", "nucleus", "chemical_shift_ppm", "assignment", "uncertainty_ppm")


def assignment_rows(datasets):
    rows = []
    for dataset in datasets:
        for observable in dataset.observables:
            if not observable.source_peak_id or not observable.assignment:
                continue
            peak = next((p for p in dataset.peaks if p.id == observable.source_peak_id), None)
            if peak is None:
                continue
            rows.append(dict(zip(FIELDS, (1, dataset.id, str(dataset.source_path or dataset.id[:8]),
                peak.id, observable.id, dataset.spectral_revision, observable.nucleus,
                peak.position_ppm, observable.assignment, observable.uncertainty_ppm))))
    return tuple(rows)


def export_assignments(path, datasets):
    rows = assignment_rows(datasets)
    if not rows:
        raise ValueError("No assigned peak observables to export. Assign peaks and check nucleus metadata first.")
    with atomic_text_export(path, encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


@dataclass(frozen=True)
class AssignmentReview:
    row_number: int
    dataset_id: str
    peak_id: str
    assignment: str
    uncertainty_ppm: float | None
    status: str
    message: str


def _signature(datasets):
    return tuple((d.id, d.spectral_revision, d.conditions.nucleus,
        tuple((p.id, p.position_ppm) for p in d.peaks),
        tuple(tuple(vars(o).items()) for o in d.observables)) for d in datasets)


@dataclass(frozen=True)
class AssignmentImport:
    rows: tuple[AssignmentReview, ...]
    signature: tuple


def review_assignments(path, datasets):
    datasets = tuple(datasets)
    if len({d.id for d in datasets}) != len(datasets):
        raise ValueError("Duplicate loaded dataset IDs")
    by_id = {d.id: d for d in datasets}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS) <= set(reader.fieldnames or ()):
            raise ValueError("Use an exported assignment CSV; required identity/revision columns are missing.")
        raw_rows = list(reader)
    keys = [(r.get("dataset_id"), r.get("peak_id")) for r in raw_rows]
    duplicates = {key for key, count in Counter(keys).items() if count > 1}
    reviewed = []
    for number, row in enumerate(raw_rows, 2):
        dataset_id, peak_id = row.get("dataset_id", ""), row.get("peak_id", "")
        assignment = (row.get("assignment") or "").strip()
        uncertainty = None
        status, message = "ready", "New assignment"
        try:
            if (dataset_id, peak_id) in duplicates:
                raise ValueError("Duplicate dataset/peak rows; neither row will be applied")
            if row["schema_version"] != "1":
                raise ValueError("Unsupported assignment CSV version")
            dataset = by_id.get(dataset_id)
            if dataset is None:
                raise ValueError("Dataset ID is not loaded; open the original .nmrsession first")
            peaks = [p for p in dataset.peaks if p.id == peak_id]
            if len(peaks) != 1:
                raise ValueError("Peak ID is missing or duplicated; no ppm-based replacement")
            if int(row["spectral_revision"]) != dataset.spectral_revision:
                raise ValueError("Spectrum has changed since export; export a fresh table")
            shift = float(row["chemical_shift_ppm"])
            if not math.isfinite(shift) or shift != peaks[0].position_ppm:
                raise ValueError("Source peak position differs from the exported observation")
            if not dataset.conditions.nucleus or row["nucleus"] != dataset.conditions.nucleus:
                raise ValueError("Missing or incompatible nucleus")
            if not assignment:
                raise ValueError("Blank assignment skipped; clear assignments in the peak editor")
            if row["uncertainty_ppm"]:
                uncertainty = float(row["uncertainty_ppm"])
                if not math.isfinite(uncertainty) or uncertainty < 0:
                    raise ValueError("Uncertainty must be finite and non-negative")
            linked = [o for o in dataset.observables if o.source_peak_id == peak_id]
            if len(linked) > 1:
                raise ValueError("Multiple observables link to this peak; resolve before importing")
            if row["observable_id"] and (not linked or linked[0].id != row["observable_id"]):
                raise ValueError("Observable identity differs; open the original session or export a fresh table")
            if linked:
                if linked[0].assignment == assignment and linked[0].uncertainty_ppm == uncertainty:
                    status, message = "unchanged", "Already matches the current observable"
                else:
                    status, message = "conflict", f"Existing assignment: {linked[0].assignment}; explicit replacement required"
        except (ValueError, TypeError, KeyError) as exc:
            status, message = "blocked", str(exc)
        reviewed.append(AssignmentReview(number, dataset_id, peak_id, assignment, uncertainty, status, message))
    return AssignmentImport(tuple(reviewed), _signature(datasets))


def apply_assignments(plan, datasets, *, replace_conflicts=False):
    datasets = tuple(datasets)
    if _signature(datasets) != plan.signature:
        raise ValueError("Assignments or spectra changed after review; import and review again")
    ready = [r for r in plan.rows if r.status == "ready" or (r.status == "conflict" and replace_conflicts)]
    # Stage through the canonical synchronizer before publishing any changes.
    target_ids = {r.dataset_id for r in ready}
    staged = {}
    for dataset in datasets:
        if dataset.id in target_ids:
            staged[dataset.id] = copy(dataset)
            staged[dataset.id].observables = deepcopy(dataset.observables)
    for row in ready:
        observable = sync_peak_observable(staged[row.dataset_id], peak_id=row.peak_id, assignment=row.assignment)
        observable.uncertainty_ppm = row.uncertainty_ppm
    for dataset in datasets:
        if dataset.id not in staged:
            continue
        existing = {o.id: o for o in dataset.observables}
        updated = []
        for item in staged[dataset.id].observables:
            target = existing.get(item.id, item)
            target.__dict__.update(vars(item))
            updated.append(target)
        dataset.observables[:] = updated
    return len(ready)
