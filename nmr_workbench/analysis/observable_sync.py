"""Synchronization between assigned peaks and NMRObservable records."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.observables import NMRObservable


class ObservableSyncError(ValueError):
    """Raised when an assigned peak cannot be represented as an observable."""


@dataclass(frozen=True)
class ObservableSyncSummary:
    created: int = 0
    updated: int = 0
    removed: int = 0


def _dataset_nucleus(dataset) -> str:
    nucleus = getattr(getattr(dataset, "conditions", None), "nucleus", None)
    if nucleus is None or not str(nucleus).strip():
        raise ObservableSyncError(
            "A dataset nucleus is required before assigned peaks can become NMR observables."
        )
    return str(nucleus).strip()


def observable_for_peak(dataset, peak_id: str) -> NMRObservable | None:
    """Return the first observable linked to ``peak_id``, if one exists."""
    for observable in dataset.observables:
        if observable.source_peak_id == peak_id:
            return observable
    return None


def sync_peak_observable(
    dataset,
    *,
    peak_id: str,
    assignment: str | None,
) -> NMRObservable | None:
    """Create, update, or remove the observable linked to one picked peak.

    Blank assignment means that the spectral peak remains picked but is not yet
    interpreted chemically, so any observable linked to that peak is removed.
    """
    peak = next((item for item in dataset.peaks if item.id == peak_id), None)
    clean_assignment = (assignment or "").strip()

    linked = [
        item for item in dataset.observables
        if item.source_peak_id == peak_id
    ]

    if peak is None or not clean_assignment:
        if linked:
            linked_ids = {item.id for item in linked}
            dataset.observables[:] = [
                item for item in dataset.observables
                if item.id not in linked_ids
            ]
        return None

    nucleus = _dataset_nucleus(dataset)

    if linked:
        observable = linked[0]
        observable.nucleus = nucleus
        observable.chemical_shift_ppm = float(peak.position_ppm)
        observable.assignment = clean_assignment
        observable.dataset_id = dataset.id
        observable.source_peak_id = peak.id

        # Enforce exactly one linked observable per picked peak.
        if len(linked) > 1:
            duplicate_ids = {item.id for item in linked[1:]}
            dataset.observables[:] = [
                item for item in dataset.observables
                if item.id not in duplicate_ids
            ]
        return observable

    observable = NMRObservable(
        nucleus=nucleus,
        chemical_shift_ppm=float(peak.position_ppm),
        assignment=clean_assignment,
        dataset_id=dataset.id,
        source_peak_id=peak.id,
    )
    dataset.observables.append(observable)
    return observable


def sync_dataset_observables(
    dataset,
    assignments: dict[str, str],
) -> ObservableSyncSummary:
    """Synchronize every peak-linked observable for one dataset.

    Observables with ``source_peak_id is None`` are preserved because they may
    have been created by another analysis workflow.
    """
    before_by_peak = {
        item.source_peak_id: (
            item.id,
            item.nucleus,
            item.chemical_shift_ppm,
            item.assignment,
        )
        for item in dataset.observables
        if item.source_peak_id is not None
    }

    live_peak_ids = {peak.id for peak in dataset.peaks}

    # Remove stale peak-linked records first, but preserve independent records.
    stale_ids = {
        item.id
        for item in dataset.observables
        if item.source_peak_id is not None
        and (
            item.source_peak_id not in live_peak_ids
            or not assignments.get(item.source_peak_id, "").strip()
        )
    }
    removed = len(stale_ids)
    if stale_ids:
        dataset.observables[:] = [
            item for item in dataset.observables
            if item.id not in stale_ids
        ]

    created = 0
    updated = 0

    for peak in dataset.peaks:
        assignment = assignments.get(peak.id, "").strip()
        if not assignment:
            continue

        before = before_by_peak.get(peak.id)
        observable = sync_peak_observable(
            dataset,
            peak_id=peak.id,
            assignment=assignment,
        )
        if observable is None:
            continue

        if before is None:
            created += 1
        else:
            after = (
                observable.id,
                observable.nucleus,
                observable.chemical_shift_ppm,
                observable.assignment,
            )
            if after != before:
                updated += 1

    return ObservableSyncSummary(
        created=created,
        updated=updated,
        removed=removed,
    )


def assignments_from_observables(dataset) -> dict[str, str]:
    """Rebuild assignment drafts from existing peak-linked observables."""
    live_peak_ids = {peak.id for peak in dataset.peaks}
    result: dict[str, str] = {}
    for observable in dataset.observables:
        if (
            observable.source_peak_id in live_peak_ids
            and observable.assignment is not None
            and observable.assignment.strip()
        ):
            result[observable.source_peak_id] = observable.assignment.strip()
    return result
