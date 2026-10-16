"""Explicit VT assignment review; observed shifts remain dataset peak values."""
from dataclasses import dataclass, replace
import math

from ..core.track_observables import ObservableBinding, TrackObservableAssociation
from .observable_sync import sync_peak_observable
from .vt_analysis import VTObservation, extract_vt_track


@dataclass(frozen=True)
class VTObservablePoint:
    observation: VTObservation
    observable_id: str | None
    assignment: str | None
    nucleus: str | None
    observable_shift_ppm: float | None
    observable_uncertainty_ppm: float | None
    binding: ObservableBinding | None
    action: str | None
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class VTObservableSeries:
    series_id: str
    track_id: str
    association: TrackObservableAssociation | None
    points: tuple[VTObservablePoint, ...]
    diagnostics: tuple[str, ...]


def associate_track(series, track_id, *, assignment, nucleus, note=""):
    """Record reviewed intent only. No member observable is created/assigned."""
    series.get_track(track_id)
    if not isinstance(assignment, str) or not isinstance(nucleus, str):
        raise ValueError("assignment and nucleus must be text")
    assignment, nucleus = assignment.strip(), nucleus.strip()
    previous = series.observable_association(track_id)
    association = TrackObservableAssociation(series.id, track_id, assignment, nucleus, note,
        **({"id": previous.id} if previous else {}))
    if previous and (previous.assignment, previous.nucleus) == (assignment, nucleus):
        association = replace(association, bindings=previous.bindings)
    series.set_observable_association(association)
    return association


def unassociate_track(series, track_id):
    """Remove reviewed intent/references, preserving dataset observables."""
    series.remove_observable_association(track_id)


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        return float(value) if math.isfinite(float(value)) else None
    except (TypeError, ValueError, OverflowError):
        return None


def extract_vt_observable_series(series, track_id):
    """Pure immutable review, including invalid points. No inferred bindings.

    Assignment comparison is exact after trimming, with explicit nucleus equality.
    Same assignment on another local observable is ambiguous, never merged.
    """
    track = series.get_track(track_id)
    data = extract_vt_track(series, track)
    association = series.observable_association(track_id)
    members = {m.id: m for m in series.members}
    points = []
    for observation in data.observations:
        identity = observation.identity
        member = members.get(identity.member_id)
        dataset = member.dataset if member else None
        linked = [o for o in dataset.observables if o.source_peak_id == identity.peak_id] if dataset and identity.peak_id else []
        observable = linked[0] if len(linked) == 1 else None
        binding = next((b for b in association.bindings if b.member_id == identity.member_id), None) if association else None
        issues = [d.code for d in observation.diagnostics]
        blockers = []
        if not observation.link_valid or not observation.member_included or observation.chemical_shift_ppm is None:
            blockers.append("invalid_track_point")
        if len(linked) > 1:
            blockers.append("duplicate_peak_observables")
        if observable and (observable.dataset_id != identity.dataset_id or sum(o.id == observable.id for o in dataset.observables) != 1):
            blockers.append("observable_identity_conflict")
        if binding and (binding.dataset_id != identity.dataset_id or binding.peak_id != identity.peak_id or not observable or observable.id != binding.observable_id):
            blockers.append("broken_observable_binding")
        if association and dataset:
            if dataset.conditions.nucleus != association.nucleus or (observable and observable.nucleus != association.nucleus):
                blockers.append("nucleus_conflict")
            if observable and observable.assignment and observable.assignment.strip() != association.assignment:
                blockers.append("assignment_conflict")
            if any(o.source_peak_id != identity.peak_id and (o.assignment or "").strip() == association.assignment for o in dataset.observables):
                blockers.append("assignment_already_used")
        if not observable:
            issues.append("missing_observable")
        action = None
        if association and not blockers:
            action = "create" if not observable else (
                "update" if (observable.assignment or "").strip() != association.assignment or _number(observable.chemical_shift_ppm) != observation.chemical_shift_ppm else "associate_existing")
        if observable and _number(observable.chemical_shift_ppm) != observation.chemical_shift_ppm:
            issues.append("observable_shift_out_of_sync")
        points.append(VTObservablePoint(observation, observable.id if observable else None,
            observable.assignment if observable else None, observable.nucleus if observable else None,
            _number(observable.chemical_shift_ppm) if observable else None,
            _number(observable.uncertainty_ppm) if observable else None,
            binding, action, tuple(dict.fromkeys(issues + blockers))))
    represented = {p.observation.identity.member_id for p in points}
    diagnostics = [f"{p.observation.identity.member_id}: {issue}" for p in points for issue in p.diagnostics]
    if association:
        diagnostics.extend(f"{b.member_id}: orphan_observable_binding" for b in association.bindings if b.member_id not in represented)
    return VTObservableSeries(series.id, track.id, association, tuple(points), tuple(diagnostics))


def confirm_member_associations(series, review, member_ids, *, confirmed=False):
    """Explicitly accept selected reviewed proposals, preflight the entire batch.

    Re-review after any context change. Never overwrite a conflicting assignment,
    collapse duplicates, or rebind a deleted observable implicitly.
    """
    if confirmed is not True:
        raise ValueError("explicit confirmation is required")
    if review.series_id != series.id or review.association is None:
        raise ValueError("review an associated track first")
    current = extract_vt_observable_series(series, review.track_id)
    if current != review:
        raise ValueError("review context changed; refresh before confirming")
    ids = tuple(member_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("select unique members to confirm")
    proposals = {p.observation.identity.member_id: p for p in current.points}
    for member_id in ids:
        if member_id not in proposals or proposals[member_id].action is None:
            raise ValueError(f"member {member_id} has no safe proposal; review conflicts")
    association = current.association
    bindings = {b.member_id: b for b in association.bindings}
    changed = []
    for member_id in ids:
        proposal = proposals[member_id]
        member = next(m for m in series.members if m.id == member_id)
        observable = sync_peak_observable(member.dataset, peak_id=proposal.observation.identity.peak_id,
            assignment=association.assignment)
        bindings[member_id] = ObservableBinding(member_id, member.dataset_id, observable.source_peak_id,
            observable.id, f"explicit_{proposal.action}")
        changed.append(observable.id)
    series.set_observable_association(replace(association, bindings=tuple(bindings.values())))
    return tuple(changed)


def association_diagnostics(series):
    """Recovery checks retain broken references without inferring replacements."""
    for association in series.observable_associations:
        try:
            review = extract_vt_observable_series(series, association.track_id)
        except KeyError:
            yield "association_track_missing", None, f"association={association.id}, track={association.track_id}"
            continue
        for point in review.points:
            for code in point.diagnostics:
                if code in {"invalid_track_point", "broken_observable_binding", "assignment_conflict", "nucleus_conflict", "duplicate_peak_observables", "observable_identity_conflict", "assignment_already_used", "missing_observable"}:
                    i = point.observation.identity
                    yield code, i.dataset_id, f"association={association.id}, track={association.track_id}, member={i.member_id}, peak={i.peak_id}, observable={point.observable_id}"
        for message in review.diagnostics:
            if "orphan_observable_binding" in message:
                yield "orphan_observable_binding", None, f"association={association.id}: {message}"
