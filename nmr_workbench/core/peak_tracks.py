"""Series resonance correspondence, independent of chemical assignment and UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from uuid import uuid4


def _id(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _text(value, name):
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")


def _history_fingerprint(dataset):
    text = json.dumps(dataset.processing_history.to_list(), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _peak_fingerprint(peak):
    # A digest for invalidation, never a second authoritative measurement store.
    text = repr((peak.position_ppm, peak.height, peak.integral, peak.linewidth_hz))
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class TrackProvenance:
    origin: str = "manual"
    method: str | None = None
    note: str | None = None

    def __post_init__(self):
        if self.origin not in {"manual", "automatic", "imported"}:
            raise ValueError("unknown tracking provenance origin")
        _text(self.method, "method")
        _text(self.note, "note")


@dataclass(frozen=True)
class PeakTrackPoint:
    member_id: str
    dataset_id: str
    acquisition_order: int
    peak_id: str | None = None
    state: str = "missing"
    provenance: TrackProvenance = field(default_factory=TrackProvenance)
    dataset_revision: int | None = None
    peak_fingerprint: str | None = None
    history_fingerprint: str | None = None
    reason: str | None = None

    def __post_init__(self):
        _id(self.member_id, "member_id")
        _id(self.dataset_id, "dataset_id")
        if type(self.acquisition_order) is not int or self.acquisition_order < 0:
            raise ValueError("invalid point acquisition_order")
        if self.state not in {"confirmed", "missing", "ambiguous", "stale"}:
            raise ValueError("unknown track point state")
        if not isinstance(self.provenance, TrackProvenance):
            raise ValueError("invalid point provenance")
        if self.peak_id is not None:
            _id(self.peak_id, "peak_id")
        if self.state == "confirmed" and self.peak_id is None:
            raise ValueError("confirmed points require a peak link")
        if self.state in {"missing", "ambiguous"} and self.peak_id is not None:
            raise ValueError("missing/ambiguous points cannot confirm a peak")
        if self.dataset_revision is not None and (type(self.dataset_revision) is not int or self.dataset_revision < 0):
            raise ValueError("invalid dataset revision")
        for name in ("peak_fingerprint", "history_fingerprint", "reason"):
            _text(getattr(self, name), name)
        if self.state == "confirmed" and any(value is None for value in
                (self.dataset_revision, self.peak_fingerprint, self.history_fingerprint)):
            raise ValueError("confirmed points require input fingerprints/revision")


@dataclass(frozen=True)
class TrackIssue:
    track_id: str
    member_id: str
    code: str


@dataclass
class PeakTrack:
    series_id: str
    label: str | None = None
    provenance: TrackProvenance = field(default_factory=TrackProvenance)
    id: str = field(default_factory=lambda: str(uuid4()))
    _points: list[PeakTrackPoint] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        _id(self.id, "track id")
        _id(self.series_id, "series_id")
        _text(self.label, "label")
        if not isinstance(self.provenance, TrackProvenance):
            raise ValueError("invalid track provenance")

    @property
    def points(self):
        return tuple(sorted(self._points, key=lambda p: p.acquisition_order))

    def _member(self, series, member_id):
        if series.id != self.series_id:
            raise ValueError("track belongs to another series")
        member = next((m for m in series.members if m.id == member_id), None)
        if member is None:
            raise KeyError(f"unknown member: {member_id}")
        return member

    def get_point(self, member_id):
        """Return the member's immutable point, or None if not yet reviewed."""
        return next((p for p in self._points if p.member_id == member_id), None)

    def _point(self, member_id):
        # Compatibility for existing consumers; new callers use get_point.
        return self.get_point(member_id)

    def _append(self, point):
        if self._point(point.member_id) is not None:
            raise ValueError("duplicate member link in track")
        self._points.append(point)

    def add_point(self, series, member_id, *, peak_id=None, state="missing", provenance=None):
        member = self._member(series, member_id)
        if self._point(member_id) is not None:
            raise ValueError("duplicate member link in track")
        if peak_id is not None:
            if state not in {"missing", "confirmed"}:
                raise ValueError("linked points must be confirmed")
            point = self._linked(member, peak_id, provenance or self.provenance)
        else:
            point = PeakTrackPoint(member_id, member.dataset_id, member.acquisition_order,
                                   state=state, provenance=provenance or self.provenance)
        self._append(point)
        return point

    @staticmethod
    def _linked(member, peak_id, provenance):
        peaks = [p for p in member.dataset.peaks if p.id == peak_id]
        if len(peaks) != 1:
            raise ValueError("peak ID must resolve to exactly one dataset-local peak")
        return PeakTrackPoint(member.id, member.dataset_id, member.acquisition_order,
                              peak_id, "confirmed", provenance,
                              member.dataset.spectral_revision, _peak_fingerprint(peaks[0]),
                              _history_fingerprint(member.dataset))

    def link_peak(self, series, member_id, peak_id, *, provenance=None):
        member = self._member(series, member_id)
        current = self._point(member_id)
        point = self._linked(member, peak_id, provenance or (current.provenance if current else self.provenance))
        if current is None:
            self._append(point)
        else:
            self._points[self._points.index(current)] = point
        return point

    def remove_point(self, member_id):
        point = self._point(member_id)
        if point is None:
            raise KeyError(f"unknown track point: {member_id}")
        self._points.remove(point)
        return point

    def mark_missing(self, member_id, *, reason="marked_missing", provenance=None):
        return self._mark_unlinked(member_id, "missing", reason, provenance)

    def mark_ambiguous(self, member_id, *, reason="marked_ambiguous", provenance=None):
        return self._mark_unlinked(member_id, "ambiguous", reason, provenance)

    def unlink_peak(self, member_id, *, provenance=None):
        return self.mark_missing(member_id, reason="unlinked", provenance=provenance)

    def _mark_unlinked(self, member_id, state, reason, provenance=None):
        current = self._point(member_id)
        if current is None:
            raise KeyError(f"unknown track point: {member_id}")
        point = replace(current, peak_id=None, state=state, reason=reason,
                        provenance=provenance or current.provenance,
                        dataset_revision=None, peak_fingerprint=None, history_fingerprint=None)
        self._points[self._points.index(current)] = point
        return point

    def ordered_points(self, series, *, order="Acquisition", included_only=False):
        if series.id != self.series_id:
            raise ValueError("track belongs to another series")
        if order == "Acquisition":
            members = series.members
        elif order in {"Temperature ↑", "Temperature ↓"}:
            members = series.temperature_sorted_members(descending=order == "Temperature ↓")
        else:
            raise ValueError("unknown track order")
        points = [self._point(m.id) for m in members if m.included or not included_only]
        # Removed-member records remain inspectable, last; included-only hides them.
        orphans = [p for p in self.points if not any(m.id == p.member_id for m in members)]
        return tuple(p for p in points if p is not None) + (() if included_only else tuple(orphans))

    def validate(self, series):
        if series.id != self.series_id:
            raise ValueError("track belongs to another series")
        issues = []
        for point in self.points:
            member = next((m for m in series.members if m.id == point.member_id), None)
            code = None
            if member is None:
                code = "member_removed"
            elif member.dataset_id != point.dataset_id:
                code = "dataset_mismatch"
            elif point.peak_id is not None:
                peaks = [p for p in member.dataset.peaks if p.id == point.peak_id]
                if not peaks:
                    code = "peak_deleted"
                elif len(peaks) > 1:
                    code = "duplicate_peak_id"
                elif point.dataset_revision != member.dataset.spectral_revision or point.history_fingerprint != _history_fingerprint(member.dataset):
                    code = "spectrum_changed"
                elif point.peak_fingerprint != _peak_fingerprint(peaks[0]):
                    code = "peak_changed"
            if code:
                issues.append(TrackIssue(self.id, point.member_id, code))
            if member is not None and not member.included:
                issues.append(TrackIssue(self.id, point.member_id, "member_excluded"))
            if point.state == "stale":
                issues.append(TrackIssue(self.id, point.member_id, "stale"))
        return tuple(issues)

    def invalidate(self, *, reason, dataset_id=None, member_id=None, peak_ids=None):
        """Explicit event API. Retain links and never silently restore confirmation."""
        _id(reason, "invalidation reason")
        for index, point in enumerate(self._points):
            if dataset_id is not None and point.dataset_id != dataset_id:
                continue
            if member_id is not None and point.member_id != member_id:
                continue
            if peak_ids is not None and point.peak_id not in peak_ids:
                continue
            if point.state == "stale":
                continue
            if point.peak_id is not None or reason in {"member_removed", "dataset_mismatch", "dataset_unloaded"}:
                self._points[index] = replace(point, state="stale", reason=reason)

    def update_stale(self, series):
        for issue in self.validate(series):
            if issue.code not in {"member_excluded", "stale"}:
                self.invalidate(member_id=issue.member_id, reason=issue.code)
        return self.validate(series)

    def to_dict(self):
        return {"schema_version": 1, "id": self.id, "series_id": self.series_id,
                "label": self.label, "provenance": asdict(self.provenance),
                "points": [asdict(p) for p in self.points]}

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
            raise ValueError("unsupported track schema_version")
        try:
            track = cls(series_id=payload["series_id"], id=payload["id"], label=payload.get("label"),
                        provenance=TrackProvenance(**payload["provenance"]))
            if not isinstance(payload["points"], list):
                raise ValueError("track points must be an array")
            for row in payload["points"]:
                row = dict(row)
                row["provenance"] = TrackProvenance(**row["provenance"])
                track._append(PeakTrackPoint(**row))
            return track
        except (TypeError, KeyError) as exc:
            raise ValueError("malformed track definition") from exc
