"""Conservative, read-only suggestions; scores are distances, not identity."""

from dataclasses import dataclass
import hashlib
import math


@dataclass(frozen=True)
class TrackCandidate:
    series_id: str
    track_id: str
    member_id: str
    dataset_id: str
    peak_id: str
    score: float
    expected_ppm: float
    window_ppm: float
    reasons: tuple[str, ...]
    context: str


def candidate_context(series, track):
    """Snapshot all correspondence inputs, excluding transient UI selections."""
    data = (track.to_dict(), [(m.id, m.dataset_id, m.included, m.acquisition_order,
            m.dataset.spectral_revision, m.dataset.processing_history.to_list(),
            [(p.id, p.position_ppm, p.height, p.linewidth_hz) for p in m.dataset.peaks])
            for m in series.members])
    return hashlib.sha256(repr(data).encode()).hexdigest()


def suggest_candidates(series, track, member_id, *, window_ppm=.15):
    """Use bracketing acquisition neighbors, otherwise the closest valid anchor.

    No extrapolation or global trajectory solving. Equally plausible peaks are
    returned together. Empty anchors/peaks/out-of-window data yield no candidates.
    """
    if isinstance(window_ppm, bool) or not isinstance(window_ppm, (int, float)) or not math.isfinite(window_ppm) or window_ppm <= 0:
        raise ValueError("Candidate window must be finite positive ppm")
    member = track._member(series, member_id)
    if not member.included:
        return ()
    broken = {i.member_id for i in track.validate(series) if i.code != "member_excluded"}
    anchors = []
    for point in track.points:
        if point.state != "confirmed" or point.member_id == member_id or point.member_id in broken:
            continue
        other = next((m for m in series.members if m.id == point.member_id and m.included), None)
        if other is None:
            continue
        peak = next((p for p in other.dataset.peaks if p.id == point.peak_id), None)
        if peak is not None and math.isfinite(peak.position_ppm):
            anchors.append((other.acquisition_order, float(peak.position_ppm), point.member_id, peak.id))
    if not anchors:
        return ()
    order = member.acquisition_order
    before = sorted((a for a in anchors if a[0] < order), reverse=True)
    after = sorted(a for a in anchors if a[0] > order)
    if before and after:
        left, right = before[0], after[0]
        weight = (order - left[0]) / (right[0] - left[0])
        expected = left[1] + weight * (right[1] - left[1])
        anchor_reason = f"Acquisition continuity between {left[2]}/{left[3]} and {right[2]}/{right[3]}"
    else:
        anchor = min(anchors, key=lambda a: (abs(a[0] - order), a[0]))
        expected = anchor[1]
        anchor_reason = f"Proximity to confirmed member/peak {anchor[2]}/{anchor[3]}; no extrapolation"
    context = candidate_context(series, track)
    counts = {}
    for peak in member.dataset.peaks:
        counts[peak.id] = counts.get(peak.id, 0) + 1
    ranked = []
    for peak in member.dataset.peaks:
        if counts[peak.id] != 1 or not math.isfinite(peak.position_ppm):
            continue
        distance = abs(peak.position_ppm - expected)
        if distance <= window_ppm:
            ranked.append(TrackCandidate(series.id, track.id, member.id, member.dataset_id,
                peak.id, 1 - distance / window_ppm, expected, window_ppm,
                (f"Δppm={distance:.6g}; window={window_ppm:g} ppm", anchor_reason,
                 "Score is proximity only; crossings/coalescence require manual review"), context))
    return tuple(sorted(ranked, key=lambda c: (-c.score, c.peak_id)))
