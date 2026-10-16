"""Import historical analysis evidence without restoring live fit/source state."""
import json

from nmr_workbench.core.analysis_evidence import VTAnalysisEvidence
from nmr_workbench.analysis import extract_vt_track
from nmr_workbench.analysis.vt_models import observable_sources


def load_analysis_evidence(path):
    try:
        with open(path, encoding="utf-8-sig") as handle:
            return VTAnalysisEvidence.from_dict(json.load(handle))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed analysis result JSON; use Export results JSON") from exc


def review_analysis_evidence(evidence, series):
    """Compare saved scientific evidence to current sources; no recalculation."""
    source = next((s for s in series if s.id == evidence.series_id), None)
    if source is None:
        return ("Source series is not loaded. Open its .nmrsession to retain this result.",)
    warnings = []
    data = json.loads(evidence.evidence_json)
    for row in data["tracks"]:
        track_id = row["snapshot"]["track_id"]
        track = next((t for t in source.tracks if t.id == track_id), None)
        if track is None:
            warnings.append(f"Track {track_id[:8]} is unavailable; result remains historical.")
            continue
        current = json.loads(json.dumps(extract_vt_track(source, track).to_dict(), allow_nan=False))
        if current != row["snapshot"]:
            warnings.append(f"Track {track_id[:8]} source/state/revision differs; result remains historical.")
        association = source.observable_association(track_id)
        if (row.get("observable_association") != (association.to_dict() if association else None)
                or row.get("source_observables", []) != observable_sources(source, track_id)):
            warnings.append(f"Track {track_id[:8]} assignments/observable association differ.")
    return tuple(warnings)


def retain_analysis_evidence(evidence, series):
    source = next((s for s in series if s.id == evidence.series_id), None)
    if source is None:
        raise ValueError("Open the source .nmrsession before retaining this historical result")
    existing = next((e for e in source.analysis_evidence if e.id == evidence.id), None)
    if existing:
        if existing != evidence:
            raise ValueError("A different saved result already uses this evidence ID")
        return False
    source.add_analysis_evidence(evidence)
    return True
