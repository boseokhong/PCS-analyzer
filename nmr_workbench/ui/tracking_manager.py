"""UI-independent manual tracking workflow and explicit suggestion acceptance."""

import json

from nmr_workbench.core import TrackProvenance
from nmr_workbench.analysis.track_candidates import candidate_context, suggest_candidates
from .series_manager import SeriesManager


class TrackingManager(SeriesManager):
    @property
    def active_track(self):
        series = self.active_series
        return next((t for t in series.tracks if t.id == self.state.selected_track_id), None) if series else None

    def select_track(self, track_id):
        track = self._series().get_track(track_id)
        self.state.selected_track_id = track.id
        return track

    def _series(self):
        if self.active_series is None:
            raise ValueError("Select a VT series first")
        return self.active_series

    def _track(self):
        if self.active_track is None:
            raise ValueError("Select or create a track first")
        return self.active_track

    def create_track(self, label=None):
        track = self._series().create_track(label=label)
        self.state.selected_track_id = track.id
        return track

    def delete_track(self):
        self._series().delete_track(self._track().id)
        self.state.selected_track_id = None

    def rename_track(self, label):
        if label is not None and not isinstance(label, str):
            raise ValueError("Track label must be text")
        self._track().label = label

    def select_peak(self, member_id, peak_id=None):
        series = self._series()
        member = next((m for m in series.members if m.id == member_id), None)
        if member is None:
            raise ValueError("Select a current series member")
        if peak_id is not None and sum(p.id == peak_id for p in member.dataset.peaks) != 1:
            raise ValueError("Peak is missing or its ID is not unique")
        self.select_member(series.id, member_id)
        self.state.selected_peak_id = peak_id
        return member

    def link_selected(self):
        if self.state.selected_peak_id is None or self.state.selected_member_id is None:
            raise ValueError("Select an existing peak in a series member")
        return self._track().link_peak(self._series(), self.state.selected_member_id,
            self.state.selected_peak_id, provenance=TrackProvenance("manual", "manual_link"))

    def mark(self, state):
        if state not in {"missing", "ambiguous", "unlinked"}:
            raise ValueError("Unknown review state")
        member_id = self.state.selected_member_id
        track = self._track()
        if member_id is None:
            raise ValueError("Select a current series member")
        if track.get_point(member_id) is None:
            track.add_point(self._series(), member_id)
        provenance = TrackProvenance("manual", f"manual_{state}")
        if state == "ambiguous":
            return track.mark_ambiguous(member_id, provenance=provenance)
        return track.unlink_peak(member_id, provenance=provenance) if state == "unlinked" else track.mark_missing(member_id, provenance=provenance)

    def suggestions(self, *, window_ppm=.15):
        return suggest_candidates(self._series(), self._track(), self.state.selected_member_id,
                                  window_ppm=window_ppm)

    def accept(self, candidate):
        series, track = self._series(), self._track()
        if (candidate.series_id, candidate.track_id, candidate.member_id) != (series.id, track.id, self.state.selected_member_id):
            raise ValueError("Suggestion belongs to a different track/member; suggest again")
        if candidate.context != candidate_context(series, track):
            raise ValueError("Suggestion inputs changed; suggest again before accepting")
        if candidate not in self.suggestions(window_ppm=candidate.window_ppm):
            raise ValueError("Suggestion is no longer valid")
        provenance = TrackProvenance("manual", "assisted_ppm_v1", json.dumps({
            "score": candidate.score, "expected_ppm": candidate.expected_ppm,
            "window_ppm": candidate.window_ppm, "reasons": candidate.reasons,
            "input_context": candidate.context, "accepted_by_user": True}, sort_keys=True))
        point = track.link_peak(series, candidate.member_id, candidate.peak_id, provenance=provenance)
        self.state.selected_peak_id = candidate.peak_id
        return point
