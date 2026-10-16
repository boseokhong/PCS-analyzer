"""Analysis-session selection/presentation/export over immutable M6 results."""

import csv
import json
from nmr_workbench.io.exchange_files import atomic_text_export

from nmr_workbench.analysis.vt_models import (VTModelConfig, VTModelResult, fit_vt_model,
    compare_vt_models, observable_sources)

from nmr_workbench.analysis import extract_vt_track, prepare_vt_coordinates, fit_vt_linear
from .series_manager import SeriesManager

COORDINATES = {"δ vs T": "delta_vs_T", "δ vs 1/T": "delta_vs_inverse_T",
               "δT vs 1/T": "deltaT_vs_inverse_T"}
FIT_MODELS = {"Linear (coordinate)": "linear", "Curie": "curie", "Curie-Weiss": "curie_weiss"}


def format_value(value):
    return "—" if value is None else f"{value:.6g}"


class VTAnalysisModel(SeriesManager):
    """Masks/ranges are local to this analysis window, never track/member edits."""
    def __init__(self, state):
        super().__init__(state)
        self.coordinate = "δ vs T"
        self.selections = {}
        self.exclusions = {}
        self.ranges = {}
        self.results = ()
        self.selected_identity = None
        self.fit_model = "linear"
        self.weighting = "unweighted"
        self.theta_bounds_k = None
        self.comparison_models = ()
        self.comparison_results = ()
        self.source_observables = {}
        self.source_associations = {}

    @property
    def selected_track_ids(self):
        return self.selections.get(self.state.selected_series_id, ())

    @property
    def temperature_range_k(self):
        return self.ranges.get(self.state.selected_series_id)

    def refresh(self):
        series = self.active_series
        self.source_observables = {}
        self.source_associations = {}
        if series is None:
            self.results = ()
            self.selected_identity = None
            self.comparison_results = ()
            return
        ids = {t.id for t in series.tracks}
        selected = tuple(t for t in self.selected_track_ids if t in ids)
        if series.id not in self.selections:
            selected = (self.state.selected_track_id,) if self.state.selected_track_id in ids else tuple(t.id for t in series.tracks[:1])
        elif self.state.selected_track_id in ids and self.state.selected_track_id not in selected:
            selected += (self.state.selected_track_id,)
        self.selections[series.id] = selected
        results = []
        comparisons = []
        for track_id in selected:
            snapshot = extract_vt_track(series, series.get_track(track_id))
            self.source_observables[track_id] = observable_sources(series, track_id)
            association = series.observable_association(track_id)
            self.source_associations[track_id] = association.to_dict() if association else None
            members = {o.identity.member_id for o in snapshot.observations}
            mask = self.exclusions.get((series.id, track_id), set()) & members
            coordinates = prepare_vt_coordinates(snapshot, domain=COORDINATES[self.coordinate],
                excluded_member_ids=mask, temperature_range_k=self.temperature_range_k)
            results.append(fit_vt_linear(coordinates) if self.fit_model == "linear" else
                           fit_vt_model(coordinates, self.configuration(self.fit_model)))
            comparisons.append(compare_vt_models(coordinates, [self.configuration(m) for m in self.comparison_models]))
        self.results = tuple(results)
        self.comparison_results = tuple(comparisons)
        if self.selected_identity not in {p.observation.identity for r in self.results for p in r.coordinates.points}:
            self.selected_identity = None

    def configuration(self, model_id):
        return VTModelConfig(model_id, weighting=self.weighting,
            theta_bounds_k=self.theta_bounds_k if model_id == "curie_weiss" else None)

    def set_model(self, model_id, *, weighting="unweighted", theta_bounds_k=None):
        if model_id not in FIT_MODELS.values():
            raise ValueError("Unknown analysis model")
        if model_id == "linear" and weighting != "unweighted":
            raise ValueError("Legacy coordinate linear fits are unweighted; comparison supports measured-sigma fits")
        # Validate before changing UI state, including bounds used in comparisons.
        VTModelConfig("curie_weiss", weighting=weighting, theta_bounds_k=theta_bounds_k)
        self.fit_model, self.weighting, self.theta_bounds_k = model_id, weighting, theta_bounds_k
        self.refresh()

    def set_comparison(self, model_ids):
        model_ids = tuple(model_ids)
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("Duplicate comparison models")
        for model_id in model_ids:
            self.configuration(model_id)
        self.comparison_models = model_ids
        self.refresh()

    def select_tracks(self, track_ids, *, active_id=None):
        series = self.active_series
        if series is None:
            raise ValueError("Select a VT series")
        track_ids = tuple(dict.fromkeys(track_ids))
        if any(t not in {track.id for track in series.tracks} for t in track_ids):
            raise ValueError("Unknown analysis track")
        if active_id is not None and active_id not in track_ids:
            raise ValueError("Active track must belong to selected tracks")
        self.selections[series.id] = track_ids
        self.state.selected_track_id = active_id or (track_ids[0] if track_ids else None)
        self.refresh()

    def set_coordinate(self, coordinate):
        if coordinate not in COORDINATES:
            raise ValueError("Unknown analysis coordinate")
        self.coordinate = coordinate
        self.refresh()

    def set_range(self, temperature_range_k):
        # M6 is the sole range validation/calculation path, even with no tracks.
        series = self.active_series
        if series is None:
            raise ValueError("Select a VT series")
        from nmr_workbench.core import PeakTrack
        data = extract_vt_track(series, series.tracks[0] if series.tracks else PeakTrack(series.id))
        validated = prepare_vt_coordinates(data, temperature_range_k=temperature_range_k).temperature_range_k
        self.ranges[series.id] = validated
        self.refresh()

    def point(self, identity):
        for result in self.results:
            for point in result.coordinates.points:
                if point.observation.identity == identity:
                    return point
        raise ValueError("Point is no longer in the current analysis; refresh")

    def set_included(self, identity, included):
        point = self.point(identity)
        if any(reason not in {"analysis_excluded", "outside_temperature_range"} for reason in point.exclusion_reasons):
            raise ValueError("This observation is invalid/excluded at source; review tracking or member state first")
        mask = self.exclusions.setdefault((identity.series_id, identity.track_id), set())
        if included:
            mask.discard(identity.member_id)
        else:
            mask.add(identity.member_id)
        self.refresh()

    def reset_masks(self):
        for track_id in self.selected_track_ids:
            self.exclusions.pop((self.state.selected_series_id, track_id), None)
        self.ranges.pop(self.state.selected_series_id, None)
        self.refresh()

    def navigation(self, identity):
        self.point(identity)
        series = self.active_series
        member = next((m for m in series.members if m.id == identity.member_id), None) if series else None
        if member is None or member.dataset_id != identity.dataset_id or not any(d is member.dataset for d in self.state.datasets):
            raise ValueError("Original dataset/member is no longer available")
        peak_id = identity.peak_id if sum(p.id == identity.peak_id for p in member.dataset.peaks) == 1 else None
        self.state.selected_track_id = identity.track_id
        self.select_member(series.id, member.id)
        self.state.selected_peak_id = peak_id
        self.selected_identity = identity
        return member.dataset_id, peak_id

    def export_rows(self):
        rows = []
        for result in self.results:
            fitted = dict(zip(result.included_identities, zip(result.fitted_values, result.residuals)))
            physical = isinstance(result, VTModelResult)
            raw_fitted = dict(zip(result.included_identities, zip(result.raw_fitted_values, result.raw_residuals))) if physical else {}
            sources = self.source_observables[result.coordinates.data.track_id]
            for point in result.coordinates.points:
                observed, identity = point.observation, point.observation.identity
                prediction, residual = fitted.get(identity, (None, None))
                row = {"series_id": identity.series_id, "track_id": identity.track_id,
                    "track_label": result.coordinates.data.track_label, "member_id": identity.member_id,
                    "member_label": observed.member_label, "dataset_id": identity.dataset_id,
                    "peak_id": identity.peak_id, "temperature_k": observed.temperature_k,
                    "temperature_source": observed.temperature_source, "coordinate": result.domain.name,
                    "x": point.x, "x_unit": result.domain.x_unit, "y": point.y, "y_unit": result.domain.y_unit,
                    "chemical_shift_ppm": observed.chemical_shift_ppm, "shift_uncertainty_ppm": observed.shift_uncertainty_ppm,
                    "y_uncertainty": point.y_uncertainty, "state": observed.point_state, "link_valid": observed.link_valid,
                    "member_included": observed.member_included, "fit_included": point.included,
                    "exclusion_reasons": ";".join(point.exclusion_reasons), "fitted_value": prediction,
                    "residual": residual, "fit_status": result.status, "slope": result.slope,
                    "intercept": result.intercept, "r_squared": result.r_squared, "n": result.point_count,
                    "decision_method": observed.provenance.method if observed.provenance else None,
                    "diagnostics": ";".join(dict.fromkeys(d.code for d in result.diagnostics
                        if not d.identities or identity in d.identities))}
                row.update(model=result.config.model_id if physical else "linear_coordinate",
                    fit_quantity="observed_chemical_shift", weighting=result.config.weighting if physical else "unweighted",
                    fit_configuration=json.dumps(result.config.to_dict()) if physical else "",
                    parameters=json.dumps([vars(p) for p in result.parameters]) if physical else "",
                    covariance=json.dumps(result.covariance) if physical else "",
                    metric_unit="ppm" if physical else result.domain.y_unit,
                    rss=result.rss if physical else None, rmse=result.rmse if physical else None,
                    aic=result.aic if physical else None, aicc=result.aicc if physical else None,
                    raw_fitted_shift_ppm=raw_fitted.get(identity, (None,None))[0],
                    raw_residual_ppm=raw_fitted.get(identity, (None,None))[1],
                    observable_ids=";".join(o["id"] for o in sources
                        if o.get("source_peak_id") == identity.peak_id and o.get("dataset_id") == identity.dataset_id),
                    assignments=";".join(o.get("assignment") or "" for o in sources
                        if o.get("source_peak_id") == identity.peak_id and o.get("dataset_id") == identity.dataset_id),
                    nuclei=";".join(o.get("nucleus") or "" for o in sources
                        if o.get("source_peak_id") == identity.peak_id and o.get("dataset_id") == identity.dataset_id))
                rows.append(row)
        return tuple(rows)

    def export_csv(self, path):
        rows = self.export_rows()
        if not rows:
            raise ValueError("Select tracks with observations before exporting")
        with atomic_text_export(path, encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def evidence(self):
        from nmr_workbench.core.analysis_evidence import VTAnalysisEvidence
        from datetime import datetime, timezone
        if not self.results:
            raise ValueError("Select analysis tracks before saving evidence")
        series = self.active_series
        tracks = []
        for index, result in enumerate(self.results):
            track_id = result.coordinates.data.track_id
            tracks.append({"snapshot": result.coordinates.data.to_dict(),
                "observable_association": self.source_associations[track_id],
                "source_observables": self.source_observables[track_id],
                "results": [result.to_dict()] + [r.to_dict() for r in self.comparison_results[index]]})
        payload = {"schema_version": 1, "series_id": series.id, "quantity": "observed_chemical_shift",
            "created_at": datetime.now(timezone.utc).isoformat(), "coordinate": self.coordinate,
            "selected_model": self.fit_model, "weighting": self.weighting,
            "comparison_models": self.comparison_models, "tracks": tracks}
        return VTAnalysisEvidence(series.id, json.dumps(payload, allow_nan=False))

    def save_evidence(self):
        evidence = self.evidence()
        self.active_series.add_analysis_evidence(evidence)
        return evidence

    def export_json(self, path):
        evidence = self.evidence().to_dict()
        with atomic_text_export(path) as handle:
            json.dump(evidence, handle, indent=2, allow_nan=False)
