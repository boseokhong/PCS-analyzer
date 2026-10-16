"""Temperature-aware reviewed pairs adapted to the existing PCS providers."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real

from ..core.vt_pcs import VTPCSPreparation
from ..core.experimental_pcs import ExperimentalPCSValue
from ..core.metadata import resolve_temperature
from ..integration.structure_aware_pcs_targets import PCSStructureTargetChoice, TARGET_SCOPES, SCOPE_SINGLE
from ..integration.pcs_structured_handoff import build_handoff_rows, validate_handoff_rows
from .pcs_derivation import builtin_pcs_providers, PCSDerivationContext
from .shift_analysis import ShiftComparisonRow, normalize_nucleus
from .vt_observables import extract_vt_observable_series


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _finite(value):
    return float(value) if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)) else None


@dataclass(frozen=True)
class VTPCSObservation:
    id: str
    dataset_id: str
    peak_id: str | None
    observable_id: str | None
    series_id: str | None
    track_id: str | None
    member_id: str | None
    assignment: str | None
    nucleus: str | None
    temperature_k: float | None
    observed_shift_ppm: float | None
    revision: int
    role: str
    valid: bool
    diagnostics: tuple[str, ...]
    provenance_json: str
    target_json: str | None = None


def vt_observations(series, track_id, *, targets=None):
    """Keep every VT point; only reviewed, valid observable bindings are usable."""
    review = extract_vt_observable_series(series, track_id)
    members = {m.id: m for m in series.members}
    result = []
    for point in review.points:
        o, i = point.observation, point.observation.identity
        member = members.get(i.member_id)
        issues = list(point.diagnostics)
        if o.temperature_k is not None and sum(p.observation.temperature_k == o.temperature_k for p in review.points) > 1:
            issues.append("repeated_observation_temperature")
        if not review.association or not point.binding:
            issues.append("incomplete_provenance")
        valid = bool(member and o.link_valid and o.member_included and o.temperature_status == "valid"
            and point.binding and point.action == "associate_existing")
        key = _json([i.series_id, i.track_id, i.member_id, i.dataset_id, point.observable_id])
        target = (targets or {}).get(key)
        provenance = {"vt_observation": asdict(o), "observable_id": point.observable_id,
            "observable_shift_ppm": point.observable_shift_ppm,
            "uncertainty_ppm": point.observable_uncertainty_ppm,
            "association": review.association.to_dict() if review.association else None,
            "binding": asdict(point.binding) if point.binding else None}
        result.append(VTPCSObservation(key, i.dataset_id, i.peak_id, point.observable_id,
            i.series_id, i.track_id, i.member_id, point.assignment, normalize_nucleus(point.nucleus),
            o.temperature_k, o.chemical_shift_ppm, o.dataset_revision or 0,
            member.dataset.experimental_role if member else "unspecified", valid, tuple(issues),
            _json(provenance), _json(asdict(target)) if target else None))
    return tuple(result)


def reference_observations(datasets, series=(), *, targets=None):
    """Explicit diamagnetic reference pool: VT overrides win over nominal metadata.

    A tracked observable has no dataset-only bypass around stale/unreviewed links.
    Standalone references require a live peak and synchronized observable shift.
    """
    result, represented = [], set()
    for item in series:
        for track in item.tracks:
            for obs in vt_observations(item, track.id, targets=targets):
                represented.add((obs.dataset_id, obs.observable_id))
                if obs.role == "diamagnetic":
                    result.append(obs)
    for dataset in datasets:
        if dataset.experimental_role != "diamagnetic":
            continue
        temperature = resolve_temperature(dataset.conditions)
        for observable in dataset.observables:
            if (dataset.id, observable.id) in represented:
                continue
            key = _json([None, None, None, dataset.id, observable.id])
            peaks = [p for p in dataset.peaks if p.id == observable.source_peak_id]
            shift = _finite(peaks[0].position_ppm) if len(peaks) == 1 else None
            issues = []
            if shift is None or shift != _finite(observable.chemical_shift_ppm):
                issues.append("stale_source_observation")
            if observable.dataset_id != dataset.id or sum(o.id == observable.id or o.source_peak_id == observable.source_peak_id for o in dataset.observables) != 1:
                issues.append("conflicting_observable_identity")
            nucleus = normalize_nucleus(observable.nucleus)
            if not nucleus or nucleus != normalize_nucleus(dataset.conditions.nucleus):
                issues.append("incompatible_nucleus")
            if not observable.assignment or temperature.status != "valid":
                issues.append("incomplete_provenance")
            observable_data = asdict(observable)
            observable_data["chemical_shift_ppm"] = _finite(observable.chemical_shift_ppm)
            observable_data["uncertainty_ppm"] = _finite(observable.uncertainty_ppm)
            provenance = {"observable": observable_data, "temperature_source": temperature.source,
                "history_digest": hashlib.sha256(json.dumps(dataset.processing_history.to_list(), sort_keys=True, default=str).encode()).hexdigest()}
            # Existing independent reference assignments have no historical revision
            # baseline; the explicit reviewed snapshot establishes it for this pair.
            target = (targets or {}).get(key)
            result.append(VTPCSObservation(key, dataset.id, observable.source_peak_id, observable.id,
                None, None, None, observable.assignment, nucleus, temperature.temperature_k, shift,
                dataset.spectral_revision, "diamagnetic", not issues, tuple(issues), _json(provenance),
                _json(asdict(target)) if target else None))
    if len({o.id for o in result}) != len(result):
        raise ValueError("duplicate reference observation identity")
    return tuple(result)


def target_from_dict(row):
    if row is None:
        return None
    values = dict(row)
    values["site_member_ref_ids"] = tuple(tuple(g) for g in values["site_member_ref_ids"])
    values["site_ids"] = tuple(values.get("site_ids", ()))
    target = PCSStructureTargetChoice(**values)
    if target.scope not in TARGET_SCOPES or not target.key or not target.site_member_ref_ids or any(not g for g in target.site_member_ref_ids):
        raise ValueError("unresolved structural target")
    refs = target.member_ref_ids
    if any(type(r) is not int or r < 0 for r in refs) or len(set(refs)) != len(refs):
        raise ValueError("invalid structural target references")
    if target.scope == SCOPE_SINGLE and not target.is_legacy_single_atom:
        raise ValueError("conflicting structural target scope")
    if target.is_legacy_single_atom and target.primary_ref_id != refs[0]:
        raise ValueError("conflicting structural target identity")
    return target


def _target_identity(target):
    return (target.site_member_ref_ids, target.averaging) if target else None


@dataclass(frozen=True)
class VTPairCandidate:
    reference: VTPCSObservation
    diagnostics: tuple[str, ...]

    @property
    def compatible(self):
        return not self.diagnostics


@dataclass(frozen=True)
class VTPairReview:
    paramagnetic: VTPCSObservation
    candidates: tuple[VTPairCandidate, ...]
    selected_reference: VTPCSObservation | None
    shift_difference_ppm: float | None
    diagnostics: tuple[str, ...]


def review_pair(paramagnetic, references, *, reference_id=None, target=None):
    """Exact Kelvin matching, no nearest temperature and no automatic choice."""
    references = tuple(references)
    if len({r.id for r in references}) != len(references):
        raise ValueError("duplicate reference observation identity")
    candidates = []
    issues = []
    if target:
        target = target_from_dict(asdict(target))
    if "repeated_observation_temperature" in paramagnetic.diagnostics:
        issues.append("repeated_observation_temperature")
    if not paramagnetic.valid:
        issues.append("stale_source_observation")
    if paramagnetic.role != "paramagnetic":
        issues.append("incompatible_role")
    if target is None:
        issues.append("unresolved_target")
    for ref in references:
        reasons = []
        if not ref.valid or not paramagnetic.valid or _finite(ref.observed_shift_ppm) is None or _finite(paramagnetic.observed_shift_ppm) is None:
            reasons.append("stale_source_observation")
        if ref.role != "diamagnetic" or paramagnetic.dataset_id == ref.dataset_id:
            reasons.append("incompatible_role")
        if not ref.nucleus or ref.nucleus != paramagnetic.nucleus:
            reasons.append("incompatible_nucleus")
        if not ref.assignment or ref.assignment.strip() != (paramagnetic.assignment or "").strip():
            reasons.append("conflicting_observable_identity")
        if _finite(ref.temperature_k) is None or ref.temperature_k <= 0 or ref.temperature_k != paramagnetic.temperature_k:
            reasons.append("temperature_mismatch")
        ref_target = target_from_dict(json.loads(ref.target_json)) if ref.target_json else None
        para_target = target_from_dict(json.loads(paramagnetic.target_json)) if paramagnetic.target_json else None
        if (target and any(t and _target_identity(t) != _target_identity(target) for t in (ref_target, para_target))) or (ref_target and para_target and _target_identity(ref_target) != _target_identity(para_target)):
            reasons.append("conflicting_structural_identity")
        candidates.append(VTPairCandidate(ref, tuple(dict.fromkeys(reasons))))
    compatible = [c.reference for c in candidates if c.compatible]
    if not compatible:
        issues.append("missing_reference")
    elif len(compatible) > 1:
        issues.extend(("repeated_reference_temperature", "ambiguous_reference"))
    selected = next((r for r in compatible if r.id == reference_id), None)
    if reference_id and selected is None:
        issues.append("invalid_reference_selection")
    if selected is None:
        issues.append("reference_selection_required")
    elif "ambiguous_reference" in issues:
        issues.remove("ambiguous_reference")
    delta = paramagnetic.observed_shift_ppm - selected.observed_shift_ppm if selected and paramagnetic.valid and paramagnetic.role == "paramagnetic" else None
    return VTPairReview(paramagnetic, tuple(candidates), selected, delta, tuple(issues))


def prepare_pair(review, *, target=None, provider_id=None, correction_ppm=None, manual_pcs_ppm=None, note="", preparation_id=None):
    """Create reproducible review evidence; provider use requires an explicit basis.

    Returning evidence never publishes to PCS Analyzer or mutates source data.
    """
    if review.selected_reference is None or review.shift_difference_ppm is None:
        raise ValueError("explicit compatible reference selection is required")
    para, ref = review.paramagnetic, review.selected_reference
    checked = review_pair(para, [ref], reference_id=ref.id, target=target)
    if checked.shift_difference_ppm is None:
        raise ValueError("pair/target compatibility changed")
    if checked.shift_difference_ppm != review.shift_difference_ppm:
        raise ValueError("inconsistent shift difference review")
    if target:
        target = target_from_dict(asdict(target))
    value = None
    if provider_id is not None:
        if not isinstance(note, str) or not note.strip():
            raise ValueError("an explicit scientific derivation basis note is required")
        providers = {p.provider_id: p for p in builtin_pcs_providers()}
        if provider_id not in providers:
            raise ValueError("unknown PCS derivation provider")
        for supplied in (correction_ppm, manual_pcs_ppm):
            if supplied is not None and _finite(supplied) is None:
                raise ValueError("derivation inputs must be finite ppm values")
        row = ShiftComparisonRow(para.assignment, para.observed_shift_ppm, ref.observed_shift_ppm,
            review.shift_difference_ppm, "Matched", para.observable_id, ref.observable_id, para.dataset_id, ref.dataset_id)
        context = PCSDerivationContext(para.dataset_id, ref.dataset_id, para.temperature_k,
            {para.assignment: correction_ppm} if correction_ppm is not None else {},
            {para.assignment: manual_pcs_ppm} if manual_pcs_ppm is not None else {})
        value = providers[provider_id].derive(row, context)
        if _finite(value.pcs_ppm) is None:
            raise ValueError("derived PCS must be finite")
    evidence = {"schema_version": 1, "paramagnetic": asdict(para), "reference": asdict(ref),
        "target": asdict(target) if target else None, "basis": provider_id, "note": note,
        "shift_difference_ppm": review.shift_difference_ppm, "contact_correction_ppm": correction_ppm,
        "manual_pcs_ppm": manual_pcs_ppm,
        "pcs_value": asdict(value) if value else None}
    record = VTPCSPreparation(para.series_id, para.track_id, para.member_id, _json(evidence),
        **({"id": preparation_id} if preparation_id else {}))
    if value:
        evidence["pcs_value"]["id"] = record.id
        record = VTPCSPreparation(record.series_id, record.track_id, record.member_id, _json(evidence), record.id)
    return record


def preparation_diagnostics(record, observations):
    """Stored evidence must match current sources before a PCS-ready adapter runs."""
    evidence = json.loads(record.evidence_json)
    current = {o.id: asdict(o) for o in observations}
    issues = []
    for key in ("paramagnetic", "reference"):
        saved = evidence[key]
        if saved.get("id") not in current:
            issues.append("missing_source_observation")
        elif current[saved["id"]] != saved:
            # JSON changes tuples to lists; compare canonical JSON rather than types.
            if _json(current[saved["id"]]) != _json(saved):
                issues.append("stale_source_observation")
    if evidence["target"] is None:
        issues.append("unresolved_target")
    if evidence["pcs_value"] is None:
        issues.append("pcs_not_defined")
    return tuple(dict.fromkeys(issues))


def pcs_ready_row(record, observations, *, target_choices=None):
    """Canonical ExperimentalPCSValue → canonical handoff row, without sending."""
    issues = preparation_diagnostics(record, observations)
    if issues:
        raise ValueError("; ".join(issues))
    evidence = json.loads(record.evidence_json)
    by_id = {o.id: o for o in observations}
    target = target_from_dict(evidence["target"])
    if target_choices is not None and not any(t.key == target.key and _target_identity(t) == _target_identity(target) for t in target_choices):
        raise ValueError("unresolved_or_changed_structural_target")
    review = review_pair(by_id[evidence["paramagnetic"]["id"]],
        [by_id[evidence["reference"]["id"]]], reference_id=evidence["reference"]["id"], target=target)
    regenerated = prepare_pair(review, target=target, provider_id=evidence["basis"],
        correction_ppm=evidence.get("contact_correction_ppm"), manual_pcs_ppm=evidence.get("manual_pcs_ppm"),
        note=evidence["note"], preparation_id=record.id)
    if json.loads(regenerated.evidence_json)["pcs_value"] != evidence["pcs_value"]:
        raise ValueError("inconsistent derivation evidence")
    value = ExperimentalPCSValue(**evidence["pcs_value"])
    value.provenance = {**value.provenance, "vt_preparation_id": record.id,
        "vt_pairing": evidence, "derivation_basis_note": evidence["note"]}
    rows = build_handoff_rows({value.assignment: value}, {value.assignment: target})
    return validate_handoff_rows(rows)[0]


def all_observations(datasets, series):
    return tuple(o for s in series for t in s.tracks for o in vt_observations(s, t.id)) + reference_observations(datasets, series)


def recovery_diagnostics(datasets, series):
    observations = all_observations(datasets, series)
    for item in series:
        for record in item.pcs_preparations:
            issues = list(preparation_diagnostics(record, observations))
            if not issues:
                try:
                    pcs_ready_row(record, observations)
                except ValueError as exc:
                    issues.append(f"invalid_preparation: {exc}")
            for code in issues:
                yield code, json.loads(record.evidence_json)["paramagnetic"].get("dataset_id"), f"preparation={record.id}, series={item.id}, track={record.track_id}, member={record.member_id}"
