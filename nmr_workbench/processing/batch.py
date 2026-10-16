"""Per-member transactions orchestrating existing processing/picking APIs; no Tk."""
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field, replace
import hashlib
from datetime import datetime, timezone
import numpy as np

from ..core.batch_recipe import BatchRecipe
from ..core import ProcessingHistory
from ..analysis.peak_picking import PeakPickingParameters, pick_positive_peaks_1d
from ..analysis.observable_sync import assignments_from_observables, sync_dataset_observables
from ..reference_presets import load_reference_preset_library
from ..ui.processing_session import ProcessingSession


def _seal(dataset):
    # Supported spectrum writes advance revision; silent external array edits are
    # outside the existing mutation contract. Metadata/peak/observable edits seal too.
    spectra = tuple((id(s), id(s.axis), id(s.intensities)) if s else None for s in
                    (dataset.raw_spectrum, dataset.processed_spectrum, dataset.restored_committed_spectrum))
    state = (dataset.id, dataset.spectral_revision, spectra, dataset.processing_history.to_list(),
             dataset.conditions, dataset.acquisition_parameters, dataset.vendor_metadata,
             dataset.vendor, dataset.source_provenance, dataset.experimental_role,
             [asdict(p) for p in dataset.peaks], [asdict(o) for o in dataset.observables])
    return hashlib.sha256(repr(state).encode()).hexdigest()


def _snapshot(dataset, committed=None):
    staged = copy(dataset)
    for name in ("conditions", "acquisition_parameters", "vendor_metadata", "source_provenance", "peaks", "observables"):
        setattr(staged, name, deepcopy(getattr(dataset, name)))
    staged.processing_history = ProcessingHistory(dataset.processing_history.steps)
    for name in ("raw_spectrum", "processed_spectrum", "restored_committed_spectrum"):
        setattr(staged, name, deepcopy(getattr(dataset, name)))
    if committed is not None:
        staged.restored_committed_spectrum = deepcopy(committed)
    return staged


@dataclass(frozen=True)
class BatchMemberResult:
    member_id: str
    dataset_id: str
    status: str
    message: str
    warnings: tuple[str, ...] = ()
    spectrum: object = field(default=None, repr=False, compare=False)
    peak_count: int = 0
    removed_peak_ids: tuple[str, ...] = ()
    revision_before: int = 0
    revision_after: int = 0
    input_spectrum: object = field(default=None, repr=False, compare=False)


@dataclass
class BatchPreview:
    series_id: str
    recipe: BatchRecipe
    inputs: dict
    results: tuple[BatchMemberResult, ...] = ()
    staged: dict = field(default_factory=dict, repr=False)
    applied: bool = False


def prepare_batch(series, recipe, *, committed_by_dataset=None):
    """Capture isolated input on the caller thread before starting numerical work."""
    if not isinstance(recipe, BatchRecipe):
        raise ValueError("Expected a batch recipe")
    config = recipe.configuration
    ids = config.get("member_ids") or [m.id for m in series.members]
    members = {m.id: m for m in series.members}
    if any(i not in members for i in ids):
        raise ValueError("Batch target member is no longer in this series")
    inputs = {}
    for member in series.members:
        if member.id not in ids:
            continue
        dataset = member.dataset
        skip = not member.included and not config.get("include_excluded", False)
        inputs[member.id] = (dataset, _seal(dataset), member.included,
            None if skip else _snapshot(dataset, (committed_by_dataset or {}).get(dataset.id)), skip)
    return BatchPreview(series.id, recipe, inputs)


def _reference_parameters(dataset, session, parameters, override):
    params = dict(parameters)
    observed = params.pop("observed_ppm", None)
    if observed is not None and "observed_ppm" not in override:
        raise ValueError("Observed reference ppm must be an explicit per-member override")
    low, high = params.pop("window_low_ppm", None), params.pop("window_high_ppm", None)
    if observed is None:
        if low is None or high is None or not np.isfinite(float(low)) or not np.isfinite(float(high)) or float(low) > float(high):
            raise ValueError("Specify a per-member observed ppm or a finite reference search window")
        candidates = [p for p in pick_positive_peaks_1d(session.committed_spectrum)
                      if float(low) <= p.position_ppm <= float(high)]
        if len(candidates) != 1:
            raise ValueError(f"Reference {'missing' if not candidates else 'ambiguous'}: {len(candidates)} candidates in window")
        observed = candidates[0].position_ppm
    if params.get("preset_id"):
        library = load_reference_preset_library()
        nucleus = dataset.conditions.nucleus
        rows = library.resonances_for(params["preset_id"], nucleus)
        matches = [r for r in rows if r.id == params.get("resonance_id")]
        if len(matches) != 1:
            raise ValueError("Reference preset/resonance is incompatible with dataset nucleus")
        resonance = matches[0]
        if not resonance.recommended_for_direct_referencing:
            raise ValueError("Preset resonance is not recommended for direct referencing; review a custom per-member target")
        if params.get("target_ppm") is not None and float(params["target_ppm"]) != resonance.ppm:
            raise ValueError("Preset target differs from selected resonance")
        params.update(target_ppm=resonance.ppm, source_id=resonance.source_id,
            source_type=library.get(params["preset_id"]).category)
        if resonance.temperature_K is not None and dataset.conditions.temperature_k != resonance.temperature_K:
            warning = "Reference preset is condition-sensitive; review target temperature applicability"
        else:
            warning = None
    else:
        warning = None
        params.setdefault("source_type", "custom")
    return {**params, "observed_ppm": observed}, warning


def _pick(dataset, spectrum, parameters):
    detected = pick_positive_peaks_1d(spectrum, parameters=PeakPickingParameters(**parameters))
    removed = tuple(p.id for p in dataset.peaks if str(p.source or "").strip().lower() == "auto")
    dataset.peaks[:] = [p for p in dataset.peaks if p.id not in removed]
    spacing = np.abs(np.diff(spectrum.axis))
    spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
    tolerance = max(float(np.median(spacing)) * 1.5, 1e-6) if spacing.size else 1e-6
    for peak in detected:
        if not any(abs(p.position_ppm - peak.position_ppm) <= tolerance for p in dataset.peaks):
            dataset.peaks.append(peak)
    dataset.peaks.sort(key=lambda p: p.position_ppm)
    # Preserve manual/independent observables, remove replaced auto-linked records.
    dataset.observables[:] = [o for o in dataset.observables if o.source_peak_id not in removed]
    return removed, f"Detected {len(detected)}; replaced {len(removed)} auto peaks; total {len(dataset.peaks)}"


def _preview_phase(session, parameters):
    """Resolve each member's pivot and reuse its ordinary phase APIs."""
    params = dict(parameters)
    mode = params.pop("mode", "manual")
    pivot_mode = params.pop("pivot_mode", "explicit")
    algorithm = params.pop("algorithm", "peak_minima")
    if mode not in {"manual", "auto"} or pivot_mode not in {"explicit", "biggest_peak"}:
        raise ValueError("Phase mode must be manual/auto; pivot mode explicit/biggest_peak")
    if set(params) - {"p0_deg", "p1_deg", "pivot_ppm", "pivot_index"}:
        raise ValueError("Unknown phase parameters")
    spectrum = session.committed_spectrum
    if spectrum is None or spectrum.domain != "frequency":
        raise ValueError("Phase correction requires frequency-domain data")
    key = "pivot_ppm" if spectrum.axis_unit.lower() == "ppm" else "pivot_index"
    if pivot_mode == "biggest_peak":
        magnitude = np.abs(spectrum.intensities)
        finite = magnitude[np.isfinite(magnitude)]
        if not finite.size or not np.any(finite > 0):
            raise ValueError("Biggest-peak pivot requires a finite nonzero signal")
        pivot = session.biggest_peak_pivot()
    else:
        pivot = params.get(key)
    if mode == "auto":
        estimate = session.preview_auto_phase(algorithm=algorithm, **{key: pivot})
        p0, p1 = estimate.p0_deg, estimate.p1_deg
    else:
        p0, p1 = params.get("p0_deg", 0.), params.get("p1_deg", 0.)
        session.preview_phase(p0_deg=p0, p1_deg=p1, **{key: pivot})
    unit = "ppm" if key == "pivot_ppm" else "index"
    return (f"Phase {mode}" + (f"/{algorithm}" if mode == "auto" else "")
            + f": {pivot_mode} pivot={float(pivot):.6g} {unit}, P0={float(p0):.6g}, P1={float(p1):.6g} deg"
            + ("; " + "; ".join(estimate.diagnostics) if mode == "auto" and estimate.diagnostics else ""))


def run_batch_preview(preview):
    """Worker-safe: only isolated datasets change; each failure discards that member."""
    config = preview.recipe.configuration
    if preview.results or preview.applied:
        raise ValueError("Batch input already processed; capture fresh inputs")
    results = []
    for member_id, (original, seal, included, dataset, skip) in preview.inputs.items():
        if skip:
            results.append(BatchMemberResult(member_id, original.id, "skipped", "Excluded member"))
            continue
        warnings, removed, messages = [], [], []
        try:
            session = ProcessingSession()
            session.select_dataset(dataset)
            input_spectrum = session.committed_spectrum
            revision_before = dataset.spectral_revision
            if config.get("start_from_raw", False):
                session.start_reprocessing_from_raw()
            enabled = [row for row in config["operations"] if row.get("enabled", True)]
            if not enabled:
                results.append(BatchMemberResult(member_id, original.id, "skipped", "No enabled operations"))
                continue
            # Match single-spectrum activation on the isolated preview copy.
            # Explicit raw/FFT recipes retain their operation order.
            if (config.get("initialize_unprocessed", False)
                    and not config.get("start_from_raw", False)
                    and not any(row["operation"] == "fourier_transform" for row in enabled)
                    and session.can_process_time_domain
                    and not dataset.processing_history.steps
                    and dataset.processed_spectrum is None):
                session.initialize_default_spectrum()
                warnings.append("Untouched raw FID prepared with default vendor preprocessing/FFT")
            for row in enabled:
                operation = row["operation"]
                override = config.get("overrides", {}).get(member_id, {}).get(operation, {})
                params = {**row.get("parameters", {}), **override}
                if operation == "fourier_transform":
                    if params:
                        raise ValueError("Fourier transform has no configurable parameters")
                    session.apply_fourier_transform()
                elif operation == "peak_pick":
                    ids, message = _pick(dataset, session.committed_spectrum, params)
                    removed.extend(ids)
                    messages.append(message)
                else:
                    if operation == "reference":
                        params, warning = _reference_parameters(dataset, session, params, override)
                        if warning:
                            warnings.append(warning)
                        session.preview_reference(**params)
                        offset = float(params["target_ppm"]) - float(params["observed_ppm"])
                    elif operation == "apodization":
                        method = session.preview_apodization_from_spectrum if session.can_baseline else session.preview_apodization
                        method(**params)
                    elif operation == "zero_fill":
                        method = session.preview_zero_fill_from_spectrum if session.can_baseline else session.preview_zero_fill
                        method(**params)
                    elif operation == "phase":
                        messages.append(_preview_phase(session, params))
                    elif operation == "baseline":
                        session.preview_baseline_correction(**params)
                    session.apply_preview()
                    if operation == "reference":
                        for peak in dataset.peaks:
                            peak.position_ppm += offset
                        sync_dataset_observables(dataset, assignments_from_observables(dataset))
                        messages.append(f"Reference offset {offset:+g} ppm")
            preview.staged[member_id] = dataset
            results.append(BatchMemberResult(member_id, dataset.id, "ready", "; ".join(messages) or "Recipe preview ready",
                tuple(warnings), session.committed_spectrum, len(dataset.peaks), tuple(removed), revision_before, dataset.spectral_revision, input_spectrum))
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            results.append(BatchMemberResult(member_id, original.id, "failed", str(exc), tuple(warnings)))
    preview.results = tuple(results)
    return preview


def preview_batch(series, recipe, **kwargs):
    return run_batch_preview(prepare_batch(series, recipe, **kwargs))


def _publish_records(current, staged):
    by_id = {v.id: v for v in current}
    result = []
    for value in staged:
        existing = by_id.get(value.id)
        if existing is not None:
            existing.__dict__.update(deepcopy(value.__dict__))
            result.append(existing)
        else:
            result.append(deepcopy(value))
    current[:] = result


def apply_batch(preview, series, *, all_series=(), confirmed=False):
    if confirmed is not True:
        raise ValueError("Explicit batch confirmation required")
    if preview.applied or preview.series_id != series.id or not preview.results:
        raise ValueError("Batch preview is unavailable/already applied or belongs to another series")
    members = {m.id: m for m in series.members}
    results = []
    for result in preview.results:
        if result.status != "ready":
            results.append(result)
            continue
        original, seal, included, _, _ = preview.inputs[result.member_id]
        member = members.get(result.member_id)
        if member is None or member.dataset is not original or member.included != included or _seal(original) != seal:
            results.append(replace(result, status="failed", message="Source/member changed after preview; review again"))
            continue
        staged = preview.staged[result.member_id]
        original.processing_history = ProcessingHistory(staged.processing_history.steps)
        original.processed_spectrum = staged.processed_spectrum
        for _ in range(staged.spectral_revision - original.spectral_revision):
            original.mark_spectrum_changed()
        _publish_records(original.peaks, staged.peaks)
        _publish_records(original.observables, staged.observables)
        events = list(original.source_provenance.get("batch_processing", []))
        events.append({"schema_version": 1, "recipe": preview.recipe.to_dict(),
            "series_id": series.id, "member_id": result.member_id,
            "committed_at": datetime.now(timezone.utc).isoformat(),
            "revision_before": result.revision_before, "revision_after": result.revision_after,
            "peak_count": result.peak_count, "removed_peak_ids": list(result.removed_peak_ids),
            "summary": result.message})
        original.source_provenance["batch_processing"] = events
        # No scientific processing history for peak detection: use its established
        # ID replacement contract and explicit track invalidation instead.
        for item in {s.id: s for s in (*all_series, series)}.values():
            if result.removed_peak_ids:
                item.invalidate_tracks(dataset_id=original.id, peak_ids=set(result.removed_peak_ids), reason="auto_peaks_removed")
            item.update_tracks()
        results.append(replace(result, status="succeeded"))
    preview.applied = True
    return tuple(results)
