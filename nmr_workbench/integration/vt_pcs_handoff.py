"""Reviewed VT transfer through the sole canonical PCS handoff boundary."""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

from logic.pcs_document import active_document, document_display_name
from ..core.vt_pcs import VTPCSHandoffReceipt, VTPCSPreparation
from ..analysis.vt_pcs import all_observations, pcs_ready_row, preparation_diagnostics
from .pcs_structured_handoff import (STATE_KEY, apply_structured_pcs_handoff,
    preview_handoff_summary, record_matches_handoff_row)


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def destination_context(state):
    if not isinstance(state, dict):
        raise ValueError("PCS destination is unavailable")
    document = active_document(state)
    if document is None and state.get("document_manager") is not None:
        raise ValueError("Activate and explicitly bind a PCS document first")
    return (document.document_id if document else None,
        document_display_name(document) if document else "Legacy PCS context",
        document.revision if document else None)


def _destination_digest(state, choices):
    context = destination_context(state)
    # Runtime-only seal. Mixed integer/string legacy Ref keys remain distinct.
    content = [id(state), context, state.get(STATE_KEY), state.get("delta_exp_values"),
        state.get("nmr_workbench_delta_exp_provenance"), [asdict(c) for c in choices]]
    return _digest(json.dumps(content, default=str, allow_nan=True))


@dataclass(frozen=True)
class VTTransferConflict:
    code: str
    preparation_id: str
    message: str
    replaceable: bool = True


@dataclass(frozen=True)
class VTTransferPreview:
    id: str
    series_id: str
    preparation_ids: tuple[str, ...]
    evidence_digests: tuple[str, ...]
    destination_document_id: str | None
    destination_name: str
    destination_digest: str
    rows: tuple
    rows_digest: str
    summary: object
    conflicts: tuple[VTTransferConflict, ...]


def preview_vt_transfer(series, datasets, all_series, preparation_ids, destination, target_choices):
    """Pure selection/PCS/source/destination validation; no implicit exclusions."""
    if series is None:
        raise ValueError("Select a VT series first")
    ids = tuple(preparation_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Select unique prepared observations to transfer")
    preparations = {p.id: p for p in series.pcs_preparations}
    if any(i not in preparations for i in ids):
        raise ValueError("Selected preparation is missing or belongs to another series")
    choices = tuple(target_choices)
    observations = all_observations(datasets, all_series)
    rows = tuple(pcs_ready_row(preparations[i], observations, target_choices=choices) for i in ids)
    # Canonical validation rejects duplicate identities/single-Ref destinations.
    summary = preview_handoff_summary(destination, rows)
    context = destination_context(destination)
    conflicts = []
    existing = destination.get(STATE_KEY, ()) or ()
    delta = destination.get("delta_exp_values", {}) or {}
    for row in rows:
        matched = [r for r in existing if record_matches_handoff_row(r, row)]
        for record in matched:
            conflicts.append(VTTransferConflict("replace_observation", row.value_id,
                f"Replace PCS observation {record.get('id') or record.get('assignment')} ({record.get('pcs_ppm')} ppm)"))
        if row.is_single_atom:
            ref_id = row.target.primary_ref_id
            for record in existing:
                groups = tuple(tuple(g) for g in record.get("target", {}).get("site_member_ref_ids", []))
                if groups == ((ref_id,),) and record.get("target", {}).get("scope") == row.target.scope and not record_matches_handoff_row(record, row):
                    conflicts.append(VTTransferConflict("distinct_single_atom_observation", row.value_id,
                        f"Ref {ref_id} already represents a different observation; choose another document or resolve it in PCS Analyzer", False))
            for key in (ref_id, str(ref_id)):
                if key in delta:
                    conflicts.append(VTTransferConflict("overwrite_legacy_ref", row.value_id,
                        f"Overwrite existing PCS at Ref {key}: {delta[key]} ppm"))
                    if isinstance(key, str):
                        conflicts.append(VTTransferConflict("legacy_ref_key_conflict", row.value_id,
                            f"Resolve the legacy string Ref key {key!r} in PCS Analyzer before transfer", False))
        for ref_id, metadata in (destination.get("nmr_workbench_delta_exp_provenance", {}) or {}).items():
            if metadata and record_matches_handoff_row(metadata, row):
                conflicts.append(VTTransferConflict("replace_prior_destination", row.value_id,
                    f"Replace prior NMR-owned destination/provenance at Ref {ref_id}; current PCS {delta.get(ref_id, delta.get(str(ref_id), 'absent'))}"))
    return VTTransferPreview(str(uuid4()), series.id, ids,
        tuple(_digest(preparations[i].evidence_json) for i in ids), context[0], context[1],
        _destination_digest(destination, choices), rows, _digest(_json([asdict(r) for r in rows])), summary, tuple(conflicts))


def apply_vt_transfer(preview, series, datasets, all_series, destination, target_choices, *, confirmed=False, replace_existing=False):
    """Apply exactly the preview after a fresh source/document/target/context check."""
    if confirmed is not True:
        raise ValueError("Explicit reviewed transfer confirmation is required")
    current = preview_vt_transfer(series, datasets, all_series, preview.preparation_ids, destination, target_choices)
    if (preview.series_id != series.id or current.evidence_digests != preview.evidence_digests
        or current.destination_digest != preview.destination_digest or current.rows_digest != preview.rows_digest
        or _digest(_json([asdict(r) for r in preview.rows])) != preview.rows_digest):
        raise ValueError("Transfer review changed; preview sources and destination again")
    if any(not c.replaceable for c in current.conflicts):
        raise ValueError("; ".join(c.message for c in current.conflicts if not c.replaceable))
    if current.conflicts and replace_existing is not True:
        raise ValueError("Destination conflicts require explicit reviewed replacement approval")
    timestamp = datetime.now(timezone.utc).isoformat()
    receipt = VTPCSHandoffReceipt(series.id, preview.preparation_ids, preview.evidence_digests,
        preview.destination_document_id, preview.destination_name, timestamp,
        bool(current.conflicts), preview.id)
    if any(r.id == receipt.id for r in series.pcs_handoff_receipts):
        raise ValueError("This reviewed transfer was already applied")
    rows = tuple(replace(r, provenance={**dict(r.provenance or {}), "vt_handoff": {
        "receipt_id": receipt.id, "destination_document_id": receipt.destination_document_id,
        "destination_name": receipt.destination_name, "transferred_at": timestamp,
        "evidence_digest": digest}}) for r, digest in zip(current.rows, current.evidence_digests))
    # Canonical mutation remains authoritative. Restore its scientific keys if a
    # synchronous host callback fails, rather than publish a half-reported transfer.
    keys = (STATE_KEY, "delta_exp_values", "nmr_workbench_delta_exp_provenance",
        "_experimental_pcs_observations_undo", "_delta_exp_undo", "_nmr_workbench_delta_exp_provenance_undo",
        "last_document_mutation_domain")
    previous = {k: deepcopy(destination[k]) for k in keys if k in destination}
    document = active_document(destination)
    document_before = (document.snapshot(), document.revision, document.dirty) if document else None
    history = document.edit_history if document else None
    history_before = (list(history.edits), history.cursor) if history is not None else None
    try:
        result = apply_structured_pcs_handoff(destination, rows)
    except Exception:
        for key in keys:
            if key in previous:
                destination[key] = previous[key]
            else:
                destination.pop(key, None)
        if document_before:
            document.replace_data(document_before[0])
            document.revision, document.dirty = document_before[1:]
            if history_before is None:
                document.edit_history = None
            else:
                history.edits[:], history.cursor = history_before
                document.edit_history = history
        raise
    series.add_pcs_handoff_receipt(receipt)
    return result, receipt


def transferred_source_diagnostics(pcs_record, observations):
    """Read-only diagnosis; never revise a previously transferred PCS value."""
    provenance = pcs_record.get("provenance", {}) or {}
    evidence = provenance.get("vt_pairing")
    if not evidence or not provenance.get("vt_preparation_id"):
        return ("no_vt_provenance",)
    para = evidence["paramagnetic"]
    preparation = VTPCSPreparation(para["series_id"], para["track_id"], para["member_id"],
        _json(evidence), provenance["vt_preparation_id"])
    return preparation_diagnostics(preparation, observations)


def receipt_diagnostics(series):
    preparations = {p.id: p for p in series.pcs_preparations}
    for receipt in series.pcs_handoff_receipts:
        for prep_id, digest in zip(receipt.preparation_ids, receipt.evidence_digests):
            current = preparations.get(prep_id)
            if current is None:
                yield "handoff_preparation_missing", f"receipt={receipt.id}, preparation={prep_id}"
            elif _digest(current.evidence_json) != digest:
                yield "handoff_preparation_changed", f"receipt={receipt.id}, preparation={prep_id}"
