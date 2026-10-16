"""Reviewable NMRObservable transfer into PCS Analyzer NMR-shift state."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable


class PCSTransferError(ValueError):
    pass


@dataclass(frozen=True)
class PCSTarget:
    ref_id: int
    label: str

    @property
    def display(self) -> str:
        label = self.label.strip()
        return f"{self.ref_id} · {label}" if label else str(self.ref_id)


@dataclass(frozen=True)
class PCSTransferRow:
    observable_id: str
    assignment: str
    chemical_shift_ppm: float
    target_ref_id: int | None
    selected: bool = True


@dataclass(frozen=True)
class PCSTransferSummary:
    applied: int
    destination: str


def extract_pcs_targets(state: dict) -> list[PCSTarget]:
    atom_by_id = state.get("atom_by_id", {}) or {}
    overrides = state.get("ref_label_overrides", {}) or {}
    selected = state.get("current_selected_ids", []) or []
    ids = [int(x) for x in selected] if selected else sorted(
        {int(x) for x in atom_by_id} | {int(x) for x in overrides}
    )
    return [
        PCSTarget(
            ref_id=rid,
            label=str(overrides.get(rid, atom_by_id.get(rid, "")) or ""),
        )
        for rid in ids
    ]


def suggest_target_ref_id(assignment: str | None, targets: Iterable[PCSTarget]) -> int | None:
    text = (assignment or "").strip()
    if not text:
        return None
    targets = list(targets)
    if text.isdigit():
        rid = int(text)
        if any(t.ref_id == rid for t in targets):
            return rid
    matches = [t.ref_id for t in targets if t.label.strip().casefold() == text.casefold()]
    return matches[0] if len(matches) == 1 else None


def build_transfer_rows(observables, state: dict) -> list[PCSTransferRow]:
    targets = extract_pcs_targets(state)
    rows = []
    for observable in observables:
        assignment = (observable.assignment or "").strip()
        if not assignment:
            continue
        rows.append(
            PCSTransferRow(
                observable_id=observable.id,
                assignment=assignment,
                chemical_shift_ppm=float(observable.chemical_shift_ppm),
                target_ref_id=suggest_target_ref_id(assignment, targets),
            )
        )
    return rows


def apply_transfer_rows(
    state: dict,
    rows: Iterable[PCSTransferRow],
    *,
    destination: str,
) -> PCSTransferSummary:
    kind = destination.strip().lower()
    if kind not in {"obs", "dia"}:
        raise PCSTransferError("destination must be 'obs' or 'dia'")

    rows = [row for row in rows if row.selected]
    if not rows:
        raise PCSTransferError("No transfer rows are selected.")
    if any(row.target_ref_id is None for row in rows):
        raise PCSTransferError("Every selected observable must be mapped to a PCS Analyzer target.")

    target_ids = [int(row.target_ref_id) for row in rows]
    if len(target_ids) != len(set(target_ids)):
        raise PCSTransferError("Two selected observables are mapped to the same PCS Analyzer target.")

    target_key = "delta_obs_values" if kind == "obs" else "delta_dia_values"
    undo_key = "_delta_obs_undo" if kind == "obs" else "_delta_dia_undo"
    state[undo_key] = dict(state.get(target_key, {}) or {})
    target = state.setdefault(target_key, {})
    for row in rows:
        target[int(row.target_ref_id)] = float(row.chemical_shift_ppm)

    try:
        from logic.nmr_delta_data_manager import recompute_delta_para, push_layers_to_nmr_if_open
        recompute_delta_para(state)
        push_layers_to_nmr_if_open(state)
    except Exception:
        pass

    updater = state.get("update_graph")
    if callable(updater):
        try:
            updater()
        except Exception:
            pass

    from logic.pcs_document import mark_document_mutated
    mark_document_mutated(state, "pcs_data")

    return PCSTransferSummary(applied=len(rows), destination=kind)
