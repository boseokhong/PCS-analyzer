"""Validated transfer of derived experimental PCS values into PCS Analyzer δ_exp."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Any


class PCSExperimentalTransferError(ValueError):
    """Raised when fitting-ready PCS values cannot be applied safely."""


@dataclass(frozen=True)
class PCSTarget:
    ref_id: int
    label: str

    @property
    def display(self) -> str:
        text = self.label.strip()
        return f"{self.ref_id} · {text}" if text else str(self.ref_id)


@dataclass(frozen=True)
class PCSExperimentalApplyRow:
    assignment: str
    pcs_ppm: float
    target_ref_id: int | None
    included: bool = True
    uncertainty_ppm: float | None = None
    method: str | None = None
    provenance: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PCSExperimentalApplySummary:
    applied: int
    added: int
    overwritten: int


def extract_pcs_targets(state: dict | None) -> list[PCSTarget]:
    if not isinstance(state, dict):
        return []

    atom_by_id = state.get("atom_by_id", {}) or {}
    overrides = state.get("ref_label_overrides", {}) or {}
    selected = state.get("current_selected_ids", []) or []

    if selected:
        ids = [int(value) for value in selected]
    else:
        ids = sorted(
            {int(value) for value in atom_by_id}
            | {int(value) for value in overrides}
        )

    return [
        PCSTarget(
            ref_id=ref_id,
            label=str(overrides.get(ref_id, atom_by_id.get(ref_id, "")) or ""),
        )
        for ref_id in ids
    ]


def suggest_target_ref_id(
    assignment: str | None,
    targets: Iterable[PCSTarget],
) -> int | None:
    text = (assignment or "").strip()
    if not text:
        return None

    target_list = list(targets)

    # Direct Ref-ID entry remains useful for advanced/manual assignment.
    if text.isdigit():
        ref_id = int(text)
        if any(target.ref_id == ref_id for target in target_list):
            return ref_id

    folded = text.casefold()
    matches = [
        target.ref_id
        for target in target_list
        if target.label.strip().casefold() == folded
    ]
    return matches[0] if len(matches) == 1 else None


def build_apply_rows(
    prepared_values: Mapping[str, Any],
    state: dict | None,
    *,
    target_overrides: Mapping[str, int] | None = None,
    excluded_assignments: Iterable[str] = (),
) -> list[PCSExperimentalApplyRow]:
    targets = extract_pcs_targets(state)
    overrides = dict(target_overrides or {})
    excluded = {str(value) for value in excluded_assignments}

    rows: list[PCSExperimentalApplyRow] = []
    for assignment, value in prepared_values.items():
        target_ref_id = overrides.get(assignment)
        if target_ref_id is None:
            target_ref_id = getattr(value, "target_ref_id", None)
        if target_ref_id is None:
            target_ref_id = suggest_target_ref_id(assignment, targets)

        provenance = dict(getattr(value, "provenance", {}) or {})
        rows.append(
            PCSExperimentalApplyRow(
                assignment=str(assignment),
                pcs_ppm=float(value.pcs_ppm),
                target_ref_id=(
                    None if target_ref_id is None else int(target_ref_id)
                ),
                included=assignment not in excluded,
                uncertainty_ppm=getattr(value, "uncertainty_ppm", None),
                method=getattr(value, "method", None),
                provenance=provenance,
            )
        )
    return rows


def validate_apply_rows(
    rows: Iterable[PCSExperimentalApplyRow],
) -> list[PCSExperimentalApplyRow]:
    selected = [row for row in rows if row.included]
    if not selected:
        raise PCSExperimentalTransferError(
            "No experimental PCS values are selected."
        )

    unmapped = [row.assignment for row in selected if row.target_ref_id is None]
    if unmapped:
        raise PCSExperimentalTransferError(
            "Every selected PCS value must be mapped to a PCS Analyzer target. "
            f"Unmapped: {', '.join(unmapped)}"
        )

    target_ids = [int(row.target_ref_id) for row in selected]
    if len(target_ids) != len(set(target_ids)):
        duplicates = sorted(
            {
                ref_id
                for ref_id in target_ids
                if target_ids.count(ref_id) > 1
            }
        )
        raise PCSExperimentalTransferError(
            "Two selected PCS values cannot use the same PCS Analyzer target. "
            f"Duplicate Ref IDs: {', '.join(map(str, duplicates))}"
        )

    return selected


def preview_apply_summary(
    state: dict,
    rows: Iterable[PCSExperimentalApplyRow],
) -> PCSExperimentalApplySummary:
    selected = validate_apply_rows(rows)
    existing = state.get("delta_exp_values", {}) or {}
    overwritten = sum(
        1 for row in selected if int(row.target_ref_id) in existing
    )
    return PCSExperimentalApplySummary(
        applied=len(selected),
        added=len(selected) - overwritten,
        overwritten=overwritten,
    )


def apply_experimental_pcs(
    state: dict,
    rows: Iterable[PCSExperimentalApplyRow],
) -> PCSExperimentalApplySummary:
    if not isinstance(state, dict):
        raise PCSExperimentalTransferError(
            "PCS Analyzer state is not available."
        )

    selected = validate_apply_rows(rows)
    summary = preview_apply_summary(state, selected)

    from logic.document_edit_history import PCS_KEYS, begin_edit, finish_edit
    edit_token = begin_edit(state, PCS_KEYS)
    current = dict(state.get("delta_exp_values", {}) or {})
    state["_delta_exp_undo"] = current

    target = state.setdefault("delta_exp_values", {})
    provenance_store = state.setdefault(
        "nmr_workbench_delta_exp_provenance",
        {},
    )

    for row in selected:
        ref_id = int(row.target_ref_id)
        target[ref_id] = float(row.pcs_ppm)
        provenance_store[ref_id] = {
            "assignment": row.assignment,
            "pcs_ppm": float(row.pcs_ppm),
            "uncertainty_ppm": (
                None
                if row.uncertainty_ppm is None
                else float(row.uncertainty_ppm)
            ),
            "method": row.method,
            "provenance": dict(row.provenance or {}),
        }

    updater = state.get("update_graph")
    if callable(updater):
        updater()

    from logic.pcs_document import mark_document_mutated
    finish_edit(state, edit_token, "Import PCS · NMR Workbench")
    mark_document_mutated(state, "pcs_data")

    return summary
