"""Reviewable handoff of experimental PCS values to PCS Analyzer.

D5d is deliberately a *data handoff* layer.  It does not calculate PCS from
coordinates, average calculated values, build residuals, or perform tensor
fitting.  Its responsibility is to preserve a validated experimental PCS value
together with the structural target plan selected in PCS Preparation.

Legacy single-atom targets continue to populate ``state['delta_exp_values']``
so the existing PCS Analyzer workflow remains available.  Grouped NMR sites,
matching ligand sites, and custom grouped targets are stored only in the
structured observation collection and are never flattened into repeated
``delta_exp_values`` entries.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .structure_aware_pcs_targets import PCSStructureTargetChoice


STATE_KEY = "experimental_pcs_observations"
STATE_SCHEMA_VERSION = 1
SOURCE_NAME = "nmr_workbench"


class PCSStructuredHandoffError(ValueError):
    """Raised when experimental PCS data cannot be handed off safely."""


@dataclass(frozen=True)
class PCSStructuredHandoffRow:
    """One validated experimental PCS value plus its structural target plan."""

    assignment: str
    pcs_ppm: float
    target: PCSStructureTargetChoice | None
    included: bool = True
    uncertainty_ppm: float | None = None
    temperature_K: float | None = None
    method: str | None = None
    source_dataset_ids: tuple[str, ...] = ()
    source_observable_ids: tuple[str, ...] = ()
    provenance: Mapping[str, Any] | None = None
    value_id: str | None = None

    @property
    def is_single_atom(self) -> bool:
        return bool(self.target and self.target.is_legacy_single_atom)

    @property
    def is_structured(self) -> bool:
        return bool(self.target and not self.target.is_legacy_single_atom)


@dataclass(frozen=True)
class PCSStructuredHandoffSummary:
    applied: int
    single_atom: int
    structured: int
    added: int
    replaced: int
    delta_exp_overwritten: int


def build_handoff_rows(
    prepared_values: Mapping[str, Any],
    target_choices: Mapping[str, PCSStructureTargetChoice | None],
    *,
    excluded_assignments: Iterable[str] = (),
) -> list[PCSStructuredHandoffRow]:
    """Combine derived PCS values with target choices selected in the UI."""

    excluded = {str(value) for value in excluded_assignments}
    rows: list[PCSStructuredHandoffRow] = []

    for assignment, value in prepared_values.items():
        assignment_text = str(assignment)
        provenance = dict(getattr(value, "provenance", {}) or {})
        rows.append(
            PCSStructuredHandoffRow(
                assignment=assignment_text,
                pcs_ppm=float(value.pcs_ppm),
                target=target_choices.get(assignment_text),
                included=assignment_text not in excluded,
                uncertainty_ppm=_optional_float(
                    getattr(value, "uncertainty_ppm", None)
                ),
                temperature_K=_optional_float(
                    getattr(value, "temperature_K", None)
                ),
                method=(
                    None
                    if getattr(value, "method", None) is None
                    else str(getattr(value, "method"))
                ),
                source_dataset_ids=tuple(
                    str(item)
                    for item in (getattr(value, "source_dataset_ids", ()) or ())
                ),
                source_observable_ids=tuple(
                    str(item)
                    for item in (getattr(value, "source_observable_ids", ()) or ())
                ),
                provenance=provenance,
                value_id=(
                    None
                    if getattr(value, "id", None) is None
                    else str(getattr(value, "id"))
                ),
            )
        )

    return rows


def validate_handoff_rows(
    rows: Iterable[PCSStructuredHandoffRow],
) -> list[PCSStructuredHandoffRow]:
    selected = [row for row in rows if row.included]
    if not selected:
        raise PCSStructuredHandoffError(
            "No experimental PCS values are selected."
        )

    unmapped = [row.assignment for row in selected if row.target is None]
    if unmapped:
        raise PCSStructuredHandoffError(
            "Every selected PCS value must have a structural target. "
            f"Unmapped: {', '.join(unmapped)}"
        )

    blank = [row.assignment for row in selected if not row.assignment.strip()]
    if blank:
        raise PCSStructuredHandoffError("PCS assignments must not be blank.")

    # Stable observable/dataset provenance distinguishes identical assignment
    # labels used by different experiment pairs. Assignment remains the final
    # compatibility fallback for older rows that have no stable IDs.
    identities = [_row_identity(row) for row in selected]
    duplicates = sorted(
        {
            row.assignment
            for row, identity in zip(selected, identities)
            if identities.count(identity) > 1
        }
    )
    if duplicates:
        raise PCSStructuredHandoffError(
            "Duplicate PCS observation identities are not allowed in one handoff: "
            + ", ".join(duplicates)
        )

    # Legacy delta_exp remains keyed by one Ref ID, therefore duplicate
    # single-atom destinations cannot be represented safely.
    single_refs = [
        int(row.target.primary_ref_id)
        for row in selected
        if row.target is not None
        and row.target.is_legacy_single_atom
        and row.target.primary_ref_id is not None
    ]
    duplicate_refs = sorted(
        {ref_id for ref_id in single_refs if single_refs.count(ref_id) > 1}
    )
    if duplicate_refs:
        raise PCSStructuredHandoffError(
            "Two selected PCS values cannot use the same single-atom target. "
            "Duplicate Ref IDs: " + ", ".join(map(str, duplicate_refs))
        )

    return selected


def preview_handoff_summary(
    state: dict,
    rows: Iterable[PCSStructuredHandoffRow],
) -> PCSStructuredHandoffSummary:
    if not isinstance(state, dict):
        raise PCSStructuredHandoffError("PCS Analyzer state is not available.")

    selected = validate_handoff_rows(rows)
    existing_records = tuple(state.get(STATE_KEY, []) or ())
    existing_delta = state.get("delta_exp_values", {}) or {}

    added = sum(
        1
        for row in selected
        if not any(_record_matches_row(record, row) for record in existing_records)
    )
    replaced = len(selected) - added
    single = [row for row in selected if row.is_single_atom]
    structured = [row for row in selected if row.is_structured]
    overwritten = sum(
        1
        for row in single
        if row.target is not None
        and row.target.primary_ref_id is not None
        and int(row.target.primary_ref_id) in existing_delta
    )

    return PCSStructuredHandoffSummary(
        applied=len(selected),
        single_atom=len(single),
        structured=len(structured),
        added=added,
        replaced=replaced,
        delta_exp_overwritten=overwritten,
    )


def apply_structured_pcs_handoff(
    state: dict,
    rows: Iterable[PCSStructuredHandoffRow],
) -> PCSStructuredHandoffSummary:
    """Store selected PCS observations and preserve legacy single-atom delta_exp.

    Structured targets are *not* interpreted here.  Their member groups and
    averaging metadata are preserved for the later PCS Analyzer integration.
    """

    if not isinstance(state, dict):
        raise PCSStructuredHandoffError("PCS Analyzer state is not available.")

    selected = validate_handoff_rows(rows)
    summary = preview_handoff_summary(state, selected)

    from logic.document_edit_history import PCS_KEYS, begin_edit, finish_edit
    edit_token = begin_edit(state, PCS_KEYS)
    state["_experimental_pcs_observations_undo"] = deepcopy(
        list(state.get(STATE_KEY, []) or [])
    )
    state["_delta_exp_undo"] = dict(state.get("delta_exp_values", {}) or {})
    state["_nmr_workbench_delta_exp_provenance_undo"] = deepcopy(
        dict(state.get("nmr_workbench_delta_exp_provenance", {}) or {})
    )

    selected_identities = {_row_identity(row) for row in selected}
    retained_records = [
        deepcopy(record)
        for record in (state.get(STATE_KEY, []) or [])
        if not any(_record_matches_row(record, row) for row in selected)
    ]

    delta_exp = state.setdefault("delta_exp_values", {})
    provenance_store = state.setdefault(
        "nmr_workbench_delta_exp_provenance",
        {},
    )

    # If an assignment previously used a legacy single-Ref destination and is
    # now handed off to another target (including a grouped target), remove only
    # the old value that NMR Workbench itself owns.  Never delete unrelated
    # manually-entered PCS Analyzer values.
    delta_exp_changed = False
    for ref_id, metadata in list(provenance_store.items()):
        metadata = metadata or {}
        metadata_identity = _identity_from_parts(
            assignment=str(metadata.get("assignment", "")),
            observable_ids=metadata.get("source_observable_ids", ()) or (),
            dataset_ids=metadata.get("source_dataset_ids", ()) or (),
        )
        # Old provenance did not retain stable IDs. Preserve its assignment
        # replacement behavior while new records use collision-safe identity.
        matches = metadata_identity in selected_identities
        if metadata_identity[0] == "assignment":
            matches = any(
                row.assignment == metadata_identity[1]
                for row in selected
            )
        if matches:
            try:
                normalized_ref = int(ref_id)
            except Exception:
                normalized_ref = ref_id
            if normalized_ref in delta_exp:
                delta_exp.pop(normalized_ref, None)
                delta_exp_changed = True
            else:
                # Preserve compatibility with state dictionaries that may
                # still carry string Ref keys.
                if ref_id in delta_exp:
                    delta_exp.pop(ref_id, None)
                    delta_exp_changed = True
            provenance_store.pop(ref_id, None)

    new_records: list[dict[str, Any]] = []
    for row in selected:
        assert row.target is not None  # validated above
        record = _row_to_record(row)
        new_records.append(record)

        if row.target.is_legacy_single_atom:
            ref_id = row.target.primary_ref_id
            if ref_id is None:
                raise PCSStructuredHandoffError(
                    f"{row.assignment}: single-atom target has no Ref ID."
                )
            ref_id = int(ref_id)
            delta_exp[ref_id] = float(row.pcs_ppm)
            provenance_store[ref_id] = {
                "assignment": row.assignment,
                "pcs_ppm": float(row.pcs_ppm),
                "uncertainty_ppm": row.uncertainty_ppm,
                "temperature_K": row.temperature_K,
                "method": row.method,
                "source_dataset_ids": list(row.source_dataset_ids),
                "source_observable_ids": list(row.source_observable_ids),
                "provenance": _json_safe(dict(row.provenance or {})),
                "structured_observation": True,
            }
            delta_exp_changed = True

    state[STATE_KEY] = retained_records + new_records

    updater = state.get("update_graph")
    if delta_exp_changed and callable(updater):
        updater()

    from logic.pcs_document import mark_document_mutated
    finish_edit(state, edit_token, "Import PCS observations · NMR Workbench")
    mark_document_mutated(state, "pcs_data")

    return summary


def _row_to_record(row: PCSStructuredHandoffRow) -> dict[str, Any]:
    target = row.target
    if target is None:
        raise PCSStructuredHandoffError(
            f"{row.assignment}: structural target is missing."
        )

    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "source": SOURCE_NAME,
        "id": row.value_id,
        "assignment": row.assignment,
        "pcs_ppm": float(row.pcs_ppm),
        "uncertainty_ppm": row.uncertainty_ppm,
        "temperature_K": row.temperature_K,
        "method": row.method,
        "source_dataset_ids": list(row.source_dataset_ids),
        "source_observable_ids": list(row.source_observable_ids),
        "target": {
            "scope": target.scope,
            "choice_key": target.key,
            "display": target.display,
            "site_member_ref_ids": [
                [int(ref_id) for ref_id in group]
                for group in target.site_member_ref_ids
            ],
            "site_ids": list(target.site_ids),
            "primary_ref_id": (
                None
                if target.primary_ref_id is None
                else int(target.primary_ref_id)
            ),
            # This is descriptive metadata from NMR-site planning.  D5d does
            # not execute the averaging operation.
            "averaging_hint": target.averaging,
            "matching_status": target.matching_status,
            "source": target.source,
        },
        "provenance": _json_safe(dict(row.provenance or {})),
    }


def _identity_from_parts(*, assignment, observable_ids, dataset_ids) -> tuple:
    observables = tuple(str(value) for value in observable_ids if value)
    if observables:
        return ("observables", *observables)
    datasets = tuple(str(value) for value in dataset_ids if value)
    if datasets:
        return ("datasets", *datasets, "assignment", str(assignment).casefold())
    return ("assignment", str(assignment))


def _row_identity(row: PCSStructuredHandoffRow) -> tuple:
    return _identity_from_parts(
        assignment=row.assignment,
        observable_ids=row.source_observable_ids,
        dataset_ids=row.source_dataset_ids,
    )


def _record_identity(record: Mapping[str, Any]) -> tuple:
    return _identity_from_parts(
        assignment=record.get("assignment", ""),
        observable_ids=record.get("source_observable_ids", ()) or (),
        dataset_ids=record.get("source_dataset_ids", ()) or (),
    )


def _record_matches_row(
    record: Mapping[str, Any],
    row: PCSStructuredHandoffRow,
) -> bool:
    identity = _record_identity(record)
    if identity == _row_identity(row):
        return True
    return (
        identity[0] == "assignment"
        and str(identity[1]) == row.assignment
    )


def record_matches_handoff_row(record, row):
    """Public identity test for review adapters, preserving legacy fallbacks."""
    return _record_matches_row(record, row)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        return float(value)
    except Exception:
        return str(value)
