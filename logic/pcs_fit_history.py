"""Immutable history records and comparison helpers for standard PCS fits."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from math import isfinite
from typing import Any, Mapping, MutableMapping, Sequence
from uuid import uuid4

from logic.pcs_fit_result_schema import deserialize_fit_result, serialize_fit_result
from logic.pcs_fit_result_status import evaluate_fit_result_status


PCS_FIT_HISTORY_SCHEMA = "pcs_fit_history_v1"


def _history_state(state):
    """Use explicit document storage when an active document is installed."""
    from logic.pcs_document import active_document, active_document_view

    if active_document(state) is None:
        return state
    return active_document_view(state)


def _mark_history_mutated(state: MutableMapping[str, Any]) -> None:
    from logic.pcs_document import mark_document_mutated
    mark_document_mutated(state, "standard_fitting")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("fit history timestamp is missing")
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("fit history timestamp is invalid") from exc
    return text


@dataclass(frozen=True)
class FitHistoryEntry:
    """One immutable wrapper around a canonical serialized fit-result payload."""

    result_id: str
    created_at: str
    sequence: int
    result_json: str
    label: str = ""

    def __post_init__(self) -> None:
        if not str(self.result_id).strip():
            raise ValueError("fit history result ID is missing")
        if int(self.sequence) < 1:
            raise ValueError("fit history sequence must be positive")
        _validate_timestamp(self.created_at)
        # Validate once at the persistence/creation boundary. Consumers always
        # receive a fresh dict, so legacy mutation cannot alter this record.
        self.result()

    def result(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.result_json)
        except json.JSONDecodeError as exc:
            raise ValueError("fit history result payload is malformed") from exc
        return deserialize_fit_result(payload)

    def to_payload(self) -> dict[str, Any]:
        return {
            "history_schema": PCS_FIT_HISTORY_SCHEMA,
            "result_id": self.result_id,
            "created_at": self.created_at,
            "sequence": self.sequence,
            "label": self.label,
            "result": self.result(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FitHistoryEntry":
        if not isinstance(payload, Mapping):
            raise ValueError("fit history entry must be an object")
        if payload.get("history_schema") != PCS_FIT_HISTORY_SCHEMA:
            raise ValueError("unsupported fit history entry schema")
        result = serialize_fit_result(payload.get("result"))
        try:
            sequence = int(payload.get("sequence"))
        except (TypeError, ValueError) as exc:
            raise ValueError("fit history sequence is invalid") from exc
        return cls(
            result_id=str(payload.get("result_id", "")).strip(),
            created_at=_validate_timestamp(payload.get("created_at")),
            sequence=sequence,
            label=str(payload.get("label", "") or ""),
            result_json=json.dumps(
                result,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )


def create_fit_history_entry(
    result: Mapping[str, Any],
    *,
    sequence: int,
    result_id: str | None = None,
    created_at: str | None = None,
    label: str = "",
) -> FitHistoryEntry:
    normalized = serialize_fit_result(result)
    return FitHistoryEntry(
        result_id=result_id or str(uuid4()),
        created_at=created_at or _utc_timestamp(),
        sequence=sequence,
        label=str(label or ""),
        result_json=json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def fit_history_entries(state: Mapping[str, Any]) -> list[FitHistoryEntry]:
    state = _history_state(state)
    entries = state.get("fit_history", ()) or ()
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return []
    return [entry for entry in entries if isinstance(entry, FitHistoryEntry)]


def find_fit_history_entry(
    state: Mapping[str, Any], result_id: str | None
) -> FitHistoryEntry | None:
    if not result_id:
        return None
    return next(
        (entry for entry in fit_history_entries(state) if entry.result_id == result_id),
        None,
    )


def _latest_entry(entries: Sequence[FitHistoryEntry]) -> FitHistoryEntry | None:
    return max(entries, key=lambda entry: entry.sequence, default=None)


def sync_last_fit_result_bridge(state: MutableMapping[str, Any]) -> dict | None:
    """Point the legacy bridge at a fresh copy of the active/latest payload."""
    state = _history_state(state)
    entries = fit_history_entries(state)
    active = find_fit_history_entry(state, state.get("active_fit_result_id"))
    latest = _latest_entry(entries)
    selected = active or latest
    if latest is not None:
        state["latest_fit_result_id"] = latest.result_id
    else:
        state.pop("latest_fit_result_id", None)
    if selected is None:
        state.pop("active_fit_result_id", None)
        state.pop("last_fit_result", None)
        return None
    state["active_fit_result_id"] = selected.result_id
    result = selected.result()
    state["last_fit_result"] = result
    return result


def append_fit_history_result(
    state: MutableMapping[str, Any], result: Mapping[str, Any], *, label: str = ""
) -> FitHistoryEntry:
    mutation_owner = state
    state = _history_state(state)
    entries = fit_history_entries(state)
    sequence = max((entry.sequence for entry in entries), default=0) + 1
    entry = create_fit_history_entry(result, sequence=sequence, label=label)
    state["fit_history"] = [*entries, entry]
    state["latest_fit_result_id"] = entry.result_id
    state["active_fit_result_id"] = entry.result_id
    sync_last_fit_result_bridge(state)
    _mark_history_mutated(mutation_owner)
    return entry


def append_fit_history_entry(
    state: MutableMapping[str, Any], entry: FitHistoryEntry
) -> FitHistoryEntry:
    """Append one already validated entry with an assigned local sequence."""
    if not isinstance(entry, FitHistoryEntry):
        raise TypeError("entry must be a FitHistoryEntry")
    mutation_owner = state
    state = _history_state(state)
    entries = fit_history_entries(state)
    if any(item.result_id == entry.result_id for item in entries):
        raise ValueError(f"fit history result ID already exists: {entry.result_id}")
    if any(item.sequence == entry.sequence for item in entries):
        raise ValueError(f"fit history run sequence already exists: {entry.sequence}")
    expected_sequence = max((item.sequence for item in entries), default=0) + 1
    if entry.sequence != expected_sequence:
        raise ValueError(
            f"fit history run sequence must be the next local value {expected_sequence}"
        )
    state["fit_history"] = [*entries, entry]
    state["latest_fit_result_id"] = entry.result_id
    state["active_fit_result_id"] = entry.result_id
    sync_last_fit_result_bridge(state)
    _mark_history_mutated(mutation_owner)
    return entry


def select_fit_history_result(
    state: MutableMapping[str, Any], result_id: str
) -> FitHistoryEntry:
    mutation_owner = state
    state = _history_state(state)
    entry = find_fit_history_entry(state, result_id)
    if entry is None:
        raise KeyError(f"unknown fit history result ID: {result_id}")
    state["active_fit_result_id"] = entry.result_id
    sync_last_fit_result_bridge(state)
    _mark_history_mutated(mutation_owner)
    return entry


def label_fit_history_entry(
    state: MutableMapping[str, Any], result_id: str, label: str
) -> FitHistoryEntry:
    mutation_owner = state
    state = _history_state(state)
    entries = fit_history_entries(state)
    entry = find_fit_history_entry(state, result_id)
    if entry is None:
        raise KeyError(f"unknown fit history result ID: {result_id}")
    updated = replace(entry, label=str(label or ""))
    state["fit_history"] = [updated if item.result_id == result_id else item for item in entries]
    _mark_history_mutated(mutation_owner)
    return updated


def delete_fit_history_entry(
    state: MutableMapping[str, Any], result_id: str
) -> bool:
    mutation_owner = state
    state = _history_state(state)
    entries = fit_history_entries(state)
    kept = [entry for entry in entries if entry.result_id != result_id]
    if len(kept) == len(entries):
        return False
    state["fit_history"] = kept
    if state.get("active_fit_result_id") == result_id:
        state.pop("active_fit_result_id", None)
    if state.get("latest_fit_result_id") == result_id:
        state.pop("latest_fit_result_id", None)
    sync_last_fit_result_bridge(state)
    _mark_history_mutated(mutation_owner)
    return True


def _clear_fit_history_state(state: MutableMapping[str, Any]) -> None:
    state["fit_history"] = []
    state.pop("active_fit_result_id", None)
    state.pop("latest_fit_result_id", None)
    state.pop("last_fit_result", None)


def clear_fit_history(state: MutableMapping[str, Any]) -> None:
    mutation_owner = state
    state = _history_state(state)
    _clear_fit_history_state(state)
    _mark_history_mutated(mutation_owner)


def serialize_fit_history(state: Mapping[str, Any]) -> dict[str, Any]:
    state = _history_state(state)
    entries = fit_history_entries(state)
    return {
        "format": PCS_FIT_HISTORY_SCHEMA,
        "active_result_id": state.get("active_fit_result_id"),
        "latest_result_id": state.get("latest_fit_result_id"),
        "entries": [entry.to_payload() for entry in entries],
    }


def install_fit_history(
    state: MutableMapping[str, Any], payload: Mapping[str, Any]
) -> list[FitHistoryEntry]:
    mutation_owner = state
    state = _history_state(state)
    if not isinstance(payload, Mapping) or payload.get("format") != PCS_FIT_HISTORY_SCHEMA:
        raise ValueError("unsupported fit history format")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("fit history entries are missing or invalid")
    entries = [FitHistoryEntry.from_payload(item) for item in raw_entries]
    ids = [entry.result_id for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("fit history contains duplicate result IDs")
    sequences = [entry.sequence for entry in entries]
    if len(sequences) != len(set(sequences)):
        raise ValueError("fit history contains duplicate run sequences")
    active = payload.get("active_result_id")
    latest = payload.get("latest_result_id")
    if active is not None and active not in ids:
        raise ValueError("fit history active result ID is invalid")
    if latest is not None and latest not in ids:
        raise ValueError("fit history latest result ID is invalid")
    actual_latest = _latest_entry(entries)
    if actual_latest is not None and latest not in (None, actual_latest.result_id):
        raise ValueError("fit history latest result ID is inconsistent")

    _clear_fit_history_state(state)
    state["fit_history"] = entries
    if actual_latest is not None:
        state["latest_fit_result_id"] = actual_latest.result_id
        state["active_fit_result_id"] = active or actual_latest.result_id
    sync_last_fit_result_bridge(state)
    _mark_history_mutated(mutation_owner)
    return entries


def _metadata(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = result.get("fit_metadata")
    return value if isinstance(value, Mapping) else {}


def _orientation(result: Mapping[str, Any]) -> Any:
    mode = result.get("mode")
    if mode == "theta_alpha_multi":
        return [result.get("theta"), result.get("alpha")]
    if mode == "euler_global":
        return [result.get("ax"), result.get("ay"), result.get("az")]
    if mode == "full_tensor":
        return result.get("euler_deg")
    return None


def compare_fit_results(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a neutral field comparison with explicit comparability warnings."""
    left_meta = _metadata(left)
    right_meta = _metadata(right)
    fields = {
        "mode": (left.get("mode"), right.get("mode")),
        "weighting_mode": (
            left_meta.get("weighting_mode", "unweighted"),
            right_meta.get("weighting_mode", "unweighted"),
        ),
        "optimizer": (left_meta.get("optimizer"), right_meta.get("optimizer")),
        "n": (left.get("n"), right.get("n")),
        "rmsd": (left.get("rmsd"), right.get("rmsd")),
        "r2": (left.get("r2"), right.get("r2")),
        "q_factor": (left.get("q_factor"), right.get("q_factor")),
        "raw_sse_per_datum": (left.get("chi2_n"), right.get("chi2_n")),
        "chi_square": (left.get("chi_square"), right.get("chi_square")),
        "delta_chi_ax": (left.get("delta_chi_ax"), right.get("delta_chi_ax")),
        "delta_chi_rh": (left.get("delta_chi_rh"), right.get("delta_chi_rh")),
        "orientation": (_orientation(left), _orientation(right)),
        "metal_position": (left.get("metal_pos"), right.get("metal_pos")),
    }
    if state is not None:
        fields["status"] = (
            evaluate_fit_result_status(left, state)["status"],
            evaluate_fit_result_status(right, state)["status"],
        )

    warnings: list[str] = []
    if fields["mode"][0] != fields["mode"][1]:
        warnings.append("These runs used different fit modes.")
    if fields["weighting_mode"][0] != fields["weighting_mode"][1]:
        warnings.append("These runs used different weighting modes.")
    if fields["n"][0] != fields["n"][1]:
        warnings.append("These runs used different numbers of observations.")
    left_ids = tuple(
        row.get("observable_id") for row in left.get("per_observable", ())
        if isinstance(row, Mapping)
    )
    right_ids = tuple(
        row.get("observable_id") for row in right.get("per_observable", ())
        if isinstance(row, Mapping)
    )
    left_fp = left_meta.get("input_fingerprint")
    right_fp = right_meta.get("input_fingerprint")
    left_digest = left_fp.get("digest") if isinstance(left_fp, Mapping) else None
    right_digest = right_fp.get("digest") if isinstance(right_fp, Mapping) else None
    if left_ids != right_ids or (
        left_digest is not None and right_digest is not None and left_digest != right_digest
    ):
        warnings.append("These runs used different fitting inputs.")

    differences = {}
    for name, (left_value, right_value) in fields.items():
        try:
            left_number = float(left_value)
            right_number = float(right_value)
        except (TypeError, ValueError):
            differences[name] = None
        else:
            differences[name] = (
                right_number - left_number
                if isfinite(left_number) and isfinite(right_number)
                else None
            )
    return {"fields": fields, "differences": differences, "warnings": warnings}


__all__ = [
    "PCS_FIT_HISTORY_SCHEMA",
    "FitHistoryEntry",
    "append_fit_history_entry",
    "append_fit_history_result",
    "clear_fit_history",
    "compare_fit_results",
    "create_fit_history_entry",
    "delete_fit_history_entry",
    "find_fit_history_entry",
    "fit_history_entries",
    "install_fit_history",
    "label_fit_history_entry",
    "select_fit_history_result",
    "serialize_fit_history",
    "sync_last_fit_result_bridge",
]
