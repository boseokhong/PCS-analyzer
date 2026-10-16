"""Validated read-only import of external PCS fit-run export records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping
from uuid import UUID

from logic.pcs_fit_fingerprint import validate_fit_input_fingerprint
from logic.pcs_fit_history import (
    FitHistoryEntry,
    append_fit_history_entry,
    create_fit_history_entry,
    find_fit_history_entry,
    fit_history_entries,
)
from logic.pcs_fit_history_export import (
    PCS_FIT_HISTORY_EXPORT_SCHEMA,
    PCS_FIT_RUN_EXPORT_SCHEMA,
)
from logic.pcs_fit_result_schema import serialize_fit_result


MAX_IMPORT_FILE_BYTES = 64 * 1024 * 1024
MAX_IMPORT_RUNS = 10_000
MAX_OBSERVATIONS_PER_RUN = 100_000
MAX_JSON_DEPTH = 64
MAX_JSON_STRING_LENGTH = 1_000_000
MAX_JSON_NODES = 2_000_000


class FitRunImportError(ValueError):
    """Raised when an export wrapper/file cannot be safely interpreted."""


@dataclass(frozen=True)
class ValidatedFitRun:
    result_id: str
    created_at_utc: str
    source_run_sequence: int
    label: str
    result: dict[str, Any]


@dataclass(frozen=True)
class FitRunImportRejection:
    index: int
    result_id: str | None
    reason: str


@dataclass(frozen=True)
class FitRunImportReport:
    imported: tuple[FitHistoryEntry, ...]
    skipped_duplicate_ids: tuple[str, ...]
    rejected: tuple[FitRunImportRejection, ...]

    @property
    def imported_count(self) -> int:
        return len(self.imported)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_duplicate_ids)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    def summary(self) -> str:
        parts = [f"Imported {self.imported_count} run(s)."]
        if self.skipped_count:
            parts.append(f"{self.skipped_count} duplicate(s) skipped.")
        if self.rejected_count:
            parts.append(f"{self.rejected_count} run(s) rejected.")
        return " ".join(parts)


def _validate_json_resources(payload: Any) -> None:
    stack = [(payload, 0)]
    nodes = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise FitRunImportError("fit-run export contains too many JSON values")
        if depth > MAX_JSON_DEPTH:
            raise FitRunImportError("fit-run export exceeds the JSON nesting limit")
        if isinstance(value, str):
            if len(value) > MAX_JSON_STRING_LENGTH:
                raise FitRunImportError("fit-run export contains an oversized string")
        elif isinstance(value, Mapping):
            stack.extend((key, depth + 1) for key in value.keys())
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)


def load_fit_run_export(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise FitRunImportError(f"could not read fit-run export: {exc}") from exc
    if size > MAX_IMPORT_FILE_BYTES:
        raise FitRunImportError("fit-run export exceeds the 64 MiB file-size limit")
    try:
        with source.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise FitRunImportError(f"fit-run export is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FitRunImportError("fit-run export root must be a JSON object")
    _validate_json_resources(payload)
    return payload


def _utc_timestamp(value: Any, name: str) -> str:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FitRunImportError(f"{name} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise FitRunImportError(f"{name} must be a UTC timestamp")
    return text


def _uuid(value: Any) -> str:
    text = str(value or "").strip()
    try:
        parsed = UUID(text)
    except (ValueError, AttributeError) as exc:
        raise FitRunImportError("history_entry.result_id must be a valid UUID") from exc
    return str(parsed)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise FitRunImportError(f"{name} must be a positive integer")
    return value


def validate_fit_run_export(payload: Mapping[str, Any]) -> ValidatedFitRun:
    if not isinstance(payload, Mapping):
        raise FitRunImportError("fit-run record must be a JSON object")
    if payload.get("export_schema") != PCS_FIT_RUN_EXPORT_SCHEMA:
        raise FitRunImportError("unsupported single-run export schema")
    _utc_timestamp(payload.get("exported_at_utc"), "exported_at_utc")
    history = payload.get("history_entry")
    if not isinstance(history, Mapping):
        raise FitRunImportError("history_entry is missing or invalid")
    result_id = _uuid(history.get("result_id"))
    created_at = _utc_timestamp(
        history.get("created_at_utc"), "history_entry.created_at_utc"
    )
    source_sequence = _positive_int(
        history.get("run_sequence"), "history_entry.run_sequence"
    )
    label = history.get("label", "")
    if not isinstance(label, str):
        raise FitRunImportError("history_entry.label must be a string")
    try:
        result = serialize_fit_result(payload.get("result"))
    except (TypeError, ValueError) as exc:
        raise FitRunImportError(f"invalid pcs_fit_result_v1: {exc}") from exc
    per_observable = result.get("per_observable") or ()
    per_point = result.get("per_point") or ()
    if len(per_observable) > MAX_OBSERVATIONS_PER_RUN or len(per_point) > MAX_OBSERVATIONS_PER_RUN:
        raise FitRunImportError("fit result exceeds the observation-count limit")
    metadata = result.get("fit_metadata")
    fingerprint = metadata.get("input_fingerprint") if isinstance(metadata, Mapping) else None
    if fingerprint is not None:
        if not isinstance(fingerprint, Mapping):
            raise FitRunImportError("fit result fingerprint metadata must be an object")
        # Unsupported or incomplete historical fingerprint versions are safe
        # to import; the existing status evaluator reports them as Unknown.
        validate_fit_input_fingerprint(fingerprint)
    return ValidatedFitRun(
        result_id=result_id,
        created_at_utc=created_at,
        source_run_sequence=source_sequence,
        label=label,
        result=result,
    )


def _run_records(payload: Mapping[str, Any]) -> list[Any]:
    schema = payload.get("export_schema")
    if schema == PCS_FIT_RUN_EXPORT_SCHEMA:
        return [payload]
    if schema != PCS_FIT_HISTORY_EXPORT_SCHEMA:
        raise FitRunImportError(f"unsupported fit-run export schema: {schema!r}")
    _utc_timestamp(payload.get("exported_at_utc"), "exported_at_utc")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise FitRunImportError("multi-run export must contain a runs list")
    if len(runs) > MAX_IMPORT_RUNS:
        raise FitRunImportError("multi-run export exceeds the run-count limit")
    return runs


def import_fit_run_export(
    state: MutableMapping[str, Any], payload: Mapping[str, Any]
) -> FitRunImportReport:
    if not isinstance(payload, Mapping):
        raise FitRunImportError("fit-run export root must be a JSON object")
    _validate_json_resources(payload)
    records = _run_records(payload)
    imported: list[FitHistoryEntry] = []
    skipped: list[str] = []
    rejected: list[FitRunImportRejection] = []

    for index, record in enumerate(records):
        candidate_id = None
        if isinstance(record, Mapping) and isinstance(record.get("history_entry"), Mapping):
            candidate_id = str(record["history_entry"].get("result_id") or "") or None
        try:
            validated = validate_fit_run_export(record)
            existing = find_fit_history_entry(state, validated.result_id)
            next_sequence = max(
                (entry.sequence for entry in fit_history_entries(state)), default=0
            ) + 1
            candidate = create_fit_history_entry(
                validated.result,
                sequence=next_sequence,
                result_id=validated.result_id,
                created_at=validated.created_at_utc,
                label=validated.label,
            )
            if existing is not None:
                if existing.result_json == candidate.result_json:
                    skipped.append(validated.result_id)
                    continue
                raise FitRunImportError(
                    "Result ID already exists with different content."
                )
            append_fit_history_entry(state, candidate)
            imported.append(candidate)
        except (FitRunImportError, TypeError, ValueError) as exc:
            rejected.append(
                FitRunImportRejection(
                    index=index,
                    result_id=candidate_id,
                    reason=str(exc),
                )
            )

    return FitRunImportReport(
        imported=tuple(imported),
        skipped_duplicate_ids=tuple(skipped),
        rejected=tuple(rejected),
    )


def import_fit_run_export_file(
    state: MutableMapping[str, Any], path: str | Path
) -> FitRunImportReport:
    return import_fit_run_export(state, load_fit_run_export(path))


__all__ = [
    "FitRunImportError",
    "FitRunImportRejection",
    "FitRunImportReport",
    "MAX_IMPORT_FILE_BYTES",
    "MAX_IMPORT_RUNS",
    "MAX_OBSERVATIONS_PER_RUN",
    "ValidatedFitRun",
    "import_fit_run_export",
    "import_fit_run_export_file",
    "load_fit_run_export",
    "validate_fit_run_export",
]
