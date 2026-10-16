"""Portable JSON and CSV export for immutable PCS fit-history records."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from logic.pcs_fit_history import FitHistoryEntry
from logic.pcs_fit_result_status import evaluate_fit_result_status


PCS_FIT_RUN_EXPORT_SCHEMA = "pcs_fit_run_export_v1"
PCS_FIT_HISTORY_EXPORT_SCHEMA = "pcs_fit_history_export_v1"

SUMMARY_COLUMNS = (
    "result_id",
    "run_sequence",
    "label",
    "created_at_utc",
    "runtime_status",
    "mode",
    "weighting_mode",
    "optimizer",
    "n",
    "rmsd",
    "r2",
    "q_factor",
    "raw_sse_per_n",
    "chi_square",
    "chi_square_per_datum",
    "delta_chi_ax",
    "delta_chi_rh",
    "orientation_convention",
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "metal_x",
    "metal_y",
    "metal_z",
)

OBSERVABLE_COLUMNS = (
    "result_id",
    "run_sequence",
    "label",
    "observation_model",
    "observable_id",
    "source_dataset_ids",
    "source_observable_ids",
    "assignment",
    "scope",
    "primary_ref_id",
    "site_ids",
    "member_ref_ids",
    "pcs_exp_ppm",
    "pcs_calc_ppm",
    "residual_ppm",
    "uncertainty_ppm",
    "weighted_residual",
    "matching_status",
    "averaging_hint",
)


def _export_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize_filename_component(value: Any, *, max_length: int = 48) -> str:
    """Return a short cross-platform-safe human-readable filename component."""
    text = str(value or "").strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip(" ._")
    return text[:max_length].rstrip(" ._")


def _mode_label(result: Mapping[str, Any]) -> str:
    return {
        "theta_alpha_multi": "A",
        "euler_global": "B",
        "full_tensor": "C",
    }.get(str(result.get("mode", "")), "unknown")


def default_fit_run_basename(entry: FitHistoryEntry) -> str:
    result = entry.result()
    base = f"pcs_fit_run_{entry.sequence:03d}_mode{_mode_label(result)}"
    label = sanitize_filename_component(entry.label)
    return f"{base}_{label}" if label else base


def default_fit_history_basename(entries: Sequence[FitHistoryEntry]) -> str:
    if len(entries) == 1:
        return default_fit_run_basename(entries[0])
    sequences = sorted(entry.sequence for entry in entries)
    return f"pcs_fit_runs_{sequences[0]:03d}-{sequences[-1]:03d}"


def _runtime_status(result: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, str]:
    status = evaluate_fit_result_status(result, state)
    return {
        "status": str(status.get("status", "unknown")),
        "reason": str(status.get("reason", "")),
    }


def build_fit_run_export(
    entry: FitHistoryEntry,
    state: Mapping[str, Any],
    *,
    exported_at_utc: str | None = None,
) -> dict[str, Any]:
    result = entry.result()
    return {
        "export_schema": PCS_FIT_RUN_EXPORT_SCHEMA,
        "exported_at_utc": exported_at_utc or _export_timestamp(),
        "history_entry": {
            "result_id": entry.result_id,
            "created_at_utc": entry.created_at,
            "run_sequence": entry.sequence,
            "label": entry.label,
        },
        "result": result,
        "runtime_status": _runtime_status(result, state),
    }


def build_fit_history_export(
    entries: Sequence[FitHistoryEntry],
    state: Mapping[str, Any],
    *,
    exported_at_utc: str | None = None,
) -> dict[str, Any]:
    timestamp = exported_at_utc or _export_timestamp()
    return {
        "export_schema": PCS_FIT_HISTORY_EXPORT_SCHEMA,
        "exported_at_utc": timestamp,
        "runs": [
            build_fit_run_export(entry, state, exported_at_utc=timestamp)
            for entry in entries
        ],
    }


def write_json_export(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    return destination


def _orientation_fields(result: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    provenance = result.get("parameter_provenance")
    orientation = provenance.get("orientation") if isinstance(provenance, Mapping) else None
    convention = orientation.get("parameterization") if isinstance(orientation, Mapping) else None
    values = orientation.get("value") if isinstance(orientation, Mapping) else None
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        values = ()
    padded = [*values[:3], None, None, None]
    return convention, padded[0], padded[1], padded[2]


def fit_summary_row(
    entry: FitHistoryEntry, state: Mapping[str, Any]
) -> dict[str, Any]:
    result = entry.result()
    metadata = result.get("fit_metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    convention, ox, oy, oz = _orientation_fields(result)
    metal = result.get("metal_pos")
    if not isinstance(metal, Sequence) or isinstance(metal, (str, bytes)):
        metal = ()
    metal_values = [*metal[:3], None, None, None]
    return {
        "result_id": entry.result_id,
        "run_sequence": entry.sequence,
        "label": entry.label,
        "created_at_utc": entry.created_at,
        "runtime_status": _runtime_status(result, state)["status"],
        "mode": result.get("mode"),
        "weighting_mode": metadata.get("weighting_mode", "unweighted"),
        "optimizer": metadata.get("optimizer"),
        "n": result.get("n"),
        "rmsd": result.get("rmsd"),
        "r2": result.get("r2"),
        "q_factor": result.get("q_factor"),
        "raw_sse_per_n": result.get("chi2_n"),
        "chi_square": result.get("chi_square"),
        "chi_square_per_datum": result.get("chi_square_per_datum"),
        "delta_chi_ax": result.get("delta_chi_ax"),
        "delta_chi_rh": result.get("delta_chi_rh"),
        "orientation_convention": convention,
        "orientation_x": ox,
        "orientation_y": oy,
        "orientation_z": oz,
        "metal_x": metal_values[0],
        "metal_y": metal_values[1],
        "metal_z": metal_values[2],
    }


def _compact_json(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def fit_observable_rows(entry: FitHistoryEntry) -> list[dict[str, Any]]:
    result = entry.result()
    records = result.get("per_observable")
    rows: list[dict[str, Any]] = []
    if records and isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        for record in records:
            if not isinstance(record, Mapping):
                continue
            rows.append({
                "result_id": entry.result_id,
                "run_sequence": entry.sequence,
                "label": entry.label,
                "observation_model": record.get("source") or "structured_v1",
                "observable_id": record.get("observable_id"),
                "source_dataset_ids": _compact_json(record.get("source_dataset_ids", [])),
                "source_observable_ids": _compact_json(record.get("source_observable_ids", [])),
                "assignment": record.get("assignment"),
                "scope": record.get("scope"),
                "primary_ref_id": record.get("primary_ref_id"),
                "site_ids": _compact_json(record.get("site_ids", [])),
                "member_ref_ids": _compact_json(record.get("site_member_ref_ids", [])),
                "pcs_exp_ppm": record.get("observed"),
                "pcs_calc_ppm": record.get("predicted"),
                "residual_ppm": record.get("residual"),
                "uncertainty_ppm": record.get("uncertainty_ppm"),
                "weighted_residual": record.get("weighted_residual"),
                "matching_status": record.get("matching_status"),
                "averaging_hint": record.get("averaging_hint"),
            })
        return rows

    per_point = result.get("per_point", ()) or ()
    for point in per_point:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 4:
            continue
        rows.append({
            "result_id": entry.result_id,
            "run_sequence": entry.sequence,
            "label": entry.label,
            "observation_model": "legacy_single_atom",
            "observable_id": "",
            "source_dataset_ids": "",
            "source_observable_ids": "",
            "assignment": "",
            "scope": "",
            "primary_ref_id": point[0],
            "site_ids": "",
            "member_ref_ids": "",
            "pcs_exp_ppm": point[1],
            "pcs_calc_ppm": point[2],
            "residual_ppm": point[3],
            "uncertainty_ppm": "",
            "weighted_residual": "",
            "matching_status": "",
            "averaging_hint": "",
        })
    return rows


def _csv_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".17g")
    return value


def _write_csv(path: str | Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_scalar(row.get(column)) for column in columns})
    return destination


def write_fit_summary_csv(
    path: str | Path,
    entries: Sequence[FitHistoryEntry],
    state: Mapping[str, Any],
) -> Path:
    return _write_csv(path, SUMMARY_COLUMNS, (fit_summary_row(entry, state) for entry in entries))


def write_fit_observables_csv(path: str | Path, entry: FitHistoryEntry) -> Path:
    return _write_csv(path, OBSERVABLE_COLUMNS, fit_observable_rows(entry))


__all__ = [
    "OBSERVABLE_COLUMNS",
    "PCS_FIT_HISTORY_EXPORT_SCHEMA",
    "PCS_FIT_RUN_EXPORT_SCHEMA",
    "SUMMARY_COLUMNS",
    "build_fit_history_export",
    "build_fit_run_export",
    "default_fit_history_basename",
    "default_fit_run_basename",
    "fit_observable_rows",
    "fit_summary_row",
    "sanitize_filename_component",
    "write_fit_observables_csv",
    "write_fit_summary_csv",
    "write_json_export",
]
