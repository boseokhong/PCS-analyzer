"""
PCS Analyzer project file support (.pcsp).

The .pcsp format is a strict section-based text format designed for
human-readable project snapshots:

    [section]
    key = value

    [table_section]
    # fixed columns
    row row row

The writer always emits the canonical format.  The parser is intentionally
strict so that small manual numeric edits remain safe and easy to diagnose.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json
from math import isfinite

from logic.pcs_prediction_model import (
    get_prediction_model,
    install_prediction_model,
)
from logic.pcs_fit_result_schema import (
    PCS_FIT_RESULT_SCHEMA,
    deserialize_fit_result,
    serialize_fit_result,
)
from logic.pcs_fit_weighting import (
    FitWeightingMode,
    fit_weighting_label,
    parse_fit_weighting_mode,
)
from logic.pcs_fit_history import (
    PCS_FIT_HISTORY_SCHEMA,
    append_fit_history_result,
    clear_fit_history,
    find_fit_history_entry,
    fit_history_entries,
    install_fit_history,
    serialize_fit_history,
)
from logic.pcs_document import (
    DOCUMENT_ENTRY_KEYS,
    DOCUMENT_TK_VARIABLE_KEYS,
    PCSDocument,
    active_document,
    active_document_path,
    capture_active_document,
    commit_pending_document_edits,
    default_document_data,
    hydrate_active_document,
    refresh_active_document_commands,
    suspend_document_tracking,
)

try:
    from app_version import APP_NAME, APP_VERSION
except Exception:  # pragma: no cover - app_version may be unavailable in tests
    APP_NAME = "PCS Analyzer"
    APP_VERSION = "unknown"


SCHEMA_VERSION = "1.1"
PCSP_EXTENSION = ".pcsp"

_TABLE_SECTIONS = {
    "original_atoms",
    "current_atoms",
    "working_atoms",
    "effective_atoms",
    "table_atoms",
    "delta_exp",
    "delta_obs",
    "delta_dia",
    "viewer.pcs_field.levels",
    "rotational_groups",  # PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE2
}


def normalize_project_path(path: str | Path) -> Path:
    path = Path(path)
    return (
        path
        if path.suffix.lower() == PCSP_EXTENSION
        else path.with_suffix(PCSP_EXTENSION)
    )


@dataclass
class AtomRow:
    ref_id: int
    element: str
    x: float
    y: float
    z: float


@dataclass
class EffectiveAtomRow:
    ref_id: int
    element: str
    label: str
    x: float
    y: float
    z: float
    source: str = "working"
    members: str = "-"


@dataclass
class TableAtomRow:
    ref_id: int
    label: str
    x: float
    y: float
    z: float
    gi: str = "none"
    delta_pcs: str = "none"
    delta_exp: str = "none"


class PCSPError(ValueError):
    """Raised when a .pcsp file cannot be parsed or applied."""


def _mode_c_preview(values: Mapping, *, legacy_incomplete: bool = False) -> dict | None:
    """Validate detached Mode C preview values (XYZ degrees, absolute Å)."""
    if legacy_incomplete and not any(
        key in values for key in ("euler_deg", "metal_pos", "dchi_ax", "dchi_rh")
    ):
        # Older writers saved only enabled/mode. There is no geometry to restore.
        return None

    result: dict[str, Any] = {"mode": "full_tensor"}
    for key in ("euler_deg", "metal_pos"):
        value = values.get(key)
        parts = value.split(",") if isinstance(value, str) else value
        try:
            if not isinstance(parts, (list, tuple)) or len(parts) != 3:
                raise ValueError("expected three numbers")
            vector = tuple(float(part) for part in parts)
            if not all(isfinite(part) for part in vector):
                raise ValueError("values must be finite")
        except (ValueError, TypeError) as exc:
            raise PCSPError(f"Invalid Mode C preview {key}: {exc}") from exc
        result[key] = vector
    for key in ("dchi_ax", "dchi_rh"):
        if key not in values:
            continue
        try:
            value = float(values[key])
            if not isfinite(value):
                raise ValueError("value must be finite")
        except (ValueError, TypeError) as exc:
            raise PCSPError(f"Invalid Mode C preview {key}: {exc}") from exc
        result[key] = value
    return result


@dataclass(frozen=True)
class ParsedPCSProject:
    """Detached, validated project input prior to legacy-state mutation."""

    path: Path
    sections: dict[str, list[tuple[int, str]]]


def _project_metadata(project: ParsedPCSProject) -> tuple[str | None, str | None]:
    metadata = parse_key_values(project.sections.get("metadata", []))
    project_name = metadata.get("project_name", "").strip() or None
    tab_color = metadata.get("tab_color", "").strip() or None
    if tab_color and not (
        len(tab_color) == 7
        and tab_color.startswith("#")
        and all(ch in "0123456789abcdefABCDEF" for ch in tab_color[1:])
    ):
        tab_color = None
    return project_name, tab_color


# ---------------------------------------------------------------------------
# Basic formatting / parsing helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _fmt_float(value: Any, digits: int = 8) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.00000000"


def _fmt_scalar(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_bool(value: str, *, default: bool = False) -> bool:
    s = str(value).strip().lower()
    if s in {"true", "yes", "1", "on"}:
        return True
    if s in {"false", "no", "0", "off"}:
        return False
    return bool(default)


def _parse_float(value: str, *, default: float = 0.0) -> float:
    s = str(value).strip()
    if s.lower() in {"", "none", "auto"}:
        return float(default)
    try:
        return float(s)
    except Exception:
        return float(default)


def _parse_int(value: str, *, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return int(default)


def _parse_csv_ints(value: str) -> list[int]:
    s = str(value).strip()
    if not s or s.lower() in {"none", "-"}:
        return []
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def _parse_csv_strings(value: str) -> list[str]:
    s = str(value).strip()
    if not s or s.lower() in {"none", "-"}:
        return []
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def _parse_vec3(value: str, *, default=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    parts = str(value).split()
    if len(parts) != 3:
        return tuple(float(x) for x in default)  # type: ignore[return-value]
    return float(parts[0]), float(parts[1]), float(parts[2])


def _strip_inline_comment(line: str) -> str:
    """
    Remove inline comments while preserving ``#RRGGBB`` color tokens.

    Inline comments begin at ``#`` only when it is the first character of a
    token *and* that token is not a six-digit hexadecimal color.  This keeps
    rows such as ``1  #FF0000  #0000FF  surface  0.3`` intact while still
    allowing ordinary trailing comments such as ``value = 1  # note``.
    """

    def _is_hex_color_token(index: int) -> bool:
        end = index + 7
        if end > len(line):
            return False
        token = line[index:end]
        if len(token) != 7 or token[0] != "#":
            return False
        if not all(ch in "0123456789abcdefABCDEF" for ch in token[1:]):
            return False
        return end == len(line) or line[end].isspace()

    for i, ch in enumerate(line):
        if ch != "#":
            continue
        if i != 0 and not line[i - 1].isspace():
            continue
        if _is_hex_color_token(i):
            continue
        return line[:i].rstrip()
    return line.rstrip()


# ---------------------------------------------------------------------------
# Low-level .pcsp reader
# ---------------------------------------------------------------------------

def read_sections(path: str | Path) -> dict[str, list[tuple[int, str]]]:
    sections: dict[str, list[tuple[int, str]]] = {"__root__": []}
    current = "__root__"

    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw_line = raw.rstrip("\n")
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip()
                if not current:
                    raise PCSPError(f"Line {lineno}: empty section name.")
                sections.setdefault(current, [])
                continue

            if current in _TABLE_SECTIONS:
                stripped = _strip_inline_comment(raw_line).strip()
                if stripped:
                    sections.setdefault(current, []).append((lineno, stripped))
                continue

            stripped = _strip_inline_comment(raw_line).strip()
            if stripped:
                sections.setdefault(current, []).append((lineno, stripped))

    return sections


def parse_key_values(lines: Iterable[tuple[int, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for lineno, line in lines:
        if "=" not in line:
            raise PCSPError(f"Line {lineno}: expected 'key = value', got: {line}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise PCSPError(f"Line {lineno}: empty key.")
        out[key] = value.strip()
    return out


def parse_atom_rows(lines: Iterable[tuple[int, str]]) -> list[AtomRow]:
    atoms: list[AtomRow] = []
    seen: set[int] = set()
    for lineno, line in lines:
        parts = line.split()
        if len(parts) != 5:
            raise PCSPError(f"Line {lineno}: atom row must be 'id element x y z'.")
        try:
            ref_id = int(parts[0])
            element = parts[1]
            x, y, z = map(float, parts[2:5])
        except Exception as exc:
            raise PCSPError(f"Line {lineno}: invalid atom row: {line}") from exc
        if ref_id in seen:
            raise PCSPError(f"Line {lineno}: duplicate atom id {ref_id}.")
        seen.add(ref_id)
        atoms.append(AtomRow(ref_id, element, x, y, z))
    return atoms


def parse_effective_atom_rows(lines: Iterable[tuple[int, str]]) -> list[EffectiveAtomRow]:
    atoms: list[EffectiveAtomRow] = []
    seen: set[int] = set()
    for lineno, line in lines:
        parts = line.split()
        if len(parts) < 8:
            raise PCSPError(
                f"Line {lineno}: effective atom row must be "
                "'id element label x y z source members'."
            )
        try:
            ref_id = int(parts[0])
            element = parts[1]
            label = parts[2]
            x, y, z = map(float, parts[3:6])
            source = parts[6]
            members = parts[7]
        except Exception as exc:
            raise PCSPError(f"Line {lineno}: invalid effective atom row: {line}") from exc
        if ref_id in seen:
            raise PCSPError(f"Line {lineno}: duplicate effective atom id {ref_id}.")
        seen.add(ref_id)
        atoms.append(EffectiveAtomRow(ref_id, element, label, x, y, z, source, members))
    return atoms


def parse_table_atom_rows(lines: Iterable[tuple[int, str]]) -> list[TableAtomRow]:
    rows: list[TableAtomRow] = []
    seen: set[int] = set()
    for lineno, line in lines:
        parts = line.split()
        if len(parts) != 8:
            raise PCSPError(
                f"Line {lineno}: table atom row must be "
                "'id label x y z G_ax delta_pcs delta_exp'."
            )
        try:
            ref_id = int(parts[0])
            label = parts[1]
            x, y, z = map(float, parts[2:5])
            gi = parts[5]
            delta_pcs = parts[6]
            delta_exp = parts[7]
        except Exception as exc:
            raise PCSPError(f"Line {lineno}: invalid table atom row: {line}") from exc
        if ref_id in seen:
            raise PCSPError(f"Line {lineno}: duplicate table atom id {ref_id}.")
        seen.add(ref_id)
        rows.append(TableAtomRow(ref_id, label, x, y, z, gi, delta_pcs, delta_exp))
    return rows


def parse_delta_rows(lines: Iterable[tuple[int, str]]) -> dict[int, float]:
    values: dict[int, float] = {}
    for lineno, line in lines:
        parts = line.split()
        if len(parts) != 2:
            raise PCSPError(f"Line {lineno}: delta row must be 'ref ppm'.")
        try:
            values[int(parts[0])] = float(parts[1])
        except Exception as exc:
            raise PCSPError(f"Line {lineno}: invalid delta row: {line}") from exc
    return values


def parse_experimental_pcs_records(section: dict[str, str]) -> list[dict[str, Any]]:
    """Parse additive structured experimental-PCS metadata.

    The payload lives in an ordinary key/value section so older PCS Analyzer
    versions can safely ignore it instead of failing on unknown table syntax.
    """

    raw = str(section.get("observations_json", "[]") or "[]").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise PCSPError("Invalid [experimental_pcs] observations_json payload.") from exc
    if not isinstance(payload, list):
        raise PCSPError("[experimental_pcs] observations_json must be a JSON list.")
    records: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise PCSPError("Each experimental PCS observation must be a JSON object.")
        records.append(item)
    return records


def parse_level_rows(lines: Iterable[tuple[int, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in lines:
        parts = line.split()
        if len(parts) != 5:
            raise PCSPError(
                f"Line {lineno}: PCS field level row must be "
                "'ppm pos_color neg_color style opacity'."
            )
        try:
            rows.append(
                {
                    "ppm": float(parts[0]),
                    "pos_color": parts[1],
                    "neg_color": parts[2],
                    "style": parts[3],
                    "opacity": float(parts[4]),
                }
            )
        except Exception as exc:
            raise PCSPError(f"Line {lineno}: invalid level row: {line}") from exc
    return rows


def parse_rotational_group_rows(lines: Iterable[tuple[int, str]]) -> list[dict[str, Any]]:
    # Parse JSON-line rotational group records from [rotational_groups].
    rows: list[dict[str, Any]] = []
    for lineno, line in lines:
        try:
            obj = json.loads(line)
        except Exception as exc:
            raise PCSPError(f"Line {lineno}: invalid rotational group JSON.") from exc
        if not isinstance(obj, dict):
            raise PCSPError(f"Line {lineno}: rotational group row must be a JSON object.")
        try:
            obj["axis_atoms"] = tuple(int(x) for x in obj.get("axis_atoms", ()))
            obj["rotating_atoms"] = tuple(int(x) for x in obj.get("rotating_atoms", ()))
            obj["ring_atoms"] = tuple(int(x) for x in obj.get("ring_atoms", ()))
            obj["ring_cycles"] = tuple(tuple(int(x) for x in cyc) for cyc in obj.get("ring_cycles", ()))
            obj["enabled"] = bool(obj.get("enabled", True))
            obj["n_samples"] = max(1, int(obj.get("n_samples", 24)))
            obj["range_start_deg"] = float(obj.get("range_start_deg", 0.0))
            obj["range_end_deg"] = float(obj.get("range_end_deg", 360.0))
            obj["planarity_rms"] = float(obj.get("planarity_rms", 0.0))
        except Exception as exc:
            raise PCSPError(f"Line {lineno}: invalid rotational group values.") from exc
        rows.append(obj)
    return rows


# ---------------------------------------------------------------------------
# State extraction helpers
# ---------------------------------------------------------------------------

def _atoms_from_state(state: dict, key: str, ids_key: str | None = None) -> list[AtomRow]:
    data = state.get(key) or []
    ids = state.get(ids_key) if ids_key else None
    if not ids:
        ids = list(range(1, len(data) + 1))
    rows: list[AtomRow] = []
    for i, atom in enumerate(data):
        try:
            element, x, y, z = atom
            ref_id = int(ids[i]) if i < len(ids) else i + 1
            rows.append(AtomRow(ref_id, str(element), float(x), float(y), float(z)))
        except Exception:
            continue
    return rows


def _effective_atoms_from_state(state: dict) -> list[EffectiveAtomRow]:
    data = state.get("atom_data_eff") or state.get("atom_data") or []
    ids = state.get("atom_ids_eff") or list(range(1, len(data) + 1))
    labels = state.get("ref_label_overrides", {}) or {}
    pseudo_ids = set(state.get("symavg_pseudo_ref_ids", set()) or set())

    member_map: dict[int, str] = {}
    for rec in state.get("symavg_records", []) or []:
        try:
            pseudo_index = int(getattr(rec, "pseudo_index"))
            if pseudo_index < len(ids):
                rid = int(ids[pseudo_index])
                members = getattr(rec, "member_indices_original", ())
                member_map[rid] = ",".join(str(int(v) + 1) for v in members) or "-"
        except Exception:
            continue

    rows: list[EffectiveAtomRow] = []
    for i, atom in enumerate(data):
        try:
            element, x, y, z = atom
            ref_id = int(ids[i]) if i < len(ids) else i + 1
            label = str(labels.get(ref_id, f"{element}{ref_id}"))
            source = "pseudo" if ref_id in pseudo_ids else "working"
            members = member_map.get(ref_id, "-" if source != "pseudo" else "?")
            rows.append(
                EffectiveAtomRow(
                    ref_id,
                    str(element),
                    label,
                    float(x),
                    float(y),
                    float(z),
                    source,
                    members,
                )
            )
        except Exception:
            continue
    return rows


def _as_table_token(value: Any) -> str:
    s = str(value).strip()
    return s if s else "none"


def _table_atoms_from_state(state: dict) -> list[TableAtomRow]:
    """Return a snapshot of the rows currently visible in the main Treeview."""
    tree = state.get("tree")
    if tree is None:
        return []

    rows: list[TableAtomRow] = []
    try:
        for item in tree.get_children():
            vals = list(tree.item(item, "values") or [])
            if len(vals) < 8:
                continue
            ref_id = int(vals[0])
            label = str(vals[1])
            x = float(vals[2])
            y = float(vals[3])
            z = float(vals[4])
            rows.append(
                TableAtomRow(
                    ref_id=ref_id,
                    label=label,
                    x=x,
                    y=y,
                    z=z,
                    gi=_as_table_token(vals[5]),
                    delta_pcs=_as_table_token(vals[6]),
                    delta_exp=_as_table_token(vals[7]),
                )
            )
    except Exception:
        return rows
    return rows


def _get_entry_float(state: dict, key: str, default: float = 0.0) -> float:
    obj = state.get(key)
    try:
        return float(obj.get() if obj is not None else default)
    except Exception:
        return float(default)


def _get_var_value(state: dict, key: str, default: Any = None) -> Any:
    obj = state.get(key)
    try:
        return obj.get()
    except Exception:
        return default


def _project_fit_weighting(value: Any) -> str:
    """Normalize persisted/UI weighting values, defaulting old projects safely."""
    try:
        return parse_fit_weighting_mode(value).value
    except ValueError:
        return FitWeightingMode.UNWEIGHTED.value


def _set_entry(state: dict, key: str, value: Any) -> None:
    obj = state.get(key)
    if obj is None:
        return
    try:
        obj.delete(0, "end")
        obj.insert(0, str(value))
    except Exception:
        pass


def _set_var(state: dict, key: str, value: Any) -> None:
    obj = state.get(key)
    if obj is None:
        return
    try:
        obj.set(value)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _write_kv(lines: list[str], key: str, value: Any) -> None:
    lines.append(f"{key} = {_fmt_scalar(value)}")


def _write_atom_table(lines: list[str], title: str, rows: list[AtomRow]) -> None:
    lines.append(f"[{title}]")
    lines.append("# id element              x              y              z")
    for r in rows:
        lines.append(
            f"{r.ref_id:<5d} {r.element:<4s} "
            f"{_fmt_float(r.x):>14s} {_fmt_float(r.y):>14s} {_fmt_float(r.z):>14s}"
        )
    lines.append("")


def _write_effective_table(lines: list[str], rows: list[EffectiveAtomRow]) -> None:
    lines.append("[effective_atoms]")
    lines.append("# id element label x y z source members")
    for r in rows:
        lines.append(
            f"{r.ref_id:<5d} {r.element:<4s} {r.label:<16s} "
            f"{_fmt_float(r.x):>14s} {_fmt_float(r.y):>14s} {_fmt_float(r.z):>14s} "
            f"{r.source:<8s} {r.members}"
        )
    lines.append("")


def _write_table_atoms(lines: list[str], rows: list[TableAtomRow]) -> None:
    lines.append("[table_atoms]")
    lines.append("# Snapshot of the currently visible main table: id label x y z G_ax delta_pcs delta_exp")
    lines.append("# id label              x              y              z           G_ax       delta_pcs       delta_exp")
    for r in rows:
        lines.append(
            f"{r.ref_id:<5d} {r.label:<16s} "
            f"{_fmt_float(r.x):>14s} {_fmt_float(r.y):>14s} {_fmt_float(r.z):>14s} "
            f"{r.gi:>14s} {r.delta_pcs:>14s} {r.delta_exp:>14s}"
        )
    lines.append("")


def _write_delta_table(lines: list[str], title: str, values: dict[int, float]) -> None:
    lines.append(f"[{title}]")
    lines.append("# ref ppm")
    for ref_id in sorted(values):
        lines.append(f"{int(ref_id):<5d} {float(values[ref_id]): .8f}")
    lines.append("")


def build_pcsp_text(state: dict, *, source_path: str | None = None) -> str:
    lines: list[str] = []
    cfg = state.get("app_settings", {}) or {}

    original = _atoms_from_state(state, "atom_data_original")
    working = _atoms_from_state(state, "atom_data_raw", "atom_ids_raw")
    if not working:
        working = _atoms_from_state(state, "atom_data")
    if not original:
        original = list(working)
    effective = _effective_atoms_from_state(state)
    table_snapshot = _table_atoms_from_state(state)

    source_path = source_path or state.get("current_project_source_path") or state.get("current_structure_path") or ""

    lines.append("# PCS Analyzer Project File")
    lines.append("# Generated by PCS Analyzer. Manual editing is intended for numeric values.")
    lines.append(f"schema_version = {SCHEMA_VERSION}")
    lines.append(f"app_name = {APP_NAME}")
    lines.append(f"app_version = {APP_VERSION}")
    lines.append("")

    lines.append("[metadata]")
    _write_kv(lines, "project_name", state.get("_document_project_name", ""))
    # No whitespace before a leading '#': the PCSP reader otherwise treats it
    # as an inline comment by design.
    tab_color = str(state.get("_document_tab_color", "") or "")
    lines.append(f"tab_color ={tab_color}")
    lines.append("")

    lines.append("[provenance]")
    _write_kv(lines, "created_at", _now_iso())
    _write_kv(lines, "source_path", source_path)
    _write_kv(lines, "source_format", "xyz/orca")
    _write_kv(lines, "coordinate_units", "angstrom")
    lines.append("")

    lines.append("[structure]")
    _write_kv(lines, "atom_id_base", 1)
    _write_kv(lines, "metal_ref_id", state.get("metal_ref_id", "none"))
    metal_el = ""
    try:
        mr = int(state.get("metal_ref_id"))
        for r in working:
            if r.ref_id == mr:
                metal_el = r.element
                break
    except Exception:
        pass
    _write_kv(lines, "metal_element", metal_el or "")
    lines.append(
        "metal_xyz = "
        f"{_fmt_float(state.get('x0', 0.0))} "
        f"{_fmt_float(state.get('y0', 0.0))} "
        f"{_fmt_float(state.get('z0', 0.0))}"
    )
    _write_kv(lines, "authoritative_atoms", "working_atoms")
    _write_kv(lines, "analysis_atoms", "effective_atoms")
    _write_kv(lines, "visible_snapshot", "table_atoms")
    lines.append("")

    _write_atom_table(lines, "original_atoms", original)
    _write_atom_table(lines, "working_atoms", working)
    _write_effective_table(lines, effective)
    _write_table_atoms(lines, table_snapshot)

    lines.append("[symmetry_averaging]")  # PCS_PATCH_AVERAGING_SETTINGS_PHASE1
    _write_kv(lines, "enabled", bool(_get_var_value(state, "symavg_enabled_var", False)))
    _write_kv(lines, "methyl_enabled", bool(_get_var_value(state, "symavg_methyl_enabled_var", True)))
    _write_kv(lines, "cf3_enabled", bool(_get_var_value(state, "symavg_cf3_enabled_var", True)))
    _write_kv(lines, "keep_original", bool(_get_var_value(state, "symavg_keep_original_var", False)))
    _write_kv(lines, "mode", "mask" if bool(_get_var_value(state, "symavg_keep_original_var", False)) else "drop")
    lines.append("")

    lines.append("[rotational_averaging]")  # PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE2
    _write_kv(lines, "enabled", bool(_get_var_value(state, "torsion_avg_enabled_var", False)))
    _write_kv(lines, "detection_mode", _get_var_value(state, "torsion_avg_detection_mode_var", "auto"))
    _write_kv(lines, "keep_reference", bool(_get_var_value(state, "torsion_avg_keep_reference_var", False)))
    _write_kv(lines, "planarity_threshold", _get_var_value(state, "torsion_avg_planarity_var", 0.10))
    _write_kv(lines, "phase", state.get("torsion_avg_phase", 2))
    lines.append("")

    lines.append("[rotational_groups]")
    lines.append("# One JSON object per configured torsional group")
    for group in (state.get("torsion_avg_groups", []) or []):
        try:
            payload = dict(group)
            for key in ("axis_atoms", "rotating_atoms", "ring_atoms"):
                payload[key] = [int(x) for x in payload.get(key, ())]
            payload["ring_cycles"] = [[int(x) for x in cyc] for cyc in payload.get("ring_cycles", ())]
            lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            pass
    lines.append("")

    selected_elements_value = state.get("_visibility_selected_elements")
    if selected_elements_value is None:
        selected_elements = []
        for el, var in (state.get("check_vars", {}) or {}).items():
            try:
                if var.get():
                    selected_elements.append(str(el))
            except Exception:
                pass
        if not selected_elements:
            selected_elements = sorted({str(atom.element) for atom in working})
    else:
        selected_elements = [str(value) for value in selected_elements_value]

    selected_ligands_value = state.get("_visibility_selected_ligands")
    if selected_ligands_value is None:
        selected_ligands = []
        for key, var in (state.get("ligand_check_vars", {}) or {}).items():
            try:
                if var.get():
                    selected_ligands.append(str(key))
            except Exception:
                pass
        if not selected_ligands and working:
            try:
                from logic.visibility import ligand_visibility_records
                selected_ligands = [record.key for record in ligand_visibility_records(state)]
            except Exception:
                selected_ligands = []
    else:
        selected_ligands = [str(value) for value in selected_ligands_value]

    lines.append("[visibility]")
    _write_kv(lines, "selected_elements", ",".join(sorted(selected_elements)))
    _write_kv(
        lines,
        "selected_ligands_json",
        json.dumps(sorted(selected_ligands), ensure_ascii=False, separators=(",", ":")),
    )
    _write_kv(lines, "residual_color_enabled", bool(_get_var_value(state, "residual_color_enabled_var", False)))
    _write_kv(lines, "residual_thr_ok", state.get("residual_thr_ok", 0.10))
    _write_kv(lines, "residual_thr_warn", state.get("residual_thr_warn", 0.30))
    lines.append("")

    lines.append("[pcs_model]")
    _write_kv(lines, "dchi_ax", _get_entry_float(state, "tensor_entry", cfg.get("default_dchi_ax", -2.0)))
    _write_kv(lines, "dchi_rh", state.get("rh_dchi_rh", 0.0))
    _write_kv(lines, "rhombic_enabled", bool(state.get("rh_calc_enabled", False)))
    _write_kv(lines, "prediction_model", get_prediction_model(state).value)
    _write_kv(lines, "pcs_min", _get_entry_float(state, "pcs_min_entry", cfg.get("default_pcs_min", -10.0)))
    _write_kv(lines, "pcs_max", _get_entry_float(state, "pcs_max_entry", cfg.get("default_pcs_max", 10.0)))
    _write_kv(lines, "pcs_interval", _get_entry_float(state, "pcs_interval_entry", cfg.get("default_pcs_interval", 0.5)))
    _write_kv(lines, "plot_90", bool(_get_var_value(state, "plot_90_var", False)))
    lines.append("")

    lines.append("[rotation]")
    _write_kv(lines, "x", _get_var_value(state, "angle_x_var", 0.0))
    _write_kv(lines, "y", _get_var_value(state, "angle_y_var", 0.0))
    _write_kv(lines, "z", _get_var_value(state, "angle_z_var", 0.0))
    _write_kv(lines, "euler_order", "XYZ")
    lines.append("")

    _write_delta_table(lines, "delta_exp", dict(state.get("delta_exp_values", {}) or {}))
    _write_delta_table(lines, "delta_obs", dict(state.get("delta_obs_values", {}) or {}))
    _write_delta_table(lines, "delta_dia", dict(state.get("delta_dia_values", {}) or {}))

    lines.append("[experimental_pcs]")
    _write_kv(lines, "format", "structured_v1")
    _write_kv(
        lines,
        "observations_json",
        json.dumps(
            list(state.get("experimental_pcs_observations", []) or []),
            ensure_ascii=False,
            separators=(",", ":"),
        ).replace("#", "\\u0023"),
    )
    lines.append("")

    from logic.conformer_workflow import project_payload
    lines.append("[multi_conformer]")
    _write_kv(lines, "payload_json", json.dumps(project_payload(
        state.get("advfit_conformers", []), state.get("advfit_mc_options", {}),
        state.get("advfit_mc_last_result")), ensure_ascii=False, allow_nan=False,
        separators=(",", ":")).replace("#", "\\u0023"))
    lines.append("")

    from logic.joint_pcs_rdc_workflow import project_payload as joint_project_payload
    # Additive optional section; existing scientific/standard-fit sections retain their format.
    lines.append("[joint_pcs_rdc]")
    _write_kv(lines, "payload_json", json.dumps(joint_project_payload(
        state.get("advfit_rdc_rows", []), state.get("advfit_rdc_options", {}),
        state.get("advfit_rdc_last_result")), ensure_ascii=False, allow_nan=False,
        separators=(",", ":")).replace("#", "\\u0023"))
    lines.append("")
    from logic.pre_workflow import project_payload as pre_project_payload
    lines.append("[pre_fitting]")
    _write_kv(lines, "payload_json", json.dumps(pre_project_payload(
        state.get("advfit_pre_obs", []), state.get("advfit_pre_options", {}),
        state.get("advfit_pre_last_result")), ensure_ascii=False, allow_nan=False,
        separators=(",", ":")).replace("#", "\\u0023"))
    lines.append("")
    from logic.lanthanide_workflow import project_payload as ml_payload
    lines.append("[multi_lanthanide]")
    _write_kv(lines, "payload_json", json.dumps(ml_payload(state.get("advfit_ml_datasets", []),
        state.get("advfit_ml_options", {}), state.get("advfit_ml_results", {})),
        ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("#", "\\u0023"))
    lines.append("")
    lines.append("[fitting]")
    _write_kv(lines, "mode", _get_var_value(state, "fit_mode_var", "theta_alpha_multi"))
    _write_kv(lines, "selected_proton_ids", "")
    _write_kv(lines, "selected_donor_ids", "")
    _write_kv(lines, "axis_mode", _get_var_value(state, "axis_mode_var", "bisector"))
    _write_kv(lines, "fit_visible_as_group", bool(_get_var_value(state, "fit_use_visible_var", True)))
    _write_kv(lines, "fit_dchi_ax", bool(_get_var_value(state, "fit_dchi_var", False)))
    _write_kv(lines, "fit_dchi_rh", bool(_get_var_value(state, "fit_dchi_rh_var", False)))
    _write_kv(lines, "use_global_search", bool(_get_var_value(state, "fit_global_search_var", False)))
    _write_kv(lines, "use_current_tensor", bool(_get_var_value(state, "fit_use_current_tensor_var", False)))
    _write_kv(
        lines,
        "weighting_mode",
        _project_fit_weighting(_get_var_value(state, "fit_weighting_var", None)),
    )
    lines.append("")

    fo = state.get("fit_override") or {}
    if isinstance(fo, dict) and fo.get("mode") == "full_tensor":
        fo = _mode_c_preview(fo)
    lines.append("[fit_override]")
    _write_kv(lines, "enabled", bool(fo))
    _write_kv(lines, "mode", fo.get("mode", "none") if isinstance(fo, dict) else "none")
    if isinstance(fo, dict):
        for key in ("theta", "alpha", "ax", "ay", "az", "axis_mode"):
            if key in fo:
                _write_kv(lines, key, fo.get(key))
        if "donor_ids" in fo:
            _write_kv(lines, "donor_ids", ",".join(str(v) for v in fo.get("donor_ids") or []))
        if fo.get("mode") == "full_tensor":
            for key in ("euler_deg", "metal_pos"):
                _write_kv(lines, key, ",".join(repr(value) for value in fo[key]))
            for key in ("dchi_ax", "dchi_rh"):
                if key in fo:
                    _write_kv(lines, key, repr(fo[key]))
    lines.append("")

    history_entries = fit_history_entries(state)
    lines.append("[fit_history]")
    _write_kv(lines, "format", PCS_FIT_HISTORY_SCHEMA if history_entries else "")
    if history_entries:
        try:
            _write_kv(
                lines,
                "history_json",
                json.dumps(
                    serialize_fit_history(state),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).replace("#", "\\u0023"),
            )
        except (TypeError, ValueError) as exc:
            print(f"[PCSP] Fit history was not persisted: {exc}")
    lines.append("")

    active_history_entry = find_fit_history_entry(
        state, state.get("active_fit_result_id")
    )
    last = (
        active_history_entry.result()
        if active_history_entry is not None
        else state.get("last_fit_result") or {}
    )
    lines.append("[last_fit_result]")
    _write_kv(lines, "available", bool(last))
    if isinstance(last, dict):
        for key in ("mode", "rmsd", "r2", "q_factor", "n", "theta", "alpha", "delta_chi_ax", "delta_chi_rh"):
            if key in last:
                _write_kv(lines, key, last.get(key))
        # A history-enabled project stores the complete payload exactly once.
        # The legacy section remains a scalar compatibility summary. States
        # without history retain the pre-L14 complete-result representation.
        if not history_entries and last.get("result_schema") == PCS_FIT_RESULT_SCHEMA:
            try:
                payload = serialize_fit_result(last)
                _write_kv(lines, "format", PCS_FIT_RESULT_SCHEMA)
                _write_kv(
                    lines,
                    "result_json",
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).replace("#", "\\u0023"),
                )
            except (TypeError, ValueError) as exc:
                print(f"[PCSP] Fit result was not persisted: {exc}")
    lines.append("")

    last_conf = state.get("last_conformer_result") or {}
    lines.append("[conformer_search]")
    _write_kv(lines, "applied", bool(state.get("conformer_applied", False)))
    _write_kv(lines, "constraint_mode", "angular")
    from logic.conformer_search_project import parse_payload as parse_search_payload
    search_record = parse_search_payload(state.get("conformer_search_session"))
    if search_record is not None:
        _write_kv(lines, "payload_json", json.dumps(search_record, ensure_ascii=False,
                  allow_nan=False, separators=(",", ":")).replace("#", "\\u0023"))
    if isinstance(last_conf, dict):
        _write_kv(lines, "last_rmsd", last_conf.get("rmsd", ""))
        _write_kv(lines, "last_q_factor", last_conf.get("q_factor", ""))
        _write_kv(lines, "last_n_clashes", last_conf.get("n_clashes", ""))
    else:
        _write_kv(lines, "last_rmsd", "")
        _write_kv(lines, "last_q_factor", "")
        _write_kv(lines, "last_n_clashes", "")
    lines.append("")

    scene = state.get("pcs_scene_kwargs", {}) or {}
    lines.append("[viewer.pcs_field]")
    _write_kv(lines, "spacing", scene.get("spacing", 0.35))
    _write_kv(lines, "padding", "auto" if scene.get("padding", None) is None else scene.get("padding"))
    _write_kv(lines, "r_mask_min", scene.get("r_mask_min", 0.8))
    _write_kv(lines, "clip_abs_ppm", "none" if scene.get("clip_abs_ppm", None) is None else scene.get("clip_abs_ppm"))
    _write_kv(lines, "show_slices", scene.get("show_slices", False))
    _write_kv(lines, "show_isosurfaces", scene.get("show_isosurfaces", True))
    _write_kv(lines, "show_labels", scene.get("show_labels", False))
    _write_kv(lines, "show_atoms", scene.get("show_atoms", True))
    _write_kv(lines, "show_bonds", scene.get("show_bonds", True))
    _write_kv(lines, "slice_opacity", scene.get("slice_opacity", 0.25))
    _write_kv(lines, "background", scene.get("background", "white"))
    _write_kv(lines, "ambient_light", scene.get("ambient_light", 0.30))
    lines.append("")

    level_styles = scene.get("level_styles") or []
    lines.append("[viewer.pcs_field.levels]")
    lines.append("# ppm pos_color neg_color style opacity")
    for lv in level_styles:
        try:
            lines.append(
                f"{float(lv.get('ppm', 1.0)):<8g} "
                f"{str(lv.get('pos_color', '#FF0000')):<10s} "
                f"{str(lv.get('neg_color', '#0000FF')):<10s} "
                f"{str(lv.get('style', 'surface')):<10s} "
                f"{float(lv.get('opacity', 0.30)):.3f}"
            )
        except Exception:
            pass
    lines.append("")

    lines.append("[viewer.plot3d]")
    _write_kv(lines, "color_mode", _get_var_value(state, "plot3d_color_mode_var", "Element"))
    _write_kv(lines, "show_labels", bool(_get_var_value(state, "plot3d_show_labels_var", False)))
    lines.append("")

    lines.append("[viewer.projection]")
    _write_kv(lines, "mode", _get_var_value(state, "projection_mode_var", "phi_cos_theta"))
    _write_kv(lines, "fixed_r", _get_var_value(state, "projection_r_var", "10.0"))
    _write_kv(lines, "show_atoms", bool(_get_var_value(state, "projection_show_atoms_var", True)))
    _write_kv(lines, "show_h", bool(_get_var_value(state, "projection_show_h_var", True)))
    lines.append("")

    lines.append("[export]")
    _write_kv(lines, "default_dpi", cfg.get("export_default_dpi", 600))
    lines.append("")

    return "\n".join(lines)


def build_document_pcsp_text(
    document: PCSDocument,
    state: dict,
    *,
    source_path: str | None = None,
) -> str:
    """Serialize one document through the unchanged canonical writer.

    The writer and the ``.pcsp`` schema intentionally remain unchanged.  This
    adapter projects plain document data into scalar stand-ins so inactive
    documents can be saved without changing the visible legacy UI.
    """
    if active_document(state) is document:
        commit_pending_document_edits(state)
        capture_active_document(state, document)
    projected = _project_document_state(document, state)
    return build_pcsp_text(projected, source_path=source_path)


class _DetachedValue:
    """Minimal Tk variable/Entry protocol used by detached project I/O."""

    def __init__(self, value: Any = None) -> None:
        self.value = value

    def get(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = value

    def delete(self, _start: Any, _end: Any) -> None:
        self.value = ""

    def insert(self, _index: Any, value: Any) -> None:
        self.value = str(value)


def _project_document_state(
    document: PCSDocument,
    application_state: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = default_document_data(application_state)
    if defaults is not None:
        values.update(deepcopy(dict(defaults)))
    values.update(document.snapshot())
    projected: dict[str, Any] = {
        "app_settings": deepcopy(
            dict(application_state.get("app_settings", {}) or {})
        )
    }
    projected.update(deepcopy(values))
    projected["_document_project_name"] = document.project_name or ""
    projected["_document_tab_color"] = document.tab_color or ""
    for key in DOCUMENT_TK_VARIABLE_KEYS:
        projected[key] = _DetachedValue(values.get(key))
    for key in DOCUMENT_ENTRY_KEYS:
        projected[key] = _DetachedValue(values.get(key, ""))
    projected["check_vars"] = {
        str(element): _DetachedValue(True)
        for element in (values.get("_visibility_selected_elements") or ())
    }
    projected["ligand_check_vars"] = {
        str(key): _DetachedValue(True)
        for key in (values.get("_visibility_selected_ligands") or ())
    }
    return projected


def load_project_document(
    path: str | Path,
    state: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any] | None = None,
) -> PCSDocument:
    """Parse and materialize a detached document without touching the UI."""
    project = parse_project_file(path)
    seed = PCSDocument(title=project.path.name, file_path=project.path)
    projected = _project_document_state(seed, state, defaults=defaults)
    _apply_parsed_project(project, projected)
    prediction_var = projected.get("pcs_prediction_model_var")
    if isinstance(prediction_var, _DetachedValue):
        prediction_var.set(projected.get("pcs_prediction_model", "axial"))
    projected["check_vars"] = {
        str(element): _DetachedValue(True)
        for element in (projected.get("_visibility_selected_elements") or ())
    }
    projected["ligand_check_vars"] = {
        str(key): _DetachedValue(True)
        for key in (projected.get("_visibility_selected_ligands") or ())
    }
    document = PCSDocument(title=project.path.name, file_path=project.path)
    capture_active_document(projected, document)
    document.project_name, document.tab_color = _project_metadata(project)
    document.revision = 0
    document.mark_clean(project.path)
    return document


def save_project_file(
    path: str | Path,
    state: dict,
    document: PCSDocument | None = None,
) -> None:
    path = normalize_project_path(path)
    document = document or active_document(state)
    text = (
        build_document_pcsp_text(document, state)
        if document is not None
        else build_pcsp_text(state)
    )
    path.write_text(text, encoding="utf-8")
    if document is not None:
        document.mark_clean(path)
        if active_document(state) is document:
            state["current_project_path"] = str(path)
    else:
        state["current_project_path"] = str(path)
    refresh_tabs = state.get("refresh_document_tabs")
    if callable(refresh_tabs):
        try:
            refresh_tabs()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Loader / state applier
# ---------------------------------------------------------------------------

def _atoms_to_state_data(rows: list[AtomRow]) -> tuple[list[tuple[str, float, float, float]], list[int]]:
    return [(r.element, r.x, r.y, r.z) for r in rows], [r.ref_id for r in rows]


def parse_project_file(path: str | Path) -> ParsedPCSProject:
    """Read and validate project input without mutating application state."""
    path = Path(path)
    sections = read_sections(path)

    # Validate every structured section consumed by the state applier before
    # the first legacy-state assignment.  Optional malformed fit history keeps
    # its established tolerant behavior and is therefore validated on apply.
    for name in (
        "__root__", "metadata", "structure", "symmetry_averaging",
        "rotational_averaging", "visibility", "pcs_model", "rotation",
        "fitting", "multi_conformer", "joint_pcs_rdc", "pre_fitting", "multi_lanthanide", "experimental_pcs", "fit_override", "fit_history",
        "last_fit_result", "conformer_search", "viewer.pcs_field", "viewer.plot3d",
        "viewer.projection",
    ):
        parse_key_values(sections.get(name, []))
    parse_atom_rows(sections.get("original_atoms", []))
    working_rows = parse_atom_rows(sections.get("working_atoms", []))
    if not working_rows:
        working_rows = parse_atom_rows(sections.get("current_atoms", []))
    if not working_rows:
        raise PCSPError("Project does not contain [working_atoms].")
    parse_table_atom_rows(sections.get("table_atoms", []))
    parse_rotational_group_rows(sections.get("rotational_groups", []))
    for name in ("delta_exp", "delta_obs", "delta_dia"):
        parse_delta_rows(sections.get(name, []))
    parse_level_rows(sections.get("viewer.pcs_field.levels", []))
    preview = parse_key_values(sections.get("fit_override", []))
    if _parse_bool(preview.get("enabled", "false")) and preview.get("mode") == "full_tensor":
        _mode_c_preview(preview, legacy_incomplete=True)
    from logic.conformer_search_project import parse_payload as parse_search_payload
    search = parse_key_values(sections.get("conformer_search", []))
    try:
        parse_search_payload(json.loads(search["payload_json"]) if search.get("payload_json") else None)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise PCSPError(f"Invalid Conformer Search section: {exc}") from exc
    experimental = parse_key_values(sections.get("experimental_pcs", []))
    parse_experimental_pcs_records(experimental)
    from logic.conformer_workflow import parse_project_payload
    mc = parse_key_values(sections.get("multi_conformer", []))
    try:
        parse_project_payload(json.loads(mc["payload_json"]) if mc.get("payload_json") else None)
    except (ValueError, TypeError, KeyError) as exc:
        raise PCSPError(f"Invalid Multi-Conformer section: {exc}") from exc
    from logic.joint_pcs_rdc_workflow import parse_project_payload as parse_joint_payload
    joint = parse_key_values(sections.get("joint_pcs_rdc", []))
    try:
        parse_joint_payload(json.loads(joint["payload_json"]) if joint.get("payload_json") else None)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise PCSPError(f"Invalid Joint PCS+RDC section: {exc}") from exc
    from logic.pre_workflow import parse_project_payload as parse_pre_payload
    pre = parse_key_values(sections.get("pre_fitting", []))
    try:
        parse_pre_payload(json.loads(pre["payload_json"]) if pre.get("payload_json") else None)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise PCSPError(f"Invalid PRE section: {exc}") from exc
    from logic.lanthanide_workflow import parse_project_payload as ml_payload
    ml = parse_key_values(sections.get("multi_lanthanide", []))
    try:
        ml_payload(json.loads(ml["payload_json"]) if ml.get("payload_json") else None)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise PCSPError(f"Invalid Multi-Lanthanide section: {exc}") from exc
    return ParsedPCSProject(path=path, sections=sections)


def _apply_parsed_project(project: ParsedPCSProject, state: dict) -> None:
    path = project.path
    sections = project.sections

    root_kv = parse_key_values(sections.get("__root__", []))
    schema = root_kv.get("schema_version", "")
    if schema and schema != SCHEMA_VERSION:
        # Strict enough to warn, permissive enough for future minor migration.
        print(f"[PCSP] Loading schema_version={schema}; expected {SCHEMA_VERSION}.")

    structure = parse_key_values(sections.get("structure", []))
    symavg = parse_key_values(sections.get("symmetry_averaging", []))
    rotavg = parse_key_values(sections.get("rotational_averaging", []))  # PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE2
    rotgroups = parse_rotational_group_rows(sections.get("rotational_groups", []))
    visibility = parse_key_values(sections.get("visibility", []))
    from logic.residual_display import load_thresholds
    state["residual_thr_ok"], state["residual_thr_warn"] = load_thresholds(
        visibility.get("residual_thr_ok", 0.10), visibility.get("residual_thr_warn", 0.30)
    )
    _set_var(state, "residual_color_enabled_var", _parse_bool(visibility.get("residual_color_enabled", "false")))
    if "selected_elements" in visibility:
        state["_visibility_selected_elements"] = tuple(
            sorted(_parse_csv_strings(visibility.get("selected_elements", "")))
        )
    else:
        state["_visibility_selected_elements"] = None

    if "selected_ligands_json" in visibility:
        try:
            parsed_ligands = json.loads(visibility.get("selected_ligands_json", "[]"))
            if not isinstance(parsed_ligands, list):
                raise ValueError("selected_ligands_json must contain a JSON list")
            state["_visibility_selected_ligands"] = tuple(
                sorted(str(value) for value in parsed_ligands)
            )
        except Exception:
            state["_visibility_selected_ligands"] = None
    else:
        # Older .pcsp files had only the element visibility field; all detected
        # ligands therefore default to visible.
        state["_visibility_selected_ligands"] = None
    pcs_model = parse_key_values(sections.get("pcs_model", []))
    rotation = parse_key_values(sections.get("rotation", []))
    fitting = parse_key_values(sections.get("fitting", []))
    experimental_pcs = parse_key_values(sections.get("experimental_pcs", []))
    fit_override = parse_key_values(sections.get("fit_override", []))
    fit_history = parse_key_values(sections.get("fit_history", []))
    last_fit_result = parse_key_values(sections.get("last_fit_result", []))
    from logic.conformer_search_project import parse_payload as parse_search_payload, preview_payload
    search = parse_key_values(sections.get("conformer_search", []))
    search_record = parse_search_payload(json.loads(search["payload_json"]) if search.get("payload_json") else None)
    state["conformer_search_session"] = search_record
    search_preview = preview_payload(search_record) if search_record else None
    state["conformer_applied"] = bool(search_record and search_record["applied"])
    state["conformer_preview"] = search_preview
    for key, field in (("conformer_preview_coords", "coords_opt"),
                       ("conformer_preview_elements", "elements"),
                       ("conformer_preview_report", "report"), ("last_conformer_result", "result")):
        state[key] = search_preview[field] if search_preview else None
    viewer_field = parse_key_values(sections.get("viewer.pcs_field", []))
    viewer_plot3d = parse_key_values(sections.get("viewer.plot3d", []))
    viewer_projection = parse_key_values(sections.get("viewer.projection", []))

    original_rows = parse_atom_rows(sections.get("original_atoms", []))
    working_rows = parse_atom_rows(sections.get("working_atoms", []))
    if not working_rows:
        # Backward compatibility: v1 project files used [current_atoms].
        working_rows = parse_atom_rows(sections.get("current_atoms", []))
    if not working_rows:
        raise PCSPError("Project does not contain [working_atoms].")

    table_rows = parse_table_atom_rows(sections.get("table_atoms", []))

    original_data, _original_ids = _atoms_to_state_data(original_rows or working_rows)
    current_data, current_ids = _atoms_to_state_data(working_rows)

    # Coordinates and metal center
    state["atom_data_original"] = list(original_data)
    state["atom_data_raw"] = list(current_data)
    state["atom_data"] = list(current_data)
    state["atom_ids_raw"] = list(current_ids)
    state["pcsp_table_atoms_snapshot"] = table_rows

    metal_ref_id = _parse_int(structure.get("metal_ref_id", "1"), default=1)
    state["metal_ref_id"] = metal_ref_id
    metal_xyz = _parse_vec3(structure.get("metal_xyz", "0 0 0"))
    state["x0"], state["y0"], state["z0"] = metal_xyz

    # UI variables / entries
    state["tensor"] = _parse_float(pcs_model.get("dchi_ax", "-2.0"), default=-2.0)
    _set_entry(state, "tensor_entry", pcs_model.get("dchi_ax", "-2.0"))
    state["rh_dchi_rh"] = _parse_float(pcs_model.get("dchi_rh", "0.0"), default=0.0)
    _set_var(state, "rh_dchi_rh_var", str(state["rh_dchi_rh"]))
    state["rh_calc_enabled"] = _parse_bool(pcs_model.get("rhombic_enabled", "false"))
    _set_var(state, "rh_calc_enabled_var", state["rh_calc_enabled"])
    install_prediction_model(
        state,
        pcs_model.get("prediction_model", "axial"),
    )

    _set_entry(state, "pcs_min_entry", pcs_model.get("pcs_min", "-10.0"))
    _set_entry(state, "pcs_max_entry", pcs_model.get("pcs_max", "10.0"))
    _set_entry(state, "pcs_interval_entry", pcs_model.get("pcs_interval", "0.5"))
    _set_var(state, "plot_90_var", _parse_bool(pcs_model.get("plot_90", "false")))

    ax = _parse_float(rotation.get("x", "0.0"))
    ay = _parse_float(rotation.get("y", "0.0"))
    az = _parse_float(rotation.get("z", "0.0"))
    state["angle_x"], state["angle_y"], state["angle_z"] = ax, ay, az
    _set_var(state, "angle_x_var", ax)
    _set_var(state, "angle_y_var", ay)
    _set_var(state, "angle_z_var", az)
    _set_entry(state, "angle_x_entry", f"{ax:.1f}")
    _set_entry(state, "angle_y_entry", f"{ay:.1f}")
    _set_entry(state, "angle_z_entry", f"{az:.1f}")

    _set_var(state, "symavg_enabled_var", _parse_bool(symavg.get("enabled", "false")))  # PCS_PATCH_AVERAGING_SETTINGS_PHASE1
    _set_var(state, "symavg_methyl_enabled_var", _parse_bool(symavg.get("methyl_enabled", "true"), default=True))
    _set_var(state, "symavg_cf3_enabled_var", _parse_bool(symavg.get("cf3_enabled", "true"), default=True))
    _set_var(state, "symavg_keep_original_var", _parse_bool(symavg.get("keep_original", "false")))

    _set_var(state, "torsion_avg_enabled_var", _parse_bool(rotavg.get("enabled", "false")))  # PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE2
    _set_var(state, "torsion_avg_detection_mode_var", rotavg.get("detection_mode", "auto"))
    _set_var(state, "torsion_avg_keep_reference_var", _parse_bool(rotavg.get("keep_reference", "false")))
    _set_var(state, "torsion_avg_planarity_var", _parse_float(rotavg.get("planarity_threshold", "0.10"), default=0.10))
    state["torsion_avg_groups"] = rotgroups
    state["torsion_avg_phase"] = _parse_int(rotavg.get("phase", "2"), default=2)

    # Data layers
    state["delta_exp_values"] = parse_delta_rows(sections.get("delta_exp", []))
    state["delta_obs_values"] = parse_delta_rows(sections.get("delta_obs", []))
    state["delta_dia_values"] = parse_delta_rows(sections.get("delta_dia", []))

    state["experimental_pcs_observations"] = parse_experimental_pcs_records(
        experimental_pcs
    )

    from logic.conformer_workflow import parse_project_payload
    mc = parse_key_values(sections.get("multi_conformer", []))
    conformers, mc_options, mc_result = parse_project_payload(json.loads(mc["payload_json"]) if mc.get("payload_json") else None)
    state["advfit_conformers"] = conformers
    state["advfit_mc_options"] = mc_options
    state["advfit_mc_last_result"] = mc_result
    from logic.joint_pcs_rdc_workflow import parse_project_payload as parse_joint_payload
    joint = parse_key_values(sections.get("joint_pcs_rdc", []))
    joint_rows, joint_options, joint_result = parse_joint_payload(json.loads(joint["payload_json"]) if joint.get("payload_json") else None)
    state["advfit_rdc_rows"] = joint_rows
    state["advfit_rdc_options"] = joint_options
    state["advfit_rdc_last_result"] = joint_result
    from logic.pre_workflow import parse_project_payload as parse_pre_payload
    pre = parse_key_values(sections.get("pre_fitting", []))
    pre_rows, pre_options, pre_result = parse_pre_payload(json.loads(pre["payload_json"]) if pre.get("payload_json") else None)
    state["advfit_pre_obs"] = pre_rows
    state["advfit_pre_options"] = pre_options
    state["advfit_pre_last_result"] = pre_result
    from logic.lanthanide_workflow import parse_project_payload as ml_payload
    ml = parse_key_values(sections.get("multi_lanthanide", []))
    ml_data, ml_options, ml_results = ml_payload(json.loads(ml["payload_json"]) if ml.get("payload_json") else None)
    state.update(advfit_ml_datasets=ml_data, advfit_ml_options=ml_options, advfit_ml_results=ml_results,
                 advfit_ml_last_result=ml_results.get(ml_options['mode']))

    # Fitting state
    _set_var(state, "fit_mode_var", fitting.get("mode", "theta_alpha_multi"))
    _set_var(state, "axis_mode_var", fitting.get("axis_mode", "bisector"))
    _set_var(state, "fit_use_visible_var", _parse_bool(fitting.get("fit_visible_as_group", "true"), default=True))
    _set_var(state, "fit_dchi_var", _parse_bool(fitting.get("fit_dchi_ax", "false")))
    _set_var(state, "fit_dchi_rh_var", _parse_bool(fitting.get("fit_dchi_rh", "false")))
    _set_var(state, "fit_global_search_var", _parse_bool(fitting.get("use_global_search", "false")))
    _set_var(state, "fit_use_current_tensor_var", _parse_bool(fitting.get("use_current_tensor", "false")))
    _set_var(
        state,
        "fit_weighting_var",
        fit_weighting_label(_project_fit_weighting(fitting.get("weighting_mode"))),
    )
    refresh_fit_initial_guess = state.get("refresh_fit_initial_guess_ui")
    if callable(refresh_fit_initial_guess):
        refresh_fit_initial_guess()

    if _parse_bool(fit_override.get("enabled", "false")):
        fo: dict[str, Any] = {"mode": fit_override.get("mode", "none")}
        for key in ("theta", "alpha", "ax", "ay", "az"):
            if key in fit_override:
                fo[key] = _parse_float(fit_override[key])
        if "axis_mode" in fit_override:
            fo["axis_mode"] = fit_override["axis_mode"]
        if "donor_ids" in fit_override:
            fo["donor_ids"] = _parse_csv_ints(fit_override["donor_ids"])
        if fo["mode"] == "full_tensor":
            fo = _mode_c_preview(fit_override, legacy_incomplete=True)
        if fo is None:
            state.pop("fit_override", None)
        else:
            state["fit_override"] = fo
    else:
        state.pop("fit_override", None)

    # Structured fit results are additive to the legacy scalar summary.  The
    # old summary was never sufficient to reconstruct a scientifically usable
    # result, so it is intentionally not fabricated on load.
    clear_fit_history(state)
    state.pop("_last_fit_result_restored", None)
    state.pop("_last_fit_result_restore_error", None)
    state["_last_fit_result_load_processed"] = True
    history_format = str(fit_history.get("format", "") or "").strip()
    history_json = str(fit_history.get("history_json", "") or "").strip()
    history_restored = False
    if history_format or history_json:
        try:
            if history_format != PCS_FIT_HISTORY_SCHEMA:
                raise ValueError(f"unsupported fit history format: {history_format}")
            if not history_json:
                raise ValueError("fit history payload is missing")
            install_fit_history(state, json.loads(history_json))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            message = f"Fit history was not restored: {exc}"
            state["_last_fit_result_restore_error"] = message
            print(f"[PCSP] {message}")
        else:
            state["_last_fit_result_restored"] = True
            history_restored = True

    # Pre-L14 projects carry one complete result in [last_fit_result]. Import
    # it as a one-entry history. A malformed new history may also fall back to
    # a valid legacy result if one is present.
    if not history_restored:
        result_format = str(last_fit_result.get("format", "") or "").strip()
        result_json = str(last_fit_result.get("result_json", "") or "").strip()
        if result_format or result_json:
            try:
                if result_format != PCS_FIT_RESULT_SCHEMA:
                    raise ValueError(f"unsupported fit result format: {result_format}")
                if not result_json:
                    raise ValueError("structured fit result payload is missing")
                restored_result = deserialize_fit_result(json.loads(result_json))
                append_fit_history_result(state, restored_result)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                message = f"Fit result was not restored: {exc}"
                state["_last_fit_result_restore_error"] = message
                print(f"[PCSP] {message}")
            else:
                state["_last_fit_result_restored"] = True

    # Viewer settings cache
    level_styles = parse_level_rows(sections.get("viewer.pcs_field.levels", []))
    padding_text = viewer_field.get("padding", "auto")
    clip_text = viewer_field.get("clip_abs_ppm", "none")
    state["pcs_scene_kwargs"] = {
        "spacing": _parse_float(viewer_field.get("spacing", "0.35"), default=0.35),
        "padding": None if padding_text.lower() in {"auto", "none", ""} else _parse_float(padding_text),
        "r_mask_min": _parse_float(viewer_field.get("r_mask_min", "0.8"), default=0.8),
        "clip_abs_ppm": None if clip_text.lower() in {"none", "auto", ""} else _parse_float(clip_text),
        "show_slices": _parse_bool(viewer_field.get("show_slices", "false")),
        "show_isosurfaces": _parse_bool(viewer_field.get("show_isosurfaces", "true"), default=True),
        "show_labels": _parse_bool(viewer_field.get("show_labels", "false")),
        "show_atoms": _parse_bool(viewer_field.get("show_atoms", "true"), default=True),
        "show_bonds": _parse_bool(viewer_field.get("show_bonds", "true"), default=True),
        "slice_opacity": _parse_float(viewer_field.get("slice_opacity", "0.25"), default=0.25),
        "background": viewer_field.get("background", "white"),
        "ambient_light": _parse_float(viewer_field.get("ambient_light", "0.30"), default=0.30),
        "level_styles": level_styles,
    }

    _set_var(state, "plot3d_color_mode_var", viewer_plot3d.get("color_mode", "Element"))
    _set_var(state, "plot3d_show_labels_var", _parse_bool(viewer_plot3d.get("show_labels", "false")))
    _set_var(state, "projection_mode_var", viewer_projection.get("mode", "phi_cos_theta"))
    _set_var(state, "projection_r_var", viewer_projection.get("fixed_r", "10.0"))
    _set_var(state, "projection_show_atoms_var", _parse_bool(viewer_projection.get("show_atoms", "true"), default=True))
    _set_var(state, "projection_show_h_var", _parse_bool(viewer_projection.get("show_h", "true"), default=True))

    # Rebuild derived structures and UI.
    apply_symavg = state.get("apply_symavg_to_state")
    if callable(apply_symavg):
        apply_symavg(state)

    refresh_averaging = state.get("refresh_averaging_status")  # PCS_PATCH_AVERAGING_SETTINGS_PHASE1
    if callable(refresh_averaging):
        refresh_averaging()

    create_checklist = state.get("create_checklist")
    if callable(create_checklist):
        # The checklist hydrates both element and ligand variables directly from
        # the document-owned plain visibility selections above.
        create_checklist(state)

    try:
        from logic.nmr_delta_data_manager import recompute_delta_para, push_layers_to_nmr_if_open
        recompute_delta_para(state)
        push_layers_to_nmr_if_open(state)
    except Exception:
        pass

    for key in ("update_graph", "plot_cartesian", "rh_refresh_table"):
        fn = state.get(key)
        if callable(fn):
            try:
                if key == "plot_cartesian":
                    fn(state)
                else:
                    fn()
            except Exception:
                pass

    workspace = state.get("mc_workspace")
    if workspace is not None:
        workspace.refresh()

    refresh_orientation = state.get("refresh_orientation_editor")
    if callable(refresh_orientation):
        try:
            refresh_orientation()
        except Exception:
            pass

    refresh_prediction_model = state.get("refresh_prediction_model_ui")
    if callable(refresh_prediction_model):
        try:
            refresh_prediction_model()
        except Exception:
            pass

    pop_fit = state.get("populate_fitting_controls")
    if callable(pop_fit):
        try:
            pop_fit(state)
        except Exception:
            pass

    refresh_fit_result = state.get("refresh_last_fit_result_ui")
    if callable(refresh_fit_result):
        try:
            refresh_fit_result()
        except Exception:
            pass

    refresh_fit_history = state.get("refresh_fit_history_ui")
    if callable(refresh_fit_history):
        try:
            refresh_fit_history()
        except Exception:
            pass

    refresh_search = state.get("refresh_conformer_search_ui")
    if callable(refresh_search):
        refresh_search()

    state["current_project_path"] = str(path)


def load_project_file(
    path: str | Path,
    state: dict,
    document: PCSDocument | None = None,
) -> None:
    """Atomically load one project into the active single-document surface."""
    project = parse_project_file(path)
    document = document or active_document(state)
    if document is None:
        _apply_parsed_project(project, state)
        return
    if active_document(state) is not document:
        raise ValueError("Only the active PCS document can be loaded")

    capture_active_document(state, document)
    backup_data = document.snapshot()
    backup_path = document.file_path
    backup_title = document.title
    backup_dirty = document.dirty
    backup_revision = document.revision
    backup_project_name = document.project_name
    backup_tab_color = document.tab_color
    try:
        with suspend_document_tracking(state):
            _apply_parsed_project(project, state)
            capture_active_document(state, document)
            document.project_name, document.tab_color = _project_metadata(project)
    except Exception:
        document.replace_data(backup_data)
        document.file_path = backup_path
        document.title = backup_title
        document.dirty = backup_dirty
        document.revision = backup_revision
        document.project_name = backup_project_name
        document.tab_color = backup_tab_color
        hydrate_active_document(state, document, refresh=True)
        raise
    # Replacing the document contents invalidates any worker launched against
    # the previous snapshot even though the freshly loaded file is clean.
    document.mark_mutated()
    document.mark_clean(project.path)
    state["current_project_path"] = str(project.path)
    if document.edit_history is not None:
        document.edit_history.clear()
    refresh_history = state.get("refresh_edit_history")
    if callable(refresh_history):
        refresh_history()
    refresh_active_document_commands(state)
    refresh_tabs = state.get("refresh_document_tabs")
    if callable(refresh_tabs):
        try:
            refresh_tabs()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dialog wrappers used by the UI menu
# ---------------------------------------------------------------------------

def save_project_dialog(state: dict) -> None:
    filedialog = state.get("filedialog")
    messagebox = state.get("messagebox")
    if filedialog is None:
        raise RuntimeError("filedialog is not available in state.")

    document_path = active_document_path(state)
    initial = str(document_path) if document_path is not None else "pcs_project.pcsp"
    path = filedialog.asksaveasfilename(
        title="Save PCS project",
        defaultextension=PCSP_EXTENSION,
        initialfile=Path(initial).name,
        filetypes=[("PCS project", "*.pcsp"), ("All files", "*.*")],
    )
    if not path:
        return

    try:
        manager = state.get("document_manager")
        if manager is not None:
            manager.save_document(path=path)
        else:
            save_project_file(path, state)
        if messagebox is not None:
            messagebox.showinfo("Save PCS project", f"Project saved:\n{path}")
    except Exception as exc:
        if messagebox is not None:
            messagebox.showerror("Save PCS project", str(exc))
        else:
            raise


def load_project_dialog(state: dict) -> None:
    filedialog = state.get("filedialog")
    messagebox = state.get("messagebox")
    if filedialog is None:
        raise RuntimeError("filedialog is not available in state.")

    path = filedialog.askopenfilename(
        title="Open PCS project",
        filetypes=[("PCS project", "*.pcsp"), ("All files", "*.*")],
    )
    if not path:
        return

    try:
        manager = state.get("document_manager")
        if manager is not None:
            manager.open_document(path, activate=True)
        else:
            load_project_file(path, state)
        refresh_tabs = state.get("refresh_document_tabs")
        if callable(refresh_tabs):
            try:
                refresh_tabs()
            except Exception:
                pass
        if messagebox is not None:
            messagebox.showinfo("Open PCS project", f"Project loaded:\n{path}")
    except Exception as exc:
        if messagebox is not None:
            messagebox.showerror("Open PCS project", str(exc))
        else:
            raise
