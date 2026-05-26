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

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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
}


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
    Remove inline comments only when # is preceded by whitespace.

    This preserves color strings such as #FF0000 inside table rows.
    """
    for i, ch in enumerate(line):
        if ch == "#" and (i == 0 or line[i - 1].isspace()):
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
                "'id label x y z G_i delta_pcs delta_exp'."
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
    lines.append("# Snapshot of the currently visible main table: id label x y z G_i delta_pcs delta_exp")
    lines.append("# id label              x              y              z            G_i       delta_pcs       delta_exp")
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

    lines.append("[symmetry_averaging]")
    _write_kv(lines, "enabled", bool(_get_var_value(state, "symavg_enabled_var", False)))
    _write_kv(lines, "methyl_enabled", True)
    _write_kv(lines, "cf3_enabled", True)
    _write_kv(lines, "keep_original", bool(_get_var_value(state, "symavg_keep_original_var", False)))
    _write_kv(lines, "mode", "mask" if bool(_get_var_value(state, "symavg_keep_original_var", False)) else "drop")
    lines.append("")

    selected_elements = []
    for el, var in (state.get("check_vars", {}) or {}).items():
        try:
            if var.get():
                selected_elements.append(str(el))
        except Exception:
            pass

    lines.append("[visibility]")
    _write_kv(lines, "selected_elements", ",".join(sorted(selected_elements)))
    _write_kv(lines, "residual_color_enabled", bool(_get_var_value(state, "residual_color_enabled_var", False)))
    _write_kv(lines, "residual_thr_ok", state.get("residual_thr_ok", 0.10))
    _write_kv(lines, "residual_thr_warn", state.get("residual_thr_warn", 0.30))
    lines.append("")

    lines.append("[pcs_model]")
    _write_kv(lines, "dchi_ax", _get_entry_float(state, "tensor_entry", cfg.get("default_dchi_ax", -2.0)))
    _write_kv(lines, "dchi_rh", state.get("rh_dchi_rh", 0.0))
    _write_kv(lines, "rhombic_enabled", bool(state.get("rh_calc_enabled", False)))
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

    lines.append("[fitting]")
    _write_kv(lines, "mode", _get_var_value(state, "fit_mode_var", "theta_alpha_multi"))
    _write_kv(lines, "selected_proton_ids", "")
    _write_kv(lines, "selected_donor_ids", "")
    _write_kv(lines, "axis_mode", _get_var_value(state, "axis_mode_var", "bisector"))
    _write_kv(lines, "fit_visible_as_group", bool(_get_var_value(state, "fit_use_visible_var", True)))
    _write_kv(lines, "fit_dchi_ax", bool(_get_var_value(state, "fit_dchi_var", False)))
    _write_kv(lines, "fit_dchi_rh", bool(_get_var_value(state, "fit_dchi_rh_var", False)))
    _write_kv(lines, "use_global_search", bool(_get_var_value(state, "fit_global_search_var", False)))
    lines.append("")

    fo = state.get("fit_override") or {}
    lines.append("[fit_override]")
    _write_kv(lines, "enabled", bool(fo))
    _write_kv(lines, "mode", fo.get("mode", "none") if isinstance(fo, dict) else "none")
    if isinstance(fo, dict):
        for key in ("theta", "alpha", "ax", "ay", "az", "axis_mode"):
            if key in fo:
                _write_kv(lines, key, fo.get(key))
        if "donor_ids" in fo:
            _write_kv(lines, "donor_ids", ",".join(str(v) for v in fo.get("donor_ids") or []))
    lines.append("")

    last = state.get("last_fit_result") or {}
    lines.append("[last_fit_result]")
    _write_kv(lines, "available", bool(last))
    if isinstance(last, dict):
        for key in ("mode", "rmsd", "r2", "q_factor", "n", "theta", "alpha", "delta_chi_ax", "delta_chi_rh"):
            if key in last:
                _write_kv(lines, key, last.get(key))
    lines.append("")

    last_conf = state.get("last_conformer_result") or {}
    lines.append("[conformer_search]")
    _write_kv(lines, "applied", bool(state.get("conformer_applied", False)))
    _write_kv(lines, "constraint_mode", "angular")
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


def save_project_file(path: str | Path, state: dict) -> None:
    path = Path(path)
    if path.suffix.lower() != PCSP_EXTENSION:
        path = path.with_suffix(PCSP_EXTENSION)
    text = build_pcsp_text(state)
    path.write_text(text, encoding="utf-8")
    state["current_project_path"] = str(path)


# ---------------------------------------------------------------------------
# Loader / state applier
# ---------------------------------------------------------------------------

def _atoms_to_state_data(rows: list[AtomRow]) -> tuple[list[tuple[str, float, float, float]], list[int]]:
    return [(r.element, r.x, r.y, r.z) for r in rows], [r.ref_id for r in rows]


def load_project_file(path: str | Path, state: dict) -> None:
    path = Path(path)
    sections = read_sections(path)

    root_kv = parse_key_values(sections.get("__root__", []))
    schema = root_kv.get("schema_version", "")
    if schema and schema != SCHEMA_VERSION:
        # Strict enough to warn, permissive enough for future minor migration.
        print(f"[PCSP] Loading schema_version={schema}; expected {SCHEMA_VERSION}.")

    structure = parse_key_values(sections.get("structure", []))
    symavg = parse_key_values(sections.get("symmetry_averaging", []))
    visibility = parse_key_values(sections.get("visibility", []))
    pcs_model = parse_key_values(sections.get("pcs_model", []))
    rotation = parse_key_values(sections.get("rotation", []))
    fitting = parse_key_values(sections.get("fitting", []))
    fit_override = parse_key_values(sections.get("fit_override", []))
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
    _set_entry(state, "tensor_entry", pcs_model.get("dchi_ax", "-2.0"))
    state["rh_dchi_rh"] = _parse_float(pcs_model.get("dchi_rh", "0.0"), default=0.0)
    _set_var(state, "rh_dchi_rh_var", str(state["rh_dchi_rh"]))
    state["rh_calc_enabled"] = _parse_bool(pcs_model.get("rhombic_enabled", "false"))
    _set_var(state, "rh_calc_enabled_var", state["rh_calc_enabled"])

    _set_entry(state, "pcs_min_entry", pcs_model.get("pcs_min", "-10.0"))
    _set_entry(state, "pcs_max_entry", pcs_model.get("pcs_max", "10.0"))
    _set_entry(state, "pcs_interval_entry", pcs_model.get("pcs_interval", "0.5"))
    _set_var(state, "plot_90_var", _parse_bool(pcs_model.get("plot_90", "false")))

    ax = _parse_float(rotation.get("x", "0.0"))
    ay = _parse_float(rotation.get("y", "0.0"))
    az = _parse_float(rotation.get("z", "0.0"))
    _set_var(state, "angle_x_var", ax)
    _set_var(state, "angle_y_var", ay)
    _set_var(state, "angle_z_var", az)
    _set_entry(state, "angle_x_entry", f"{ax:.1f}")
    _set_entry(state, "angle_y_entry", f"{ay:.1f}")
    _set_entry(state, "angle_z_entry", f"{az:.1f}")

    _set_var(state, "symavg_enabled_var", _parse_bool(symavg.get("enabled", "false")))
    _set_var(state, "symavg_keep_original_var", _parse_bool(symavg.get("keep_original", "false")))

    # Data layers
    state["delta_exp_values"] = parse_delta_rows(sections.get("delta_exp", []))
    state["delta_obs_values"] = parse_delta_rows(sections.get("delta_obs", []))
    state["delta_dia_values"] = parse_delta_rows(sections.get("delta_dia", []))

    # Fitting state
    _set_var(state, "fit_mode_var", fitting.get("mode", "theta_alpha_multi"))
    _set_var(state, "axis_mode_var", fitting.get("axis_mode", "bisector"))
    _set_var(state, "fit_use_visible_var", _parse_bool(fitting.get("fit_visible_as_group", "true"), default=True))
    _set_var(state, "fit_dchi_var", _parse_bool(fitting.get("fit_dchi_ax", "false")))
    _set_var(state, "fit_dchi_rh_var", _parse_bool(fitting.get("fit_dchi_rh", "false")))
    _set_var(state, "fit_global_search_var", _parse_bool(fitting.get("use_global_search", "false")))

    if _parse_bool(fit_override.get("enabled", "false")):
        fo: dict[str, Any] = {"mode": fit_override.get("mode", "none")}
        for key in ("theta", "alpha", "ax", "ay", "az"):
            if key in fit_override:
                fo[key] = _parse_float(fit_override[key])
        if "axis_mode" in fit_override:
            fo["axis_mode"] = fit_override["axis_mode"]
        if "donor_ids" in fit_override:
            fo["donor_ids"] = _parse_csv_ints(fit_override["donor_ids"])
        state["fit_override"] = fo
    else:
        state.pop("fit_override", None)

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

    create_checklist = state.get("create_checklist")
    if callable(create_checklist):
        create_checklist(state)
        selected = set(_parse_csv_strings(visibility.get("selected_elements", "")))
        if selected:
            for el, var in (state.get("check_vars", {}) or {}).items():
                try:
                    var.set(str(el) in selected)
                except Exception:
                    pass

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

    pop_fit = state.get("populate_fitting_controls")
    if callable(pop_fit):
        try:
            pop_fit(state)
        except Exception:
            pass

    state["current_project_path"] = str(path)


# ---------------------------------------------------------------------------
# Dialog wrappers used by the UI menu
# ---------------------------------------------------------------------------

def save_project_dialog(state: dict) -> None:
    filedialog = state.get("filedialog")
    messagebox = state.get("messagebox")
    if filedialog is None:
        raise RuntimeError("filedialog is not available in state.")

    initial = state.get("current_project_path") or "pcs_project.pcsp"
    path = filedialog.asksaveasfilename(
        title="Save PCS project",
        defaultextension=PCSP_EXTENSION,
        initialfile=Path(initial).name,
        filetypes=[("PCS project", "*.pcsp"), ("All files", "*.*")],
    )
    if not path:
        return

    try:
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
        load_project_file(path, state)
        if messagebox is not None:
            messagebox.showinfo("Open PCS project", f"Project loaded:\n{path}")
    except Exception as exc:
        if messagebox is not None:
            messagebox.showerror("Open PCS project", str(exc))
        else:
            raise
