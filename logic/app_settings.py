# logic/app_settings.py

from __future__ import annotations

import re
import json
from copy import deepcopy
from pathlib import Path


DEFAULT_MAIN_GEOMETRY = "1200x880"
MIN_MAIN_WIDTH = 900
MIN_MAIN_HEIGHT = 650


DEFAULT_APP_SETTINGS = {
    "theme_variant": "light",
    "theme_accent": "green",

    "open_2d_plot_on_start": True,
    "auto_open_3d_on_load": True,
    "remember_window_geometry": True,

    "remember_recent_files": True,
    "max_recent_files": 8,

    "default_dchi_ax": -2.0,
    "default_pcs_min": -10.0,
    "default_pcs_max": 10.0,
    "default_pcs_interval": 0.5,

    "export_default_dpi": 600,
}


def get_settings_path() -> Path:
    """
    Return the JSON path used to store application settings.
    Saved next to the project entry for now.
    """
    return Path(__file__).resolve().parent.parent / "app_settings.json"


def build_default_settings() -> dict:
    return deepcopy(DEFAULT_APP_SETTINGS)


def parse_tk_geometry(geometry: str):
    """
    Parse a Tk geometry string.

    Accepted examples:
        1200x880
        1200x880+100+100
        1200x880-10+50

    Returns:
        (width, height, x, y) or None
    """
    if not geometry:
        return None

    m = re.match(r"^\s*(\d+)x(\d+)([+-]\d+)?([+-]\d+)?\s*$", str(geometry))
    if not m:
        return None

    width = int(m.group(1))
    height = int(m.group(2))
    x = int(m.group(3)) if m.group(3) else None
    y = int(m.group(4)) if m.group(4) else None

    return width, height, x, y


def is_reasonable_main_geometry(geometry: str) -> bool:
    """
    Return True only for geometry values that are safe to reuse.

    This prevents accidentally saving/restoring tiny, half-initialized,
    minimized, or clearly invalid window sizes.
    """
    parsed = parse_tk_geometry(geometry)
    if parsed is None:
        return False

    width, height, x, y = parsed

    if width < MIN_MAIN_WIDTH:
        return False

    if height < MIN_MAIN_HEIGHT:
        return False

    # Keep this permissive for multi-monitor setups.
    # This only rejects obviously broken coordinates.
    if x is not None and abs(x) > 10000:
        return False

    if y is not None and abs(y) > 10000:
        return False

    return True


def get_safe_main_geometry(root, previous_geometry: str | None = None) -> str:
    """
    Return a geometry string safe to save.

    If the current geometry is invalid, too small, minimized, or unavailable,
    keep the previous valid geometry instead of overwriting it.
    """
    if root is None:
        if previous_geometry and is_reasonable_main_geometry(previous_geometry):
            return previous_geometry
        return DEFAULT_MAIN_GEOMETRY

    try:
        win_state = root.state()
        if win_state not in ("normal", "zoomed"):
            if previous_geometry and is_reasonable_main_geometry(previous_geometry):
                return previous_geometry
            return DEFAULT_MAIN_GEOMETRY
    except Exception:
        if previous_geometry and is_reasonable_main_geometry(previous_geometry):
            return previous_geometry
        return DEFAULT_MAIN_GEOMETRY

    try:
        root.update_idletasks()
    except Exception:
        pass

    try:
        geometry = root.geometry()
    except Exception:
        if previous_geometry and is_reasonable_main_geometry(previous_geometry):
            return previous_geometry
        return DEFAULT_MAIN_GEOMETRY

    if is_reasonable_main_geometry(geometry):
        return geometry

    if previous_geometry and is_reasonable_main_geometry(previous_geometry):
        return previous_geometry

    return DEFAULT_MAIN_GEOMETRY


def _coerce_settings(raw: dict) -> dict:
    """
    Merge external JSON data with defaults and coerce obvious types safely.
    Invalid values fall back to defaults.
    """
    cfg = build_default_settings()

    if not isinstance(raw, dict):
        return cfg

    # strings
    theme_variant = str(raw.get("theme_variant", cfg["theme_variant"])).strip().lower()
    if theme_variant in ("light", "dark"):
        cfg["theme_variant"] = theme_variant

    theme_accent = str(raw.get("theme_accent", cfg["theme_accent"])).strip().lower()
    if theme_accent in ("blue", "green", "orange", "purple"):
        cfg["theme_accent"] = theme_accent

    # bools
    for key in (
        "open_2d_plot_on_start",
        "auto_open_3d_on_load",
        "remember_window_geometry",
        "remember_recent_files",
    ):
        if key in raw:
            cfg[key] = bool(raw[key])

    # ints
    try:
        v = int(raw.get("max_recent_files", cfg["max_recent_files"]))
        if v >= 1:
            cfg["max_recent_files"] = v
    except Exception:
        pass

    try:
        v = int(raw.get("export_default_dpi", cfg["export_default_dpi"]))
        if v > 0:
            cfg["export_default_dpi"] = v
    except Exception:
        pass

    # floats
    try:
        cfg["default_dchi_ax"] = float(raw.get("default_dchi_ax", cfg["default_dchi_ax"]))
    except Exception:
        pass

    try:
        cfg["default_pcs_min"] = float(raw.get("default_pcs_min", cfg["default_pcs_min"]))
    except Exception:
        pass

    try:
        cfg["default_pcs_max"] = float(raw.get("default_pcs_max", cfg["default_pcs_max"]))
    except Exception:
        pass

    try:
        cfg["default_pcs_interval"] = float(raw.get("default_pcs_interval", cfg["default_pcs_interval"]))
    except Exception:
        pass

    # logical sanity
    if cfg["default_pcs_min"] >= cfg["default_pcs_max"]:
        cfg["default_pcs_min"] = DEFAULT_APP_SETTINGS["default_pcs_min"]
        cfg["default_pcs_max"] = DEFAULT_APP_SETTINGS["default_pcs_max"]

    if cfg["default_pcs_interval"] <= 0:
        cfg["default_pcs_interval"] = DEFAULT_APP_SETTINGS["default_pcs_interval"]

    return cfg


def load_app_state() -> dict:
    """
    Load the persisted application state.

    Returns:
        {
            "app_settings": {...},
            "recent_files": [...],
            "main_window_geometry": "1200x880+100+100" | None,
        }
    """
    path = get_settings_path()

    if not path.exists():
        return {
            "app_settings": build_default_settings(),
            "recent_files": [],
            "main_window_geometry": None,
        }

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "app_settings": build_default_settings(),
            "recent_files": [],
            "main_window_geometry": None,
        }

    if not isinstance(data, dict):
        data = {}

    app_settings = _coerce_settings(data.get("app_settings", {}))

    recent_files = data.get("recent_files", [])
    if not isinstance(recent_files, list):
        recent_files = []
    recent_files = [str(p) for p in recent_files if str(p).strip()]

    main_window_geometry = data.get("main_window_geometry")
    if main_window_geometry is not None:
        main_window_geometry = str(main_window_geometry)

    # Do not restore invalid or tiny geometry.
    if main_window_geometry and not is_reasonable_main_geometry(main_window_geometry):
        main_window_geometry = None

    return {
        "app_settings": app_settings,
        "recent_files": recent_files,
        "main_window_geometry": main_window_geometry,
    }


def save_app_state(state: dict, *, save_geometry: bool = True) -> None:
    """
    Persist app settings and lightweight UI state to JSON.

    Parameters
    ----------
    state : dict
        Main PCS Analyzer state.
    save_geometry : bool
        If True, save the main window geometry only if it is valid.
        If False, preserve the previous geometry from app_settings.json.

    Notes
    -----
    Use save_geometry=False for frequent saves such as recent-file updates.
    Use save_geometry=True on normal application close.
    """
    path = get_settings_path()

    old_state = load_app_state()
    previous_geometry = old_state.get("main_window_geometry")

    app_settings = _coerce_settings(state.get("app_settings", {}) or {})
    recent_files = state.get("recent_files", []) or []

    geometry = previous_geometry

    if save_geometry and app_settings.get("remember_window_geometry", True):
        geometry = get_safe_main_geometry(
            state.get("root"),
            previous_geometry=previous_geometry,
        )
    elif previous_geometry and is_reasonable_main_geometry(previous_geometry):
        geometry = previous_geometry
    else:
        geometry = None

    payload = {
        "app_settings": app_settings,
        "recent_files": [str(p) for p in recent_files if str(p).strip()],
        "main_window_geometry": geometry,
    }

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )