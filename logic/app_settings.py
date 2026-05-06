# logic/app_settings.py

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


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
            "main_window_geometry": "1190x900+100+100" | None,
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

    app_settings = _coerce_settings(data.get("app_settings", {}))

    recent_files = data.get("recent_files", [])
    if not isinstance(recent_files, list):
        recent_files = []
    recent_files = [str(p) for p in recent_files if str(p).strip()]

    main_window_geometry = data.get("main_window_geometry")
    if main_window_geometry is not None:
        main_window_geometry = str(main_window_geometry)

    return {
        "app_settings": app_settings,
        "recent_files": recent_files,
        "main_window_geometry": main_window_geometry,
    }


def save_app_state(state: dict) -> None:
    """
    Persist app settings and lightweight UI state to JSON.
    """
    path = get_settings_path()

    app_settings = state.get("app_settings", {}) or {}
    recent_files = state.get("recent_files", []) or []

    geometry = None
    try:
        if app_settings.get("remember_window_geometry", True):
            geometry = state["root"].geometry()
    except Exception:
        geometry = None

    payload = {
        "app_settings": _coerce_settings(app_settings),
        "recent_files": [str(p) for p in recent_files if str(p).strip()],
        "main_window_geometry": geometry,
    }

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )