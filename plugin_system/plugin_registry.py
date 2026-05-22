# plugin_system/plugin_registry.py

from __future__ import annotations

import json
import sys
from pathlib import Path


def get_app_base_dir() -> Path:
    """
    Source mode:
        project root

    PyInstaller onedir:
        folder containing PCS Analyzer.exe
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent.parent


BASE_DIR = get_app_base_dir()

PLUGIN_DIR = BASE_DIR / "plugins"
INSTALLED_DIR = PLUGIN_DIR / "installed"
REGISTRY_PATH = PLUGIN_DIR / "plugins.json"


def ensure_plugin_dirs() -> None:
    PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
    INSTALLED_DIR.mkdir(parents=True, exist_ok=True)

    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.write_text("{}", encoding="utf-8")


def load_registry() -> dict:
    ensure_plugin_dirs()

    try:
        with REGISTRY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_registry(registry: dict) -> None:
    ensure_plugin_dirs()

    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def set_plugin_enabled(plugin_id: str, enabled: bool) -> None:
    registry = load_registry()
    if plugin_id in registry:
        registry[plugin_id]["enabled"] = bool(enabled)
        save_registry(registry)


def remove_plugin_from_registry(plugin_id: str) -> None:
    registry = load_registry()
    if plugin_id in registry:
        registry.pop(plugin_id)
        save_registry(registry)