# plugin_system/plugin_registry.py

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Mapping
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


class PluginRegistryError(Exception):
    """Base class for local plugin registry errors."""


class RegistryReadError(PluginRegistryError):
    """Raised when the registry cannot be read."""


class RegistryCorruptError(RegistryReadError):
    """Raised when the registry does not contain valid JSON."""


class RegistryValidationError(PluginRegistryError):
    """Raised when registry JSON has an invalid structure."""


class RegistryWriteError(PluginRegistryError):
    """Raised when the registry cannot be written atomically."""


_OPTIONAL_STRING_DEFAULTS = {
    "name": None,
    "version": "",
    "author": "",
    "description": "",
    "dir": None,
}
_OPTIONAL_STRING_FIELDS = (*_OPTIONAL_STRING_DEFAULTS, "type")


def _ensure_plugin_directories() -> None:
    try:
        PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
        INSTALLED_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RegistryWriteError(
            f"Could not create plugin directories under {PLUGIN_DIR}: {exc}"
        ) from exc


def validate_registry(data: object) -> dict:
    """Validate and normalize the existing dictionary-based registry format."""
    if not isinstance(data, Mapping):
        raise RegistryValidationError("Plugin registry must be a JSON object.")

    normalized = {}
    for plugin_id, raw_record in data.items():
        if not isinstance(plugin_id, str) or not plugin_id.strip():
            raise RegistryValidationError(
                "Plugin registry keys must be non-empty strings."
            )
        if not isinstance(raw_record, Mapping):
            raise RegistryValidationError(
                f"Registry record for {plugin_id!r} must be a JSON object."
            )

        record = dict(raw_record)
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise RegistryValidationError(
                f"Registry record for {plugin_id!r} requires a non-empty string id."
            )
        if record_id != plugin_id:
            raise RegistryValidationError(
                f"Registry record id {record_id!r} does not match key {plugin_id!r}."
            )

        for field_name in _OPTIONAL_STRING_FIELDS:
            if field_name in record and not isinstance(record[field_name], str):
                raise RegistryValidationError(
                    f"Registry field {field_name!r} for {plugin_id!r} must be a string."
                )

        if "dir" in record and not record["dir"].strip():
            raise RegistryValidationError(
                f"Registry field 'dir' for {plugin_id!r} must not be empty."
            )

        if "enabled" in record and not isinstance(record["enabled"], bool):
            raise RegistryValidationError(
                f"Registry field 'enabled' for {plugin_id!r} must be boolean."
            )

        for field_name, default in _OPTIONAL_STRING_DEFAULTS.items():
            if field_name not in record:
                record[field_name] = plugin_id if default is None else default
        record.setdefault("enabled", False)

        normalized[plugin_id] = record

    return normalized


def get_installed_plugin_dir(
    plugin_id: str,
    record: Mapping | None = None,
) -> Path:
    """Interpret legacy absolute or installed-root-relative plugin paths."""
    directory = plugin_id if record is None else record.get("dir", plugin_id)
    path = Path(directory)
    if path.is_absolute():
        return path
    return INSTALLED_DIR / path


def _write_registry_atomic(registry: dict) -> None:
    _ensure_plugin_directories()
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=REGISTRY_PATH.parent,
            prefix=f".{REGISTRY_PATH.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(registry, temp_file, indent=2, ensure_ascii=False)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, REGISTRY_PATH)
    except Exception as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RegistryWriteError(
            f"Could not write plugin registry {REGISTRY_PATH}: {exc}"
        ) from exc


def ensure_plugin_dirs() -> None:
    _ensure_plugin_directories()

    if not REGISTRY_PATH.exists():
        _write_registry_atomic({})


def load_registry() -> dict:
    ensure_plugin_dirs()

    try:
        with REGISTRY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise RegistryCorruptError(
            f"Plugin registry {REGISTRY_PATH} contains invalid JSON: {exc}"
        ) from exc
    except OSError as exc:
        raise RegistryReadError(
            f"Could not read plugin registry {REGISTRY_PATH}: {exc}"
        ) from exc

    return validate_registry(data)


def save_registry(registry: dict) -> None:
    normalized = validate_registry(registry)
    _write_registry_atomic(normalized)


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
