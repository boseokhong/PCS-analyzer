# plugin_system/plugin_loader.py

from __future__ import annotations

import importlib.util
import json
import sys
import traceback as traceback_module
from pathlib import Path

from app_version import APP_VERSION
from plugin_system import plugin_installer
from plugin_system.plugin_manifest import ManifestValidationError, resolve_entry_path
from plugin_system.plugin_registry import (
    INSTALLED_DIR,
    PluginRegistryError,
    ensure_plugin_dirs,
    load_registry,
)
from plugin_system.plugin_version import PluginVersionError, check_app_compatibility


_MISSING = object()


def _load_result(
    plugin_id: str | None,
    *,
    status: str,
    category: str,
    phase: str,
    message: str,
    error: str | None = None,
    exception: BaseException | None = None,
    path: Path | None = None,
    enabled: bool | None = None,
    manifest: dict | None = None,
    module=None,
    warnings: list[dict] | None = None,
    reused_module: bool = False,
    **extra,
) -> dict:
    """Build the stable, UI-independent plugin load result shape."""
    result = {
        "ok": status != "failed",
        "plugin_id": plugin_id,
        "status": status,
        "category": category,
        "phase": phase,
        "message": message,
        "error": error,
        "exception_type": type(exception).__name__ if exception else None,
        "traceback": (
            "".join(
                traceback_module.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )
            if exception
            else None
        ),
        "path": str(path) if path is not None else None,
        "enabled": enabled,
        "manifest": manifest,
        "module": module,
        "warnings": list(warnings or ()),
        "reused_module": reused_module,
    }
    result.update(extra)
    return result


def _read_manifest(plugin_dir: Path) -> dict:
    manifest_path = plugin_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object.")
    return manifest


def load_manifest(plugin_dir: Path) -> dict | None:
    """Return a legacy manifest result while preserving the historic public API."""
    try:
        return _read_manifest(plugin_dir)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None


def _compatibility_warning(field: str, value: object, detail: str) -> dict:
    return {
        "category": "invalid_compatibility_metadata",
        "field": field,
        "value": value,
        "message": f"Ignoring invalid {field} value {value!r}: {detail}",
    }


def _check_legacy_compatibility(manifest: dict) -> tuple[object | None, list[dict]]:
    """Check valid compatibility bounds without rejecting malformed legacy data."""
    warnings = []
    valid_bounds = {}

    for field in ("min_app_version", "max_app_version"):
        if field not in manifest:
            continue
        value = manifest[field]
        try:
            if field == "min_app_version":
                check_app_compatibility(APP_VERSION, min_app_version=value)
            else:
                check_app_compatibility(APP_VERSION, max_app_version=value)
        except PluginVersionError as exc:
            warnings.append(_compatibility_warning(field, value, str(exc)))
        else:
            valid_bounds[field] = value

    try:
        compatibility = check_app_compatibility(
            APP_VERSION,
            min_app_version=valid_bounds.get("min_app_version"),
            max_app_version=valid_bounds.get("max_app_version"),
        )
    except PluginVersionError as exc:
        warnings.append(
            _compatibility_warning(
                "min_app_version/max_app_version",
                {
                    "min_app_version": valid_bounds.get("min_app_version"),
                    "max_app_version": valid_bounds.get("max_app_version"),
                },
                str(exc),
            )
        )
        return None, warnings

    return compatibility, warnings


def _canonical_module_path(module) -> Path | None:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None
    try:
        return Path(module_file).resolve(strict=False)
    except (OSError, RuntimeError):
        return None


def _restore_module(sys_name: str, previous) -> None:
    if previous is _MISSING:
        sys.modules.pop(sys_name, None)
    else:
        sys.modules[sys_name] = previous


def _remove_new_alias_modules(prefixes: tuple[str, ...], previous_names: set[str]) -> None:
    for name in tuple(sys.modules):
        if name in previous_names:
            continue
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            sys.modules.pop(name, None)


def load_plugin(plugin_id: str, record: dict, app) -> dict:
    enabled = bool(record.get("enabled", False)) if isinstance(record, dict) else None
    warnings = []

    try:
        directory = record.get("dir", "")
        plugin_dir = Path(directory)
    except (AttributeError, TypeError, ValueError) as exc:
        return _load_result(
            plugin_id,
            status="failed",
            category="path_error",
            phase="locate",
            message="The installed plugin path is invalid.",
            error=str(exc),
            exception=exc,
            enabled=enabled,
        )

    # Preserve the original loader's interpretation of legacy relative paths.
    if not plugin_dir.is_absolute():
        plugin_dir = INSTALLED_DIR / plugin_id
    plugin_dir = plugin_dir.resolve(strict=False)

    if not plugin_dir.is_dir():
        return _load_result(
            plugin_id,
            status="failed",
            category="path_error",
            phase="locate",
            message="The installed plugin directory is missing or invalid.",
            error=f"Plugin directory is not a directory: {plugin_dir}",
            path=plugin_dir,
            enabled=enabled,
        )

    manifest_path = plugin_dir / "manifest.json"
    if not manifest_path.exists():
        return _load_result(
            plugin_id,
            status="failed",
            category="missing_manifest",
            phase="manifest",
            message="The plugin manifest is missing.",
            error=f"Manifest does not exist: {manifest_path}",
            path=manifest_path,
            enabled=enabled,
        )

    try:
        manifest = _read_manifest(plugin_dir)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return _load_result(
            plugin_id,
            status="failed",
            category="malformed_manifest",
            phase="manifest",
            message="The plugin manifest could not be read.",
            error=str(exc),
            exception=exc,
            path=manifest_path,
            enabled=enabled,
        )

    manifest_id = manifest.get("id")
    if manifest_id is not None and manifest_id != plugin_id:
        warnings.append(
            {
                "category": "legacy_metadata",
                "field": "id",
                "value": manifest_id,
                "message": (
                    f"Manifest id {manifest_id!r} differs from registry id {plugin_id!r}."
                ),
            }
        )

    entry = manifest.get("entry", "plugin.py")
    try:
        entry_path = resolve_entry_path(plugin_dir, entry)
    except ManifestValidationError as exc:
        return _load_result(
            plugin_id,
            status="failed",
            category="path_error",
            phase="entry",
            message="The plugin entry path is unsafe or invalid.",
            error=str(exc),
            exception=exc,
            path=plugin_dir,
            enabled=enabled,
            manifest=manifest,
            warnings=warnings,
        )

    if not entry_path.is_file():
        return _load_result(
            plugin_id,
            status="failed",
            category="missing_entry",
            phase="entry",
            message="The plugin entry file is missing.",
            error=f"Plugin entry file does not exist: {entry_path}",
            path=entry_path,
            enabled=enabled,
            manifest=manifest,
            warnings=warnings,
        )

    compatibility, compatibility_warnings = _check_legacy_compatibility(manifest)
    warnings.extend(compatibility_warnings)
    if compatibility is not None and not compatibility.compatible:
        reason = compatibility.reason or "Plugin is incompatible with this application."
        return _load_result(
            plugin_id,
            status="failed",
            category="incompatible",
            phase="compatibility",
            message="The plugin is not compatible with this PCS Analyzer version.",
            error=reason,
            path=entry_path,
            enabled=enabled,
            manifest=manifest,
            warnings=warnings,
        )

    module_name = f"pcs_plugin_{plugin_id}"
    public_name = plugin_id
    entry_identity = entry_path.resolve(strict=False)
    existing_module = sys.modules.get(module_name)
    reused_module = False

    if existing_module is not None:
        if _canonical_module_path(existing_module) != entry_identity:
            return _load_result(
                plugin_id,
                status="failed",
                category="import_error",
                phase="import",
                message="A conflicting plugin module is already loaded.",
                error=(
                    f"Module {module_name!r} is loaded from "
                    f"{_canonical_module_path(existing_module)!s}, not {entry_identity}."
                ),
                path=entry_path,
                enabled=enabled,
                manifest=manifest,
                warnings=warnings,
            )
        module = existing_module
        reused_module = True
    else:
        module = None

    original_sys_path = list(sys.path)
    previous_names = set(sys.modules)
    old_module = sys.modules.get(module_name, _MISSING)
    old_public_module = sys.modules.get(public_name, _MISSING)
    import_completed = reused_module

    try:
        sys.path.insert(0, str(plugin_dir))

        if not reused_module:
            spec = importlib.util.spec_from_file_location(
                module_name,
                entry_path,
                submodule_search_locations=[str(plugin_dir)],
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Could not create plugin loader.")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            sys.modules[public_name] = module
            try:
                spec.loader.exec_module(module)
            except ModuleNotFoundError as exc:
                _restore_module(module_name, old_module)
                _restore_module(public_name, old_public_module)
                _remove_new_alias_modules((module_name, public_name), previous_names)
                return _load_result(
                    plugin_id,
                    status="failed",
                    category="dependency_error",
                    phase="import",
                    message="A plugin dependency could not be imported.",
                    error=str(exc),
                    exception=exc,
                    path=entry_path,
                    enabled=enabled,
                    manifest=manifest,
                    warnings=warnings,
                    missing_module=exc.name,
                )
            except Exception as exc:
                _restore_module(module_name, old_module)
                _restore_module(public_name, old_public_module)
                _remove_new_alias_modules((module_name, public_name), previous_names)
                return _load_result(
                    plugin_id,
                    status="failed",
                    category="import_error",
                    phase="import",
                    message="The plugin raised an exception while importing.",
                    error=str(exc),
                    exception=exc,
                    path=entry_path,
                    enabled=enabled,
                    manifest=manifest,
                    warnings=warnings,
                )
            import_completed = True
        else:
            # This alias supports existing plugins that import their own ID.
            sys.modules[public_name] = module

        register = getattr(module, "register", None)
        if not callable(register):
            if not reused_module:
                _restore_module(module_name, old_module)
                _restore_module(public_name, old_public_module)
                _remove_new_alias_modules((module_name, public_name), previous_names)
            return _load_result(
                plugin_id,
                status="failed",
                category="missing_register",
                phase="register",
                message="The plugin does not define register(app).",
                error="register(app) function is missing or not callable.",
                path=entry_path,
                enabled=enabled,
                manifest=manifest,
                module=module if reused_module else None,
                warnings=warnings,
                reused_module=reused_module,
            )

        previous_plugin_id = getattr(app, "current_plugin_id", None)
        previous_manifest = getattr(app, "current_plugin_manifest", None)

        try:
            app.current_plugin_id = plugin_id
            app.current_plugin_manifest = manifest
            register(app)
        except Exception as exc:
            return _load_result(
                plugin_id,
                status="failed",
                category="register_error",
                phase="register",
                message="The plugin raised an exception during registration.",
                error=str(exc),
                exception=exc,
                path=entry_path,
                enabled=enabled,
                manifest=manifest,
                module=module,
                warnings=warnings,
                reused_module=reused_module,
            )
        finally:
            app.current_plugin_id = previous_plugin_id
            app.current_plugin_manifest = previous_manifest

        return _load_result(
            plugin_id,
            status="loaded",
            category="success",
            phase="register",
            message="Plugin loaded successfully.",
            path=entry_path,
            enabled=enabled,
            manifest=manifest,
            module=module,
            warnings=warnings,
            reused_module=reused_module,
        )
    except Exception as exc:
        if not import_completed:
            _restore_module(module_name, old_module)
            _restore_module(public_name, old_public_module)
            _remove_new_alias_modules((module_name, public_name), previous_names)
        return _load_result(
            plugin_id,
            status="failed",
            category="import_error",
            phase="import",
            message="The plugin could not be imported.",
            error=str(exc),
            exception=exc,
            path=entry_path,
            enabled=enabled,
            manifest=manifest,
            module=module if import_completed else None,
            warnings=warnings,
            reused_module=reused_module,
        )
    finally:
        sys.path[:] = original_sys_path


def load_enabled_plugins(app) -> list[dict]:
    try:
        ensure_plugin_dirs()
        registry = load_registry()
    except PluginRegistryError as exc:
        return [
            _load_result(
                None,
                status="failed",
                category="registry_error",
                phase="registry",
                message="The plugin registry could not be loaded.",
                error=str(exc),
                exception=exc,
            )
        ]

    results = []
    for plugin_id, record in registry.items():
        if not record.get("enabled", False):
            results.append(
                _load_result(
                    plugin_id,
                    status="skipped",
                    category="disabled",
                    phase="selection",
                    message="Plugin is disabled.",
                    enabled=False,
                )
            )
            continue

        try:
            results.append(load_plugin(plugin_id, record, app))
        except Exception as exc:
            # Keep an unforeseen loader defect in one plugin from blocking others.
            results.append(
                _load_result(
                    plugin_id,
                    status="failed",
                    category="load_error",
                    phase="load",
                    message="The plugin could not be loaded.",
                    error=str(exc),
                    exception=exc,
                    enabled=True,
                )
            )

    return results


def inspect_plugin_file(py_path: str | Path) -> dict:
    return plugin_installer.inspect_plugin_file(py_path)


def install_plugin_file(py_path: str | Path, **kwargs) -> dict:
    return plugin_installer.install_plugin_file(py_path, **kwargs)


def install_plugin_folder(folder_path: str | Path, **kwargs) -> dict:
    return plugin_installer.install_plugin_folder(folder_path, **kwargs)


def install_plugin_zip(zip_path: str | Path, **kwargs) -> dict:
    return plugin_installer.install_plugin_zip(zip_path, **kwargs)
