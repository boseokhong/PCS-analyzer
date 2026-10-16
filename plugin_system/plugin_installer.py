from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import traceback
import uuid
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path

from app_version import APP_VERSION
from plugin_system import plugin_registry
from plugin_system.plugin_manifest import (
    ManifestValidationError,
    PLUGIN_ID_PATTERN,
    check_manifest_compatibility,
    resolve_entry_path,
    validate_manifest,
)
from plugin_system.plugin_version import PluginVersionError, parse_version


MAX_ZIP_ENTRIES = 5_000
MAX_ZIP_TOTAL_SIZE = 512 * 1024 * 1024
MAX_ZIP_FILE_SIZE = 256 * 1024 * 1024


class PluginInstallerError(Exception):
    def __init__(
        self,
        message: str,
        *,
        restart_required: bool = False,
        recovery_paths: tuple[Path, ...] = (),
    ) -> None:
        super().__init__(message)
        self.restart_required = restart_required
        self.recovery_paths = recovery_paths


class PackageValidationError(PluginInstallerError):
    """Raised when a candidate plugin package is invalid or unsafe."""


class PackageTransactionError(PluginInstallerError):
    """Raised when an install/update transaction cannot be completed."""


class PluginRemovalError(PluginInstallerError):
    """Raised when a plugin removal transaction cannot be completed."""


def _error_result(exc: Exception) -> dict:
    if isinstance(exc, PluginInstallerError):
        restart_required = exc.restart_required
        recovery_paths = [str(path) for path in exc.recovery_paths]
    else:
        restart_required = False
        recovery_paths = []

    return {
        "ok": False,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "restart_required": restart_required,
        "recovery_paths": recovery_paths,
    }


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _module_belongs_to(module: object, plugin_dir: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        return _path_is_within(Path(module_file).resolve(), plugin_dir.resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _inspect_plugin_file_or_raise(py_path: Path) -> dict:
    if not py_path.is_file():
        raise PackageValidationError(f"File does not exist: {py_path}")

    module_name = f"_pcs_plugin_inspect_{uuid.uuid4().hex}"
    plugin_dir = py_path.parent.resolve()
    original_sys_path = list(sys.path)
    modules_before = set(sys.modules)

    try:
        sys.path.insert(0, str(plugin_dir))
        spec = importlib.util.spec_from_file_location(
            module_name,
            py_path,
            submodule_search_locations=[str(plugin_dir)],
        )
        if spec is None or spec.loader is None:
            raise PackageValidationError("Could not create plugin inspection loader.")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        info = getattr(module, "PLUGIN_INFO", None)
        if not isinstance(info, dict):
            raise PackageValidationError("PLUGIN_INFO dictionary is missing.")
        if not callable(getattr(module, "register", None)):
            raise PackageValidationError("register(app) function is missing.")
        return dict(info)
    except PluginInstallerError:
        raise
    except Exception as exc:
        raise PackageValidationError(
            "Plugin inspection failed:\n" + traceback.format_exc()
        ) from exc
    finally:
        sys.path[:] = original_sys_path
        for name in tuple(sys.modules):
            if name == module_name or (
                name not in modules_before
                and _module_belongs_to(sys.modules.get(name), plugin_dir)
            ):
                sys.modules.pop(name, None)


def _manifest_from_plugin_info(info: Mapping, source_name: str) -> dict:
    plugin_id = info.get("id") or info.get("name") or Path(source_name).stem
    manifest = {
        "id": plugin_id,
        "name": info.get("name", plugin_id),
        "version": info.get("version", ""),
        "author": info.get("author", ""),
        "description": info.get("description", ""),
        "entry": "plugin.py",
        "standalone": bool(info.get("standalone", True)),
        "dependencies": info.get("dependencies", []),
    }
    for field_name in (
        "schema_version",
        "type",
        "optional_dependencies",
        "min_app_version",
        "max_app_version",
        "license",
        "category",
    ):
        if field_name in info:
            manifest[field_name] = info[field_name]
    return manifest


def inspect_plugin_file(py_path: str | Path) -> dict:
    """Execute a single-file plugin to inspect its current PLUGIN_INFO format."""
    path = Path(py_path)
    try:
        info = _inspect_plugin_file_or_raise(path)
        manifest = _manifest_from_plugin_info(info, path.name)
        validate_manifest(manifest)
        return {
            "ok": True,
            "plugin_id": manifest["id"],
            "info": info,
        }
    except Exception as exc:
        return _error_result(exc)


def _load_and_validate_manifest(
    package_root: Path,
    *,
    expected_plugin_id: object | None,
    expected_version: object | None,
) -> dict:
    manifest_path = package_root / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            raw_manifest = json.load(manifest_file)
    except FileNotFoundError as exc:
        raise PackageValidationError(
            f"manifest.json was not found in {package_root}."
        ) from exc
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise PackageValidationError(
            f"manifest.json in {package_root} is invalid: {exc}"
        ) from exc
    except OSError as exc:
        raise PackageValidationError(
            f"Could not read manifest.json in {package_root}: {exc}"
        ) from exc

    try:
        manifest = validate_manifest(raw_manifest, plugin_dir=package_root)
    except ManifestValidationError as exc:
        raise PackageValidationError(f"Invalid plugin manifest: {exc}") from exc

    compatibility = check_manifest_compatibility(manifest, APP_VERSION)
    if not compatibility.compatible:
        raise PackageValidationError(
            compatibility.reason or "Plugin is not compatible with this application version."
        )

    entry_path = resolve_entry_path(package_root, manifest["entry"])
    if not entry_path.is_file() or entry_path.is_symlink():
        raise PackageValidationError(
            f"Plugin entry file does not exist or is unsafe: {manifest['entry']}"
        )

    if expected_plugin_id is not None:
        if (
            not isinstance(expected_plugin_id, str)
            or PLUGIN_ID_PATTERN.fullmatch(expected_plugin_id) is None
        ):
            raise PackageValidationError(
                "expected_plugin_id must match ^[a-z][a-z0-9_]*$."
            )
        if manifest["id"] != expected_plugin_id:
            raise PackageValidationError(
                f"Package id {manifest['id']!r} does not match expected id "
                f"{expected_plugin_id!r}."
            )

    if expected_version is not None:
        try:
            expected = parse_version(expected_version, field_name="expected_version")
            actual = parse_version(manifest["version"], field_name="version")
        except PluginVersionError as exc:
            raise PackageValidationError(str(exc)) from exc
        if actual != expected:
            raise PackageValidationError(
                f"Package version {manifest['version']!r} does not match expected "
                f"version {str(expected_version)!r}."
            )

    return manifest


def _find_folder_package_root(folder: Path) -> Path:
    if not folder.is_dir():
        raise PackageValidationError(f"Plugin folder does not exist: {folder}")
    if folder.is_symlink():
        raise PackageValidationError("Plugin folder must not be a symbolic link.")
    if (folder / "manifest.json").is_file():
        return folder

    entries = list(folder.iterdir())
    candidates = [
        entry
        for entry in entries
        if entry.is_dir() and not entry.is_symlink() and (entry / "manifest.json").is_file()
    ]
    if len(entries) == 1 and len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise PackageValidationError(
            "Plugin folder layout is ambiguous; expected one package root."
        )
    raise PackageValidationError("manifest.json was not found in the selected folder.")


def _reject_folder_symlinks(folder: Path) -> None:
    for path in folder.rglob("*"):
        if path.is_symlink():
            raise PackageValidationError(
                f"Plugin folder contains a symbolic link: {path.relative_to(folder)}"
            )


def _safe_zip_parts(name: str) -> tuple[str, ...]:
    if "\x00" in name:
        raise PackageValidationError("ZIP member names must not contain NUL characters.")
    portable_name = name.replace("\\", "/")
    if portable_name.startswith("/") or re.match(r"^[A-Za-z]:", portable_name):
        raise PackageValidationError(f"ZIP member uses an absolute path: {name!r}")

    parts = tuple(
        part for part in portable_name.split("/") if part not in ("", ".")
    )
    if ".." in parts:
        raise PackageValidationError(
            f"ZIP member contains parent-directory traversal: {name!r}"
        )
    if any(":" in part for part in parts):
        raise PackageValidationError(f"ZIP member contains an unsafe colon: {name!r}")
    return parts


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    unix_mode = (info.external_attr >> 16) & 0o177777
    file_type = stat.S_IFMT(unix_mode)
    if file_type == stat.S_IFLNK:
        raise PackageValidationError(f"ZIP symbolic links are not allowed: {info.filename}")
    if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise PackageValidationError(
            f"ZIP member is not a regular file or directory: {info.filename}"
        )
    if info.flag_bits & 0x1:
        raise PackageValidationError(
            f"Encrypted ZIP members are not supported: {info.filename}"
        )
    if info.file_size > MAX_ZIP_FILE_SIZE:
        raise PackageValidationError(
            f"ZIP member exceeds the {MAX_ZIP_FILE_SIZE}-byte file limit: "
            f"{info.filename}"
        )


def _extract_zip_safely(zip_path: Path, extraction_root: Path) -> None:
    try:
        archive = zipfile.ZipFile(zip_path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise PackageValidationError(f"Could not open plugin ZIP: {exc}") from exc

    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            raise PackageValidationError(
                f"ZIP contains {len(infos)} entries; limit is {MAX_ZIP_ENTRIES}."
            )

        declared_total = 0
        planned = []
        normalized_names = set()
        root = extraction_root.resolve()

        for info in infos:
            _validate_zip_info(info)
            parts = _safe_zip_parts(info.filename)
            if not parts:
                continue
            normalized_name = "/".join(parts).casefold()
            if normalized_name in normalized_names:
                raise PackageValidationError(
                    f"ZIP contains duplicate member path: {info.filename!r}"
                )
            normalized_names.add(normalized_name)

            destination = root.joinpath(*parts).resolve(strict=False)
            if not _path_is_within(destination, root):
                raise PackageValidationError(
                    f"ZIP member escapes the staging directory: {info.filename!r}"
                )

            declared_total += info.file_size
            if declared_total > MAX_ZIP_TOTAL_SIZE:
                raise PackageValidationError(
                    f"ZIP exceeds the {MAX_ZIP_TOTAL_SIZE}-byte total size limit."
                )
            planned.append((info, destination))

        streamed_total = 0
        for info, destination in planned:
            if info.is_dir() or info.filename.endswith(("/", "\\")):
                destination.mkdir(parents=True, exist_ok=True)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            try:
                with archive.open(info, "r") as source, destination.open("xb") as target:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        streamed_total += len(chunk)
                        if written > MAX_ZIP_FILE_SIZE:
                            raise PackageValidationError(
                                f"Extracted ZIP member exceeds the file limit: {info.filename}"
                            )
                        if streamed_total > MAX_ZIP_TOTAL_SIZE:
                            raise PackageValidationError(
                                "Extracted ZIP data exceeds the total size limit."
                            )
                        target.write(chunk)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise PackageValidationError(
                    f"Could not extract ZIP member {info.filename!r}: {exc}"
                ) from exc

            if written != info.file_size:
                raise PackageValidationError(
                    f"Extracted size differs from ZIP metadata for {info.filename!r}."
                )


def _find_extracted_package_root(extraction_root: Path) -> Path:
    root_manifest = extraction_root / "manifest.json"
    entries = list(extraction_root.iterdir())
    wrapper_candidates = [
        entry
        for entry in entries
        if entry.is_dir() and (entry / "manifest.json").is_file()
    ]
    if root_manifest.is_file():
        if wrapper_candidates:
            raise PackageValidationError(
                "ZIP package layout is ambiguous; both root and wrapper manifests exist."
            )
        return extraction_root

    if (
        len(entries) == 1
        and entries[0].is_dir()
        and (entries[0] / "manifest.json").is_file()
    ):
        return entries[0]

    if wrapper_candidates or entries:
        raise PackageValidationError(
            "ZIP package layout is ambiguous; expected a root manifest or one wrapper directory."
        )
    raise PackageValidationError("ZIP package is empty.")


def _legacy_manifest_id(path: Path) -> str | None:
    try:
        data = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and isinstance(data.get("id"), str):
        return data["id"]
    return None


def _assert_safe_destructive_path(
    path: Path,
    plugin_id: str,
    canonical_target: Path,
) -> Path:
    if path.is_symlink():
        raise PackageTransactionError(
            f"Existing plugin path must not be a symbolic link: {path}"
        )
    resolved = path.resolve()
    protected = {
        plugin_registry.BASE_DIR.resolve(),
        plugin_registry.PLUGIN_DIR.resolve(),
        plugin_registry.INSTALLED_DIR.resolve(),
    }
    for protected_path in protected:
        if resolved == protected_path or _path_is_within(protected_path, resolved):
            raise PackageTransactionError(
                f"Refusing destructive operation on protected path: {resolved}"
            )

    if not resolved.is_dir():
        raise PackageTransactionError(
            f"Existing plugin path is not a safe directory: {resolved}"
        )

    installed_root = plugin_registry.INSTALLED_DIR.resolve()
    canonical = canonical_target.resolve(strict=False)
    if resolved == canonical:
        return resolved
    if _path_is_within(resolved, installed_root):
        if _legacy_manifest_id(resolved) == plugin_id:
            return resolved
        raise PackageTransactionError(
            f"Legacy plugin directory does not identify {plugin_id!r}: {resolved}"
        )

    if resolved.name == plugin_id and _legacy_manifest_id(resolved) == plugin_id:
        return resolved
    raise PackageTransactionError(
        f"Refusing suspicious legacy plugin path outside the installed root: {resolved}"
    )


def _backup_path(path: Path, purpose: str) -> Path:
    return path.with_name(f".{path.name}.{purpose}-{uuid.uuid4().hex}")


def _record_from_manifest(manifest: Mapping, existing: Mapping | None) -> dict:
    record = dict(existing or {})
    record.update(
        {
            "id": manifest["id"],
            "name": manifest["name"],
            "version": manifest["version"],
            "author": manifest.get("author", ""),
            "description": manifest.get("description", ""),
            "dir": manifest["id"],
            "enabled": existing.get("enabled", False) if existing else True,
        }
    )
    if "type" in manifest:
        record["type"] = manifest["type"]
    return record


def _rollback_install(
    target: Path,
    installed_new: bool,
    backup: Path | None,
    previous_path: Path | None,
) -> tuple[str, ...]:
    errors = []
    if installed_new and target.exists():
        try:
            shutil.rmtree(target)
        except OSError as exc:
            errors.append(f"could not remove replacement {target}: {exc}")
    if backup is not None and backup.exists() and previous_path is not None:
        try:
            os.replace(backup, previous_path)
        except OSError as exc:
            errors.append(f"could not restore backup {backup}: {exc}")
    return tuple(errors)


def _commit_staged_package(package_root: Path, manifest: Mapping) -> dict:
    plugin_id = manifest["id"]
    registry = plugin_registry.load_registry()
    previous_record = registry.get(plugin_id)
    target = plugin_registry.INSTALLED_DIR / plugin_id
    target.parent.mkdir(parents=True, exist_ok=True)

    previous_path = None
    if previous_record is not None:
        recorded_path = plugin_registry.get_installed_plugin_dir(
            plugin_id, previous_record
        )
        if recorded_path.exists():
            previous_path = _assert_safe_destructive_path(
                recorded_path, plugin_id, target
            )
        elif target.exists():
            previous_path = _assert_safe_destructive_path(target, plugin_id, target)
    elif target.exists():
        raise PackageTransactionError(
            f"Canonical plugin directory exists without a registry record: {target}"
        )

    if (
        previous_path is not None
        and previous_path.resolve() != target.resolve(strict=False)
        and target.exists()
    ):
        raise PackageTransactionError(
            f"Both legacy and canonical plugin directories exist for {plugin_id!r}."
        )

    backup = None
    installed_new = False
    try:
        if previous_path is not None:
            backup = _backup_path(previous_path, "backup")
            os.replace(previous_path, backup)

        os.replace(package_root, target)
        installed_new = True

        updated_registry = dict(registry)
        updated_registry[plugin_id] = _record_from_manifest(manifest, previous_record)
        committed_record = plugin_registry.validate_registry(updated_registry)[plugin_id]
        plugin_registry.save_registry(updated_registry)
    except Exception as exc:
        rollback_errors = _rollback_install(
            target, installed_new, backup, previous_path
        )
        recovery_paths = tuple(
            path for path in (backup, target if installed_new else None) if path and path.exists()
        )
        detail = f"Plugin installation transaction failed: {exc}"
        if rollback_errors:
            detail += "; rollback incomplete: " + "; ".join(rollback_errors)
        raise PackageTransactionError(
            detail,
            restart_required=bool(rollback_errors),
            recovery_paths=recovery_paths,
        ) from exc

    warnings = []
    restart_required = False
    recovery_paths = []
    if backup is not None and backup.exists():
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            warnings.append(f"Old plugin backup could not be deleted: {exc}")
            restart_required = True
            recovery_paths.append(str(backup))

    return {
        "ok": True,
        "plugin_id": plugin_id,
        "info": committed_record,
        "updated": previous_record is not None,
        "restart_required": restart_required,
        "warnings": warnings,
        "recovery_paths": recovery_paths,
    }


def _cleanup_staging(stage: Path, result: dict) -> None:
    if not stage.exists():
        return
    try:
        shutil.rmtree(stage)
    except OSError as exc:
        result.setdefault("warnings", []).append(
            f"Temporary staging directory could not be deleted: {exc}"
        )
        result["restart_required"] = True
        result.setdefault("recovery_paths", []).append(str(stage))


def _run_staged_install(
    prepare: Callable[[Path], Path],
    *,
    expected_plugin_id: object | None,
    expected_version: object | None,
) -> dict:
    try:
        plugin_registry.ensure_plugin_dirs()
        stage = Path(
            tempfile.mkdtemp(prefix=".plugin-stage-", dir=plugin_registry.PLUGIN_DIR)
        )
    except Exception as exc:
        return _error_result(
            PackageTransactionError(f"Could not create plugin staging directory: {exc}")
        )

    result = None
    try:
        package_root = prepare(stage)
        manifest = _load_and_validate_manifest(
            package_root,
            expected_plugin_id=expected_plugin_id,
            expected_version=expected_version,
        )
        result = _commit_staged_package(package_root, manifest)
    except Exception as exc:
        result = _error_result(exc)
    finally:
        _cleanup_staging(stage, result)
    return result


def install_plugin_file(
    py_path: str | Path,
    *,
    expected_plugin_id: object | None = None,
    expected_version: object | None = None,
) -> dict:
    source = Path(py_path)

    def prepare(stage: Path) -> Path:
        if not source.is_file():
            raise PackageValidationError(f"File does not exist: {source}")
        package_root = stage / "package"
        package_root.mkdir()
        staged_file = package_root / "plugin.py"
        shutil.copy2(source, staged_file)
        info = _inspect_plugin_file_or_raise(staged_file)
        manifest = _manifest_from_plugin_info(info, source.name)
        (package_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return package_root

    return _run_staged_install(
        prepare,
        expected_plugin_id=expected_plugin_id,
        expected_version=expected_version,
    )


def install_plugin_folder(
    folder_path: str | Path,
    *,
    expected_plugin_id: object | None = None,
    expected_version: object | None = None,
) -> dict:
    source = Path(folder_path)

    def prepare(stage: Path) -> Path:
        source_root = _find_folder_package_root(source)
        _reject_folder_symlinks(source_root)
        package_root = stage / "package"
        shutil.copytree(source_root, package_root)
        return package_root

    return _run_staged_install(
        prepare,
        expected_plugin_id=expected_plugin_id,
        expected_version=expected_version,
    )


def install_plugin_zip(
    zip_path: str | Path,
    *,
    expected_plugin_id: object | None = None,
    expected_version: object | None = None,
) -> dict:
    source = Path(zip_path)

    def prepare(stage: Path) -> Path:
        if not source.is_file():
            raise PackageValidationError(f"ZIP file does not exist: {source}")
        extraction_root = stage / "extracted"
        extraction_root.mkdir()
        _extract_zip_safely(source, extraction_root)
        return _find_extracted_package_root(extraction_root)

    return _run_staged_install(
        prepare,
        expected_plugin_id=expected_plugin_id,
        expected_version=expected_version,
    )


def remove_plugin(plugin_id: str) -> dict:
    if not isinstance(plugin_id, str) or PLUGIN_ID_PATTERN.fullmatch(plugin_id) is None:
        return _error_result(
            PluginRemovalError("plugin_id must match ^[a-z][a-z0-9_]*$.")
        )

    try:
        registry = plugin_registry.load_registry()
        record = registry.get(plugin_id)
        if record is None:
            return {
                "ok": True,
                "plugin_id": plugin_id,
                "removed": False,
                "restart_required": False,
                "warnings": [],
                "recovery_paths": [],
            }

        target = plugin_registry.get_installed_plugin_dir(plugin_id, record)
        if not target.exists():
            updated_registry = dict(registry)
            updated_registry.pop(plugin_id)
            plugin_registry.save_registry(updated_registry)
            return {
                "ok": True,
                "plugin_id": plugin_id,
                "removed": True,
                "missing_files": True,
                "restart_required": False,
                "warnings": [],
                "recovery_paths": [],
            }

        canonical = plugin_registry.INSTALLED_DIR / plugin_id
        target = _assert_safe_destructive_path(target, plugin_id, canonical)
        backup = _backup_path(target, "remove")

        try:
            os.replace(target, backup)
        except OSError as exc:
            raise PluginRemovalError(
                f"Could not move plugin files for removal: {exc}",
                restart_required=True,
            ) from exc

        try:
            updated_registry = dict(registry)
            updated_registry.pop(plugin_id)
            plugin_registry.save_registry(updated_registry)
        except Exception as exc:
            try:
                os.replace(backup, target)
            except OSError as rollback_exc:
                raise PluginRemovalError(
                    f"Registry update failed ({exc}) and plugin files could not be "
                    f"restored ({rollback_exc}).",
                    restart_required=True,
                    recovery_paths=(backup,),
                ) from exc
            raise PluginRemovalError(
                f"Registry update failed; plugin files were restored: {exc}"
            ) from exc

        warnings = []
        restart_required = False
        recovery_paths = []
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            warnings.append(f"Removed plugin backup could not be deleted: {exc}")
            restart_required = True
            recovery_paths.append(str(backup))

        return {
            "ok": True,
            "plugin_id": plugin_id,
            "removed": True,
            "restart_required": restart_required,
            "warnings": warnings,
            "recovery_paths": recovery_paths,
        }
    except Exception as exc:
        return _error_result(exc)
