# plugin_system/plugin_loader.py

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
import shutil
import tempfile
import zipfile
from pathlib import Path

from plugin_system.plugin_registry import (
    INSTALLED_DIR,
    load_registry,
    save_registry,
    ensure_plugin_dirs,
)

def load_manifest(plugin_dir: Path) -> dict | None:
    manifest_path = plugin_dir / "manifest.json"

    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def load_plugin(plugin_id: str, record: dict, app) -> dict:
    plugin_dir = Path(record.get("dir", ""))

    if not plugin_dir.is_absolute():
        plugin_dir = INSTALLED_DIR / plugin_id

    manifest = load_manifest(plugin_dir)
    if manifest is None:
        return {
            "ok": False,
            "plugin_id": plugin_id,
            "error": f"Missing or invalid manifest.json in {plugin_dir}",
        }

    entry = manifest.get("entry", "plugin.py")
    entry_path = plugin_dir / entry

    if not entry_path.exists():
        return {
            "ok": False,
            "plugin_id": plugin_id,
            "error": f"Plugin entry file does not exist: {entry_path}",
        }

    plugin_dir_str = str(plugin_dir)
    if plugin_dir_str not in sys.path:
        sys.path.insert(0, plugin_dir_str)

    try:
        module_name = f"pcs_plugin_{plugin_id}"
        public_package_name = plugin_id

        spec = importlib.util.spec_from_file_location(
            module_name,
            entry_path,
            submodule_search_locations=[str(plugin_dir)],
        )

        module = importlib.util.module_from_spec(spec)

        if spec.loader is None:
            raise RuntimeError("Could not create plugin loader.")

        sys.modules[module_name] = module

        # For folder-type plugins, also expose the plugin under its actual plugin_id.
        # This is needed when internal files use absolute imports:
        #     from pcs_motion_explorer.core... import ...
        old_public_module = sys.modules.get(public_package_name)
        sys.modules[public_package_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception:
            # Restore old module only if loading failed.
            if old_public_module is not None:
                sys.modules[public_package_name] = old_public_module
            else:
                sys.modules.pop(public_package_name, None)
            raise

        if not hasattr(module, "register"):
            return {
                "ok": False,
                "plugin_id": plugin_id,
                "error": "register(app) function is missing.",
            }

        module.register(app)

        return {
            "ok": True,
            "plugin_id": plugin_id,
            "module": module,
            "manifest": manifest,
        }

    except Exception:
        return {
            "ok": False,
            "plugin_id": plugin_id,
            "error": traceback.format_exc(),
        }


def load_enabled_plugins(app) -> list[dict]:
    ensure_plugin_dirs()

    registry = load_registry()
    results = []

    for plugin_id, record in registry.items():
        if not record.get("enabled", False):
            continue

        result = load_plugin(plugin_id, record, app)
        results.append(result)

    return results


def _safe_plugin_id(text: str) -> str:
    text = str(text or "").strip().lower()
    out = []

    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_", "."):
            out.append("_")

    plugin_id = "".join(out).strip("_")
    return plugin_id or "unnamed_plugin"


def inspect_plugin_file(py_path: str | Path) -> dict:
    """
    Temporarily load a .py plugin to read PLUGIN_INFO.
    Does not call register(app).
    """
    py_path = Path(py_path)

    if not py_path.exists():
        return {"ok": False, "error": f"File does not exist: {py_path}"}

    try:
        module_name = f"_pcs_plugin_inspect_{py_path.stem}"
        plugin_dir = py_path.parent

        spec = importlib.util.spec_from_file_location(
            module_name,
            py_path,
            submodule_search_locations=[str(plugin_dir)],
        )

        module = importlib.util.module_from_spec(spec)

        if spec.loader is None:
            raise RuntimeError("Could not create plugin loader.")

        # Important for @dataclass and other introspection-heavy decorators.
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        finally:
            # Inspection-only import. Remove it to avoid stale temporary modules.
            sys.modules.pop(module_name, None)

        info = getattr(module, "PLUGIN_INFO", None)
        if not isinstance(info, dict):
            return {
                "ok": False,
                "error": "PLUGIN_INFO dictionary is missing.",
            }

        if not hasattr(module, "register"):
            return {
                "ok": False,
                "error": "register(app) function is missing.",
            }

        plugin_id = _safe_plugin_id(
            info.get("id") or info.get("name") or py_path.stem
        )

        return {
            "ok": True,
            "plugin_id": plugin_id,
            "info": info,
        }

    except Exception:
        return {
            "ok": False,
            "error": traceback.format_exc(),
        }


def _write_manifest(plugin_dir: Path, info: dict, entry: str = "plugin.py") -> dict:
    plugin_id = _safe_plugin_id(info.get("id") or info.get("name") or plugin_dir.name)

    manifest = {
        "id": plugin_id,
        "name": info.get("name", plugin_id),
        "version": info.get("version", ""),
        "author": info.get("author", ""),
        "description": info.get("description", ""),
        "entry": entry,
        "type": info.get("type", "window"),
        "standalone": bool(info.get("standalone", True)),
        "dependencies": info.get("dependencies", []),
    }

    manifest_path = plugin_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def _register_installed_plugin(plugin_id: str, manifest: dict) -> dict:
    registry = load_registry()

    registry[plugin_id] = {
        "id": plugin_id,
        "name": manifest.get("name", plugin_id),
        "version": manifest.get("version", ""),
        "author": manifest.get("author", ""),
        "description": manifest.get("description", ""),
        "dir": plugin_id,
        "enabled": True,
        "type": manifest.get("type", "window"),
    }

    save_registry(registry)
    return registry[plugin_id]


def install_plugin_file(py_path: str | Path) -> dict:
    """
    Install a single .py plugin.

    External input:
        some_plugin.py

    Internal installed form:
        plugins/installed/<plugin_id>/
            manifest.json
            plugin.py
    """
    ensure_plugin_dirs()
    py_path = Path(py_path)

    inspected = inspect_plugin_file(py_path)
    if not inspected.get("ok"):
        return inspected

    info = inspected["info"]
    plugin_id = inspected["plugin_id"]

    target_dir = INSTALLED_DIR / plugin_id

    if target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "plugin.py"

    shutil.copy2(py_path, target_file)

    manifest = _write_manifest(target_dir, info, entry="plugin.py")
    rec = _register_installed_plugin(plugin_id, manifest)

    return {
        "ok": True,
        "plugin_id": plugin_id,
        "info": rec,
    }


def _find_manifest_dir(folder: Path) -> Path | None:
    """
    Accept either:
      folder/manifest.json
    or
      folder/<plugin_name>/manifest.json
    """
    folder = Path(folder)

    if (folder / "manifest.json").exists():
        return folder

    children = [p for p in folder.iterdir() if p.is_dir()]
    for child in children:
        if (child / "manifest.json").exists():
            return child

    return None


def install_plugin_folder(folder_path: str | Path) -> dict:
    """
    Install a folder-type plugin.
    The folder must contain manifest.json and an entry file.
    """
    ensure_plugin_dirs()
    folder_path = Path(folder_path)

    src_dir = _find_manifest_dir(folder_path)
    if src_dir is None:
        return {
            "ok": False,
            "error": "manifest.json was not found in the selected folder.",
        }

    manifest_path = src_dir / "manifest.json"

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return {
            "ok": False,
            "error": traceback.format_exc(),
        }

    plugin_id = _safe_plugin_id(manifest.get("id") or manifest.get("name") or src_dir.name)
    entry = manifest.get("entry", "plugin.py")

    if not (src_dir / entry).exists():
        return {
            "ok": False,
            "error": f"Entry file does not exist: {entry}",
        }

    target_dir = INSTALLED_DIR / plugin_id

    if target_dir.exists():
        shutil.rmtree(target_dir)

    shutil.copytree(src_dir, target_dir)

    # Ensure manifest has normalized id/name fields.
    manifest["id"] = plugin_id
    manifest.setdefault("name", plugin_id)
    manifest.setdefault("entry", entry)

    with (target_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    rec = _register_installed_plugin(plugin_id, manifest)

    return {
        "ok": True,
        "plugin_id": plugin_id,
        "info": rec,
    }


def install_plugin_zip(zip_path: str | Path) -> dict:
    """
    Install a ZIP plugin package.
    The zip must contain either manifest.json at root,
    or one top-level folder containing manifest.json.
    """
    ensure_plugin_dirs()
    zip_path = Path(zip_path)

    if not zip_path.exists():
        return {
            "ok": False,
            "error": f"ZIP file does not exist: {zip_path}",
        }

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(td)
        except Exception:
            return {
                "ok": False,
                "error": traceback.format_exc(),
            }

        return install_plugin_folder(td)