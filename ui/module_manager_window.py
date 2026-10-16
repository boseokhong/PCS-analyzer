# ui/module_manager_window.py

from __future__ import annotations

from ui.widget_style import ui_text_color, ui_label_style, ui_font_name, ui_native_colors
import queue
import threading
import traceback
import webbrowser
from collections.abc import Mapping
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from app_version import OFFICIAL_PLUGIN_CATALOG_URL
from plugin_system import plugin_downloader, plugin_installer, plugin_repository
from plugin_system.plugin_loader import (
    install_plugin_file,
    install_plugin_folder,
    install_plugin_zip,
)
from plugin_system.plugin_registry import (
    PluginRegistryError,
    load_registry,
    set_plugin_enabled,
)
from ui.style import get_app_fonts


_BROWSE_STATE_PRIORITY = {
    "not_installed": 10,
    "installed": 20,
    "update_available": 30,
    "incompatible": 40,
    "deprecated": 50,
    "unavailable": 60,
    "blocked": 70,
    "local_error": 80,
}

_BROWSE_STATUS_LABELS = {
    "not_installed": "Not installed",
    "installed": "Installed",
    "update_available": "Update available",
    "incompatible": "Incompatible",
    "deprecated": "Deprecated",
    "unavailable": "Unavailable",
    "blocked": "Blocked",
    "local_error": "Local error",
}


def _warning_text(warning: object) -> str:
    if isinstance(warning, Mapping):
        return str(warning.get("message") or warning)
    return str(warning)


def _loader_diagnostics_by_id(load_results: object) -> dict[str, dict]:
    diagnostics = {}
    if not isinstance(load_results, (list, tuple)):
        return diagnostics
    for result in load_results:
        if not isinstance(result, Mapping):
            continue
        plugin_id = result.get("plugin_id")
        if isinstance(plugin_id, str):
            diagnostics[plugin_id] = dict(result)
    return diagnostics


def _diagnostic_summary(result: Mapping | None) -> str:
    if result is None:
        return "Not loaded in this session"

    status = result.get("status")
    category = result.get("category")
    if status == "loaded":
        summary = "Loaded"
    elif status == "skipped" and category == "disabled":
        summary = "Disabled"
    elif status == "skipped":
        summary = "Skipped"
    elif status == "failed":
        summary = f"Failed ({category or 'unknown'})"
    else:
        summary = "Not loaded in this session"

    if result.get("warnings"):
        summary += " — warning"
    return summary


def _build_installed_rows(
    registry: Mapping,
    load_results: object,
) -> list[dict]:
    """Create Installed-tab rows without accessing Tk widgets."""
    diagnostics = _loader_diagnostics_by_id(load_results)
    rows = []
    for plugin_id, raw_record in sorted(registry.items(), key=lambda item: str(item[0])):
        record = dict(raw_record) if isinstance(raw_record, Mapping) else {}
        diagnostic = diagnostics.get(plugin_id)
        rows.append(
            {
                "plugin_id": plugin_id,
                "name": record.get("name") or plugin_id,
                "version": record.get("version") or "",
                "author": record.get("author") or "",
                "enabled": bool(record.get("enabled", False)),
                "diagnostic_summary": _diagnostic_summary(diagnostic),
                "diagnostic": diagnostic,
                "record": record,
            }
        )
    return rows


def _build_browse_rows(classification: object | None) -> list[dict]:
    """Merge catalog classifications into one deterministic Browse sequence."""
    if classification is None:
        return []

    by_plugin_id = {}

    def consider(state_name: str, item) -> None:
        row = _browse_row(state_name, item)
        existing = by_plugin_id.get(row["plugin_id"])
        if existing is None or (
            _BROWSE_STATE_PRIORITY[state_name]
            > _BROWSE_STATE_PRIORITY[existing["state"]]
        ):
            by_plugin_id[row["plugin_id"]] = row

    for item in getattr(classification, "available", ()):
        consider("not_installed", item)
    for item in getattr(classification, "installed", ()):
        consider("installed", item)
    for item in getattr(classification, "updates", ()):
        consider("update_available", item)
    for item in getattr(classification, "incompatible", ()):
        consider("incompatible", item)
    for item in getattr(classification, "deprecated_unavailable", ()):
        consider(item.remote.status, item)
    for item in getattr(classification, "local_errors", ()):
        consider("local_error", item)

    return sorted(
        by_plugin_id.values(),
        key=lambda row: (row["name"].casefold(), row["plugin_id"]),
    )


def _browse_row(state_name: str, item) -> dict:
    remote = item.remote
    local_record = (
        dict(item.local_record) if item.local_record is not None else None
    )
    installed_version = (local_record or {}).get("version", "")
    if state_name == "update_available" and installed_version:
        version_display = f"{installed_version} → {remote.latest_version}"
    else:
        version_display = remote.latest_version

    can_install = state_name == "not_installed" and remote.status == "active"
    can_update = state_name == "update_available" and remote.status == "active"
    link_actions = []
    if remote.source_url:
        link_actions.append("repository")
    if remote.homepage_url:
        link_actions.append("homepage")

    return {
        "plugin_id": remote.id,
        "remote": remote,
        "local_record": local_record,
        "name": remote.name,
        "latest_version": remote.latest_version,
        "installed_version": installed_version,
        "version_display": version_display,
        "state": state_name,
        "status_label": _BROWSE_STATUS_LABELS[state_name],
        "reason": item.reason,
        "can_install": can_install,
        "can_update": can_update,
        "primary_action": (
            "install" if can_install else "update" if can_update else None
        ),
        "source_url": remote.source_url,
        "homepage_url": remote.homepage_url,
        "link_actions": tuple(link_actions),
    }


def _build_other_state_rows(classification: object | None) -> list[dict]:
    """Create lifecycle rows while preserving every backend distinction."""
    if classification is None:
        return []

    rows = []
    for item in classification.incompatible:
        rows.append(_classified_state_row("incompatible", item))

    for item in classification.deprecated_unavailable:
        rows.append(_classified_state_row(item.remote.status, item))

    for item in classification.local_only:
        record = dict(item.local_record)
        rows.append(
            {
                "state": "local-only",
                "plugin_id": item.plugin_id,
                "name": record.get("name") or item.plugin_id,
                "local_version": record.get("version") or "",
                "remote_version": "",
                "reason": "Plugin is installed but is not present in the catalog.",
                "remote": None,
                "local_record": record,
            }
        )

    for item in classification.local_errors:
        rows.append(_classified_state_row("local-error", item))

    order = {
        "incompatible": 0,
        "deprecated": 1,
        "unavailable": 2,
        "blocked": 3,
        "local-only": 4,
        "local-error": 5,
    }
    return sorted(
        rows,
        key=lambda row: (order.get(row["state"], 99), str(row["plugin_id"])),
    )


def _classified_state_row(state_name: str, item) -> dict:
    local = dict(item.local_record) if item.local_record is not None else None
    return {
        "state": state_name,
        "plugin_id": item.remote.id,
        "name": item.remote.name,
        "local_version": (local or {}).get("version", ""),
        "remote_version": item.remote.latest_version,
        "reason": item.reason or "",
        "remote": item.remote,
        "local_record": local,
    }


def _format_loader_details(plugin_id: str, result: Mapping | None) -> str:
    if result is None:
        return (
            f"Plugin: {plugin_id}\n"
            "Status: Not loaded in this session\n\n"
            "No loader result is available for this plugin in the current session."
        )

    lines = [
        f"Plugin: {plugin_id}",
        f"Status: {result.get('status') or 'unknown'}",
        f"Category: {result.get('category') or 'unknown'}",
        f"Phase: {result.get('phase') or 'unknown'}",
    ]
    if result.get("message"):
        lines.extend(("", f"Message:\n{result['message']}"))
    if result.get("missing_module"):
        lines.extend(("", f"Missing dependency:\n{result['missing_module']}"))
    warnings = result.get("warnings") or []
    if warnings:
        lines.extend(("", "Warnings:"))
        lines.extend(f"- {_warning_text(warning)}" for warning in warnings)
    if result.get("error"):
        lines.extend(("", f"Technical detail:\n{result['error']}"))
    if result.get("traceback"):
        lines.extend(("", f"Traceback:\n{result['traceback']}"))
    return "\n".join(lines)


def _format_operation_details(action: str, result: Mapping) -> str:
    lines = [f"Operation: {action}", f"Succeeded: {'Yes' if result.get('ok') else 'No'}"]
    for label, key in (
        ("Stage", "stage"),
        ("Error type", "error_type"),
        ("Message", "message"),
        ("Technical detail", "error"),
        ("Cleanup warning", "cleanup_warning"),
        ("Activation note", "activation_note"),
    ):
        value = result.get(key)
        if value:
            lines.extend(("", f"{label}:\n{value}"))

    warnings = result.get("warnings") or []
    if warnings:
        lines.extend(("", "Warnings:"))
        lines.extend(f"- {_warning_text(warning)}" for warning in warnings)
    lines.extend(("", f"Backend restart required: {'Yes' if result.get('restart_required') else 'No'}"))
    recovery_paths = result.get("recovery_paths") or []
    if recovery_paths:
        lines.extend(("", "Recovery paths:"))
        lines.extend(f"- {path}" for path in recovery_paths)
    if result.get("traceback"):
        lines.extend(("", f"Traceback:\n{result['traceback']}"))
    return "\n".join(lines)


def _format_catalog_details(plugin, *, reason: str | None = None) -> str:
    lines = [
        f"Name: {plugin.name}",
        f"ID: {plugin.id}",
        f"Version: {plugin.latest_version}",
        f"Author: {plugin.author}",
        f"Status: {plugin.status}",
        f"Category: {plugin.category or ''}",
        f"License: {plugin.license or ''}",
        "",
        "Description:",
        plugin.description,
    ]
    if reason:
        lines.extend(("", f"Reason:\n{reason}"))
    if plugin.min_app_version or plugin.max_app_version:
        lines.extend(
            (
                "",
                "Compatibility:",
                f"Minimum app version: {plugin.min_app_version or 'none'}",
                f"Maximum app version: {plugin.max_app_version or 'none'}",
            )
        )
    if plugin.homepage_url:
        lines.extend(("", f"Homepage:\n{plugin.homepage_url}"))
    if plugin.source_url:
        lines.extend(("", f"Source:\n{plugin.source_url}"))
    return "\n".join(lines)


def _format_browse_details(row: Mapping) -> str:
    plugin = row["remote"]
    lines = [
        f"Name: {plugin.name}",
        f"ID: {plugin.id}",
        f"Catalog version: {plugin.latest_version}",
        f"Installed version: {row.get('installed_version') or 'not installed'}",
        f"Status: {row['status_label']}",
        f"Catalog lifecycle: {plugin.status}",
        f"Author: {plugin.author}",
        f"Category: {plugin.category or ''}",
        f"License: {plugin.license or ''}",
        "",
        "Description:",
        plugin.description,
        "",
        "Compatibility:",
        f"Minimum app version: {plugin.min_app_version or 'none'}",
        f"Maximum app version: {plugin.max_app_version or 'none'}",
    ]
    if row.get("reason"):
        lines.extend(("", f"Reason:\n{row['reason']}"))
    if plugin.source_url:
        lines.extend(("", f"Repository:\n{plugin.source_url}"))
    if plugin.homepage_url:
        lines.extend(("", f"Homepage:\n{plugin.homepage_url}"))
    return "\n".join(lines)


def open_module_manager_window(state: dict):
    root = state["root"]
    existing = state.get("module_manager_window")
    if existing is not None:
        try:
            if existing.winfo_exists():
                existing.lift()
                existing.focus_force()
                return
        except Exception:
            pass

    win = tk.Toplevel(root)
    win.title("Module Manager")
    win.geometry("1050x720")
    state["module_manager_window"] = win
    fonts = get_app_fonts(state)

    outer = ttk.Frame(win, padding=10)
    outer.pack(fill="both", expand=True)
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(2, weight=1)

    title_row = ttk.Frame(outer)
    title_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    ttk.Label(
        title_row,
        text="Module Manager",
        font=fonts.get("section_large", ("Segoe UI", 11, "bold")),
    ).pack(side="left")
    ttk.Label(
        title_row,
        text="Catalog operations require an internet connection.",
        style=ui_label_style(title_row, "muted"),
    ).pack(side="right")

    warning_box = ttk.LabelFrame(outer, text="Security notice")
    warning_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
    warning_label = ttk.Label(
        warning_box,
        text=(
            "Only install modules from trusted sources. Python plugins can execute "
            "code on your computer. SHA-256 verifies package integrity, not safety."
        ),
        style=ui_label_style(root, "warning"),
        wraplength=980,
        justify="left",
    )
    warning_label.pack(fill="x", padx=8, pady=6)
    warning_label.bind("<Configure>", lambda event: warning_label.configure(wraplength=max(1, event.width)))

    notebook = ttk.Notebook(outer)
    notebook.grid(row=2, column=0, sticky="nsew")
    browse_tab = ttk.Frame(notebook, padding=6)
    installed_tab = ttk.Frame(notebook, padding=6)
    updates_tab = ttk.Frame(notebook, padding=6)
    notebook.add(browse_tab, text="Browse")
    notebook.add(installed_tab, text="Installed")
    notebook.add(updates_tab, text="Updates")

    browse_tree = _make_tree(
        browse_tab,
        (
            ("name", "Name", 230),
            ("version", "Version", 150),
            ("author", "Author", 170),
            ("category", "Category", 130),
            ("status", "Status", 125),
        ),
        height=6,
    )
    browse_description = tk.StringVar(value="Select a catalog module to see its description.")
    browse_description_label = ttk.Label(
        browse_tab,
        textvariable=browse_description,
        wraplength=950,
        justify="left",
    )
    browse_description_label.pack(side="bottom", before=browse_tree.master, fill="x", pady=(6, 4))
    browse_description_label.bind("<Configure>", lambda event: browse_description_label.configure(
        wraplength=max(1, event.width)))
    browse_buttons = ttk.Frame(browse_tab)
    browse_buttons.pack(side="bottom", before=browse_description_label, fill="x")

    installed_tree = _make_tree(
        installed_tab,
        (
            ("enabled", "Enabled", 75),
            ("name", "Name", 215),
            ("version", "Version", 90),
            ("author", "Author", 145),
            ("id", "ID", 190),
            ("diagnostic", "Session status", 225),
        ),
        height=6,
    )
    installed_buttons = ttk.Frame(installed_tab)
    installed_buttons.pack(side="bottom", before=installed_tree.master, fill="x", pady=(6, 0))
    installed_import_row = ttk.Frame(installed_buttons, padding=0)
    installed_import_row.pack(fill="x")
    installed_state_row = ttk.Frame(installed_buttons, padding=0)
    installed_state_row.pack(fill="x", pady=(6, 0))
    installed_links_row = ttk.Frame(installed_buttons, padding=0)
    installed_links_row.pack(fill="x", pady=(6, 0))

    updates_tree = _make_tree(
        updates_tab,
        (
            ("name", "Name", 250),
            ("installed", "Installed", 110),
            ("available", "Available", 110),
            ("author", "Author", 190),
            ("status", "Status", 100),
        ),
        height=6,
    )
    updates_buttons = ttk.Frame(updates_tab)
    updates_buttons.pack(side="bottom", before=updates_tree.master, fill="x", pady=(6, 0))

    other_frame = ttk.LabelFrame(outer, text="Other states")
    other_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
    other_tree = _make_tree(
        other_frame,
        (
            ("state", "State", 105),
            ("name", "Name", 220),
            ("local", "Installed", 95),
            ("remote", "Catalog", 95),
            ("reason", "Reason", 430),
        ),
        height=3,
    )

    status_row = ttk.Frame(outer)
    status_row.grid(row=4, column=0, sticky="ew", pady=(8, 0))
    status_var = tk.StringVar(value="Loading local modules…")
    ttk.Label(status_row, textvariable=status_var, style=ui_label_style(status_row, "muted")).pack(
        side="left", fill="x", expand=True
    )
    progress = ttk.Progressbar(status_row, mode="indeterminate", length=120)
    progress.pack(side="right", padx=(8, 0))

    catalog_url = state.get("plugin_catalog_url") or OFFICIAL_PLUGIN_CATALOG_URL
    catalog_state = {"catalog": None, "classification": None, "error": None}
    row_state = {"browse": {}, "installed": {}, "updates": {}, "other": {}}
    last_details = {"title": "Module Manager details", "text": "No details available."}
    worker_results: queue.Queue = queue.Queue()
    worker_busy = {"value": False}
    background_controls = []

    def set_last_details(title: str, text: str) -> None:
        last_details["title"] = title
        last_details["text"] = text

    def show_last_details() -> None:
        _show_details_dialog(win, last_details["title"], last_details["text"])

    details_button = ttk.Button(status_row, text="Last operation details", command=show_last_details)
    details_button.pack(side="right", padx=(8, 0))

    def set_background_busy(busy: bool, label: str | None = None) -> None:
        worker_busy["value"] = busy
        for control in background_controls:
            control.configure(state="disabled" if busy else "normal")
        if busy:
            progress.start(12)
            if label:
                status_var.set(label)
        else:
            progress.stop()

    def start_background(label: str, operation, callback) -> None:
        if worker_busy["value"]:
            messagebox.showinfo("Module Manager", "Another catalog operation is still running.")
            return
        set_background_busy(True, label)

        def worker() -> None:
            try:
                value = operation()
                worker_results.put((callback, value, None, None))
            except Exception as exc:
                worker_results.put((callback, None, exc, traceback.format_exc()))

        threading.Thread(target=worker, daemon=True, name="plugin-catalog-worker").start()

    def poll_worker_results() -> None:
        try:
            while True:
                callback, value, exc, trace = worker_results.get_nowait()
                set_background_busy(False)
                if exc is not None:
                    failure = {
                        "ok": False,
                        "message": "Unexpected background operation failure.",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "traceback": trace,
                    }
                    handle_operation_result("Catalog operation", failure)
                else:
                    try:
                        callback(value)
                    except Exception as callback_exc:
                        failure = {
                            "ok": False,
                            "message": "The catalog operation completed, but its UI refresh failed.",
                            "error": str(callback_exc),
                            "error_type": type(callback_exc).__name__,
                            "traceback": traceback.format_exc(),
                        }
                        handle_operation_result("Catalog UI refresh", failure)
        except queue.Empty:
            pass
        try:
            if win.winfo_exists():
                root.after(100, poll_worker_results)
        except tk.TclError:
            pass

    def current_registry() -> dict | None:
        try:
            return load_registry()
        except PluginRegistryError as exc:
            failure = {
                "ok": False,
                "message": "The local plugin registry could not be loaded.",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
            set_last_details("Registry error", _format_operation_details("Load registry", failure))
            messagebox.showerror("Module Manager", failure["message"])
            status_var.set("Could not load the local plugin registry.")
            return None

    def refresh_local_views(message: str | None = None) -> None:
        registry = current_registry()
        if registry is None:
            return
        catalog = catalog_state["catalog"]
        classification = None
        if catalog is not None:
            try:
                classification = plugin_repository.classify_catalog(catalog, registry)
            except plugin_repository.PluginRepositoryError as exc:
                failure = {
                    "ok": False,
                    "message": "Catalog classification failed.",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }
                set_last_details(
                    "Catalog classification error",
                    _format_operation_details("Classify catalog", failure),
                )
        catalog_state["classification"] = classification
        render_browse(classification)
        render_installed(registry)
        render_updates(classification)
        render_other_states(classification)
        if message:
            status_var.set(message)
        elif catalog is None:
            status_var.set(f"{len(registry)} installed module(s). Catalog not loaded.")
        elif classification is None:
            status_var.set(
                f"{len(registry)} installed module(s). Catalog classification unavailable."
            )
        else:
            status_var.set(
                f"{len(registry)} installed, "
                f"{len(row_state['browse'])} catalog plugin(s), "
                f"{len(classification.updates)} update(s)."
            )

    def render_browse(classification) -> None:
        browse_tree.delete(*browse_tree.get_children())
        row_state["browse"].clear()
        for row in _build_browse_rows(classification):
            plugin = row["remote"]
            row_state["browse"][plugin.id] = row
            browse_tree.insert(
                "",
                "end",
                iid=plugin.id,
                values=(
                    plugin.name,
                    row["version_display"],
                    plugin.author,
                    plugin.category or "",
                    row["status_label"],
                ),
            )
        update_browse_selection()

    def render_installed(registry: Mapping) -> None:
        installed_tree.delete(*installed_tree.get_children())
        row_state["installed"].clear()
        for row in _build_installed_rows(registry, state.get("plugin_load_results")):
            plugin_id = row["plugin_id"]
            row_state["installed"][plugin_id] = row
            installed_tree.insert(
                "",
                "end",
                iid=plugin_id,
                values=(
                    "Yes" if row["enabled"] else "No",
                    row["name"],
                    row["version"],
                    row["author"],
                    plugin_id,
                    row["diagnostic_summary"],
                ),
            )
        update_installed_links()

    def render_updates(classification) -> None:
        updates_tree.delete(*updates_tree.get_children())
        row_state["updates"].clear()
        if classification is None:
            update_update_links()
            return
        for item in classification.updates:
            plugin = item.remote
            local_version = (item.local_record or {}).get("version", "")
            row_state["updates"][plugin.id] = item
            updates_tree.insert(
                "",
                "end",
                iid=plugin.id,
                values=(
                    plugin.name,
                    local_version,
                    plugin.latest_version,
                    plugin.author,
                    plugin.status,
                ),
            )
        update_update_links()

    def render_other_states(classification) -> None:
        other_tree.delete(*other_tree.get_children())
        row_state["other"].clear()
        for index, row in enumerate(_build_other_state_rows(classification)):
            iid = f"other-{index}"
            row_state["other"][iid] = row
            other_tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    row["state"],
                    row["name"],
                    row["local_version"],
                    row["remote_version"],
                    row["reason"],
                ),
            )

    def selected(tree, group: str, *, warn: bool = True):
        selection = tree.selection()
        if not selection:
            if warn:
                messagebox.showwarning("Module Manager", "No module selected.")
            return None
        return row_state[group].get(selection[0])

    def refresh_main_plugin_menu() -> None:
        refresh_fn = state.get("refresh_plugin_menu")
        if callable(refresh_fn):
            try:
                refresh_fn()
            except Exception as exc:
                failure = {
                    "ok": False,
                    "message": "The Modules menu could not be refreshed.",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }
                set_last_details(
                    "Modules menu refresh error",
                    _format_operation_details("Refresh Modules menu", failure),
                )

    def refresh_after_local_change(message: str) -> None:
        refresh_main_plugin_menu()
        refresh_local_views(message)

    def fetch_catalog() -> None:
        start_background(
            "Fetching plugin catalog…",
            lambda: plugin_repository.fetch_catalog_result(catalog_url),
            finish_catalog_fetch,
        )

    def finish_catalog_fetch(fetch_result) -> None:
        if not fetch_result.ok:
            error = fetch_result.error
            failure = {
                "ok": False,
                "message": "The plugin catalog could not be loaded.",
                "error": error.message if error else "Unknown repository error.",
                "error_type": error.code if error else "repository_error",
            }
            catalog_state["error"] = error
            set_last_details(
                "Catalog fetch error",
                _format_operation_details("Fetch catalog", failure),
            )
            messagebox.showerror("Plugin catalog", failure["message"])
            refresh_local_views("Catalog fetch failed; local modules remain available.")
            return
        catalog_state["catalog"] = fetch_result.catalog
        catalog_state["error"] = None
        refresh_local_views("Plugin catalog refreshed.")

    def confirm_install_security(plugin_name: str = "this plugin") -> bool:
        return messagebox.askyesno(
            "Install plugin?",
            (
                f"Install {plugin_name}?\n\n"
                "Only install modules from trusted sources. Python plugins can "
                "execute code, access files, and launch external processes.\n\n"
                "Do you want to continue?"
            ),
        )

    def run_selected_browse_action() -> None:
        row = selected(browse_tree, "browse")
        if row is None or row["primary_action"] is None:
            return
        plugin = row["remote"]
        is_update = row["primary_action"] == "update"
        prompt_name = f"the update for {plugin.name}" if is_update else plugin.name
        if not confirm_install_security(prompt_name):
            return
        start_background(
            (
                f"Downloading and updating {plugin.name}…"
                if is_update
                else f"Downloading and installing {plugin.name}…"
            ),
            lambda: plugin_downloader.install_catalog_plugin(plugin),
            lambda result: finish_install(result, plugin, is_update=is_update),
        )

    def update_selected_catalog_plugin() -> None:
        item = selected(updates_tree, "updates")
        if item is None or not confirm_install_security(f"the update for {item.remote.name}"):
            return
        plugin = item.remote
        start_background(
            f"Downloading and updating {plugin.name}…",
            lambda: plugin_downloader.install_catalog_plugin(plugin),
            lambda result: finish_install(result, plugin, is_update=True),
        )

    def finish_install(result: Mapping, plugin, *, is_update: bool) -> None:
        result_for_view = dict(result)
        if is_update and result.get("ok"):
            result_for_view["activation_note"] = (
                "The updated files are installed on disk. Restart PCS Analyzer to "
                "execute the new plugin code. A Modules-menu refresh may re-register "
                "the previous in-memory module; it does not activate the new version."
            )
        action = "Update module" if is_update else "Install module"
        handle_operation_result(action, result_for_view)
        if not result.get("ok"):
            return
        refresh_main_plugin_menu()
        refresh_local_views(
            (
                f"Updated {plugin.name} on disk. Restart required to execute "
                f"version {plugin.latest_version}."
                if is_update
                else f"Installed {plugin.name}; Modules menu refreshed."
            )
        )

    def run_local_install(action: str, installer, path: str) -> None:
        # Deliberately synchronous: single-file inspection may execute legacy
        # plugin code that assumes it is running on Tk's main thread.
        try:
            result = installer(path)
        except Exception as exc:
            result = {
                "ok": False,
                "message": f"{action} failed.",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
        result_for_view = dict(result)
        if result.get("ok") and result.get("updated"):
            result_for_view["activation_note"] = (
                "The replacement files are installed on disk. Restart PCS Analyzer "
                "to execute the new plugin code. A Modules-menu refresh may "
                "re-register the previous in-memory module."
            )
        handle_operation_result(action, result_for_view)
        if result.get("ok"):
            refresh_after_local_change(
                (
                    f"Updated {result.get('plugin_id', 'plugin')} on disk; restart "
                    "required to execute the replacement."
                    if result.get("updated")
                    else f"Installed {result.get('plugin_id', 'plugin')}; "
                    "Modules menu refreshed."
                )
            )

    def install_local_file() -> None:
        path = filedialog.askopenfilename(
            title="Install .py plugin",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        if path and confirm_install_security("this local Python plugin"):
            run_local_install("Install .py module", install_plugin_file, path)

    def install_local_folder() -> None:
        path = filedialog.askdirectory(title="Install plugin folder")
        if path and confirm_install_security("this local plugin folder"):
            run_local_install("Install module folder", install_plugin_folder, path)

    def install_local_zip() -> None:
        path = filedialog.askopenfilename(
            title="Install plugin ZIP",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
        )
        if path and confirm_install_security("this local ZIP plugin"):
            run_local_install("Install module ZIP", install_plugin_zip, path)

    def set_selected_enabled(enabled: bool) -> None:
        row = selected(installed_tree, "installed")
        if row is None:
            return
        try:
            set_plugin_enabled(row["plugin_id"], enabled)
        except PluginRegistryError as exc:
            result = {
                "ok": False,
                "message": "The plugin state could not be saved.",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
            handle_operation_result("Enable/disable module", result)
            return
        refresh_after_local_change(
            f"{row['plugin_id']} {'enabled' if enabled else 'disabled'}; "
            "Modules menu refreshed."
        )

    def remove_selected() -> None:
        row = selected(installed_tree, "installed")
        if row is None:
            return
        if not messagebox.askyesno(
            "Remove module",
            f"Remove module '{row['name']}'?\n\nIts installed files will be removed.",
        ):
            return
        result = plugin_installer.remove_plugin(row["plugin_id"])
        handle_operation_result("Remove module", result)
        if result.get("ok"):
            refresh_after_local_change(
                f"Removed {row['plugin_id']}; Modules menu refreshed."
            )

    def handle_operation_result(action: str, result: Mapping) -> None:
        details = _format_operation_details(action, result)
        set_last_details(f"{action} details", details)
        if not result.get("ok"):
            summary = result.get("message") or result.get("error") or f"{action} failed."
            messagebox.showerror(action, summary)
            status_var.set(f"{action} failed. See Last operation details.")
            return

        warnings = result.get("warnings") or []
        restart_required = bool(result.get("restart_required"))
        activation_note = result.get("activation_note")
        summary = f"{action} completed successfully."
        if activation_note:
            summary += f"\n\n{activation_note}"
        if restart_required:
            summary += "\n\nThe backend reports that a restart is required."
        if warnings or result.get("recovery_paths"):
            summary += "\n\nWarnings or recovery information are available in details."
        if activation_note or restart_required or warnings or result.get("recovery_paths"):
            messagebox.showwarning(action, summary)
        else:
            messagebox.showinfo(action, summary)

    def show_browse_details() -> None:
        row = selected(browse_tree, "browse")
        if row is not None:
            _show_details_dialog(
                win, row["name"], _format_browse_details(row)
            )

    def show_update_details() -> None:
        item = selected(updates_tree, "updates")
        if item is not None:
            browse_row = row_state["browse"].get(item.remote.id)
            text = (
                _format_browse_details(browse_row)
                if browse_row is not None
                else _format_catalog_details(item.remote)
            )
            _show_details_dialog(win, item.remote.name, text)

    def show_installed_details() -> None:
        row = selected(installed_tree, "installed")
        if row is not None:
            _show_details_dialog(
                win,
                f"{row['name']} diagnostics",
                _format_loader_details(row["plugin_id"], row["diagnostic"]),
            )

    def show_other_details() -> None:
        row = selected(other_tree, "other")
        if row is None:
            return
        if row["remote"] is not None:
            text = _format_catalog_details(row["remote"], reason=row["reason"])
        else:
            text = (
                f"Name: {row['name']}\nID: {row['plugin_id']}\n"
                f"State: {row['state']}\nInstalled version: {row['local_version']}\n\n"
                f"Reason:\n{row['reason']}"
            )
        _show_details_dialog(win, row["name"], text)

    def open_catalog_url(url: str | None, label: str) -> None:
        if not url:
            return
        try:
            opened = webbrowser.open_new_tab(url)
        except Exception as exc:
            messagebox.showerror(
                "Module Manager", f"Could not open {label.lower()}:\n\n{exc}"
            )
            return
        if opened is False:
            messagebox.showerror(
                "Module Manager", f"No browser accepted the {label.lower()} URL."
            )

    def update_browse_selection(_event=None) -> None:
        row = selected(browse_tree, "browse", warn=False)
        browse_description.set(
            row["remote"].description
            if row is not None
            else "Select a catalog module to see its description."
        )
        browse_action_button.grid_remove()
        browse_repository_button.grid_remove()
        browse_homepage_button.grid_remove()
        if row is None:
            return
        if row["primary_action"] is not None:
            browse_action_button.configure(
                text="Update" if row["primary_action"] == "update" else "Install"
            )
            browse_action_button.grid()
        if row["source_url"]:
            browse_repository_button.grid()
        if row["homepage_url"]:
            browse_homepage_button.grid()

    def update_installed_links(_event=None) -> None:
        repository_installed_button.pack_forget()
        homepage_installed_button.pack_forget()
        row = selected(installed_tree, "installed", warn=False)
        browse_row = (
            row_state["browse"].get(row["plugin_id"]) if row is not None else None
        )
        if browse_row is None:
            return
        if browse_row["source_url"]:
            repository_installed_button.pack(side="left", padx=(6, 0))
        if browse_row["homepage_url"]:
            homepage_installed_button.pack(side="left", padx=(6, 0))

    def update_update_links(_event=None) -> None:
        repository_update_button.pack_forget()
        homepage_update_button.pack_forget()
        item = selected(updates_tree, "updates", warn=False)
        if item is None:
            return
        if item.remote.source_url:
            repository_update_button.pack(side="left", padx=(6, 0))
        if item.remote.homepage_url:
            homepage_update_button.pack(side="left", padx=(6, 0))

    def open_installed_url(field_name: str, label: str) -> None:
        row = selected(installed_tree, "installed", warn=False)
        browse_row = (
            row_state["browse"].get(row["plugin_id"]) if row is not None else None
        )
        open_catalog_url(browse_row.get(field_name) if browse_row else None, label)

    def open_update_url(field_name: str, label: str) -> None:
        item = selected(updates_tree, "updates", warn=False)
        open_catalog_url(
            getattr(item.remote, field_name, None) if item is not None else None,
            label,
        )

    browse_buttons.columnconfigure(8, weight=1)
    browse_action_button = ttk.Button(
        browse_buttons, text="Install", command=run_selected_browse_action
    )
    browse_action_button.grid(row=0, column=0, sticky="w")
    ttk.Button(browse_buttons, text="Details", command=show_browse_details).grid(
        row=0, column=1, padx=(6, 0), sticky="w"
    )
    browse_repository_button = ttk.Button(
        browse_buttons,
        text="Repository",
        command=lambda: open_catalog_url(
            (selected(browse_tree, "browse", warn=False) or {}).get("source_url"),
            "Repository",
        ),
    )
    browse_repository_button.grid(row=0, column=2, padx=(6, 0), sticky="w")
    browse_homepage_button = ttk.Button(
        browse_buttons,
        text="Homepage",
        command=lambda: open_catalog_url(
            (selected(browse_tree, "browse", warn=False) or {}).get("homepage_url"),
            "Homepage",
        ),
    )
    browse_homepage_button.grid(row=0, column=3, padx=(6, 0), sticky="w")
    refresh_catalog_button = ttk.Button(
        browse_buttons, text="Refresh Catalog", command=fetch_catalog
    )
    refresh_catalog_button.grid(row=0, column=9, sticky="e")
    background_controls.extend((browse_action_button, refresh_catalog_button))

    local_file_button = ttk.Button(
        installed_import_row, text="Install .py Plugin", command=install_local_file
    )
    local_file_button.pack(side="left")
    local_folder_button = ttk.Button(
        installed_import_row, text="Install Folder", command=install_local_folder
    )
    local_folder_button.pack(side="left", padx=(6, 0))
    local_zip_button = ttk.Button(
        installed_import_row, text="Install ZIP", command=install_local_zip
    )
    local_zip_button.pack(side="left", padx=(6, 12))
    enable_button = ttk.Button(
        installed_state_row, text="Enable", command=lambda: set_selected_enabled(True)
    )
    enable_button.pack(side="left")
    disable_button = ttk.Button(
        installed_state_row, text="Disable", command=lambda: set_selected_enabled(False)
    )
    disable_button.pack(side="left", padx=(6, 0))
    remove_button = ttk.Button(installed_state_row, text="Remove", command=remove_selected)
    remove_button.pack(side="left", padx=(6, 0))
    ttk.Button(
        installed_state_row, text="Details", command=show_installed_details
    ).pack(side="left", padx=(6, 0))
    repository_installed_button = ttk.Button(
        installed_links_row,
        text="Repository",
        command=lambda: open_installed_url("source_url", "Repository"),
    )
    homepage_installed_button = ttk.Button(
        installed_links_row,
        text="Homepage",
        command=lambda: open_installed_url("homepage_url", "Homepage"),
    )
    ttk.Button(
        installed_import_row,
        text="Refresh",
        command=lambda: refresh_local_views("Local module list refreshed."),
    ).pack(side="right")

    update_button = ttk.Button(
        updates_buttons, text="Update", command=update_selected_catalog_plugin
    )
    update_button.pack(side="left")
    ttk.Button(updates_buttons, text="Details", command=show_update_details).pack(
        side="left", padx=(6, 0)
    )
    repository_update_button = ttk.Button(
        updates_buttons,
        text="Repository",
        command=lambda: open_update_url("source_url", "Repository"),
    )
    homepage_update_button = ttk.Button(
        updates_buttons,
        text="Homepage",
        command=lambda: open_update_url("homepage_url", "Homepage"),
    )
    background_controls.append(update_button)
    background_controls.extend(
        (
            local_file_button,
            local_folder_button,
            local_zip_button,
            enable_button,
            disable_button,
            remove_button,
        )
    )

    ttk.Button(other_frame, text="Details", command=show_other_details).pack(
        anchor="e", padx=6, pady=(0, 6)
    )

    browse_tree.bind("<<TreeviewSelect>>", update_browse_selection)
    browse_tree.bind("<Double-1>", lambda _event: show_browse_details())
    installed_tree.bind("<<TreeviewSelect>>", update_installed_links)
    installed_tree.bind("<Double-1>", lambda _event: show_installed_details())
    updates_tree.bind("<<TreeviewSelect>>", update_update_links)
    updates_tree.bind("<Double-1>", lambda _event: show_update_details())
    other_tree.bind("<Double-1>", lambda _event: show_other_details())

    def on_close() -> None:
        if worker_busy["value"]:
            messagebox.showinfo(
                "Module Manager",
                "Please wait for the current catalog operation to finish before closing.",
            )
            return
        try:
            win.destroy()
        finally:
            state["module_manager_window"] = None

    win.protocol("WM_DELETE_WINDOW", on_close)
    update_browse_selection()
    update_installed_links()
    update_update_links()
    refresh_local_views()
    poll_worker_results()
    fetch_catalog()


def _make_tree(parent, column_specs, *, height: int) -> ttk.Treeview:
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)
    columns = tuple(spec[0] for spec in column_specs)
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=height)
    for name, heading, width in column_specs:
        tree.heading(name, text=heading)
        tree.column(name, width=width, anchor="w")
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    horizontal = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=scrollbar.set, xscrollcommand=horizontal.set)
    tree.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")
    horizontal.grid(row=1, column=0, sticky="ew")
    return tree


def _show_details_dialog(parent, title: str, text: str) -> None:
    dialog = ui_native_colors(tk.Toplevel(parent), background="surface")
    dialog.title(title)
    dialog.geometry("760x520")
    frame = ttk.Frame(dialog, padding=10)
    frame.pack(fill="both", expand=True)
    text_widget = tk.Text(frame, wrap="word", font=ui_font_name(frame, "report"))
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)
    text_widget.insert("1.0", text)
    text_widget.configure(state="disabled")
    text_widget.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=(0, 10))
    dialog.transient(parent)
    dialog.lift()
