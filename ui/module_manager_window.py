# ui/module_manager_window.py

from __future__ import annotations

import json
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ui.style import get_app_fonts

from plugin_system.plugin_registry import (
    INSTALLED_DIR,
    load_registry,
    save_registry,
    remove_plugin_from_registry,
)
from plugin_system.plugin_loader import (
    install_plugin_file,
    install_plugin_folder,
    install_plugin_zip,
)


def open_module_manager_window(state: dict):
    root = state["root"]

    win = state.get("module_manager_window")
    if win is not None:
        try:
            if win.winfo_exists():
                win.lift()
                win.focus_force()
                return
        except Exception:
            pass

    win = tk.Toplevel(root)
    win.title("Module Manager")
    win.geometry("900x560")
    state["module_manager_window"] = win

    outer = ttk.Frame(win, padding=10)
    outer.pack(fill="both", expand=True)

    fonts = get_app_fonts(state)

    # ============================================================
    # Header
    # ============================================================
    title_row = ttk.Frame(outer)
    title_row.pack(fill="x", pady=(0, 8))

    ttk.Label(
        title_row,
        text="Installed Modules",
        font=fonts.get("section_large", ("Segoe UI", 11, "bold")),
    ).pack(side="left")

    ttk.Label(
        title_row,
        text="Changes usually require restarting PCS Analyzer.",
        foreground="gray",
    ).pack(side="right")

    # ============================================================
    # Security notice
    # ============================================================
    warning_box = ttk.LabelFrame(outer, text="Security notice")
    warning_box.pack(fill="x", pady=(0, 8))

    ttk.Label(
        warning_box,
        text=(
            "Only install modules from trusted sources. "
            "Python plugins can execute code on your computer."
        ),
        foreground="#8a5a00",
        wraplength=820,
        justify="left",
    ).pack(anchor="w", padx=8, pady=6)

    # ============================================================
    # Treeview
    # ============================================================
    tree_frame = ttk.Frame(outer)
    tree_frame.pack(fill="both", expand=True)

    cols = ("enabled", "name", "version", "author", "id", "type")
    tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=15)

    headers = {
        "enabled": "Enabled",
        "name": "Name",
        "version": "Version",
        "author": "Author",
        "id": "ID",
        "type": "Type",
    }

    widths = {
        "enabled": 80,
        "name": 220,
        "version": 90,
        "author": 150,
        "id": 180,
        "type": 100,
    }

    for col in cols:
        tree.heading(col, text=headers[col])
        tree.column(col, width=widths[col], anchor="w")

    yscroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=yscroll.set)

    tree.pack(side="left", fill="both", expand=True)
    yscroll.pack(side="right", fill="y")

    # ============================================================
    # Status
    # ============================================================
    status_var = tk.StringVar(value="Ready.")
    ttk.Label(
        outer,
        textvariable=status_var,
        foreground="gray",
    ).pack(anchor="w", pady=(8, 4))

    # ============================================================
    # Internal helpers
    # ============================================================
    def refresh():
        tree.delete(*tree.get_children())
        registry = load_registry()

        inserted = 0

        for plugin_id, rec in registry.items():
            try:
                manifest = _load_manifest_for_record(plugin_id, rec)

                name = rec.get("name") or manifest.get("name") or plugin_id
                version = rec.get("version") or manifest.get("version") or ""
                author = rec.get("author") or manifest.get("author") or ""
                ptype = manifest.get("type") or rec.get("type") or "plugin"

                tree.insert(
                    "",
                    "end",
                    iid=str(plugin_id),
                    values=(
                        "Yes" if rec.get("enabled", False) else "No",
                        name,
                        version,
                        author,
                        plugin_id,
                        ptype,
                    ),
                )
                inserted += 1

            except Exception as e:
                print(f"[Module Manager] Failed to insert module row: {plugin_id}")
                print(e)

        status_var.set(
            f"{len(registry)} module(s) registered, {inserted} shown."
        )

    def refresh_main_plugin_menu():
        fn = state.get("refresh_plugin_menu")
        if callable(fn):
            try:
                fn()
            except Exception as e:
                print("[Module Manager] Failed to refresh plugin menu:")
                print(e)

    def selected_plugin_id():
        sel = tree.selection()
        if not sel:
            messagebox.showwarning("Module Manager", "No module selected.")
            return None
        return sel[0]

    def set_enabled(enabled: bool):
        plugin_id = selected_plugin_id()
        if not plugin_id:
            return

        registry = load_registry()
        if plugin_id not in registry:
            messagebox.showerror("Module Manager", f"Module not found: {plugin_id}")
            return

        registry[plugin_id]["enabled"] = bool(enabled)
        save_registry(registry)

        refresh()
        refresh_main_plugin_menu()
        status_var.set(
            f"{plugin_id} {'enabled' if enabled else 'disabled'}. "
            "Module menu refreshed."
        )

    def remove_selected():
        plugin_id = selected_plugin_id()
        if not plugin_id:
            return

        registry = load_registry()
        rec = registry.get(plugin_id)
        if not rec:
            return

        name = rec.get("name", plugin_id)

        ok = messagebox.askyesno(
            "Remove module",
            f"Remove module '{name}'?\n\n"
            "This will delete its installed plugin folder.",
        )
        if not ok:
            return

        plugin_dir = _plugin_dir_for_record(plugin_id, rec)

        remove_plugin_from_registry(plugin_id)

        if plugin_dir.exists():
            try:
                shutil.rmtree(plugin_dir)
            except Exception as e:
                messagebox.showwarning(
                    "Remove module",
                    (
                        "Registry entry was removed, but the plugin folder "
                        f"could not be deleted:\n\n{e}"
                    ),
                )

        refresh()
        refresh_main_plugin_menu()
        status_var.set(f"{plugin_id} removed. Module menu refreshed.")

    def install_py():
        path = filedialog.askopenfilename(
            title="Install .py plugin",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        if not path:
            return

        if not _confirm_install_security():
            return

        result = install_plugin_file(path)
        _handle_install_result(result)
        refresh()
        refresh_main_plugin_menu()

    def install_folder():
        path = filedialog.askdirectory(
            title="Install plugin folder",
        )
        if not path:
            return

        if not _confirm_install_security():
            return

        result = install_plugin_folder(path)
        _handle_install_result(result)
        refresh()

    def install_zip():
        path = filedialog.askopenfilename(
            title="Install plugin ZIP",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
        )
        if not path:
            return

        if not _confirm_install_security():
            return

        result = install_plugin_zip(path)
        _handle_install_result(result)
        refresh()

    def _handle_install_result(result: dict):
        if not result.get("ok"):
            messagebox.showerror(
                "Install module",
                f"Installation failed:\n\n{result.get('error', '')}",
            )
            status_var.set("Installation failed.")
            return

        info = result.get("info", {})
        name = info.get("name") or result.get("plugin_id", "plugin")

        messagebox.showinfo(
            "Install module",
            f"Module installed:\n{name}\n\n"
            "If this was an update of an existing plugin, restart PCS Analyzer for a clean reload.",
        )
        status_var.set(f"Installed {name}. Module menu refreshed.")

    def _confirm_install_security() -> bool:
        return messagebox.askyesno(
            "Install plugin?",
            (
                "Only install modules from trusted sources.\n\n"
                "Python plugins can execute code on your computer, "
                "including file access and external process execution.\n\n"
                "Do you want to continue?"
            ),
        )

    # ============================================================
    # Buttons
    # ============================================================
    btns = ttk.Frame(outer)
    btns.pack(fill="x", pady=(8, 0))

    ttk.Button(
        btns,
        text="Install .py Plugin",
        command=install_py,
    ).pack(side="left")

    ttk.Button(
        btns,
        text="Install Folder",
        command=install_folder,
    ).pack(side="left", padx=(6, 0))

    ttk.Button(
        btns,
        text="Install ZIP",
        command=install_zip,
    ).pack(side="left", padx=(6, 12))

    ttk.Separator(btns, orient="vertical").pack(side="left", fill="y", padx=8)

    ttk.Button(
        btns,
        text="Enable",
        command=lambda: set_enabled(True),
    ).pack(side="left")

    ttk.Button(
        btns,
        text="Disable",
        command=lambda: set_enabled(False),
    ).pack(side="left", padx=(6, 0))

    ttk.Button(
        btns,
        text="Remove",
        command=remove_selected,
    ).pack(side="left", padx=(6, 0))

    ttk.Button(
        btns,
        text="Refresh",
        command=refresh,
    ).pack(side="right")

    # ============================================================
    # Close handling
    # ============================================================
    def on_close():
        try:
            win.destroy()
        finally:
            state["module_manager_window"] = None

    win.protocol("WM_DELETE_WINDOW", on_close)

    refresh()


def _plugin_dir_for_record(plugin_id: str, rec: dict) -> Path:
    d = rec.get("dir", plugin_id)
    p = Path(d)

    if p.is_absolute():
        return p

    return INSTALLED_DIR / d


def _load_manifest_for_record(plugin_id: str, rec: dict) -> dict:
    plugin_dir = _plugin_dir_for_record(plugin_id, rec)
    manifest_path = plugin_dir / "manifest.json"

    if not manifest_path.exists():
        return {}

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}