from __future__ import annotations

import re
from pathlib import Path
from tkinter import messagebox

SUPPORTED_PROJECT_EXTS = {".pcsp"}
SUPPORTED_STRUCTURE_EXTS = {".xyz", ".out", ".log"}


def _split_drop_files(raw: str) -> list[str]:
    """
    Split a TkinterDnD file-drop payload into clean file paths.

    Common payload examples:
        C:/path/file.xyz
        {C:/path with spaces/file.xyz}
        {C:/a.pcsp} {C:/b.xyz}
    """
    if not raw:
        return []

    parts = re.findall(r"\{[^}]+\}|[^\s]+", str(raw))
    paths: list[str] = []

    for part in parts:
        part = part.strip()
        if part.startswith("{") and part.endswith("}"):
            part = part[1:-1]
        if part:
            paths.append(part)

    return paths


def handle_dropped_file(state: dict, path: str) -> None:
    p = Path(path)
    ext = p.suffix.lower()

    if not p.exists():
        messagebox.showerror("Drag and Drop", f"File does not exist:\n{path}")
        return

    if ext in SUPPORTED_PROJECT_EXTS:
        fn = state.get("open_project_file")
        if callable(fn):
            fn(str(p))
        else:
            messagebox.showerror("Drag and Drop", "Project loader is not available.")
        return

    if ext in SUPPORTED_STRUCTURE_EXTS:
        fn = state.get("load_structure_file")
        if callable(fn):
            fn(str(p))
        else:
            messagebox.showerror("Drag and Drop", "Structure loader is not available.")
        return

    messagebox.showwarning(
        "Unsupported file",
        "Supported drag-and-drop files are:\n"
        "*.pcsp\n"
        "*.xyz, *.out, *.log",
    )


def enable_main_window_dnd(state: dict) -> bool:
    """
    Enable drag-and-drop support for the main window.

    Requires the root window to have been created with tkinterdnd2.TkinterDnD.Tk().
    Returns True only if at least one widget was successfully registered.
    """
    try:
        from tkinterdnd2 import DND_FILES
    except Exception:
        return False

    root = state.get("root")
    if root is None:
        return False

    def _on_drop(event):
        paths = _split_drop_files(getattr(event, "data", ""))
        if not paths:
            return

        # Keep behavior unambiguous: process only the first dropped file.
        if len(paths) > 1:
            try:
                messagebox.showinfo(
                    "Drag and Drop",
                    "Multiple files were dropped. Only the first file will be opened.",
                )
            except Exception:
                pass

        handle_dropped_file(state, paths[0])

    targets = [
        root,
        state.get("center_frame"),
        state.get("right_frame"),
        state.get("plots_nb"),
        state.get("tree"),
        state.get("input_frame"),
    ]

    ok = False
    for widget in targets:
        if widget is None:
            continue
        try:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", _on_drop)
            ok = True
        except Exception:
            pass

    return ok
