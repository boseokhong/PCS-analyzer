# ui/about_window.py

import tkinter as tk
from tkinter import ttk
import webbrowser


def open_about_window(state: dict):
    root = state["root"]
    win = state.get("about_window")

    if win is not None:
        try:
            if win.winfo_exists():
                win.lift()
                win.focus_force()
                return
        except Exception:
            pass

    win = tk.Toplevel(root)
    win.title("About PCS Analyzer")
    win.geometry("500x380")
    win.resizable(False, False)
    state["about_window"] = win

    outer = ttk.Frame(win, padding=14)
    outer.pack(fill="both", expand=True)

    app_bg = getattr(state["root"], "_app_bg", "#F5F6FA")
    variant = state.get("app_settings", {}).get("theme_variant", "light")
    link_fg = "#1a73e8" if variant == "light" else "#7fb2ff"

    def make_link(parent, text, url, pady=(0, 2)):
        lbl = tk.Label(
            parent,
            text=text,
            fg=link_fg,
            bg=app_bg,
            cursor="hand2",
            anchor="w",
            justify="left",
            bd=0,
            relief="flat",
            font=("Segoe UI", 9, "underline"),
        )
        lbl.pack(anchor="w", pady=pady)
        lbl.bind("<Button-1>", lambda e: webbrowser.open(url))
        return lbl

    ttk.Label(
        outer,
        text="PCS Analyzer",
        font=("Segoe UI", 12, "bold"),
    ).pack(anchor="w", pady=(0, 4))

    ttk.Label(
        outer,
        text="Version 1.3.3",
    ).pack(anchor="w", pady=(0, 10))

    body = (
        "Python-based desktop application for the analysis, visualization,\n"
        "and fitting of pseudocontact chemical shifts (PCS).\n\n"
    )

    ttk.Label(
        outer,
        text=body,
        justify="left",
    ).pack(anchor="w")

    ttk.Separator(outer).pack(fill="x", pady=10)

    ttk.Label(
        outer,
        text="Links",
        font=("Segoe UI", 10, "bold"),
    ).pack(anchor="w", pady=(0, 4))

    make_link(
        outer,
        "DOI: 10.5281/zenodo.18752129",
        "https://doi.org/10.5281/zenodo.18752129",
        pady=(0, 2),
    )

    make_link(
        outer,
        "GitHub: github.com/boseokhong/PCS-analyzer",
        "https://github.com/boseokhong/PCS-analyzer",
        pady=(0, 8),
    )

    ttk.Label(
        outer,
        text="Rights",
        font=("Segoe UI", 10, "bold"),
    ).pack(anchor="w", pady=(0, 4))

    ttk.Label(
        outer,
        text="Copyright (c) 2026 Boseok Hong",
        justify="left",
    ).pack(anchor="w", pady=(0, 2))

    ttk.Label(
        outer,
        text="Licensed under the BSD 3-Clause License.",
        justify="left",
    ).pack(anchor="w")

    ttk.Button(
        outer,
        text="Close",
        command=win.destroy,
    ).pack(anchor="e", pady=(14, 0))

    def _on_close():
        try:
            win.destroy()
        finally:
            state["about_window"] = None

    win.protocol("WM_DELETE_WINDOW", _on_close)