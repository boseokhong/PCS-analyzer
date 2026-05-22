# ui/update_window.py

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
import webbrowser

from app_version import (
    APP_NAME,
    APP_VERSION,
    GITHUB_LATEST_RELEASE_API,
    GITHUB_RELEASES_URL,
)
from logic.update_checker import check_latest_release


def open_update_result_window(state: dict, result):
    root = state["root"]

    # Avoid opening multiple update result windows.
    old = state.get("update_window")
    if old is not None:
        try:
            if old.winfo_exists():
                old.destroy()
        except Exception:
            pass

    win = tk.Toplevel(root)
    win.title("Check for Updates")
    win.geometry("460x200")
    win.resizable(False, False)
    state["update_window"] = win

    outer = ttk.Frame(win, padding=14)
    outer.pack(fill="both", expand=True)

    # Layout:
    # row 0 = message area, expandable
    # row 1 = bottom button bar, fixed
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(0, weight=1)
    outer.rowconfigure(1, weight=0)

    def _on_close():
        try:
            win.destroy()
        finally:
            state["update_window"] = None

    # ----------------------------
    # Build message
    # ----------------------------
    if result.error:
        message = (
            "Could not check for updates.\n\n"
            f"Current version: {result.current_version}\n"
            f"Error: {result.error}"
        )

    elif result.status == "update_available":
        message = (
            f"A newer version of {APP_NAME} is available.\n\n"
            f"Current version: {result.current_version}\n"
            f"Latest version: {result.latest_version}"
        )

    elif result.status == "up_to_date":
        message = (
            f"{APP_NAME} is up to date.\n\n"
            f"Current version: {result.current_version}\n"
            f"Latest version: {result.latest_version}"
        )

    elif result.status == "ahead_of_release":
        message = (
            "This build is newer than the latest public release.\n\n"
            f"Current version: {result.current_version}\n"
            f"Latest public release: {result.latest_version}\n\n"
            "This can happen when you are using a development or pre-release build."
        )

    else:
        message = (
            "Could not check for updates.\n\n"
            f"Current version: {result.current_version}\n"
            f"Error: {result.error or 'Unknown update status.'}"
        )

    msg_frame = ttk.Frame(outer)
    msg_frame.grid(row=0, column=0, sticky="nsew")

    ttk.Label(
        msg_frame,
        text=message,
        justify="left",
        wraplength=440,
    ).pack(anchor="nw", fill="x")

    # ----------------------------
    # Fixed bottom button bar
    # ----------------------------
    btns = ttk.Frame(outer)
    btns.grid(row=1, column=0, sticky="ew", pady=(14, 0))
    btns.columnconfigure(0, weight=1)

    right_btns = ttk.Frame(btns)
    right_btns.grid(row=0, column=1, sticky="e")

    if result.status == "update_available":
        ttk.Button(
            right_btns,
            text="Open Update Page",
            command=lambda: webbrowser.open(result.release_url),
        ).pack(side="left", padx=(0, 6))

    elif result.status == "ahead_of_release":
        ttk.Button(
            right_btns,
            text="Open Release Page",
            command=lambda: webbrowser.open(result.release_url),
        ).pack(side="left", padx=(0, 6))

    elif result.error:
        ttk.Button(
            right_btns,
            text="Open Release Page",
            command=lambda: webbrowser.open(result.release_url),
        ).pack(side="left", padx=(0, 6))

    ttk.Button(
        right_btns,
        text="Close",
        command=_on_close,
    ).pack(side="left")

    win.protocol("WM_DELETE_WINDOW", _on_close)

def check_for_updates_ui(state: dict):
    root = state["root"]

    def worker():
        result = check_latest_release(
            current_version=APP_VERSION,
            api_url=GITHUB_LATEST_RELEASE_API,
            fallback_release_url=GITHUB_RELEASES_URL,
        )

        root.after(0, lambda: open_update_result_window(state, result))

    threading.Thread(target=worker, daemon=True).start()