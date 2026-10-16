"""Pinned Control Panel tool customization."""

from __future__ import annotations

from ui.widget_style import ui_font_name
from ui.style import get_app_fonts
import tkinter as tk
from tkinter import ttk

from logic.control_panel_layout import DEFAULT_PINNED_TOOLS
from plugin_system.tool_registry import ensure_tool_registry


def open_customize_tools_window(state: dict) -> None:
    existing = state.get("customize_tools_window")
    if existing is not None:
        try:
            if existing.winfo_exists():
                existing.lift()
                return
        except Exception:
            pass

    top = tk.Toplevel(state["root"])
    state["customize_tools_window"] = top
    top.title("Customize Tools")
    top.geometry("480x520")

    outer = ttk.Frame(top, padding=10)
    outer.pack(fill="both", expand=True)
    search_var = tk.StringVar()
    ttk.Label(outer, text="Search:").pack(anchor="w")
    search = ttk.Entry(outer, textvariable=search_var)
    search.pack(fill="x", pady=(0, 8))

    list_frame = ttk.Frame(outer)
    list_frame.pack(fill="both", expand=True)
    canvas = tk.Canvas(list_frame, highlightthickness=0)
    scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
    content = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=content, anchor="nw")
    content.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    registry = ensure_tool_registry(state)
    preferences = state["app_settings"]["control_panel"]
    variables = {}
    known_ids = {tool.id for tool in registry.available()}
    pinned_order = [tool_id for tool_id in preferences["pinned_tools"] if tool_id in known_ids]

    order_frame = ttk.LabelFrame(outer, text="Pinned order", padding=6)
    order_frame.pack(fill="x", pady=(8, 0))
    order_list = tk.Listbox(order_frame, height=5, exportselection=False)
    order_list.pack(side="left", fill="x", expand=True)
    order_buttons = ttk.Frame(order_frame)
    order_buttons.pack(side="right", padx=(6, 0))

    def _render_order():
        order_list.delete(0, "end")
        for tool_id in pinned_order:
            tool = registry.get(tool_id)
            if tool is not None:
                order_list.insert("end", tool.label)

    def _sync_order():
        selected = {tool_id for tool_id, variable in variables.items() if variable.get()}
        pinned_order[:] = [tool_id for tool_id in pinned_order if tool_id in selected]
        pinned_order.extend(
            tool.id for tool in registry.available()
            if tool.id in selected and tool.id not in pinned_order
        )
        _render_order()

    def _move(delta):
        selection = order_list.curselection()
        if not selection:
            return
        index = selection[0]
        target = max(0, min(index + delta, len(pinned_order) - 1))
        if target == index:
            return
        pinned_order[index], pinned_order[target] = pinned_order[target], pinned_order[index]
        _render_order()
        order_list.selection_set(target)

    ttk.Button(order_buttons, text="Up", width=6, command=lambda: _move(-1)).pack(pady=(0, 3))
    ttk.Button(order_buttons, text="Down", width=6, command=lambda: _move(1)).pack()

    def render(*_args):
        for child in content.winfo_children():
            child.destroy()
        query = search_var.get().strip().lower()
        last_category = None
        for tool in registry.available():
            if query and query not in tool.label.lower() and query not in tool.category.lower():
                continue
            if tool.category != last_category:
                ttk.Label(content, text=tool.category, font=ui_font_name(content, "ui_bold")).pack(anchor="w", pady=(7, 2))
                last_category = tool.category
            variable = variables.setdefault(
                tool.id, tk.BooleanVar(value=tool.id in preferences["pinned_tools"])
            )
            ttk.Checkbutton(content, text=tool.label, variable=variable, command=_sync_order).pack(anchor="w", padx=(10, 0))

    def apply():
        _sync_order()
        preferences["pinned_tools"] = list(pinned_order)
        state.get("refresh_control_panel_tools", lambda: None)()
        state.get("save_app_preferences", lambda: None)()

    def reset():
        for tool_id, variable in variables.items():
            variable.set(tool_id in DEFAULT_PINNED_TOOLS)
        pinned_order[:] = [tool_id for tool_id in DEFAULT_PINNED_TOOLS if tool_id in known_ids]
        _render_order()
        apply()

    search_var.trace_add("write", render)
    render()
    _render_order()
    buttons = ttk.Frame(outer)
    buttons.pack(fill="x", pady=(8, 0))
    ttk.Button(buttons, text="Reset defaults", command=reset).pack(side="left")
    ttk.Button(buttons, text="Apply", command=apply).pack(side="right", padx=(4, 0))
    ttk.Button(buttons, text="Close", command=lambda: (apply(), top.destroy())).pack(side="right")
    top.protocol("WM_DELETE_WINDOW", lambda: (apply(), top.destroy()))


__all__ = ["open_customize_tools_window"]
