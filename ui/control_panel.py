"""Scrollable, independently collapsible PCS Control Panel."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from logic.control_panel_layout import (
    default_control_panel_preferences,
    normalize_control_panel_preferences,
    reorder_sections,
)
from ui.control_panel_style import (
    ControlPanelMetrics,
    configure_control_panel_styles,
    control_panel_hierarchy_fonts,
    control_panel_palette,
)
from ui.tooltips import ToolTip
from ui.pane_gesture import (
    PANEL_SNAP_THRESHOLD_RATIO, control_panel_drag_should_toggle,
    control_panel_release_action,
)


SECTION_LABELS = {
    "tensor": "Tensor",
    "orientation": "Orientation",
    "susceptibility": "Susceptibility details",
    "symmetry": "Symmetry averaging",
    "elements": "Element visibility",
    "plot_display": "Plot & display",
    "tools": "Quick tools",
}


class _Section:
    def __init__(self, panel, section_id: str, label: str) -> None:
        self.panel = panel
        self.section_id = section_id
        metrics = panel.metrics
        self.frame = ttk.Frame(panel.inner, style="ControlPanel.Section.TFrame")
        self.header = ttk.Frame(
            self.frame,
            style="ControlPanel.Header.TFrame",
            padding=(metrics.header_x, metrics.header_y),
        )
        self.header.pack(fill="x")
        self.header.columnconfigure(2, weight=1)
        self.handle = ttk.Label(
            self.header,
            text="⠿",
            cursor="sb_v_double_arrow",
            style="ControlPanel.Handle.TLabel",
        )
        self.handle.grid(row=0, column=0, sticky="w", padx=(0, metrics.column_gap))
        self.toggle = ttk.Button(
            self.header,
            width=2,
            style="ControlPanel.Chevron.TButton",
            command=self.toggle_expanded,
        )
        self.toggle.grid(row=0, column=1, sticky="w")
        self.title = ttk.Label(
            self.header,
            style="ControlPanel.Header.TLabel",
            font=panel.hierarchy_fonts["header"],
            cursor="hand2",
        )
        self.title.grid(row=0, column=2, sticky="ew", padx=(metrics.column_gap, 0))
        self.actions = ttk.Frame(self.header, style="ControlPanel.Header.TFrame")
        self.actions.grid(row=0, column=3, sticky="e", padx=(metrics.column_gap, 0))
        self.body = ttk.Frame(
            self.frame,
            style="ControlPanel.Body.TFrame",
            padding=(metrics.body_x, metrics.body_y),
        )

        self.handle.bind("<ButtonPress-1>", self._drag_start)
        self.handle.bind("<B1-Motion>", self._drag_motion)
        self.handle.bind("<ButtonRelease-1>", self._drag_end)
        self.header.bind("<Button-1>", lambda _event: self.toggle_expanded())
        self.title.bind("<Button-1>", lambda _event: self.toggle_expanded())
        self.set_expanded(section_id in panel.preferences["expanded"], persist=False)
        self.label = label

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = str(value)
        self._update_header()

    def _update_header(self):
        indicator = "▾" if self.expanded else "▸"
        self.toggle.configure(text=indicator)
        self.title.configure(text=self._label)

    def set_expanded(self, expanded: bool, *, persist: bool = True) -> None:
        self.expanded = bool(expanded)
        if self.expanded:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()
        if hasattr(self, "_label"):
            self._update_header()
        if persist:
            self.panel._section_expansion_changed(self.section_id, self.expanded)

    def toggle_expanded(self) -> None:
        self.set_expanded(not self.expanded)

    def _drag_start(self, event) -> None:
        self.panel.begin_drag(self.section_id, event.y_root)

    def _drag_motion(self, event) -> None:
        self.panel.drag_to(event.y_root)

    def _drag_end(self, event) -> None:
        self.panel.end_drag(event.y_root)


class ScrollableControlPanel:
    """Canvas-backed section host with persistent collapse and ordering state."""

    COLLAPSE_SETTING_KEY = "control_panel_collapsed"
    DOCK_SIDE_SETTING_KEY = "control_panel_side"
    DEFAULT_DOCK_SIDE = "right"
    SNAP_THRESHOLD_RATIO = PANEL_SNAP_THRESHOLD_RATIO

    def __init__(self, parent, state: dict) -> None:
        self.state = state
        self.host = parent
        self.main_container = parent.master
        self.center_frame = state.get("center_frame")
        self._host_pack_options = self._safe_pack_options(parent.pack_info())
        self._host_visible = True
        self._gesture_start_x = None
        self._gesture_armed = False
        self._gutter_hover = False

        configure_control_panel_styles(parent)
        self.hierarchy_fonts = control_panel_hierarchy_fonts(parent)
        self.metrics = ControlPanelMetrics.from_widget(parent)
        self.palette = control_panel_palette(parent)
        settings = state.setdefault("app_settings", {})
        self.preferences = normalize_control_panel_preferences(
            settings.get("control_panel")
        )
        settings["control_panel"] = self.preferences
        self.collapsed = bool(settings.get(self.COLLAPSE_SETTING_KEY, False))
        self.dock_side = self._normalize_dock_side(
            settings.get(self.DOCK_SIDE_SETTING_KEY, self.DEFAULT_DOCK_SIDE)
        )
        settings[self.DOCK_SIDE_SETTING_KEY] = self.dock_side

        self.frame = ttk.Frame(parent, style="ControlPanel.Root.TFrame")
        self.frame.pack(
            fill="both",
            expand=True,
            padx=(self.metrics.outer_x, self.metrics.outer_x),
            pady=self.metrics.outer_y,
        )
        title = ttk.Frame(self.frame, style="ControlPanel.Root.TFrame")
        title.pack(fill="x", pady=(0, self.metrics.section_gap))
        ttk.Label(title, text=" Control Panel", style="ControlPanel.Title.TLabel", font=self.hierarchy_fonts["header"]).pack(side="left")
        icon_manager = state.get("icons")
        reset_icon = icon_manager.get("reset", size=20) if icon_manager is not None else ""
        self.dock_button = ttk.Button(
            title,
            text="◧",
            width=2,
            style="ControlPanel.TitleUtility.TButton",
            command=self.toggle_dock_side,
        )
        self.dock_button.pack(side="right")
        self._dock_tooltip = ToolTip(self.dock_button, "")
        self._update_dock_button()
        ttk.Button(
            title,
            text="Reset layout",
            image=reset_icon,
            compound="left",
            style="ControlPanel.Secondary.TButton",
            command=self.reset_layout,
        ).pack(side="right", padx=(0, self.metrics.action_gap))

        body = ttk.Frame(self.frame, style="ControlPanel.Root.TFrame")
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            body,
            highlightthickness=0,
            borderwidth=0,
            width=self.metrics.panel_width,
            background=self.palette["panel"],
        )
        self.scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.inner = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._update_scrollregion)
        self.canvas.bind("<Configure>", self._resize_inner)
        self.canvas.bind("<MouseWheel>", self._mousewheel)
        self.inner.bind("<MouseWheel>", self._mousewheel)

        self.sections = {
            section_id: _Section(self, section_id, SECTION_LABELS[section_id])
            for section_id in self.preferences["section_order"]
        }
        self.drop_indicator = ttk.Separator(self.inner, orient="horizontal")
        self._dragged_id = None
        self._drop_index = None
        self._pack_sections()

        self._apply_dock_side(self.dock_side, persist=False)
        self._create_boundary_gutter()
        self.main_container.bind("<Configure>", self._container_changed, add="+")
        self.host.bind("<Configure>", self._container_changed, add="+")
        self.frame.after_idle(self._install_scroll_bindings)
        self.frame.after_idle(self._restore_host_visibility)

        state["toggle_control_panel"] = self.toggle_collapsed
        state["collapse_control_panel"] = self.collapse
        state["expand_control_panel"] = self.expand
        state["toggle_control_panel_side"] = self.toggle_dock_side
        state["set_control_panel_side"] = self.set_dock_side

    @staticmethod
    def _safe_pack_options(info: dict) -> dict:
        allowed = {"anchor", "expand", "fill", "ipadx", "ipady", "padx", "pady", "side"}
        return {key: value for key, value in info.items() if key in allowed}

    @classmethod
    def _normalize_dock_side(cls, side: str) -> str:
        return "left" if str(side).strip().lower() == "left" else cls.DEFAULT_DOCK_SIDE

    def _pack_host_for_dock_side(self) -> None:
        """Reserve the inspector width before the expanding workspace."""
        options = dict(self._host_pack_options)
        options["side"] = self.dock_side
        center = self.center_frame
        try:
            if center is not None and center.winfo_exists() and center.winfo_manager() == "pack":
                self.host.pack(**options, before=center)
            else:
                self.host.pack(**options)
        except tk.TclError:
            self.host.pack(**options)

    def _update_dock_button(self) -> None:
        if not hasattr(self, "dock_button"):
            return
        target = "left" if self.dock_side == "right" else "right"
        self.dock_button.configure(text="◧" if target == "left" else "◨")
        if hasattr(self, "_dock_tooltip"):
            self._dock_tooltip.update_text(f"Move Control Panel to the {target}")

    def _apply_dock_side(self, side: str, *, persist: bool = True) -> None:
        side = self._normalize_dock_side(side)
        self.dock_side = side
        self.state.setdefault("app_settings", {})[self.DOCK_SIDE_SETTING_KEY] = side

        if self._host_visible and self.host.winfo_manager() == "pack":
            self.host.pack_forget()
            self._pack_host_for_dock_side()

        self._update_dock_button()
        if persist:
            callback = self.state.get("save_app_preferences")
            if callable(callback):
                callback()
        if hasattr(self, "boundary_gutter"):
            self.main_container.after_idle(self._position_boundary_gutter)

    def set_dock_side(self, side: str) -> None:
        self._apply_dock_side(side)

    def toggle_dock_side(self) -> None:
        target = "left" if self.dock_side == "right" else "right"
        self._apply_dock_side(target)

    def section(self, section_id: str):
        return self.sections[section_id].body

    def section_actions(self, section_id: str):
        return self.sections[section_id].actions

    def refresh_styles(self) -> None:
        """Refresh scoped colors/metrics after an application theme/font change."""
        configure_control_panel_styles(self.frame)
        self.hierarchy_fonts = control_panel_hierarchy_fonts(self.frame)
        self.metrics = ControlPanelMetrics.from_widget(self.frame)
        self.palette = control_panel_palette(self.frame)
        self.canvas.configure(
            width=self.metrics.panel_width,
            background=self.palette["panel"],
        )
        self._refresh_gutter_theme()
        for section in self.sections.values():
            section.header.configure(
                padding=(self.metrics.header_x, self.metrics.header_y)
            )
            section.body.configure(
                padding=(self.metrics.body_x, self.metrics.body_y)
            )
        self._pack_sections()
        refresh_elements = self.state.get("refresh_element_visibility_styles")
        if callable(refresh_elements):
            refresh_elements()
        self.frame.after_idle(self._position_boundary_gutter)

    def _create_boundary_gutter(self) -> None:
        root = self.main_container.winfo_toplevel()
        bg = getattr(root, "_app_bg", self.palette["panel"])
        self.boundary_gutter = tk.Canvas(
            self.main_container,
            width=self.metrics.gutter_hit,
            highlightthickness=0,
            borderwidth=0,
            background=bg,
            cursor="sb_h_double_arrow",
        )
        self.boundary_gutter.bind("<Enter>", self._gutter_enter, add="+")
        self.boundary_gutter.bind("<Leave>", self._gutter_leave, add="+")
        self.boundary_gutter.bind("<ButtonPress-1>", self._gutter_press, add="+")
        self.boundary_gutter.bind("<B1-Motion>", self._gutter_drag, add="+")
        self.boundary_gutter.bind("<ButtonRelease-1>", self._gutter_release, add="+")
        self.boundary_gutter.bind("<Configure>", lambda _event: self._draw_gutter(), add="+")
        self._position_boundary_gutter()

    def _refresh_gutter_theme(self) -> None:
        if not hasattr(self, "boundary_gutter"):
            return
        root = self.main_container.winfo_toplevel()
        bg = getattr(root, "_app_bg", self.palette["panel"])
        self.boundary_gutter.configure(background=bg)
        self._draw_gutter()

    def _container_changed(self, _event=None) -> None:
        self.main_container.after_idle(self._position_boundary_gutter)

    def _position_boundary_gutter(self) -> None:
        if not hasattr(self, "boundary_gutter") or not self.main_container.winfo_exists():
            return
        try:
            container_width = max(1, self.main_container.winfo_width())
            hit = self.metrics.gutter_hit
            if self.collapsed or not self._host_visible:
                x = 0 if self.dock_side == "left" else max(0, container_width - hit)
            elif self.dock_side == "left":
                x = self.host.winfo_x() + self.host.winfo_width() - hit // 2
            else:
                x = self.host.winfo_x() - hit // 2
            x = max(0, min(container_width - hit, int(x)))
            self.boundary_gutter.place(x=x, y=0, width=hit, relheight=1.0)
            self.boundary_gutter.lift()
            self._draw_gutter()
        except tk.TclError:
            return

    def _draw_gutter(self) -> None:
        if not hasattr(self, "boundary_gutter"):
            return
        gutter = self.boundary_gutter
        try:
            gutter.delete("all")
            width = max(1, gutter.winfo_width())
            height = max(1, gutter.winfo_height())
            color = (
                self.palette["gutter_armed"]
                if self._gesture_armed
                else self.palette["gutter_hover"] if self._gutter_hover
                else self.palette["gutter_idle"]
            )
            x = width / 2.0
            gutter.create_line(
                x, 0, x, height,
                fill=color,
                width=1,
            )
        except tk.TclError:
            return

    def _gutter_enter(self, _event=None) -> None:
        self._gutter_hover = True
        self._draw_gutter()

    def _gutter_leave(self, _event=None) -> None:
        if self._gesture_start_x is None:
            self._gutter_hover = False
            self._gesture_armed = False
            self._draw_gutter()

    def _gutter_press(self, event) -> None:
        self._gesture_start_x = int(event.x_root)
        self._gesture_armed = False
        try:
            self.boundary_gutter.grab_set()
        except tk.TclError:
            pass
        self._draw_gutter()

    def _gutter_drag(self, event) -> None:
        if self._gesture_start_x is None:
            return
        delta = int(event.x_root) - self._gesture_start_x
        self._gesture_armed = control_panel_drag_should_toggle(
            delta,
            self.metrics.panel_width,
            collapsed=self.collapsed,
            threshold_ratio=self.SNAP_THRESHOLD_RATIO,
            dock_side=self.dock_side,
        )
        self._draw_gutter()

    def _gutter_release(self, event) -> None:
        if self._gesture_start_x is None:
            return
        delta = int(event.x_root) - self._gesture_start_x
        was_collapsed = self.collapsed
        action = control_panel_release_action(
            delta,
            self.metrics.panel_width,
            collapsed=was_collapsed,
            threshold_ratio=self.SNAP_THRESHOLD_RATIO,
            dock_side=self.dock_side,
            scale=self.metrics.scale,
        )
        self._gesture_start_x = None
        self._gesture_armed = False
        try:
            self.boundary_gutter.grab_release()
        except tk.TclError:
            pass
        if action == "toggle":
            self.toggle_collapsed()
        elif action == "expand":
            # The collapsed-edge affordance also works as a simple restore click.
            self.expand()
        self._draw_gutter()

    def _restore_host_visibility(self) -> None:
        if self.collapsed:
            self._set_host_visible(False, persist=False)
        else:
            self._set_host_visible(True, persist=False)
        self._position_boundary_gutter()

    def _set_host_visible(self, visible: bool, *, persist: bool = True) -> None:
        visible = bool(visible)
        if visible == self._host_visible and self.host.winfo_manager() == ("pack" if visible else ""):
            self.collapsed = not visible
            if persist:
                self._persist_collapsed_state()
            self._position_boundary_gutter()
            return

        if visible:
            self._pack_host_for_dock_side()
            self._host_visible = True
            self.collapsed = False
        else:
            try:
                focus = self.host.focus_get()
                if focus is not None and str(focus).startswith(str(self.host)):
                    self.main_container.focus_set()
            except Exception:
                pass
            self.host.pack_forget()
            self._host_visible = False
            self.collapsed = True

        if persist:
            self._persist_collapsed_state()
        self.main_container.after_idle(self._position_boundary_gutter)

    def collapse(self) -> None:
        self._set_host_visible(False)

    def expand(self) -> None:
        self._set_host_visible(True)

    def toggle_collapsed(self) -> None:
        self._set_host_visible(self.collapsed)

    def _persist_collapsed_state(self) -> None:
        self.state.setdefault("app_settings", {})[self.COLLAPSE_SETTING_KEY] = bool(self.collapsed)
        callback = self.state.get("save_app_preferences")
        if callable(callback):
            callback()

    def _update_scrollregion(self, _event=None) -> None:
        # Defer until Tk has resolved section geometry.  A direct bbox("all")
        # update can preserve an old y-offset after the layout becomes shorter,
        # which makes blank canvas margin appear scrollable.
        self.frame.after_idle(self._sync_scrollregion)
        self.frame.after_idle(self._install_scroll_bindings)

    def _sync_scrollregion(self) -> None:
        if not self.canvas.winfo_exists():
            return
        try:
            viewport_w = max(1, int(self.canvas.winfo_width()))
            viewport_h = max(1, int(self.canvas.winfo_height()))
            content_h = max(0, int(self.inner.winfo_reqheight()))

            # Never expose scrollable space below the actual section content.
            # When content fits, make the scroll region exactly the viewport and
            # force the origin back to zero.
            region_h = max(viewport_h, content_h)
            self.canvas.configure(scrollregion=(0, 0, viewport_w, region_h))

            if content_h <= viewport_h:
                self.canvas.yview_moveto(0.0)
                return

            first, _last = self.canvas.yview()
            max_first = max(0.0, 1.0 - (viewport_h / float(content_h)))
            if first > max_first:
                self.canvas.yview_moveto(max_first)
            elif first < 0.0:
                self.canvas.yview_moveto(0.0)
        except tk.TclError:
            return

    def _install_scroll_bindings(self) -> None:
        def visit(widget):
            if not getattr(widget, "_pcs_panel_wheel_bound", False):
                widget.bind("<MouseWheel>", self._mousewheel, add="+")
                widget._pcs_panel_wheel_bound = True
            for child in widget.winfo_children():
                visit(child)
        visit(self.inner)

    def _resize_inner(self, event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)
        self.frame.after_idle(self._sync_scrollregion)

    def _mousewheel(self, event):
        if (
            event.widget.winfo_class() in {"Scale", "Entry", "TEntry", "TCombobox"}
            or getattr(event.widget, "_pcs_control_slider", False)
        ):
            return None
        try:
            if self.inner.winfo_reqheight() <= self.canvas.winfo_height():
                self.canvas.yview_moveto(0.0)
                return "break"
            first, last = self.canvas.yview()
            direction = int(-event.delta / 120)
            if (direction < 0 and first <= 0.0) or (direction > 0 and last >= 1.0):
                return "break"
            self.canvas.yview_scroll(direction, "units")
        except tk.TclError:
            pass
        return "break"

    def _pack_sections(self) -> None:
        self.drop_indicator.pack_forget()
        for section in self.sections.values():
            section.frame.pack_forget()
        for section_id in self.preferences["section_order"]:
            self.sections[section_id].frame.pack(
                fill="x", pady=(0, self.metrics.section_gap)
            )
        self._update_scrollregion()

    def _section_expansion_changed(self, section_id: str, expanded: bool) -> None:
        values = list(self.preferences["expanded"])
        if expanded and section_id not in values:
            values.append(section_id)
        elif not expanded and section_id in values:
            values.remove(section_id)
        self.preferences["expanded"] = values
        self._persist()

    def begin_drag(self, section_id: str, _root_y: int) -> None:
        self._dragged_id = section_id
        self._drop_index = self.preferences["section_order"].index(section_id)

    def _index_for_root_y(self, root_y: int) -> int:
        order = [value for value in self.preferences["section_order"] if value != self._dragged_id]
        for index, section_id in enumerate(order):
            frame = self.sections[section_id].frame
            midpoint = frame.winfo_rooty() + frame.winfo_height() / 2
            if root_y < midpoint:
                return index
        return len(order)

    def drag_to(self, root_y: int) -> None:
        if self._dragged_id is None:
            return
        self._drop_index = self._index_for_root_y(root_y)
        order = [value for value in self.preferences["section_order"] if value != self._dragged_id]
        self.drop_indicator.pack_forget()
        before = None if self._drop_index >= len(order) else self.sections[order[self._drop_index]].frame
        if before is None:
            self.drop_indicator.pack(fill="x", pady=2)
        else:
            self.drop_indicator.pack(fill="x", pady=2, before=before)

    def end_drag(self, root_y: int) -> None:
        if self._dragged_id is None:
            return
        target = self._index_for_root_y(root_y)
        self.preferences["section_order"] = reorder_sections(
            self.preferences["section_order"], self._dragged_id, target
        )
        self._dragged_id = None
        self._drop_index = None
        self._pack_sections()
        refresh_tools = self.state.get("refresh_control_panel_tools")
        if callable(refresh_tools):
            refresh_tools()
        self._persist()

    def reset_layout(self) -> None:
        self.preferences.clear()
        self.preferences.update(default_control_panel_preferences())
        for section_id, section in self.sections.items():
            section.set_expanded(
                section_id in self.preferences["expanded"], persist=False
            )
        self._pack_sections()
        self.canvas.after_idle(lambda: self.canvas.yview_moveto(0.0))
        refresh_tools = self.state.get("refresh_control_panel_tools")
        if callable(refresh_tools):
            refresh_tools()
        self._persist()

    def _persist(self) -> None:
        self.state.setdefault("app_settings", {})["control_panel"] = self.preferences
        callback = self.state.get("save_app_preferences")
        if callable(callback):
            callback()


__all__ = [
    "SECTION_LABELS",
    "ScrollableControlPanel",
    "control_panel_drag_should_toggle",
]
