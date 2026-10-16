"""Compact application menu row and document edit-history presentation."""
import tkinter as tk
from tkinter import ttk
import sys

from logic.document_edit_history import PCS_KEYS, history_for, move_history
from logic.pcs_document import (active_document, capture_active_document,
                                refresh_follow_active_surfaces, suspend_document_tracking)
from logic.pcs_model_state import update_canonical_tensor, update_prediction_model
from logic.pcs_prediction_model import get_prediction_model, install_prediction_model
from logic.pcs_tensor_state import apply_canonical_tensor_to_legacy
from ui.style import get_app_fonts
from ui.theme_palette import presentation_palette, history_palette, theme_id, theme_mode
from ui.ribbon.icon_service import tk_scale_factor, nearest_packaged_icon_size
from ui.tooltips import ToolTip


def _popup_bounds(anchor):
    """Use the anchor's monitor work area, including negative desktop origins."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class MonitorInfo(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        monitor_from_point = user32.MonitorFromPoint
        monitor_from_point.argtypes = [wintypes.POINT, wintypes.DWORD]
        monitor_from_point.restype = wintypes.HANDLE
        get_monitor_info = user32.GetMonitorInfoW
        get_monitor_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(MonitorInfo)]
        get_monitor_info.restype = wintypes.BOOL
        point = wintypes.POINT(anchor.winfo_rootx(),
                              anchor.winfo_rooty() + anchor.winfo_height())
        monitor = monitor_from_point(point, 2)  # nearest monitor
        info = MonitorInfo()
        info.cbSize = ctypes.sizeof(info)
        if get_monitor_info(monitor, ctypes.byref(info)):
            work = info.rcWork
            return work.left, work.top, work.right, work.bottom
    left, top = anchor.winfo_vrootx(), anchor.winfo_vrooty()
    return left, top, left + anchor.winfo_vrootwidth(), top + anchor.winfo_vrootheight()


def _popup_position(anchor, width, height, bounds):
    left, top, right, bottom = bounds
    x = max(left, min(anchor.winfo_rootx(), right - width))
    y = anchor.winfo_rooty() + anchor.winfo_height()
    if y + height > bottom:
        y = anchor.winfo_rooty() - height
    return x, max(top, min(y, bottom - height))


def apply_history_move(state, direction, count=1):
    """Restore recorded values and refresh through the existing presentation owners."""
    finish_gesture = state.get("finish_parameter_gesture")
    if callable(finish_gesture):
        finish_gesture()
    check_pending = state.get("check_edit_history_pending_inputs")
    if callable(check_pending):
        check_pending()
    previous = state.get("pcs_tensor_state")
    previous_model = get_prediction_model(state)
    labels, touched = move_history(state, direction, count)
    target = state.get("pcs_tensor_state")
    target_model = get_prediction_model(state)
    model_changed = "pcs_prediction_model" in touched and target_model is not previous_model
    errors = []
    if touched.intersection(("x0", "y0", "z0", "metal_ref_id")):
        state.pop("structure_graph", None)
    with suspend_document_tracking(state):
        from ui.structure_history import project_structure_controls, prepare_structure_views
        project_structure_controls(state, state.pop("_edit_history_restored_controls", {}))
        if "current_structure_path" in touched:
            try:
                prepare_structure_views(state)
            except Exception as exc:
                errors.append(f"Structure presentation: {exc}")
        if touched.intersection(PCS_KEYS):
            from logic.nmr_delta_data_manager import recompute_delta_para
            recompute_delta_para(state)
        if "pcs_tensor_state" in touched and target is not None:
            state["pcs_tensor_state"] = previous
            try:
                changed = update_canonical_tensor(state, target, source=f"history:{direction}",
                                                 legacy_callback=state.get("update_graph"))
                if not changed:
                    apply_canonical_tensor_to_legacy(state, target)
                    if callable(state.get("update_graph")):
                        state["update_graph"]()
            except Exception as exc:
                # Values are already committed; retain the moved cursor on refresh failure.
                apply_canonical_tensor_to_legacy(state, target)
                errors.append(str(exc))
        elif not model_changed and callable(state.get("update_graph")):
            try:
                state["update_graph"]()
            except Exception as exc:
                errors.append(str(exc))
        if model_changed:
            # Mixed moves project the restored tensor first. The policy event
            # must never import stale legacy entry values during its redraw.
            install_prediction_model(state, previous_model)
            try:
                update_prediction_model(state, target_model, source=f"history:{direction}")
            except Exception as exc:
                errors.append(f"Prediction model: {exc}")
        for key, args in (("apply_symavg_to_state", (state,)), ("plot_cartesian", (state,)),
                          ("rh_refresh_table", ()), ("refresh_orientation_editor", ()),
                          ("refresh_fit_result_apply_ui", ()), ("refresh_fitting_commands", ()),
                          ("refresh_fit_result_validity_ui", ()), ("refresh_pcs_data_commands", ()),
                          ("refresh_prediction_model_ui", ()),
                          ("refresh_gax_rhombicity_commands", ()),
                          ("refresh_conformer_commands", ())):
            callback = state.get(key)
            if callable(callback):
                try:
                    callback(*args)
                except Exception as exc:
                    errors.append(f"{key}: {exc}")
    capture_search = state.get("capture_conformer_search_state")
    if callable(capture_search):
        capture_search()
    if "atom_data_raw" in touched or touched.intersection(PCS_KEYS) or "metal_ref_id" in touched:
        from logic.nmr_delta_data_manager import push_layers_to_nmr_if_open
        try:
            push_layers_to_nmr_if_open(state)
        except Exception as exc:
            errors.append(f"NMR compatibility: {exc}")
    capture_active_document(state)
    if "fit_history" in touched:
        # Fit History reads the document adapter, so capture restored records
        # before asking its presenters to read them.
        with suspend_document_tracking(state):
            for key in ("refresh_fit_history_ui", "refresh_last_fit_result_ui"):
                callback = state.get(key)
                if callable(callback):
                    try:
                        callback()
                    except Exception as exc:
                        errors.append(f"{key}: {exc}")
    recorded = state.get("edit_history_recorded")
    if callable(recorded):
        recorded()
    refresh_follow_active_surfaces(state, include_legacy=False)
    for key in ("refresh_document_tabs", "refresh_active_document_commands", "refresh_edit_history"):
        callback = state.get(key)
        if callable(callback):
            callback()
    if errors:
        state["messagebox"].showwarning("Edit history", "Values restored with refresh warnings:\n" + "\n".join(errors))
    return labels


class EditHistoryMenuBar(tk.Frame):
    """Existing menus and adjacent 30×24 buttons inside a 28-pixel menu row."""
    def __init__(self, root, state):
        super().__init__(root, borderwidth=0)
        self.state = state
        self.menu_buttons = []
        self.actions = {}
        self._history_hovered = None
        self._history_pressed = None
        self._history_paints = {}
        self._history_images = {}
        self.tooltips = []
        self.popup = None
        self.popup_previous_focus = None
        self.popup_document = None
        self._bindings = []
        self.projected_inputs = {}
        self.projected_document = None
        state["menus"] = {}
        for label in ("File", "View", "Modules", "Settings", "Help"):
            button = tk.Menubutton(self, text=label, underline=0,
                                   relief="flat", borderwidth=0, takefocus=True)
            # Tk's native Menubutton posting requires its menu to be a descendant.
            # Construct it here, before command presenters populate the same menu.
            menu = tk.Menu(button, tearoff=0)
            button.configure(menu=menu)
            state["menus"][label] = menu
            button.pack(side="left")
            self.menu_buttons.append(button)
            binding = root.bind(f"<Alt-{label[0].lower()}>",
                                lambda e, b=button, m=menu: self.post_menu(b, m), add="+")
            self._bindings.append((f"<Alt-{label[0].lower()}>", binding))
        self.separator = tk.Frame(self, width=1)
        self.separator.pack(side="left", fill="y", padx=(9, 9), pady=5)
        for direction in ("undo", "redo"):
            frame = tk.Frame(self, borderwidth=0, highlightthickness=1)
            frame.pack(side="left", pady=2)
            frame.pack_propagate(False)
            # Classic Windows Tk Buttons stipple disabled images even when the
            # supplied PNG is already grey. ttk preserves that exact image while
            # retaining a real disabled state and normal keyboard semantics.
            action = ttk.Button(frame, command=lambda d=direction: self.invoke(d),
                                style=f"{direction}.History.Action.TButton", padding=0, takefocus=True)
            arrow = tk.Button(frame, text="▾", command=lambda d=direction: self.show_history(d),
                              relief="flat", borderwidth=0, padx=0, pady=0, takefocus=True,
                              highlightthickness=1)
            self.actions[direction] = (frame, action, arrow)
            for widget in (frame, action, arrow):
                widget.bind("<Enter>", lambda e, d=direction: self._history_enter(d), add="+")
                widget.bind("<Leave>", lambda e, d=direction: self._history_leave(e, d), add="+")
            for button in (action, arrow):
                button.bind("<FocusIn>", lambda e, d=direction: self._paint_history(d), add="+")
                button.bind("<FocusOut>", lambda e, d=direction: self._paint_history(d), add="+")
                button.bind("<ButtonPress-1>", lambda e, d=direction: self._history_press(d, True), add="+")
                button.bind("<ButtonRelease-1>", lambda e, d=direction: self._history_press(d, False), add="+")
            self.tooltips.append(ToolTip(action, f"{direction.title()}"))
            self.tooltips.append(ToolTip(arrow, f"{direction.title()} history"))
        state["refresh_edit_history"] = self.refresh
        state["edit_history_recorded"] = self.record_projection
        state["check_edit_history_pending_inputs"] = self.check_pending_inputs
        state["is_history_input_unchanged"] = self.is_input_unchanged
        from ui.keyboard_shortcuts import SHORTCUTS
        for _, _, command, _, sequences in SHORTCUTS:
            if not command.startswith("history."):
                continue
            for sequence in sequences:
                identity = root.bind(sequence, lambda e, d=command.split(".")[1]: self.shortcut(e, d), add="+")
                self._bindings.append((sequence, identity))
        self.bind("<Destroy>", self.cleanup, add="+")
        self.refresh_styles()
        self.refresh()

    def post_menu(self, button, menu):
        menu.tk_popup(button.winfo_rootx(), button.winfo_rooty() + button.winfo_height())
        return "break"

    def refresh_styles(self):
        p = presentation_palette(theme_id(self))
        self.configure(background=p["field"])
        self.separator.configure(background=p["edge"])
        fonts = get_app_fonts(self)
        scale = tk_scale_factor(self)
        row_height = round(28 * scale)
        for b in self.menu_buttons:
            b.configure(background=p["field"], foreground=p["text"], activebackground=p["hover"],
                        activeforeground=p["text"], font=fonts["ui"], padx=7, pady=3)
            row_height = max(row_height, b.winfo_reqheight())
        self.configure(height=row_height)
        self.pack_propagate(False)
        colors = history_palette(theme_id(self))
        self._history_colors = colors
        self._history_paints.clear()
        style = ttk.Style(self)
        style.layout("History.Action.TButton", [
            ("Button.border", {"sticky": "nswe", "children": [
                ("Button.padding", {"sticky": "nswe", "children": [
                    ("Button.label", {"sticky": "nswe"})]})]})])
        style.configure("History.Action.TButton", background=p["field"],
                        borderwidth=0, relief="flat", padding=0, anchor="center")
        style.map("History.Action.TButton", background=[
            ("disabled", p["field"]), ("pressed", colors["pressed"]),
            ("active", colors["hover"])])
        for direction, (frame, action, arrow) in self.actions.items():
            bg, fg = colors[direction]
            width, height, drop = round(30 * scale), round(24 * scale), round(8 * scale)
            # Split an odd spare pixel between top/bottom at 125% DPI rather
            # than rounding both margins up and clipping the 24-pixel button.
            spare = max(0, row_height - height)
            frame.pack_configure(pady=(spare // 2, spare - spare // 2))
            frame.configure(width=width, height=height)
            action.place(x=0, y=0, width=width-drop-2, height=height-2)
            arrow.place(x=width-drop-2, y=0, width=drop, height=height-2)
            icon = f"history_{direction}" + ("_dark" if theme_mode(theme_id(self)) == "dark" else "")
            size = nearest_packaged_icon_size(16, scale)
            for enabled in (True, False):
                self._history_images[direction, enabled] = self.state["icons"].get(
                    icon if enabled else icon + "_disabled", size=size)
            # An explicit disabled image is essential: ttk also stipples when
            # the same image is used for both normal and disabled states.
            normal_image = self._history_images[direction, True]
            disabled_image = self._history_images[direction, False] or normal_image
            action.configure(image=(normal_image, "disabled", disabled_image) if normal_image else "")
            arrow.configure(background=bg, foreground=fg, activebackground=p["hover"],
                            activeforeground=fg, disabledforeground=colors["disabled_foreground"],
                            highlightbackground=bg, highlightcolor=p["accent"], font=fonts["ui_small"])
            self._paint_history(direction)

    def _history_enter(self, direction):
        previous = self._history_hovered
        self._history_hovered = direction
        if previous is not None and previous != direction:
            self._paint_history(previous)
        self._paint_history(direction)

    def _history_leave(self, event, direction):
        frame = self.actions[direction][0]
        # Crossing between the icon and its dropdown stays inside one button.
        if (frame.winfo_rootx() <= event.x_root < frame.winfo_rootx() + frame.winfo_width()
                and frame.winfo_rooty() <= event.y_root < frame.winfo_rooty() + frame.winfo_height()):
            return
        if self._history_hovered == direction:
            self._history_hovered = None
        self._paint_history(direction)

    def _history_press(self, direction, pressed):
        self._history_pressed = direction if pressed else None
        self._paint_history(direction)

    def _paint_history(self, direction):
        """Paint existing buttons without changing focus, grabs or commands."""
        colors = self._history_colors
        frame, action, arrow = self.actions[direction]
        enabled = str(action.cget("state")) != "disabled"
        hovered = enabled and self._history_hovered == direction
        pressed = enabled and self._history_pressed == direction
        focused = enabled and self.focus_get() in (action, arrow)
        signature = (enabled, hovered, pressed, focused)
        if self._history_paints.get(direction) == signature:
            return
        self._history_paints[direction] = signature
        bg, fg = colors[direction]
        face = colors["pressed"] if pressed else colors["hover"] if hovered else bg
        edge = colors["focus"] if focused else colors["edge"] if hovered or pressed else bg
        frame.configure(background=face, highlightbackground=edge)
        # Retain the whole split control's face while moving between the image
        # and its native dropdown, even when only one child has pointer hover.
        ttk.Style(self).configure(f"{direction}.History.Action.TButton", background=face)
        arrow.configure(background=face, highlightbackground=face,
                        activebackground=colors["pressed"] if pressed else colors["hover"],
                        foreground=fg)
        # The ttk image state specification selects the disabled PNG without
        # adding the toolkit's automatic stipple effect.

    def refresh(self):
        if self.popup is not None:
            self.close_popup()
        history = history_for(self.state)
        document = active_document(self.state)
        identity = document.document_id if document else None
        if identity != self.projected_document:
            self.record_projection()
        for index, (direction, (_, action, arrow)) in enumerate(self.actions.items()):
            enabled = history is not None and history.available(self.state, direction)
            for b in (action, arrow):
                b.configure(state="normal" if enabled else "disabled", takefocus=enabled)
            self._paint_history(direction)
            labels = history.labels(direction) if history else []
            text = f"{direction.title()}: {labels[0]}" if enabled else f"No available {direction}"
            self.tooltips[index*2].update_text(text + (" (Ctrl+Z)" if direction == "undo" else " (Ctrl+Y)"))

    def input_values(self):
        values = {}
        for key in ("tensor_entry", "angle_x_entry", "angle_y_entry", "angle_z_entry",
                    "angle_x_var", "angle_y_var", "angle_z_var", "rh_dchi_rh_var", "rh_calc_enabled_var"):
            widget = self.state.get(key)
            if callable(getattr(widget, "get", None)):
                values[key] = str(widget.get())
        for key, variable in (self.state.get("tensor_euler_vars") or {}).items():
            values[f"tensor_euler_{key}"] = str(variable.get())
        return values

    def record_projection(self):
        document = active_document(self.state)
        self.projected_document = document.document_id if document else None
        self.projected_inputs = self.input_values()

    def check_pending_inputs(self):
        if self.input_values() != self.projected_inputs:
            raise ValueError("Finish or cancel pending tensor/orientation input edits before Undo/Redo.")

    def is_input_unchanged(self, key):
        document = active_document(self.state)
        history = document.edit_history if document else None
        return (history is not None and bool(history.edits)
                and document.document_id == self.projected_document
                and key in self.projected_inputs
                and self.input_values().get(key) == self.projected_inputs[key])

    def shortcut(self, event, direction):
        # Widget-local editing has priority, even in auxiliary windows.
        if event.widget.winfo_toplevel() is not self.winfo_toplevel():
            return None
        if event.widget.winfo_class() in ("Entry", "TEntry", "TCombobox", "Spinbox", "TSpinbox", "Text"):
            return None
        history = history_for(self.state)
        if history is not None and history.labels(direction):
            self.invoke(direction)
        return "break"

    def invoke(self, direction, count=1):
        self.close_popup()
        try:
            apply_history_move(self.state, direction, count)
        except ValueError as exc:
            self.state["messagebox"].showwarning("Edit history", str(exc))
        self.refresh()

    def close_popup(self):
        if self.popup is not None:
            popup, self.popup = self.popup, None
            previous, self.popup_previous_focus = self.popup_previous_focus, None
            if popup.winfo_exists():
                popup.grab_release()
                popup.destroy()
                # On Windows, destroying the focused override-redirect window
                # can leave the application without keyboard focus; entry class
                # mouse bindings alone then fail to reactivate it.
                target = self.winfo_toplevel()
                if (previous is not None and previous.winfo_exists()
                        and previous.winfo_toplevel() is target):
                    target = previous
                target.focus_force()

    def show_history(self, direction):
        self.close_popup()
        history = history_for(self.state)
        if history is None or not history.available(self.state, direction):
            self.refresh()
            return
        from logic.pcs_document import active_document
        document = active_document(self.state)
        self.popup_document = document.document_id
        self.popup_previous_focus = self.focus_get()
        p = presentation_palette(theme_id(self))
        popup = tk.Toplevel(self)
        popup.withdraw()
        popup.overrideredirect(True)
        popup.transient(self.winfo_toplevel())
        popup.configure(background=p["edge"])
        self.popup = popup
        frame = tk.Frame(popup, background=p["field"], padx=1, pady=1)
        frame.pack(fill="both", expand=True)
        font = get_app_fonts(self)["ui"]
        tk.Label(frame, text=f"{direction.title()} history — {document.title}",
                 background=p["field"], foreground=p["text"], font=font,
                 anchor="w", padx=10, pady=7).pack(fill="x")
        listing = tk.Listbox(frame, font=font, background=p["field"], foreground=p["text"],
                             selectbackground=p["selected"], selectforeground=p["text"],
                             selectmode="extended", exportselection=False, activestyle="none",
                             borderwidth=0, highlightthickness=1, highlightcolor=p["accent"],
                             height=min(10, len(history.labels(direction))), width=40)
        for label in history.labels(direction):
            listing.insert("end", label)
        listing.pack(fill="both", expand=True)
        footer = tk.Label(frame, background=p["surface"], foreground=p["text"], font=font,
                          text=f"{direction.title()} 1 action · This session", anchor="w", padx=10, pady=7)
        footer.pack(fill="x")
        def select(index):
            listing.selection_clear(0, "end")
            listing.selection_set(0, index)
            listing.activate(index)
            footer.configure(text=f"{direction.title()} {index+1} action(s) · This session")
        def apply(_event=None):
            current = active_document(self.state)
            if current is None or current.document_id != self.popup_document:
                self.close_popup()
                return "break"
            count = listing.index("active") + 1
            self.invoke(direction, count)
            return "break"
        listing.bind("<Motion>", lambda e: select(listing.nearest(e.y)))
        listing.bind("<ButtonRelease-1>", apply)
        listing.bind("<Return>", apply)
        for key, delta in (("Up", -1), ("Down", 1)):
            def step(e, d=delta):
                select(max(0, min(listing.size()-1, listing.index("active")+d)))
                listing.see(listing.index("active"))
                return "break"
            listing.bind(f"<{key}>", step)
        def dismiss(_event=None):
            self.close_popup()
            self.actions[direction][2].focus_set()
            return "break"
        popup.bind("<Escape>", dismiss)
        def outside(e):
            if not (popup.winfo_rootx() <= e.x_root < popup.winfo_rootx()+popup.winfo_width()
                    and popup.winfo_rooty() <= e.y_root < popup.winfo_rooty()+popup.winfo_height()):
                self.close_popup()
                return "break"
        popup.bind("<Button-1>", outside, add="+")
        anchor = self.actions[direction][2]
        popup.update_idletasks()
        x, y = _popup_position(anchor, popup.winfo_reqwidth(), popup.winfo_reqheight(),
                               _popup_bounds(anchor))
        # A leading '+' followed by a negative value is an absolute desktop
        # coordinate; '-N' alone means an offset from the screen's right edge.
        popup.geometry(f"+{x}+{y}")
        popup.deiconify()
        popup.grab_set()
        select(0)
        listing.focus_force()

    def cleanup(self, event):
        if event.widget is not self:
            return
        self.close_popup()
        for sequence, identity in self._bindings:
            self.winfo_toplevel().unbind(sequence, identity)
