"""Tk adapter for the isolated viewer; never imports Qt/VTK in the host."""

from logic.molecule_viewer_process import MoleculeViewerProcess
from logic.molecule_viewer_protocol import matches_scene
from logic.pcs_document import active_document, document_display_name
from logic.selection_context import ensure_selection_hub, publish_ref_selection

ADAPTER_KEY = "_isolated_molecule_viewer"
PROBE_MAX_ATOMS = 1000  # Limits the existing dense bond calculation in this experiment only.


def host_ui_theme(state):
    """Resolve existing Tk presentation colors into plain private metadata."""
    import re
    root = state["root"]
    bg, fg = getattr(root, "_app_bg", None), getattr(root, "_app_fg", None)
    if not all(isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color) for color in (bg, fg)):
        return None
    palette = getattr(root, "_app_palette", None)
    if palette is not None:
        return dict(surface=palette["surface"], text=palette["text"],
                    field=palette["field"], edge=palette["edge"],
                    hover=palette["hover"], pressed=palette["pressed"],
                    focus=palette["accent"], disabled=palette["disabled"],
                    disabled_text=palette["disabled_text"])
    from tkinter import ttk
    style = ttk.Style(root)
    def color(name, option, fallback, states=()):
        value = style.lookup(name, option, state=states)
        return value if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) else fallback
    return dict(surface=bg, text=fg, field=color("TEntry", "fieldbackground", bg),
                edge=color("TButton", "bordercolor", fg), hover=color("TButton", "background", bg, ("active",)),
                pressed=color("TButton", "background", bg, ("pressed",)),
                focus=color("Treeview", "selectbackground", fg, ("selected",)),
                disabled=color("TEntry", "fieldbackground", bg, ("disabled",)),
                disabled_text=color("TButton", "foreground", fg, ("disabled",)))


def host_display_controls(state):
    from ui.plot_3d_window import _get_3d_atom_scale, _get_3d_bond_scale, _get_color_mode, _get_show_labels
    from ui.style import get_app_fonts
    font = get_app_fonts(state)["ui"]
    return dict(color_mode=_get_color_mode(state), show_labels=_get_show_labels(state),
                atom_scale=_get_3d_atom_scale(state), bond_tolerance=_get_3d_bond_scale(state) - 1.,
                ui_font=[font[0], font[1]])


def apply_display_changes(state, changes):
    """Apply validated prototype edits through existing Tk owners on the Tk thread."""
    from logic.molecule_viewer_display import validate_display_change
    if any(not validate_display_change(key, value) for key, value in changes.items()):
        raise ValueError("Invalid display change")
    # Check all destinations before changing any value.
    for key, entry_key in (("atom_scale", "atom_scale_entry"), ("bond_tolerance", "bond_tol_entry")):
        if key in changes and state.get(entry_key) is None:
            raise ValueError("The host display control is unavailable")
    import tkinter as tk
    for key, state_key, variable_type in (("color_mode", "plot3d_color_mode_var", tk.StringVar),
                                          ("show_labels", "plot3d_show_labels_var", tk.BooleanVar)):
        if key in changes:
            variable = state.get(state_key)
            if not callable(getattr(variable, "set", None)):
                variable = variable_type(master=state["root"], value=changes[key])
                state[state_key] = variable
            else:
                variable.set(changes[key])
    for key, entry_key in (("atom_scale", "atom_scale_entry"), ("bond_tolerance", "bond_tol_entry")):
        if key in changes:
            entry = state[entry_key]
            entry.delete(0, "end")
            entry.insert(0, repr(float(changes[key])))
    if any(key in changes for key in ("atom_scale", "bond_tolerance")):
        state["update_graph"]()
    from ui.plot_3d_window import _draw_3d_plot
    _draw_3d_plot(state)


def host_display_signature(state):
    """Read effective display preferences through their existing UI owners."""
    from ui.plot_3d_window import (
        _get_3d_atom_scale, _get_3d_bond_scale, _get_color_mode,
        _get_show_labels, _get_plot_font_settings,
    )
    from ui.style import get_app_fonts
    return (_get_3d_atom_scale(state), _get_3d_bond_scale(state),
            _get_color_mode(state), _get_show_labels(state),
            tuple(sorted(_get_plot_font_settings(state).items())),
            tuple(get_app_fonts(state)["ui"][:2]),
            getattr(state.get("root"), "_app_bg", None), getattr(state.get("root"), "_app_fg", None))


def open_matplotlib_for_document(state, document_id):
    """Legacy Tk window is opened only by the host for the active document."""
    document = active_document(state)
    if document is None or document.document_id != document_id:
        return
    from ui.plot_3d_window import open_matplotlib_3d_window
    return open_matplotlib_3d_window(state)


def offer_matplotlib_fallback(state, message):
    """Report a new-viewer failure and offer the retained host-rendered window."""
    if state["messagebox"].askyesno("3D Structure", str(message)[:1024] +
            "\n\nOpen the Matplotlib viewer instead?", parent=state["root"]):
        from ui.plot_3d_window import open_matplotlib_3d_window
        return open_matplotlib_3d_window(state)


def host_scene(state):
    """Read existing display adapters on the Tk thread, preserving their semantics."""
    from ui.plot_3d_window import (
        _get_3d_structure, _get_3d_view_data, _get_3d_atom_scale, _get_3d_bond_scale, calculate_bonds,
    )
    from logic.molecule_viewer_snapshot import prototype_scene_from_view_data
    from ui.molecule_viewer_presentation import host_presentation
    from logic.molecule_viewer_protocol import validate_scene

    document = active_document(state)
    if document is None:
        raise ValueError("The prototype requires an active PCS document")
    data = _get_3d_view_data(state)
    if data is not None and (len(data["ref_ids"]) + len(data.get("reference_ref_ids", []))) > PROBE_MAX_ATOMS:
        raise ValueError("The isolated molecular viewer supports at most 1000 visible markers")
    scene = prototype_scene_from_view_data(
        data, document_id=document.document_id, scene_revision=document.revision,
        bonds=[], atom_scale=_get_3d_atom_scale(state),
    )
    bonds = (calculate_bonds(data["coords"], data["elements"], _get_3d_bond_scale(state))
             if data is not None else [])
    scene["bonds"] = [[int(i), int(j)] for i, j in bonds]
    scene["document_title"] = document_display_name(document)[:512]
    scene["structure_loaded"] = bool(_get_3d_structure(state)[0])
    scene["presentation"] = host_presentation(state, data)
    scene["display_controls"] = host_display_controls(state)
    theme = host_ui_theme(state)
    if theme is not None:
        scene["ui_theme"] = theme
    return validate_scene(scene)


class MoleculeViewerHost:
    """All methods run on the Tk thread; pipe workers never own application state."""

    def __init__(self, state, *, snapshot_factory=host_scene,
                 process_factory=MoleculeViewerProcess, report_error=None,
                 display_signature_factory=None):
        self.state = state
        self.root = state["root"]
        self.snapshot_factory = snapshot_factory
        self.report_error = report_error or self._show_error
        self.hub = ensure_selection_hub(state)
        self.manager = state["document_manager"]
        self.process = process_factory(snapshot_factory(state))
        self.display_signature_factory = (display_signature_factory or
            (host_display_signature if snapshot_factory is host_scene else None))
        self.display_signature = None
        self.closing = False
        self.finished = False
        self.scene_dirty = False
        self.selection_dirty = True
        self.error = None
        self._reported = False
        self._close_callbacks = []
        self._timer = None
        self._selection_token = self._document_token = None
        try:
            self._selection_token = self.hub.subscribe(self._selection_changed)
            self._document_token = self.manager.subscribe(self._document_changed)
            self._schedule()
            state[ADAPTER_KEY] = self
        except Exception:
            self._detach()
            self.process.request_close()
            # Failed attachment has no Tk timer. Reap only the process in a worker;
            # no document/widget access is allowed there.
            import threading
            import time
            process = self.process
            def reap_unattached():
                while process.state not in ("closed", "failed"):
                    process.poll()
                    time.sleep(0.02)
            threading.Thread(target=reap_unattached, daemon=True).start()
            raise

    def _show_error(self, message):
        self.state["messagebox"].showerror("3D Structure", message, parent=self.root)

    def _schedule(self):
        self._timer = self.root.after(30, self._tick)

    def _selection_changed(self, context):
        if not self.closing:
            self.selection_dirty = True
            document = active_document(self.state)
            if document is None or document.document_id != self.process.scene["document_id"]:
                self.scene_dirty = True

    def _document_changed(self, event):
        self.request_refresh()

    def request_refresh(self):
        if not self.closing:
            self.scene_dirty = True

    def _detach(self):
        if self._selection_token is not None:
            self.hub.unsubscribe(self._selection_token)
            self._selection_token = None
        if self._document_token is not None:
            self.manager.unsubscribe(self._document_token)
            self._document_token = None

    def close(self, on_closed=None):
        """Call only after host exit is confirmed; wait asynchronously for cleanup."""
        if self.finished:
            if on_closed is not None:
                on_closed()
            return
        if on_closed is not None and on_closed not in self._close_callbacks:
            self._close_callbacks.append(on_closed)
        if not self.closing:
            self.closing = True
            self._detach()
            self.process.request_close()

    def _tick(self):
        self._timer = None
        try:
            if self.process.state in ("closing", "closed", "failed") or self.process.error:
                self.close()
            if not self.closing:
                if self.display_signature_factory is not None:
                    signature = self.display_signature_factory(self.state)
                    if signature != self.display_signature:
                        self.scene_dirty = True
                        self.display_signature = signature
                document = active_document(self.state)
                if (document is None or document.document_id != self.process.scene["document_id"]
                        or document.revision != self.process.scene["scene_revision"]):
                    self.scene_dirty = True
                if self.scene_dirty:
                    self.process.update_scene(self.snapshot_factory(self.state))
                    self.scene_dirty = False
                    self.selection_dirty = True
                if self.selection_dirty:
                    self.process.set_selection(self.hub.context.ref_ids)
                    self.selection_dirty = False
            selection_revision = self.hub.context.revision
            events = self.process.poll()
            display_changes = {}
            if self.process.state in ("closing", "closed", "failed") or self.process.error:
                self.close()
            for event in events:
                if event["type"] == "display_change" and not self.closing:
                    document = active_document(self.state)
                    if (document is not None and document.document_id == self.process.scene["document_id"]
                            and document.revision == self.process.scene["scene_revision"]
                            and matches_scene(event, self.process.scene)):
                        display_changes[event["key"]] = event["value"]
                    continue
                if event["type"] not in ("selection", "open_matplotlib") or self.closing or self.scene_dirty:
                    continue
                document = active_document(self.state)
                if (document is None or document.document_id != self.process.scene["document_id"]
                        or document.revision != self.process.scene["scene_revision"]
                        or self.hub.context.revision != selection_revision
                        or not matches_scene(event, self.process.scene)):
                    continue
                if event["type"] == "open_matplotlib":
                    open_matplotlib_for_document(self.state, document.document_id)
                else:
                    publish_ref_selection(self.state, event["ref_ids"], source="3d_structure")
            if display_changes and not self.closing:
                apply_display_changes(self.state, display_changes)
                self.scene_dirty = True
        except Exception as exc:
            # A presentation/transport failure must not unwind Tk or document hydration.
            self.error = f"{type(exc).__name__}: {str(exc)[:768]}"
            self.close()
        if self.process.state in ("closed", "failed"):
            self.close()
            if self.process.cleanup_complete.is_set():
                self.finished = True
                if self.state.get(ADAPTER_KEY) is self:
                    self.state.pop(ADAPTER_KEY)
                self.state["_isolated_molecule_viewer_diagnostics"] = {
                    "error": self.error or self.process.error,
                    "stderr": self.process.diagnostics,
                    "cleanup_errors": list(self.process.cleanup_errors),
                }
                callbacks, self._close_callbacks = self._close_callbacks, []
                error = self.error or self.process.error
                if self.process.cleanup_errors:
                    error = error or "; ".join(self.process.cleanup_errors)
                if error and not self._reported and not callbacks:
                    self._reported = True
                    self.report_error(error)
                for callback in callbacks:
                    callback()
                return
        self._schedule()


def open_isolated_molecule_viewer(state, *, offer_fallback=False):
    """Open the isolated viewer from Tk; repeated clicks reuse the current session."""
    if state.get("_molecule_viewer_host_exiting", False):
        return None
    existing = state.get(ADAPTER_KEY)
    if existing is not None and not existing.finished:
        if existing.closing:
            if not getattr(existing, "reopen_requested", False):
                existing.reopen_requested = True
                def reopen():
                    if state.get("_molecule_viewer_host_exiting", False):
                        return
                    from tkinter import TclError
                    try:
                        root_exists = state["root"].winfo_exists()
                    except TclError:
                        root_exists = False
                    if root_exists:
                        open_isolated_molecule_viewer(state, offer_fallback=offer_fallback)
                existing.close(reopen)
        else:
            existing.request_refresh()
            existing.process.request_focus()
        return existing
    try:
        if offer_fallback:
            return MoleculeViewerHost(state, report_error=lambda message: offer_matplotlib_fallback(state, message))
        return MoleculeViewerHost(state)
    except Exception as exc:
        if offer_fallback:
            return offer_matplotlib_fallback(state, exc)
        state["messagebox"].showerror("3D Structure", str(exc), parent=state["root"])
        return None


def request_isolated_viewer_refresh(state):
    adapter = state.get(ADAPTER_KEY)
    if adapter is not None:
        adapter.request_refresh()


def close_isolated_viewer_before_host_exit(state, on_closed):
    state["_molecule_viewer_host_exiting"] = True
    adapter = state.get(ADAPTER_KEY)
    if adapter is None:
        on_closed()
    else:
        adapter.close(on_closed)
