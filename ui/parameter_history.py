"""Scoped parameter edit/gesture adapter; scientific values remain in core state."""
from functools import wraps
from math import isfinite

from logic.document_edit_history import FIT_KEYS, begin_edit, finish_edit
from logic.pcs_document import active_document_id


def parameter_edit(label, state=None):
    """Record a confirmed edit, or let the active drag own its whole mutation."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            owner = state if state is not None else args[0]
            if (owner.get("_pcs_document_tracking_suppressed") or
                    owner.get("_parameter_history_depth") or owner.get("_parameter_gesture")):
                return function(*args, **kwargs)
            token = begin_edit(owner, FIT_KEYS)
            owner["_parameter_history_depth"] = 1
            confirmed = False
            try:
                result = function(*args, **kwargs)
                confirmed = result is not False
                return result
            finally:
                owner.pop("_parameter_history_depth", None)
                changed = finish_edit(owner, token, label)
                if confirmed and not changed:
                    recorded = owner.get("edit_history_recorded")
                    if callable(recorded):
                        recorded()
        return wrapped
    return decorate


def attach_parameter_gesture(slider, state, label):
    """Use explicit widget lifecycle callbacks before variable writes and on release."""
    def start():
        end()
        if not state.get("_pcs_document_tracking_suppressed"):
            state["_parameter_gesture"] = (active_document_id(state), begin_edit(state, FIT_KEYS), label, slider)
    def end():
        gesture = state.pop("_parameter_gesture", None)
        if gesture and gesture[0] == active_document_id(state):
            gesture[3]._dragging = False
            changed = finish_edit(state, gesture[1], gesture[2])
            if not changed and callable(state.get("edit_history_recorded")):
                state["edit_history_recorded"]()
    slider._edit_gesture_start = start
    slider._edit_gesture_end = end
    state["finish_parameter_gesture"] = end


@parameter_edit("Edit tensor parameters")
def commit_parameter_update(state):
    original_preview = state.get("fit_override")
    previous_tensor = state.get("pcs_tensor_state")
    preview_changed = False
    try:
        # Mode C rendering projects its preview tensor back into this entry.
        # An explicit axial edit must update that working preview as well;
        # its geometry and the immutable recorded fit remain unchanged.
        if isinstance(original_preview, dict) and original_preview.get("mode") == "full_tensor" and "dchi_ax" in original_preview:
            text = state["tensor_entry"].get()
            value = float(text) if text else 1.0
            if not isfinite(value):
                raise ValueError("Δχ_ax must be finite.")
            state["fit_override"] = dict(original_preview, dchi_ax=value)
            preview_changed = True
        state["update_graph"]()
    except ValueError as exc:
        if preview_changed and state.get("pcs_tensor_state") is previous_tensor:
            # Display-input validation happens before tensor installation.
            state["fit_override"] = original_preview
        state["messagebox"].showerror("Parameter input", str(exc))
        return False


def unchanged_parameter_input(state, key):
    bar = state.get("edit_history_bar")
    return (bar is not None and bar.projected_document == active_document_id(state)
            and key in bar.projected_inputs
            and bar.input_values().get(key) == bar.projected_inputs[key])


def install_center_menu(tree, state):
    """Offer an explicit assigned-centre action for the raw Ref under the pointer."""
    import tkinter as tk
    from logic.parameter_edits import assign_metal_center
    from logic.pcs_document import refresh_follow_active_surfaces
    from ui.theme_palette import presentation_palette, theme_id
    from ui.style import get_app_fonts
    menu = tk.Menu(tree, tearoff=0)
    state["tree_center_menu"] = menu
    def show(event):
        row = tree.identify_row(event.y)
        if not row:
            return
        try:
            ref = int(tree.item(row, "values")[0])
        except (IndexError, TypeError, ValueError):
            return
        document_id = active_document_id(state)
        def apply():
            if active_document_id(state) != document_id:
                return
            try:
                end = state.get("finish_parameter_gesture")
                if callable(end):
                    end()
                check = state.get("check_edit_history_pending_inputs")
                if callable(check):
                    check()
                changed = assign_metal_center(state, ref)
            except ValueError as exc:
                state["messagebox"].showerror("Metal center", str(exc))
                return
            if changed:
                state["update_graph"]()
                state["plot_cartesian"](state)
                from logic.nmr_delta_data_manager import push_layers_to_nmr_if_open
                push_layers_to_nmr_if_open(state)
                refresh_follow_active_surfaces(state, include_legacy=False)
        p = presentation_palette(theme_id(tree))
        menu.configure(background=p["field"], foreground=p["text"],
                       activebackground=p["selected"], activeforeground=p["text"],
                       font=get_app_fonts(state)["ui"])
        menu.delete(0, "end")
        menu.add_command(label=f"Set Ref {ref} as metal center (clear preview)", command=apply,
                         state="normal" if ref in (state.get("atom_ids_raw") or []) else "disabled")
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"
    tree.bind("<Button-3>", show, add="+")
