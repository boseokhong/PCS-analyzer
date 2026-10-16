"""Document history for the existing fitted-geometry preview and its removal."""
from copy import deepcopy
from logic.document_edit_history import FIT_KEYS, begin_edit, finish_edit, snapshot, install
from logic.pcs_tensor_state import apply_canonical_tensor_to_legacy
from logic.fitting import apply_fit_to_views


def change_geometry_preview(state, *, clear=False):
    finish_gesture = state.get("finish_parameter_gesture")
    if callable(finish_gesture):
        finish_gesture()
    check = state.get("check_edit_history_pending_inputs")
    if callable(check):
        check()
    if clear and not state.get("fit_override"):
        return False
    token = begin_edit(state, FIT_KEYS)
    before = snapshot(state, FIT_KEYS)
    previous_tensor = state.get("pcs_tensor_state")
    try:
        if clear:
            state.pop("fit_override", None)
        else:
            result = apply_fit_to_views(state, notify=False)
            if result is False or not state.get("last_fit_result"):
                return False
            state["fit_override"] = deepcopy(state["fit_override"])
        # Existing Mode C preview writes fitted strengths to legacy inputs.
        # Include their confirmed redraw/canonical import in this one action.
        state["update_graph"]()
        refresh = state.get("refresh_fitting_commands")
        if callable(refresh):
            refresh()
        return True
    except ValueError:
        # Display validation precedes canonical installation. Reject that
        # incomplete preview transaction, including its legacy strength inputs.
        # Post-install failures retain accepted values and a reversible edit.
        if state.get("pcs_tensor_state") is previous_tensor:
            install(state, FIT_KEYS, before)
            if previous_tensor is not None:
                apply_canonical_tensor_to_legacy(state, previous_tensor)
        raise
    finally:
        mode = (state.get("last_fit_result") or {}).get("mode")
        caption = {"theta_alpha_multi": "Mode A", "euler_global": "Mode B",
                   "full_tensor": "Mode C"}.get(mode, "Fit")
        finish_edit(state, token, "Clear geometry preview" if clear else f"Preview geometry · {caption}")


def geometry_preview_command(state, *, clear=False):
    """Surface pending-input or redraw failures at the UI command boundary."""
    try:
        return change_geometry_preview(state, clear=clear)
    except Exception as exc:
        state["messagebox"].showerror("Geometry preview", str(exc))
        return False
