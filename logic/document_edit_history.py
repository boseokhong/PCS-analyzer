"""Document-scoped, in-memory scientific edit history (no UI or optimizers)."""
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from logic.pcs_document import active_document


FIT_KEYS = ("pcs_tensor_state", "fit_override")
STRUCTURE_KEYS = ("atom_data_raw", "atom_data", "atom_ids_raw",
                  "atom_data_conformer", "conformer_applied")
PCS_DATA_KEYS = ("delta_exp_values", "delta_obs_values", "delta_dia_values",
                 "experimental_pcs_observations", "nmr_workbench_delta_exp_provenance")
PCS_KEYS = PCS_DATA_KEYS + ("_delta_exp_undo", "_delta_obs_undo", "_delta_dia_undo",
                           "_experimental_pcs_observations_undo",
                           "_nmr_workbench_delta_exp_provenance_undo")
# Ordinary edits do not include result records or rendering caches. Reset can
# explicitly guard/snapshot the Fit History it clears without making every fit
# calculation a change to the global scientific-context fingerprint.
CONTEXT_KEYS = FIT_KEYS + tuple(k for k in STRUCTURE_KEYS if k != "atom_data") + (
    "atom_data_original", "metal_ref_id", "x0", "y0", "z0",
    "pcs_prediction_model", "torsion_avg_groups",
    "torsion_avg_phase",
) + PCS_DATA_KEYS
# A source replacement is a real edit even when its coordinates are identical.
CONTEXT_KEYS += ("current_structure_path",)


def _digest(value):
    if isinstance(value, dict):
        return tuple(sorted((str(k), _digest(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_digest(v) for v in value)
    if hasattr(value, "dtype") and hasattr(value, "tobytes"):
        return (str(value.dtype), value.shape, value.tobytes())
    return value


def fingerprint(state):
    values = tuple((k, k in state, _digest(state.get(k))) for k in CONTEXT_KEYS)
    return sha256(repr(values).encode("utf-8")).hexdigest()


def snapshot(state, keys):
    return {k: deepcopy(state[k]) for k in keys if k in state}


def install(state, keys, values):
    for key in keys:
        if key in values:
            state[key] = deepcopy(values[key])
        else:
            state.pop(key, None)


@dataclass
class Edit:
    label: str
    keys: tuple
    before: dict
    after: dict
    before_context: str
    after_context: str
    controls_before: dict = field(default_factory=dict)
    controls_after: dict = field(default_factory=dict)
    guard_keys: tuple = ()


@dataclass
class EditHistory:
    edits: list = field(default_factory=list)
    cursor: int = 0
    limit: int = 100

    def clear(self):
        self.edits.clear()
        self.cursor = 0

    def labels(self, direction):
        items = self.edits[:self.cursor][::-1] if direction == "undo" else self.edits[self.cursor:]
        return [e.label for e in items]

    def available(self, state, direction):
        if direction == "undo":
            if self.cursor == 0:
                return False
            edit = self.edits[self.cursor - 1]
            expected, values = edit.after_context, edit.after
        else:
            if self.cursor >= len(self.edits):
                return False
            edit = self.edits[self.cursor]
            expected, values = edit.before_context, edit.before
        return (fingerprint(state) == expected and
                repr(_digest(snapshot(state, edit.guard_keys))) ==
                repr(_digest(snapshot(values, edit.guard_keys))))


def history_for(state):
    document = active_document(state)
    if document is None:
        return None
    if document.edit_history is None:
        document.edit_history = EditHistory()
    return document.edit_history


def begin_edit(state, keys):
    # An independently initiated edit must not become part of a held UI drag.
    if state.get("_parameter_gesture"):
        end = state.get("finish_parameter_gesture")
        if callable(end):
            end()
    document = active_document(state)
    if document is None:
        return None
    return (document.document_id, tuple(keys), snapshot(state, keys), fingerprint(state), document.revision)


def finish_edit(state, token, label, *, controls_before=None, controls_after=None, guard_keys=()):
    if token is None:
        return False
    identity, keys, before, context, revision = token
    document = active_document(state)
    if document is None or document.document_id != identity:
        raise ValueError("The edit belongs to a different document")
    after = snapshot(state, keys)
    after_context = fingerprint(state)
    controls_before = controls_before or {}
    controls_after = controls_after or {}
    if (context == after_context and repr(_digest(before)) == repr(_digest(after))
            and controls_before == controls_after):
        return False
    history = history_for(state)
    expected = (history.edits[history.cursor - 1].after_context if history.cursor else
                history.edits[0].before_context if history.edits else context)
    if expected != context:
        # An untracked scientific edit starts a new safe history boundary.
        history.clear()
    del history.edits[history.cursor:]
    history.edits.append(Edit(str(label), keys, before, after, context, after_context,
                              deepcopy(controls_before), deepcopy(controls_after), tuple(guard_keys)))
    if len(history.edits) > history.limit:
        del history.edits[0]
    history.cursor = len(history.edits)
    if document.revision == revision:
        document.mark_mutated()
    recorded = state.get("edit_history_recorded")
    if callable(recorded):
        recorded()
    refresh = state.get("refresh_edit_history")
    if callable(refresh):
        refresh()
    return True


def move_history(state, direction, count=1):
    """Preflight the entire move, then install only the changed plain values."""
    if direction not in ("undo", "redo") or isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Invalid history operation")
    history = history_for(state)
    if history is None:
        raise ValueError("No active document history")
    remaining = history.cursor if direction == "undo" else len(history.edits) - history.cursor
    if count > remaining:
        raise ValueError("History position is unavailable")
    selected = (history.edits[history.cursor-count:history.cursor] if direction == "undo"
                else history.edits[history.cursor:history.cursor+count])
    guarded = {k for edit in selected for k in edit.guard_keys}
    shadow = {k: deepcopy(state[k]) for k in set(CONTEXT_KEYS) | guarded if k in state}
    cursor = history.cursor
    touched = set()
    labels = []
    controls = {}
    for _ in range(count):
        edit = history.edits[cursor - 1 if direction == "undo" else cursor]
        expected = edit.after_context if direction == "undo" else edit.before_context
        if fingerprint(shadow) != expected:
            raise ValueError("Scientific inputs changed outside this history. Undo/Redo cannot safely restore this edit.")
        source = edit.after if direction == "undo" else edit.before
        if repr(_digest(snapshot(shadow, edit.guard_keys))) != repr(_digest(snapshot(source, edit.guard_keys))):
            raise ValueError("Recorded values changed outside this history. Undo/Redo cannot safely restore this edit.")
        install(shadow, edit.keys, edit.before if direction == "undo" else edit.after)
        target = edit.before_context if direction == "undo" else edit.after_context
        if fingerprint(shadow) != target:
            raise ValueError("History snapshot does not match its scientific context")
        touched.update(edit.keys)
        labels.append(edit.label)
        controls.update(edit.controls_before if direction == "undo" else edit.controls_after)
        cursor += -1 if direction == "undo" else 1
    install(state, touched, shadow)
    history.cursor = cursor
    # Plain UI projection values only; the UI owner applies them to its widgets.
    state["_edit_history_restored_controls"] = deepcopy(controls)
    active_document(state).mark_mutated()
    return labels, touched
