"""Validation and presentation-only residual threshold changes."""
from math import isfinite


def validate_thresholds(ok, warn):
    try:
        pair = float(ok), float(warn)
    except (TypeError, ValueError) as exc:
        raise ValueError("Enter two finite numbers with 0 ≤ green < orange (ppm).") from exc
    if not all(isfinite(value) for value in pair) or not 0 <= pair[0] < pair[1]:
        raise ValueError("Enter two finite numbers with 0 ≤ green < orange (ppm).")
    return pair


def load_thresholds(ok, warn):
    try:
        return validate_thresholds(ok, warn)
    except ValueError:
        return 0.10, 0.30


def mark_display_modified(state):
    # Display edits are saved, but do not advance scientific revision / invalidate workers.
    from logic.pcs_document import active_document, capture_active_document
    document = active_document(state)
    if document is not None and not state.get("_pcs_document_tracking_suppressed"):
        capture_active_document(state, document)
        document.dirty = True
        refresh = state.get("refresh_document_tabs")
        if callable(refresh):
            refresh()


def retag_residual_tables(state):
    from logic.table_utils import classify_residual_tag
    scientific_tags = {"ok", "warn", "bad", "none"}
    for tree_key, residuals in (
        ("tree", state.get("_main_row_residuals", {})),
        ("rh_tree", state.get("_rh_row_residuals", {})),
    ):
        tree = state.get(tree_key)
        if tree is None:
            continue
        for item in tree.get_children():
            tags = tuple(tag for tag in tree.item(item, "tags") if tag not in scientific_tags)
            tree.item(item, tags=tags + classify_residual_tag(state, residuals.get(item)))
