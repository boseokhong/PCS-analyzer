"""Persistent, UI-independent Control Panel layout preferences."""

from __future__ import annotations


CONTROL_PANEL_SECTION_ORDER = (
    "tensor",
    "orientation",
    "plot_display",
    "symmetry",
    "elements",
    "susceptibility",
    "tools",
)
DEFAULT_EXPANDED_SECTIONS = (
    "tensor",
    "orientation",
    "plot_display",
    "elements",
)
DEFAULT_PINNED_TOOLS = (
    "builtin.pcs_plot_2d",
    "builtin.structure_3d",
    "builtin.projection",
    "builtin.nmr_workbench",
)


def default_control_panel_preferences() -> dict:
    return {
        "section_order": list(CONTROL_PANEL_SECTION_ORDER),
        "expanded": list(DEFAULT_EXPANDED_SECTIONS),
        "pinned_tools": list(DEFAULT_PINNED_TOOLS),
    }


def normalize_control_panel_preferences(raw) -> dict:
    """Merge safe known values with defaults and retain new sections."""
    defaults = default_control_panel_preferences()
    if not isinstance(raw, dict):
        return defaults

    raw_order = raw.get("section_order")
    order = []
    if isinstance(raw_order, (list, tuple)):
        for value in raw_order:
            section_id = str(value)
            if section_id in CONTROL_PANEL_SECTION_ORDER and section_id not in order:
                order.append(section_id)
    order.extend(
        section_id
        for section_id in CONTROL_PANEL_SECTION_ORDER
        if section_id not in order
    )

    raw_expanded = raw.get("expanded")
    if isinstance(raw_expanded, (list, tuple, set)):
        expanded = [
            section_id
            for section_id in order
            if section_id in {str(value) for value in raw_expanded}
        ]
    else:
        expanded = defaults["expanded"]

    raw_pinned = raw.get("pinned_tools")
    if isinstance(raw_pinned, (list, tuple)):
        pinned = []
        for value in raw_pinned:
            tool_id = str(value).strip()
            if tool_id and tool_id not in pinned:
                pinned.append(tool_id)
    else:
        pinned = defaults["pinned_tools"]

    return {
        "section_order": order,
        "expanded": expanded,
        "pinned_tools": pinned,
    }


def reorder_sections(order, moved_id: str, target_index: int) -> list[str]:
    """Return a normalized order with one section moved vertically."""
    normalized = normalize_control_panel_preferences(
        {"section_order": list(order), "expanded": []}
    )["section_order"]
    if moved_id not in normalized:
        return normalized
    normalized.remove(moved_id)
    target_index = max(0, min(int(target_index), len(normalized)))
    normalized.insert(target_index, moved_id)
    return normalized


__all__ = [
    "CONTROL_PANEL_SECTION_ORDER",
    "DEFAULT_EXPANDED_SECTIONS",
    "DEFAULT_PINNED_TOOLS",
    "default_control_panel_preferences",
    "normalize_control_panel_preferences",
    "reorder_sections",
]
