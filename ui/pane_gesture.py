"""Shared Control Panel boundary gesture policy, independent of geometry/Tk."""

PANEL_SNAP_THRESHOLD_RATIO = 0.55


def control_panel_drag_should_toggle(
    delta_x: float,
    panel_width: float,
    *,
    collapsed: bool,
    threshold_ratio: float = PANEL_SNAP_THRESHOLD_RATIO,
    dock_side: str = "right",
) -> bool:
    """Normalize left/right edge gestures using the existing panel threshold."""
    threshold = max(1.0, float(panel_width) * float(threshold_ratio))
    direction = -1.0 if str(dock_side).strip().lower() == "left" else 1.0
    normalized_delta = float(delta_x) * direction
    return (-normalized_delta >= threshold) if collapsed else (normalized_delta >= threshold)


def control_panel_release_action(
    delta_x, panel_width, *, collapsed, scale=1.0,
    threshold_ratio=PANEL_SNAP_THRESHOLD_RATIO, dock_side="right",
):
    """Snap on release; a click on a collapsed boundary also restores it."""
    if control_panel_drag_should_toggle(
        delta_x, panel_width, collapsed=collapsed,
        threshold_ratio=threshold_ratio, dock_side=dock_side,
    ):
        return "toggle"
    if collapsed and abs(delta_x) <= max(3, int(round(3 * scale))):
        return "expand"
    return None
