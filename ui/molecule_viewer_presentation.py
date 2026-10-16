"""Adapt existing Matplotlib presentation owners to plain child scene values."""


def host_presentation(state, data):
    # Imports stay at the UI boundary. Color normalization/PCS lookup remain in
    # the existing owner; the child never recalculates PCS or chooses a colormap.
    from ui.plot_3d_window import (
        _build_atom_colors_and_sizes, _get_color_mode, _get_show_labels,
        _get_plot_font_settings,
    )
    from matplotlib.colors import to_rgba
    from matplotlib.font_manager import FontProperties, findfont
    import math

    labels = data["labels"] if data is not None else []
    refs = data["ref_ids"] if data is not None else []
    coords = data["coords"] if data is not None else []
    colors, _sizes, norm, cmap = _build_atom_colors_and_sizes(state, labels, refs)
    font = _get_plot_font_settings(state)
    pcs_values = []
    for ref in refs:
        number = (state.get("pcs_by_id") or {}).get(ref)
        try:
            number = float(number) if number is not None else None
        except (TypeError, ValueError, OverflowError):
            number = None  # Optional display data must not break Element mode.
        pcs_values.append(number if number is not None and math.isfinite(number) else None)
    axis_length = (float(max(*(max(row[i] for row in coords) - min(row[i] for row in coords)
                               for i in range(3)), 1.0) * 0.35) if len(coords) else None)
    scale = None
    if norm is not None and cmap is not None:
        scale = {"range": [float(norm.vmin), float(norm.vmax)], "title": "PCS (ppm)",
                 "rgba": [[float(v) for v in cmap(i / 255)] for i in range(256)]}
    moving_refs = set()
    torsion_var = state.get("torsion_avg_enabled_var")
    if torsion_var is not None and bool(torsion_var.get()):
        for group in state.get("torsion_avg_groups", []) or []:
            if bool(group.get("enabled", True)):
                moving_refs.update(int(ref) for ref in group.get("rotating_atoms", ()))
    return {
        "color_mode": _get_color_mode(state),
        "atom_rgba": [[float(v) for v in to_rgba(color)] for color in colors],
        "pcs_values": pcs_values,
        "show_labels": _get_show_labels(state), "axis_length": axis_length,
        "font_family": font["family"], "label_size": font["plot_label"],
        "font_file": findfont(FontProperties(family=font["family"])),
        "axis_label_size": font["viewer_label"], "tick_size": font["plot_tick"],
        "color_scale": scale,
        "averaged_ref_ids": [int(ref) for ref in refs if ref in moving_refs],
    }
