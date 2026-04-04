# tools/plot_3d_pyvista.py
"""
PyVista-based interactive 3D PCS field viewer.
Provides an isosurface / slice visualization of the PCS tensor field
together with a molecular structure overlay.
"""

from __future__ import annotations

import json
import os
from tkinter import colorchooser

import numpy as np

try:
    import pyvista as pv
except Exception as exc:  # noqa: BLE001
    pv = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from ui.plot_3d_window import _get_3d_view_data, calculate_bonds
from logic.chem_constants import covalent_radii, CPK_COLORS

# Default per-level style entries used when building the level table.
# Each entry: (ppm_value, pos_color, neg_color, style, opacity)
DEFAULT_LEVEL_STYLES: list[tuple] = [
    (1.0,  "#FF0000", "#0000FF", "mesh", 0.05),
    (10.0, "#FF0000", "#0000FF", "surface", 0.30),
]

# DEFAULT_LEVEL_STYLES: list[tuple] = [
#     (1.0,  "#E84040", "#4070E8", "surface", 0.18),
#     (2.0,  "#D94B4B", "#4B6FD9", "surface", 0.20),
#     (5.0,  "#C03030", "#3055C0", "surface", 0.24),
#     (10.0, "#A02020", "#2040A0", "surface", 0.30),
# ]

CAMERA_PRESETS: dict[str, str] = {
    "Iso":   "iso",
    "Top":   "xy",
    "Front": "xz",
    "Side":  "yz",
}

BACKGROUND_OPTIONS: dict[str, str] = {
    "White":      "white",
    "Light grey": "#EEEEEE",
    "Dark":       "#222222",
    "Black":      "black",
}

STYLE_OPTIONS = ("surface", "mesh", "both")

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _radius_for_element(el: str) -> float:
    return float(covalent_radii.get(el, 0.77)) * 0.40


def _get_tensor_values(state: dict) -> tuple[float, float]:
    """Read dchi_ax and dchi_rh from the application state."""
    try:
        dchi_ax = float(state["tensor_entry"].get() or 0.0)
    except Exception:  # noqa: BLE001
        dchi_ax = float(state.get("tensor", 0.0) or 0.0)

    try:
        dchi_rh = float(state.get("rh_dchi_rh", 0.0) or 0.0)
    except Exception:  # noqa: BLE001
        dchi_rh = 0.0

    return dchi_ax, dchi_rh


def _parse_contour_levels(text: str) -> tuple[float, ...]:
    """Parse a comma-separated string of ppm values into a sorted tuple."""
    parts = [p.strip() for p in str(text).split(",")]
    vals = []
    for p in parts:
        if not p:
            continue
        vals.append(abs(float(p)))
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        return (1.0, 2.0, 4.0, 8.0)
    return tuple(vals)


# ---------------------------------------------------------------------------
# Auto-padding estimation
# ---------------------------------------------------------------------------

def _estimate_auto_padding(
    coords: np.ndarray,
    dchi_ax: float,
    dchi_rh: float,
    contour_levels: tuple[float, ...],
    *,
    min_padding: float = 10.0,
    max_padding: float = 30.0,
    safety: float = 2.0,
) -> float:
    xyz = np.asarray(coords, dtype=float)
    if xyz.size == 0:
        return float(min_padding)

    positive_levels = [abs(float(v)) for v in contour_levels if abs(float(v)) > 1e-12]
    if not positive_levels:
        return float(min_padding)

    level_min = min(positive_levels)
    dchi_eff = max(abs(float(dchi_ax)), abs(float(dchi_rh)), 1e-6)

    const = 1e4 / (12.0 * np.pi)
    r_est = (const * dchi_eff / level_min) ** (1.0 / 3.0)

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    span = max(maxs - mins)
    half_span = 0.5 * float(span)

    padding = max(min_padding, safety * r_est - half_span)
    padding = max(min_padding, min(max_padding, padding))
    return float(padding)


# ---------------------------------------------------------------------------
# PCS grid construction
# ---------------------------------------------------------------------------

def _build_pcs_grid(
    coords: np.ndarray,
    dchi_ax: float,
    dchi_rh: float,
    *,
    spacing: float = 0.35,
    padding: float = 5.0,
    r_mask_min: float = 0.8,
    clip_abs_ppm: float | None = None,
) -> tuple["pv.ImageData", np.ndarray]:
    """Build a uniform voxel grid containing the PCS scalar field."""
    xyz = np.asarray(coords, dtype=float)
    mins = xyz.min(axis=0) - float(padding)
    maxs = xyz.max(axis=0) + float(padding)

    nx = int(np.ceil((maxs[0] - mins[0]) / spacing)) + 1
    ny = int(np.ceil((maxs[1] - mins[1]) / spacing)) + 1
    nz = int(np.ceil((maxs[2] - mins[2]) / spacing)) + 1

    grid = pv.ImageData()
    grid.origin = tuple(mins)
    grid.spacing = (spacing, spacing, spacing)
    grid.dimensions = (nx, ny, nz)

    xs = mins[0] + np.arange(nx) * spacing
    ys = mins[1] + np.arange(ny) * spacing
    zs = mins[2] + np.arange(nz) * spacing

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    r = np.sqrt(X**2 + Y**2 + Z**2)
    r_safe = np.where(r < 1e-12, np.inf, r)

    cos_theta = np.clip(Z / r_safe, -1.0, 1.0)
    sin2_theta = 1.0 - cos_theta**2
    phi = np.arctan2(Y, X)

    gax = (3.0 * cos_theta**2 - 1.0) / (r_safe**3)
    grh = (1.5 * sin2_theta * np.cos(2.0 * phi)) / (r_safe**3)

    pcs = (float(dchi_ax) * gax + float(dchi_rh) * grh) * 1e4 / (12.0 * np.pi)
    pcs = np.where(r >= float(r_mask_min), pcs, np.nan)

    if clip_abs_ppm is not None and clip_abs_ppm > 0:
        pcs = np.clip(pcs, -float(clip_abs_ppm), float(clip_abs_ppm))

    grid.point_data["pcs"] = pcs.ravel(order="F")
    return grid, pcs


# ---------------------------------------------------------------------------
# Scene building helpers
# ---------------------------------------------------------------------------

def get_cpk_color(atom_label: str) -> str:
    if not atom_label:
        return CPK_COLORS["default"]
    if len(atom_label) >= 2 and atom_label[:2] in CPK_COLORS:
        return CPK_COLORS[atom_label[:2]]
    return CPK_COLORS.get(atom_label[0], CPK_COLORS["default"])

def _add_bonds(plotter, coords: np.ndarray, elements: list[str]) -> None:
    try:
        bonds = calculate_bonds(coords, elements)
    except Exception:
        bonds = []

    bond_radius = 0.08

    for i, j in bonds:
        line = pv.Line(coords[i], coords[j], resolution=1)
        tube = line.tube(radius=bond_radius)
        plotter.add_mesh(tube, color="#555A60", smooth_shading=True)


def _add_atoms(
    plotter,
    coords: np.ndarray,
    labels: list[str],
    elements: list[str],
    ref_ids: list[int],
    selected_ref: int | None = None,
) -> None:
    for xyz, _lbl, el, rid in zip(coords, labels, elements, ref_ids):
        radius = _radius_for_element(el)
        color = CPK_COLORS.get(el, CPK_COLORS["default"])

        sphere = pv.Sphere(radius=radius, center=xyz, theta_resolution=28, phi_resolution=28)
        plotter.add_mesh(sphere, color=color, smooth_shading=True, specular=0.25, ambient=0.18)


def _add_labels(plotter, coords: np.ndarray, labels: list[str], ref_ids: list[int]) -> None:
    if len(coords) == 0:
        return
    label_points = np.asarray(coords, dtype=float)
    label_text = [f"{rid}:{lbl}" for rid, lbl in zip(ref_ids, labels)]
    plotter.add_point_labels(
        label_points,
        label_text,
        font_size=10,
        point_size=0,
        shape_opacity=0.0,
        always_visible=False,
    )


def _add_isosurface_for_level(
    plotter,
    grid,
    level_ppm: float,
    pos_color: str,
    neg_color: str,
    style: str,
    opacity: float,
    level_index: int,
    ambient: float = 0.2,
) -> None:
    """Render positive and negative isosurfaces for a single PCS level."""
    mesh_kwargs_base = dict(smooth_shading=True)

    def _add(surf, color: str, suffix: str) -> None:
        name_base = f"pcs_lv{level_index}_{suffix}"
        if style in ("surface", "both"):
            plotter.add_mesh(
                surf, color=color, opacity=opacity, style="surface",
                ambient=ambient,
                name=name_base + "_surf", **mesh_kwargs_base,
            )
        if style in ("mesh", "both"):
            plotter.add_mesh(
                surf, color=color, opacity=min(opacity * 1.5, 1.0), style="wireframe",
                line_width=1,
                name=name_base + "_wire",
            )

    try:
        surf_pos = grid.contour(isosurfaces=[float(level_ppm)], scalars="pcs")
        if surf_pos.n_points > 0:
            _add(surf_pos, pos_color, "pos")
    except Exception:  # noqa: BLE001
        pass

    try:
        surf_neg = grid.contour(isosurfaces=[-float(level_ppm)], scalars="pcs")
        if surf_neg.n_points > 0:
            _add(surf_neg, neg_color, "neg")
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Main scene population
# ---------------------------------------------------------------------------

def _populate_pcs_scene(
    plotter,
    state: dict,
    *,
    spacing: float = 0.35,
    padding: float | None = None,
    r_mask_min: float = 0.8,
    clip_abs_ppm: float | None = None,
    level_styles: list[dict] | None = None,
    show_slices: bool = False,
    show_isosurfaces: bool = True,
    show_labels: bool = False,
    show_atoms: bool = True,
    show_bonds: bool = True,
    slice_opacity: float = 0.25,
    background: str = "white",
    ambient_light: float = 0.3,
) -> None:
    """Clear the plotter and rebuild the full PCS scene."""
    data = _get_3d_view_data(state)
    if not data:
        raise RuntimeError("No 3D data available.")

    coords   = np.asarray(data["coords"],   dtype=float)
    labels   = list(data["labels"])
    elements = list(data["elements"])
    ref_ids  = list(data["ref_ids"])

    dchi_ax, dchi_rh = _get_tensor_values(state)

    # Resolve level styles
    if level_styles is None or len(level_styles) == 0:
        level_styles = [
            {"ppm": ppm, "pos_color": pc, "neg_color": nc, "style": st, "opacity": op}
            for ppm, pc, nc, st, op in DEFAULT_LEVEL_STYLES
        ]

    contour_levels = tuple(float(ls["ppm"]) for ls in level_styles)

    if padding is None:
        padding = _estimate_auto_padding(
            coords, dchi_ax, dchi_rh, contour_levels,
            min_padding=10.0, max_padding=30.0, safety=2.0,
        )

    # Determine selected atom highlight
    tree = state.get("tree")
    selected_ref = None
    if tree is not None:
        sel = tree.selection()
        if sel:
            try:
                selected_ref = int(tree.item(sel[0], "values")[0])
            except Exception:  # noqa: BLE001
                pass

    grid, _pcs = _build_pcs_grid(
        coords, dchi_ax, dchi_rh,
        spacing=spacing,
        padding=padding,
        r_mask_min=r_mask_min,
        clip_abs_ppm=clip_abs_ppm,
    )

    pcs_flat  = np.asarray(grid.point_data["pcs"], dtype=float)
    pcs_valid = pcs_flat[np.isfinite(pcs_flat)]

    if pcs_valid.size == 0:
        vlim = 10.0
    elif clip_abs_ppm is not None and clip_abs_ppm > 0:
        vlim = float(clip_abs_ppm)
    else:
        vlim = float(np.nanpercentile(np.abs(pcs_valid), 98.0))
        if not np.isfinite(vlim) or vlim <= 0:
            vlim = 10.0

    # --- Clear and configure ---
    plotter.clear()
    plotter.set_background(background)

    # --- Slice planes ---
    if show_slices:
        try:
            first = True
            for normal in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                slc = grid.slice(normal=normal, origin=(0, 0, 0))
                plotter.add_mesh(
                    slc,
                    scalars="pcs",
                    cmap="coolwarm",
                    clim=[-vlim, vlim],
                    opacity=slice_opacity,
                    show_scalar_bar=first,
                    scalar_bar_args={"title": "PCS (ppm)", "vertical": True} if first else None,
                )
                first = False
        except Exception:  # noqa: BLE001
            pass

    # --- Isosurfaces per level ---
    if show_isosurfaces:
        for idx, ls in enumerate(level_styles):
            _add_isosurface_for_level(
                plotter, grid,
                level_ppm=float(ls["ppm"]),
                pos_color=str(ls["pos_color"]),
                neg_color=str(ls["neg_color"]),
                style=str(ls["style"]),
                opacity=float(ls["opacity"]),
                level_index=idx,
                ambient=float(ambient_light),
            )

    # --- Molecular structure ---
    if show_bonds:
        _add_bonds(plotter, coords, elements)
    if show_atoms:
        _add_atoms(plotter, coords, labels, elements, ref_ids, selected_ref=selected_ref)
    if show_labels:
        _add_labels(plotter, coords, labels, ref_ids)

    plotter.add_axes()


# ---------------------------------------------------------------------------
# Refresh / open viewer
# ---------------------------------------------------------------------------

def _collect_scene_kwargs(state: dict) -> dict:
    """Pull current UI settings from state into a kwargs dict for _populate_pcs_scene."""
    return state.get("pcs_scene_kwargs", {})


def refresh_pcs_viewer(state: dict, **kwargs) -> None:
    """Re-render the existing PyVista viewer, or open one if none exists."""
    plotter = state.get("pyvista_field_plotter")
    if plotter is None:
        open_pcs_viewer(state, **kwargs)
        return

    _populate_pcs_scene(plotter, state, **kwargs)
    try:
        plotter.render()
        plotter.update()
    except Exception:  # noqa: BLE001
        pass


def open_pcs_viewer(state: dict, **kwargs) -> None:
    """Open the interactive PyVista viewer window."""
    if pv is None:
        msg = (
            "PyVista import failed.\n\n"
            f"Original error:\n{_IMPORT_ERROR}\n\n"
            "Install with:\n  pip install pyvista vtk"
        )
        mb = state.get("messagebox")
        if mb:
            mb.showerror("PyVista", msg)
        else:
            print(msg)
        return

    existing = state.get("pyvista_field_plotter")
    if existing is not None:
        refresh_pcs_viewer(state, **kwargs)
        return

    plotter = pv.Plotter(window_size=(900, 900))
    state["pyvista_field_plotter"] = plotter

    _populate_pcs_scene(plotter, state, **kwargs)

    try:
        plotter.camera_position = "iso"
    except Exception:  # noqa: BLE001
        pass

    try:
        def _on_viewer_closed(*_args):
            state["pyvista_field_plotter"] = None
        plotter.iren.add_observer("ExitEvent", _on_viewer_closed)
    except Exception:  # noqa: BLE001
        pass

    plotter.show(
        title="PCS Field Viewer",
        auto_close=False,
        interactive=True,
        interactive_update=True,
    )


def close_pcs_viewer(state: dict) -> None:
    """Close and dispose of the active PyVista viewer."""
    plotter = state.get("pyvista_field_plotter")
    if plotter is None:
        return
    try:
        plotter.iren.terminate_app()
    except Exception:  # noqa: BLE001
        pass
    try:
        plotter.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        plotter.ren_win.Finalize()
    except Exception:  # noqa: BLE001
        pass
    state["pyvista_field_plotter"] = None


# ---------------------------------------------------------------------------
# PNG export
# ---------------------------------------------------------------------------

def save_pcs_field_png(
    state: dict,
    *,
    dpi: int = 600,
    width_inch: float = 6.0,
    transparent: bool = False,
    scene_kwargs: dict | None = None,
) -> None:
    """Export the current scene to a high-resolution PNG via an offscreen renderer."""
    plotter = state.get("pyvista_field_plotter")
    if plotter is None:
        mb = state.get("messagebox")
        msg = "No PCS viewer is currently open."
        if mb:
            mb.showwarning("Save PNG", msg)
        else:
            print(msg)
        return

    filedialog = state.get("filedialog")
    if filedialog is None:
        raise RuntimeError("state['filedialog'] not found.")

    path = filedialog.asksaveasfilename(
        title="Save PCS field image",
        defaultextension=".png",
        filetypes=[("PNG image", "*.png")],
    )
    if not path:
        return

    target_px = int(round(float(width_inch) * int(dpi)))

    try:
        off = pv.Plotter(off_screen=True, window_size=(target_px, target_px))
        _populate_pcs_scene(off, state, **(scene_kwargs or {}))
        try:
            off.camera_position = plotter.camera_position
        except Exception:  # noqa: BLE001
            off.camera_position = "iso"

        off.screenshot(path, transparent_background=bool(transparent))
        off.close()

        mb = state.get("messagebox")
        msg = (
            f"Saved: {path}\n\n"
            f"Size: {target_px} × {target_px} px  "
            f"({width_inch:.2f} in @ {dpi} dpi)\n"
            f"Transparent background: {transparent}"
        )
        if mb:
            mb.showinfo("Save PNG", msg)
        else:
            print(msg)

    except Exception as exc:  # noqa: BLE001
        mb = state.get("messagebox")
        if mb:
            mb.showerror("Save PNG", f"Failed:\n{exc}")
        else:
            print(f"Save PNG failed: {exc}")


# ---------------------------------------------------------------------------
# Preset serialisation
# ---------------------------------------------------------------------------

def _gather_preset(ui: dict) -> dict:
    """Collect all UI variable values into a JSON-serialisable dict."""
    level_styles = []
    for row in ui["level_rows"]:
        level_styles.append({
            "ppm":       float(row["ppm"].get()),
            "pos_color": row["pos_color"].get(),
            "neg_color": row["neg_color"].get(),
            "style":     row["style"].get(),
            "opacity":   float(row["opacity"].get()),
        })
    return {
        "level_styles":     level_styles,
        "spacing":          ui["var_spacing"].get(),
        "padding_mode":     ui["var_padding_mode"].get(),
        "padding":          ui["var_padding"].get(),
        "r_mask_min":       ui["var_rmask"].get(),
        "clip_abs_ppm":     ui["var_clip"].get(),
        "show_slices":      bool(ui["var_show_slices"].get()),
        "show_isosurfaces": bool(ui["var_show_iso"].get()),
        "show_atoms":       bool(ui["var_show_atoms"].get()),
        "show_bonds":       bool(ui["var_show_bonds"].get()),
        "show_labels":      bool(ui["var_show_labels"].get()),
        "slice_opacity":    float(ui["var_slice_opacity"].get()),
        "background":       ui["var_background"].get(),
        "ambient_light":    float(ui["var_ambient"].get()),
        "png_dpi":          ui["var_dpi"].get(),
        "png_width_inch":   ui["var_width_inch"].get(),
        "png_transparent":  bool(ui["var_png_transparent"].get()),
    }


def _apply_preset(ui: dict, preset: dict) -> None:
    """Push a loaded preset dict into the UI variables."""
    def _set(var, key, default=""):
        var.set(preset.get(key, default))

    _set(ui["var_spacing"],         "spacing",          "0.35")
    _set(ui["var_padding_mode"],    "padding_mode",     "auto")
    _set(ui["var_padding"],         "padding",          "8.0")
    _set(ui["var_rmask"],           "r_mask_min",       "0.01")
    _set(ui["var_clip"],            "clip_abs_ppm",     "")
    _set(ui["var_show_slices"],     "show_slices",      False)
    _set(ui["var_show_iso"],        "show_isosurfaces", True)
    _set(ui["var_show_atoms"],      "show_atoms",       True)
    _set(ui["var_show_bonds"],      "show_bonds",       True)
    _set(ui["var_show_labels"],     "show_labels",      False)
    _set(ui["var_slice_opacity"],   "slice_opacity",    0.25)
    _set(ui["var_background"],      "background",       "White")
    _set(ui["var_ambient"],         "ambient_light",    0.3)
    _set(ui["var_dpi"],             "png_dpi",          "600")
    _set(ui["var_width_inch"],      "png_width_inch",   "6.0")
    _set(ui["var_png_transparent"], "png_transparent",  False)

    if "level_styles" in preset:
        ui["rebuild_level_rows_fn"](preset["level_styles"])


def save_preset_to_file(ui: dict, filedialog) -> None:
    path = filedialog.asksaveasfilename(
        title="Save preset",
        defaultextension=".json",
        filetypes=[("JSON preset", "*.json")],
    )
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_gather_preset(ui), f, indent=2)


def load_preset_from_file(ui: dict, filedialog) -> None:
    path = filedialog.askopenfilename(
        title="Load preset",
        filetypes=[("JSON preset", "*.json"), ("All files", "*.*")],
    )
    if not path or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        preset = json.load(f)
    _apply_preset(ui, preset)


# ---------------------------------------------------------------------------
# Control panel (Tkinter UI)
# ---------------------------------------------------------------------------

def open_pyvista_field(state: dict) -> None:
    """
    Open the PCS field control panel.
    Only one instance is allowed; subsequent calls raise the existing window.
    """
    root = state.get("root")
    if root is None:
        raise RuntimeError("state['root'] not found.")

    old = state.get("pyvista_field_ctrl_win")
    if old is not None:
        try:
            if old.winfo_exists():
                old.lift()
                old.focus_force()
                return
        except Exception:  # noqa: BLE001
            pass

    import tkinter as tk
    from tkinter import ttk

    # -----------------------------------------------------------------------
    # Style — "PCS.*" prefix only.
    # Never touch ".", "TFrame", "TButton" etc. — those are process-wide and
    # would overwrite the parent application's theme.
    # -----------------------------------------------------------------------
    BG       = "#F5F5F5"
    FG       = "#1A1A1A"
    ACCENT   = "#2D6EBB"
    SEP_CLR  = "#D0D0D0"
    ENTRY_BG = "#FFFFFF"

    _s = ttk.Style()
    # Do NOT call _s.theme_use() here — it is process-wide.
    _s.configure("PCS.TFrame",      background=BG)
    _s.configure("PCS.TLabel",      background=BG, foreground=FG, font=("Segoe UI", 9))
    _s.configure("PCS.TLabelframe", background=BG, foreground=FG)
    _s.configure("PCS.TLabelframe.Label",
                 background=BG, foreground=ACCENT, font=("Segoe UI", 9, "bold"))
    _s.configure("PCS.TCheckbutton", background=BG, foreground=FG, font=("Segoe UI", 9))
    _s.configure("PCS.TRadiobutton", background=BG, foreground=FG, font=("Segoe UI", 9))
    _s.configure("PCS.TEntry",       fieldbackground=ENTRY_BG, foreground=FG)
    _s.configure("PCS.TCombobox",    fieldbackground=ENTRY_BG, foreground=FG)
    _s.configure("PCS.TButton",      font=("Segoe UI", 9))
    _s.configure("PCS.TSeparator",   background=SEP_CLR)
    _s.configure("PCS.Accent.TButton",
                 foreground="white", background=ACCENT, font=("Segoe UI", 9, "bold"))
    _s.map("PCS.Accent.TButton",
           background=[("active", "#1A4F99"), ("pressed", "#163E80")])

    # -----------------------------------------------------------------------
    # Window
    # -----------------------------------------------------------------------
    win = tk.Toplevel(root)
    win.title("PCS Field Viewer — Controls")
    win.geometry("420x820")
    win.configure(bg=BG)
    win.resizable(True, True)
    state["pyvista_field_ctrl_win"] = win

    # Scrollable canvas
    canvas = tk.Canvas(win, bg=BG, highlightthickness=0)
    scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    outer = ttk.Frame(canvas, padding=(14, 10, 14, 10), style="PCS.TFrame")
    canvas_window = canvas.create_window((0, 0), window=outer, anchor="nw")

    def _on_frame_configure(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(event):
        canvas.itemconfig(canvas_window, width=event.width)

    outer.bind("<Configure>", _on_frame_configure)
    canvas.bind("<Configure>", _on_canvas_configure)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    win.bind_all("<MouseWheel>", _on_mousewheel)

    # -----------------------------------------------------------------------
    # Layout helpers — every widget gets a PCS.* style
    # -----------------------------------------------------------------------
    def _section(parent, title: str) -> ttk.LabelFrame:
        f = ttk.LabelFrame(parent, text=title, padding=(8, 4, 8, 8),
                           style="PCS.TLabelframe")
        f.pack(fill="x", pady=(0, 8))
        return f

    def _row_entry(parent, label: str, textvariable, row: int,
                   col_offset: int = 0) -> ttk.Entry:
        ttk.Label(parent, text=label, style="PCS.TLabel").grid(
            row=row, column=col_offset, sticky="w", padx=(0, 6), pady=3)
        ent = ttk.Entry(parent, textvariable=textvariable,
                        width=14, style="PCS.TEntry")
        ent.grid(row=row, column=col_offset + 1, sticky="ew", pady=3)
        parent.columnconfigure(col_offset + 1, weight=1)
        return ent

    def _separator(parent) -> None:
        ttk.Separator(parent, orient="horizontal",
                      style="PCS.TSeparator").pack(fill="x", pady=6)

    # -----------------------------------------------------------------------
    # UI variables
    # -----------------------------------------------------------------------
    var_spacing      = tk.StringVar(value="0.35")
    var_padding_mode = tk.StringVar(value="auto")
    var_padding      = tk.StringVar(value="8.0")
    var_rmask        = tk.StringVar(value="0.01")
    var_clip         = tk.StringVar(value="")

    var_show_slices = tk.BooleanVar(value=False)
    var_show_iso    = tk.BooleanVar(value=True)
    var_show_labels = tk.BooleanVar(value=False)
    var_show_atoms  = tk.BooleanVar(value=True)
    var_show_bonds  = tk.BooleanVar(value=True)

    var_slice_opacity   = tk.DoubleVar(value=0.25)
    var_ambient         = tk.DoubleVar(value=0.3)
    var_surface_opacity = tk.DoubleVar(value=0.22)  # kept for preset compatibility

    var_background      = tk.StringVar(value="White")
    var_png_transparent = tk.BooleanVar(value=False)
    var_dpi             = tk.StringVar(value="600")
    var_width_inch      = tk.StringVar(value="6.0")

    # -----------------------------------------------------------------------
    # Section: Grid parameters
    # -----------------------------------------------------------------------
    sec_grid = _section(outer, "Grid Parameters")
    _row_entry(sec_grid, "Grid spacing (Å)", var_spacing, 0)
    _row_entry(sec_grid, "Metal mask r (Å)", var_rmask,   1)
    _row_entry(sec_grid, "Clip ±ppm",        var_clip,    2)

    pad_frame = ttk.Frame(sec_grid, style="PCS.TFrame")
    pad_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=3)
    ttk.Label(pad_frame, text="Padding (Å)", style="PCS.TLabel").pack(side="left")
    ttk.Radiobutton(pad_frame, text="Auto",   variable=var_padding_mode, value="auto",
                    style="PCS.TRadiobutton").pack(side="left", padx=(10, 4))
    ttk.Radiobutton(pad_frame, text="Manual", variable=var_padding_mode, value="manual",
                    style="PCS.TRadiobutton").pack(side="left", padx=(0, 4))
    ttk.Entry(pad_frame, textvariable=var_padding, width=7,
              style="PCS.TEntry").pack(side="left")

    # -----------------------------------------------------------------------
    # Section: Isosurface levels (dynamic table)
    # -----------------------------------------------------------------------
    sec_levels = _section(outer, "Isosurface Levels")

    hdr = ttk.Frame(sec_levels, style="PCS.TFrame")
    hdr.pack(fill="x")
    for col, (txt, w) in enumerate([
        ("ppm", 8), ("Pos", 4), ("Neg", 4),
        ("Style", 11), ("Opacity", 8), ("", 3),
    ]):
        ttk.Label(hdr, text=txt, style="PCS.TLabel",
                  font=("Segoe UI", 8, "bold"), width=w, anchor="w").grid(
            row=0, column=col, padx=2, sticky="w")

    level_rows_frame = ttk.Frame(sec_levels, style="PCS.TFrame")
    level_rows_frame.pack(fill="x")

    level_rows: list[dict] = []

    def _color_btn(parent, var: tk.StringVar, row: int, col: int) -> tk.Button:
        """Color swatch button — opens a color chooser dialog."""
        btn = tk.Button(parent, bg=var.get(), width=3, relief="flat", cursor="hand2")

        def _pick():
            result = colorchooser.askcolor(color=var.get(), title="Pick colour")
            if result and result[1]:
                var.set(result[1])
                btn.configure(bg=result[1])

        btn.configure(command=_pick)
        btn.grid(row=row, column=col, padx=2, pady=2)
        var.trace_add("write", lambda *_: btn.configure(bg=var.get()))
        return btn

    def _add_level_row(
        ppm: float = 1.0,
        pos_color: str = "#E84040",
        neg_color: str = "#4070E8",
        style_val: str = "surface",
        opacity_val: float = 0.20,
    ) -> None:
        f = ttk.Frame(level_rows_frame, style="PCS.TFrame")
        f.pack(fill="x", pady=1)

        v_ppm     = tk.StringVar(value=str(ppm))
        v_pos     = tk.StringVar(value=pos_color)
        v_neg     = tk.StringVar(value=neg_color)
        v_style   = tk.StringVar(value=style_val)
        v_opacity = tk.StringVar(value=str(opacity_val))

        ttk.Entry(f, textvariable=v_ppm, width=6,
                  style="PCS.TEntry").grid(row=0, column=0, padx=2)
        _color_btn(f, v_pos, 0, 1)
        _color_btn(f, v_neg, 0, 2)
        ttk.Combobox(f, textvariable=v_style, values=STYLE_OPTIONS,
                     width=8, state="readonly",
                     style="PCS.TCombobox").grid(row=0, column=3, padx=2)
        ttk.Entry(f, textvariable=v_opacity, width=5,
                  style="PCS.TEntry").grid(row=0, column=4, padx=2)

        def _remove(frame=f):
            idx = next((i for i, rr in enumerate(level_rows)
                        if rr["frame"] is frame), None)
            if idx is not None:
                level_rows.pop(idx)
                frame.destroy()

        ttk.Button(f, text="✕", width=2, command=_remove,
                   style="PCS.TButton").grid(row=0, column=5, padx=2)

        level_rows.append({
            "frame": f, "ppm": v_ppm, "pos_color": v_pos, "neg_color": v_neg,
            "style": v_style, "opacity": v_opacity,
        })

    def _rebuild_level_rows(styles: list[dict]) -> None:
        for row in list(level_rows):
            row["frame"].destroy()
        level_rows.clear()
        for ls in styles:
            _add_level_row(
                ppm=ls.get("ppm", 1.0),
                pos_color=ls.get("pos_color", "#E84040"),
                neg_color=ls.get("neg_color", "#4070E8"),
                style_val=ls.get("style", "surface"),
                opacity_val=ls.get("opacity", 0.20),
            )

    for ppm, pc, nc, st, op in DEFAULT_LEVEL_STYLES:
        _add_level_row(ppm, pc, nc, st, op)

    ttk.Button(sec_levels, text="+ Add level", command=lambda: _add_level_row(),
               style="PCS.TButton").pack(anchor="w", pady=(4, 0))

    # -----------------------------------------------------------------------
    # Section: Display toggles
    # -----------------------------------------------------------------------
    sec_disp = _section(outer, "Display")
    disp_grid = ttk.Frame(sec_disp, style="PCS.TFrame")
    disp_grid.pack(fill="x")
    for i, (txt, var) in enumerate([
        ("Isosurfaces",  var_show_iso),
        ("Slice planes", var_show_slices),
        ("Atoms",        var_show_atoms),
        ("Bonds",        var_show_bonds),
        ("Labels",       var_show_labels),
    ]):
        ttk.Checkbutton(disp_grid, text=txt, variable=var,
                        style="PCS.TCheckbutton").grid(
            row=i // 3, column=i % 3, sticky="w", padx=8, pady=2)

    # -----------------------------------------------------------------------
    # Section: Appearance
    # -----------------------------------------------------------------------
    sec_app = _section(outer, "Appearance")

    bg_frame = ttk.Frame(sec_app, style="PCS.TFrame")
    bg_frame.pack(fill="x", pady=2)
    ttk.Label(bg_frame, text="Background", width=14,
              style="PCS.TLabel").pack(side="left")
    ttk.Combobox(bg_frame, textvariable=var_background,
                 values=list(BACKGROUND_OPTIONS.keys()),
                 state="readonly", width=12,
                 style="PCS.TCombobox").pack(side="left")

    def _slider_row(parent, label: str, variable: tk.DoubleVar) -> None:
        f = ttk.Frame(parent, style="PCS.TFrame")
        f.pack(fill="x", pady=3)
        ttk.Label(f, text=label, width=14, style="PCS.TLabel").pack(side="left")
        tk.Scale(
            f, from_=0.05, to=1.0, resolution=0.01,
            orient="horizontal", variable=variable,
            bg=BG, highlightthickness=0, length=200,
            troughcolor=SEP_CLR, sliderrelief="flat",
        ).pack(side="left")
        ttk.Label(f, textvariable=variable, width=4,
                  style="PCS.TLabel").pack(side="left", padx=(4, 0))

    _slider_row(sec_app, "Slice opacity", var_slice_opacity)
    _slider_row(sec_app, "Ambient light", var_ambient)

    # -----------------------------------------------------------------------
    # Section: Camera presets
    # -----------------------------------------------------------------------
    sec_cam = _section(outer, "Camera")
    cam_frame = ttk.Frame(sec_cam, style="PCS.TFrame")
    cam_frame.pack(fill="x")

    def _set_camera(preset_key: str) -> None:
        plotter = state.get("pyvista_field_plotter")
        if plotter is None:
            status_var.set("Open the viewer first.")
            return
        try:
            plotter.camera_position = CAMERA_PRESETS[preset_key]
            plotter.render()
            plotter.update()
        except Exception as exc:  # noqa: BLE001
            status_var.set(f"Camera error: {exc}")

    for i, name in enumerate(CAMERA_PRESETS):
        ttk.Button(cam_frame, text=name, width=7,
                   command=lambda n=name: _set_camera(n),
                   style="PCS.TButton").grid(row=0, column=i, padx=3, pady=2)

    # -----------------------------------------------------------------------
    # Section: PNG export
    # -----------------------------------------------------------------------
    sec_png = _section(outer, "Export PNG")
    png_grid = ttk.Frame(sec_png, style="PCS.TFrame")
    png_grid.pack(fill="x")
    _row_entry(png_grid, "DPI",          var_dpi,        0)
    _row_entry(png_grid, "Width (inch)", var_width_inch, 1)
    ttk.Checkbutton(sec_png, text="Transparent background",
                    variable=var_png_transparent,
                    style="PCS.TCheckbutton").pack(anchor="w", pady=(4, 0))

    # -----------------------------------------------------------------------
    # Status bar
    # -----------------------------------------------------------------------
    _separator(outer)
    status_var = tk.StringVar(value="Ready.")
    ttk.Label(outer, textvariable=status_var, style="PCS.TLabel",
              foreground="#666666", font=("Segoe UI", 8)).pack(
        fill="x", pady=(0, 6))

    # -----------------------------------------------------------------------
    # Build scene kwargs from current UI state
    # -----------------------------------------------------------------------
    def _build_scene_kwargs() -> dict:
        ls = [
            {
                "ppm":       float(r["ppm"].get()),
                "pos_color": r["pos_color"].get(),
                "neg_color": r["neg_color"].get(),
                "style":     r["style"].get(),
                "opacity":   float(r["opacity"].get()),
            }
            for r in level_rows
        ]
        clip_raw = var_clip.get().strip()
        clip_val = None if not clip_raw else float(clip_raw)
        padding  = None if var_padding_mode.get() == "auto" else float(var_padding.get())
        bg_val   = BACKGROUND_OPTIONS.get(var_background.get(), "white")

        return dict(
            spacing=float(var_spacing.get()),
            padding=padding,
            r_mask_min=float(var_rmask.get()),
            clip_abs_ppm=clip_val,
            level_styles=ls,
            show_slices=bool(var_show_slices.get()),
            show_isosurfaces=bool(var_show_iso.get()),
            show_labels=bool(var_show_labels.get()),
            show_atoms=bool(var_show_atoms.get()),
            show_bonds=bool(var_show_bonds.get()),
            slice_opacity=float(var_slice_opacity.get()),
            background=bg_val,
            ambient_light=float(var_ambient.get()),
        )

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------
    def _refresh_view() -> None:
        try:
            kwargs = _build_scene_kwargs()
            state["pcs_scene_kwargs"] = kwargs
            if state.get("pyvista_field_plotter") is None:
                open_pcs_viewer(state, **kwargs)
            else:
                refresh_pcs_viewer(state, **kwargs)
            levels_str = ", ".join(r["ppm"].get() for r in level_rows)
            status_var.set(f"Rendered  |  levels: {levels_str} ppm")
        except Exception as exc:  # noqa: BLE001
            status_var.set(f"Error: {exc}")

    def _close_viewer() -> None:
        close_pcs_viewer(state)
        status_var.set("Viewer closed.")

    def _reset_defaults() -> None:
        var_spacing.set("0.35")
        var_padding_mode.set("auto")
        var_padding.set("8.0")
        var_rmask.set("0.01")
        var_clip.set("")
        var_show_slices.set(False)
        var_show_iso.set(True)
        var_show_labels.set(False)
        var_show_atoms.set(True)
        var_show_bonds.set(True)
        var_slice_opacity.set(0.25)
        var_ambient.set(0.3)
        var_background.set("White")
        var_dpi.set("600")
        var_width_inch.set("6.0")
        var_png_transparent.set(False)
        _rebuild_level_rows([
            {"ppm": p, "pos_color": pc, "neg_color": nc, "style": st, "opacity": op}
            for p, pc, nc, st, op in DEFAULT_LEVEL_STYLES
        ])
        status_var.set("Defaults restored.")

    ui_refs = dict(
        level_rows=level_rows,
        var_spacing=var_spacing,
        var_padding_mode=var_padding_mode,
        var_padding=var_padding,
        var_rmask=var_rmask,
        var_clip=var_clip,
        var_show_slices=var_show_slices,
        var_show_iso=var_show_iso,
        var_show_atoms=var_show_atoms,
        var_show_bonds=var_show_bonds,
        var_show_labels=var_show_labels,
        var_surface_opacity=var_surface_opacity,
        var_slice_opacity=var_slice_opacity,
        var_ambient=var_ambient,
        var_background=var_background,
        var_dpi=var_dpi,
        var_width_inch=var_width_inch,
        var_png_transparent=var_png_transparent,
        rebuild_level_rows_fn=_rebuild_level_rows,
    )

    filedialog = state.get("filedialog")

    # -----------------------------------------------------------------------
    # Button bars
    # -----------------------------------------------------------------------
    _separator(outer)

    btn_bar_top = ttk.Frame(outer, style="PCS.TFrame")
    btn_bar_top.pack(fill="x", pady=(0, 4))
    ttk.Button(btn_bar_top, text="Save Preset", style="PCS.TButton",
               command=lambda: save_preset_to_file(ui_refs, filedialog),
               ).pack(side="left", padx=(0, 4))
    ttk.Button(btn_bar_top, text="Load Preset", style="PCS.TButton",
               command=lambda: load_preset_from_file(ui_refs, filedialog),
               ).pack(side="left", padx=(0, 4))
    ttk.Button(btn_bar_top, text="Reset Defaults", style="PCS.TButton",
               command=_reset_defaults,
               ).pack(side="left")
    ttk.Button(btn_bar_top, text="Save PNG", style="PCS.TButton",
               command=lambda: save_pcs_field_png(
                   state,
                   dpi=int(var_dpi.get()),
                   width_inch=float(var_width_inch.get()),
                   transparent=bool(var_png_transparent.get()),
                   scene_kwargs=_build_scene_kwargs(),
               )).pack(side="right")

    btn_bar_bot = ttk.Frame(outer, style="PCS.TFrame")
    btn_bar_bot.pack(fill="x", pady=(0, 8))
    ttk.Button(btn_bar_bot, text="Close Viewer", style="PCS.TButton",
               command=_close_viewer,
               ).pack(side="left")
    ttk.Button(btn_bar_bot, text="Open / Refresh Viewer",
               style="PCS.Accent.TButton",
               command=_refresh_view,
               ).pack(side="right")

    # -----------------------------------------------------------------------
    # Window close handler
    # -----------------------------------------------------------------------
    def _on_close():
        win.unbind_all("<MouseWheel>")
        try:
            win.destroy()
        finally:
            state["pyvista_field_ctrl_win"] = None

    win.protocol("WM_DELETE_WINDOW", _on_close)