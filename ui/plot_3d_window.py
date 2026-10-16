# ui/plot_3d_window.py
# PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE3

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from logic.chem_constants import CPK_COLORS, covalent_radii
from logic.visibility import visible_ref_id_set
from logic.molecule_view_data import apply_molecule_fit_override, prepare_molecule_view_data
from logic.molecule_figure_export import save_molecule_figure
from logic.fitting import _angles_to_rotation_multi
from logic.selection_context import ensure_selection_hub, publish_ref_selection


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj3d.proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self):
        return min(self._xyz[2], self._xyz[2])


def add_arrow3d(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

def _get_plot_font_settings(state):
    """
    Read Matplotlib plot font settings from app settings.

    Uses:
        font_family_plot
        font_size_plot
        font_scale

    These are shared plot-level settings for Matplotlib/PyVista-style viewers.
    """
    cfg = state.get("app_settings", {}) or {}

    family = str(cfg.get("font_family_plot", "DejaVu Sans")).strip() or "DejaVu Sans"

    try:
        base_size = int(cfg.get("font_size_plot", 9))
    except Exception:
        base_size = 9

    try:
        scale = float(cfg.get("font_scale", 1.0))
    except Exception:
        scale = 1.0

    # Keep values in a safe visual range.
    if base_size < 6:
        base_size = 6
    if base_size > 48:
        base_size = 48

    if scale < 0.5:
        scale = 0.5
    if scale > 2.5:
        scale = 2.5

    plot_label = max(6, int(round(base_size * scale)))
    plot_tick = max(5, int(round((base_size - 1) * scale)))
    viewer_label = max(6, int(round((base_size + 1) * scale)))

    return {
        "family": family,
        "plot_label": plot_label,
        "plot_tick": plot_tick,
        "viewer_label": viewer_label,
    }

def get_cpk_color(atom_label: str) -> str:
    if not atom_label:
        return CPK_COLORS["default"]

    s = str(atom_label).strip()

    # Pseudo labels used by symmetry averaging, if they ever reach the 3D viewer.
    if s.startswith("MeH@"):
        el = "H"
    elif s.startswith("CF3F@"):
        el = "F"
    elif len(s) >= 2 and s[1].islower():
        el = s[:2]   # Cl, Br, Nd, Dy ...
    else:
        el = s[:1]   # C, H, N, O ...

    return CPK_COLORS.get(el, CPK_COLORS["default"])

def set_axes_equal(ax):
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    xr = abs(xlim[1] - xlim[0])
    yr = abs(ylim[1] - ylim[0])
    zr = abs(zlim[1] - zlim[0])

    xm = np.mean(xlim)
    ym = np.mean(ylim)
    zm = np.mean(zlim)

    radius = 0.5 * max([xr, yr, zr])
    ax.set_xlim3d([xm - radius, xm + radius])
    ax.set_ylim3d([ym - radius, ym + radius])
    ax.set_zlim3d([zm - radius, zm + radius])

def _get_3d_bond_scale(state):
    """
    Read bond tolerance from the main components UI.

    components.py stores:
        state["bond_tol_entry"]

    The entry value is interpreted as an additive tolerance:
        actual bond scale = 1.00 + tol

    Example:
        entry = 0.05  -> scale = 1.05
        entry = 0.10  -> scale = 1.10
    """
    fallback = 1.05

    entry = state.get("bond_tol_entry")
    if entry is not None:
        try:
            text = str(entry.get()).strip()
            tol = float(text) if text else 0.05
            scale = 1.00 + tol

            # Safe range to avoid absurd bonding.
            if 0.50 <= scale <= 3.00:
                return scale
        except Exception:
            pass

    return fallback

def calculate_bonds(atom_coords, atom_elements, bond_scale=1.05):
    from scipy.spatial.distance import pdist, squareform

    atom_coords = np.asarray(atom_coords, dtype=float)
    if len(atom_coords) < 2:
        return []

    def _radius_key(label):
        s = str(label).strip()

        if s.startswith("MeH@"):
            return "H"
        if s.startswith("CF3F@"):
            return "F"
        if len(s) >= 2 and s[1].islower():
            return s[:2]
        return s[:1]

    try:
        bond_scale = float(bond_scale)
    except Exception:
        bond_scale = 1.05

    if not (0.50 <= bond_scale <= 3.00):
        bond_scale = 1.05

    bonds = []
    distances = squareform(pdist(atom_coords))
    elements = [_radius_key(el) for el in atom_elements]

    for i in range(len(atom_coords)):
        for j in range(i + 1, len(atom_coords)):
            ri = covalent_radii.get(elements[i], 0.0)
            rj = covalent_radii.get(elements[j], 0.0)
            rsum = ri + rj

            if rsum <= 0.0:
                continue

            if distances[i, j] <= rsum * bond_scale:
                bonds.append((i, j))

    return bonds

def _get_3d_structure(state):
    """
    Return the raw structure used for the 3D viewer.

    The 3D Structure viewer should display the real molecular structure.
    Symmetry-averaged pseudo atoms are used only for 2D/table/fitting views.
    """
    # PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE3
    torsion_var = state.get("torsion_avg_enabled_var")
    torsion_enabled = bool(torsion_var.get()) if torsion_var is not None else False
    avg_data = state.get("torsion_avg_display_atom_data") if torsion_enabled else None
    atom_data = (
        avg_data
        or state.get("atom_data_raw")
        or state.get("atom_data")
        or []
    )

    ref_ids = state.get("atom_ids_raw")

    if not ref_ids or len(ref_ids) != len(atom_data):
        ref_ids = list(range(1, len(atom_data) + 1))

    return atom_data, ref_ids


def _get_selected_elements(state):
    """
    Reuse the main element checklist from the main window.
    """
    check_vars = state.get("check_vars", {})
    if not check_vars:
        return None
    return {el for el, var in check_vars.items() if var.get()}


def _apply_fit_override_to_raw_coords(state, abs_coords, ref_ids, metal):
    """Compatibility wrapper for existing callers of the fitted display path."""
    return apply_molecule_fit_override(
        abs_coords, ref_ids, metal, state.get("fit_override"),
        donor_rotation=_angles_to_rotation_multi,
    )


def _get_3d_view_data(state):
    """Adapt document/Tk values to the shared, UI-independent coordinate path."""
    atom_data, raw_ref_ids = _get_3d_structure(state)
    if not atom_data:
        return None
    visible_ids = visible_ref_id_set(state, atom_data=atom_data, ref_ids=raw_ref_ids)
    angles = tuple(
        float(state[key].get()) if key in state else 0.0
        for key in ("angle_x_var", "angle_y_var", "angle_z_var")
    )
    torsion_var = state.get("torsion_avg_enabled_var")
    keep_var = state.get("torsion_avg_keep_reference_var")
    keep_reference = (
        bool(torsion_var.get()) if torsion_var is not None else False
    ) and (bool(keep_var.get()) if keep_var is not None else False)
    raw = (state.get("atom_data_raw") or state.get("atom_data") or []) if keep_reference else []
    moving_refs = set()
    if keep_reference and raw:
        raw_ids = list(state.get("atom_ids_raw") or range(1, len(raw) + 1))
        if len(raw_ids) == len(raw):
            for group in state.get("torsion_avg_groups", []) or []:
                if bool(group.get("enabled", True)):
                    moving_refs.update(int(r) for r in group.get("rotating_atoms", ()))
    return prepare_molecule_view_data(
        atom_data, raw_ref_ids,
        metal=(state["x0"], state["y0"], state["z0"]), angles=angles,
        fit_override=state.get("fit_override"), visible_ids=visible_ids,
        reference_atom_data=raw,
        reference_ref_ids=state.get("atom_ids_raw") or (),
        moving_ref_ids=moving_refs, donor_rotation=_angles_to_rotation_multi,
    )


def _get_color_mode(state):
    var = state.get("plot3d_color_mode_var")
    getter = getattr(var, "get", None)
    return getter() if callable(getter) else (var if var is not None else "Element")


def _get_show_labels(state):
    var = state.get("plot3d_show_labels_var")
    getter = getattr(var, "get", None)
    return bool(getter()) if callable(getter) else bool(var)

def _get_3d_atom_scale(state):
    """
    Read atom marker scale from the main components UI.
    components.py stores:
        state["atom_scale_entry"]
    Default:
        1.0
    """
    fallback = 1.0

    entry = state.get("atom_scale_entry")
    if entry is not None:
        try:
            text = str(entry.get()).strip()
            scale = float(text) if text else fallback

            # Safe range to avoid invisible or enormous atoms.
            if 0.10 <= scale <= 10.0:
                return scale
        except Exception:
            pass

    return fallback

def _build_atom_colors_and_sizes(state, labels, ref_ids):
    """
    Return per-atom colors and marker sizes based on the selected color mode.
    """
    color_mode = _get_color_mode(state)
    atom_scale = _get_3d_atom_scale(state)

    sizes = []

    for label in labels:
        s = str(label).strip()

        if s.startswith("MeH@"):
            el = "H"
        elif s.startswith("CF3F@"):
            el = "F"
        elif len(s) >= 2 and s[1].islower():
            el = s[:2]
        else:
            el = s[:1]

        radius = covalent_radii.get(el, 1.0)
        sizes.append(radius * 55 * atom_scale)

    if color_mode == "PCS":
        pcs_by_id = state.get("pcs_by_id", {}) or {}

        pcs_values = []
        valid_values = []
        for rid in ref_ids:
            v = pcs_by_id.get(rid, None)
            if v is None:
                pcs_values.append(None)
            else:
                val = float(v)
                pcs_values.append(val)
                valid_values.append(val)

        if valid_values:
            vmax = max(abs(min(valid_values)), abs(max(valid_values)))
            vmax = max(vmax, 1e-6)
            norm = Normalize(vmin=-vmax, vmax=vmax)
            cmap = plt.cm.RdBu_r
            colors = [
                cmap(norm(v)) if v is not None else (0.75, 0.75, 0.75, 1.0)
                for v in pcs_values
            ]
            return colors, sizes, norm, cmap
        else:
            colors = [get_cpk_color(lbl) for lbl in labels]
            return colors, sizes, None, None

    colors = [get_cpk_color(lbl) for lbl in labels]
    return colors, sizes, None, None


def _draw_3d_plot(state):
    isolated_viewer = state.get("_isolated_molecule_viewer")
    if isolated_viewer is not None:
        isolated_viewer.request_refresh()
    fig = state.get("plot3d_figure")
    canvas = state.get("plot3d_canvas")
    if fig is None or canvas is None:
        return

    from logic.pcs_document import active_document_id
    from ui.document_integrations import active_document_title

    win = state.get("plot3d_popup")
    try:
        if win is not None and win.winfo_exists():
            win.title(active_document_title(state, "3D Structure (Matplotlib)"))
    except Exception:
        pass
    current_document_id = active_document_id(state)
    same_document = state.get("plot3d_rendered_document_id") == current_document_id

    font_cfg = _get_plot_font_settings(state)
    plot_font_family = font_cfg["family"]
    plot_label = font_cfg["plot_label"]
    plot_tick = font_cfg["plot_tick"]
    viewer_label = font_cfg["viewer_label"]

    saved_view = None
    saved_limits = None

    if fig.axes:
        old_ax = fig.axes[0]
        try:
            saved_view = (old_ax.elev, old_ax.azim)
        except Exception:
            saved_view = None
        try:
            saved_limits = (
                old_ax.get_xlim3d(),
                old_ax.get_ylim3d(),
                old_ax.get_zlim3d(),
            )
        except Exception:
            saved_limits = None
        if not same_document:
            saved_limits = None

    fig.clear()

    data = _get_3d_view_data(state)
    if data is None:
        ax = fig.add_subplot(111, projection="3d")
        message = ("No structure loaded.\nLoad a structure in PCS Analyzer to display it here."
                   if not _get_3d_structure(state)[0] else
                   "No visible atoms.\nCheck the element filters in PCS Analyzer.")
        ax.text2D(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
                  fontfamily=plot_font_family, fontsize=plot_label)
        ax.set_axis_off()
        state["plot3d_rendered_document_id"] = current_document_id
        canvas.draw_idle()
        return

    labels = data["labels"]
    coords = data["coords"]
    ref_ids = data["ref_ids"]
    elements = data["elements"]

    ax = fig.add_subplot(111, projection="3d")

    # PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE4
    ref_coords_overlay = data.get("reference_coords")
    ref_labels_overlay = data.get("reference_labels") or []
    if ref_coords_overlay is not None and len(ref_coords_overlay):
        for lbl, coord in zip(ref_labels_overlay, ref_coords_overlay):
            cpk = get_cpk_color(lbl)
            ax.scatter(
                coord[0], coord[1], coord[2],
                s=24,
                facecolors=cpk,
                edgecolors=cpk,
                linewidths=0.6,
                alpha=0.72,
                marker="o",
            )

    colors, sizes, norm, cmap = _build_atom_colors_and_sizes(state, labels, ref_ids)

    torsion_var = state.get("torsion_avg_enabled_var")
    torsion_enabled = bool(torsion_var.get()) if torsion_var is not None else False
    torsion_avg_ref_ids = set()
    if torsion_enabled:
        for group in (state.get("torsion_avg_groups", []) or []):
            if bool(group.get("enabled", True)):
                torsion_avg_ref_ids.update(int(r) for r in group.get("rotating_atoms", ()))

    bond_scale = _get_3d_bond_scale(state)
    bonds = calculate_bonds(coords, elements, bond_scale=bond_scale)
    for i, j in bonds:
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            [coords[i, 2], coords[j, 2]],
            color="gray",
            lw=2.0,
            alpha=0.9,
        )

    pick_pairs = []

    show_labels = _get_show_labels(state)
    selected_ref = ensure_selection_hub(state).context.primary_ref_id

    for label, (x, y, z), rid, color, size in zip(labels, coords, ref_ids, colors, sizes):
        is_avg = rid in torsion_avg_ref_ids
        is_selected = selected_ref is not None and rid == selected_ref

        if is_avg:
            # Ensemble-averaged coordinates are derived representative positions:
            # hollow marker, element-coloured edge, slightly larger than reference.
            size_avg = size * (1.18 if not is_selected else 1.35)
            pt = ax.scatter(
                x, y, z,
                s=size_avg,
                alpha=0.95,
                facecolors="none",
                edgecolors=get_cpk_color(label),
                linewidths=1.25,
                marker="o",
                picker=True,
            )
            if is_selected:
                ax.scatter(
                    x, y, z,
                    s=size_avg * 1.75,
                    facecolors="none",
                    edgecolors="Yellow",
                    linewidths=1.5,
                    marker="o",
                )
        else:
            edgecolor = "Yellow" if is_selected else "white"
            linewidth = 1.5 if is_selected else 0.5
            if is_selected:
                size = size * 1.25
            pt = ax.scatter(
                x, y, z,
                color=color,
                s=size,
                alpha=0.9,
                edgecolors=edgecolor,
                linewidths=linewidth,
                picker=True,
            )

        pick_pairs.append((pt, rid))

        if show_labels:
            ax.text(
                x, y, z,
                str(label),
                fontsize=plot_label,
                fontfamily=plot_font_family,
            )

    # Dynamic representation legend.  Do not show a Reference entry unless
    # the reference overlay is actually enabled and present.
    avg_visible = any(rid in torsion_avg_ref_ids for rid in ref_ids)
    ref_visible = ref_coords_overlay is not None and len(ref_coords_overlay) > 0
    avg_handles = []
    if ref_visible:
        avg_handles.append(Line2D([], [], marker="o", linestyle="None",
                                  markerfacecolor="0.35", markeredgecolor="0.35",
                                  markersize=5.5, label="Reference"))
    if avg_visible:
        avg_handles.append(Line2D([], [], marker="o", linestyle="None",
                                  markerfacecolor="none", markeredgecolor="0.35",
                                  markeredgewidth=1.2, markersize=6.0, label="Ensemble avg"))
    if avg_handles:
        avg_leg = ax.legend(
            handles=avg_handles,
            loc="upper right",
            frameon=False,
            fontsize=plot_tick,
            handletextpad=0.6,
            labelspacing=0.4,
            borderpad=0.2,
        )
        avg_leg.set_title("Averaging", prop={"size": plot_tick, "weight": "bold"})

    axis_len = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), np.ptp(coords[:, 2]), 1.0) * 0.35
    add_arrow3d(ax, 0, 0, 0, axis_len, 0, 0, mutation_scale=18, ec="black", fc="blue")
    add_arrow3d(ax, 0, 0, 0, 0, axis_len, 0, mutation_scale=18, ec="black", fc="green")
    add_arrow3d(ax, 0, 0, 0, 0, 0, axis_len, mutation_scale=18, ec="black", fc="red")
    ax.text(axis_len, 0, 0, "X", color="blue", fontsize=viewer_label, fontfamily=plot_font_family, weight="bold")
    ax.text(0, axis_len, 0, "Y", color="green", fontsize=viewer_label, fontfamily=plot_font_family, weight="bold")
    ax.text(0, 0, axis_len, "Z", color="red", fontsize=viewer_label, fontfamily=plot_font_family, weight="bold")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    if saved_limits is not None:
        try:
            ax.set_xlim3d(saved_limits[0])
            ax.set_ylim3d(saved_limits[1])
            ax.set_zlim3d(saved_limits[2])
        except Exception:
            set_axes_equal(ax)
    else:
        set_axes_equal(ax)

    if saved_view is not None:
        try:
            ax.view_init(elev=saved_view[0], azim=saved_view[1])
        except Exception:
            pass

    color_mode = _get_color_mode(state)
    if color_mode == "PCS" and norm is not None and cmap is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.03)
        cbar.set_label("PCS (ppm)", fontsize=plot_label, fontfamily=plot_font_family)
        cbar.ax.tick_params(labelsize=plot_tick)

        for tick_label in cbar.ax.get_yticklabels():
            tick_label.set_fontfamily(plot_font_family)

    old_cid = state.get("plot3d_click_cid")
    if old_cid is not None:
        try:
            fig.canvas.mpl_disconnect(old_cid)
        except Exception:
            pass
        state["plot3d_click_cid"] = None

    def _on_click(event):
        for artist, rid in pick_pairs:
            contains, _ = artist.contains(event)
            if contains:
                publish_ref_selection(
                    state,
                    [rid],
                    source="3d_structure",
                )
                break

    state["plot3d_click_cid"] = fig.canvas.mpl_connect("button_press_event", _on_click)

    try:
        fig.tight_layout(pad=0.1)
    except Exception:
        pass

    canvas.draw_idle()
    state["plot3d_rendered_document_id"] = current_document_id

def open_isolated_pyvista_structure_window(state):
    """Launch the child viewer without importing Qt/VTK into the Tk process."""
    from ui.molecule_viewer_host import open_isolated_molecule_viewer
    return open_isolated_molecule_viewer(state)


def open_pyvista_structure_window(state):
    """
    Open a separate PyVista/Qt 3D structure viewer.

    This viewer reuses the same 3D data pipeline as the Matplotlib viewer:
    - raw structure
    - metal-centered coordinates
    - fit override
    - x/y/z rotation
    - element checklist
    - bond tolerance
    - atom scale
    """
    try:
        import pyvista as pv
        from pyvistaqt import BackgroundPlotter
    except Exception as exc:
        state["messagebox"].showerror(
            "PyVista 3D Viewer",
            "PyVista viewer requires pyvista and pyvistaqt.\n\n"
            "Install with:\n"
            "python -m pip install pyvista pyvistaqt\n\n"
            f"Error:\n{exc}"
        )
        return

    data = _get_3d_view_data(state)
    if data is None:
        state["messagebox"].showwarning(
            "PyVista 3D Viewer",
            "No visible atoms."
        )
        return

    labels = data["labels"]
    coords = data["coords"]
    ref_ids = data["ref_ids"]
    elements = data["elements"]

    atom_scale = _get_3d_atom_scale(state)
    bond_scale = _get_3d_bond_scale(state)

    plotter = BackgroundPlotter(title="PyVista 3D Structure")
    plotter.set_background("white")

    # Bonds
    bonds = calculate_bonds(coords, elements, bond_scale=bond_scale)
    for i, j in bonds:
        p0 = coords[i]
        p1 = coords[j]
        line = pv.Line(p0, p1)
        plotter.add_mesh(
            line,
            color="gray",
            line_width=4,
            name=f"bond_{i}_{j}",
        )

    # Atoms
    for idx, (label, xyz, rid) in enumerate(zip(labels, coords, ref_ids)):
        color = get_cpk_color(label)

        s = str(label).strip()
        if s.startswith("MeH@"):
            el = "H"
        elif s.startswith("CF3F@"):
            el = "F"
        elif len(s) >= 2 and s[1].islower():
            el = s[:2]
        else:
            el = s[:1]

        radius = covalent_radii.get(el, 1.0)
        sphere_radius = max(0.05, radius * 0.25 * atom_scale)

        sphere = pv.Sphere(
            radius=sphere_radius,
            center=tuple(float(v) for v in xyz),
            theta_resolution=32,
            phi_resolution=32,
        )

        plotter.add_mesh(
            sphere,
            color=color,
            smooth_shading=True,
            name=f"atom_{rid}_{idx}",
        )

    # Axis arrows
    axis_len = max(
        np.ptp(coords[:, 0]),
        np.ptp(coords[:, 1]),
        np.ptp(coords[:, 2]),
        1.0,
    ) * 0.35

    plotter.add_arrows(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[axis_len, 0.0, 0.0]]),
        color="blue",
        mag=1.0,
    )
    plotter.add_arrows(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, axis_len, 0.0]]),
        color="green",
        mag=1.0,
    )
    plotter.add_arrows(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, axis_len]]),
        color="red",
        mag=1.0,
    )

    plotter.add_text("PyVista 3D Structure", font_size=10)
    plotter.show_axes()
    plotter.reset_camera()

    # Keep a reference so the window is not immediately garbage-collected.
    state["plot3d_pyvista_plotter"] = plotter

def open_3d_plot_window(state):
    """Default 3D entry; keep Qt/VTK isolated from the Tk host."""
    from ui.molecule_viewer_host import open_isolated_molecule_viewer
    return open_isolated_molecule_viewer(state, offer_fallback=True)


def open_matplotlib_3d_window(state):
    """
    Open or focus the integrated 3D structure viewer.
    The 3D view follows the current element filter and current rotation state.
    """
    win = state.get("plot3d_popup")
    if win is not None:
        try:
            if win.winfo_exists():
                win.lift()
                win.focus_force()
                _draw_3d_plot(state)
                return
        except Exception:
            pass

    win = tk.Toplevel(state["root"])
    from ui.document_integrations import active_document_title

    win.title(active_document_title(state, "3D Structure (Matplotlib)"))
    win.geometry("650x560")

    outer = ttk.Frame(win)
    outer.pack(fill=tk.BOTH, expand=True)

    control = ttk.Frame(outer)
    control.pack(fill=tk.X, padx=8, pady=8)

    ttk.Label(control, text="Color mode:").pack(side=tk.LEFT)

    color_mode_var = tk.StringVar(value=_get_color_mode(state))
    state["plot3d_color_mode_var"] = color_mode_var
    color_mode_box = ttk.Combobox(
        control,
        textvariable=color_mode_var,
        values=["Element", "PCS"],
        state="readonly",
        width=12,
    )
    color_mode_box.pack(side=tk.LEFT, padx=(6, 12))

    show_labels_var = tk.BooleanVar(value=_get_show_labels(state))
    state["plot3d_show_labels_var"] = show_labels_var
    ttk.Checkbutton(
        control,
        text="Show labels",
        variable=show_labels_var,
        command=lambda: _draw_3d_plot(state),
    ).pack(side=tk.LEFT, padx=(0, 12))

    ttk.Button(
        control,
        text="⟳ Refresh",
        command=lambda: _draw_3d_plot(state),
    ).pack(side=tk.RIGHT)

    ttk.Button(
        control,
        text="💾 Export plot",
        command=lambda: _save_3d_figure(state),
    ).pack(side=tk.RIGHT, padx=(0, 6))

    fig_frame = ttk.Frame(outer)
    fig_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    fig = plt.Figure(figsize=(4.8, 4.8), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=fig_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, fig_frame)
    toolbar.update()

    state["plot3d_popup"] = win
    state["plot3d_figure"] = fig
    state["plot3d_canvas"] = canvas
    state.setdefault("plot3d_click_cid", None)

    color_mode_box.bind("<<ComboboxSelected>>", lambda e: _draw_3d_plot(state))

    def _on_close():
        # Display preferences remain authoritative after the legacy widgets close.
        try:
            win.destroy()
        finally:
            state["plot3d_popup"] = None
            state["plot3d_figure"] = None
            state["plot3d_canvas"] = None
            state["plot3d_click_cid"] = None
            state["plot3d_rendered_document_id"] = None
            isolated_viewer = state.get("_isolated_molecule_viewer")
            if isolated_viewer is not None:
                isolated_viewer.request_refresh()

    win.protocol("WM_DELETE_WINDOW", _on_close)

    _draw_3d_plot(state)

def _save_3d_figure(state):
    fig = state.get("plot3d_figure")
    if fig is None:
        state["messagebox"].showerror("3D Structure", "No figure available.")
        return

    path = state["filedialog"].asksaveasfilename(
        title="Save 3D figure",
        defaultextension=".png",
        filetypes=[
            ("PNG image", "*.png"),
            ("PDF file", "*.pdf"),
            ("SVG file", "*.svg"),
            ("All files", "*.*"),
        ],
    )
    if not path:
        return

    try:
        save_molecule_figure(fig, path)
        state["messagebox"].showinfo("3D Structure", f"Saved:\n{path}")
    except Exception as exc:
        state["messagebox"].showerror("3D Structure", f"Save failed:\n{exc}")
