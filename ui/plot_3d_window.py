# ui/plot_3d_window.py

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from logic.chem_constants import CPK_COLORS, covalent_radii
from logic.rotate_align import rotate_coordinates, rotate_euler
from logic.fitting import _angles_to_rotation_multi


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


def get_cpk_color(atom_label: str) -> str:
    if not atom_label:
        return CPK_COLORS["default"]
    s = str(atom_label).strip()
    if len(s) >= 2 and s[1].islower():
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


def calculate_bonds(atom_coords, atom_elements):
    from scipy.spatial.distance import pdist, squareform

    bonds = []
    distances = squareform(pdist(atom_coords))

    for i in range(len(atom_coords)):
        for j in range(i + 1, len(atom_coords)):
            rsum = covalent_radii.get(atom_elements[i], 0.0) + covalent_radii.get(atom_elements[j], 0.0)
            if distances[i, j] <= rsum * 1.03:
                bonds.append((i, j))
    return bonds


def _get_3d_structure(state):
    """
    Return the structure used for the 3D viewer.
    Prefer effective coordinates when symmetry averaging is enabled.
    """
    atom_data = state.get("atom_data_eff") or state.get("atom_data_raw") or state.get("atom_data") or []
    ref_ids = state.get("atom_ids_eff") or state.get("atom_ids_raw") or list(range(1, len(atom_data) + 1))
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
    """
    Apply fit override to raw coordinates using the same logic used by the main 2D/table pipeline.
    """
    fo = state.get("fit_override")
    if not fo:
        return abs_coords, metal

    mode = (fo.get("mode") or "").lower()

    if mode == "theta_alpha_multi":
        id2idx = {rid: i for i, rid in enumerate(ref_ids)}
        donor_ids = fo.get("donor_ids") or []
        if donor_ids:
            donor_pts = [abs_coords[id2idx[rid]] for rid in donor_ids if rid in id2idx]
            if donor_pts:
                abs_coords = _angles_to_rotation_multi(
                    points=abs_coords,
                    metal=metal,
                    donor_points=donor_pts,
                    theta_deg=fo.get("theta", 0.0),
                    alpha_deg=fo.get("alpha", 0.0),
                    axis_mode=fo.get("axis_mode", "bisector"),
                )

    elif mode == "euler_global":
        ax_deg = float(fo.get("ax", 0.0))
        ay_deg = float(fo.get("ay", 0.0))
        az_deg = float(fo.get("az", 0.0))
        coords0 = abs_coords - metal
        rot0 = rotate_euler(coords0, ax_deg, ay_deg, az_deg)
        abs_coords = rot0 + metal

    elif mode == "full_tensor":
        euler_deg = fo.get("euler_deg", (0.0, 0.0, 0.0))
        ax_deg, ay_deg, az_deg = euler_deg

        metal_pos = fo.get("metal_pos")
        if metal_pos is not None:
            metal = np.array(metal_pos, dtype=float)

        coords0 = abs_coords - metal
        rot0 = rotate_euler(coords0, ax_deg, ay_deg, az_deg)
        abs_coords = rot0 + metal

    return abs_coords, metal


def _get_3d_view_data(state):
    """
    Build the currently visible 3D coordinates using:
    - raw structure
    - current metal center
    - fit override
    - current x/y/z rotation
    - current element checklist
    """
    atom_data, raw_ref_ids = _get_3d_structure(state)
    if not atom_data:
        return None

    selected_elements = _get_selected_elements(state)

    coords_abs = np.array([[x, y, z] for _, x, y, z in atom_data], dtype=float)
    labels = [a for a, *_ in atom_data]
    ref_ids = list(raw_ref_ids)

    metal = np.array([state["x0"], state["y0"], state["z0"]], dtype=float)

    coords_abs, metal = _apply_fit_override_to_raw_coords(state, coords_abs, ref_ids, metal)

    coords0 = coords_abs - metal

    ax_deg = float(state["angle_x_var"].get()) if "angle_x_var" in state else 0.0
    ay_deg = float(state["angle_y_var"].get()) if "angle_y_var" in state else 0.0
    az_deg = float(state["angle_z_var"].get()) if "angle_z_var" in state else 0.0

    coords_rot = rotate_coordinates(coords0, ax_deg, ay_deg, az_deg, (0.0, 0.0, 0.0))

    rows = []
    for label, coord, rid in zip(labels, coords_rot, ref_ids):
        element = label
        if selected_elements is not None and element not in selected_elements:
            continue
        rows.append((label, coord, rid))

    if not rows:
        return None

    out_labels = [r[0] for r in rows]
    out_coords = np.array([r[1] for r in rows], dtype=float)
    out_ref_ids = [r[2] for r in rows]
    out_elements = list(out_labels)

    return {
        "labels": out_labels,
        "coords": out_coords,
        "ref_ids": out_ref_ids,
        "elements": out_elements,
    }


def _get_color_mode(state):
    var = state.get("plot3d_color_mode_var")
    return var.get() if var is not None else "Element"


def _get_show_labels(state):
    var = state.get("plot3d_show_labels_var")
    return bool(var.get()) if var is not None else False


def _build_atom_colors_and_sizes(state, labels, ref_ids):
    """
    Return per-atom colors and marker sizes based on the selected color mode.
    """
    color_mode = _get_color_mode(state)

    sizes = []
    for el in labels:
        radius = covalent_radii.get(el, 1.0)
        sizes.append(radius * 55)

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
    fig = state.get("plot3d_figure")
    canvas = state.get("plot3d_canvas")
    if fig is None or canvas is None:
        return

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

    fig.clear()

    data = _get_3d_view_data(state)
    if data is None:
        ax = fig.add_subplot(111, projection="3d")
        ax.text2D(0.5, 0.5, "No visible atoms.", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        canvas.draw_idle()
        return

    labels = data["labels"]
    coords = data["coords"]
    ref_ids = data["ref_ids"]
    elements = data["elements"]

    ax = fig.add_subplot(111, projection="3d")

    colors, sizes, norm, cmap = _build_atom_colors_and_sizes(state, labels, ref_ids)

    bonds = calculate_bonds(coords, elements)
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
    current_tree = state.get("tree")
    selected_ref = None
    if current_tree is not None:
        sel = current_tree.selection()
        if sel:
            try:
                selected_ref = int(current_tree.item(sel[0], "values")[0])
            except Exception:
                selected_ref = None

    for label, (x, y, z), rid, color, size in zip(labels, coords, ref_ids, colors, sizes):
        edgecolor = "white"
        linewidth = 0.5

        if selected_ref is not None and rid == selected_ref:
            edgecolor = "Yellow"
            linewidth = 1.5
            size = size * 1.25

        pt = ax.scatter(
            x,
            y,
            z,
            color=color,
            s=size,
            alpha=0.9,
            edgecolors=edgecolor,
            linewidths=linewidth,
            picker=True,
        )
        pick_pairs.append((pt, rid))

        if show_labels:
            ax.text(x, y, z, str(label), fontsize=8)

    axis_len = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), np.ptp(coords[:, 2]), 1.0) * 0.35
    add_arrow3d(ax, 0, 0, 0, axis_len, 0, 0, mutation_scale=18, ec="black", fc="blue")
    add_arrow3d(ax, 0, 0, 0, 0, axis_len, 0, mutation_scale=18, ec="black", fc="green")
    add_arrow3d(ax, 0, 0, 0, 0, 0, axis_len, mutation_scale=18, ec="black", fc="red")
    ax.text(axis_len, 0, 0, "X", color="blue", fontsize=11, weight="bold")
    ax.text(0, axis_len, 0, "Y", color="green", fontsize=11, weight="bold")
    ax.text(0, 0, axis_len, "Z", color="red", fontsize=11, weight="bold")

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
        cbar.set_label("PCS [ppm]", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    old_cid = state.get("plot3d_click_cid")
    if old_cid is not None:
        try:
            fig.canvas.mpl_disconnect(old_cid)
        except Exception:
            pass
        state["plot3d_click_cid"] = None

    def _on_click(event):
        row_by_id = state.get("row_by_id", {})
        tree = state.get("tree")
        if tree is None:
            return

        for artist, rid in pick_pairs:
            contains, _ = artist.contains(event)
            if contains:
                item = row_by_id.get(rid)
                if item:
                    tree.selection_set(item)
                    tree.focus(item)
                    tree.see(item)
                break

    state["plot3d_click_cid"] = fig.canvas.mpl_connect("button_press_event", _on_click)

    fig.tight_layout(pad=0.1)
    canvas.draw_idle()


def open_3d_plot_window(state):
    """
    Open or focus the integrated 3D structure viewer.
    The 3D view follows the current element filter and current rotation state.
    """
    atom_data, _ = _get_3d_structure(state)
    if not atom_data:
        state["messagebox"].showerror("Error", "No atom data loaded.")
        return

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
    win.title("3D Structure")
    win.geometry("600x560")

    outer = ttk.Frame(win)
    outer.pack(fill=tk.BOTH, expand=True)

    control = ttk.Frame(outer)
    control.pack(fill=tk.X, padx=8, pady=8)

    ttk.Label(control, text="Color mode:").pack(side=tk.LEFT)

    color_mode_var = tk.StringVar(value="Element")
    state["plot3d_color_mode_var"] = color_mode_var
    color_mode_box = ttk.Combobox(
        control,
        textvariable=color_mode_var,
        values=["Element", "PCS"],
        state="readonly",
        width=12,
    )
    color_mode_box.pack(side=tk.LEFT, padx=(6, 12))

    show_labels_var = tk.BooleanVar(value=False)
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
        try:
            win.destroy()
        finally:
            state["plot3d_popup"] = None
            state["plot3d_figure"] = None
            state["plot3d_canvas"] = None
            state["plot3d_color_mode_var"] = None
            state["plot3d_show_labels_var"] = None
            state["plot3d_click_cid"] = None

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
        fig.savefig(path, dpi=600, bbox_inches="tight")
        state["messagebox"].showinfo("3D Structure", f"Saved:\n{path}")
    except Exception as exc:
        state["messagebox"].showerror("3D Structure", f"Save failed:\n{exc}")