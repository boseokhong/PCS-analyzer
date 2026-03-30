# ui/projection_window.py

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from logic.chem_constants import CPK_COLORS


def open_projection_window(state):
    """
    Open or focus the integrated projection viewer.
    This viewer supports both phi/cos(theta) and Mollweide projections.
    """
    root = state["root"]
    win = state.get("projection_popup")

    if win is not None:
        try:
            if win.winfo_exists():
                win.lift()
                win.focus_force()
                return
        except Exception:
            pass

    win = tk.Toplevel(root)
    win.title("Projection Viewer")
    win.geometry("980x600")

    outer = ttk.Frame(win)
    outer.pack(fill=tk.BOTH, expand=True)

    control = ttk.Frame(outer)
    control.pack(fill=tk.X, padx=8, pady=8)

    ttk.Label(control, text="Projection mode:").pack(side=tk.LEFT)

    mode_var = tk.StringVar(value="φ / cos(θ)")
    state["projection_mode_var"] = mode_var

    mode_box = ttk.Combobox(
        control,
        textvariable=mode_var,
        values=["φ / cos(θ)", "Mollweide"],
        state="readonly",
        width=18,
    )
    mode_box.pack(side=tk.LEFT, padx=(6, 10))

    ttk.Label(control, text="Fixed r [Å]:").pack(side=tk.LEFT)

    r_var = tk.StringVar(value="10.0")
    state["projection_r_var"] = r_var
    r_entry = ttk.Entry(control, textvariable=r_var, width=8)
    r_entry.pack(side=tk.LEFT, padx=(6, 10))

    show_atoms_var = tk.BooleanVar(value=True)
    state["projection_show_atoms_var"] = show_atoms_var
    ttk.Checkbutton(
        control,
        text="Show atoms",
        variable=show_atoms_var,
        command=lambda: _draw_projection_plot(state),
    ).pack(side=tk.LEFT, padx=(0, 10))

    show_h_var = tk.BooleanVar(value=True)
    state["projection_show_h_var"] = show_h_var
    ttk.Checkbutton(
        control,
        text="Show H",
        variable=show_h_var,
        command=lambda: _draw_projection_plot(state),
    ).pack(side=tk.LEFT, padx=(0, 10))

    ttk.Button(
        control,
        text="⟳ Refresh",
        command=lambda: _draw_projection_plot(state),
    ).pack(side=tk.RIGHT, padx=(6, 0))

    ttk.Button(
        control,
        text="💾 Export Figure",
        command=lambda: _export_projection_figure(state),
    ).pack(side=tk.RIGHT)

    fig_frame = ttk.Frame(outer)
    fig_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    fig = plt.Figure(figsize=(7, 6), dpi=120)
    canvas = FigureCanvasTkAgg(fig, master=fig_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, fig_frame)
    toolbar.update()

    state["projection_popup"] = win
    state["projection_figure"] = fig
    state["projection_canvas"] = canvas

    mode_box.bind("<<ComboboxSelected>>", lambda e: _draw_projection_plot(state))
    r_entry.bind("<Return>", lambda e: _draw_projection_plot(state))
    r_entry.bind("<FocusOut>", lambda e: _draw_projection_plot(state))

    def _on_close():
        try:
            win.destroy()
        finally:
            state["projection_popup"] = None
            state["projection_figure"] = None
            state["projection_canvas"] = None
            state["projection_mode_var"] = None
            state["projection_r_var"] = None
            state["projection_show_atoms_var"] = None
            state["projection_show_h_var"] = None
            state["projection_click_cid"] = None

    win.protocol("WM_DELETE_WINDOW", _on_close)

    _draw_projection_plot(state)


def _get_projection_tensor(state):
    """
    Return axial and rhombic tensor values in units of 1e-32 m^3.
    """
    try:
        dchi_ax = float(state["tensor_entry"].get() or 1.0)
    except Exception:
        dchi_ax = 1.0

    try:
        dchi_rh = float(state.get("rh_dchi_rh", 0.0) or 0.0)
    except Exception:
        dchi_rh = 0.0

    return dchi_ax, dchi_rh


def _get_projection_coords(state):
    """
    Return rotated coordinates, display labels, and ref_ids using the current app state.
    This uses the same filtered and rotated coordinates as the main PCS view.
    """
    fn = state.get("filter_atoms")
    if fn is None:
        return None, None, None

    if not state.get("atom_data"):
        return None, None, None

    try:
        polar_data, rotated_sel = fn(state)
    except Exception:
        return None, None, None

    if not rotated_sel:
        return None, None, None

    coords = np.array([[x, y, z] for x, y, z in rotated_sel], dtype=float)
    labels = [row[0] for row in polar_data]
    ref_ids = list(state.get("current_selected_ids", []) or [])

    n = min(len(coords), len(labels), len(ref_ids))
    if n == 0:
        return None, None, None

    return coords[:n], labels[:n], ref_ids[:n]


def _pcs_from_grid(gax, grh, dchi_ax, dchi_rh):
    """
    Compute PCS values on a projection grid.
    """
    return (dchi_ax * gax + dchi_rh * grh) * 1e4 / (12.0 * np.pi)


def _get_atom_color(label):
    """
    Resolve a display color from an atom label.
    Supports labels such as H, C, Cl, Br, N1, C12, etc.
    """
    if not label:
        return CPK_COLORS["default"]

    if len(label) >= 2 and label[:2] in CPK_COLORS:
        return CPK_COLORS[label[:2]]

    return CPK_COLORS.get(label[0], CPK_COLORS["default"])


def _nice_levels(vlim):
    """
    Return visually nice contour levels with human-friendly spacing.
    """
    if vlim <= 0:
        return np.array([-1.0, 1.0])

    candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    step = next((c for c in candidates if vlim / c <= 6), 20)

    levels = np.arange(-vlim, vlim + step * 0.1, step)
    levels = levels[np.abs(levels) > step * 0.01]

    if len(levels) < 2:
        levels = np.array([-step, step], dtype=float)

    return levels


def _draw_projection_plot(state):
    fig = state.get("projection_figure")
    canvas = state.get("projection_canvas")
    if fig is None or canvas is None:
        return

    fig.clear()
    click_pairs = []

    mode_var = state.get("projection_mode_var")
    r_var = state.get("projection_r_var")
    show_atoms_var = state.get("projection_show_atoms_var")
    show_h_var = state.get("projection_show_h_var")

    mode = mode_var.get() if mode_var is not None else "φ / cos(θ)"

    try:
        r_fixed = float(r_var.get()) if r_var is not None else 10.0
        if r_fixed <= 0:
            r_fixed = 10.0
    except Exception:
        r_fixed = 10.0

    show_atoms = bool(show_atoms_var.get()) if show_atoms_var is not None else True
    show_h = bool(show_h_var.get()) if show_h_var is not None else True

    dchi_ax, dchi_rh = _get_projection_tensor(state)

    n_theta, n_phi = 180, 360
    theta_grid = np.linspace(1e-3, np.pi - 1e-3, n_theta)
    phi_grid = np.linspace(-np.pi, np.pi, n_phi)

    TH, PH = np.meshgrid(theta_grid, phi_grid)
    cos2 = np.cos(TH) ** 2
    sin2 = 1.0 - cos2

    gax = (3.0 * cos2 - 1.0) / (r_fixed ** 3)
    grh = 1.5 * sin2 * np.cos(2.0 * PH) / (r_fixed ** 3)
    pcs_map = _pcs_from_grid(gax, grh, dchi_ax, dchi_rh)

    vlim = max(
        abs(float(np.nanpercentile(pcs_map, 2))),
        abs(float(np.nanpercentile(pcs_map, 98))),
        0.01,
    )

    if mode == "φ / cos(θ)":
        cos_theta = np.cos(theta_grid)

        # Fixed square-style layout similar to the original integrated code
        ax = fig.add_axes([0.10, 0.10, 0.72, 0.82])

        cf = ax.contourf(
            phi_grid,
            cos_theta,
            pcs_map.T,
            levels=60,
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
        )
        ax.contour(
            phi_grid,
            cos_theta,
            pcs_map.T,
            levels=_nice_levels(vlim),
            colors="white",
            linewidths=0.5,
            alpha=0.5,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("φ [rad]", fontsize=9)
        ax.set_ylabel(r"$\cos\theta$", fontsize=9)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels(["−π", "−π/2", "0", "π/2", "π"], fontsize=8)
        ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.4)
        ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.4)
        ax.tick_params(labelsize=8)
        ax.set_title(
            rf"$\phi$ / $\cos\theta$ projection   r={r_fixed:.2f} Å   "
            rf"$\Delta\chi_{{ax}}$={dchi_ax:+.2f}   "
            rf"$\Delta\chi_{{rh}}$={dchi_rh:+.2f}",
            fontsize=9,
        )

        cax = fig.add_axes([0.85, 0.10, 0.03, 0.82])
        cb = fig.colorbar(cf, cax=cax)
        cb.set_label("PCS [ppm]", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        if show_atoms:
            coords, labels, ref_ids = _get_projection_coords(state)
            if coords is not None:
                r_atom = np.linalg.norm(coords, axis=1)
                r_atom = np.where(r_atom > 1e-6, r_atom, 1e-6)
                theta_atom = np.arccos(np.clip(coords[:, 2] / r_atom, -1.0, 1.0))
                phi_atom = np.arctan2(coords[:, 1], coords[:, 0])

                for i, label in enumerate(labels):
                    if (not show_h) and label and label[0] == "H":
                        continue
                    pt = ax.scatter(
                        phi_atom[i],
                        np.cos(theta_atom[i]),
                        color=_get_atom_color(label),
                        s=30,
                        zorder=5,
                        edgecolors="white",
                        linewidths=0.4,
                        picker=True,
                    )
                    click_pairs.append((pt, ref_ids[i]))

    else:
        lat = np.pi / 2.0 - TH

        ax = fig.add_subplot(111, projection="mollweide")
        cf = ax.contourf(
            PH,
            lat,
            pcs_map,
            levels=60,
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
        )
        ax.contour(
            PH,
            lat,
            pcs_map,
            levels=_nice_levels(vlim),
            colors="white",
            linewidths=0.5,
            alpha=0.5,
        )

        ax.set_xlabel("φ [rad]", fontsize=8)
        ax.set_ylabel(r"$\frac{\pi}{2}-\theta$ [rad]", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            rf"Mollweide projection   r={r_fixed:.2f} Å   "
            rf"$\Delta\chi_{{ax}}$={dchi_ax:+.2f}   "
            rf"$\Delta\chi_{{rh}}$={dchi_rh:+.2f}",
            fontsize=9,
        )

        cb = fig.colorbar(cf, ax=ax, fraction=0.025, pad=0.05)
        cb.set_label("PCS [ppm]", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        if show_atoms:
            coords, labels, ref_ids = _get_projection_coords(state)
            if coords is not None:
                r_atom = np.linalg.norm(coords, axis=1)
                r_atom = np.where(r_atom > 1e-6, r_atom, 1e-6)
                theta_atom = np.arccos(np.clip(coords[:, 2] / r_atom, -1.0, 1.0))
                phi_atom = np.arctan2(coords[:, 1], coords[:, 0])
                lat_atom = np.pi / 2.0 - theta_atom

                for i, label in enumerate(labels):
                    if (not show_h) and label and label[0] == "H":
                        continue
                    pt = ax.scatter(
                        phi_atom[i],
                        lat_atom[i],
                        color=_get_atom_color(label),
                        s=26,
                        zorder=5,
                        edgecolors="white",
                        linewidths=0.4,
                        picker=True,
                    )
                    click_pairs.append((pt, ref_ids[i]))

    old_cid = state.get("projection_click_cid")
    if old_cid is not None:
        try:
            fig.canvas.mpl_disconnect(old_cid)
        except Exception:
            pass
        state["projection_click_cid"] = None

    def _on_click(event):
        row_by_id = state.get("row_by_id", {})
        tree = state.get("tree")
        if tree is None:
            return

        for artist, ref_id in click_pairs:
            contains, _ = artist.contains(event)
            if contains:
                item = row_by_id.get(ref_id)
                if item:
                    tree.selection_set(item)
                    tree.focus(item)
                    tree.see(item)
                break

    state["projection_click_cid"] = fig.canvas.mpl_connect("button_press_event", _on_click)

    try:
        if mode != "φ / cos(θ)":
            fig.tight_layout(pad=0.3)
    except Exception:
        pass

    canvas.draw_idle()


def _export_projection_figure(state):
    fig = state.get("projection_figure")
    if fig is None:
        state["messagebox"].showerror("Projection", "No figure available.")
        return

    path = state["filedialog"].asksaveasfilename(
        title="Export projection figure",
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
        state["messagebox"].showinfo("Projection", f"Saved:\n{path}")
    except Exception as exc:
        state["messagebox"].showerror("Projection", f"Export failed:\n{exc}")