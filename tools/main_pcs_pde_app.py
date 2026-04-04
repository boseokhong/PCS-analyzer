# tools/main_pcs_pde_app.py
"""
PCS-PDE Application — main entry point (FFT).
"""

from __future__ import annotations

import csv
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import numpy as np

try:
    import pyvista as pv
except Exception:
    pv = None

from logic.xyz_loader import load_orca_data, pick_orca_tensor_at_temperature
from tools.logic_spindens_loader import load_spindens_3d
from tools.ui_pcs_pde_control import ControlPanel, StatusBar
from tools.ui_pcs_pde_viewer import (
    compute_pcs_pde_result,
    open_or_refresh_pcs_pde_view,
    export_pcs_pde_png,
    close_pcs_pde_view,
    show_oblique_pcs_slice_plot,
)
from tools.logic_pcs_pde import rank2_chi, PYFFTW_AVAILABLE, PYFFTW_THREADS

AVOGADRO = 6.02214129e23


def convert_orca_chi_to_angstrom3(chi_raw: np.ndarray, temperature: float) -> np.ndarray:
    chi_raw = np.asarray(chi_raw, dtype=float)
    T = float(temperature)
    if T <= 0.0:
        raise ValueError("Temperature must be positive.")
    return (4.0 * np.pi * 1.0e24 * chi_raw) / (AVOGADRO * T)


class InfoPanel(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self._text = tk.Text(
            frame,
            state="disabled",
            wrap="word",
            font=("Consolas", 8),
            relief="flat",
            background="#f8f8f8",
            foreground="#222222",
            borderwidth=0,
            padx=8,
            pady=6,
        )
        sb = ttk.Scrollbar(frame, command=self._text.yview)
        self._text.configure(yscrollcommand=sb.set)
        self._text.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        self.set_text("No files loaded.\n\nUse File → Open ORCA output to begin.")

    def set_text(self, text: str):
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("end", text)
        self._text.configure(state="disabled")

    def update_from_session(self, session: "AppSession"):
        lines: list[str] = []

        def _sep(title: str = ""):
            lines.append("\n" + "─" * 54 + "\n")
            if title:
                lines.append(f" {title}\n")
                lines.append("─" * 54 + "\n")

        if session.orca_path or session.dens_path:
            _sep("Files")
            if session.orca_path:
                lines.append(f"  ORCA : {session.orca_path}\n")
            if session.dens_path:
                lines.append(f"  Dens : {session.dens_path}\n")

        if session.orca is not None and session.orca.get("tensors_by_temp"):
            temps = sorted(session.orca["tensors_by_temp"].keys())
            _sep("Available Temperatures")
            lines.append(f"  Count : {len(temps)}\n")
            lines.append(f"  Range : {temps[0]:g} – {temps[-1]:g} K\n")
            if len(temps) <= 10:
                lines.append("  Values: " + ", ".join(f"{t:g}" for t in temps) + " K\n")
            else:
                head = ", ".join(f"{t:g}" for t in temps[:5])
                tail = ", ".join(f"{t:g}" for t in temps[-3:])
                lines.append(f"  Values: {head}, …, {tail} K\n")

        if session.temperature is not None:
            lines.append(f"\n  → Selected : {session.temperature:g} K\n")

        _sep("Susceptibility Tensors")

        raw = None
        converted = None
        used_temp = None
        r2 = None

        if session.last_result is not None:
            used_temp = session.last_result.get("temperature", session.temperature)
            raw = session.last_result.get("chi_raw", session.chi_raw)
            converted = session.last_result.get("chi_converted", session.chi)
            r2 = session.last_result.get("chi_rank2", None)
        else:
            used_temp = session.temperature
            raw = session.chi_raw
            converted = session.chi
            if converted is not None:
                r2 = rank2_chi(converted)

        if used_temp is not None:
            lines.append(f"  Temperature used in current setup : {float(used_temp):.6g} K\n\n")

        if raw is not None:
            lines.append("  [1] Raw from ORCA output  (cm³·K/mol, CGS molar Curie)\n\n")
            _fmt_mat(lines, np.asarray(raw, dtype=float), indent=6)
            lines.append("\n")

        if converted is not None:
            lines.append("  [2] Converted  (Å³/molecule, Gaussian convention)\n")
            lines.append("      formula: 4π × 1e24 × χ_raw / (N_A × T)\n\n")
            _fmt_mat(lines, np.asarray(converted, dtype=float), indent=6)
            lines.append("\n")

        if r2 is not None:
            tr_in = float(np.trace(np.asarray(converted, dtype=float))) if converted is not None else np.nan
            tr_out = float(np.trace(np.asarray(r2, dtype=float)))
            lines.append("  [3] Rank-2 traceless part  (used in PCS calculation)\n")
            lines.append("      χ_rank2 = χ − (Tr(χ)/3)·I\n")
            lines.append(f"      Tr(χ_converted) = {tr_in:.6e} Å³\n")
            lines.append(f"      Tr(χ_rank2)     = {tr_out:.2e}  (≈0 expected)\n\n")
            _fmt_mat(lines, np.asarray(r2, dtype=float), indent=6)
            lines.append("\n")

            an = _anisotropy_summary(np.asarray(r2, dtype=float))
            vals = np.asarray(an["eigvals"], dtype=float)
            vecs = np.asarray(an["eigvecs"], dtype=float)

            lines.append("  [4] Eigenvalues in Mehring order  |xx| < |yy| < |zz|\n\n")
            lines.append(f"      eig_1 = {vals[0]:+14.6e} Å³    ({_to_e32_m3(vals[0]):+12.6f} ×10^-32 m³)\n")
            lines.append(f"      eig_2 = {vals[1]:+14.6e} Å³    ({_to_e32_m3(vals[1]):+12.6f} ×10^-32 m³)\n")
            lines.append(f"      eig_3 = {vals[2]:+14.6e} Å³    ({_to_e32_m3(vals[2]):+12.6f} ×10^-32 m³)\n\n")

            lines.append("  [5] Magnetic Susceptibility Anisotropy Tensor - axial representation\n")
            lines.append("      ax_1 = 3*eigvals(3)/2\n")
            lines.append("      rh_1 = (eigvals(1)-eigvals(2))/2\n\n")
            lines.append(f"      ax_1 = {an['ax_1']:+14.6e} Å³    ({_to_e32_m3(an['ax_1']):+12.6f} ×10^-32 m³)\n")
            lines.append(f"      rh_1 = {an['rh_1']:+14.6e} Å³    ({_to_e32_m3(an['rh_1']):+12.6f} ×10^-32 m³)\n\n")

            lines.append("  [6] Magnetic Susceptibility Anisotropy Tensor - general representation\n")
            lines.append("      ax_2 = eigvals(3)-((eigvals(1)+eigvals(2))/2)\n")
            lines.append("      rh_2 = eigvals(1)-eigvals(2)\n\n")
            lines.append(f"      ax_2 = {an['ax_2']:+14.6e} Å³    ({_to_e32_m3(an['ax_2']):+12.6f} ×10^-32 m³)\n")
            lines.append(f"      rh_2 = {an['rh_2']:+14.6e} Å³    ({_to_e32_m3(an['rh_2']):+12.6f} ×10^-32 m³)\n\n")

            lines.append("  [7] Rhombicity ratio and isotropic part\n")
            lines.append("      rh_rel = abs((eigvals(1)-eigvals(2))/eigvals(1))\n")
            lines.append("      iso    = trace(chi)/3\n\n")
            lines.append(f"      rh_rel = {an['rh_rel']:.6f}\n")
            lines.append(f"      iso    = {an['iso']:+14.6e} Å³    ({_to_e32_m3(an['iso']):+12.6f} ×10^-32 m³)\n\n")

            lines.append("  [8] Principal axes (columns = eigenvectors for eig_1, eig_2, eig_3)\n\n")
            _fmt_mat(lines, vecs, indent=6)
            lines.append("\n")

        lines.append("  [Use in solver]\n")
        lines.append("      PDE   : FFT spectral solver\n")
        lines.append("      Point : χ_rank2 applied inside point_pcs_from_tensor()\n")
        if PYFFTW_AVAILABLE:
            lines.append(f"      FFT   : pyfftw  ({PYFFTW_THREADS} threads)\n")
        else:
            lines.append("      FFT   : numpy.fft  (pyfftw not installed)\n")

        if session.dens is not None:
            _sep("Spin Density Grid")
            rho = session.dens["rho"]
            ox, oy, oz = np.asarray(session.dens["origin"], dtype=float)
            dx, dy, dz = session.dens["spacing"]

            lines.append(f"  Shape   : {session.dens['shape']}\n")
            lines.append(f"  Origin  : ({ox:.4f}, {oy:.4f}, {oz:.4f}) Å\n")
            lines.append(f"  Spacing : ({dx:.4f}, {dy:.4f}, {dz:.4f}) Å\n")
            lines.append(f"  ρ min   : {rho.min():.4e}\n")
            lines.append(f"  ρ max   : {rho.max():.4e}\n")
            lines.append(f"  ρ |max| : {np.max(np.abs(rho)):.4e}\n")

        if session.last_result is not None:
            pcs = session.last_result["pcs_field"]
            _sep("PCS Field Result")
            lines.append(f"  min  : {np.nanmin(pcs):.6f} ppm\n")
            lines.append(f"  max  : {np.nanmax(pcs):.6f} ppm\n")
            lines.append(f"  |max|: {np.nanmax(np.abs(pcs)):.6f} ppm\n")

            for k in ("temperature", "fft_pad_factor", "normalize_density", "normalization_target"):
                if session.last_result.get(k) is not None:
                    lines.append(f"  {k} : {session.last_result[k]}\n")

            atom_rows = session.last_result.get("atom_rows") or []
            if atom_rows:
                _sep("PCS Atom Comparison")
                header = (
                    f"  {'Ref':>4}  {'Atom':<4}  "
                    f"{'X':>9}  {'Y':>9}  {'Z':>9}  "
                    f"{'PCS_PDE':>10}  {'PCS_Point':>10}  {'Δ(PDE-Pt)':>10}\n"
                )
                lines.append(header)
                lines.append("  " + "-" * 80 + "\n")

                for row in atom_rows:
                    pde_s = f"{row['pcs_pde']:.4f}" if np.isfinite(row["pcs_pde"]) else "nan"
                    pt_s = f"{row['pcs_point']:.4f}" if np.isfinite(row["pcs_point"]) else "nan"
                    dlt_s = f"{row['delta']:.4f}" if np.isfinite(row["delta"]) else "nan"
                    lines.append(
                        f"  {row['ref']:>4}  {row['atom']:<4}  "
                        f"{row['x']:>9.4f}  {row['y']:>9.4f}  {row['z']:>9.4f}  "
                        f"{pde_s:>10}  {pt_s:>10}  {dlt_s:>10}\n"
                    )

        self.set_text("".join(lines) if lines else "No data loaded.")


def _fmt_mat(lines: list[str], mat: np.ndarray, indent: int = 4) -> None:
    pad = " " * indent
    for row in mat:
        lines.append(pad + "  ".join(f"{v:+14.6e}" for v in row) + "\n")


def _mehring_eigensystem(chi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigen-decomposition in Mehring order: |xx| < |yy| < |zz|
    Returns
    -------
    eigvals, eigvecs
        eigvals : shape (3,)
        eigvecs : shape (3,3), columns correspond to eigvals
    """
    vals, vecs = np.linalg.eigh(np.asarray(chi, dtype=float))
    order = np.argsort(np.abs(vals))
    vals = vals[order]
    vecs = vecs[:, order]
    return vals, vecs


def _anisotropy_summary(chi: np.ndarray) -> dict:
    """
    Build anisotropy summary from a 3x3 susceptibility tensor.
    Uses the formulas requested by the user.
    """
    chi = np.asarray(chi, dtype=float)
    vals, vecs = _mehring_eigensystem(chi)

    e1, e2, e3 = [float(v) for v in vals]

    # axial representation
    ax_1 = 3.0 * e3 / 2.0
    rh_1 = (e1 - e2) / 2.0

    # general representation
    ax_2 = e3 - ((e1 + e2) / 2.0)
    rh_2 = (e1 - e2)

    # rhombicity ratio
    rh_rel = abs((e1 - e2) / e1) if abs(e1) > 1e-20 else np.nan

    # isotropic part
    iso = float(np.trace(chi) / 3.0)

    return {
        "eigvals": vals,
        "eigvecs": vecs,
        "ax_1": float(ax_1),
        "rh_1": float(rh_1),
        "ax_2": float(ax_2),
        "rh_2": float(rh_2),
        "rh_rel": float(rh_rel) if np.isfinite(rh_rel) else np.nan,
        "iso": float(iso),
    }


def _to_e32_m3(value_angstrom3: float) -> float:
    """
    Convert Å^3 to units of 1e-32 m^3.
    1 Å^3 = 1e-30 m^3 = 100 * 1e-32 m^3
    """
    return float(value_angstrom3) * 100.0


def _tensor_view_defaults() -> dict:
    return {
        "show_atoms": False,
        "show_bonds": True,
        "show_labels": False,
        "show_grid": False,
        "surface_style": "mesh",   # surface / mesh / both
        "opacity": 0.20,
        "surface_color": "#AAAAAA",
        "positive_axis_color": "#FF0000",
        "negative_axis_color": "#0000FF",
        "background_color": "white",
        "tensor_scale": 3.2,

        "camera_preset": "iso",    # iso / xy / xz / yz
        "png_dpi": 600,
        "png_width_inch": 6.0,
        "png_transparent": False,

        "gif_temp_start": "",
        "gif_temp_end": "",
        "gif_temp_step": "1",

        # scaling mode:
        # per_frame : each frame normalized independently
        # global    : common scale across selected temperature series
        # absolute  : use actual tensor magnitude with a display factor
        "tensor_scaling_mode": "absolute",

        # used only when tensor_scaling_mode == "absolute"
        # lengths = absolute_scale_factor * |eigval|_(in 1e-32 m^3 units)
        "absolute_scale_factor": 0.10,

        # structure-based fitting parameters for absolute mode
        "absolute_ref_tensor_e32": 5.0,
        "absolute_target_fraction": 0.35,

        "window_size": (900, 760),
    }

def _compute_structure_camera_state(
    atoms: list[tuple],
    preset: str = "iso",
    margin: float = 1.25,
) -> dict:
    coords = np.array([[x, y, z] for _, x, y, z in atoms], dtype=float)
    if len(coords) == 0:
        raise ValueError("No atoms available for camera fit.")

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = 0.5 * (mins + maxs)
    span_vec = maxs - mins
    span = float(np.max(span_vec))
    if span < 1e-8:
        span = 1.0

    dist = float(span * margin * 2.2)

    preset = str(preset).strip().lower()
    if preset == "xy":
        direction = np.array([0.0, 0.0, 1.0], dtype=float)
        view_up = np.array([0.0, 1.0, 0.0], dtype=float)
    elif preset == "xz":
        direction = np.array([0.0, -1.0, 0.0], dtype=float)
        view_up = np.array([0.0, 0.0, 1.0], dtype=float)
    elif preset == "yz":
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
        view_up = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        direction = np.array([1.0, 1.0, 1.0], dtype=float)
        direction /= np.linalg.norm(direction)
        view_up = np.array([0.0, 0.0, 1.0], dtype=float)

    position = center + direction * dist

    return {
        "camera_position": [tuple(position), tuple(center), tuple(view_up)],
        "position": tuple(position),
        "focal_point": tuple(center),
        "view_up": tuple(view_up),
        "clipping_range": (0.01, dist * 10.0),
        "view_angle": 30.0,
        "parallel_projection": True,
        "parallel_scale": span * margin * 0.60,
        "window_size": None,
    }

def _apply_camera_preset(plotter, preset: str):
    preset = str(preset).lower().strip()
    if preset == "xy":
        plotter.camera_position = "xy"
    elif preset == "xz":
        plotter.camera_position = "xz"
    elif preset == "yz":
        plotter.camera_position = "yz"
    else:
        plotter.camera_position = "iso"

def _save_tensor_plotter_png(plotter, path: str, *, dpi: int = 600, width_inch: float = 6.0, transparent: bool = False):
    if plotter is None:
        raise RuntimeError("No tensor spheroid viewer is open.")

    target_px = int(round(float(width_inch) * int(dpi)))
    img = plotter.screenshot(
        filename=path,
        window_size=(target_px, target_px),
        transparent_background=bool(transparent),
        return_img=False,
    )
    return img

def _capture_plotter_camera_state(plotter) -> dict:
    """
    Capture current camera / window state so it can be restored exactly
    after the scene is rebuilt for each GIF frame.
    """
    if plotter is None:
        raise RuntimeError("Plotter is None.")

    cam = plotter.camera
    if cam is None:
        raise RuntimeError("Plotter camera is not available.")

    state = {
        "camera_position": plotter.camera_position,
        "position": tuple(float(v) for v in cam.GetPosition()),
        "focal_point": tuple(float(v) for v in cam.GetFocalPoint()),
        "view_up": tuple(float(v) for v in cam.GetViewUp()),
        "clipping_range": tuple(float(v) for v in cam.GetClippingRange()),
        "view_angle": float(cam.GetViewAngle()),
        "parallel_projection": bool(cam.GetParallelProjection()),
        "parallel_scale": float(cam.GetParallelScale()),
    }

    try:
        state["window_size"] = tuple(int(v) for v in plotter.window_size)
    except Exception:
        state["window_size"] = None

    return state

def _restore_plotter_camera_state(plotter, state: dict) -> None:
    """
    Restore a camera state captured by _capture_plotter_camera_state().
    """
    if plotter is None or not state:
        return

    try:
        win_size = state.get("window_size")
        if win_size is not None:
            try:
                plotter.window_size = win_size
            except Exception:
                pass

        cam = plotter.camera
        if cam is None:
            return

        try:
            cam.SetPosition(*state["position"])
            cam.SetFocalPoint(*state["focal_point"])
            cam.SetViewUp(*state["view_up"])
        except Exception:
            pass

        try:
            cam.SetViewAngle(float(state["view_angle"]))
        except Exception:
            pass

        try:
            cam.SetParallelProjection(bool(state["parallel_projection"]))
        except Exception:
            pass

        try:
            cam.SetParallelScale(float(state["parallel_scale"]))
        except Exception:
            pass

        try:
            cr = state.get("clipping_range")
            if cr is not None and len(cr) == 2:
                cam.SetClippingRange(float(cr[0]), float(cr[1]))
        except Exception:
            pass

        try:
            plotter.camera_position = state["camera_position"]
        except Exception:
            pass

    except Exception:
        pass

def _save_tensor_temperature_gif(
    *,
    plotter,
    atoms: list[tuple],
    tensors_by_temp: dict[float, np.ndarray],
    metal_xyz: np.ndarray,
    opts: dict,
    path: str,
):
    if plotter is None:
        raise RuntimeError("No tensor spheroid viewer is open.")
    if not tensors_by_temp:
        raise RuntimeError("No temperature-dependent tensors are available.")

    all_temps = sorted(float(t) for t in tensors_by_temp.keys())

    start_raw = str(opts.get("gif_temp_start", "")).strip()
    end_raw = str(opts.get("gif_temp_end", "")).strip()
    step_raw = str(opts.get("gif_temp_step", "1")).strip()

    start = float(start_raw) if start_raw else None
    end = float(end_raw) if end_raw else None

    try:
        step = int(float(step_raw)) if step_raw else 1
    except Exception:
        step = 1
    if step < 1:
        step = 1

    temps = []
    for T in all_temps:
        if start is not None and T < start:
            continue
        if end is not None and T > end:
            continue
        temps.append(T)

    temps = temps[::step]

    if not temps:
        raise RuntimeError("No temperatures fall within the requested GIF range.")

    scaling_mode = str(opts.get("tensor_scaling_mode", "per_frame")).strip().lower()
    global_max_abs = None

    if scaling_mode == "global":
        global_max_abs = _global_max_abs_eigval_from_selected_temps(
            tensors_by_temp,
            temps,
        )

    # --- build first frame once and freeze the exact camera / zoom ---
    first_T = temps[0]
    chi_raw_0 = np.asarray(tensors_by_temp[first_T], dtype=float)
    chi_conv_0 = convert_orca_chi_to_angstrom3(chi_raw_0, first_T)
    chi_r2_0 = rank2_chi(chi_conv_0)

    _draw_tensor_spheroid_scene(
        plotter,
        atoms=atoms,
        chi_r2=chi_r2_0,
        used_temp=first_T,
        opts=opts,
        metal_xyz=metal_xyz,
        global_max_abs=global_max_abs,
    )

    # GIF comparison is visually more stable in parallel projection
    try:
        plotter.camera.SetParallelProjection(True)
    except Exception:
        pass

    try:
        plotter.render()
    except Exception:
        pass

    frozen_camera_state = _compute_structure_camera_state(
        atoms,
        preset=str(opts.get("camera_preset", "iso")),
        margin=1.25,
    )

    plotter.open_gif(path)
    try:
        for T in temps:
            chi_raw = np.asarray(tensors_by_temp[T], dtype=float)
            chi_conv = convert_orca_chi_to_angstrom3(chi_raw, T)
            chi_r2 = rank2_chi(chi_conv)

            _draw_tensor_spheroid_scene(
                plotter,
                atoms=atoms,
                chi_r2=chi_r2,
                used_temp=T,
                opts=opts,
                metal_xyz=metal_xyz,
                global_max_abs=global_max_abs,
            )

            _restore_plotter_camera_state(plotter, frozen_camera_state)

            try:
                plotter.render()
            except Exception:
                pass

            plotter.write_frame()
    finally:
        try:
            if getattr(plotter, "mwriter", None) is not None:
                plotter.mwriter.close()
        except Exception:
            pass

def _safe_axis_lengths_from_eigvals(
    eigvals: np.ndarray,
    base_scale: float = 3.2,
    global_max_abs: float | None = None,
    scaling_mode: str = "per_frame",
    absolute_scale_factor: float = 0.10,
    min_length: float = 0.03,
) -> np.ndarray:
    vals = np.asarray(eigvals, dtype=float)
    mags = np.abs(vals)

    if np.all(mags < 1e-20):
        return np.array([min_length, min_length, min_length], dtype=float)

    mode = str(scaling_mode).strip().lower()

    if mode == "absolute":
        # Convert Å^3 to units of 1e-32 m^3 for more intuitive display scaling
        mags_e32 = mags * 100.0
        lengths = float(absolute_scale_factor) * mags_e32
        lengths = np.maximum(lengths, float(min_length))
        return lengths

    if mode == "global":
        ref = float(abs(global_max_abs)) if global_max_abs is not None else float(np.max(mags))
    else:
        # per_frame fallback
        ref = float(np.max(mags))

    if ref < 1e-20:
        ref = 1.0

    scaled = mags / ref
    scaled = np.clip(scaled, 0.0, None)

    return 0.35 + base_scale * scaled


def _build_tensor_spheroid_surface(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    scale: float = 3.2,
    global_max_abs: float | None = None,
    scaling_mode: str = "per_frame",
    absolute_scale_factor: float = 0.10,
):
    """
    Build an ellipsoid from traceless tensor eigenvalues/eigenvectors.
    Surface point sign is determined from the signed quadratic form in the
    principal-axis frame, allowing + / - colouring.
    """
    eigvals = np.asarray(eigvals, dtype=float)
    eigvecs = np.asarray(eigvecs, dtype=float)

    lengths = _safe_axis_lengths_from_eigvals(
        eigvals,
        base_scale=scale,
        global_max_abs=global_max_abs,
        scaling_mode=scaling_mode,
        absolute_scale_factor=absolute_scale_factor,
    )

    sph = pv.ParametricEllipsoid(
        float(lengths[0]),
        float(lengths[1]),
        float(lengths[2]),
        u_res=96,
        v_res=48,
        w_res=48,
    )

    pts_local = np.asarray(sph.points, dtype=float)

    sgn = np.sign(eigvals)
    sgn[sgn == 0] = 1.0

    a, b, c = [float(v) for v in lengths]
    q = (
        sgn[0] * (pts_local[:, 0] / a) ** 2
        + sgn[1] * (pts_local[:, 1] / b) ** 2
        + sgn[2] * (pts_local[:, 2] / c) ** 2
    )

    sph.points = pts_local @ eigvecs.T
    sph.point_data["tensor_sign"] = q

    return sph, lengths


def _molecular_span_from_atoms(atoms: list[tuple]) -> float:
    coords = np.array([[x, y, z] for _, x, y, z in atoms], dtype=float)
    if len(coords) == 0:
        return 1.0
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = float(np.max(maxs - mins))
    return span if span > 1e-12 else 1.0


def _structure_based_absolute_scale_factor(
    atoms: list[tuple],
    ref_tensor_e32: float = 5.0,
    target_fraction_of_mol_span: float = 0.35,
) -> float:
    mol_span = _molecular_span_from_atoms(atoms)
    ref_tensor_e32 = float(ref_tensor_e32)
    target_fraction_of_mol_span = float(target_fraction_of_mol_span)

    if ref_tensor_e32 <= 1e-12:
        return 0.10
    if target_fraction_of_mol_span <= 0.0:
        target_fraction_of_mol_span = 0.35

    return float(target_fraction_of_mol_span * mol_span / ref_tensor_e32)


def _guess_metal_index_from_atoms(atoms: list[tuple]) -> int:
    elements = [el for el, *_ in atoms]
    preferred = {
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
        "Sc", "Y", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    }
    for i, el in enumerate(elements):
        if el in preferred:
            return i
    return 0


def _add_tensor_molecule(plotter, atoms: list[tuple], show_atoms=True, show_bonds=True, show_labels=False):
    from tools.logic_structure_helpers import get_cpk_color, radius_for_element, calculate_bonds

    coords = np.array([[x, y, z] for _, x, y, z in atoms], dtype=float)
    elements = [el for el, *_ in atoms]

    if show_bonds and len(coords) >= 2:
        try:
            bonds = calculate_bonds(coords, elements)
        except Exception:
            bonds = []

        for i, j in bonds:
            line = pv.Line(coords[i], coords[j], resolution=1)
            tube = line.tube(radius=0.08)
            plotter.add_mesh(tube, color="#555A60", smooth_shading=True)

    if show_atoms:
        for xyz, el in zip(coords, elements):
            sphere = pv.Sphere(
                radius=radius_for_element(el),
                center=xyz,
                theta_resolution=28,
                phi_resolution=28,
            )
            plotter.add_mesh(
                sphere,
                color=get_cpk_color(el),
                smooth_shading=True,
                specular=0.25,
                ambient=0.18,
            )

    if show_labels:
        label_points = np.asarray(coords, dtype=float)
        texts = [f"{i+1}:{el}" for i, el in enumerate(elements)]
        plotter.add_point_labels(
            label_points,
            texts,
            font_size=10,
            point_size=0,
            shape_opacity=0.0,
            always_visible=False,
        )


def _add_tensor_axes_lines(
    plotter,
    eigvecs: np.ndarray,
    eigvals: np.ndarray,
    lengths: np.ndarray,
    center: np.ndarray,
    positive_color: str = "#FF0000",
    negative_color: str = "#0000FF",
):
    eigvecs = np.asarray(eigvecs, dtype=float)
    eigvals = np.asarray(eigvals, dtype=float)
    center = np.asarray(center, dtype=float)

    for i in range(3):
        vec = eigvecs[:, i]
        L = float(lengths[i])

        p1 = center - vec * L
        p2 = center + vec * L

        color = positive_color if eigvals[i] >= 0 else negative_color

        line = pv.Line(p1, p2, resolution=1)
        plotter.add_mesh(line, color=color, line_width=3)

        label_pos = center + vec * (L * 1.02)
        axis_name = ["x", "y", "z"][i]
        sign_txt = "+" if eigvals[i] >= 0 else "-"

        plotter.add_point_labels(
            [label_pos],
            [f"{axis_name}{sign_txt}"],
            font_size=12,
            point_size=0,
            shape_opacity=0.12,
            always_visible=True,
            text_color="black",
        )


def _draw_tensor_spheroid_scene(
    plotter,
    *,
    atoms: list[tuple],
    chi_r2: np.ndarray,
    used_temp: float | None,
    opts: dict,
    metal_xyz: np.ndarray,
    global_max_abs: float | None = None,
):
    chi_r2 = rank2_chi(np.asarray(chi_r2, dtype=float))

    an = _anisotropy_summary(chi_r2)
    eigvals = np.asarray(an["eigvals"], dtype=float)
    eigvecs = np.asarray(an["eigvecs"], dtype=float)

    spheroid, lengths = _build_tensor_spheroid_surface(
        eigvals=eigvals,
        eigvecs=eigvecs,
        scale=float(opts["tensor_scale"]),
        global_max_abs=global_max_abs,
        scaling_mode=str(opts.get("tensor_scaling_mode", "per_frame")),
        absolute_scale_factor=float(opts.get("absolute_scale_factor", 0.10)),
    )
    spheroid.translate(metal_xyz, inplace=True)

    plotter.clear()
    plotter.set_background(str(opts["background_color"]))

    _add_tensor_molecule(
        plotter,
        atoms,
        show_atoms=bool(opts["show_atoms"]),
        show_bonds=bool(opts["show_bonds"]),
        show_labels=bool(opts["show_labels"]),
    )

    style = str(opts["surface_style"])
    opacity = float(opts["opacity"])
    surface_color = str(opts["surface_color"])

    if style in ("surface", "both"):
        plotter.add_mesh(
            spheroid,
            color=surface_color,
            opacity=opacity,
            smooth_shading=True,
            specular=0.25,
        )

    if style in ("mesh", "both"):
        plotter.add_mesh(
            spheroid,
            color=surface_color,
            style="wireframe",
            line_width=1,
            opacity=min(1.0, opacity * 1.2),
        )

    _add_tensor_axes_lines(
        plotter,
        eigvecs=eigvecs,
        eigvals=eigvals,
        lengths=lengths,
        center=metal_xyz,
        positive_color=str(opts["positive_axis_color"]),
        negative_color=str(opts["negative_axis_color"]),
    )

    temp_txt = f"{float(used_temp):g} K" if used_temp is not None else "N/A"
    scale_mode = str(opts.get("tensor_scaling_mode", "per_frame"))
    abs_fac = float(opts.get("absolute_scale_factor", 0.10))

    plotter.add_text(
        "Tensor spheroid (traceless rank-2 tensor)\n"
        f"T = {temp_txt}\n"
        f"scale mode = {scale_mode}\n"
        f"abs. factor = {abs_fac:.3f}\n"
        f"ax₂ = {_to_e32_m3(an['ax_2']):+.3f} ×10^-32 m³\n"
        f"rh₂ = {_to_e32_m3(an['rh_2']):+.3f} ×10^-32 m³\n"
        f"rh_rel = {an['rh_rel']:.4f}",
        position="upper_left",
        font_size=11,
    )

    plotter.add_axes()

    if bool(opts["show_grid"]):
        plotter.show_grid()

    try:
        _apply_camera_preset(plotter, str(opts.get("camera_preset", "iso")))
    except Exception:
        pass

    return an


def _selected_temperatures_from_opts(
    tensors_by_temp: dict[float, np.ndarray],
    opts: dict,
) -> list[float]:
    all_temps = sorted(float(t) for t in tensors_by_temp.keys())

    start_raw = str(opts.get("gif_temp_start", "")).strip()
    end_raw = str(opts.get("gif_temp_end", "")).strip()
    step_raw = str(opts.get("gif_temp_step", "1")).strip()

    start = float(start_raw) if start_raw else None
    end = float(end_raw) if end_raw else None

    try:
        step = int(float(step_raw)) if step_raw else 1
    except Exception:
        step = 1
    if step < 1:
        step = 1

    temps = []
    for T in all_temps:
        if start is not None and T < start:
            continue
        if end is not None and T > end:
            continue
        temps.append(T)

    return temps[::step]


def _global_max_abs_eigval_from_selected_temps(
    tensors_by_temp: dict[float, np.ndarray],
    temps: list[float],
) -> float:
    vals_all = []

    for T in temps:
        chi_raw = np.asarray(tensors_by_temp[T], dtype=float)
        chi_conv = convert_orca_chi_to_angstrom3(chi_raw, T)
        chi_r2 = rank2_chi(chi_conv)
        eigvals = np.linalg.eigvalsh(chi_r2)
        vals_all.extend(np.abs(eigvals).tolist())

    if not vals_all:
        return 1.0

    vmax = float(np.max(vals_all))
    return vmax if vmax > 1e-20 else 1.0


class AppSession:
    orca_path: Optional[str] = None
    dens_path: Optional[str] = None
    orca: Optional[dict] = None
    dens: Optional[dict] = None
    temperature: Optional[float] = None
    chi_raw: Optional[np.ndarray] = None
    chi: Optional[np.ndarray] = None
    last_result: Optional[dict] = None
    last_view_params: Optional[dict] = None
    viewer_plotter = None


class AppWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("PCS-PDE Viewer")
        self.geometry("1040x760")
        self.minsize(760, 520)

        self._session = AppSession()
        self._compute_thread: Optional[threading.Thread] = None

        self._tensor_view_opts = _tensor_view_defaults()
        self._tensor_view_plotter = None
        self._tensor_view_ctrl = None

        self._build_menu()
        self._build_body()
        self._build_statusbar()
        self._update_info()

    def _build_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open ORCA output…", command=self._load_orca)
        file_menu.add_command(label="Open spin density…", command=self._load_dens)
        file_menu.add_separator()
        file_menu.add_command(label="Export PCS to NumPy…", command=self._export_npy)
        file_menu.add_command(label="Export atom PCS to CSV…", command=self._export_atom_pcs_csv)
        file_menu.add_command(label="Export PNG…", command=lambda: self._export_png(self._ctrl.get_params()))
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self._on_quit)
        menubar.add_cascade(label="File", menu=file_menu)

        run_menu = tk.Menu(menubar, tearoff=False)
        run_menu.add_command(label="Run computation (▶)", command=lambda: self._on_run(self._ctrl.get_params()))
        run_menu.add_command(label="Open / Refresh Viewer", command=lambda: self._refresh_viewer(self._ctrl.get_params()))
        run_menu.add_separator()
        run_menu.add_command(label="Open Oblique PCS Slice…", command=self._open_oblique_slice_dialog)
        menubar.add_cascade(label="Run", menu=run_menu)

        plots_menu = tk.Menu(menubar, tearoff=False)
        plots_menu.add_command(label="PDE vs Point PCS", command=self._plot_pde_vs_point)
        plots_menu.add_command(label="Residual (PDE - Point)", command=self._plot_pde_minus_point)
        plots_menu.add_separator()
        plots_menu.add_command(label="Tensor spheroid", command=self._open_tensor_spheroid_options)
        menubar.add_cascade(label="Plots", menu=plots_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _build_body(self):
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        left_frame = ttk.LabelFrame(paned, text="Parameters", padding=0)
        self._ctrl = ControlPanel(
            left_frame,
            on_run_callback=self._on_run,
            on_refresh_view_callback=self._refresh_viewer,
            on_export_png_callback=self._export_png,
        )
        self._ctrl.pack(fill="both", expand=True)
        paned.add(left_frame, weight=0)

        right_frame = ttk.LabelFrame(paned, text="Session", padding=0)
        self._info = InfoPanel(right_frame)
        self._info.pack(fill="both", expand=True)
        paned.add(right_frame, weight=1)

    def _build_statusbar(self):
        self._status = StatusBar(self)
        self._status.pack(side="bottom", fill="x")

    def _update_info(self):
        self._info.update_from_session(self._session)

    def _show_about(self):
        messagebox.showinfo(
            "About PCS-PDE Viewer",
            "PCS-PDE Viewer\n\n"
            "FFT PDE comparison mode.\n"
            "Tensor conversion :\n"
            "χ = 4π × 1e24 × χ_raw / (N_A × T)"
        )

    def _load_orca(self):
        path = filedialog.askopenfilename(
            title="Select ORCA output file",
            filetypes=[("ORCA output", "*.out *.log"), ("All files", "*.*")],
        )
        if not path:
            return

        self._status.set(f"Loading {path} …")
        self.update_idletasks()

        try:
            orca = load_orca_data(path)
        except Exception as exc:
            messagebox.showerror("ORCA load error", f"Failed to read ORCA file:\n{exc}")
            self._status.set("ORCA load failed.")
            return

        if not orca["atoms"]:
            messagebox.showerror("ORCA load error", "No atomic coordinates found.")
            return
        if not orca["tensors_by_temp"]:
            messagebox.showerror("ORCA load error", "No susceptibility tensors found.")
            return

        self._session.orca_path = path
        self._session.orca = orca

        temps = sorted(orca["tensors_by_temp"].keys())
        self._ctrl.set_temperatures(temps)

        t, chi_raw = pick_orca_tensor_at_temperature(orca["tensors_by_temp"], None)
        self._session.temperature = t
        self._session.chi_raw = np.asarray(chi_raw, dtype=float)
        self._session.chi = convert_orca_chi_to_angstrom3(chi_raw, t)

        self._status.set(
            f"ORCA loaded: {len(orca['atoms'])} atoms, {len(temps)} temperature(s) "
            f"[{temps[0]:g}–{temps[-1]:g} K]"
        )
        self._update_info()

    def _load_dens(self):
        path = filedialog.askopenfilename(
            title="Select spin density file",
            filetypes=[("Spin density grid", "*.3d"), ("All files", "*.*")],
        )
        if not path:
            return

        self._status.set(f"Loading {path} …")
        self.update_idletasks()

        try:
            dens = load_spindens_3d(path)
        except Exception as exc:
            messagebox.showerror("Density load error", f"Failed to read spindens.3d:\n{exc}")
            self._status.set("Density load failed.")
            return

        self._session.dens_path = path
        self._session.dens = dens

        nx, ny, nz = dens["shape"]
        self._status.set(
            f"Density loaded: {nx}×{ny}×{nz} grid, "
            f"ρ ∈ [{dens['rho'].min():.2e}, {dens['rho'].max():.2e}]"
        )
        self._update_info()

    def _prepare_chi_for_params(self, params: dict):
        if self._session.orca is None:
            raise ValueError("Please load an ORCA output file first.")
        temp_val = params.get("temperature")
        t, chi_raw = pick_orca_tensor_at_temperature(
            self._session.orca["tensors_by_temp"],
            temperature=temp_val,
        )
        chi_converted = convert_orca_chi_to_angstrom3(chi_raw, t)
        return t, np.asarray(chi_raw, dtype=float), np.asarray(chi_converted, dtype=float)

    def _on_run(self, params: Optional[dict] = None):
        if params is None:
            params = self._ctrl.get_params()

        if self._session.orca is None:
            messagebox.showwarning("No ORCA data", "Please load an ORCA output file first.")
            return
        if self._session.dens is None:
            messagebox.showwarning("No density data", "Please load a spin density file first.")
            return
        if self._compute_thread and self._compute_thread.is_alive():
            messagebox.showinfo("Busy", "A computation is already running. Please wait.")
            return

        try:
            t, chi_raw, chi_converted = self._prepare_chi_for_params(params)
        except Exception as exc:
            messagebox.showerror("Tensor error", str(exc))
            return

        self._session.temperature = t
        self._session.chi_raw = chi_raw
        self._session.chi = chi_converted
        self._session.last_view_params = dict(params)

        try:
            dens = load_spindens_3d(self._session.dens_path)
            self._session.dens = dens
        except Exception as exc:
            messagebox.showerror("Density reload error", str(exc))
            return

        origin_ang = np.asarray(dens["origin"], dtype=float)
        dx, dy, dz = dens["spacing"]
        spacing_ang = (float(dx), float(dy), float(dz))

        self._update_info()
        self._status.start_busy(f"Computing PCS at {t:g} K…")

        def _worker():
            try:
                result = compute_pcs_pde_result(
                    atoms=self._session.orca["atoms"],
                    origin=origin_ang,
                    spacing=spacing_ang,
                    rho=self._session.dens["rho"],
                    chi_tensor=self._session.chi,
                    params=params,
                    chi_raw=self._session.chi_raw,
                    temperature=self._session.temperature,
                )
                self._session.last_result = result
                self.after(0, lambda: self._on_compute_done(success=True, params=params))
            except Exception as exc:
                print(traceback.format_exc())
                self.after(0, lambda: self._on_compute_done(success=False, error=str(exc), params=params))

        self._compute_thread = threading.Thread(target=_worker, daemon=True)
        self._compute_thread.start()

    def _on_compute_done(self, success: bool, error: str = "", params: Optional[dict] = None):
        self._update_info()
        if success:
            pcs = self._session.last_result["pcs_field"]
            self._status.stop_busy(
                f"Done. PCS ∈ [{np.nanmin(pcs):.3f}, {np.nanmax(pcs):.3f}] ppm"
            )

            if self._session.viewer_plotter is not None:
                if not self._is_plotter_alive(self._session.viewer_plotter):
                    self._session.viewer_plotter = None

            if self._session.viewer_plotter is not None:
                try:
                    self._session.viewer_plotter = open_or_refresh_pcs_pde_view(
                        self._session.last_result,
                        params or self._session.last_view_params or self._ctrl.get_params(),
                        plotter=self._session.viewer_plotter,
                    )
                except Exception as exc:
                    self._session.viewer_plotter = None
                    messagebox.showerror("Viewer refresh error", str(exc))
        else:
            self._status.stop_busy("Computation failed.")
            messagebox.showerror("Computation error", f"PCS computation failed:\n{error}")

    def _refresh_viewer(self, params: Optional[dict] = None):
        if params is None:
            params = self._ctrl.get_params()

        if self._session.last_result is None:
            messagebox.showinfo("No result", "Run a computation first.")
            return

        self._session.last_view_params = dict(params)

        if not self._is_plotter_alive(self._session.viewer_plotter):
            self._session.viewer_plotter = None

        try:
            self._session.viewer_plotter = open_or_refresh_pcs_pde_view(
                self._session.last_result,
                params,
                plotter=self._session.viewer_plotter,
            )
            self._status.set("Viewer opened/refreshed.")
        except Exception as exc:
            self._session.viewer_plotter = None
            messagebox.showerror("Viewer error", str(exc))

    def _export_png(self, params: Optional[dict] = None):
        if params is None:
            params = self._ctrl.get_params()

        if self._session.last_result is None:
            messagebox.showinfo("Nothing to export", "Run a computation first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save PNG image",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            export_pcs_pde_png(
                self._session.last_result,
                params,
                path,
                dpi=int(params.get("png_dpi", 600)),
                width_inch=float(params.get("png_width_inch", 6.0)),
                transparent=bool(params.get("png_transparent", False)),
            )
            self._status.set(f"PNG exported: {path}")
        except Exception as exc:
            messagebox.showerror("Export PNG error", str(exc))

    def _export_npy(self, params: Optional[dict] = None):
        if self._session.last_result is None:
            messagebox.showinfo("Nothing to export", "Run a computation first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save PCS field",
            defaultextension=".npz",
            filetypes=[("NumPy archive", "*.npz"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            np.savez_compressed(
                path,
                pcs_field=self._session.last_result["pcs_field"],
                source_term=self._session.last_result["source_term"],
                origin=self._session.last_result["origin"],
                spacing=np.array(self._session.last_result["spacing"]),
                temperature=self._session.last_result.get("temperature"),
                chi_raw=self._session.last_result.get("chi_raw"),
                chi_converted=self._session.last_result.get("chi_converted"),
                chi_rank2=self._session.last_result.get("chi_rank2"),
                fft_pad_factor=self._session.last_result.get("fft_pad_factor"),
                normalize_density=self._session.last_result.get("normalize_density"),
                normalization_target=self._session.last_result.get("normalization_target"),
            )
            self._status.set(f"Exported: {path}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    def _export_atom_pcs_csv(self):
        if self._session.last_result is None:
            messagebox.showinfo("Nothing to export", "Run a computation first.")
            return

        rows = self._session.last_result.get("atom_rows")
        if not rows:
            messagebox.showinfo("Nothing to export", "No atom PCS rows are available.")
            return

        path = filedialog.asksaveasfilename(
            title="Save atom PCS comparison",
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow([
                    "Ref", "Atom", "X", "Y", "Z",
                    "PCS_PDE_ppm", "PCS_Point_ppm", "Delta_PDE_minus_Point_ppm",
                ])
                for row in rows:
                    w.writerow([
                        row["ref"], row["atom"],
                        row["x"], row["y"], row["z"],
                        row["pcs_pde"], row["pcs_point"], row["delta"],
                    ])
            self._status.set(f"Exported: {path}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    def _plot_pde_vs_point(self):
        if self._session.last_result is None:
            messagebox.showinfo("No result", "Run a computation first.")
            return

        rows = self._session.last_result.get("atom_rows") or []
        if not rows:
            messagebox.showinfo("No data", "No atom PCS rows are available.")
            return

        import matplotlib.pyplot as plt

        x = np.array([r["pcs_point"] for r in rows], dtype=float)
        y = np.array([r["pcs_pde"] for r in rows], dtype=float)
        labels = [str(r["ref"]) for r in rows]

        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            messagebox.showinfo("No data", "No valid PCS values found.")
            return

        x = x[valid]
        y = y[valid]
        labels = [lab for lab, ok in zip(labels, valid) if ok]

        diff = y - x
        rmsd = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))

        lo = min(float(np.min(x)), float(np.min(y)))
        hi = max(float(np.max(x)), float(np.max(y)))
        span = hi - lo
        pad = 0.05 * span if span > 1e-12 else 1.0

        fig, ax = plt.subplots(figsize=(5.4, 5.2))
        ax.scatter(x, y, s=34)

        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            "--",
            linewidth=1.0,
            label="y = x",
        )

        for xi, yi, lab in zip(x, y, labels):
            ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7)

        ax.set_xlabel("Point-dipole PCS / ppm")
        ax.set_ylabel("PDE PCS / ppm")
        ax.set_title(f"PDE vs Point PCS\nRMSD = {rmsd:.4f} ppm   MAE = {mae:.4f} ppm")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plt.show()

    def _plot_pde_minus_point(self):
        if self._session.last_result is None:
            messagebox.showinfo("No result", "Run a computation first.")
            return

        rows = self._session.last_result.get("atom_rows") or []
        if not rows:
            messagebox.showinfo("No data", "No atom PCS rows are available.")
            return

        import matplotlib.pyplot as plt

        x = np.array([r["pcs_point"] for r in rows], dtype=float)
        d = np.array([r["delta"] for r in rows], dtype=float)
        labels = [str(r["ref"]) for r in rows]

        valid = np.isfinite(x) & np.isfinite(d)
        if not np.any(valid):
            messagebox.showinfo("No data", "No valid residual values found.")
            return

        x = x[valid]
        d = d[valid]
        labels = [lab for lab, ok in zip(labels, valid) if ok]

        rmsd = float(np.sqrt(np.mean(d ** 2)))
        max_abs = float(np.max(np.abs(d)))

        fig, ax = plt.subplots(figsize=(5.4, 5.2))
        ax.scatter(x, d, s=34)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)

        for xi, di, lab in zip(x, d, labels):
            ax.annotate(lab, (xi, di), textcoords="offset points", xytext=(4, 4), fontsize=7)

        ax.set_xlabel("Point-dipole PCS / ppm")
        ax.set_ylabel("PDE - Point / ppm")
        ax.set_title(f"Residual plot\nRMSD = {rmsd:.4f} ppm   max|Δ| = {max_abs:.4f} ppm")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

    def _open_tensor_spheroid_options(self):
        if pv is None:
            messagebox.showerror(
                "PyVista not available",
                "PyVista could not be imported.\nInstall with:\n\npip install pyvista vtk"
            )
            return

        old = getattr(self, "_tensor_view_ctrl", None)
        if old is not None:
            try:
                if old.winfo_exists():
                    old.lift()
                    old.focus_force()
                    return
            except Exception:
                pass

        win = tk.Toplevel(self)
        win.title("Tensor spheroid controls")
        win.geometry("380x760")
        win.resizable(True, True)
        win.transient(self)
        self._tensor_view_ctrl = win

        opts = self._tensor_view_opts

        vars_ = {
            "show_atoms": tk.BooleanVar(value=opts["show_atoms"]),
            "show_bonds": tk.BooleanVar(value=opts["show_bonds"]),
            "show_labels": tk.BooleanVar(value=opts["show_labels"]),
            "show_grid": tk.BooleanVar(value=opts["show_grid"]),
            "surface_style": tk.StringVar(value=opts["surface_style"]),
            "opacity": tk.DoubleVar(value=float(opts["opacity"])),
            "surface_color": tk.StringVar(value=opts["surface_color"]),
            "positive_axis_color": tk.StringVar(value=opts["positive_axis_color"]),
            "negative_axis_color": tk.StringVar(value=opts["negative_axis_color"]),
            "background_color": tk.StringVar(value=opts["background_color"]),
            "tensor_scale": tk.DoubleVar(value=float(opts["tensor_scale"])),

            "camera_preset": tk.StringVar(value=opts["camera_preset"]),
            "png_dpi": tk.IntVar(value=int(opts["png_dpi"])),
            "png_width_inch": tk.DoubleVar(value=float(opts["png_width_inch"])),
            "png_transparent": tk.BooleanVar(value=bool(opts["png_transparent"])),

            "gif_temp_start": tk.StringVar(value=str(opts["gif_temp_start"])),
            "gif_temp_end": tk.StringVar(value=str(opts["gif_temp_end"])),
            "gif_temp_step": tk.StringVar(value=str(opts["gif_temp_step"])),

            "tensor_scaling_mode": tk.StringVar(value=str(opts["tensor_scaling_mode"])),
            "absolute_scale_factor": tk.DoubleVar(value=float(opts.get("absolute_scale_factor", 0.10))),
            "absolute_ref_tensor_e32": tk.DoubleVar(value=float(opts.get("absolute_ref_tensor_e32", 5.0))),
            "absolute_target_fraction": tk.DoubleVar(value=float(opts.get("absolute_target_fraction", 0.35))),
        }

        # scrollable container
        canvas = tk.Canvas(win, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        frm = ttk.Frame(canvas, padding=12)
        frm.columnconfigure(1, weight=1)

        frm_id = canvas.create_window((0, 0), window=frm, anchor="nw")

        def _on_frame_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfigure(frm_id, width=event.width)

        frm.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            try:
                if event.num == 4:
                    canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    canvas.yview_scroll(1, "units")
                else:
                    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)

        row = 0

        ttk.Label(frm, text="Display", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        row += 1

        ttk.Checkbutton(frm, text="Show atoms", variable=vars_["show_atoms"]).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(frm, text="Show bonds", variable=vars_["show_bonds"]).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(frm, text="Show labels", variable=vars_["show_labels"]).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(frm, text="Show grid", variable=vars_["show_grid"]).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Label(frm, text="Scaling mode").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Combobox(
            frm,
            textvariable=vars_["tensor_scaling_mode"],
            values=["per_frame", "global", "absolute"],
            state="readonly",
        ).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(frm, text="Absolute factor (absolute mode only)").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=vars_["absolute_scale_factor"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(frm, text="Reference tensor (×10^-32 m^3)").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=vars_["absolute_ref_tensor_e32"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(frm, text="Target size / molecule span").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=vars_["absolute_target_fraction"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        def _fit_absolute_factor_to_molecule():
            try:
                atoms = list(self._session.orca["atoms"]) if self._session.orca is not None else []
                if not atoms:
                    messagebox.showinfo("No structure", "Load an ORCA structure first.")
                    return

                ref_tensor = float(vars_["absolute_ref_tensor_e32"].get())
                target_fraction = float(vars_["absolute_target_fraction"].get())

                fitted = _structure_based_absolute_scale_factor(
                    atoms,
                    ref_tensor_e32=ref_tensor,
                    target_fraction_of_mol_span=target_fraction,
                )
                vars_["absolute_scale_factor"].set(float(fitted))
            except Exception as exc:
                messagebox.showerror("Fit error", str(exc))

        ttk.Button(
            frm,
            text="Fit absolute factor to molecule",
            command=_fit_absolute_factor_to_molecule,
        ).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(2, 6))
        row += 1

        ttk.Label(
            frm,
            text=(
                "per_frame: each frame normalized independently\n"
                "global: one common scale over selected temperature range\n"
                "absolute: true tensor-size comparison\n"
                "Use 'Fit absolute factor to molecule' to set a structure-based display factor."
            ),
            foreground="gray",
            justify="left",
            wraplength=320,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row += 1

        ttk.Label(frm, text="Surface style").grid(row=row, column=0, sticky="w", pady=(8, 2))
        ttk.Combobox(
            frm,
            textvariable=vars_["surface_style"],
            values=["surface", "mesh", "both"],
            state="readonly"
        ).grid(row=row, column=1, sticky="ew", pady=(8, 2))
        row += 1

        ttk.Label(frm, text="Opacity").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=vars_["opacity"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(frm, text="Tensor scale").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=vars_["tensor_scale"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(
            frm,
            text="Opacity: 0.05–1.0   |   Tensor scale: e.g. 1.0–6.0",
            foreground="gray",
            justify="left",
            wraplength=320,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 2))
        row += 1

        ttk.Label(
            frm,
            text=(
                "Tensor scale is used for per_frame/global modes.\n"
                "Absolute factor is used only for absolute mode."
            ),
            foreground="gray",
            justify="left",
            wraplength=320,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 4))
        row += 1

        def _choose_color(var):
            from tkinter import colorchooser
            c = colorchooser.askcolor(color=var.get(), title="Choose color")
            if c and c[1]:
                var.set(c[1])

        for key, label in [
            ("surface_color", "Surface color"),
            ("positive_axis_color", "+ axis color"),
            ("negative_axis_color", "- axis color"),
            ("background_color", "Background color"),
        ]:
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=2)
            rf = ttk.Frame(frm)
            rf.grid(row=row, column=1, sticky="ew", pady=2)
            rf.columnconfigure(0, weight=1)
            ttk.Entry(rf, textvariable=vars_[key]).grid(row=0, column=0, sticky="ew")
            ttk.Button(rf, text="...", width=3, command=lambda v=vars_[key]: _choose_color(v)).grid(row=0, column=1, padx=(4, 0))
            row += 1

        ttk.Separator(frm, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 8))
        row += 1

        ttk.Label(frm, text="Camera", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        row += 1

        ttk.Label(frm, text="Camera preset").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Combobox(
            frm,
            textvariable=vars_["camera_preset"],
            values=["iso", "xy", "xz", "yz"],
            state="readonly",
        ).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Separator(frm, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 8))
        row += 1

        ttk.Label(frm, text="PNG export", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        row += 1

        ttk.Label(frm, text="PNG DPI").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=vars_["png_dpi"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(frm, text="PNG width (inch)").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=vars_["png_width_inch"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Checkbutton(frm, text="Transparent PNG background", variable=vars_["png_transparent"]).grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        row += 1

        ttk.Separator(frm, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 8))
        row += 1

        ttk.Label(frm, text="GIF export", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        row += 1

        ttk.Label(
            frm,
            text=(
                "Creates a temperature sweep GIF using the selected temperature range.\n"
                "Camera stays fixed; tensor shape/size changes with temperature."
            ),
            foreground="gray",
            justify="left",
            wraplength=320,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 2))
        row += 1

        ttk.Label(frm, text="Start T (K)").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=vars_["gif_temp_start"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(frm, text="End T (K)").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=vars_["gif_temp_end"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(frm, text="Step").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=vars_["gif_temp_step"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(
            frm,
            text="Leave start/end blank to use all available temperatures. Step selects every n-th temperature.",
            foreground="gray",
            justify="left",
            wraplength=320,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 2))
        row += 1

        def _apply():
            try:
                opacity = float(vars_["opacity"].get())
                tensor_scale = float(vars_["tensor_scale"].get())
                absolute_scale_factor = float(vars_["absolute_scale_factor"].get())
                ref_tensor_e32 = float(vars_["absolute_ref_tensor_e32"].get())
                target_fraction = float(vars_["absolute_target_fraction"].get())

                if not (0.0 < opacity <= 1.0):
                    raise ValueError("Opacity must be between 0 and 1.")
                if tensor_scale <= 0.0:
                    raise ValueError("Tensor scale must be positive.")
                if absolute_scale_factor <= 0.0:
                    raise ValueError("Absolute scale factor must be positive.")
                if ref_tensor_e32 <= 0.0:
                    raise ValueError("Reference tensor must be positive.")
                if target_fraction <= 0.0:
                    raise ValueError("Target size / molecule span must be positive.")

            except Exception as exc:
                messagebox.showerror("Input error", str(exc))
                return False

            for k, v in vars_.items():
                self._tensor_view_opts[k] = v.get()

            return True

        def _open_refresh():
            if not _apply():
                return
            self._plot_tensor_spheroid_pyvista(refresh_only=True)

        def _close_viewer():
            plotter = getattr(self, "_tensor_view_plotter", None)
            if plotter is None:
                return

            try:
                iren = getattr(plotter, "iren", None)
                if iren is not None:
                    try:
                        iren.terminate_app()
                    except Exception:
                        pass

                try:
                    plotter.close()
                except Exception:
                    pass

                ren_win = getattr(plotter, "ren_win", None)
                if ren_win is not None:
                    try:
                        ren_win.Finalize()
                    except Exception:
                        pass

            finally:
                self._tensor_view_plotter = None

        def _save_png():
            if not _apply():
                return
            plotter = getattr(self, "_tensor_view_plotter", None)
            if plotter is None:
                messagebox.showinfo("No viewer", "Open the tensor spheroid viewer first.")
                return

            path = filedialog.asksaveasfilename(
                title="Save tensor spheroid PNG",
                defaultextension=".png",
                filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            )
            if not path:
                return

            try:
                try:
                    _apply_camera_preset(plotter, str(self._tensor_view_opts["camera_preset"]))
                    plotter.render()
                except Exception:
                    pass

                _save_tensor_plotter_png(
                    plotter,
                    path,
                    dpi=int(self._tensor_view_opts["png_dpi"]),
                    width_inch=float(self._tensor_view_opts["png_width_inch"]),
                    transparent=bool(self._tensor_view_opts["png_transparent"]),
                )
                messagebox.showinfo("Saved", f"Saved PNG:\n{path}")
            except Exception as exc:
                messagebox.showerror("PNG export error", str(exc))

        def _save_gif():
            if not _apply():
                return
            plotter = getattr(self, "_tensor_view_plotter", None)
            if plotter is None:
                messagebox.showinfo("No viewer", "Open the tensor spheroid viewer first.")
                return

            if self._session.orca is None or not self._session.orca.get("tensors_by_temp"):
                messagebox.showinfo("No tensor data", "No temperature-dependent tensors are loaded.")
                return

            path = filedialog.asksaveasfilename(
                title="Save tensor spheroid temperature GIF",
                defaultextension=".gif",
                filetypes=[("GIF animation", "*.gif"), ("All files", "*.*")],
            )
            if not path:
                return

            atoms = list(self._session.orca["atoms"])

            if self._session.last_result is not None and self._session.last_result.get("metal_xyz") is not None:
                metal_xyz = np.asarray(self._session.last_result["metal_xyz"], dtype=float)
            else:
                metal_idx = _guess_metal_index_from_atoms(atoms)
                metal_xyz = np.asarray(atoms[metal_idx][1:], dtype=float)

            try:
                try:
                    _apply_camera_preset(plotter, str(self._tensor_view_opts["camera_preset"]))
                    try:
                        plotter.camera.SetParallelProjection(True)
                    except Exception:
                        pass
                    plotter.render()
                except Exception:
                    pass

                _save_tensor_temperature_gif(
                    plotter=plotter,
                    atoms=atoms,
                    tensors_by_temp=self._session.orca["tensors_by_temp"],
                    metal_xyz=metal_xyz,
                    opts=self._tensor_view_opts,
                    path=path,
                )
                messagebox.showinfo("Saved", f"Saved temperature GIF:\n{path}")

            except Exception as exc:
                messagebox.showerror("GIF export error", str(exc))

        btnf = ttk.Frame(frm)
        btnf.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        btnf.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(btnf, text="Apply", command=_apply).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(btnf, text="Open / Refresh Viewer", command=_open_refresh).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(btnf, text="Close Viewer", command=_close_viewer).grid(row=0, column=2, sticky="ew", padx=(4, 0))

        btnf2 = ttk.Frame(frm)
        btnf2.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        btnf2.columnconfigure((0, 1), weight=1)

        ttk.Button(btnf2, text="Save PNG", command=_save_png).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(btnf2, text="Save GIF", command=_save_gif).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        def _on_close():
            try:
                canvas.unbind_all("<MouseWheel>")
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")
            except Exception:
                pass

            try:
                win.destroy()
            finally:
                self._tensor_view_ctrl = None

        win.protocol("WM_DELETE_WINDOW", _on_close)

    def _plot_tensor_spheroid_pyvista(self, refresh_only: bool = False):
        if pv is None:
            messagebox.showerror(
                "PyVista not available",
                "PyVista could not be imported.\nInstall with:\n\npip install pyvista vtk"
            )
            return

        if self._session.orca is None:
            messagebox.showinfo("No ORCA data", "Load an ORCA output file first.")
            return

        used_temp = None
        chi_r2 = None

        if self._session.last_result is not None:
            used_temp = self._session.last_result.get("temperature", self._session.temperature)
            chi_r2 = self._session.last_result.get("chi_rank2", None)

        if chi_r2 is None:
            if self._session.chi is None:
                messagebox.showinfo("No tensor", "No converted tensor is available.")
                return
            used_temp = self._session.temperature
            chi_r2 = rank2_chi(np.asarray(self._session.chi, dtype=float))

        chi_r2 = rank2_chi(np.asarray(chi_r2, dtype=float))

        opts = self._tensor_view_opts
        atoms = list(self._session.orca["atoms"])

        global_max_abs = None
        if str(opts.get("tensor_scaling_mode", "per_frame")) == "global":
            try:
                sel_temps = _selected_temperatures_from_opts(
                    self._session.orca["tensors_by_temp"],
                    opts,
                )
                global_max_abs = _global_max_abs_eigval_from_selected_temps(
                    self._session.orca["tensors_by_temp"],
                    sel_temps,
                )
            except Exception:
                global_max_abs = None

        if self._session.last_result is not None and self._session.last_result.get("metal_xyz") is not None:
            metal_xyz = np.asarray(self._session.last_result["metal_xyz"], dtype=float)
        else:
            metal_idx = _guess_metal_index_from_atoms(atoms)
            metal_xyz = np.asarray(atoms[metal_idx][1:], dtype=float)

        plotter = self._tensor_view_plotter
        alive = self._is_tensor_plotter_alive(plotter)

        if not alive:
            plotter = pv.Plotter(window_size=opts.get("window_size", (900, 760)))
            self._tensor_view_plotter = plotter

            try:
                def _on_close(*_args):
                    self._tensor_view_plotter = None

                plotter.iren.add_observer("ExitEvent", _on_close)
            except Exception:
                pass

        try:
            _draw_tensor_spheroid_scene(
                plotter,
                atoms=atoms,
                chi_r2=chi_r2,
                used_temp=used_temp,
                opts=opts,
                metal_xyz=metal_xyz,
                global_max_abs=global_max_abs,
            )

            if alive:
                try:
                    _apply_camera_preset(plotter, str(opts.get("camera_preset", "iso")))
                except Exception:
                    pass

                try:
                    plotter.render()
                except Exception:
                    raise

                try:
                    plotter.update()
                except Exception:
                    pass
            else:
                plotter.show(
                    title="Tensor spheroid",
                    auto_close=False,
                    interactive=True,
                    interactive_update=True,
                )

        except Exception:
            old = self._tensor_view_plotter
            try:
                if old is not None:
                    try:
                        iren = getattr(old, "iren", None)
                        if iren is not None:
                            try:
                                iren.terminate_app()
                            except Exception:
                                pass

                        old.close()

                        ren_win = getattr(old, "ren_win", None)
                        if ren_win is not None:
                            try:
                                ren_win.Finalize()
                            except Exception:
                                pass
                    except Exception:
                        pass
            finally:
                self._tensor_view_plotter = None

            plotter = pv.Plotter(window_size=opts.get("window_size", (900, 760)))
            self._tensor_view_plotter = plotter

            try:
                def _on_close(*_args):
                    self._tensor_view_plotter = None

                plotter.iren.add_observer("ExitEvent", _on_close)
            except Exception:
                pass

            _draw_tensor_spheroid_scene(
                plotter,
                atoms=atoms,
                chi_r2=chi_r2,
                used_temp=used_temp,
                opts=opts,
                metal_xyz=metal_xyz,
                global_max_abs=global_max_abs,
            )

            plotter.show(
                title="Tensor spheroid",
                auto_close=False,
                interactive=True,
                interactive_update=True,
            )

    def _is_plotter_alive(self, plotter) -> bool:
        if plotter is None:
            return False
        try:
            _ = plotter.renderer.actors
            return True
        except Exception:
            return False

    def _is_tensor_plotter_alive(self, plotter) -> bool:
        if plotter is None:
            return False

        try:
            renderer = getattr(plotter, "renderer", None)
            ren_win = getattr(plotter, "ren_win", None)
            iren = getattr(plotter, "iren", None)

            if renderer is None or ren_win is None or iren is None:
                return False

            _ = renderer.actors
            _ = ren_win.GetSize()
            _ = iren.interactor

            return True
        except Exception:
            return False

    def _on_quit(self):
        try:
            close_pcs_pde_view(self._session.viewer_plotter)
        except Exception:
            pass

        try:
            plotter = getattr(self, "_tensor_view_plotter", None)
            if plotter is not None:
                try:
                    iren = getattr(plotter, "iren", None)
                    if iren is not None:
                        try:
                            iren.terminate_app()
                        except Exception:
                            pass
                    plotter.close()
                except Exception:
                    pass
                finally:
                    self._tensor_view_plotter = None
        except Exception:
            pass

        self.destroy()

    def _build_atom_choice_labels(self) -> list[str]:
        if self._session.orca is None or not self._session.orca.get("atoms"):
            return []

        labels = []
        for i, atom in enumerate(self._session.orca["atoms"], start=1):
            el, x, y, z = atom
            labels.append(f"{i}: {el}  ({x:.3f}, {y:.3f}, {z:.3f})")
        return labels

    def _get_selected_z_axis(self, mode: str) -> np.ndarray:
        if mode == "cartesian_z":
            return np.array([0.0, 0.0, 1.0], dtype=float)

        if mode == "tensor_principal_z":
            if self._session.last_result is None:
                raise ValueError("No computed result is available.")

            chi_r2 = self._session.last_result.get("chi_rank2")
            if chi_r2 is None:
                raise ValueError("chi_rank2 is not available in the last result.")

            chi_r2 = np.asarray(chi_r2, dtype=float)
            evals, evecs = np.linalg.eigh(chi_r2)
            idx = int(np.argmax(np.abs(evals)))
            vec = np.asarray(evecs[:, idx], dtype=float)
            n = np.linalg.norm(vec)
            if n < 1e-12:
                raise ValueError("Failed to determine tensor principal axis.")
            return vec / n

        raise ValueError(f"Unknown z-axis mode: {mode}")

    def _open_oblique_slice_plot(
        self,
        *,
        z_axis_mode: str,
        atom_index_1based: int,
        plane_tol_atoms: float,
        levels: int,
        custom_levels_text: str,
        show_atom_labels: bool,
        save_path: str | None = None,
    ):
        if self._session.last_result is None:
            messagebox.showwarning("No result", "Run a computation first.")
            return

        if self._session.orca is None or not self._session.orca.get("atoms"):
            messagebox.showwarning("No structure", "No atomic structure is loaded.")
            return

        atoms = self._session.orca["atoms"]
        n_atoms = len(atoms)

        if not (1 <= atom_index_1based <= n_atoms):
            messagebox.showerror("Invalid atom", f"Atom index must be between 1 and {n_atoms}.")
            return

        result = self._session.last_result
        metal_xyz = np.asarray(result["metal_xyz"], dtype=float)

        try:
            z_axis = self._get_selected_z_axis(z_axis_mode)
        except Exception as exc:
            messagebox.showerror("Z-axis error", str(exc))
            return

        atom_idx0 = atom_index_1based - 1
        target_xyz = np.asarray(atoms[atom_idx0][1:], dtype=float)
        user_vector = target_xyz - metal_xyz

        if np.linalg.norm(user_vector) < 1e-12:
            messagebox.showerror("Vector error", "Chosen atom is at the metal position.")
            return

        axis_label = (
            "Cartesian z-axis" if z_axis_mode == "cartesian_z"
            else "Tensor principal z-axis"
        )

        try:
            show_oblique_pcs_slice_plot(
                atoms=atoms,
                pcs_field=result["pcs_field"],
                ext=result["ext"],
                metal_xyz=metal_xyz,
                z_axis=z_axis,
                user_vector=user_vector,
                plane_tol_atoms=float(plane_tol_atoms),
                levels=int(levels),
                contour_levels_text=custom_levels_text,
                cmap="custom_blue_red",
                show_atom_labels=bool(show_atom_labels),
                title=f"PCS slice | metal + {axis_label} + atom {atom_index_1based}",
                save_path=save_path,
                dpi=600,
                transparent=False,
            )
        except Exception as exc:
            messagebox.showerror("Slice plot error", str(exc))

    def _open_oblique_slice_dialog(self):
        if self._session.last_result is None:
            messagebox.showwarning("No result", "Run a computation first.")
            return

        if self._session.orca is None or not self._session.orca.get("atoms"):
            messagebox.showwarning("No structure", "Load ORCA data first.")
            return

        win = tk.Toplevel(self)
        win.title("Open Oblique PCS Slice")
        win.geometry("480x320")
        win.resizable(True, True)
        win.transient(self)

        outer = ttk.Frame(win, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)

        atom_choices = self._build_atom_choice_labels()

        z_axis_var = tk.StringVar(value="cartesian_z")
        atom_choice_var = tk.StringVar(value=atom_choices[0] if atom_choices else "")
        plane_tol_var = tk.StringVar(value="0.8")
        levels_var = tk.StringVar(value="31")
        custom_levels_var = tk.StringVar(value="")
        show_labels_var = tk.BooleanVar(value=False)

        row = 0

        ttk.Label(outer, text="Z-axis definition").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            outer,
            textvariable=z_axis_var,
            values=["cartesian_z", "tensor_principal_z"],
            state="readonly",
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(outer, text="Direction atom").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            outer,
            textvariable=atom_choice_var,
            values=atom_choices,
            state="readonly",
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        ttk.Label(outer, text="Atom plane tolerance (Å)").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(outer, textvariable=plane_tol_var).grid(
            row=row, column=1, sticky="ew", pady=4
        )
        row += 1

        ttk.Label(outer, text="Contour levels").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(outer, textvariable=levels_var).grid(
            row=row, column=1, sticky="ew", pady=4
        )
        row += 1

        ttk.Label(outer, text="Custom contour levels (ppm)").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(outer, textvariable=custom_levels_var).grid(
            row=row, column=1, sticky="ew", pady=4
        )
        row += 1

        ttk.Checkbutton(
            outer,
            text="Show atom labels",
            variable=show_labels_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 8))
        row += 1

        info = (
            "Plane definition:\n"
            "metal centre + selected z-axis + vector from metal to chosen atom\n\n"
            "Custom contour levels example: 1,2,5,10\n"
            "(leave blank for automatic symmetric levels)"
        )
        ttk.Label(
            outer,
            text=info,
            foreground="gray",
            justify="left",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 10))
        row += 1

        btns = ttk.Frame(outer)
        btns.grid(row=row, column=0, columnspan=2, sticky="ew")
        btns.columnconfigure((0, 1, 2), weight=1)

        def _parse_inputs():
            choice = atom_choice_var.get().strip()
            if not choice:
                messagebox.showerror("Missing atom", "Choose a direction atom.")
                return None

            try:
                atom_index = int(choice.split(":")[0].strip())
            except Exception:
                messagebox.showerror("Atom parse error", f"Could not parse atom index from:\n{choice}")
                return None

            try:
                plane_tol = float(plane_tol_var.get())
                levels = int(levels_var.get())
            except Exception:
                messagebox.showerror(
                    "Input error",
                    "Plane tolerance and contour levels must be numeric."
                )
                return None

            return {
                "z_axis_mode": z_axis_var.get(),
                "atom_index_1based": atom_index,
                "plane_tol_atoms": plane_tol,
                "levels": levels,
                "custom_levels_text": custom_levels_var.get().strip(),
                "show_atom_labels": bool(show_labels_var.get()),
            }

        def _run():
            parsed = _parse_inputs()
            if parsed is None:
                return
            self._open_oblique_slice_plot(
                z_axis_mode=parsed["z_axis_mode"],
                atom_index_1based=parsed["atom_index_1based"],
                plane_tol_atoms=parsed["plane_tol_atoms"],
                levels=parsed["levels"],
                custom_levels_text=parsed["custom_levels_text"],
                show_atom_labels=parsed["show_atom_labels"],
                save_path=None,
            )

        def _save():
            parsed = _parse_inputs()
            if parsed is None:
                return

            path = filedialog.asksaveasfilename(
                title="Save oblique PCS slice plot",
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

            self._open_oblique_slice_plot(
                z_axis_mode=parsed["z_axis_mode"],
                atom_index_1based=parsed["atom_index_1based"],
                plane_tol_atoms=parsed["plane_tol_atoms"],
                levels=parsed["levels"],
                custom_levels_text=parsed["custom_levels_text"],
                show_atom_labels=parsed["show_atom_labels"],
                save_path=path,
            )

        ttk.Button(btns, text="Open plot", command=_run).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(btns, text="Save plot...", command=_save).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(btns, text="Close", command=win.destroy).grid(
            row=0, column=2, sticky="ew", padx=(4, 0)
        )


def main():
    app = AppWindow()
    app.mainloop()


if __name__ == "__main__":
    main()