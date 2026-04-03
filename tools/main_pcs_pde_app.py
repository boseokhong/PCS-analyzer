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

from logic.xyz_loader import load_orca_data, pick_orca_tensor_at_temperature
from tools.logic_spindens_loader import load_spindens_3d
from tools.ui_pcs_pde_control import ControlPanel, StatusBar
from tools.ui_pcs_pde_viewer import show_pcs_pde_view, show_oblique_pcs_slice_plot
from tools.logic_pcs_pde import rank2_chi, PYFFTW_AVAILABLE, PYFFTW_THREADS

AVOGADRO = 6.02214129e23


def convert_orca_chi_to_angstrom3(chi_raw: np.ndarray, temperature: float) -> np.ndarray:
    """
    Convert ORCA susceptibility tensor from cm³·K/mol to Å³/molecule.

        chi = 4*pi * 1e24 * chi_raw / (N_A * T)
    """
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

        ttk.Label(self, text="Session Info", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", padx=10, pady=(10, 4)
        )

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


class AppSession:
    orca_path: Optional[str] = None
    dens_path: Optional[str] = None
    orca: Optional[dict] = None
    dens: Optional[dict] = None
    temperature: Optional[float] = None
    chi_raw: Optional[np.ndarray] = None
    chi: Optional[np.ndarray] = None
    last_result: Optional[dict] = None


class AppWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("PCS-PDE Viewer")
        self.geometry("1000x720")
        self.minsize(700, 500)

        self._session = AppSession()
        self._compute_thread: Optional[threading.Thread] = None

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
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        run_menu = tk.Menu(menubar, tearoff=False)
        run_menu.add_command(label="Run computation (▶)", command=self._on_run)
        run_menu.add_separator()
        run_menu.add_command(label="Open Oblique PCS Slice…", command=self._open_oblique_slice_dialog)
        menubar.add_cascade(label="Run", menu=run_menu)

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
            on_export_callback=self._export_npy,
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

        temp_val = params.get("temperature")
        try:
            t, chi_raw = pick_orca_tensor_at_temperature(
                self._session.orca["tensors_by_temp"],
                temperature=temp_val,
            )
            chi_converted = convert_orca_chi_to_angstrom3(chi_raw, t)
        except Exception as exc:
            messagebox.showerror("Tensor error", str(exc))
            return

        self._session.temperature = t
        self._session.chi_raw = np.asarray(chi_raw, dtype=float)
        self._session.chi = np.asarray(chi_converted, dtype=float)

        order = params.get("density_order", "C")
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
                result = show_pcs_pde_view(
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
                self.after(0, lambda: self._on_compute_done(success=True))
            except Exception as exc:
                print(traceback.format_exc())
                self.after(0, lambda: self._on_compute_done(success=False, error=str(exc)))

        self._compute_thread = threading.Thread(target=_worker, daemon=True)
        self._compute_thread.start()

    def _on_compute_done(self, success: bool, error: str = ""):
        self._update_info()
        if success:
            pcs = self._session.last_result["pcs_field"]
            self._status.stop_busy(
                f"Done. PCS ∈ [{np.nanmin(pcs):.3f}, {np.nanmax(pcs):.3f}] ppm"
            )
        else:
            self._status.stop_busy("Computation failed.")
            messagebox.showerror("Computation error", f"PCS computation failed:\n{error}")

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


def main():
    app = AppWindow()
    app.mainloop()


if __name__ == "__main__":
    main()