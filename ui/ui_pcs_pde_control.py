# ui/ui_pcs_pde_control.py
"""
Control panel GUI for the PCS-PDE viewer (FFT).
"""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog
from typing import Callable, Optional

from ui.style import get_app_fonts


STYLE_OPTIONS = ("surface", "mesh", "both")
CAMERA_PRESETS = ("iso", "xy", "xz", "yz")


def _labeled_entry(parent, row, label, variable, tooltip="", width=10):
    lbl = ttk.Label(parent, text=label)
    lbl.grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
    entry = ttk.Entry(parent, textvariable=variable, width=width)
    entry.grid(row=row, column=1, sticky="ew", pady=2)
    if tooltip:
        _ToolTip(entry, tooltip)
        _ToolTip(lbl, tooltip)
    return entry


def _labeled_check(parent, row, label, variable, tooltip=""):
    cb = ttk.Checkbutton(parent, text=label, variable=variable)
    cb.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
    if tooltip:
        _ToolTip(cb, tooltip)
    return cb


def _section_header(parent, row, text):
    fonts = get_app_fonts(parent)
    sep = ttk.Separator(parent, orient="horizontal")
    sep.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 2))
    lbl = ttk.Label(parent, text=text, font=fonts.get("section", ("TkDefaultFont", 9, "bold")))
    lbl.grid(row=row + 1, column=0, columnspan=2, sticky="w", pady=(0, 4))
    return lbl


def _choose_color(var: tk.StringVar):
    initial = str(var.get()).strip() or "#ffffff"
    result = colorchooser.askcolor(color=initial, title="Choose color")
    if result and result[1]:
        var.set(result[1])


def _color_swatch_button(parent, variable: tk.StringVar):
    btn = tk.Button(
        parent,
        width=3,
        relief="flat",
        bd=1,
        cursor="hand2",
        command=lambda: _choose_color(variable),
    )

    def _refresh(*_args):
        val = str(variable.get()).strip() or "#ffffff"
        try:
            btn.configure(bg=val, activebackground=val)
        except Exception:
            btn.configure(bg="#ffffff", activebackground="#ffffff")

    variable.trace_add("write", _refresh)
    _refresh()
    return btn


def _labeled_color(parent, row, label, variable, tooltip=""):
    lbl = ttk.Label(parent, text=label)
    lbl.grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)

    wrap = ttk.Frame(parent)
    wrap.grid(row=row, column=1, sticky="ew", pady=2)
    wrap.columnconfigure(0, weight=1)

    ent = ttk.Entry(wrap, textvariable=variable, width=12)
    ent.grid(row=0, column=0, sticky="ew")

    btn = _color_swatch_button(wrap, variable)
    btn.grid(row=0, column=1, padx=(6, 0))

    if tooltip:
        _ToolTip(lbl, tooltip)
        _ToolTip(ent, tooltip)
        _ToolTip(btn, tooltip)

    return ent, btn


class _ToolTip:
    def __init__(self, widget, text: str):
        self._widget = widget
        self._text = text
        self._tip: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            self._tip,
            text=self._text,
            background="#fffde7",
            relief="solid",
            borderwidth=1,
            font=get_app_fonts(self._widget).get("ui_small", ("TkDefaultFont", 8)),
            wraplength=280,
            justify="left",
            padx=4,
            pady=3,
        )
        lbl.pack()

    def _hide(self, _event=None):
        if self._tip:
            self._tip.destroy()
            self._tip = None


class StatusBar(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._msg_var = tk.StringVar(value="Ready.")
        self._label = ttk.Label(self, textvariable=self._msg_var, anchor="w")
        self._label.pack(side="left", fill="x", expand=True, padx=6)
        self._progress = ttk.Progressbar(self, mode="indeterminate", length=120)
        self._progress.pack(side="right", padx=6, pady=2)

    def set(self, text: str):
        self._msg_var.set(text)
        self._label.update_idletasks()

    def start_busy(self, text: str = "Computing…"):
        self._msg_var.set(text)
        self._progress.start(12)

    def stop_busy(self, text: str = "Done."):
        self._progress.stop()
        self._msg_var.set(text)


class ControlPanel(ttk.Frame):
    DEFAULT_LEVEL_STYLES = [
        {"ppm": 0.5, "pos_color": "#ff0000", "neg_color": "#0000ff", "style": "mesh",    "opacity": 0.05},
        {"ppm": 2.0, "pos_color": "#ff0000", "neg_color": "#0000ff", "style": "surface", "opacity": 0.18},
        {"ppm": 5.0, "pos_color": "#ff0000", "neg_color": "#0000ff", "style": "surface", "opacity": 0.30},
    ]

    DEFAULTS: dict = {
        "temperature": "",
        "fft_pad_factor": 2,
        "normalize_density": True,
        "normalization_target": 1.0,
        "density_isovalue": 0.005,
        "show_atoms": False,
        "atom_elements": [],
        "show_bonds": True,
        "bond_color": "#555A60",
        "bond_tolerance": 0.05,
        "show_density": False,
        "show_pcs": True,
        "show_labels": False,
        "show_grid": False,
        "show_outline": False,
        "background_color": "white",
        "camera_preset": "iso",
        "camera_projection": "perspective",
        "density_color": "#27af91",
        "density_style": "both",
        "density_opacity": 0.15,
        "ambient_light": 0.50,
        "smooth_pcs_display": False,
        "smooth_pcs_sigma": 1.0,
        "png_dpi": 150,
        "png_width_inch": 6.0,
        "png_transparent": False,
        "export_view": "preset",
        "level_styles": DEFAULT_LEVEL_STYLES,
    }

    def __init__(
        self,
        parent,
        on_run_callback: Callable[[dict], None],
        on_refresh_view_callback: Optional[Callable[[dict], None]] = None,
        on_export_png_callback: Optional[Callable[[dict], None]] = None,
        on_oblique_slice_callback: Optional[Callable[[], None]] = None,
        on_compare_plot_callback: Optional[Callable[[], None]] = None,
        on_residual_plot_callback: Optional[Callable[[], None]] = None,
        on_tensor_spheroid_callback: Optional[Callable[[], None]] = None,
        on_export_numpy_callback: Optional[Callable[[], None]] = None,
        on_export_atom_csv_callback: Optional[Callable[[], None]] = None,
        on_apply_camera_callback: Optional[Callable[[dict], None]] = None,
        on_save_camera_callback: Optional[Callable[[], None]] = None,
        on_restore_camera_callback: Optional[Callable[[], None]] = None,
        temperatures: Optional[list[float]] = None,
        initial_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        self._on_run = on_run_callback
        self._on_refresh_view = on_refresh_view_callback
        self._on_export_png = on_export_png_callback
        self._on_oblique_slice = on_oblique_slice_callback
        self._on_compare_plot = on_compare_plot_callback
        self._on_residual_plot = on_residual_plot_callback
        self._on_tensor_spheroid = on_tensor_spheroid_callback
        self._on_export_numpy = on_export_numpy_callback
        self._on_export_atom_csv = on_export_atom_csv_callback
        self._on_apply_camera = on_apply_camera_callback
        self._on_save_camera = on_save_camera_callback
        self._on_restore_camera = on_restore_camera_callback
        self._temperatures = temperatures or []
        self._elements: list[str] = []
        self._atom_element_vars: dict[str, tk.BooleanVar] = {}

        initial = dict(initial_params or {})
        if "density_isovalue" not in initial and "density_iso" in initial:
            initial["density_isovalue"] = initial["density_iso"]
        initial.pop("density_iso", None)
        initial.pop("auto_scale_pcs_levels", None)
        params = {**self.DEFAULTS, **initial}
        if "level_styles" not in params or not params["level_styles"]:
            params["level_styles"] = list(self.DEFAULT_LEVEL_STYLES)

        self._vars: dict[str, tk.Variable] = {}
        self._level_rows: list[dict] = []
        self._build_ui(params)

    def _refresh_temperature_summary(self):
        temps = sorted(float(t) for t in (self._temperatures or []))
        if not temps:
            self._temp_summary_var.set("No temperature list loaded.")
            return

        if len(temps) <= 8:
            values_txt = ", ".join(f"{t:g}" for t in temps)
        else:
            values_txt = (
                f"{', '.join(f'{t:g}' for t in temps[:4])}, …, "
                f"{', '.join(f'{t:g}' for t in temps[-3:])}"
            )

        self._temp_summary_var.set(
            f"Available: {len(temps)} temperature(s)\n"
            f"Range: {temps[0]:g}–{temps[-1]:g} K\n"
            f"Values: {values_txt}"
        )

    def _add_level_row(
        self,
        parent,
        *,
        ppm: float = 2.0,
        pos_color: str = "#ff0000",
        neg_color: str = "#0000ff",
        style_val: str = "both",
        opacity_val: float = 0.05,
    ):
        rowf = ttk.Frame(parent)
        rowf.pack(fill="x", pady=1)

        v_ppm = tk.StringVar(value=str(ppm))
        v_pos = tk.StringVar(value=pos_color)
        v_neg = tk.StringVar(value=neg_color)
        v_style = tk.StringVar(value=style_val)
        v_opacity = tk.StringVar(value=str(opacity_val))

        ttk.Entry(rowf, textvariable=v_ppm, width=8).grid(row=0, column=0, padx=2)
        _color_swatch_button(rowf, v_pos).grid(row=0, column=1, padx=2)
        _color_swatch_button(rowf, v_neg).grid(row=0, column=2, padx=2)
        ttk.Combobox(
            rowf,
            textvariable=v_style,
            values=list(STYLE_OPTIONS),
            state="readonly",
            width=9,
        ).grid(row=0, column=3, padx=2)
        ttk.Entry(rowf, textvariable=v_opacity, width=8).grid(row=0, column=4, padx=2)

        def _remove():
            try:
                self._level_rows.remove(level_info)
            except ValueError:
                pass
            rowf.destroy()

        ttk.Button(rowf, text="✕", width=2, command=_remove).grid(row=0, column=5, padx=2)

        level_info = {
            "frame": rowf,
            "ppm": v_ppm,
            "pos_color": v_pos,
            "neg_color": v_neg,
            "style": v_style,
            "opacity": v_opacity,
        }
        self._level_rows.append(level_info)

    def _rebuild_level_rows(self, styles: list[dict]):
        for row in list(self._level_rows):
            try:
                row["frame"].destroy()
            except Exception:
                pass
        self._level_rows.clear()

        for ls in styles:
            self._add_level_row(
                self._level_rows_frame,
                ppm=ls.get("ppm", 2.0),
                pos_color=ls.get("pos_color", "#ff0000"),
                neg_color=ls.get("neg_color", "#0000ff"),
                style_val=ls.get("style", "surface"),
                opacity_val=ls.get("opacity", 0.30),
            )

    def _build_ui_legacy(self, params: dict):
        self.columnconfigure(0, weight=1)

        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas, padding=(12, 8, 12, 8))
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(inner_id, width=e.width))

        def _on_mousewheel(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)

        inner.columnconfigure(1, weight=1)
        r = 0

        _section_header(inner, r, "Data")
        r += 2

        self._vars["temperature"] = tk.StringVar(value=str(params["temperature"]))
        lbl = ttk.Label(inner, text="Temperature (K)")
        lbl.grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
        _ToolTip(lbl, "Target temperature. Leave blank to use the first available tensor.")

        self._temp_summary_var = tk.StringVar(value="No temperature list loaded.")
        self._temp_combo = ttk.Combobox(
            inner,
            textvariable=self._vars["temperature"],
            values=[""] + [f"{t:g}" for t in sorted(self._temperatures)],
            width=10,
            state="readonly" if self._temperatures else "normal",
        )
        self._temp_combo.grid(row=r, column=1, sticky="ew", pady=2)
        r += 1

        self._temp_summary_label = ttk.Label(
            inner,
            textvariable=self._temp_summary_var,
            foreground="gray",
            font=get_app_fonts(self).get("ui_small", ("TkDefaultFont", 8)),
            wraplength=220,
            justify="left",
        )
        self._temp_summary_label.grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 4))
        r += 1
        self._refresh_temperature_summary()

        _section_header(inner, r, "FFT Settings")
        r += 2

        self._vars["fft_pad_factor"] = tk.IntVar(value=int(params["fft_pad_factor"]))
        _labeled_entry(inner, r, "Padding factor", self._vars["fft_pad_factor"])
        r += 1

        self._vars["normalize_density"] = tk.BooleanVar(value=bool(params["normalize_density"]))
        _labeled_check(inner, r, "Normalize density integral", self._vars["normalize_density"])
        r += 1

        self._vars["normalization_target"] = tk.DoubleVar(value=float(params["normalization_target"]))
        _labeled_entry(inner, r, "Normalization target", self._vars["normalization_target"])
        r += 1

        _section_header(inner, r, "Density Surface")
        r += 2

        self._vars["density_isovalue"] = tk.DoubleVar(value=float(params["density_isovalue"]))
        _labeled_entry(inner, r, "Density isovalue (a.u.)", self._vars["density_isovalue"])
        r += 1

        self._vars["density_style"] = tk.StringVar(value=str(params["density_style"]))
        ttk.Label(inner, text="Density style").grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Combobox(
            inner, textvariable=self._vars["density_style"],
            values=["surface", "mesh", "both"], state="readonly", width=10,
        ).grid(row=r, column=1, sticky="ew", pady=2)
        r += 1

        self._vars["density_color"] = tk.StringVar(value=str(params["density_color"]))
        _labeled_color(inner, r, "Density color", self._vars["density_color"])
        r += 1

        self._vars["density_opacity"] = tk.DoubleVar(value=float(params["density_opacity"]))
        _labeled_entry(inner, r, "Density opacity", self._vars["density_opacity"])
        r += 1

        self._vars["show_density"] = tk.BooleanVar(value=bool(params["show_density"]))
        _labeled_check(inner, r, "Show spin density surface", self._vars["show_density"])
        r += 1

        _section_header(inner, r, "PCS Isosurface Levels")
        r += 2

        hdr = ttk.Frame(inner)
        hdr.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 2))
        for col, txt in enumerate(("ppm", "Pos", "Neg", "Style", "Opacity", "")):
            ttk.Label(hdr, text=txt, font=get_app_fonts(self).get("ui_small_bold", ("TkDefaultFont", 8, "bold"))).grid(
                row=0, column=col, padx=2, sticky="w"
            )
        r += 1

        self._level_rows_frame = ttk.Frame(inner)
        self._level_rows_frame.grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1

        self._rebuild_level_rows(params.get("level_styles", self.DEFAULT_LEVEL_STYLES))

        ttk.Button(
            inner,
            text="+ Add level",
            command=lambda: self._add_level_row(self._level_rows_frame),
        ).grid(row=r, column=0, columnspan=2, sticky="w", pady=(2, 0))
        r += 1

        self._vars["show_pcs"] = tk.BooleanVar(value=bool(params["show_pcs"]))
        _labeled_check(inner, r, "Show PCS isosurfaces", self._vars["show_pcs"])
        r += 1

        _section_header(inner, r, "Display")
        r += 2

        self._vars["show_atoms"] = tk.BooleanVar(value=bool(params["show_atoms"]))
        self._vars["show_bonds"] = tk.BooleanVar(value=bool(params["show_bonds"]))
        self._vars["show_labels"] = tk.BooleanVar(value=bool(params["show_labels"]))
        self._vars["show_grid"] = tk.BooleanVar(value=bool(params["show_grid"]))
        self._vars["show_outline"] = tk.BooleanVar(value=bool(params["show_outline"]))

        disp_frame = ttk.Frame(inner)
        disp_frame.grid(row=r, column=0, columnspan=2, sticky="ew", pady=2)

        disp_frame.columnconfigure(0, weight=1)
        disp_frame.columnconfigure(1, weight=1)
        disp_frame.columnconfigure(2, weight=1)

        ttk.Checkbutton(
            disp_frame,
            text="Show atoms",
            variable=self._vars["show_atoms"],
        ).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=2)

        ttk.Checkbutton(
            disp_frame,
            text="Show bonds",
            variable=self._vars["show_bonds"],
        ).grid(row=0, column=1, sticky="w", padx=(0, 8), pady=2)

        ttk.Checkbutton(
            disp_frame,
            text="Show labels",
            variable=self._vars["show_labels"],
        ).grid(row=0, column=2, sticky="w", pady=2)

        ttk.Checkbutton(
            disp_frame,
            text="Show grid",
            variable=self._vars["show_grid"],
        ).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=2)

        ttk.Checkbutton(
            disp_frame,
            text="Show outline",
            variable=self._vars["show_outline"],
        ).grid(row=1, column=1, sticky="w", padx=(0, 8), pady=2)

        r += 1

        _section_header(inner, r, "Appearance")
        r += 2

        self._vars["background_color"] = tk.StringVar(value=str(params["background_color"]))
        ttk.Label(inner, text="Background").grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Combobox(
            inner,
            textvariable=self._vars["background_color"],
            values=["white", "black", "gray", "lightgray", "darkgray", "#1a1a2e", "#0d1117"],
            width=10,
        ).grid(row=r, column=1, sticky="ew", pady=2)
        r += 1

        self._vars["ambient_light"] = tk.DoubleVar(value=float(params["ambient_light"]))
        _labeled_entry(inner, r, "Ambient light", self._vars["ambient_light"])
        r += 1

        self._vars["smooth_pcs_display"] = tk.BooleanVar(value=bool(params["smooth_pcs_display"]))
        _labeled_check(inner, r, "Smooth PCS display (Gaussian)", self._vars["smooth_pcs_display"])
        r += 1

        self._vars["smooth_pcs_sigma"] = tk.DoubleVar(value=float(params["smooth_pcs_sigma"]))
        _labeled_entry(inner, r, "Smooth sigma (voxels)", self._vars["smooth_pcs_sigma"])
        r += 1

        _section_header(inner, r, "Camera")
        r += 2

        self._vars["camera_preset"] = tk.StringVar(value=str(params["camera_preset"]))
        ttk.Label(inner, text="Camera preset").grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Combobox(
            inner,
            textvariable=self._vars["camera_preset"],
            values=list(CAMERA_PRESETS),
            state="readonly",
            width=10,
        ).grid(row=r, column=1, sticky="ew", pady=2)
        r += 1

        _section_header(inner, r, "Export PNG")
        r += 2

        self._vars["png_dpi"] = tk.IntVar(value=int(params["png_dpi"]))
        _labeled_entry(inner, r, "DPI", self._vars["png_dpi"])
        r += 1

        self._vars["png_width_inch"] = tk.DoubleVar(value=float(params["png_width_inch"]))
        _labeled_entry(inner, r, "Width (inch)", self._vars["png_width_inch"])
        r += 1

        self._vars["png_transparent"] = tk.BooleanVar(value=bool(params["png_transparent"]))
        _labeled_check(inner, r, "Transparent background", self._vars["png_transparent"])
        r += 1

        sep2 = ttk.Separator(inner, orient="horizontal")
        sep2.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(14, 8))
        r += 1

        btn_frame = ttk.Frame(inner)
        btn_frame.grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        ttk.Button(btn_frame, text="▶ Run computation", style="Accent.TButton", command=self._run).grid(
            row=0, column=0, padx=(0, 4), sticky="ew", ipady=4
        )
        ttk.Button(btn_frame, text="Open / Refresh Viewer", command=self._refresh_view).grid(
            row=0, column=1, padx=(4, 0), sticky="ew", ipady=4
        )

        btn_frame2 = ttk.Frame(inner)
        btn_frame2.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        r += 1
        btn_frame2.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(btn_frame2, text="Export PNG…", command=self._export_png).grid(
            row=0, column=0, padx=(0, 4), sticky="ew"
        )
        ttk.Button(btn_frame2, text="Save preset…", command=self._save_preset).grid(
            row=0, column=1, padx=4, sticky="ew"
        )
        ttk.Button(btn_frame2, text="Load preset…", command=self._load_preset).grid(
            row=0, column=2, padx=(4, 0), sticky="ew"
        )

        ttk.Button(inner, text="Reset to defaults", command=self._reset).grid(
            row=r, column=0, columnspan=2, pady=(6, 0), sticky="ew"
        )
        r += 1

        ttk.Label(
            inner,
            text=(
                "Use Run computation to recalculate the PDE field.\n"
                "Use Open / Refresh Viewer to redraw only the scene with the current display settings."
            ),
            foreground="gray",
            font=get_app_fonts(self).get("ui_small", ("TkDefaultFont", 8)),
            wraplength=240,
            justify="left",
        ).grid(row=r, column=0, columnspan=2, pady=(10, 4), sticky="w")

    def _build_ui(self, params: dict):
        """Build the compact tabbed control panel.

        The application menu remains available; buttons in the Analysis tab
        are additional entry points wired to the same AppWindow callbacks.
        """
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")
        self._notebook = notebook

        def _scroll_tab(title: str):
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=title)
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)

            canvas = tk.Canvas(tab, borderwidth=0, highlightthickness=0)
            sb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=sb.set)
            canvas.grid(row=0, column=0, sticky="nsew")
            sb.grid(row=0, column=1, sticky="ns")

            inner = ttk.Frame(canvas, padding=(12, 8, 12, 10))
            win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
            inner.columnconfigure(1, weight=1)
            inner.bind("<Configure>", lambda _e, c=canvas: c.configure(scrollregion=c.bbox("all")))
            canvas.bind("<Configure>", lambda e, c=canvas, w=win_id: c.itemconfigure(w, width=e.width))

            def _wheel(event, c=canvas):
                delta = -1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else 1
                c.yview_scroll(delta, "units")

            canvas.bind("<MouseWheel>", _wheel)
            canvas.bind("<Button-4>", _wheel)
            canvas.bind("<Button-5>", _wheel)
            inner.bind("<MouseWheel>", _wheel)
            return inner

        calc = _scroll_tab("Calculation")
        surf = _scroll_tab("Surfaces")
        view = _scroll_tab("View & Export")
        analysis = _scroll_tab("Analysis")

        # Calculation -----------------------------------------------------
        r = 0
        _section_header(calc, r, "Data")
        r += 2
        self._vars["temperature"] = tk.StringVar(value=str(params["temperature"]))
        ttk.Label(calc, text="Temperature (K)").grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
        self._temp_summary_var = tk.StringVar(value="No temperature list loaded.")
        self._temp_combo = ttk.Combobox(
            calc,
            textvariable=self._vars["temperature"],
            values=[""] + [f"{t:g}" for t in sorted(self._temperatures)],
            width=10,
            state="readonly" if self._temperatures else "normal",
        )
        self._temp_combo.grid(row=r, column=1, sticky="ew", pady=2)
        r += 1
        self._temp_summary_label = ttk.Label(
            calc, textvariable=self._temp_summary_var, foreground="gray",
            font=get_app_fonts(self).get("ui_small", ("TkDefaultFont", 8)),
            wraplength=260, justify="left",
        )
        self._temp_summary_label.grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._refresh_temperature_summary()
        r += 1

        _section_header(calc, r, "FFT / PDE Settings")
        r += 2
        self._vars["fft_pad_factor"] = tk.IntVar(value=int(params["fft_pad_factor"]))
        _labeled_entry(calc, r, "Padding factor", self._vars["fft_pad_factor"])
        r += 1
        self._vars["normalize_density"] = tk.BooleanVar(value=bool(params["normalize_density"]))
        _labeled_check(calc, r, "Normalize density integral", self._vars["normalize_density"])
        r += 1
        self._vars["normalization_target"] = tk.DoubleVar(value=float(params["normalization_target"]))
        _labeled_entry(calc, r, "Normalization target", self._vars["normalization_target"])
        r += 1
        ttk.Button(calc, text="▶ Run computation", style="Accent.TButton", command=self._run).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(14, 4), ipady=4
        )

        # Surfaces --------------------------------------------------------
        r = 0
        _section_header(surf, r, "Density Surface")
        r += 2
        self._vars["density_isovalue"] = tk.DoubleVar(value=float(params["density_isovalue"]))
        _labeled_entry(surf, r, "Density isovalue (a.u.)", self._vars["density_isovalue"])
        r += 1
        self._vars["density_style"] = tk.StringVar(value=str(params["density_style"]))
        ttk.Label(surf, text="Density style").grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Combobox(surf, textvariable=self._vars["density_style"], values=list(STYLE_OPTIONS),
                     state="readonly", width=10).grid(row=r, column=1, sticky="ew", pady=2)
        r += 1
        self._vars["density_color"] = tk.StringVar(value=str(params["density_color"]))
        _labeled_color(surf, r, "Density color", self._vars["density_color"])
        r += 1
        self._vars["density_opacity"] = tk.DoubleVar(value=float(params["density_opacity"]))
        _labeled_entry(surf, r, "Density opacity", self._vars["density_opacity"])
        r += 1
        self._vars["show_density"] = tk.BooleanVar(value=bool(params["show_density"]))
        _labeled_check(surf, r, "Show spin density surface", self._vars["show_density"])
        r += 1

        _section_header(surf, r, "PCS Isosurface Levels")
        r += 2
        hdr = ttk.Frame(surf)
        hdr.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 2))
        for col, txt in enumerate(("ppm", "Pos", "Neg", "Style", "Opacity", "")):
            ttk.Label(hdr, text=txt, font=get_app_fonts(self).get(
                "ui_small_bold", ("TkDefaultFont", 8, "bold"))).grid(row=0, column=col, padx=2, sticky="w")
        r += 1
        self._level_rows_frame = ttk.Frame(surf)
        self._level_rows_frame.grid(row=r, column=0, columnspan=2, sticky="ew")
        self._rebuild_level_rows(params.get("level_styles", self.DEFAULT_LEVEL_STYLES))
        r += 1
        ttk.Button(surf, text="+ Add level", command=lambda: self._add_level_row(self._level_rows_frame)).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(3, 0))
        r += 1
        self._vars["show_pcs"] = tk.BooleanVar(value=bool(params["show_pcs"]))
        _labeled_check(surf, r, "Show PCS isosurfaces", self._vars["show_pcs"])
        r += 1
        self._vars["smooth_pcs_display"] = tk.BooleanVar(value=bool(params["smooth_pcs_display"]))
        _labeled_check(surf, r, "Smooth PCS display (Gaussian)", self._vars["smooth_pcs_display"])
        r += 1
        self._vars["smooth_pcs_sigma"] = tk.DoubleVar(value=float(params["smooth_pcs_sigma"]))
        _labeled_entry(surf, r, "Smooth sigma (voxels)", self._vars["smooth_pcs_sigma"])
        r += 1
        ttk.Button(surf, text="Open / Refresh Viewer", command=self._refresh_view).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(14, 4), ipady=3)

        # View & Export ---------------------------------------------------
        r = 0
        _section_header(view, r, "Display")
        r += 2
        self._vars["show_atoms"] = tk.BooleanVar(value=bool(params["show_atoms"]))
        _labeled_check(view, r, "Show atoms", self._vars["show_atoms"])
        r += 1

        self._atom_details_open = tk.BooleanVar(value=False)
        self._atom_details_button = ttk.Button(
            view, text="▸ Atom elements…", command=self._toggle_atom_details,
        )
        self._atom_details_button.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 3))
        r += 1
        self._atom_elements_frame = ttk.Frame(view, padding=(8, 2, 0, 5))
        self._atom_elements_frame.grid(row=r, column=0, columnspan=2, sticky="ew")
        self._atom_elements_frame.grid_remove()
        self._rebuild_atom_element_checks(
            [], selected=params.get("atom_elements", [])
        )
        r += 1

        self._vars["show_bonds"] = tk.BooleanVar(value=bool(params["show_bonds"]))
        _labeled_check(view, r, "Show bonds", self._vars["show_bonds"])
        r += 1
        self._vars["bond_color"] = tk.StringVar(value=str(params.get("bond_color", "#555A60")))
        _labeled_color(view, r, "Bond color", self._vars["bond_color"],
                       "Colour used for all rendered bond tubes.")
        r += 1
        self._vars["bond_tolerance"] = tk.DoubleVar(value=float(params.get("bond_tolerance", 0.05)))
        _labeled_entry(
            view, r, "Bond tolerance", self._vars["bond_tolerance"],
            "Bond cutoff = (r_cov,i + r_cov,j) × (1 + tolerance).",
        )
        r += 1

        for key, label in (("show_labels", "Show labels"), ("show_grid", "Show grid"),
                           ("show_outline", "Show outline")):
            self._vars[key] = tk.BooleanVar(value=bool(params[key]))
            _labeled_check(view, r, label, self._vars[key])
            r += 1
        _section_header(view, r, "Appearance")
        r += 2
        self._vars["background_color"] = tk.StringVar(value=str(params["background_color"]))
        _labeled_color(view, r, "Background", self._vars["background_color"])
        r += 1
        self._vars["ambient_light"] = tk.DoubleVar(value=float(params["ambient_light"]))
        _labeled_entry(view, r, "Ambient light", self._vars["ambient_light"])
        r += 1
        _section_header(view, r, "Camera")
        r += 2
        self._vars["camera_preset"] = tk.StringVar(value=str(params["camera_preset"]))
        ttk.Label(view, text="Camera preset").grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Combobox(view, textvariable=self._vars["camera_preset"], values=list(CAMERA_PRESETS),
                     state="readonly", width=10).grid(row=r, column=1, sticky="ew", pady=2)
        r += 1
        self._vars["camera_projection"] = tk.StringVar(
            value=str(params.get("camera_projection", "perspective"))
        )
        ttk.Label(view, text="Projection").grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Combobox(
            view, textvariable=self._vars["camera_projection"],
            values=["perspective", "orthographic"], state="readonly", width=12,
        ).grid(row=r, column=1, sticky="ew", pady=2)
        r += 1
        camera_buttons = ttk.Frame(view)
        camera_buttons.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(5, 2))
        camera_buttons.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(camera_buttons, text="Apply", command=self._apply_camera).grid(
            row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(camera_buttons, text="Save view", command=lambda: self._call_optional(self._on_save_camera)).grid(
            row=0, column=1, sticky="ew", padx=3)
        ttk.Button(camera_buttons, text="Restore", command=lambda: self._call_optional(self._on_restore_camera)).grid(
            row=0, column=2, sticky="ew", padx=(3, 0))
        r += 1
        ttk.Label(
            view, text="Camera controls affect the complete 3D scene.",
            foreground="gray", wraplength=260, justify="left",
        ).grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 4))
        r += 1
        _section_header(view, r, "Export PNG")
        r += 2
        self._vars["png_dpi"] = tk.IntVar(value=int(params["png_dpi"]))
        _labeled_entry(view, r, "DPI", self._vars["png_dpi"])
        r += 1
        self._vars["png_width_inch"] = tk.DoubleVar(value=float(params["png_width_inch"]))
        _labeled_entry(view, r, "Width (inch)", self._vars["png_width_inch"])
        r += 1
        self._vars["png_transparent"] = tk.BooleanVar(value=bool(params["png_transparent"]))
        _labeled_check(view, r, "Transparent background", self._vars["png_transparent"])
        r += 1
        self._vars["export_view"] = tk.StringVar(value=str(params.get("export_view", "preset")))
        ttk.Label(view, text="Export view").grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Combobox(
            view, textvariable=self._vars["export_view"],
            values=["current", "preset"], state="readonly", width=10,
        ).grid(row=r, column=1, sticky="ew", pady=2)
        r += 1
        ttk.Button(view, text="Export PNG…", command=self._export_png).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(10, 3))
        r += 1
        preset_buttons = ttk.Frame(view)
        preset_buttons.grid(row=r, column=0, columnspan=2, sticky="ew", pady=3)
        preset_buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(preset_buttons, text="Save preset…", command=self._save_preset).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(preset_buttons, text="Load preset…", command=self._load_preset).grid(row=0, column=1, sticky="ew", padx=(3, 0))
        r += 1
        ttk.Button(view, text="Reset to defaults", command=self._reset).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(3, 8))

        # Analysis --------------------------------------------------------
        r = 0
        _section_header(analysis, r, "PCS Slice")
        r += 2
        ttk.Label(analysis, text="Oblique PCS plane through the metal centre.", foreground="gray",
                  wraplength=260, justify="left").grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 5))
        r += 1
        ttk.Button(analysis, text="Open Oblique PCS Slice…",
                   command=lambda: self._call_optional(self._on_oblique_slice)).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=2)
        r += 1
        _section_header(analysis, r, "PDE Comparison")
        r += 2
        ttk.Button(analysis, text="PDE PCS vs Point-dipole PCS…",
                   command=lambda: self._call_optional(self._on_compare_plot)).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=2)
        r += 1
        ttk.Button(analysis, text="PDE − Point residuals…",
                   command=lambda: self._call_optional(self._on_residual_plot)).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=2)
        r += 1
        _section_header(analysis, r, "Tensor")
        r += 2
        ttk.Button(analysis, text="Tensor spheroid…",
                   command=lambda: self._call_optional(self._on_tensor_spheroid)).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=2)
        r += 1
        _section_header(analysis, r, "Data Export")
        r += 2
        ttk.Button(analysis, text="Export atom PCS to CSV…",
                   command=lambda: self._call_optional(self._on_export_atom_csv)).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=2)
        r += 1
        ttk.Button(analysis, text="Export PCS to NumPy…",
                   command=lambda: self._call_optional(self._on_export_numpy)).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=2)

    @staticmethod
    def _call_optional(callback):
        if callback is not None:
            callback()

    def _apply_camera(self):
        if self._on_apply_camera is not None:
            self._on_apply_camera(self.get_params())

    def _toggle_atom_details(self):
        opened = not bool(self._atom_details_open.get())
        self._atom_details_open.set(opened)
        if opened:
            self._atom_elements_frame.grid()
            self._atom_details_button.configure(text="▾ Atom elements…")
        else:
            self._atom_elements_frame.grid_remove()
            self._atom_details_button.configure(text="▸ Atom elements…")

    def _rebuild_atom_element_checks(self, elements: list[str], selected=None):
        frame = getattr(self, "_atom_elements_frame", None)
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()

        unique = sorted({str(el).strip() for el in elements if str(el).strip()})
        self._elements = unique
        selected_set = {str(el) for el in (unique if selected is None else selected)}
        self._atom_element_vars = {}

        if not unique:
            ttk.Label(frame, text="Load an ORCA output to list elements.", foreground="gray").grid(
                row=0, column=0, sticky="w")
            return

        for idx, el in enumerate(unique):
            var = tk.BooleanVar(value=el in selected_set)
            self._atom_element_vars[el] = var
            ttk.Checkbutton(frame, text=el, variable=var).grid(
                row=idx // 4, column=idx % 4, sticky="w", padx=(0, 12), pady=1)

        buttons = ttk.Frame(frame)
        buttons.grid(row=(len(unique) + 3) // 4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Button(buttons, text="All", width=6,
                   command=lambda: [v.set(True) for v in self._atom_element_vars.values()]).pack(side="left")
        ttk.Button(buttons, text="None", width=6,
                   command=lambda: [v.set(False) for v in self._atom_element_vars.values()]).pack(side="left", padx=(4, 0))

    def set_elements(self, elements: list[str]):
        had_element_controls = bool(self._atom_element_vars)
        previous = {el for el, var in self._atom_element_vars.items() if bool(var.get())}
        new_unique = sorted({str(el).strip() for el in elements if str(el).strip()})
        selected = previous.intersection(new_unique) if had_element_controls else new_unique
        self._rebuild_atom_element_checks(new_unique, selected=selected)

    def _levels_to_list(self) -> list[dict]:
        out = []
        for row in self._level_rows:
            try:
                ppm = abs(float(row["ppm"].get()))
                if ppm <= 0:
                    continue
                out.append({
                    "ppm": ppm,
                    "pos_color": str(row["pos_color"].get()),
                    "neg_color": str(row["neg_color"].get()),
                    "style": str(row["style"].get()),
                    "opacity": float(row["opacity"].get()),
                })
            except Exception:
                continue

        if not out:
            out = list(self.DEFAULT_LEVEL_STYLES)
        out.sort(key=lambda d: float(d["ppm"]))
        return out

    def get_params(self) -> dict:
        raw = {}
        for k, var in self._vars.items():
            try:
                raw[k] = var.get()
            except tk.TclError:
                raw[k] = self.DEFAULTS.get(k)

        try:
            raw["temperature"] = float(raw["temperature"]) if str(raw["temperature"]).strip() else None
        except (ValueError, TypeError):
            raw["temperature"] = None

        raw["fft_pad_factor"] = int(raw.get("fft_pad_factor", 2))
        raw["normalize_density"] = bool(raw.get("normalize_density", True))
        raw["normalization_target"] = float(raw.get("normalization_target", 1.0))
        raw["density_isovalue"] = float(raw.get("density_isovalue", 0.005))
        raw["density_opacity"] = float(raw.get("density_opacity", 0.15))
        raw["atom_elements"] = [
            el for el, var in self._atom_element_vars.items() if bool(var.get())
        ]
        raw["bond_color"] = str(raw.get("bond_color", "#555A60"))
        raw["bond_tolerance"] = float(raw.get("bond_tolerance", 0.05))
        raw["ambient_light"] = float(raw.get("ambient_light", 0.50))
        raw["smooth_pcs_display"] = bool(raw.get("smooth_pcs_display", False))
        raw["smooth_pcs_sigma"] = float(raw.get("smooth_pcs_sigma", 1.0))
        raw["png_dpi"] = int(raw.get("png_dpi", 150))
        raw["png_width_inch"] = float(raw.get("png_width_inch", 6.0))
        raw["png_transparent"] = bool(raw.get("png_transparent", False))
        raw["camera_projection"] = str(raw.get("camera_projection", "perspective"))
        raw["export_view"] = str(raw.get("export_view", "preset"))
        raw["level_styles"] = self._levels_to_list()
        return raw

    def apply_params(self, params: dict):
        incoming = dict(params or {})
        # Backward compatibility for presets saved before density_isovalue
        # became the canonical key.
        if "density_isovalue" not in incoming and "density_iso" in incoming:
            incoming["density_isovalue"] = incoming["density_iso"]
        incoming.pop("density_iso", None)
        incoming.pop("auto_scale_pcs_levels", None)
        merged = {**self.DEFAULTS, **incoming}

        for k, v in merged.items():
            if k == "level_styles":
                continue
            var = self._vars.get(k)
            if var is not None:
                try:
                    var.set(v)
                except Exception:
                    pass

        self._rebuild_level_rows(merged.get("level_styles", self.DEFAULT_LEVEL_STYLES))
        if self._elements:
            self._rebuild_atom_element_checks(
                self._elements, selected=merged.get("atom_elements", self._elements)
            )

    def set_temperatures(self, temps: list[float]):
        self._temperatures = sorted(float(t) for t in temps)
        if getattr(self, "_temp_combo", None) is not None:
            self._temp_combo.configure(values=[""] + [f"{t:g}" for t in self._temperatures])

        cur = str(self._vars["temperature"].get()).strip()
        allowed = {f"{t:g}" for t in self._temperatures}
        if cur and cur not in allowed:
            self._vars["temperature"].set("")
        self._refresh_temperature_summary()

    def update_status(self, text: str):
        pass

    def _run(self):
        self._on_run(self.get_params())

    def _refresh_view(self):
        if self._on_refresh_view is not None:
            self._on_refresh_view(self.get_params())

    def _export_png(self):
        if self._on_export_png is not None:
            self._on_export_png(self.get_params())

    def _reset(self):
        self.apply_params(self.DEFAULTS)

    def _save_preset(self):
        path = filedialog.asksaveasfilename(
            title="Save preset",
            defaultextension=".json",
            filetypes=[("JSON preset", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        data = self.get_params()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        self.update_status(f"Preset saved: {Path(path).name}")

    def _load_preset(self):
        path = filedialog.askopenfilename(
            title="Load preset",
            filetypes=[("JSON preset", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.apply_params(data)
        self.update_status(f"Preset loaded: {Path(path).name}")
