# tools/ui_pcs_pde_control.py
"""
Control panel GUI for the PCS-PDE viewer (FFT).
"""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog
from typing import Callable, Optional


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
    sep = ttk.Separator(parent, orient="horizontal")
    sep.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 2))
    lbl = ttk.Label(parent, text=text, font=("TkDefaultFont", 9, "bold"))
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
            font=("TkDefaultFont", 8),
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
        "auto_scale_pcs_levels": True,
        "density_iso": 0.005,
        "show_atoms": False,
        "show_bonds": True,
        "show_density": False,
        "show_pcs": True,
        "show_labels": False,
        "show_grid": False,
        "show_outline": False,
        "background_color": "white",
        "camera_preset": "iso",
        "density_color": "#27af91",
        "density_style": "both",
        "density_opacity": 0.15,
        "ambient_light": 0.50,
        "smooth_pcs_display": False,
        "smooth_pcs_sigma": 1.0,
        "png_dpi": 600,
        "png_width_inch": 6.0,
        "png_transparent": False,
        "level_styles": DEFAULT_LEVEL_STYLES,
    }

    def __init__(
        self,
        parent,
        on_run_callback: Callable[[dict], None],
        on_refresh_view_callback: Optional[Callable[[dict], None]] = None,
        on_export_png_callback: Optional[Callable[[dict], None]] = None,
        temperatures: Optional[list[float]] = None,
        initial_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        self._on_run = on_run_callback
        self._on_refresh_view = on_refresh_view_callback
        self._on_export_png = on_export_png_callback
        self._temperatures = temperatures or []

        params = {**self.DEFAULTS, **(initial_params or {})}
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

    def _build_ui(self, params: dict):
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
            font=("TkDefaultFont", 8),
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

        self._vars["auto_scale_pcs_levels"] = tk.BooleanVar(value=bool(params["auto_scale_pcs_levels"]))
        _labeled_check(inner, r, "Auto-scale PCS contour levels", self._vars["auto_scale_pcs_levels"])
        r += 1

        self._vars["normalization_target"] = tk.DoubleVar(value=float(params["normalization_target"]))
        _labeled_entry(inner, r, "Normalization target", self._vars["normalization_target"])
        r += 1

        _section_header(inner, r, "Density Surface")
        r += 2

        self._vars["density_iso"] = tk.DoubleVar(value=float(params["density_iso"]))
        _labeled_entry(inner, r, "Density isovalue fraction", self._vars["density_iso"])
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
            ttk.Label(hdr, text=txt, font=("TkDefaultFont", 8, "bold")).grid(
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
            font=("TkDefaultFont", 8),
            wraplength=240,
            justify="left",
        ).grid(row=r, column=0, columnspan=2, pady=(10, 4), sticky="w")

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
        raw["density_iso"] = float(raw.get("density_iso", 0.005))
        raw["density_opacity"] = float(raw.get("density_opacity", 0.15))
        raw["ambient_light"] = float(raw.get("ambient_light", 0.50))
        raw["smooth_pcs_display"] = bool(raw.get("smooth_pcs_display", False))
        raw["smooth_pcs_sigma"] = float(raw.get("smooth_pcs_sigma", 1.0))
        raw["png_dpi"] = int(raw.get("png_dpi", 600))
        raw["png_width_inch"] = float(raw.get("png_width_inch", 6.0))
        raw["png_transparent"] = bool(raw.get("png_transparent", False))
        raw["level_styles"] = self._levels_to_list()
        return raw

    def apply_params(self, params: dict):
        merged = {**self.DEFAULTS, **(params or {})}

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