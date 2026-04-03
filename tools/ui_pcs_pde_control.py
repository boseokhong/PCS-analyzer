# tools/ui_pcs_pde_control.py
"""
Control panel GUI for the PCS-PDE viewer (FFT).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, colorchooser
from typing import Callable, Optional


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
            wraplength=260,
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
    DEFAULTS: dict = {
        "temperature": "",
        "fft_pad_factor": 2,
        "normalize_density": True,
        "normalization_target": 1.0,
        "auto_scale_pcs_levels": True,
        "density_iso": 0.005,
        "pcs_level_neg": -2.0,
        "pcs_level_pos": 2.0,
        "show_bonds": True,
        "show_density": True,
        "show_pcs": True,
        "show_labels": False,
        "show_grid": False,
        "show_outline": False,
        "background_color": "white",
        "pcs_pos_color": "#ff0000",
        "pcs_neg_color": "#0000ff",
        "pcs_style": "surface",
        "pcs_opacity": 0.30,
        "density_color": "#27af91",
        "density_style": "both",
        "density_opacity": 0.15,
        "ambient_light": 0.50,
        "smooth_pcs_display": False,
        "smooth_pcs_sigma": 1.0,
    }

    def __init__(
        self,
        parent,
        on_run_callback: Callable[[dict], None],
        on_export_callback: Optional[Callable[[dict], None]] = None,
        temperatures: Optional[list[float]] = None,
        initial_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        self._on_run = on_run_callback
        self._on_export = on_export_callback
        self._temperatures = temperatures or []
        params = {**self.DEFAULTS, **(initial_params or {})}
        self._vars: dict[str, tk.Variable] = {}
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
        _labeled_entry(
            inner, r, "Padding factor", self._vars["fft_pad_factor"],
            tooltip=(
                "Zero padding added symmetrically by multiples of the original grid size.\n"
                "Example: factor 2 expands each axis to 5× the original size."
            )
        )
        r += 1

        self._vars["normalize_density"] = tk.BooleanVar(value=bool(params["normalize_density"]))
        _labeled_check(
            inner, r, "Normalize density integral", self._vars["normalize_density"],
            tooltip="Scale density so that the integrated density matches the target value."
        )
        r += 1

        self._vars["auto_scale_pcs_levels"] = tk.BooleanVar(value=bool(params["auto_scale_pcs_levels"]))
        _labeled_check(
            inner, r, "Auto-scale PCS contour levels",
            self._vars["auto_scale_pcs_levels"],
            tooltip=(
                "When density normalization is enabled, multiply PCS contour levels "
                "by the normalization scale factor for more comparable cone shapes."
            )
        )
        r += 1

        self._vars["normalization_target"] = tk.DoubleVar(value=float(params["normalization_target"]))
        _labeled_entry(
            inner, r, "Normalization target", self._vars["normalization_target"],
            tooltip="Target value for the integrated density after normalization."
        )
        r += 1

        _section_header(inner, r, "Visualisation")
        r += 2

        self._vars["density_iso"] = tk.DoubleVar(value=params["density_iso"])
        _labeled_entry(inner, r, "Density isovalue", self._vars["density_iso"])
        r += 1

        self._vars["pcs_level_neg"] = tk.DoubleVar(value=params["pcs_level_neg"])
        _labeled_entry(inner, r, "PCS level − (ppm)", self._vars["pcs_level_neg"])
        r += 1

        self._vars["pcs_level_pos"] = tk.DoubleVar(value=params["pcs_level_pos"])
        _labeled_entry(inner, r, "PCS level + (ppm)", self._vars["pcs_level_pos"])
        r += 1

        self._vars["show_bonds"] = tk.BooleanVar(value=params["show_bonds"])
        _labeled_check(inner, r, "Show bonds", self._vars["show_bonds"])
        r += 1

        self._vars["show_density"] = tk.BooleanVar(value=params["show_density"])
        _labeled_check(inner, r, "Show spin density surface", self._vars["show_density"])
        r += 1

        self._vars["show_pcs"] = tk.BooleanVar(value=params["show_pcs"])
        _labeled_check(inner, r, "Show PCS surface", self._vars["show_pcs"])
        r += 1

        self._vars["show_labels"] = tk.BooleanVar(value=params["show_labels"])
        _labeled_check(inner, r, "Show labels", self._vars["show_labels"])
        r += 1

        self._vars["show_grid"] = tk.BooleanVar(value=params["show_grid"])
        _labeled_check(inner, r, "Show grid", self._vars["show_grid"])
        r += 1

        self._vars["show_outline"] = tk.BooleanVar(value=params["show_outline"])
        _labeled_check(inner, r, "Show outline", self._vars["show_outline"])
        r += 1

        self._vars["density_style"] = tk.StringVar(value=str(params["density_style"]))
        lbl = ttk.Label(inner, text="Density style")
        lbl.grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Combobox(
            inner,
            textvariable=self._vars["density_style"],
            values=["surface", "mesh", "both"],
            state="readonly",
            width=10,
        ).grid(row=r, column=1, sticky="ew", pady=2)
        r += 1

        self._vars["density_color"] = tk.StringVar(value=str(params["density_color"]))
        _labeled_color(
            inner, r, "Density color", self._vars["density_color"],
            tooltip="Spin density isosurface colour."
        )
        r += 1

        self._vars["density_opacity"] = tk.DoubleVar(value=float(params["density_opacity"]))
        _labeled_entry(inner, r, "Density opacity", self._vars["density_opacity"])
        r += 1

        self._vars["pcs_style"] = tk.StringVar(value=str(params["pcs_style"]))
        lbl = ttk.Label(inner, text="PCS style")
        lbl.grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Combobox(
            inner,
            textvariable=self._vars["pcs_style"],
            values=["surface", "mesh", "both"],
            state="readonly",
            width=10,
        ).grid(row=r, column=1, sticky="ew", pady=2)
        r += 1

        self._vars["pcs_pos_color"] = tk.StringVar(value=str(params["pcs_pos_color"]))
        _labeled_color(
            inner, r, "PCS + color", self._vars["pcs_pos_color"],
            tooltip="Positive PCS isosurface colour."
        )
        r += 1

        self._vars["pcs_neg_color"] = tk.StringVar(value=str(params["pcs_neg_color"]))
        _labeled_color(
            inner, r, "PCS − color", self._vars["pcs_neg_color"],
            tooltip="Negative PCS isosurface colour."
        )
        r += 1

        self._vars["pcs_opacity"] = tk.DoubleVar(value=float(params["pcs_opacity"]))
        _labeled_entry(inner, r, "PCS opacity", self._vars["pcs_opacity"])
        r += 1

        self._vars["background_color"] = tk.StringVar(value=params["background_color"])
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
        _labeled_check(
            inner, r, "Smooth PCS display (Gaussian)",
            self._vars["smooth_pcs_display"],
            tooltip="Apply Gaussian smoothing to the PCS field before rendering. Does not affect computed values.",
        )
        r += 1

        self._vars["smooth_pcs_sigma"] = tk.DoubleVar(value=float(params["smooth_pcs_sigma"]))
        _labeled_entry(
            inner, r, "  Smooth sigma (voxels)", self._vars["smooth_pcs_sigma"],
            tooltip="Standard deviation of the Gaussian kernel in voxels. Larger = smoother.",
        )
        r += 1

        sep2 = ttk.Separator(inner, orient="horizontal")
        sep2.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(14, 8))
        r += 1

        btn_frame = ttk.Frame(inner)
        btn_frame.grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1
        btn_frame.columnconfigure((0, 1), weight=1)

        ttk.Button(btn_frame, text="▶  Run", style="Accent.TButton", command=self._run).grid(
            row=0, column=0, padx=(0, 4), sticky="ew", ipady=4
        )

        if self._on_export is not None:
            ttk.Button(btn_frame, text="⬇  Export NPZ…", command=self._export).grid(
                row=0, column=1, padx=(4, 0), sticky="ew", ipady=4
            )

        ttk.Button(inner, text="Reset to defaults", command=self._reset).grid(
            row=r, column=0, columnspan=2, pady=(6, 0), sticky="ew"
        )
        r += 1

        # bottom help text
        ttk.Label(
            inner,
            text=(
                "FFT mode: padding factor, density normalization, "
                "and contour rendering options are the main controls."
            ),
            foreground="gray",
            font=("TkDefaultFont", 8),
            wraplength=220,
        ).grid(row=r, column=0, columnspan=2, pady=(10, 4), sticky="w")

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
        raw["pcs_opacity"] = float(raw.get("pcs_opacity", 0.30))
        raw["density_opacity"] = float(raw.get("density_opacity", 0.15))
        raw["ambient_light"] = float(raw.get("ambient_light", 0.50))
        raw["smooth_pcs_display"] = bool(raw.get("smooth_pcs_display", False))
        raw["smooth_pcs_sigma"] = float(raw.get("smooth_pcs_sigma", 1.0))
        return raw

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

    def _export(self):
        if self._on_export:
            self._on_export(self.get_params())

    def _reset(self):
        for k, v in self.DEFAULTS.items():
            var = self._vars.get(k)
            if var is not None:
                var.set(v)