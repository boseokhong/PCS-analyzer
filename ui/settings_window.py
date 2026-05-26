# ui/settings_window.py

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from logic.app_settings import save_app_state, build_default_settings

def open_settings_window(state: dict):
    root = state["root"]
    win = state.get("settings_window")

    if win is not None:
        try:
            if win.winfo_exists():
                win.lift()
                win.focus_force()
                return
        except Exception:
            pass

    app_settings = state.setdefault("app_settings", build_default_settings())

    win = tk.Toplevel(root)
    win.title("Settings")
    win.geometry("700x660")
    win.minsize(640, 540)
    state["settings_window"] = win

    outer = ttk.Frame(win, padding=10)
    outer.pack(fill="both", expand=True)

    nb = ttk.Notebook(outer)
    nb.pack(fill="both", expand=True)

    tab_general = ttk.Frame(nb)
    tab_appearance = ttk.Frame(nb)

    nb.add(tab_general, text="General")
    nb.add(tab_appearance, text="Appearance")

    # ----------------------------
    # Tk variables
    # ----------------------------
    vars_ = {
        "open_2d_plot_on_start": tk.BooleanVar(value=bool(app_settings.get("open_2d_plot_on_start", True))),
        "auto_open_3d_on_load": tk.BooleanVar(value=bool(app_settings.get("auto_open_3d_on_load", True))),
        "remember_window_geometry": tk.BooleanVar(value=bool(app_settings.get("remember_window_geometry", True))),
        "remember_recent_files": tk.BooleanVar(value=bool(app_settings.get("remember_recent_files", True))),
        "max_recent_files": tk.StringVar(value=str(app_settings.get("max_recent_files", 8))),
        "default_dchi_ax": tk.StringVar(value=str(app_settings.get("default_dchi_ax", -2.0))),
        "default_pcs_min": tk.StringVar(value=str(app_settings.get("default_pcs_min", -10.0))),
        "default_pcs_max": tk.StringVar(value=str(app_settings.get("default_pcs_max", 10.0))),
        "default_pcs_interval": tk.StringVar(value=str(app_settings.get("default_pcs_interval", 0.5))),
        "export_default_dpi": tk.StringVar(value=str(app_settings.get("export_default_dpi", 600))),
        "theme_variant": tk.StringVar(value=str(app_settings.get("theme_variant", "light"))),
        "theme_accent": tk.StringVar(value=str(app_settings.get("theme_accent", "green"))),
        "font_family_ui": tk.StringVar(value=str(app_settings.get("font_family_ui", "Segoe UI"))),
        "font_size_ui": tk.StringVar(value=str(app_settings.get("font_size_ui", 10))),
        "font_family_table": tk.StringVar(value=str(app_settings.get("font_family_table", "Segoe UI"))),
        "font_size_table": tk.StringVar(value=str(app_settings.get("font_size_table", 10))),
        "font_family_report": tk.StringVar(value=str(app_settings.get("font_family_report", "Consolas"))),
        "font_size_report": tk.StringVar(value=str(app_settings.get("font_size_report", 9))),
        "font_family_plot": tk.StringVar(value=str(app_settings.get("font_family_plot", "DejaVu Sans"))),
        "font_size_plot": tk.StringVar(value=str(app_settings.get("font_size_plot", 9))),
        "font_scale": tk.StringVar(value=str(app_settings.get("font_scale", 1.0))),
    }

    # ----------------------------
    # General tab
    # ----------------------------
    startup_box = ttk.LabelFrame(tab_general, text="Startup", padding=10)
    startup_box.pack(fill="x", padx=10, pady=(10, 6))

    ttk.Checkbutton(
        startup_box,
        text="Open 2D PCS plot on startup",
        variable=vars_["open_2d_plot_on_start"],
    ).pack(anchor="w", pady=2)

    ttk.Checkbutton(
        startup_box,
        text="Auto-open 3D structure after loading XYZ",
        variable=vars_["auto_open_3d_on_load"],
    ).pack(anchor="w", pady=2)

    ttk.Checkbutton(
        startup_box,
        text="Remember window geometry",
        variable=vars_["remember_window_geometry"],
    ).pack(anchor="w", pady=2)

    recent_box = ttk.LabelFrame(tab_general, text="Recent files", padding=10)
    recent_box.pack(fill="x", padx=10, pady=6)

    ttk.Checkbutton(
        recent_box,
        text="Remember recent files",
        variable=vars_["remember_recent_files"],
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

    ttk.Label(recent_box, text="Maximum recent files:").grid(row=1, column=0, sticky="w", pady=(8, 2))
    ttk.Spinbox(
        recent_box,
        from_=1,
        to=20,
        textvariable=vars_["max_recent_files"],
        width=8,
    ).grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(8, 2))

    defaults_box = ttk.LabelFrame(tab_general, text="Defaults", padding=10)
    defaults_box.pack(fill="x", padx=10, pady=6)

    ttk.Label(defaults_box, text="Default Δχ_ax (E-32 m³):").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Entry(defaults_box, textvariable=vars_["default_dchi_ax"], width=12).grid(row=0, column=1, sticky="w", padx=(10, 0), pady=2)

    ttk.Label(defaults_box, text="Default PCS range:").grid(row=1, column=0, sticky="w", pady=(10, 2))

    range_row = ttk.Frame(defaults_box)
    range_row.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(10, 2))

    ttk.Label(range_row, text="Min").pack(side="left")
    ttk.Entry(range_row, textvariable=vars_["default_pcs_min"], width=8).pack(side="left", padx=(4, 10))
    ttk.Label(range_row, text="Max").pack(side="left")
    ttk.Entry(range_row, textvariable=vars_["default_pcs_max"], width=8).pack(side="left", padx=(4, 10))
    ttk.Label(range_row, text="Interval").pack(side="left")
    ttk.Entry(range_row, textvariable=vars_["default_pcs_interval"], width=8).pack(side="left", padx=(4, 0))

    export_box = ttk.LabelFrame(tab_general, text="Export", padding=10)
    export_box.pack(fill="x", padx=10, pady=6)

    ttk.Label(export_box, text="Default export DPI:").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Combobox(
        export_box,
        textvariable=vars_["export_default_dpi"],
        values=["150", "300", "600", "900", "1200"],
        state="readonly",
        width=10,
    ).grid(row=0, column=1, sticky="w", padx=(10, 0), pady=2)

    # ----------------------------
    # Appearance tab
    # ----------------------------
    appearance_box = ttk.LabelFrame(tab_appearance, text="Theme", padding=10)
    appearance_box.pack(fill="x", padx=10, pady=(10, 6))

    ttk.Label(appearance_box, text="Theme:").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Combobox(
        appearance_box,
        textvariable=vars_["theme_variant"],
        values=["light", "dark"],
        state="readonly",
        width=14,
    ).grid(row=0, column=1, sticky="w", padx=(10, 0), pady=2)

    ttk.Label(appearance_box, text="Accent color:").grid(row=1, column=0, sticky="w", pady=(10, 2))
    ttk.Combobox(
        appearance_box,
        textvariable=vars_["theme_accent"],
        values=["blue", "green", "orange", "purple"],
        state="readonly",
        width=14,
    ).grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(10, 2))

    fonts_box = ttk.LabelFrame(tab_appearance, text="Fonts", padding=10)
    fonts_box.pack(fill="x", padx=10, pady=6)
    fonts_box.columnconfigure(1, weight=1)

    common_ui_fonts = ["Segoe UI", "Arial", "Helvetica", "TkDefaultFont"]
    common_report_fonts = ["Consolas", "Courier New", "Menlo", "Monaco", "TkFixedFont"]
    common_plot_fonts = ["DejaVu Sans", "Arial", "Helvetica", "Segoe UI"]

    def _font_row(row, label, family_key, size_key, values):
        ttk.Label(fonts_box, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Combobox(
            fonts_box,
            textvariable=vars_[family_key],
            values=values,
            width=18,
        ).grid(row=row, column=1, sticky="ew", padx=(10, 8), pady=2)
        ttk.Spinbox(
            fonts_box,
            from_=6,
            to=48,
            textvariable=vars_[size_key],
            width=6,
        ).grid(row=row, column=2, sticky="w", pady=2)

    _font_row(0, "UI font:", "font_family_ui", "font_size_ui", common_ui_fonts)
    _font_row(1, "Table font:", "font_family_table", "font_size_table", common_ui_fonts)
    _font_row(2, "Report font:", "font_family_report", "font_size_report", common_report_fonts)
    _font_row(3, "Plot font:", "font_family_plot", "font_size_plot", common_plot_fonts)

    ttk.Label(fonts_box, text="Global scale:").grid(row=4, column=0, sticky="w", pady=(8, 2))
    ttk.Combobox(
        fonts_box,
        textvariable=vars_["font_scale"],
        values=["0.85", "0.90", "1.0", "1.10", "1.20", "1.35", "1.50"],
        width=8,
    ).grid(row=4, column=1, sticky="w", padx=(10, 0), pady=(8, 2))

    ttk.Label(
        fonts_box,
        text=(
            "UI/Table/Report fonts affect Tk widgets. Plot font size is used as the "
            "base size for Matplotlib/PyVista labels."
        ),
        foreground="#666666",
        wraplength=600,
        justify="left",
    ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))

    ttk.Label(
        tab_appearance,
        text="Changes to appearance are applied when you click Apply or OK.",
        foreground="#666666",
    ).pack(anchor="w", padx=14, pady=(4, 0))

    # ----------------------------
    # Buttons
    # ----------------------------
    btn_row = ttk.Frame(outer)
    btn_row.pack(fill="x", pady=(10, 0))

    def _validate_settings():
        try:
            max_recent_files = int(vars_["max_recent_files"].get())
            default_dchi_ax = float(vars_["default_dchi_ax"].get())
            default_pcs_min = float(vars_["default_pcs_min"].get())
            default_pcs_max = float(vars_["default_pcs_max"].get())
            default_pcs_interval = float(vars_["default_pcs_interval"].get())
            export_default_dpi = int(vars_["export_default_dpi"].get())
            font_size_ui = int(vars_["font_size_ui"].get())
            font_size_table = int(vars_["font_size_table"].get())
            font_size_report = int(vars_["font_size_report"].get())
            font_size_plot = int(vars_["font_size_plot"].get())
            font_scale = float(vars_["font_scale"].get())
        except Exception:
            raise ValueError("One or more numeric settings are invalid.")

        if max_recent_files < 1:
            raise ValueError("Maximum recent files must be at least 1.")
        if default_pcs_min >= default_pcs_max:
            raise ValueError("Default PCS min must be smaller than max.")
        if default_pcs_interval <= 0:
            raise ValueError("Default PCS interval must be greater than 0.")
        if export_default_dpi <= 0:
            raise ValueError("Default export DPI must be greater than 0.")
        for label, size in (
            ("UI font size", font_size_ui),
            ("Table font size", font_size_table),
            ("Report font size", font_size_report),
            ("Plot font size", font_size_plot),
        ):
            if not (6 <= size <= 48):
                raise ValueError(f"{label} must be between 6 and 48.")
        if not (0.5 <= font_scale <= 2.5):
            raise ValueError("Global font scale must be between 0.5 and 2.5.")

        def _font_family(key, fallback):
            value = vars_[key].get().strip()
            return value or fallback

        return {
            "theme_variant": vars_["theme_variant"].get().strip(),
            "theme_accent": vars_["theme_accent"].get().strip(),
            "open_2d_plot_on_start": bool(vars_["open_2d_plot_on_start"].get()),
            "auto_open_3d_on_load": bool(vars_["auto_open_3d_on_load"].get()),
            "remember_window_geometry": bool(vars_["remember_window_geometry"].get()),
            "remember_recent_files": bool(vars_["remember_recent_files"].get()),
            "max_recent_files": max_recent_files,
            "default_dchi_ax": default_dchi_ax,
            "default_pcs_min": default_pcs_min,
            "default_pcs_max": default_pcs_max,
            "default_pcs_interval": default_pcs_interval,
            "export_default_dpi": export_default_dpi,
            "font_family_ui": _font_family("font_family_ui", "Segoe UI"),
            "font_size_ui": font_size_ui,
            "font_family_table": _font_family("font_family_table", "Segoe UI"),
            "font_size_table": font_size_table,
            "font_family_report": _font_family("font_family_report", "Consolas"),
            "font_size_report": font_size_report,
            "font_family_plot": _font_family("font_family_plot", "DejaVu Sans"),
            "font_size_plot": font_size_plot,
            "font_scale": font_scale,
        }

    def _apply_settings():
        try:
            new_settings = _validate_settings()
        except Exception as e:
            state["messagebox"].showerror("Settings", str(e))
            return False

        state["app_settings"].update(new_settings)

        # Appearance: apply immediately
        try:
            from ui.style import apply_style
            style = apply_style(
                state["root"],
                variant=state["app_settings"]["theme_variant"],
                accent=state["app_settings"]["theme_accent"],
                settings=state["app_settings"],
            )
            state["style"] = style
            state["fonts"] = getattr(state["root"], "_app_fonts", {})
        except Exception:
            pass

        # Refresh visible views if possible
        try:
            if "update_graph" in state:
                state["update_graph"]()
        except Exception:
            pass

        try:
            has_structure = bool(state.get("atom_data") or state.get("atom_data_raw"))
            if has_structure and "rh_refresh_table" in state:
                state["rh_refresh_table"]()
        except Exception:
            pass

        try:
            save_app_state(state)
        except Exception:
            pass

        try:
            if "rebuild_recent_files_menu" in state:
                state["rebuild_recent_files_menu"]()
        except Exception:
            pass

        try:
            if "refresh_checklist_ui" in state:
                state["refresh_checklist_ui"]()
        except Exception:
            pass

        try:
            for key in ("angle_x_slider", "angle_y_slider", "rh_angle_z_slider"):
                w = state.get(key)
                if w is not None and "apply_scale_theme" in state:
                    state["apply_scale_theme"](w)
        except Exception:
            pass

        return True

    def _on_apply():
        _apply_settings()

    def _on_ok():
        if _apply_settings():
            _on_close()

    def _on_close():
        try:
            win.destroy()
        finally:
            state["settings_window"] = None

    ttk.Button(btn_row, text="Apply", command=_on_apply).pack(side="right")
    ttk.Button(btn_row, text="OK", command=_on_ok).pack(side="right", padx=(0, 6))
    ttk.Button(btn_row, text="Cancel", command=_on_close).pack(side="right", padx=(0, 6))

    win.protocol("WM_DELETE_WINDOW", _on_close)