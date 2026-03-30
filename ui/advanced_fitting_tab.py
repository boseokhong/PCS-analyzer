# ui/advanced_fitting_tab.py

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from logic.advanced_physics import (
    Conformer,
    get_current_coords_and_metal,
    get_tensor_values,
    get_exp_pcs,
    get_selected_ids,
    fit_multiconf,
    fit_joint_pcs_rdc,
    fit_pre,
    pre_gamma1,
    pre_gamma2,
    LANTHANIDE_DB,
    fit_multilanthanid,
)
from logic.xyz_loader import load_structure

# ============================================================================
# Public builder
# ============================================================================
def build_advanced_fitting_tab(state: dict, parent_notebook: ttk.Notebook):
    advanced_frame = ttk.Frame(parent_notebook)
    parent_notebook.add(advanced_frame, text="🔍 Advanced Fitting")

    outer = ttk.Frame(advanced_frame, padding=4)
    outer.pack(fill="both", expand=True)

    sub_nb = ttk.Notebook(outer)
    sub_nb.pack(fill="both", expand=True)

    _build_multiconformer_tab(state, sub_nb)
    _build_joint_pcs_rdc_tab(state, sub_nb)
    _build_pre_tab(state, sub_nb)
    _build_multilanthanide_tab(state, sub_nb)

    state["advanced_fitting_nb"] = sub_nb
    return advanced_frame

# ============================================================================
# Common helpers
# ============================================================================
# template export
def _export_rdc_template():
    path = filedialog.asksaveasfilename(
        title="Save RDC template",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
    )
    if not path:
        return

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("atom1,atom2,rdc_exp,rdc_err,nuc1,nuc2\n")
        f.write("H12,N3,8.25,0.50,H,N\n")

    messagebox.showinfo("RDC Template", f"Template saved:\n{path}")

def _export_pre_template():
    path = filedialog.asksaveasfilename(
        title="Save PRE template",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
    )
    if not path:
        return

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("ref,r2_para\n")
        f.write("12,145.3\n")

    messagebox.showinfo("PRE Template", f"Template saved:\n{path}")

def _run_in_thread(fn, state: dict):
    threading.Thread(target=fn, args=(state,), daemon=True).start()

def _show_traceback_error(state: dict, title: str):
    import traceback
    tb = traceback.format_exc()
    state["root"].after(0, lambda: messagebox.showerror(title, tb))

def _set_textbox_content(text_widget, content: str):
    text_widget.config(state="normal")
    text_widget.delete("1.0", "end")
    text_widget.insert("end", content)
    text_widget.config(state="disabled")

def _fill_listbox(lb, lines):
    if lb is None:
        return
    lb.delete(0, "end")
    for line in lines:
        lb.insert("end", line)

def _build_result_panel(parent, state: dict, key_prefix: str, text_height: int = 5, fig_size=(4.0, 2.0)):
    body = ttk.Frame(parent, padding=(6, 2, 6, 6))
    body.pack(fill="both", expand=True)

    summary_frame = ttk.LabelFrame(body, text="Fit Summary", padding=4)
    summary_frame.pack(fill="both", expand=True)

    text_key = f"{key_prefix}_result_text"
    state[text_key] = scrolledtext.ScrolledText(
        summary_frame,
        height=text_height,
        wrap="none",
        font=("Courier", 9),
    )
    state[text_key].pack(fill="both", expand=True)

    plot_frame = ttk.LabelFrame(body, text="Result Plot", padding=4)
    plot_frame.pack(fill="both", expand=True, pady=(4, 0))

    fig_key = f"{key_prefix}_fig"
    canvas_key = f"{key_prefix}_canvas"

    state[fig_key] = plt.Figure(figsize=fig_size, dpi=100)
    state[canvas_key] = FigureCanvasTkAgg(state[fig_key], master=plot_frame)
    state[canvas_key].get_tk_widget().pack(fill="both", expand=True)

    return body

def _draw_identity_scatter(ax, x, y, title, xlabel="Exp.", ylabel="Pred."):
    vmin = min(x.min(), y.min()) * 1.1
    vmax = max(x.max(), y.max()) * 1.1
    ax.plot([vmin, vmax], [vmin, vmax], "--", color="gray", alpha=0.5)
    ax.scatter(x, y, s=24, alpha=0.85)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)

def _build_placeholder_tab(nb: ttk.Notebook, title: str, message: str):
    tab = ttk.Frame(nb)
    nb.add(tab, text=title)

    outer = ttk.Frame(tab, padding=10)
    outer.pack(fill="both", expand=True)

    ttk.Label(
        outer,
        text=message,
        anchor="center",
        justify="center"
    ).pack(expand=True)

# Apply tensor to main helper
def _apply_tensor_to_main(state: dict, dchi_ax: float, dchi_rh: float | None = None):
    """
    Apply fitted tensor values back to the main application state
    and refresh the main tensor-dependent views.
    """
    tensor_entry = state.get("tensor_entry")
    if tensor_entry is not None:
        try:
            tensor_entry.delete(0, tk.END)
            tensor_entry.insert(0, f"{float(dchi_ax):.6g}")
        except Exception:
            pass

    if dchi_rh is not None:
        try:
            state["rh_dchi_rh"] = float(dchi_rh)
        except Exception:
            pass

        rh_var = state.get("rh_dchi_rh_var")
        if rh_var is not None:
            try:
                rh_var.set(f"{float(dchi_rh):g}")
            except Exception:
                pass

    # Refresh tensor-derived labels if available
    try:
        if "maybe_calc_chi" in state:
            state["maybe_calc_chi"](quiet=True)
    except Exception:
        pass

    # Refresh main PCS / cartesian / dependent views
    try:
        if "update_graph" in state:
            state["update_graph"]()
    except Exception:
        pass

    try:
        if "plot_cartesian_graph_fn" in state:
            state["plot_cartesian_graph_fn"](state)
    except Exception:
        pass

    # Refresh rhombicity table if the helper exists
    try:
        refresh_fn = state.get("rh_refresh_table")
        if callable(refresh_fn):
            refresh_fn()
    except Exception:
        pass

    # Refresh diagnostic panel
    try:
        from logic.diagnostic import update_diagnostic_panel
        update_diagnostic_panel(state)
    except Exception:
        pass

# ============================================================================
# Multi-Conformer tab
# ============================================================================
def _build_multiconformer_tab(state: dict, nb: ttk.Notebook):
    tab = ttk.Frame(nb)
    nb.add(tab, text="Multi-Conformer")

    state["advfit_conformers"] = []

    top = ttk.Frame(tab, padding=(6, 6, 6, 2))
    top.pack(fill="x")

    btn_row = ttk.Frame(top)
    btn_row.pack(fill="x")

    ttk.Button(
        btn_row,
        text="Use Current Structure",
        command=lambda: _mc_add_current_structure(state),
        width=18,
    ).pack(side="left", padx=4)

    ttk.Button(
        btn_row,
        text="Load XYZ",
        command=lambda: _mc_load_conformer_xyz(state),
        width=10,
    ).pack(side="left", padx=4)

    ttk.Button(
        btn_row,
        text="Clear",
        command=lambda: _mc_clear_conformers(state),
        width=8,
    ).pack(side="left", padx=4)

    ttk.Button(
        btn_row,
        text="Run Fit",
        command=lambda: _mc_run_threaded(state),
        width=10,
    ).pack(side="right")

    # Keep this inside btn_row as requested
    ttk.Label(
        btn_row,
        text="Same atom order is required across conformers.",
    ).pack(anchor="w", pady=(0, 3))

    opt_row = ttk.Frame(top)
    opt_row.pack(fill="x", pady=(4, 2))

    state["advfit_mc_fit_weights_var"] = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        opt_row,
        text="Fit weights",
        variable=state["advfit_mc_fit_weights_var"],
    ).pack(side="left")

    state["advfit_mc_fit_rh_var"] = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        opt_row,
        text="Fit Δχ_rh",
        variable=state["advfit_mc_fit_rh_var"],
    ).pack(side="left", padx=8)

    ttk.Button(
        opt_row,
        text="Apply to Main",
        command=lambda: _mc_apply_to_main(state),
        width=12,
    ).pack(side="right")

    list_box = ttk.LabelFrame(top, text="Conformers", padding=4)
    list_box.pack(fill="x", pady=(2, 2))

    state["advfit_mc_listbox"] = tk.Listbox(
        list_box,
        height=3,
        exportselection=False,
    )
    state["advfit_mc_listbox"].pack(side="left", fill="x", expand=True)

    list_scroll = ttk.Scrollbar(
        list_box,
        orient="vertical",
        command=state["advfit_mc_listbox"].yview,
    )
    list_scroll.pack(side="left", fill="y")
    state["advfit_mc_listbox"].configure(yscrollcommand=list_scroll.set)

    _build_result_panel(tab, state, "advfit_mc", text_height=5, fig_size=(4.0, 2.0))

def _mc_add_current_structure(state: dict):
    try:
        coords, labels, metal, polar = get_current_coords_and_metal(state)
        if len(coords) == 0:
            messagebox.showwarning("Multi-Conformer", "No structure is currently loaded.")
            return

        conf = Conformer(
            name=f"Current structure ({len(coords)} atoms)",
            coords=coords.copy(),
            labels=labels,
        )
        state["advfit_conformers"].append(conf)
        _mc_refresh_conformer_list(state)

    except Exception as e:
        messagebox.showerror("Multi-Conformer", str(e))

def _mc_load_conformer_xyz(state: dict):
    path = filedialog.askopenfilename(
        title="Load conformer structure",
        filetypes=[("Structure files", "*.xyz *.out *.log"), ("All files", "*.*")],
    )
    if not path:
        return

    try:
        atoms = load_structure(path)

        if not state.get("atom_data"):
            messagebox.showwarning("Multi-Conformer", "Please load the main structure first.")
            return

        metal_abs = np.array([state["x0"], state["y0"], state["z0"]], float)
        raw_coords = np.array([[x, y, z] for _, x, y, z in atoms], dtype=float)
        centered = raw_coords - metal_abs
        labels = [f"{el}{i+1}" for i, (el, *_) in enumerate(atoms)]

        conf = Conformer(
            name=os.path.splitext(os.path.basename(path))[0],
            coords=centered,
            labels=labels,
        )
        state["advfit_conformers"].append(conf)
        _mc_refresh_conformer_list(state)

    except Exception as e:
        messagebox.showerror("Multi-Conformer", str(e))

def _mc_clear_conformers(state: dict):
    state["advfit_conformers"].clear()
    _mc_refresh_conformer_list(state)

def _mc_refresh_conformer_list(state: dict):
    lb = state.get("advfit_mc_listbox")
    lines = []

    for i, conf in enumerate(state["advfit_conformers"]):
        name = conf.name
        if len(name) > 28:
            name = name[:25] + "..."
        lines.append(f"{i+1}. {name}  ({conf.weight:.3f})")

    _fill_listbox(lb, lines)

def _mc_run_threaded(state: dict):
    _run_in_thread(_mc_run, state)

def _mc_run(state: dict):
    try:
        conformers = state["advfit_conformers"]
        if len(conformers) < 2:
            state["root"].after(
                0,
                lambda: messagebox.showwarning(
                    "Multi-Conformer",
                    "At least 2 conformers are required."
                )
            )
            return

        dchi_ax, dchi_rh = get_tensor_values(state)
        obs_pcs = get_exp_pcs(state)
        ids = get_selected_ids(state)

        if not obs_pcs:
            state["root"].after(
                0,
                lambda: messagebox.showwarning(
                    "Multi-Conformer",
                    "No experimental δ_Exp values are available."
                )
            )
            return

        fit_rh = state["advfit_mc_fit_rh_var"].get()

        result = fit_multiconf(
            conformers,
            obs_pcs,
            ids,
            dchi_ax,
            dchi_rh if fit_rh else 0.0,
            fit_weights=state["advfit_mc_fit_weights_var"].get(),
        )

        state["root"].after(0, lambda r=result: _mc_show_result(state, r))

    except Exception:
        _show_traceback_error(state, "Multi-Conformer")

def _mc_show_result(state: dict, result: dict):
    state["advfit_mc_last_result"] = result

    lines = []
    lines.append("Multi-Conformer Fit")
    lines.append("-" * 46)
    lines.append(f"Δχ_ax = {result['dchi_ax']:+.4f} × 10^-32 m^3")
    lines.append(f"Δχ_rh = {result['dchi_rh']:+.4f} × 10^-32 m^3")
    lines.append(f"RMSD  = {result['rmsd']:.4f} ppm   (N={result['n']})")
    lines.append("")
    lines.append("Conformer weights:")
    for i, (conf, w) in enumerate(zip(state["advfit_conformers"], result["weights"])):
        lines.append(f"  {i+1}. {conf.name}: w={w:.4f}")

    lines.append("")
    lines.append("Ref\tδ_exp\tδ_pred\tResidual")
    for rid, exp, pred, res in result["per_point"]:
        lines.append(f"{rid}\t{exp:+.3f}\t{pred:+.3f}\t{res:+.3f}")

    _set_textbox_content(state["advfit_mc_result_text"], "\n".join(lines))

    _mc_refresh_conformer_list(state)
    _mc_draw_result_plot(state, result)

def _mc_apply_to_main(state: dict):
    result = state.get("advfit_mc_last_result")
    if not result:
        messagebox.showwarning(
            "Multi-Conformer",
            "No fit result is available to apply.",
        )
        return

    try:
        _apply_tensor_to_main(
            state,
            dchi_ax=result["dchi_ax"],
            dchi_rh=result["dchi_rh"],
        )
        messagebox.showinfo(
            "Multi-Conformer",
            "Applied fitted tensor values to the main app.",
        )
    except Exception as e:
        messagebox.showerror("Multi-Conformer", str(e))

def _mc_draw_result_plot(state: dict, result: dict):
    fig = state["advfit_mc_fig"]
    fig.clear()

    if not result["per_point"]:
        state["advfit_mc_canvas"].draw_idle()
        return

    exp_v = np.array([p[1] for p in result["per_point"]], dtype=float)
    pred_v = np.array([p[2] for p in result["per_point"]], dtype=float)
    res_v = np.array([p[3] for p in result["per_point"]], dtype=float)

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    _draw_identity_scatter(ax1, exp_v, pred_v, "Correlation")

    ax2.bar(range(len(res_v)), res_v, alpha=0.8)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_title("Residuals", fontsize=9)
    ax2.set_ylabel("pred - exp", fontsize=8)
    ax2.tick_params(labelsize=7)

    labels = [f"C{i+1}" for i in range(len(result["weights"]))]
    ax3.bar(labels, result["weights"], alpha=0.85)
    ax3.set_title("Weights", fontsize=9)
    ax3.set_ylabel("Weight", fontsize=8)
    ax3.tick_params(labelsize=7)

    fig.tight_layout()
    state["advfit_mc_canvas"].draw_idle()

# ============================================================================
# Joint PCS + RDC tab
# ============================================================================
def _build_joint_pcs_rdc_tab(state: dict, nb: ttk.Notebook):
    tab = ttk.Frame(nb)
    nb.add(tab, text="Joint PCS+RDC")

    state["advfit_rdc_rows"] = []

    top = ttk.Frame(tab, padding=(6, 6, 6, 2))
    top.pack(fill="x")

    file_row = ttk.Frame(top)
    file_row.pack(fill="x")

    ttk.Button(
        file_row,
        text="Load RDC CSV",
        command=lambda: _rdc_load_csv(state),
        width=12,
    ).pack(side="left")

    ttk.Button(
        file_row,
        text="Export Template",
        command=_export_rdc_template,
        width=14,
    ).pack(side="left", padx=(6, 0))

    state["advfit_rdc_file_label"] = ttk.Label(
        file_row,
        text="No file loaded",
        foreground="gray",
    )
    state["advfit_rdc_file_label"].pack(side="left", padx=8)

    ttk.Label(
        top,
        text="RDC atom labels must match the current structure labels.",
    ).pack(anchor="w", pady=(3, 3))

    param_row = ttk.Frame(top)
    param_row.pack(fill="x", pady=(2, 2))

    ttk.Label(param_row, text="B0 [T]").pack(side="left")
    state["advfit_rdc_B0_var"] = tk.StringVar(value="14.1")
    ttk.Entry(param_row, textvariable=state["advfit_rdc_B0_var"], width=6).pack(side="left", padx=(2, 8))

    ttk.Label(param_row, text="T [K]").pack(side="left")
    state["advfit_rdc_T_var"] = tk.StringVar(value="298")
    ttk.Entry(param_row, textvariable=state["advfit_rdc_T_var"], width=6).pack(side="left", padx=(2, 8))

    ttk.Label(param_row, text="RDC weight").pack(side="left")
    state["advfit_rdc_weight_var"] = tk.StringVar(value="1.0")
    ttk.Entry(param_row, textvariable=state["advfit_rdc_weight_var"], width=6).pack(side="left", padx=(2, 8))

    ttk.Button(
        param_row,
        text="Apply to Main",
        command=lambda: _rdc_apply_to_main(state),
        width=12,
    ).pack(side="right", padx=(4, 0))

    ttk.Button(
        param_row,
        text="Run Fit",
        command=lambda: _rdc_run_threaded(state),
        width=10,
    ).pack(side="right")

    _build_result_panel(tab, state, "advfit_rdc", text_height=6, fig_size=(4.0, 2.0))

def _rdc_load_csv(state: dict):
    path = filedialog.askopenfilename(
        title="Load RDC data",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
    )
    if not path:
        return

    try:
        import pandas as pd

        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        rows = []
        for _, row in df.iterrows():
            rows.append(
                {
                    "atom1": str(row.get("atom1", row.get("h", ""))),
                    "atom2": str(row.get("atom2", row.get("n", ""))),
                    "rdc_exp": float(row.get("rdc_exp", row.get("d", 0))),
                    "rdc_err": float(row.get("rdc_err", 0.5)),
                    "nuc1": str(row.get("nuc1", "H")),
                    "nuc2": str(row.get("nuc2", "N")),
                }
            )

        state["advfit_rdc_rows"] = rows
        state["advfit_rdc_file_label"].config(
            text=f"{os.path.basename(path)} ({len(rows)} bonds)",
            foreground="green",
        )

    except Exception as e:
        messagebox.showerror("Joint PCS+RDC", str(e))

def _rdc_run_threaded(state: dict):
    _run_in_thread(_rdc_run, state)

def _rdc_run(state: dict):
    try:
        coords, labels, metal, polar = get_current_coords_and_metal(state)
        dchi_ax, dchi_rh = get_tensor_values(state)
        obs_pcs = get_exp_pcs(state)
        ids = get_selected_ids(state)
        rdc_rows = state.get("advfit_rdc_rows", [])

        if not rdc_rows:
            state["root"].after(
                0,
                lambda: messagebox.showwarning(
                    "Joint PCS+RDC",
                    "Please load an RDC CSV file first.",
                ),
            )
            return

        if not obs_pcs:
            state["root"].after(
                0,
                lambda: messagebox.showwarning(
                    "Joint PCS+RDC",
                    "No experimental δ_Exp values are available.",
                ),
            )
            return

        B0 = float(state["advfit_rdc_B0_var"].get())
        T = float(state["advfit_rdc_T_var"].get())
        w_rdc = float(state["advfit_rdc_weight_var"].get())

        result = fit_joint_pcs_rdc(
            coords=coords,
            labels=labels,
            obs_pcs=obs_pcs,
            ids=ids,
            rdc_rows=rdc_rows,
            dchi_ax0=dchi_ax,
            dchi_rh0=dchi_rh,
            B0=B0,
            T=T,
            w_rdc=w_rdc,
        )

        state["root"].after(0, lambda r=result: _rdc_show_result(state, r))

    except Exception:
        _show_traceback_error(state, "Joint PCS+RDC")

def _rdc_show_result(state: dict, result: dict):
    state["advfit_rdc_last_result"] = result

    lines = []
    lines.append("Joint PCS+RDC Fit")
    lines.append("-" * 46)
    lines.append(f"Δχ_ax    = {result['dchi_ax']:+.4f} × 10^-32 m^3")
    lines.append(f"Δχ_rh    = {result['dchi_rh']:+.4f} × 10^-32 m^3")
    lines.append(f"PCS RMSD = {result['rmsd_pcs']:.4f} ppm")
    lines.append(f"B0 = {result['B0']:.2f} T,   T = {result['T']:.1f} K")
    lines.append("")
    lines.append("PCS results:")
    lines.append("Ref\tδ_exp\tδ_pred\tResidual")
    for rid, exp, pred, res in result["per_point_pcs"]:
        lines.append(f"{rid}\t{exp:+.3f}\t{pred:+.3f}\t{res:+.3f}")

    if result["rdc_results"]:
        lines.append("")
        lines.append("RDC results:")
        lines.append("Pred [Hz]\tExp [Hz]\tResidual")
        for row in result["rdc_results"]:
            lines.append(f"{row['rdc_pred']:+.2f}\t{row['rdc_exp']:+.2f}\t{row['residual']:+.2f}")

    _set_textbox_content(state["advfit_rdc_result_text"], "\n".join(lines))
    _rdc_draw_result_plot(state, result)

def _rdc_apply_to_main(state: dict):
    result = state.get("advfit_rdc_last_result")
    if not result:
        messagebox.showwarning(
            "Joint PCS+RDC",
            "No fit result is available to apply.",
        )
        return

    try:
        _apply_tensor_to_main(
            state,
            dchi_ax=result["dchi_ax"],
            dchi_rh=result["dchi_rh"],
        )
        messagebox.showinfo(
            "Joint PCS+RDC",
            "Applied fitted tensor values to the main app.",
        )
    except Exception as e:
        messagebox.showerror("Joint PCS+RDC", str(e))

def _rdc_draw_result_plot(state: dict, result: dict):
    fig = state["advfit_rdc_fig"]
    fig.clear()

    pcs_points = result["per_point_pcs"]
    rdc_points = result["rdc_results"]

    if not pcs_points:
        state["advfit_rdc_canvas"].draw_idle()
        return

    ncols = 3 if rdc_points else 2
    gs = gridspec.GridSpec(1, ncols, figure=fig, wspace=0.42)

    exp_pcs = np.array([p[1] for p in pcs_points], dtype=float)
    pred_pcs = np.array([p[2] for p in pcs_points], dtype=float)
    res_pcs = np.array([p[3] for p in pcs_points], dtype=float)

    ax1 = fig.add_subplot(gs[0, 0])
    _draw_identity_scatter(ax1, exp_pcs, pred_pcs, "PCS Corr.", xlabel="PCS exp.", ylabel="PCS pred.")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(len(res_pcs)), res_pcs, alpha=0.8)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_title("PCS Residuals", fontsize=9)
    ax2.set_ylabel("pred - exp", fontsize=8)
    ax2.tick_params(labelsize=7)

    if rdc_points:
        exp_rdc = np.array([r["rdc_exp"] for r in rdc_points], dtype=float)
        pred_rdc = np.array([r["rdc_pred"] for r in rdc_points], dtype=float)

        ax3 = fig.add_subplot(gs[0, 2])
        _draw_identity_scatter(ax3, exp_rdc, pred_rdc, "RDC Corr.", xlabel="RDC exp.", ylabel="RDC pred.")

    fig.tight_layout()
    state["advfit_rdc_canvas"].draw_idle()

# ============================================================================
# PRE tab
# ============================================================================
def _build_pre_tab(state: dict, nb: ttk.Notebook):
    tab = ttk.Frame(nb)
    nb.add(tab, text="PRE")

    state["advfit_pre_obs"] = {}

    top = ttk.Frame(tab, padding=(6, 6, 6, 2))
    top.pack(fill="x")

    file_row = ttk.Frame(top)
    file_row.pack(fill="x")

    ttk.Button(
        file_row,
        text="Load PRE CSV",
        command=lambda: _pre_load_csv(state),
        width=12,
    ).pack(side="left")

    ttk.Button(
        file_row,
        text="Export Template",
        command=_export_pre_template,
        width=14,
    ).pack(side="left", padx=(6, 0))

    state["advfit_pre_file_label"] = ttk.Label(
        file_row,
        text="No file loaded",
        foreground="gray",
    )
    state["advfit_pre_file_label"].pack(side="left", padx=8)

    ttk.Label(
        top,
        text="PRE input should contain Ref ID and Gamma2 (R2_para).",
    ).pack(anchor="w", pady=(3, 3))

    param_row = ttk.Frame(top)
    param_row.pack(fill="x", pady=(2, 2))

    ttk.Label(param_row, text="B0 [T]").pack(side="left")
    state["advfit_pre_B0_var"] = tk.StringVar(value="14.1")
    ttk.Entry(param_row, textvariable=state["advfit_pre_B0_var"], width=6).pack(side="left", padx=(2, 8))

    ttk.Label(param_row, text="μeff [μB]").pack(side="left")
    state["advfit_pre_mu_var"] = tk.StringVar(value="3.68")
    ttk.Entry(param_row, textvariable=state["advfit_pre_mu_var"], width=6).pack(side="left", padx=(2, 8))

    ttk.Label(param_row, text="Ln").pack(side="left")
    state["advfit_pre_ln_var"] = tk.StringVar(value="Nd")
    ln_box = ttk.Combobox(
        param_row,
        textvariable=state["advfit_pre_ln_var"],
        values=list(LANTHANIDE_DB.keys()),
        width=5,
        state="readonly",
    )
    ln_box.pack(side="left", padx=(2, 8))
    ln_box.bind(
        "<<ComboboxSelected>>",
        lambda e: state["advfit_pre_mu_var"].set(
            str(LANTHANIDE_DB[state["advfit_pre_ln_var"].get()]["mu_eff"])
        ),
    )

    ttk.Button(
        param_row,
        text="Run Fit",
        command=lambda: _pre_run_threaded(state),
        width=10,
    ).pack(side="right")

    _build_result_panel(tab, state, "advfit_pre", text_height=5, fig_size=(4.0, 2.0))

def _pre_load_csv(state: dict):
    path = filedialog.askopenfilename(
        title="Load PRE data",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
    )
    if not path:
        return

    try:
        import pandas as pd

        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        obs = {}

        ref_col = next((c for c in ["ref", "ref_id", "id"] if c in df.columns), None)
        r2_col = next((c for c in ["r2_para", "r2", "gamma2", "g2"] if c in df.columns), None)
        label_col = next((c for c in ["atom_label", "label", "atom"] if c in df.columns), None)

        id_col = ref_col or label_col
        if id_col and r2_col:
            for _, row in df.iterrows():
                try:
                    rid = int(row[id_col])
                    obs[rid] = float(row[r2_col])
                except Exception:
                    pass

        state["advfit_pre_obs"] = obs
        state["advfit_pre_file_label"].config(
            text=f"{os.path.basename(path)} ({len(obs)} values)",
            foreground="green",
        )

    except Exception as e:
        messagebox.showerror("PRE", str(e))

def _pre_run_threaded(state: dict):
    _run_in_thread(_pre_run, state)

def _pre_run(state: dict):
    try:
        coords, labels, metal, polar = get_current_coords_and_metal(state)
        ids = get_selected_ids(state)
        obs = state.get("advfit_pre_obs", {})

        if not obs:
            state["root"].after(
                0,
                lambda: messagebox.showwarning(
                    "PRE",
                    "Please load a PRE CSV file first.",
                ),
            )
            return

        mu_eff = float(state["advfit_pre_mu_var"].get())
        B0 = float(state["advfit_pre_B0_var"].get())

        r_ang = np.linalg.norm(coords, axis=1)

        result = fit_pre(
            r_ang=r_ang,
            obs_gamma2=obs,
            ids=ids,
            mu_eff=mu_eff,
            B0=B0,
        )

        state["root"].after(0, lambda r=result: _pre_show_result(state, r))

    except Exception:
        _show_traceback_error(state, "PRE")

def _pre_show_result(state: dict, result: dict):
    lines = []
    lines.append("PRE Fit (Solomon-Bloembergen)")
    lines.append("-" * 46)
    lines.append(f"tau_c   = {result['tau_c'] * 1e9:.4f} ns")
    lines.append(f"mu_eff  = {result['mu_eff']:.2f} μB")
    lines.append(f"omega_H = {result['omega_H']:.3e} rad/s")
    lines.append(f"RMSD    = {result['rmsd']:.3e} s^-1")
    lines.append("")
    lines.append("Ref\tr [Å]\tGamma2_exp\tGamma2_pred")
    for rid, exp, pred, r in result["per_point"]:
        lines.append(f"{rid}\t{r:.2f}\t{exp:.2e}\t{pred:.2e}")

    _set_textbox_content(state["advfit_pre_result_text"], "\n".join(lines))
    _pre_draw_result_plot(state, result)

def _pre_draw_result_plot(state: dict, result: dict):
    fig = state["advfit_pre_fig"]
    fig.clear()

    tau_c = result["tau_c"]
    mu_eff = result["mu_eff"]
    omega_H = result["omega_H"]

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.42)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    r_curve = np.linspace(3.0, 35.0, 300)
    g2_curve = pre_gamma2(r_curve, mu_eff, tau_c, omega_H)

    ax1.plot(r_curve, g2_curve, lw=1.5, ls="--")
    pts = result["per_point"]
    if pts:
        r_obs = np.array([p[3] for p in pts], dtype=float)
        g2_exp = np.array([p[1] for p in pts], dtype=float)
        g2_pred = np.array([p[2] for p in pts], dtype=float)
        ax1.scatter(r_obs, g2_exp, s=24, alpha=0.85)
        ax1.scatter(r_obs, g2_pred, s=18, marker="^", alpha=0.85)

    ax1.set_yscale("log")
    ax1.set_xlabel("r [Å]", fontsize=8)
    ax1.set_ylabel("Gamma2 [s^-1]", fontsize=8)
    ax1.set_title("PRE Profile", fontsize=9)
    ax1.tick_params(labelsize=7)

    tau_range = np.logspace(-11, -7, 250)
    g2_tau = pre_gamma2(np.full(250, 10.0), mu_eff, tau_range, omega_H)
    ax2.semilogx(tau_range * 1e9, g2_tau, lw=1.5)
    ax2.axvline(tau_c * 1e9, ls="--", lw=1.2)
    ax2.set_xlabel("tau_c [ns]", fontsize=8)
    ax2.set_ylabel("Gamma2 (r=10 Å)", fontsize=8)
    ax2.set_title("tau_c Dependence", fontsize=9)
    ax2.tick_params(labelsize=7)

    fig.tight_layout()
    state["advfit_pre_canvas"].draw_idle()

# ============================================================================
# Multi-Lanthanide tab
# ============================================================================
def _build_multilanthanide_tab(state: dict, nb: ttk.Notebook):
    tab = ttk.Frame(nb)
    nb.add(tab, text="Multi-Lanthanide")

    state["advfit_ml_datasets"] = []

    top = ttk.Frame(tab, padding=(6, 6, 6, 2))
    top.pack(fill="x")

    row1 = ttk.Frame(top)
    row1.pack(fill="x")

    ttk.Label(row1, text="Ln").pack(side="left")
    state["advfit_ml_ln_var"] = tk.StringVar(value="Nd")
    ttk.Combobox(
        row1,
        textvariable=state["advfit_ml_ln_var"],
        values=list(LANTHANIDE_DB.keys()),
        width=5,
        state="readonly",
    ).pack(side="left", padx=(2, 8))

    ttk.Button(
        row1,
        text="Add CSV",
        command=lambda: _ml_add_csv(state),
        width=10,
    ).pack(side="left")

    ttk.Button(
        row1,
        text="Use Current δ_Exp",
        command=lambda: _ml_add_current_dataset(state),
        width=16,
    ).pack(side="left", padx=4)

    ttk.Button(
        row1,
        text="Clear",
        command=lambda: _ml_clear_datasets(state),
        width=8,
    ).pack(side="left", padx=4)

    ttk.Button(
        row1,
        text="Run Fit",
        command=lambda: _ml_run_threaded(state),
        width=10,
    ).pack(side="right")

    ttk.Label(
        top,
        text="Each dataset should contain Ref ID and experimental PCS values.",
    ).pack(anchor="w", pady=(3, 3))

    list_box = ttk.LabelFrame(top, text="Datasets", padding=4)
    list_box.pack(fill="x", pady=(2, 2))

    state["advfit_ml_listbox"] = tk.Listbox(
        list_box,
        height=3,
        exportselection=False,
    )
    state["advfit_ml_listbox"].pack(side="left", fill="x", expand=True)

    list_scroll = ttk.Scrollbar(
        list_box,
        orient="vertical",
        command=state["advfit_ml_listbox"].yview,
    )
    list_scroll.pack(side="left", fill="y")
    state["advfit_ml_listbox"].configure(yscrollcommand=list_scroll.set)

    _build_result_panel(tab, state, "advfit_ml", text_height=5, fig_size=(4.0, 2.0))

def _ml_add_csv(state: dict):
    path = filedialog.askopenfilename(
        title="Load PCS dataset",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
    )
    if not path:
        return

    try:
        import pandas as pd

        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        obs = {}
        ref_col = next((c for c in ["ref", "ref_id", "id"] if c in df.columns), None)
        pcs_col = next((c for c in ["pcs_exp", "delta_exp", "d_exp", "ppm"] if c in df.columns), None)

        if ref_col and pcs_col:
            for _, row in df.iterrows():
                try:
                    obs[int(row[ref_col])] = float(row[pcs_col])
                except Exception:
                    pass

        ln = state["advfit_ml_ln_var"].get()
        state["advfit_ml_datasets"].append(
            {
                "name": os.path.basename(path),
                "ln": ln,
                "obs": obs,
            }
        )
        _ml_refresh_dataset_list(state)

    except Exception as e:
        messagebox.showerror("Multi-Lanthanide", str(e))

def _ml_add_current_dataset(state: dict):
    obs = dict(get_exp_pcs(state))
    if not obs:
        messagebox.showwarning(
            "Multi-Lanthanide",
            "No experimental δ_Exp values are available.",
        )
        return

    ln = state["advfit_ml_ln_var"].get()
    state["advfit_ml_datasets"].append(
        {
            "name": f"Current δ_Exp ({ln})",
            "ln": ln,
            "obs": obs,
        }
    )
    _ml_refresh_dataset_list(state)

def _ml_clear_datasets(state: dict):
    state["advfit_ml_datasets"].clear()
    _ml_refresh_dataset_list(state)

def _ml_refresh_dataset_list(state: dict):
    lb = state.get("advfit_ml_listbox")
    lines = []

    for i, ds in enumerate(state["advfit_ml_datasets"]):
        name = ds["name"]
        if len(name) > 28:
            name = name[:25] + "..."
        lines.append(f"{i+1}. [{ds['ln']}] {name} ({len(ds['obs'])})")

    _fill_listbox(lb, lines)

def _ml_run_threaded(state: dict):
    _run_in_thread(_ml_run, state)

def _ml_run(state: dict):
    try:
        datasets = state["advfit_ml_datasets"]
        if len(datasets) < 2:
            state["root"].after(
                0,
                lambda: messagebox.showwarning(
                    "Multi-Lanthanide",
                    "At least 2 datasets are required.",
                ),
            )
            return

        coords, labels, metal, polar = get_current_coords_and_metal(state)
        ids = get_selected_ids(state)

        ln_list = [ds["ln"] for ds in datasets]
        fit_datasets = [{"obs": ds["obs"]} for ds in datasets]

        result = fit_multilanthanid(
            coords=coords,
            ids=ids,
            datasets=fit_datasets,
            lanthanides=ln_list,
        )

        state["root"].after(0, lambda r=result: _ml_show_result(state, r))

    except Exception:
        _show_traceback_error(state, "Multi-Lanthanide")

def _ml_show_result(state: dict, result: dict):
    lines = []
    lines.append("Multi-Lanthanide Fit")
    lines.append("-" * 46)

    for res in result["results"]:
        lines.append("")
        lines.append(f"[{res['lanthanide']}]")
        lines.append(f"Δχ_ax = {res['dchi_ax']:+.4f} × 10^-32 m^3")
        lines.append(f"Δχ_rh = {res['dchi_rh']:+.4f} × 10^-32 m^3")
        lines.append(f"RMSD  = {res['rmsd']:.4f} ppm   (N={res['n']})")
        lines.append("Ref\tδ_exp\tδ_pred\tResidual")
        for rid, exp, pred, resid in res["per_point"]:
            lines.append(f"{rid}\t{exp:+.3f}\t{pred:+.3f}\t{resid:+.3f}")

    _set_textbox_content(state["advfit_ml_result_text"], "\n".join(lines))
    _ml_draw_result_plot(state, result)

def _ml_draw_result_plot(state: dict, result: dict):
    fig = state["advfit_ml_fig"]
    fig.clear()

    results = result["results"]
    if not results:
        state["advfit_ml_canvas"].draw_idle()
        return

    ncols = len(results)
    gs = gridspec.GridSpec(1, ncols, figure=fig, wspace=0.4)

    for i, res in enumerate(results):
        if not res["per_point"]:
            continue

        exp_v = np.array([p[1] for p in res["per_point"]], dtype=float)
        pred_v = np.array([p[2] for p in res["per_point"]], dtype=float)

        ax = fig.add_subplot(gs[0, i])
        _draw_identity_scatter(ax, exp_v, pred_v, f"{res['lanthanide']}\nRMSD={res['rmsd']:.3f}")

    fig.tight_layout()
    state["advfit_ml_canvas"].draw_idle()