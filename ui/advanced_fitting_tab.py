# ui/advanced_fitting_tab.py

from ui.widget_style import ui_text_color, ui_label_style
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
from logic.pcs_document import (
    active_document,
    active_document_view,
    apply_worker_document_updates,
    capture_worker_identity,
    worker_completion_matches,
    worker_document_state,
)
from ui.style import get_app_fonts
from ui.icons import TAB_ICON_SIZE, TOOLBAR_ICON_SIZE, BUTTON_ICON_SIZE

def _icon(state, name, size=BUTTON_ICON_SIZE):
    manager = state.get("icons") if isinstance(state, dict) else None
    if manager is None:
        return ""
    try:
        return manager.get(name, size=size)
    except Exception:
        return ""


# ============================================================================
# Public builder
# ============================================================================
def build_advanced_fitting_tab(state: dict, parent_notebook: ttk.Notebook):
    advanced_frame = ttk.Frame(parent_notebook)
    parent_notebook.add(advanced_frame, text="Advanced Fitting", image=_icon(state, "run_fit", TAB_ICON_SIZE), compound=tk.LEFT)

    outer = ttk.Frame(advanced_frame, padding=4)
    outer.pack(fill="both", expand=True)

    sub_nb = ttk.Notebook(outer)
    sub_nb.pack(fill="both", expand=True)

    _build_multiconformer_tab(state, sub_nb)
    _build_joint_pcs_rdc_tab(state, sub_nb)
    _build_pre_tab(state, sub_nb)
    _build_multilanthanide_tab(state, sub_nb)

    for key in (
        "advfit_mc_running",
        "advfit_rdc_running",
        "advfit_pre_running",
        "advfit_ml_running",
    ):
        state.setdefault(key, False)
    state["advanced_fitting_nb"] = sub_nb
    state["advanced_fitting_callbacks"] = advanced_fitting_callbacks(state)
    return advanced_frame


def advanced_fitting_callbacks(state: dict):
    """Return thin adapters to the existing Advanced Fitting UI callbacks."""
    callbacks = {
        "analysis.advanced.multiconformer.input.use_current": (
            lambda: _mc_add_current_structure(state)
        ),
        "analysis.advanced.multiconformer.input.load_xyz": (
            lambda: _mc_load_conformer_xyz(state)
        ),
        "analysis.advanced.multiconformer.input.clear": (
            lambda: _mc_clear_conformers(state)
        ),
        "analysis.advanced.multiconformer.run": lambda: _mc_run_threaded(state),
        "analysis.advanced.multiconformer.results.apply": (
            lambda: _mc_apply_to_main(state)
        ),
        "analysis.advanced.rdc.input.load_csv": lambda: _rdc_load_csv(state),
        "analysis.advanced.rdc.run": lambda: _rdc_run_threaded(state),
        "analysis.advanced.rdc.results.apply": lambda: _rdc_apply_to_main(state),
        "analysis.advanced.rdc.output.export_template": _export_rdc_template,
        "analysis.advanced.pre.input.load_csv": lambda: _pre_load_csv(state),
        "analysis.advanced.pre.run": lambda: _pre_run_threaded(state),
        "analysis.advanced.pre.output.export_template": _export_pre_template,
        "analysis.advanced.multilanthanide.input.use_current": (
            lambda: _ml_add_current_dataset(state)
        ),
        "analysis.advanced.multilanthanide.input.add_csv": (
            lambda: _ml_add_csv(state)
        ),
        "analysis.advanced.multilanthanide.input.clear": (
            lambda: _ml_clear_datasets(state)
        ),
        "analysis.advanced.multilanthanide.run": lambda: _ml_run_threaded(state),
    }
    workspace = state.get("mc_workspace")
    if workspace is not None:
        callbacks.update(workspace.callbacks())
    joint_workspace = state.get("rdc_workspace")
    if joint_workspace is not None:
        callbacks.update(joint_workspace.callbacks())
    pre_workspace = state.get("pre_workspace")
    if pre_workspace is not None:
        callbacks.update(pre_workspace.callbacks())
    ml_workspace = state.get("ml_workspace")
    if ml_workspace is not None:
        callbacks.update(ml_workspace.callbacks())
    return callbacks

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
        f.write("ref_id,gamma2,gamma2_err,nucleus\n")
        f.write("12,145.3,7.3,1H\n")

    messagebox.showinfo("PRE Template", f"Template saved:\n{path}")

def _refresh_fitting_commands(state: dict):
    refresh = state.get("refresh_fitting_commands")
    if callable(refresh):
        refresh()


def _set_running(state: dict, key: str, running: bool):
    state[key] = bool(running)
    _refresh_fitting_commands(state)


def _run_in_thread(fn, state: dict, busy_key: str):
    if state.get(busy_key, False):
        return
    state[f"{busy_key}_worker_identity"] = capture_worker_identity(state)
    worker_state = worker_document_state(state)
    worker_state[f"{busy_key}_worker_identity"] = state[
        f"{busy_key}_worker_identity"
    ]
    _set_running(state, busy_key, True)

    def _target():
        try:
            fn(worker_state)
        finally:
            state["root"].after(0, lambda: _set_running(state, busy_key, False))

    try:
        threading.Thread(target=_target, daemon=True).start()
    except Exception:
        _set_running(state, busy_key, False)
        raise


def _worker_result_is_current(state: dict, busy_key: str) -> bool:
    identity = state.get(f"{busy_key}_worker_identity")
    # Preserve isolated embedding behavior without weakening the integrated
    # app, where every launched worker has an explicit document identity.
    return identity is None or worker_completion_matches(state, identity)


def _set_document_result(state: dict, busy_key: str, key: str, result: dict) -> bool:
    identity = state.get(f"{busy_key}_worker_identity")
    manager = state.get("document_manager")
    origin = manager.document_by_id(identity.document_id) if manager is not None and identity is not None else None
    if origin is not None and origin is not active_document(state):
        apply_worker_document_updates(
            state, identity, {key: result}, domain=f"{busy_key}_result"
        )
        return False
    try:
        active_document_view(state)[key] = result
    except RuntimeError:
        state[key] = result
    else:
        if origin is not None:
            origin.mark_mutated()
            refresh_tabs = state.get("refresh_document_tabs")
            if callable(refresh_tabs):
                refresh_tabs()
    return True

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

def _plot_font_sizes(state: dict | None = None):
    fonts = get_app_fonts(state or {})
    return {
        "title": fonts.get("plot_title", 9),
        "label": fonts.get("plot_label", 8),
        "tick": fonts.get("plot_tick", 7),
        "legend": fonts.get("plot_legend", 7),
        "annotation": fonts.get("plot_annotation", 7),
    }

def _build_result_panel(parent, state: dict, key_prefix: str, text_height: int = 5, fig_size=(4.0, 2.0)):
    body = ttk.Frame(parent, padding=(6, 2, 6, 6))
    body.pack(fill="both", expand=True)

    fonts = get_app_fonts(state)

    summary_frame = ttk.LabelFrame(body, text="Fit Summary", padding=4)
    summary_frame.pack(fill="both", expand=True)

    text_key = f"{key_prefix}_result_text"
    state[text_key] = scrolledtext.ScrolledText(
        summary_frame,
        height=text_height,
        wrap="none",
        font=fonts.get("report", ("Consolas", 9)),
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

def _draw_identity_scatter(ax, x, y, title, xlabel="Exp.", ylabel="Pred.", state: dict | None = None):
    vmin = min(x.min(), y.min()) * 1.1
    vmax = max(x.max(), y.max()) * 1.1
    fs = _plot_font_sizes(state)
    ax.plot([vmin, vmax], [vmin, vmax], "--", color="gray", alpha=0.5)
    ax.scatter(x, y, s=24, alpha=0.85)
    ax.set_xlabel(xlabel, fontsize=fs["label"])
    ax.set_ylabel(ylabel, fontsize=fs["label"])
    ax.set_title(title, fontsize=fs["title"])
    ax.tick_params(labelsize=fs["tick"])

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
    from ui.multiconformer_workspace import MultiConformerWorkspace
    return MultiConformerWorkspace(state, nb)


def _mc_add_current_structure(state):
    return state["mc_workspace"].use_current()


def _mc_load_conformer_xyz(state):
    return state["mc_workspace"].load_xyz()


def _mc_clear_conformers(state):
    return state["mc_workspace"].clear()


def _mc_run_threaded(state):
    return state["mc_workspace"].run()


def _mc_apply_to_main(state):
    return state["mc_workspace"].apply()


def _mc_refresh_conformer_list(state):
    return state["mc_workspace"].refresh()


# ============================================================================
# Joint PCS + RDC tab
# ============================================================================
def _build_joint_pcs_rdc_tab(state: dict, nb: ttk.Notebook):
    from ui.joint_pcs_rdc_workspace import JointPCSRDCWorkspace
    return JointPCSRDCWorkspace(state, nb)


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
        _refresh_fitting_commands(state)

    except Exception as e:
        messagebox.showerror("Joint PCS+RDC", str(e))

def _rdc_run_threaded(state: dict):
    _run_in_thread(_rdc_run, state, "advfit_rdc_running")

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
    if not _worker_result_is_current(state, "advfit_rdc_running"):
        return
    if not _set_document_result(state, "advfit_rdc_running", "advfit_rdc_last_result", result):
        return
    _refresh_fitting_commands(state)

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
    _draw_identity_scatter(ax1, exp_pcs, pred_pcs, "PCS Corr.", xlabel="PCS exp.", ylabel="PCS pred.", state=state)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(len(res_pcs)), res_pcs, alpha=0.8)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_title("PCS Residuals", fontsize=_plot_font_sizes(state)["title"])
    ax2.set_ylabel("pred - exp", fontsize=_plot_font_sizes(state)["label"])
    ax2.tick_params(labelsize=_plot_font_sizes(state)["tick"])

    if rdc_points:
        exp_rdc = np.array([r["rdc_exp"] for r in rdc_points], dtype=float)
        pred_rdc = np.array([r["rdc_pred"] for r in rdc_points], dtype=float)

        ax3 = fig.add_subplot(gs[0, 2])
        _draw_identity_scatter(ax3, exp_rdc, pred_rdc, "RDC Corr.", xlabel="RDC exp.", ylabel="RDC pred.", state=state)

    fig.tight_layout()
    state["advfit_rdc_canvas"].draw_idle()

# ============================================================================
# PRE tab
# ============================================================================
def _build_pre_tab(state: dict, nb: ttk.Notebook):
    from ui.pre_workspace import PREWorkspace
    return PREWorkspace(state, nb)


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
        _refresh_fitting_commands(state)

    except Exception as e:
        messagebox.showerror("PRE", str(e))

def _pre_run_threaded(state: dict):
    _run_in_thread(_pre_run, state, "advfit_pre_running")

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
    if not _worker_result_is_current(state, "advfit_pre_running"):
        return
    if not _set_document_result(state, "advfit_pre_running", "advfit_pre_last_result", result):
        return
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
    ax1.set_xlabel("r [Å]", fontsize=_plot_font_sizes(state)["label"])
    ax1.set_ylabel("Gamma2 [s^-1]", fontsize=_plot_font_sizes(state)["label"])
    ax1.set_title("PRE Profile", fontsize=_plot_font_sizes(state)["title"])
    ax1.tick_params(labelsize=_plot_font_sizes(state)["tick"])

    tau_range = np.logspace(-11, -7, 250)
    g2_tau = pre_gamma2(np.full(250, 10.0), mu_eff, tau_range, omega_H)
    ax2.semilogx(tau_range * 1e9, g2_tau, lw=1.5)
    ax2.axvline(tau_c * 1e9, ls="--", lw=1.2)
    ax2.set_xlabel("tau_c [ns]", fontsize=_plot_font_sizes(state)["label"])
    ax2.set_ylabel("Gamma2 (r=10 Å)", fontsize=_plot_font_sizes(state)["label"])
    ax2.set_title("tau_c Dependence", fontsize=_plot_font_sizes(state)["title"])
    ax2.tick_params(labelsize=_plot_font_sizes(state)["tick"])

    fig.tight_layout()
    state["advfit_pre_canvas"].draw_idle()

# ============================================================================
# Multi-Lanthanide tab
# ============================================================================
def _build_multilanthanide_tab(state: dict, nb: ttk.Notebook):
    from ui.lanthanide_workspace import LanthanideWorkspace
    return LanthanideWorkspace(state, nb)


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
        _refresh_fitting_commands(state)

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
    _refresh_fitting_commands(state)

def _ml_clear_datasets(state: dict):
    state["advfit_ml_datasets"].clear()
    _ml_refresh_dataset_list(state)
    _refresh_fitting_commands(state)

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
    _run_in_thread(_ml_run, state, "advfit_ml_running")

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
    if not _worker_result_is_current(state, "advfit_ml_running"):
        return
    if not _set_document_result(state, "advfit_ml_running", "advfit_ml_last_result", result):
        return
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
