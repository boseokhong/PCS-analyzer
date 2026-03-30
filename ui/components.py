# ui/components.py

import subprocess, sys, os
import tkinter as tk
import tkinter.ttk as ttk
import threading
from ui.style import apply_style
from tkinter import filedialog, simpledialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from logic.command_processor import process_command as _pc
from logic.xyz_loader import load_structure
from logic.func_group_collapse import collapse_methyl_groups, collapse_cf3_groups

from logic.plot_pcs import plot_graph
from logic.plot_cartesian import plot_cartesian_graph
from logic.table_utils import (
    update_molar_value, update_table, on_delta_entry_change, calculate_tensor_components_ui, calculate_tensor_components_ui_ax_rh,
    export_delta_exp_template, import_delta_exp_file, import_delta_exp_from_clipboard, undo_last_delta_import, clear_delta_exp
)
from logic.rotate_align import rotate_coordinates, rotate_euler
from logic.chem_constants import CPK_COLORS
from logic.fitting import (
    populate_fitting_controls, apply_fit_to_views,
    fit_theta_alpha_multi, fit_euler_global, fit_full_tensor, _angles_to_rotation_multi
)
from logic.include_rhombic import build_rh_table_rows
from logic.diagnostic import update_diagnostic_panel

from ui.plot_3d_window import open_3d_plot_window
from ui.nmr_spectrum_window import NMRSpectrumWindow
from logic.nmr_delta_data_manager import push_layers_to_nmr_if_open

from ui.projection_window import open_projection_window
from ui.pcs_plot_window import open_pcs_plot_popup

from ui.advanced_fitting_tab import build_advanced_fitting_tab
from ui.conformer_search import run_conformer_search_gui

def get_cpk_color(atom):
    return CPK_COLORS.get(atom, CPK_COLORS['default'])

def _sep(parent, orient='horizontal', pady=8, fill='x'):
    s = ttk.Separator(parent, orient=orient)
    s.pack(fill=fill, pady=pady)
    return s

def open_nmr_window(state):
    """Open (or focus) the NMR spectrum window and store the handle in state."""
    root = state['root']
    win = state.get('nmr_win')

    if win is not None:
        try:
            if win.winfo_exists():
                win.lift()
                win.focus_force()
                return
        except Exception:
            pass

    # Keep a reference so the NMR window can call table plotting after imports (drawer actions)
    state["plot_cartesian_graph_fn"] = plot_cartesian_graph

    # Pass state + plot fn into the window (drawer needs them)
    win = NMRSpectrumWindow(root, state=state, plot_cartesian_graph_fn=plot_cartesian_graph)
    state['nmr_win'] = win

    def _on_pick_refs(ref_ids: list[int]):
        """Select one or multiple refs in the main Treeview."""
        tree = state.get("tree")
        row_by_id = state.get("row_by_id", {})

        if tree is None or not row_by_id:
            return

        items = [row_by_id[r] for r in ref_ids if r in row_by_id]
        if not items:
            return

        tree.selection_set(items)
        tree.focus(items[0])
        tree.see(items[0])

    win.set_pick_callback(_on_pick_refs)

    def _on_close():
        try:
            win.destroy()
        finally:
            state['nmr_win'] = None

    win.protocol("WM_DELETE_WINDOW", _on_close)

    # Push layers (PCS/OBS/DIA/PARA) based on state flags
    push_layers_to_nmr_if_open(state)

# conformer search helper
def open_conformer_search(state):
    import csv
    import os
    import tempfile

    atom_data = state.get("atom_data_raw") or state.get("atom_data") or []
    atom_ids = state.get("atom_ids_raw") or list(range(1, len(atom_data) + 1))

    if not atom_data or not atom_ids:
        state["messagebox"].showwarning("Conformer Search", "No structure loaded.")
        return

    delta_exp = state.get("delta_exp_values", {}) or {}
    if not delta_exp:
        state["messagebox"].showwarning(
            "Conformer Search",
            "No δ_Exp data found. Please import δ_Exp first."
        )
        return

    # Keep original structure once
    if state.get("atom_data_raw") is None and state.get("atom_data") is not None:
        state["atom_data_raw"] = list(state["atom_data"])

    tmpdir = tempfile.gettempdir()

    # ---- temporary XYZ for conformer_search ----
    # ── 좌표계 주의 ──────────────────────────────────────────────────────────
    # atom_data_raw는 load_structure()가 반환한 절대 좌표 (metal 원점 이동 전).
    # metal_xyz(x0, y0, z0)도 동일 좌표계의 절대 위치.
    # conformer_search 내부 calc_pcs()에서 coords - metal을 직접 계산하므로
    # 이 두 값은 반드시 같은 좌표계(절대)여야 함.
    # ※ atom_data_eff / filter_atoms()의 좌표는 metal 원점 이동 후이므로 절대 사용 금지.
    xyz_path = os.path.join(tmpdir, "pcs_analyzer_conformer_input.xyz")
    with open(xyz_path, "w", encoding="utf-8") as f:
        f.write(f"{len(atom_data)}\n")
        f.write("PCS Analyzer export for conformer search\n")
        for atom, x, y, z in atom_data:
            f.write(f"{atom} {float(x):.8f} {float(y):.8f} {float(z):.8f}\n")

    # ---- temporary PCS CSV (Ref, δ_exp) ----
    pcs_path = os.path.join(tmpdir, "pcs_analyzer_conformer_pcs.csv")
    with open(pcs_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Ref", "δ_exp"])
        for rid in atom_ids:
            if rid in delta_exp:
                w.writerow([rid, float(delta_exp[rid])])

    try:
        dchi_ax = float(state["tensor_entry"].get() or 0.0)
    except Exception:
        dchi_ax = float(state.get("tensor", 0.0) or 0.0)

    try:
        dchi_rh = float(state.get("rh_dchi_rh", 0.0) or 0.0)
    except Exception:
        dchi_rh = 0.0

    metal_xyz = (
        float(state.get("x0", 0.0)),
        float(state.get("y0", 0.0)),
        float(state.get("z0", 0.0)),
    )

    # Best simple fallback:
    # use explicit metal_ref_id if available, otherwise 1
    metal_idx_1b = int(state.get("metal_ref_id", 1))

    # Euler angles:
    # start simple with current UI rotation if present, else 0,0,0
    try:
        euler_deg = (
            float(state["angle_x_var"].get()) if "angle_x_var" in state else 0.0,
            float(state["angle_y_var"].get()) if "angle_y_var" in state else 0.0,
            float(state["angle_z_var"].get()) if "angle_z_var" in state else 0.0,
        )
    except Exception:
        euler_deg = (0.0, 0.0, 0.0)

    def _on_preview_result(payload):
        state["conformer_preview"] = payload
        state["conformer_preview_coords"] = payload["coords_opt"]
        state["conformer_preview_elements"] = payload["elements"]
        state["conformer_preview_report"] = payload["report"]
        state["conformer_applied"] = False

        try:
            box = state.get("conformer_result_box")
            if box is not None:
                box.delete("1.0", "end")
                box.insert("end", payload["report"])
        except Exception:
            pass

        try:
            state["conformer_status_var"].set(
                f"Done! | RMSD={payload['result']['rmsd']:.4f} ppm"
            )
        except Exception:
            pass

        try:
            from ui.plot_3d_window import _draw_3d_plot
            _draw_3d_plot(state)
        except Exception:
            pass

        state["messagebox"].showinfo(
            "Conformer Search",
            "Preview result received.\nReview the result, then click 'Apply Conformer Preview'."
        )

    new_initial_data = {
        "xyz_path": xyz_path,
        "pcs_path": pcs_path,
        "metal_idx_1b": metal_idx_1b,
        "metal_xyz": metal_xyz,
        "dchi_ax": dchi_ax,
        "dchi_rh": dchi_rh,
        "euler_deg": euler_deg,
    }

    # embedded instance가 있으면 새 창 대신 데이터만 refresh
    cs_app = state.get("conformer_search_app")
    if cs_app is not None:
        cs_app.initial_data = new_initial_data
        cs_app._prefill_from_initial_data()
        cs_app._auto_detect_if_ready()
        try:
            state["plots_nb"].select(state["conformer_tab"])
        except Exception:
            pass
        state["conformer_status_var"].set(
            "Update complete — Check Setup tab and run search")
        return

    # fallback: embedded app 없을 때만 Toplevel로 열기
    run_conformer_search_gui(
        master=state["root"],
        on_preview_result=lambda payload: _conformer_preview_callback(state, payload),
        initial_data=new_initial_data,
    )

def _conformer_preview_callback(state, payload):
    """Shared result-receipt callback for both embedded and Toplevel modes."""
    state["conformer_preview"] = payload
    state["conformer_preview_coords"] = payload["coords_opt"]
    state["conformer_preview_elements"] = payload["elements"]
    state["conformer_preview_report"] = payload["report"]
    state["conformer_applied"] = False

    try:
        result = payload["result"]
        # Include candidate rank in the status if available.
        rank_str = ""
        report   = payload.get("report", "")
        import re
        m = re.search(r"Candidate #(\d+)\s*/\s*(\d+)", report)
        if m:
            rank_str = f" [#{m.group(1)}/{m.group(2)}]"
        state["conformer_status_var"].set(
            f"Preview ready{rank_str} | RMSD={result['rmsd']:.4f} ppm"
            " — click '✅ Apply'"
        )
    except Exception:
        pass

    try:
        from ui.plot_3d_window import _draw_3d_plot
        _draw_3d_plot(state)
    except Exception:
        pass

def apply_conformer_preview(state):
    payload = state.get("conformer_preview")
    if not payload:
        state["messagebox"].showwarning("Conformer Preview", "No preview result available.")
        return

    coords = payload.get("coords_opt")
    atom_data_src = state.get("atom_data_raw") or state.get("atom_data") or []

    if coords is None or not atom_data_src:
        state["messagebox"].showerror("Conformer Preview", "Missing preview coordinates.")
        return

    if len(coords) != len(atom_data_src):
        state["messagebox"].showerror(
            "Conformer Preview",
            "Atom count mismatch between current structure and preview result."
        )
        return

    # coords_opt는 conformer_search가 절대 좌표 기준으로 최적화한 결과.
    # atom_data_raw(절대 좌표)에 덮어쓰는 것이므로 좌표계 일치함.
    # atom_data_eff / filter_atoms()는 이후 apply_symavg_to_state()가 재계산.
    new_atom_data = []
    for (atom, _, _, _), (x, y, z) in zip(atom_data_src, coords):
        new_atom_data.append((atom, float(x), float(y), float(z)))

    # Keep raw only once
    if state.get("atom_data_raw") is None and state.get("atom_data") is not None:
        state["atom_data_raw"] = list(state["atom_data"])

    state["atom_data_conformer"] = new_atom_data
    state["atom_data_raw"] = list(new_atom_data)
    state["atom_data"] = list(new_atom_data)
    state["atom_ids_raw"] = list(range(1, len(new_atom_data) + 1))
    state["conformer_applied"] = True
    state["last_conformer_result"] = payload["result"]

    try:
        apply_symavg_to_state(state)
    except Exception:
        pass

    # Put report into result box if available
    try:
        state["conformer_status_var"].set("Preview applied to current structure.")
    except Exception:
        pass

    try:
        box = state.get("conformer_result_box")
        if box is not None:
            box.delete("1.0", "end")
            box.insert("end", payload["report"])
    except Exception:
        pass

    # Refresh main app views
    try:
        state["update_graph"]()
    except Exception:
        pass

    try:
        state["plot_cartesian"](state)
    except Exception:
        try:
            plot_cartesian_graph(state)
        except Exception:
            pass

    try:
        push_layers_to_nmr_if_open(state)
    except Exception:
        pass

    try:
        from ui.plot_3d_window import _draw_3d_plot
        _draw_3d_plot(state)
    except Exception:
        pass

    state["messagebox"].showinfo("Conformer Preview", "Preview coordinates applied.")

def discard_conformer_preview(state):
    state["conformer_preview"] = None
    state["conformer_preview_coords"] = None
    state["conformer_preview_elements"] = None
    state["conformer_preview_report"] = ""
    state["conformer_applied"] = False

    try:
        state["conformer_status_var"].set("Preview discarded.")
    except Exception:
        pass

    try:
        box = state.get("conformer_result_box")
        if box is not None:
            box.delete("1.0", "end")
    except Exception:
        pass

    try:
        from ui.plot_3d_window import _draw_3d_plot
        _draw_3d_plot(state)
    except Exception:
        pass

    state["messagebox"].showinfo("Conformer Preview", "Preview discarded.")

def revert_conformer_to_original(state):
    raw = state.get("atom_data_original")
    if not raw:
        state["messagebox"].showwarning("Conformer Preview", "No original structure stored.")
        return

    state["atom_data_raw"] = list(raw)
    state["atom_data"] = list(raw)
    state["atom_ids_raw"] = list(range(1, len(raw) + 1))
    state["atom_data_conformer"] = None
    state["conformer_applied"] = False

    try:
        apply_symavg_to_state(state)
    except Exception:
        pass

    try:
        state["update_graph"]()
    except Exception:
        pass

    try:
        state["plot_cartesian"](state)
    except Exception:
        try:
            plot_cartesian_graph(state)
        except Exception:
            pass

    try:
        push_layers_to_nmr_if_open(state)
    except Exception:
        pass

    try:
        state["conformer_status_var"].set("Reverted to original structure.")
    except Exception:
        pass

    try:
        from ui.plot_3d_window import _draw_3d_plot
        _draw_3d_plot(state)
    except Exception:
        pass

    state["messagebox"].showinfo("Conformer Preview", "Reverted to original structure.")

# select - table - plot
def _on_tree_select_update_spectrum(state):
    """Highlight selected Ref in the NMR spectrum window (if open)."""
    win = state.get("nmr_win")
    if win is None:
        return
    try:
        if not win.winfo_exists():
            state["nmr_win"] = None
            return
    except Exception:
        return

    tree = state.get("tree")
    sel = tree.selection() if tree is not None else ()
    if not sel:
        return

    try:
        ref_id = int(tree.item(sel[0], "values")[0])
    except Exception:
        return

    # This method will be added to NMRSpectrumWindow
    try:
        win.highlight_ref(ref_id)
    except Exception:
        pass

def _on_tree_select_update_3d(state):
    """Refresh the 3D viewer highlight when the main table selection changes."""
    win = state.get("plot3d_popup")
    if win is None:
        return
    try:
        if not win.winfo_exists():
            state["plot3d_popup"] = None
            return
    except Exception:
        return

    try:
        from ui.plot_3d_window import _draw_3d_plot
        _draw_3d_plot(state)
    except Exception:
        pass

def _on_tree_select_update_pcs(state):
    """Refresh the 2D PCS plot highlight when the main table selection changes."""
    try:
        tensor_text = state['tensor_entry'].get()
        tensor = float(tensor_text) if tensor_text else 1.0
    except Exception:
        tensor = 1.0
    pcs_values = state.get("pcs_values")
    if pcs_values is None:
        try:
            pcs_min = float(state['pcs_min_entry'].get())
            pcs_max = float(state['pcs_max_entry'].get())
            pcs_interval = float(state['pcs_interval_entry'].get())
            pcs_values = np.arange(pcs_min, pcs_max + pcs_interval, pcs_interval)
        except Exception:
            return
    theta_values = state.get("theta_values")
    if theta_values is None:
        theta_values = np.linspace(0, 2 * np.pi, 500)
        state["theta_values"] = theta_values
    try:
        polar_data, _ = state["filter_atoms"](state)
    except Exception:
        polar_data = None
    try:
        plot_graph(state, pcs_values, theta_values, tensor, polar_data=polar_data)
    except Exception:
        pass

# integrated helper
def _on_tree_select_update_views(state):
    _on_tree_select_update_spectrum(state)
    _on_tree_select_update_3d(state)
    _on_tree_select_update_pcs(state)

def build_app():
    state = {}
    # Tk and common modules
    state['tk'] = tk; state['ttk'] = ttk
    state['filedialog'] = filedialog; state['simpledialog'] = simpledialog; state['messagebox'] = messagebox
    state['FigureCanvas'] = FigureCanvasTkAgg; state['NavigationToolbar2Tk'] = NavigationToolbar2Tk
    # handle for the NMR spectrum window
    state['pcs_by_id'] = {}
    state['nmr_win'] = None
    state['pcs_plot_popup'] = None
    state['pcs_figure_popup'] = None
    state['pcs_canvas_popup'] = None
    state['pcs_click_cid'] = None
    state['pcs_popup_click_cid'] = None
    # projection plot window
    state['projection_popup'] = None
    state['projection_figure'] = None
    state['projection_canvas'] = None
    state['projection_mode_var'] = None
    state['projection_r_var'] = None
    state['projection_show_atoms_var'] = None
    state['projection_show_h_var'] =None
    state['projection_click_cid'] = None
    # 3D plot window
    state['plot3d_popup'] = None
    state['plot3d_figure'] = None
    state['plot3d_canvas'] = None
    state['plot3d_color_mode_var'] = None
    state['plot3d_show_labels_var'] = None
    state['plot3d_click_cid'] = None
    # conformer search - preview / apply state
    state['conformer_preview'] = None
    state['conformer_preview_coords'] = None
    state['conformer_preview_elements'] = None
    state['conformer_preview_report'] = ""
    state['conformer_applied'] = False
    state['atom_data_raw'] = None
    state['atom_data_conformer'] = None
    state['atom_data_original'] = None

    # Window size
    root = tk.Tk(); root.title("PCS Analyzer"); root.geometry("1180x915"); state['root'] = root
    apply_style(root, variant="light", accent="green")  # darkmode : variant="dark"

    # Frames
    main_frame = ttk.Frame(root)
    main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    center_frame = ttk.Frame(main_frame)
    center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    state['center_frame'] = center_frame

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side=tk.LEFT, fill=tk.Y)
    state['right_frame'] = right_frame

    state['left_frame'] = None

    # Embedded PCS plot removed.
    # # PCS figure/canvas
    # pcs_figure = plt.figure(figsize=(4,4), dpi=150); state['pcs_figure']=pcs_figure
    # pcs_canvas = FigureCanvasTkAgg(pcs_figure, master=left_frame); pcs_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    # state['pcs_canvas'] = pcs_canvas
    state['pcs_figure'] = None
    state['pcs_canvas'] = None
    state['pcs_click_cid'] = None
    state['pcs_popup_click_cid'] = None

    # Table
    table_frame = ttk.Frame(center_frame); table_frame.pack(side=tk.TOP, fill=tk.X, padx=3, pady=0)
    columns = ('Ref','Atom','X','Y','Z','G_i','δ_PCS','δ_Exp');
    tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=12)
    for col in columns: tree.heading(col, text=col)
    widths = {'Ref':10,'Atom':10,'X':20,'Y':20,'Z':20,'G_i':30,'δ_PCS':20,'δ_Exp':20}
    for c,w in widths.items(): tree.column(c, width=w)
    scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set); scrollbar.pack(side=tk.RIGHT, fill=tk.Y); tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    state['tree']=tree

    table_btns = ttk.Frame(center_frame)
    table_btns.pack(side=tk.TOP, fill=tk.X, padx=3, pady=(4, 6))
    ttk.Button(
        table_btns, text="💾 Export δ_Exp Template",
        command=lambda: export_delta_exp_template(state)
    ).pack(side=tk.LEFT, padx=(0, 6))
    ttk.Button(
        table_btns, text="📂 Import δ_Exp",
        command=lambda: import_delta_exp_file(state, plot_cartesian_graph)
    ).pack(side=tk.LEFT)
    ttk.Button(
        table_btns, text="📝 Paste δ_Exp",
               command=lambda: import_delta_exp_from_clipboard(state, plot_cartesian_graph)).pack(side=tk.LEFT, padx=6)
    ttk.Button(
        table_btns, text="↺ Undo",
               command=lambda: undo_last_delta_import(state, plot_cartesian_graph)).pack(side=tk.LEFT)
    ttk.Button(
        table_btns, text="❌ Clear δ_Exp",
        command=lambda: clear_delta_exp(state, plot_cartesian_graph)
    ).pack(side=tk.LEFT, padx=6)

    _sep(center_frame, orient='horizontal', pady=6, fill='x')

    # --- tab ---
    # Plots notebook
    plots_nb = ttk.Notebook(center_frame)
    plots_nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=3, pady=0)
    state['plots_nb'] = plots_nb

    # -------------------------
    # --- Cartesian tab UI ---
    # -------------------------
    cartesian_tab = ttk.Frame(plots_nb)
    plots_nb.add(cartesian_tab, text="📈 Plot")

    cartesian_figure = plt.Figure(figsize=(4, 3), dpi=100);
    state['cartesian_figure'] = cartesian_figure
    cartesian_canvas = FigureCanvasTkAgg(cartesian_figure, master=cartesian_tab)
    cartesian_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    state['cartesian_canvas'] = cartesian_canvas

    # --------------------------
    # --- Diagnostic tab UI ---
    # --------------------------
    diagtab = ttk.Frame(plots_nb)
    plots_nb.add(diagtab, text="📊 Diagnostic")
    state["diagtab"] = diagtab

    # 1) button row (TOP)
    diag_btnrow = ttk.Frame(diagtab)
    diag_btnrow.pack(fill=tk.X, padx=6, pady=(6, 4))

    ttk.Button(
        diag_btnrow,
        text="🔄 Update Diagnostic",
        command=lambda: update_diagnostic_panel(state)
    ).pack(side=tk.LEFT)

    state["diag_fit_intercept_var"] = tk.BooleanVar(value=False)
    tk.Checkbutton(
        diag_btnrow,
        variable=state["diag_fit_intercept_var"],
        text="Force b = 0 (through origin)",
        bg="#F5F6FA",
        activebackground="#F5F6FA",
        highlightthickness=0,
        relief="flat"
    ).pack(side=tk.LEFT, padx=10)

    # 2) summary box (TOP)
    state["diag_result_box"] = tk.Text(diagtab, font=("Courier", 9), height=6)
    state["diag_result_box"].pack(fill=tk.X, padx=6, pady=(0, 6))

    # 3) figures container (BOTTOM)
    diag_top = ttk.Frame(diagtab)
    diag_top.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

    ## Figure 1: δexp vs Gax (left)
    #fig1 = plt.Figure(figsize=(4, 3), dpi=100)
    #state["diag_fig_linearity"] = fig1
    #cv1 = FigureCanvasTkAgg(fig1, master=diag_top)
    #cv1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    #state["diag_canvas_linearity"] = cv1

    # Figure 2: residual vs phi / Grh (right)
    fig2 = plt.Figure(figsize=(4, 3), dpi=100)
    state["diag_fig_resphi"] = fig2
    cv2 = FigureCanvasTkAgg(fig2, master=diag_top)
    cv2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    state["diag_canvas_resphi"] = cv2

    # --------------------------
    # --- Rhombicity tab UI ----
    # --------------------------
    rhtab = ttk.Frame(plots_nb, width=560)  # Fixed width
    plots_nb.add(rhtab, text="📊 Rhombicity")

    # Prevent the tab frame from resizing to the requested size of its children.
    rhtab.grid_propagate(False)

    # Use grid layout only inside the Rhombicity tab.
    rhtab.rowconfigure(0, weight=0)  # top bar
    rhtab.rowconfigure(1, weight=0)  # z-rotation bar
    rhtab.rowconfigure(2, weight=1)  # table area
    rhtab.columnconfigure(0, weight=1)

    # Top control row
    rh_top = ttk.Frame(rhtab)
    rh_top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)

    # Δχ_rh input (unit: 1e-32 m^3)
    ttk.Label(rh_top, text="Δχ_rh values (E-32 m³):").pack(side="left", padx=(0, 6))

    # Default state value
    state.setdefault("rh_dchi_rh", 0.0)

    # Entry variable
    state["rh_dchi_rh_var"] = tk.StringVar(value=f"{state.get('rh_dchi_rh', 0.0):g}")
    rh_dchi_entry = ttk.Entry(rh_top, textvariable=state["rh_dchi_rh_var"], width=10)
    rh_dchi_entry.pack(side="left")
    state["rh_dchi_entry"] = rh_dchi_entry

    # Rhombicity tab table
    cols = ("Ref", "Atom", "r", "theta(deg)", "phi(deg)",
            "Gi_ax", "Gi_rh", "δ_PCS(ax)", "δ_PCS(ax+rh)", "δ_Exp", "res(ax)", "res(ax+rh)")

    # ---------------------------
    # Z-rotation row (between top and table)
    # ---------------------------
    rh_zrow = ttk.Frame(rhtab)
    rh_zrow.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))

    ttk.Label(rh_zrow, text="Rotate around Z-axis (degrees):").pack(side="left", padx=(0, 6))

    # z state vars
    state.setdefault("angle_z_var", tk.DoubleVar(value=0.0))
    # on_angle_slider / on_angle_entry_commit must support axis 'z'
    azf = ttk.Frame(rh_zrow)
    azf.pack(side="left", fill="x", expand=True)

    angle_z_slider = tk.Scale(
        azf,
        from_=-180, to=180,
        orient=tk.HORIZONTAL,
        variable=state["angle_z_var"],
        resolution=0.1,
        command=lambda v: on_angle_slider(state, 'z', v),
        bg="#F5F6FA",
        activebackground="#F5F6FA",
        highlightthickness=0
    )
    angle_z_slider.pack(side="left", fill="x", expand=True)

    angle_z_entry = tk.Entry(azf, width=6)
    angle_z_entry.pack(side="left", padx=(6, 0))
    angle_z_entry.delete(0, tk.END)
    angle_z_entry.insert(0, f"{float(state['angle_z_var'].get()):.1f}")

    state["angle_z_entry"] = angle_z_entry
    angle_z_entry.bind('<Return>', lambda e: on_angle_entry_commit(state, 'z'))
    angle_z_entry.bind('<FocusOut>', lambda e: on_angle_entry_commit(state, 'z'))

    # ---------------------------
    # Use grid layout for the table frame as well(CHANGED row: 1 -> 2)
    # ---------------------------
    rh_table_frame = ttk.Frame(rhtab)
    rh_table_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))

    rh_tree = ttk.Treeview(rh_table_frame, columns=cols, show="headings", height=18)

    # Vertical and horizontal scrollbars
    rh_ys = ttk.Scrollbar(rh_table_frame, orient="vertical", command=rh_tree.yview)
    rh_xs = ttk.Scrollbar(rh_table_frame, orient="horizontal", command=rh_tree.xview)
    rh_tree.configure(yscrollcommand=rh_ys.set, xscrollcommand=rh_xs.set)

    # Column widths (all with stretch=False)
    widths = (40, 50, 50, 80, 80, 80, 80, 80, 105, 60, 60, 80)
    for c, w in zip(cols, widths):
        rh_tree.heading(c, text=c)
        rh_tree.column(c, width=w, minwidth=40, anchor="center", stretch=False)

    # Grid layout inside the table frame
    rh_table_frame.rowconfigure(0, weight=1)
    rh_table_frame.rowconfigure(1, weight=0)
    rh_table_frame.columnconfigure(0, weight=1)
    rh_table_frame.columnconfigure(1, weight=0)

    rh_tree.grid(row=0, column=0, sticky="nsew")
    rh_ys.grid(row=0, column=1, sticky="ns")
    rh_xs.grid(row=1, column=0, sticky="ew")

    state["rh_tree"] = rh_tree

    def _rh_refresh_table():
        for it in rh_tree.get_children():
            rh_tree.delete(it)
        try:
            rows = build_rh_table_rows(state, filter_atoms)  # filter_atoms는 기존 components.py 함수
        except Exception as e:
            state["messagebox"].showerror("Rhombicity", str(e))
            return
        for row in rows:
            rh_tree.insert("", "end", values=row)
        if hasattr(state["root"], "_stripe_treeview"):
            state["root"]._stripe_treeview(rh_tree)

    state["rh_refresh_table"] = _rh_refresh_table

    def _apply_dchi_rh():
        s = state["rh_dchi_rh_var"].get().strip()
        try:
            v = float(s) if s else 0.0
        except Exception:
            state["messagebox"].showerror("Rhombicity", "Invalid Δχ_rh value.")
            return

        # Δχ_rh 저장 (입력값은 '1e-32 m^3' 스케일)
        state["rh_dchi_rh"] = v

        # Apply = Refresh
        _rh_refresh_table()

    # Apply on Enter
    rh_dchi_entry.bind("<Return>", lambda e: _apply_dchi_rh())

    # Apply btn
    ttk.Button(rh_top, text="🔄 Update", command=_apply_dchi_rh).pack(side="left", padx=(6, 12))

    # Rhombicity ON/OFF toggle checkbox
    state.setdefault("rh_calc_enabled", False)
    state["rh_calc_enabled_var"] = tk.BooleanVar(value=state["rh_calc_enabled"])

    def _on_toggle_rh_calc():
        state["rh_calc_enabled"] = bool(state["rh_calc_enabled_var"].get())
        try:
            _apply_dchi_rh()
        except Exception:
            pass

    ttk.Checkbutton(
        rh_top,
        text="Enable Δχ_rh",
        variable=state["rh_calc_enabled_var"],
        command=_on_toggle_rh_calc
    ).pack(side="left", padx=(10, 0))

    def _calc_chi_tensor_from_ui():
        ph = state.get("chi_placeholder_text")
        if state.get("rh_calc_enabled", False):
            calculate_tensor_components_ui_ax_rh(
                chi_mol_entry=state["chi_mol_entry"],
                molar_value_label=state["molar_value_label"],
                rh_dchi_entry=rh_dchi_entry,
                tensor_xx_label=state["tensor_xx_label"],
                tensor_yy_label=state["tensor_yy_label"],
                tensor_zz_label=state["tensor_zz_label"],
                messagebox=state["messagebox"],
                quiet=False,
                placeholder_text=ph,
            )
        else:
            calculate_tensor_components_ui(
                state["chi_mol_entry"],
                state["molar_value_label"],
                state["tensor_xx_label"],
                state["tensor_yy_label"],
                state["tensor_zz_label"],
                state["messagebox"],
                quiet=False,
                placeholder_text=ph,
            )

    ttk.Button(rh_top, text="🧮 Calc χ(xx,yy,zz)", command=_calc_chi_tensor_from_ui) \
        .pack(side="left", padx=(4, 0))

    # --------------------------
    # --- Fitting tab UI -------
    # --------------------------
    fittab = ttk.Frame(plots_nb)
    plots_nb.add(fittab, text="🔍 Fitting")

    # left(Settings), right(Protons)
    fit_top = ttk.Frame(fittab)
    fit_top.pack(fill=tk.BOTH, expand=True)

    settings_col = ttk.Frame(fit_top)  # left
    settings_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

    protons_col = ttk.Frame(fit_top)  # right
    protons_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Mode selection : inside settings_col
    mode_row = ttk.Frame(settings_col)
    mode_row.pack(fill=tk.X, pady=4)
    state['fit_mode_var'] = tk.StringVar(value='theta_alpha_multi')
    tk.Radiobutton(mode_row, text="[Mode A] Ligand-donor fit",
                   variable=state['fit_mode_var'], value='theta_alpha_multi',
                   command=lambda: switch_fit_mode(state),
                   bg="#F5F6FA",
                   activebackground="#F5F6FA",
                   highlightthickness=0,
                   relief="flat"
                   ).pack(side=tk.LEFT)
    tk.Radiobutton(mode_row, text="[Mode B] Euler fit",
                   variable=state['fit_mode_var'], value='euler_global',
                   command=lambda: switch_fit_mode(state),
                   bg="#F5F6FA",
                   activebackground="#F5F6FA",
                   highlightthickness=0,
                   relief="flat"
                   ).pack(side=tk.LEFT, padx=10)
    tk.Radiobutton(mode_row, text="[Mode C] 8-param fit",
                   variable=state['fit_mode_var'], value='full_tensor',
                   command=lambda: switch_fit_mode(state),
                   bg="#F5F6FA",
                   activebackground="#F5F6FA",
                   highlightthickness=0,
                   relief="flat"
                   ).pack(side=tk.LEFT, padx=10)

    # A/B anchor frame : inside settings_col
    state['fit_anchor'] = ttk.Frame(settings_col)
    state['fit_anchor'].pack(fill=tk.X)

    # --- Mode A (Settings) ---
    frameA = ttk.LabelFrame(settings_col, text="Settings", width=200, height=230)
    state['fit_frameA'] = frameA
    frameA.pack_propagate(False)
    frameA.pack(fill=tk.X, pady=4, before=state['fit_anchor'])

    labelA = ttk.Label(
        frameA,
        text=("Donor atoms list : define the ligand donor atoms and decide "
              "how to establish the axis/vector used in the fitting."),
        justify="left"
    )
    labelA.pack(anchor='w', padx=6, pady=(6, 0), fill="x")

    def _adjust_wrapA(event):
        labelA.configure(wraplength=event.width - 12)  # padding 고려

    frameA.bind("<Configure>", _adjust_wrapA)

    _donor_box = ttk.Frame(frameA)
    _donor_box.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    state['fit_donor_list'] = tk.Listbox(_donor_box, selectmode=tk.EXTENDED,
                                         exportselection=False, height=6)
    state['fit_donor_list'].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    _donor_scroll = ttk.Scrollbar(_donor_box, orient=tk.VERTICAL,
                                  command=state['fit_donor_list'].yview)
    _donor_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    state['fit_donor_list'].configure(yscrollcommand=_donor_scroll.set)

    rowA = ttk.Frame(frameA);
    rowA.pack(fill=tk.X, pady=(0, 6), padx=6)
    ttk.Label(rowA, text="Axis vector mode:").pack(side=tk.LEFT)
    state['axis_mode_var'] = tk.StringVar(value='bisector')
    ttk.Combobox(rowA, textvariable=state['axis_mode_var'],
                 values=['bisector', 'normal', 'pca', 'centroid', 'average', 'first'],
                 width=12, state="readonly").pack(side=tk.LEFT, padx=6)

    # --- Mode B (Settings) ---
    frameB = ttk.LabelFrame(settings_col, text="Settings", width=200, height=230)
    state['fit_frameB'] = frameB
    frameB.pack_propagate(False)

    labelB = ttk.Label(
        frameB,
        text=("No donors required. Rigid-body fitting of the selected set of "
              "protons in the global frame (ax, ay, az)."),
        justify="left"
    )
    labelB.pack(anchor='w', padx=6, pady=6, fill="x")

    def _adjust_wrapB(event):
        labelB.configure(wraplength=event.width - 12)

    frameB.bind("<Configure>", _adjust_wrapB)

    # initiate- hide mode B
    frameB.pack_forget()

    # --- Mode C (Settings) ---
    frameC = ttk.LabelFrame(settings_col, text="Settings", width=200, height=230)
    state['fit_frameC'] = frameC
    frameC.pack_propagate(False)

    labelC = ttk.Label(
        frameC,
        text=("8-parameter fitting of the selected protons.\n"
              "Optimizes Δχ_ax, optional Δχ_rh, metal position (x, y, z; range ±1 Å),\n"
              "and Euler angles (α,β,γ) in the global frame. Molecule stays rigid. \n"
              "Two stages fit: Differential Evolution → Levenberg-Marquardt.\n"
              "Recommended: ≥8 protons with δ_exp, enable 'Global Search (DE)'."),
        justify="left"
    )
    labelC.pack(anchor='w', padx=6, pady=6, fill="x")

    def _adjust_wrapC(event):
        labelC.configure(wraplength=event.width - 12)

    frameC.bind("<Configure>", _adjust_wrapC)
    # initiate- hide mode C
    frameC.pack_forget()

    # --- Protons to fit ---
    state['fit_protons_label'] = ttk.Label(protons_col, text="Protons to fit")
    state['fit_protons_label'].pack(anchor='w', pady=(6, 0))

    _proton_box = ttk.Frame(protons_col)
    _proton_box.pack(fill=tk.BOTH, expand=True)

    state['fit_proton_list'] = tk.Listbox(_proton_box, selectmode=tk.EXTENDED,
                                          exportselection=False, height=12)
    state['fit_proton_list'].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    _proton_scroll = ttk.Scrollbar(_proton_box, orient=tk.VERTICAL,
                                   command=state['fit_proton_list'].yview)
    _proton_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    state['fit_proton_list'].configure(yscrollcommand=_proton_scroll.set)

    opts = ttk.Frame(fittab);
    opts.pack(fill=tk.X, pady=4)
    state['fit_use_visible_var'] = tk.BooleanVar(value=False)
    tk.Checkbutton(opts, text="Use visible atoms as rigid group",
                   variable=state['fit_use_visible_var'],
                   state="disabled",  # WIP!!!!
                   bg="#F5F6FA",
                   activebackground="#F5F6FA",
                   highlightthickness=0,
                   relief="flat"
                   ).pack(side=tk.LEFT)
    state['fit_dchi_var'] = tk.BooleanVar(value=True)
    tk.Checkbutton(opts, text="Fit Δχ_ax",
                   variable=state['fit_dchi_var'],
                   bg="#F5F6FA",
                   activebackground="#F5F6FA",
                   highlightthickness=0,
                   relief="flat"
                   ).pack(side=tk.LEFT, padx=12)

    state['fit_dchi_rh_var'] = tk.BooleanVar(value=False)
    tk.Checkbutton(opts, text="Fit Δχ_rh",
                    variable=state['fit_dchi_rh_var'],
                    bg="#F5F6FA",
                    activebackground="#F5F6FA",
                    highlightthickness=0,
                    relief="flat"
                    ).pack(side=tk.LEFT, padx=12)
    state['fit_global_search_var'] = tk.BooleanVar(value=False)
    tk.Checkbutton(opts, text="Global search (DE)",
                   variable=state['fit_global_search_var'],
                   bg="#F5F6FA",
                   activebackground="#F5F6FA",
                   highlightthickness=0,
                   relief="flat"
                   ).pack(side=tk.LEFT, padx=12)

    btns = ttk.Frame(fittab);
    btns.pack(fill=tk.X, pady=4)
    ttk.Button(btns, text="🔄 Refresh",
               command=lambda: populate_fitting_controls(state)).pack(side=tk.LEFT, padx=3)
    ttk.Button(btns, text="🧮 Run fit",
               command=lambda: run_fit_from_ui(state)).pack(side=tk.LEFT, padx=3)
    ttk.Button(btns, text="⚙️ Apply to plot",
               command=lambda: apply_fit_to_views(state)).pack(side=tk.LEFT, padx=3)
    ttk.Button(btns, text="💾 Export plot",
               command=lambda: export_fit_plot(state)).pack(side=tk.RIGHT, padx=3)

    state['fit_status_var'] = tk.StringVar(value="Ready.")
    fit_status_label = tk.Label(btns,
                                textvariable=state['fit_status_var'],
                                font=("Helvetica", 8),
                                fg="#777777",
                                bg="#F5F6FA"
                                )
    fit_status_label.pack(side=tk.LEFT, padx=(8, 0))

    def _fit_progress_cb(msg):
        state['fit_status_var'].set(msg)
        try:
            state['root'].update_idletasks()
        except Exception:
            pass

    state['fit_progress_cb'] = _fit_progress_cb

    # body container
    fit_body = ttk.Frame(fittab)
    fit_body.pack(fill=tk.BOTH, expand=True, pady=4)

    fit_body.columnconfigure(0, weight=4)
    fit_body.columnconfigure(1, weight=5)
    fit_body.rowconfigure(0, weight=1)

    fit_left = ttk.Frame(fit_body)
    fit_left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

    fit_right = ttk.Frame(fit_body)
    fit_right.grid(row=0, column=1, sticky="nsew")

    state['fit_result_box'] = tk.Text(fit_left, height=10, font=("Courier", 9))
    state['fit_result_box'].pack(fill=tk.BOTH, expand=True)

    fit_fig = Figure(figsize=(6, 3), dpi=50)
    fit_ax = fit_fig.add_subplot(111)
    fit_ax.set_xlabel("δ_Exp")
    fit_ax.set_ylabel("δ_Pred")
    fit_ax.set_title("Fit correlation")

    fit_canvas = FigureCanvasTkAgg(fit_fig, master=fit_right)
    fit_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    state['fit_corr_fig'] = fit_fig
    state['fit_corr_ax'] = fit_ax
    state['fit_corr_canvas'] = fit_canvas

    # -------------------------------
    # --- Advanced Fitting tab UI ---
    # -------------------------------
    build_advanced_fitting_tab(state, plots_nb)

    # -------------------------------
    # --- Conformer search tab UI ---
    # -------------------------------
    cstab = ttk.Frame(plots_nb)
    plots_nb.add(cstab, text="🔍 Conformer Search")
    state["conformer_tab"] = cstab

    # ── 액션 바 ──────────────────────────────────────────────────────────────
    cs_btns = ttk.Frame(cstab)
    cs_btns.pack(fill=tk.X, padx=6, pady=(4, 2))

    ttk.Button(cs_btns, text="🔄 Sync structure & PCS",
               command=lambda: open_conformer_search(state)).pack(side=tk.LEFT, padx=3)
    ttk.Button(cs_btns, text="✅ Apply candidate",
               command=lambda: apply_conformer_preview(state)).pack(side=tk.LEFT, padx=3)
    ttk.Button(cs_btns, text="↺ Revert",
               command=lambda: revert_conformer_to_original(state)).pack(side=tk.LEFT, padx=3)
    ttk.Button(cs_btns, text="🗑 Discard",
               command=lambda: discard_conformer_preview(state)).pack(side=tk.LEFT, padx=3)

    state["conformer_status_var"] = tk.StringVar(
        value="Load XYZ and δ_Exp, then press the '🔄 Sync' button.")
    ttk.Label(cs_btns, textvariable=state["conformer_status_var"],
              foreground="gray").pack(side=tk.LEFT, padx=(10, 0))

    # ── 4개 서브탭 직접 embed ────────────────────────────────────────────────
    cs_embed_frame = ttk.Frame(cstab)
    cs_embed_frame.pack(fill=tk.BOTH, expand=True)

    cs_app = run_conformer_search_gui(
        embed_in=cs_embed_frame,
        on_preview_result=lambda payload: _conformer_preview_callback(state, payload),
    )
    state["conformer_search_app"] = cs_app
    # _result_box는 ScrolledText(disabled) — _show_result()가 직접 씀
    state["conformer_result_box"] = cs_app._result_box

    # Right inputs region
    input_frame = ttk.Frame(right_frame); input_frame.pack(fill=tk.Y, padx=10, pady=10); state['input_frame']=input_frame

    # Δχ_ax
    tf = ttk.Frame(input_frame); tf.pack(pady=3)
    ttk.Label(tf, text="Δχ_ax values (E-32 m³):", font=("default",9,"bold")).pack(side=tk.LEFT)
    tensor_entry = tk.Entry(tf, width=5); tensor_entry.pack(side=tk.LEFT, padx=5); state['tensor_entry']=tensor_entry

    # PCS range
    prf = ttk.Frame(input_frame); prf.pack(pady=3)
    ttk.Label(prf, text="PCS plot range (ppm)", font=("default",9,"bold")).pack(side=tk.TOP, pady=0)
    pef = ttk.Frame(prf); pef.pack(side=tk.TOP, pady=0)
    ttk.Label(pef, text="Min:").pack(side=tk.LEFT); pcs_min_entry = tk.Entry(pef, width=5); pcs_min_entry.pack(side=tk.LEFT, padx=5)
    ttk.Label(pef, text="/").pack(side=tk.LEFT); ttk.Label(pef, text="Max:").pack(side=tk.LEFT); pcs_max_entry = tk.Entry(pef, width=5); pcs_max_entry.pack(side=tk.LEFT, padx=5)
    state['pcs_min_entry']=pcs_min_entry; state['pcs_max_entry']=pcs_max_entry

    # Interval
    pif = ttk.Frame(input_frame); pif.pack(pady=3)
    ttk.Label(pif, text="PCS plot interval (ppm):", font=("default",9,"bold")).pack(side=tk.LEFT)
    pcs_interval_entry = tk.Entry(pif, width=5); pcs_interval_entry.pack(side=tk.LEFT, padx=5); state['pcs_interval_entry']=pcs_interval_entry

    # Toggle 0-90 / 0-180
    prt = ttk.Frame(input_frame); prt.pack(pady=0); ttk.Label(prt, text="Half/Quarter plot toggle", font=("default",9,"bold")).pack(side=tk.LEFT)
    plot_90_var = tk.BooleanVar(value=False); state['plot_90_var']=plot_90_var
    tk.Checkbutton(prt, variable=plot_90_var,
                   command=lambda: state['update_graph'](),
                   bg="#F5F6FA",
                   activebackground="#F5F6FA",
                   highlightthickness=0,
                   relief="flat"
                   ).pack(side=tk.LEFT)

    # Update/Reset
    bf = ttk.Frame(input_frame); bf.pack(pady=3)
    ttk.Button(bf, text="🔄 Update", command=lambda: state['update_graph']()).pack(side=tk.LEFT, padx=2)
    ttk.Button(bf, text="↺ Reset", command=lambda: reset_values(state)).pack(side=tk.LEFT, padx=2)

    # Frame - align middle
    for f in (tf, prf, pif, prt, bf):
        f.pack_configure(anchor="center")

    _sep(input_frame)

    # Molar labels and chi_mol
    state['molar_value_label'] = ttk.Label(input_frame, text="Δχ_mol_ax : N/A m³/mol")
    state['molar_value_label'].pack(pady=0)

    ttk.Label(input_frame, text="χ_mol from Exp. (m³/mol):").pack()

    chi_mol_entry = tk.Entry(input_frame)
    chi_mol_entry.pack()
    state['chi_mol_entry'] = chi_mol_entry

    state['tensor_xx_label'] = ttk.Label(input_frame, text="χ_xx: N/A m³/mol")
    state['tensor_xx_label'].pack()
    state['tensor_yy_label'] = ttk.Label(input_frame, text="χ_yy: N/A m³/mol")
    state['tensor_yy_label'].pack()
    state['tensor_zz_label'] = ttk.Label(input_frame, text="χ_zz: N/A m³/mol")
    state['tensor_zz_label'].pack()

    # ---- Auto-calc on/off ----
    state.setdefault("chi_auto_calc_var", tk.BooleanVar(value=True))

    tk.Checkbutton(
        input_frame,
        text="Auto-calc (default: ON)",
        variable=state["chi_auto_calc_var"],
        bg="#F5F6FA",
        activebackground="#F5F6FA",
        highlightthickness=0,
        relief="flat"
    ).pack(pady=(2, 0))

    # ---- Placeholder (blank -> traceless) ----
    CHI_PLACEHOLDER = "blank → χ_iso=0"
    state["chi_placeholder_text"] = CHI_PLACEHOLDER
    state["chi_placeholder_active"] = False

    def _set_placeholder():
        # 빈칸이면 placeholder를 넣고 회색으로 표시
        if (chi_mol_entry.get() or "").strip() == "":
            chi_mol_entry.delete(0, tk.END)
            chi_mol_entry.insert(0, CHI_PLACEHOLDER)
            try:
                chi_mol_entry.config(fg="#888888")
            except Exception:
                pass
            state["chi_placeholder_active"] = True

    def _clear_placeholder():
        # placeholder 상태면 지우고 검정 글씨로
        if state.get("chi_placeholder_active", False):
            chi_mol_entry.delete(0, tk.END)
            try:
                chi_mol_entry.config(fg="#000000")
            except Exception:
                pass
            state["chi_placeholder_active"] = False

    # 최초에 placeholder 표시
    _set_placeholder()

    chi_mol_entry.bind("<FocusIn>", lambda e: _clear_placeholder())
    chi_mol_entry.bind("<FocusOut>", lambda e: (_set_placeholder(), state.get("schedule_chi_autocalc", lambda: None)()))

    # ---- Auto-calc core (NO popup on typing) ----
    state["_chi_autocalc_after_id"] = None

    def _maybe_calc_chi(_event=None, *, quiet=True):
        """Auto-calc가 켜져 있을 때만 χ_xx/yy/zz 계산.
        quiet=True: 입력 중간/placeholder일 땐 에러팝업 띄우지 않음
        """
        if not state["chi_auto_calc_var"].get():
            return

        # placeholder면 '빈칸' 취급
        if state.get("chi_placeholder_active", False):
            # 빈칸(traceless)로 계산하도록 그냥 entry를 빈 것으로 간주해야 함
            # -> table_utils 쪽에서 entry.get()을 읽으므로, 여기서는 임시로 ""로 두고 호출
            # 가장 안전하게는 entry 내용을 건드리지 말고, table_utils가 placeholder를 모르게 하는 것
            # 따라서 placeholder 상태에서는 아예 계산 호출만 해도 되고(=traceless),
            # calculate_tensor...가 entry.get()==""일 때 traceless로 가므로 아래처럼 "임시로" 처리.
            pass

        try:
            if state.get("rh_calc_enabled", False):
                rh_ent = state.get("rh_dchi_entry")
                if rh_ent is None:
                    return
                calculate_tensor_components_ui_ax_rh(
                    chi_mol_entry=state["chi_mol_entry"],
                    molar_value_label=state["molar_value_label"],
                    rh_dchi_entry=rh_ent,
                    tensor_xx_label=state["tensor_xx_label"],
                    tensor_yy_label=state["tensor_yy_label"],
                    tensor_zz_label=state["tensor_zz_label"],
                    messagebox=state["messagebox"],
                    # 아래 quiet 인자는 table_utils 수정(2번) 후에만 사용
                    quiet=quiet,
                    placeholder_text=CHI_PLACEHOLDER,
                )
            else:
                calculate_tensor_components_ui(
                    state["chi_mol_entry"],
                    state["molar_value_label"],
                    state["tensor_xx_label"],
                    state["tensor_yy_label"],
                    state["tensor_zz_label"],
                    state["messagebox"],
                    quiet=quiet,
                    placeholder_text=CHI_PLACEHOLDER,
                )
        except Exception:
            return

    state["maybe_calc_chi"] = _maybe_calc_chi

    def _schedule_chi_autocalc():
        """KeyRelease에서 바로 계산하지 말고 200ms 뒤에 1번만 실행(디바운스)."""
        try:
            if state["_chi_autocalc_after_id"] is not None:
                state["root"].after_cancel(state["_chi_autocalc_after_id"])
        except Exception:
            pass
        state["_chi_autocalc_after_id"] = state["root"].after(200, lambda: _maybe_calc_chi(None, quiet=True))

    state["schedule_chi_autocalc"] = _schedule_chi_autocalc

    chi_mol_entry.bind("<Return>", lambda e: _maybe_calc_chi(e, quiet=True))
    chi_mol_entry.bind("<KeyRelease>", lambda e: _schedule_chi_autocalc())

    _sep(input_frame)

    # File load
    ttk.Button(input_frame,
               text="📂 Load xyz File",
               command=lambda: load_xyz_file(state)
               ).pack(anchor="center", pady=3)

    # ---- Symmetry averaging (Me/CF3) ----
    state.setdefault("symavg_enabled_var", tk.BooleanVar(value=False))
    def _on_toggle_symavg():
        # Rebuild effective coordinates if a structure is loaded
        try:
            apply_symavg_to_state(state)
        except Exception as e:
            state["messagebox"].showwarning("Symmetry average", f"Failed:\n{e}")
            return
        tree = state.get("tree")
        if tree is not None:
            try:
                tree.selection_remove(tree.selection())
            except Exception:
                pass
        update_graph(state)
        try:
            populate_fitting_controls(state)
        except Exception:
            pass

    tk.Checkbutton(
        input_frame,
        text="Coordinate average (eg. CH₃)",
        variable=state["symavg_enabled_var"],
        command=_on_toggle_symavg,
        bg="#F5F6FA",
        activebackground="#F5F6FA",
        highlightthickness=0,
        relief="flat",
    ).pack(anchor="w", pady=(0, 0))

    state.setdefault("symavg_keep_original_var", tk.BooleanVar(value=False))
    tk.Checkbutton(
        input_frame,
        text="Keep original atoms",
        variable=state["symavg_keep_original_var"],
        command=lambda: (_on_toggle_symavg()),
        bg="#F5F6FA",
        activebackground="#F5F6FA",
        highlightthickness=0,
        relief="flat"
    ).pack(anchor="w", pady=(0, 0))

    _sep(input_frame)

    # Angle controls
    ttk.Label(input_frame, text="Rotate around X-axis (degrees):").pack()
    axf = ttk.Frame(input_frame); axf.pack(fill=tk.X)
    angle_x_var = tk.DoubleVar(); state['angle_x_var']=angle_x_var
    angle_x_slider = tk.Scale(axf, from_=-180, to=180, orient=tk.HORIZONTAL, variable=angle_x_var, resolution=0.1, command=lambda v: on_angle_slider(state, 'x', v), bg="#F5F6FA", activebackground="#F5F6FA", highlightthickness=0)
    angle_x_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    angle_x_entry = tk.Entry(axf, width=6); angle_x_entry.pack(side=tk.RIGHT, padx=5); angle_x_entry.insert(0,'0.0'); state['angle_x_entry']=angle_x_entry
    angle_x_entry.bind('<Return>', lambda e: on_angle_entry_commit(state, 'x'))
    angle_x_entry.bind('<FocusOut>', lambda e: on_angle_entry_commit(state, 'x'))

    ttk.Label(input_frame, text="Rotate around Y-axis (degrees):").pack()
    ayf = ttk.Frame(input_frame); ayf.pack(fill=tk.X)
    angle_y_var = tk.DoubleVar(); state['angle_y_var']=angle_y_var
    angle_y_slider = tk.Scale(ayf, from_=-180, to=180, orient=tk.HORIZONTAL, variable=angle_y_var, resolution=0.1, command=lambda v: on_angle_slider(state,'y',v), bg="#F5F6FA", activebackground="#F5F6FA", highlightthickness=0)
    angle_y_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    angle_y_entry = tk.Entry(ayf, width=6); angle_y_entry.pack(side=tk.RIGHT, padx=5); angle_y_entry.insert(0,'0.0'); state['angle_y_entry']=angle_y_entry
    angle_y_entry.bind('<Return>', lambda e: on_angle_entry_commit(state, 'y'))
    angle_y_entry.bind('<FocusOut>', lambda e: on_angle_entry_commit(state, 'y'))

    _sep(input_frame)

    # Open viewers buttons
    opf = ttk.Frame(input_frame)
    opf.pack(fill=tk.X, padx=0, pady=0)

    ttk.Label(opf, text="Open viewers", font=("default", 9, "bold")).grid(
        row=0, column=0, columnspan=2, pady=3
    )

    opf.columnconfigure(0, weight=1)
    opf.columnconfigure(1, weight=1)

    ttk.Button(
        opf,
        text="2D PCS Plot",
        command=lambda: open_pcs_plot_popup(state)
    ).grid(row=1, column=0, sticky="ew", padx=2, pady=(0, 4))

    ttk.Button(
        opf,
        text="3D structure",
        command=lambda: open_3d_plot_window(state)
    ).grid(row=1, column=1, sticky="ew", padx=2, pady=(0, 4))

    ttk.Button(
        opf,
        text="Projection",
        command=lambda: open_projection_window(state)
    ).grid(row=2, column=0, sticky="ew", padx=2)

    ttk.Button(
        opf,
        text="NMR Spectrum",
        command=lambda: open_nmr_window(state)
    ).grid(row=2, column=1, sticky="ew", padx=2)

    _sep(input_frame)

    # Checklist
    ttk.Label(input_frame, text="Select elements to display", font=("default",9,"bold")).pack(pady=0)
    checklist_frame = ttk.Frame(input_frame); checklist_frame.pack(pady=3, fill=tk.BOTH, expand=True); state['checklist_frame']=checklist_frame
    ttk.Label(input_frame, text="Rotation and atom selection can be\n applied after updating PCS information.", font=("Helvetica",8,"italic"), justify="center", anchor="center").pack(pady=0)

    _sep(input_frame)

    _sep(root, orient='horizontal', pady=0, fill='x')

    # Command bar
    cmdf = ttk.Frame(root)
    cmdf.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
    ttk.Label(cmdf, text="Command :").pack(side=tk.LEFT)
    command_entry = tk.Entry(cmdf, width=50); command_entry.pack(side=tk.LEFT, padx=5); state['command_entry']=command_entry
    ttk.Button(cmdf, text="Run", command=lambda: _pc(state)).pack(side=tk.LEFT, padx=5)
    ttk.Button(cmdf, text="Exit", command=lambda: root.destroy()).pack(side=tk.RIGHT, padx=0)

    # Export buttons
    sbf = ttk.Frame(cmdf); sbf.pack(side=tk.RIGHT, padx=15)
    ttk.Label(sbf, text="Export data ", font=("default",9,"bold")).grid(row=1, column=0, sticky="ew", pady=3)
    ttk.Button(sbf, text="💾 Save plot",  command=lambda: on_save_plot_any(state)).grid( row=1, column=1, sticky="ew", padx=2, pady=1)
    ttk.Button(sbf, text="💾 Save table", command=lambda: on_save_table_any(state)).grid(row=1, column=2, sticky="ew", padx=2, pady=1)

    # Defaults
    state['delta_exp_values'] = {}
    reset_values(state)  # sets defaults and draws initial plots
    # Bindings
    tree.bind("<Double-1>", lambda e: on_delta_entry_change(state, e, state['delta_exp_values'], plot_cartesian_graph))
    tree.bind("<<TreeviewSelect>>", lambda e: _on_tree_select_update_views(state))

    state['root'].after(150, lambda: open_pcs_plot_popup(state))
    state['root'].after(250, lambda: state['update_graph']())
    return state

def switch_fit_mode(state):
    m = state['fit_mode_var'].get()

    # Hide all mode-specific setting frames first
    state['fit_frameA'].pack_forget()
    state['fit_frameB'].pack_forget()
    state['fit_frameC'].pack_forget()
    anchor = state['fit_anchor']

    if m == 'theta_alpha_multi':
        state['fit_frameA'].pack(fill=state['tk'].X, pady=4, before=anchor)
    elif m == 'euler_global':
        state['fit_frameB'].pack(fill=state['tk'].X, pady=4, before=anchor)
    elif m == 'full_tensor':
        state['fit_frameC'].pack(fill=state['tk'].X, pady=4, before=anchor)

    try:
        populate_fitting_controls(state)
    except Exception:
        pass

def run_fit_from_ui(state):
    mode = state['fit_mode_var'].get()

    psel = state['fit_proton_list'].curselection()
    if not psel:
        state['messagebox'].showerror("Fitting", "Select ≥3 protons (Ref IDs).")
        return
    proton_ids = [int(state['fit_proton_list'].get(i)) for i in psel]

    if mode == 'theta_alpha_multi':
        dsel = state['fit_donor_list'].curselection()
        if not dsel:
            state['messagebox'].showerror("Fitting", "Select ≥1 donor IDs.")
            return
        donor_ids = [int(state['fit_donor_list'].get(i).split(':')[0]) for i in dsel]
    else:
        donor_ids = []

    def _finish_success(res):
        state['last_fit_result'] = res
        _show_fit_result(state, res)
        update_fit_correlation_plot_from_result(state, res)
        state['fit_status_var'].set("Fit completed.")

    def _finish_error(msg):
        state['fit_status_var'].set("Fit failed.")
        state['messagebox'].showerror("Fitting", msg)

    def _worker():
        try:
            state['root'].after(0, lambda: state['fit_status_var'].set("Collecting atoms..."))

            if mode == 'theta_alpha_multi':
                res = fit_theta_alpha_multi(
                    state,
                    donor_ids=donor_ids,
                    proton_ids=proton_ids,
                    axis_mode=state['axis_mode_var'].get(),
                    fit_visible_as_group=state['fit_use_visible_var'].get(),
                    fit_delta_chi=state['fit_dchi_var'].get(),
                    fit_delta_chi_rh=state['fit_dchi_rh_var'].get(),
                    use_global_search=state['fit_global_search_var'].get(),
                    progress_cb=state.get('fit_progress_cb'),
                )

            elif mode == 'euler_global':
                res = fit_euler_global(
                    state,
                    proton_ids=proton_ids,
                    fit_visible_as_group=state['fit_use_visible_var'].get(),
                    fit_delta_chi=state['fit_dchi_var'].get(),
                    fit_delta_chi_rh=state['fit_dchi_rh_var'].get(),
                    use_global_search=state['fit_global_search_var'].get(),
                    progress_cb=state.get('fit_progress_cb'),
                )

            elif mode == 'full_tensor':
                res = fit_full_tensor(
                    state,
                    proton_ids=proton_ids,
                    fit_delta_chi=state['fit_dchi_var'].get(),
                    fit_delta_chi_rh=state['fit_dchi_rh_var'].get(),
                    use_global_search=state['fit_global_search_var'].get(),
                    progress_cb=state.get('fit_progress_cb'),
                )

            else:
                raise ValueError(f"Unknown fitting mode: {mode}")

            state['root'].after(0, lambda r=res: _finish_success(r))

        except Exception as e:
            state['root'].after(0, lambda m=str(e): _finish_error(m))

    state['fit_status_var'].set("Preparing fit...")
    threading.Thread(target=_worker, daemon=True).start()

def _show_fit_result(state, res):
    box = state['fit_result_box']
    box.delete("1.0", "end")

    # ratio threshold warning
    state.setdefault("rh_ratio_warn_threshold", 0.5)  # ratio threshold 0.5

    mode = res.get('mode')

    if mode == 'theta_alpha_multi':
        box.insert("end", f"[Mode A] axis={res.get('axis_mode')}\n")
        box.insert("end", f"Donors: {res['donor_ids']}\n")
        box.insert("end", f"θ* = {res['theta']:.2f}°, α* = {res['alpha']:.2f}°\n")

    elif mode == 'euler_global':
        box.insert("end", "[Mode B] Euler global\n")
        box.insert("end", f"ax = {res['ax']:.2f}°, ay = {res['ay']:.2f}°, az = {res['az']:.2f}°\n")

    elif mode == 'full_tensor':
        box.insert("end", "[Mode C] Full tensor fit\n")
        if 'metal_pos' in res:
            mx, my, mz = res['metal_pos']
            box.insert("end", f"Metal position = ({mx:.3f}, {my:.3f}, {mz:.3f})\n")
        if 'euler_deg' in res:
            ea, eb, eg = res['euler_deg']
            box.insert("end", f"Euler angles = ({ea:.2f}°, {eb:.2f}°, {eg:.2f}°)\n")

    else:
        box.insert("end", f"[Unknown mode] {mode}\n")

    # --- χ tensor parameters ---
    box.insert("end", f"Δχ_ax = {res['delta_chi_ax']:.3e} (E-32 m³)\n")

    if 'delta_chi_rh' in res:
        dchi_ax = res['delta_chi_ax']
        dchi_rh = res['delta_chi_rh']
        box.insert("end", f"Δχ_rh = {dchi_rh:.3e} (E-32 m³)\n")
        if abs(dchi_ax) > 0:
            ratio = abs(dchi_rh / dchi_ax)
            box.insert("end", f"|Δχ_rh / Δχ_ax| = {ratio:.3f}\n")
            thr = float(state.get("rh_ratio_warn_threshold", 0.5))
            if ratio > thr:
                box.insert(
                    "end",
                    f"⚠ Warning: |Δχ_rh/Δχ_ax| exceeds {thr:.2f}. "
                    "Rhombic fit may be unstable or φ-reference/convention may be ill-defined.\n"
                )
        else:
            box.insert("end", "|Δχ_rh / Δχ_ax| = undefined (Δχ_ax = 0)\n")

    # --- φ reference / z-rotation meaning (mode-dependent) ---
    if mode == 'theta_alpha_multi':
        # In Mode A, the φ reference follows the user-defined z-rotation.
        try:
            az_ref = float(state.get('angle_z_var', 0.0).get())
            box.insert("end", f"φ reference (user z-rotation) = {az_ref:.1f}°\n")
        except Exception:
            pass

    elif mode == 'euler_global':
        # In Mode B, az is part of the fitted Euler angles.
        box.insert("end", f"Euler z-angle (az) = {res['az']:.2f}°  (φ reference is defined by Euler convention)\n")

    elif mode == 'full_tensor':
        # In Mode C, the Euler convention is defined by the fitted tensor orientation.
        if 'euler_deg' in res:
            _, _, eg = res['euler_deg']
            box.insert("end", f"Euler z-angle = {eg:.2f}°  (tensor orientation from full-tensor fit)\n")

    # --- fit quality ---
    box.insert("end", f"RMSD = {res['rmsd']:.3f} ppm  (N = {res['n']})\n")
    r2 = res.get('r2', float('nan'))
    if not np.isnan(r2):
        box.insert("end", f"R² = {r2:.4f}\n")
    qf = res.get('q_factor', float('nan'))
    if not np.isnan(qf):
        box.insert("end", f"Q-factor = {qf:.4f}\n")
    chi2_n = res.get('chi2_n', float('nan'))
    if not np.isnan(chi2_n):
        box.insert("end", f"χ²/N = {chi2_n:.4f} ppm²\n")
    box.insert("end", "\n")

    box.insert("end", "Ref\tδ_exp\tδ_pred\tresid\n")
    for rid, exp, pred, r in res['per_point']:
        box.insert("end", f"{rid}\t{exp:.3f}\t{pred:.3f}\t{r:+.3f}\n")

# Fitting function - correlation plot
def update_fit_correlation_plot_from_result(state, res):
    fig = state.get('fit_corr_fig')
    canvas = state.get('fit_corr_canvas')
    if fig is None or canvas is None:
        return

    per_point = res.get('per_point', [])
    fig.clear()

    if not per_point:
        canvas.draw_idle()
        return

    exp_values = np.array([row[1] for row in per_point], float)
    pred_values = np.array([row[2] for row in per_point], float)
    resid_values = np.array([row[3] for row in per_point], float)
    ref_ids = [row[0] for row in per_point]

    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[2.5, 1.0],
        hspace=0.08,
        left=0.15, right=0.92, top=0.90, bottom=0.16
    )

    ax_corr = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1])

    mn = min(exp_values.min(), pred_values.min())
    mx = max(exp_values.max(), pred_values.max())
    if mn == mx:
        mn -= 1.0
        mx += 1.0

    pad = 0.08 * (mx - mn)
    lo = mn - pad
    hi = mx + pad

    ax_corr.plot([lo, hi], [lo, hi], linestyle='--', color='gray', linewidth=0.9, alpha=0.7)

    sc = ax_corr.scatter(
        exp_values,
        pred_values,
        c=np.abs(resid_values),
        cmap='viridis',
        s=45,
        alpha=0.9
    )

    for rid, x, y in zip(ref_ids, exp_values, pred_values):
        ax_corr.annotate(
            str(rid),
            (x, y),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=6,
            color='dimgray'
        )

    ax_corr.set_xlim(lo, hi)
    ax_corr.set_ylim(lo, hi)
    ax_corr.set_xlabel("Experimental δ (ppm)")
    ax_corr.set_ylabel("Predicted δ (ppm)")
    ax_corr.set_title(
        f"Fit correlation   R²={res.get('r2', float('nan')):.4f}   "
        f"RMSD={res.get('rmsd', float('nan')):.4f} ppm   "
        f"Q={res.get('q_factor', float('nan')):.4f}",
        fontsize=9
    )
    ax_corr.grid(alpha=0.2)

    cbar = fig.colorbar(sc, ax=ax_corr, fraction=0.04, pad=0.02)
    cbar.set_label("|residual| (ppm)")

    colors = []
    for r in resid_values:
        ar = abs(r)
        if ar <= 0.05:
            colors.append("#2ecc71")
        elif ar <= 0.15:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    ax_res.bar(np.arange(len(resid_values)), resid_values, color=colors, alpha=0.9)
    ax_res.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_res.set_ylabel("resid")
    ax_res.set_xticks(np.arange(len(ref_ids)))
    ax_res.set_xticklabels([str(r) for r in ref_ids], rotation=45, ha='right', fontsize=6)
    ax_res.grid(axis='y', alpha=0.2)

    canvas.draw_idle()

# Fitting function - correlation plot export
def export_fit_plot(state):
    fig = state.get('fit_corr_fig')
    if fig is None:
        state['messagebox'].showwarning("Export", "No fit plot available.")
        return

    path = state['filedialog'].asksaveasfilename(
        title="Save fit plot",
        defaultextension=".png",
        filetypes=[
            ("PNG image", "*.png"),
            ("PDF file", "*.pdf"),
        ],
    )
    if not path:
        return

    fig.savefig(path, dpi=600, bbox_inches="tight")
    state['messagebox'].showinfo("Export", f"Saved fit plot:\n{path}")

def on_save_plot_any(state):
    fd = state['filedialog'].asksaveasfilename(
        title="Save plot data",
        defaultextension=".xlsx",
        filetypes=[
            ("Excel workbook", "*.xlsx"),
            ("CSV (PCS & Atoms, two files)", "*.csv"),
            ("PNG image (polar plot)", "*.png"),
        ],
    )
    if not fd:
        return

    base, ext = os.path.splitext(fd)
    ext = ext.lower()

    if ext == ".xlsx":
        pcs_values, theta_values, tensor, polar_data = recompute_plot_inputs(state)
        from logic.export_utils import save_to_excel
        save_to_excel(pcs_values, theta_values, tensor, fd, polar_data)
        state['messagebox'].showinfo("Export", f"Saved Excel:\n{fd}")

    elif ext == ".csv":
        pcs_values, theta_values, tensor, polar_data = recompute_plot_inputs(state)
        from logic.export_utils import save_to_csv
        pcs_path, atoms_path = save_to_csv(pcs_values, theta_values, tensor, fd, polar_data)
        state['messagebox'].showinfo("Export", f"Saved CSV:\n{pcs_path}\n{atoms_path}")

        # Origin
        try:
            script = os.path.join(os.path.dirname(__file__), '..', 'tools', 'plot_pcs_origin.py')
            script = os.path.abspath(script)

            cmd = [sys.executable, script, pcs_path]
            if state['plot_90_var'].get():
                cmd.append('--half')

            kwargs = {}
            if os.name == 'nt':
                DETACHED = getattr(subprocess, 'DETACHED_PROCESS', 0)
                NEWGROUP = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
                kwargs['creationflags'] = DETACHED | NEWGROUP

            subprocess.Popen(cmd, **kwargs)

        except Exception as e:
            state['messagebox'].showwarning(
                "Origin",
                f"CSV saved but failed to start Origin automation:\n{e}"
            )

    elif ext == ".png":
        win = state.get("pcs_plot_popup")
        canvas = state.get("pcs_canvas_popup")
        try:
            if win is None or not win.winfo_exists() or canvas is None:
                state['messagebox'].showerror(
                    "Export", "No 2D PCS plot window is open.\nPlease open the 2D PCS Plot window first."
                )
                return

        except Exception:

            state['messagebox'].showerror(
                "Export", "No valid 2D PCS plot window is available."
            )
            return
        canvas.print_figure(fd, dpi=600)
        state['messagebox'].showinfo("Export", f"Saved PNG:\n{fd}")

    else:
        # 확장자 없으면 .xlsx로 저장
        fd_x = base + ".xlsx"
        pcs_values, theta_values, tensor, polar_data = recompute_plot_inputs(state)
        from logic.export_utils import save_to_excel
        save_to_excel(pcs_values, theta_values, tensor, fd_x, polar_data)
        state['messagebox'].showinfo("Export", f"No/unknown extension. Saved as Excel:\n{fd_x}")

def on_save_table_any(state):
    fd = state['filedialog'].asksaveasfilename(
        title="Save table",
        defaultextension=".xlsx",
        filetypes=[
            ("Excel workbook", "*.xlsx"),
            ("CSV (table only)", "*.csv"),
        ],
    )
    if not fd:
        return

    base, ext = os.path.splitext(fd)
    ext = ext.lower()

    # rh_tree는 없을 수도 있음 (Rhombicity 탭을 아직 생성 안 했거나)
    rh_tree = state.get("rh_tree", None)

    if ext == ".xlsx":
        # 메인 + Rhombicity(있으면)를 같은 xlsx에 시트로 저장
        from logic.export_utils import export_tables_to_excel
        export_tables_to_excel(state['tree'], fd, rh_tree=rh_tree)
        state['messagebox'].showinfo("Export", f"Saved Excel table:\n{fd}")

    elif ext == ".csv":
        # CSV는 파일 2개로 저장: *_table.csv (+ 있으면 *_rhombicity.csv)
        from logic.export_utils import export_tables_to_csv
        table_path, rh_path = export_tables_to_csv(state['tree'], fd, rh_tree=rh_tree)

        if rh_path:
            state['messagebox'].showinfo("Export", f"Saved CSV tables:\n{table_path}\n{rh_path}")
        else:
            state['messagebox'].showinfo("Export", f"Saved CSV table:\n{table_path}")

    else:
        fd_x = base + ".xlsx"
        from logic.export_utils import export_tables_to_excel
        export_tables_to_excel(state['tree'], fd_x, rh_tree=rh_tree)
        state['messagebox'].showinfo("Export", f"No/unknown extension. Saved as Excel:\n{fd_x}")

def recompute_plot_inputs(state):
    tensor = state['tensor_entry'].get()
    tensor = float(tensor) if tensor else 1.0
    pcs_min = float(state['pcs_min_entry'].get()); pcs_max = float(state['pcs_max_entry'].get()); pcs_interval = float(state['pcs_interval_entry'].get())
    pcs_values = np.arange(pcs_min, pcs_max + pcs_interval, pcs_interval)
    theta_values = state.get('theta_values')
    if theta_values is None:
        theta_values = np.linspace(0, 2*np.pi, 500); state['theta_values']=theta_values
    polar_data, _ = filter_atoms(state)
    return pcs_values, theta_values, tensor, polar_data

def update_graph(state):
    tensor = state['tensor_entry'].get()
    tensor = float(tensor) if tensor else 1.0

    pcs_min = float(state['pcs_min_entry'].get())
    pcs_max = float(state['pcs_max_entry'].get())
    pcs_interval = float(state['pcs_interval_entry'].get())
    pcs_values = np.arange(pcs_min, pcs_max + pcs_interval, pcs_interval)
    state['pcs_values'] = pcs_values

    if state.get('atom_data'):
        polar_data, rotated_coords = filter_atoms(state)
        update_table(state, polar_data, rotated_coords, tensor, state['delta_exp_values'])
        try:
            state['root']._stripe_treeview(state['tree'])
        except Exception:
            pass
    else:
        polar_data = None

    theta_values = state.get('theta_values')
    if theta_values is None:
        theta_values = np.linspace(0, 2*np.pi, 500)
    state['theta_values'] = theta_values

    plot_graph(state, pcs_values, theta_values, tensor, polar_data=polar_data)
    plot_cartesian_graph(state)
    update_molar_value(state, tensor)
    try:
        if state.get("chi_auto_calc_var") and state["chi_auto_calc_var"].get():
            if "maybe_calc_chi" in state:
                state["root"].after(0, lambda: state["maybe_calc_chi"](None))
    except Exception as e:
        print("auto-calc failed:", e)

    if 'fit_proton_list' in state:
        try:
            populate_fitting_controls(state)
        except Exception:
            pass

    # projection window
    try:
        win = state.get("projection_popup")
        if win is not None and win.winfo_exists():
            from ui.projection_window import _draw_projection_plot
            _draw_projection_plot(state)
    except Exception:
        pass

    # 3d plot window
    try:
        win3d = state.get("plot3d_popup")
        if win3d is not None and win3d.winfo_exists():
            from ui.plot_3d_window import _draw_3d_plot
            _draw_3d_plot(state)
    except Exception:
        pass

def reset_values(state):
    state['tensor_entry'].delete(0, tk.END); state['tensor_entry'].insert(0, '-2.0')
    state['pcs_min_entry'].delete(0, tk.END); state['pcs_min_entry'].insert(0, '-10')
    state['pcs_max_entry'].delete(0, tk.END); state['pcs_max_entry'].insert(0, '10')
    state['pcs_interval_entry'].delete(0, tk.END); state['pcs_interval_entry'].insert(0, '0.5')
    state['angle_x_var'].set(0); state['angle_y_var'].set(0);

    state.get('delta_exp_values', {}).clear()
    state.pop('last_rotated_coords', None)

    if 'angle_z_var' in state:
        state['angle_z_var'].set(0)
    if 'angle_z_entry' in state:
        state['angle_z_entry'].delete(0, tk.END)
        state['angle_z_entry'].insert(0, "0.0")

    state.get('delta_exp_values', {}).clear()
    state.pop('last_rotated_coords', None)

    # Reset fit override as well; otherwise Rh/PCS can remain overridden.
    state.pop('fit_override', None)
    state.pop('last_fit_result', None)

    state['atom_data'] = [];
    state['check_vars'] = {}
    state['pcs_values'] = __import__('numpy').arange(-10, 10.5, 0.5)
    for w in state['checklist_frame'].winfo_children(): w.destroy()
    for r in state['tree'].get_children(): state['tree'].delete(r)

    if 'theta_values' not in state:
        state['theta_values'] = __import__('numpy').linspace(0, 2 * __import__("numpy").pi, 500)

    # Rhombicity tab reset
    state['rh_dchi_rh'] = 0.0
    if 'rh_dchi_rh_var' in state:
        state['rh_dchi_rh_var'].set("0")

    state['rh_calc_enabled'] = False
    if 'rh_calc_enabled_var' in state:
        state['rh_calc_enabled_var'].set(False)

    rh_tree = state.get('rh_tree')
    if rh_tree is not None:
        for it in rh_tree.get_children():
            rh_tree.delete(it)

    # conformer search reset
    state['atom_data_raw'] = None
    state['atom_data_eff'] = []
    state['atom_ids_eff'] = []
    state['ref_label_overrides'] = {}
    state['symavg_pseudo_ref_ids'] = set()
    state['symavg_records'] = []
    state['metal_ref_id'] = None
    state['atom_data_original'] = None
    state['atom_data_conformer'] = None

    state['conformer_preview'] = None
    state['conformer_preview_coords'] = None
    state['conformer_preview_elements'] = None
    state['conformer_preview_report'] = ""
    state['conformer_applied'] = False

    if 'conformer_status_var' in state:
        state['conformer_status_var'].set("Ready.")
    if 'conformer_result_box' in state:
        state['conformer_result_box'].delete("1.0", "end")

    update_graph(state)

    # (4) 가능하면 Rh 테이블도 "현재 상태"로 다시 채우기 (원하면 주석 해제)
    # try:
    #     state['plots_nb'].update_idletasks()
    #     # Rh table은 atom_data가 비면 rows가 없을 거라 결과도 비어있음(정상)
    #     # 나중에 xyz 로드 후엔 update_graph/populate가 다시 채움
    # except Exception:
    #     pass

def on_angle_slider(state, axis, value):
    if axis == 'x':
        state['angle_x_entry'].delete(0, tk.END)
        state['angle_x_entry'].insert(0, f"{float(value):.1f}")
    elif axis == 'y':
        state['angle_y_entry'].delete(0, tk.END)
        state['angle_y_entry'].insert(0, f"{float(value):.1f}")
    elif axis == 'z':
        if 'angle_z_entry' in state:
            state['angle_z_entry'].delete(0, tk.END)
            state['angle_z_entry'].insert(0, f"{float(value):.1f}")
    update_graph(state)

def on_angle_entry_commit(state, axis):
    try:
        if axis == 'x':
            val = float(state['angle_x_entry'].get())
            state['angle_x_var'].set(val)
        elif axis == 'y':
            val = float(state['angle_y_entry'].get())
            state['angle_y_var'].set(val)
        elif axis == 'z':
            val = float(state['angle_z_entry'].get())
            state['angle_z_var'].set(val)
    except ValueError:
        return
    state['update_graph']()

def apply_symavg_to_state(state):
    """
    Build effective coordinates (2D/table only) from raw structure
    according to symmetry-average UI toggles.

    - 3D viewer always uses raw coordinates (state["atom_data"])
    - 2D plot + table use state["atom_data_eff"]

    Modes:
        enabled = False → raw structure as it is
        enabled = True  →
            drop mode (default): averaged H/F delete + only pseudo used
            mask mode (keep original): original H/F + pseudo
    """
    # --- raw structure  ---
    raw = state.get("atom_data_raw") or state.get("atom_data") or []
    if not raw:
        state["atom_data_eff"] = []
        state["atom_ids_eff"] = []
        state["symavg_pseudo_ref_ids"] = set()
        state["symavg_records"] = []
        return

    enabled_var = state.get("symavg_enabled_var")
    enabled = bool(enabled_var.get()) if enabled_var is not None else False

    # --- symmetry average OFF → raw as it is ---
    if not enabled:
        state["atom_data_eff"] = list(raw)
        state["atom_ids_eff"] = list(range(1, len(raw) + 1))
        state["symavg_pseudo_ref_ids"] = set()
        state["symavg_records"] = []
        return

    # --- keep original atoms  ---
    keep_var = state.get("symavg_keep_original_var")
    keep_original = bool(keep_var.get()) if keep_var is not None else False

    # collapse mode
    mode = "mask" if keep_original else "drop"

    # --- 시작 구조 ---
    atom_data = list(raw)
    ids = list(range(1, len(raw) + 1))

    records_all = []
    # Ref-ID -> display label override for table (pseudo atoms)
    # Example: { 153: "MeH@C12", 154: "CF3F@C7" }
    label_overrides = {}


    def _apply_collapse_with_ids(atom_data_in, ids_in, collapse_fn):
        """
        collapse_* 함수 적용 + ref id 동기화
        mask mode:
            original ids 유지 + pseudo ids append
        drop mode:
            masked indices 제거 후 pseudo ids append
        """

        out_atoms, records, masked = collapse_fn(atom_data_in)

        # --- mask mode ---
        if mode == "mask":
            next_id = (max(ids_in) if ids_in else 0) + 1
            pseudo_ids = list(range(next_id, next_id + len(records)))
            out_ids = list(ids_in) + pseudo_ids
            return out_atoms, out_ids, records, set(pseudo_ids)

        # --- drop mode ---
        kept_ids = [rid for i, rid in enumerate(ids_in) if i not in masked]

        next_id = (max(ids_in) if ids_in else 0) + 1
        pseudo_ids = list(range(next_id, next_id + len(records)))
        out_ids = kept_ids + pseudo_ids

        return out_atoms, out_ids, records, set(pseudo_ids)

    # 1) Methyl collapse
    def _do_me(atoms):
        return collapse_methyl_groups(
            atoms,
            mode=mode,
            pseudo_element="H",
            require_carbon_substituent_count=1,
        )

    atom_data, ids, rec_me, pseudo_me = _apply_collapse_with_ids(
        atom_data, ids, _do_me
    )
    records_all.extend(rec_me)

    # Map pseudo Ref IDs to human-readable labels for table display
    # NOTE: record.pseudo_index must refer to the index in the returned out_atoms list
    for rec in rec_me:
        try:
            rid = ids[rec.pseudo_index]
            label_overrides[rid] = rec.label
        except Exception:
            pass

    # 2) CF3 collapse
    def _do_cf(atoms):
        return collapse_cf3_groups(
            atoms,
            mode=mode,
            pseudo_element="F",
            require_carbon_substituent_count=1,
        )

    atom_data, ids, rec_cf, pseudo_cf = _apply_collapse_with_ids(
        atom_data, ids, _do_cf
    )
    records_all.extend(rec_cf)

    for rec in rec_cf:
        try:
            rid = ids[rec.pseudo_index]
            label_overrides[rid] = rec.label
        except Exception:
            pass

    # --- pseudo ref ids (plot에서 marker='x' 구분용) ---
    pseudo_ref_ids = set()
    pseudo_ref_ids.update(pseudo_me)
    pseudo_ref_ids.update(pseudo_cf)

    # --- state 반영 ---
    state["atom_data_eff"] = atom_data
    state["atom_ids_eff"] = ids
    state["symavg_pseudo_ref_ids"] = pseudo_ref_ids
    state["symavg_records"] = records_all
    state["ref_label_overrides"] = label_overrides

def load_xyz_file(state):
    path = state['filedialog'].askopenfilename(filetypes=[
        ("Structure files", "*.xyz *.out *.log"),
        ("XYZ", "*.xyz"),
        ("ORCA output", "*.out *.log"),
        ("All files", "*.*"),
    ])
    if not path:
        return

    # reset
    state.get('delta_exp_values', {}).clear()
    state.pop('last_rotated_coords', None)
    state.pop('row_by_id', None)
    state.pop('current_selected_ids', None)

    atom_data = load_structure(path)

    # reset conformer search
    state['conformer_preview'] = None
    state['conformer_preview_coords'] = None
    state['conformer_preview_elements'] = None
    state['conformer_preview_report'] = ""
    state['conformer_applied'] = False

    if 'conformer_status_var' in state:
        state['conformer_status_var'].set(
            "Structure loaded. Click '🔄 Sync' after loading δ_Exp")

    # Keep raw for 3D plot
    state["atom_data_original"] = list(atom_data)
    state["atom_data_raw"] = list(atom_data)
    state["atom_data"] = list(atom_data)
    state["atom_ids_raw"] = list(range(1, len(atom_data) + 1))

    # Build effective (2D/table) data based on checkbox
    apply_symavg_to_state(state)

    target_atom = state['simpledialog'].askstring("Input center atom", "Enter the center atom (element) :")
    if not target_atom:
        return
    tgt = None
    metal_ref_id = None

    for idx, (atom, x, y, z) in enumerate(atom_data, start=1):
        if atom == target_atom:
            tgt = (x, y, z)
            metal_ref_id = idx
            break

    if tgt is None:
        state['messagebox'].showerror("Error", f"Atom {target_atom} not found in the file.")
        return

    state['x0'], state['y0'], state['z0'] = tgt
    state['metal_ref_id'] = metal_ref_id

    create_checklist(state)
    apply_symavg_to_state(state)
    update_graph(state)
    populate_fitting_controls(state)

    # Open the 3D viewer automatically after loading the structure file.
    try:
        state['root'].after(50, lambda: open_3d_plot_window(state))
    except Exception as e:
        state['messagebox'].showwarning("3D viewer", f"Could not open 3D window:\n{e}")

def create_checklist(state):
    for w in state['checklist_frame'].winfo_children(): w.destroy()
    atom_types = sorted(set([a for a, *_ in state['atom_data']]))
    canvas = tk.Canvas(state['checklist_frame'], width=100, height=60); scrollbar = ttk.Scrollbar(state['checklist_frame'], orient='vertical', command=canvas.yview); scroll = tk.Frame(canvas)
    scroll.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0,0), window=scroll, anchor="nw"); canvas.configure(yscrollcommand=scrollbar.set)
    check_vars = {}
    for i, tp in enumerate(atom_types):
        var = tk.BooleanVar(value=True)
        cb = tk.Checkbutton(
            scroll,
            text=tp,
            variable=var,
            command=lambda: (update_graph(state), populate_fitting_controls(state))
        )
        row = i//3; col=i%3; cb.grid(row=row, column=col, sticky='w', padx=5, pady=2); check_vars[tp]=var
    canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
    state['check_vars']=check_vars
    populate_fitting_controls(state)

def filter_atoms(state):
    sel = {a for a, v in state['check_vars'].items() if v.get()}

    # Use effective coords for 2D/table, keep raw for 3D separately
    atom_data = state.get("atom_data_eff") or state.get("atom_data") or []
    ids = state.get("atom_ids_eff") or state.get("atom_ids") or list(range(1, len(atom_data) + 1))

    abs_coords = np.array([[x, y, z] for a, x, y, z in atom_data])
    metal = np.array([state['x0'], state['y0'], state['z0']])

    fo = state.get('fit_override')
    if fo:
        mode = (fo.get('mode') or '').lower()

        # --- Mode A: donor-axis (theta/alpha) ---
        if mode == 'theta_alpha_multi':
            id2idx = {rid: i for i, rid in enumerate(ids)}
            donor_ids = fo.get('donor_ids') or []
            if donor_ids:
                donor_pts = [abs_coords[id2idx[rid]] for rid in donor_ids if rid in id2idx]
                abs_coords = _angles_to_rotation_multi(
                    points=abs_coords,
                    metal=metal,
                    donor_points=donor_pts,
                    theta_deg=fo.get('theta', 0.0),
                    alpha_deg=fo.get('alpha', 0.0),
                    axis_mode=fo.get('axis_mode', 'bisector')
                )

        # --- Mode B: global Euler (ax/ay/az) ---
        elif mode == 'euler_global':
            ax = float(fo.get('ax', 0.0))
            ay = float(fo.get('ay', 0.0))
            az = float(fo.get('az', 0.0))
            coords0 = abs_coords - metal
            rot0 = rotate_euler(coords0, ax, ay, az)
            abs_coords = rot0 + metal

        # --- Mode C: full tensor ---
        elif mode == 'full_tensor':
            euler_deg = fo.get('euler_deg', (0.0, 0.0, 0.0))
            ax_e, ay_e, az_e = euler_deg

            metal_pos = fo.get('metal_pos')
            if metal_pos is not None:
                metal = np.array(metal_pos, dtype=float)

            coords0 = abs_coords - metal
            rot0 = rotate_euler(coords0, ax_e, ay_e, az_e)
            abs_coords = rot0 + metal

            if 'dchi_ax' in fo:
                try:
                    state['tensor_entry'].delete(0, tk.END)
                    state['tensor_entry'].insert(0, f"{float(fo['dchi_ax']):g}")
                except Exception:
                    pass

    coords0 = abs_coords - metal

    ax = float(state['angle_x_var'].get())
    ay = float(state['angle_y_var'].get())
    az = float(state['angle_z_var'].get())
    rotated = rotate_coordinates(coords0, ax, ay, az, (0, 0, 0))

    polar = []
    rotated_sel = []
    selected_ids = []

    label_overrides = state.get("ref_label_overrides", {}) or {}

    for idx, ((atom, *_), (dx, dy, dz)) in enumerate(zip(atom_data, rotated)):
        if atom in sel:
            r = (dx*dx + dy*dy + dz*dz) ** 0.5
            theta = np.arccos(dz / r) if r != 0 else 0.0

            ref_id = ids[idx]
            atom_label = label_overrides.get(ref_id, atom)

            polar.append((atom_label, r, theta))
            rotated_sel.append((dx, dy, dz))
            selected_ids.append(ref_id)

    state['current_selected_ids'] = selected_ids
    return polar, rotated_sel

def wire(state):
    state['update_graph'] = lambda : update_graph(state)
    state['plot_cartesian'] = plot_cartesian_graph
    state['filter_atoms'] = filter_atoms
    state['update_table'] = update_table

    return state
