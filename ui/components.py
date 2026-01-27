# ui/components.py

import subprocess, sys, os
import tkinter as tk
import tkinter.ttk as ttk
from ui.style import apply_style
from tkinter import filedialog, simpledialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from logic.command_processor import process_command as _pc
from logic.xyz_loader import parse_xyz
from logic.plot_pcs import plot_graph, update_figsize
from logic.plot_cartesian import plot_cartesian_graph
from logic.plot_3d import open_3d_plot_window
from logic.mollweide_projection import open_theta_phi_plot as open_mollweide_plot
from logic.table_utils import (
    update_molar_value, update_table, on_delta_entry_change, calculate_tensor_components_ui, calculate_tensor_components_ui_ax_rh,
    export_delta_exp_template, import_delta_exp_file, import_delta_exp_from_clipboard, undo_last_delta_import, clear_delta_exp
)
from logic.rotate_align import rotate_coordinates, rotate_euler
from logic.chem_constants import CPK_COLORS
from logic.fitting import (
    populate_fitting_controls, apply_fit_to_views,
    fit_theta_alpha_multi, fit_euler_global, _angles_to_rotation_multi
)
from logic.include_rhombic import build_rh_table_rows

def get_cpk_color(atom):
    return CPK_COLORS.get(atom, CPK_COLORS['default'])

def _sep(parent, orient='horizontal', pady=8, fill='x'):
    s = ttk.Separator(parent, orient=orient)
    s.pack(fill=fill, pady=pady)
    return s

def build_app():
    state = {}
    # Tk and common modules
    state['tk'] = tk; state['ttk'] = ttk
    state['filedialog'] = filedialog; state['simpledialog'] = simpledialog; state['messagebox'] = messagebox
    state['FigureCanvas'] = FigureCanvasTkAgg; state['NavigationToolbar2Tk'] = NavigationToolbar2Tk

    root = tk.Tk(); root.title("PCS Analyzer"); root.geometry("1400x890"); state['root'] = root
    apply_style(root, variant="light", accent="green")  # darkmode : variant="dark"

    # Frames
    main_frame = ttk.Frame(root)
    main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    left_frame = ttk.Frame(main_frame); left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); state['left_frame']=left_frame
    center_frame = ttk.Frame(main_frame); center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); state['center_frame']=center_frame
    right_frame = ttk.Frame(main_frame); right_frame.pack(side=tk.LEFT, fill=tk.Y); state['right_frame']=right_frame

    # PCS figure/canvas
    pcs_figure = plt.figure(figsize=(4,4), dpi=150); state['pcs_figure']=pcs_figure
    pcs_canvas = FigureCanvasTkAgg(pcs_figure, master=left_frame); pcs_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    state['pcs_canvas'] = pcs_canvas

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
        table_btns, text="Export δ_Exp Template",
        command=lambda: export_delta_exp_template(state)
    ).pack(side=tk.LEFT, padx=(0, 6))
    ttk.Button(
        table_btns, text="Import δ_Exp",
        command=lambda: import_delta_exp_file(state, plot_cartesian_graph)
    ).pack(side=tk.LEFT)
    ttk.Button(
        table_btns, text="Paste δ_Exp",
               command=lambda: import_delta_exp_from_clipboard(state, plot_cartesian_graph)).pack(side=tk.LEFT, padx=6)
    ttk.Button(
        table_btns, text="Undo",
               command=lambda: undo_last_delta_import(state, plot_cartesian_graph)).pack(side=tk.LEFT)
    ttk.Button(
        table_btns, text="Clear δ_Exp",
        command=lambda: clear_delta_exp(state, plot_cartesian_graph)
    ).pack(side=tk.LEFT, padx=6)

    _sep(center_frame, orient='horizontal', pady=6, fill='x')

    # --- tab ---
    # Plots notebook
    plots_nb = ttk.Notebook(center_frame)
    plots_nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=3, pady=0)
    state['plots_nb'] = plots_nb

    # --- Cartesian tab UI ---
    cartesian_tab = ttk.Frame(plots_nb)
    plots_nb.add(cartesian_tab, text="Plot")

    cartesian_figure = plt.Figure(figsize=(4, 3), dpi=100);
    state['cartesian_figure'] = cartesian_figure
    cartesian_canvas = FigureCanvasTkAgg(cartesian_figure, master=cartesian_tab)
    cartesian_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    state['cartesian_canvas'] = cartesian_canvas

    # --- Rhombicity tab UI ---
    rhtab = ttk.Frame(plots_nb, width=560)  # Fixed width
    plots_nb.add(rhtab, text="Rhombicity")

    # 탭 프레임이 자식의 reqsize에 끌려가지 않게
    rhtab.grid_propagate(False)

    # rhtab 내부는 grid로만 배치
    rhtab.rowconfigure(0, weight=0)  # top bar
    rhtab.rowconfigure(1, weight=0)  # z-rotation bar (NEW)
    rhtab.rowconfigure(2, weight=1)  # table area (CHANGED: was row 1)
    rhtab.columnconfigure(0, weight=1)

    # 상단 버튼 영역
    rh_top = ttk.Frame(rhtab)
    rh_top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)

    # Δχ_rh 입력 (단위: 1e-32 m^3)
    ttk.Label(rh_top, text="Δχ_rh values (E-32 m³):").pack(side="left", padx=(0, 6))

    # state 기본값
    state.setdefault("rh_dchi_rh", 0.0)

    # Entry 변수
    state["rh_dchi_rh_var"] = tk.StringVar(value=f"{state.get('rh_dchi_rh', 0.0):g}")
    rh_dchi_entry = ttk.Entry(rh_top, textvariable=state["rh_dchi_rh_var"], width=10)
    rh_dchi_entry.pack(side="left")

    # Rhombicity 탭 table
    cols = ("Ref", "Atom", "r", "theta(deg)", "phi(deg)",
            "Gi_ax", "Gi_rh", "δ_PCS(ax)", "δ_PCS(ax+rh)", "δ_Exp", "res(ax)", "res(ax+rh)")

    # ---------------------------
    # NEW: Z-rotation row (between top and table)
    # ---------------------------
    rh_zrow = ttk.Frame(rhtab)
    rh_zrow.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))

    ttk.Label(rh_zrow, text="Rotate around Z-axis (degrees):").pack(side="left", padx=(0, 6))

    # z state vars
    state.setdefault("angle_z_var", tk.DoubleVar(value=0.0))
    # NOTE: on_angle_slider / on_angle_entry_commit must support axis 'z'
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
    # table frame도 grid로 (CHANGED row: 1 -> 2)
    # ---------------------------
    rh_table_frame = ttk.Frame(rhtab)
    rh_table_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))

    rh_tree = ttk.Treeview(rh_table_frame, columns=cols, show="headings", height=18)

    # 스크롤바 (세로/가로)
    rh_ys = ttk.Scrollbar(rh_table_frame, orient="vertical", command=rh_tree.yview)
    rh_xs = ttk.Scrollbar(rh_table_frame, orient="horizontal", command=rh_tree.xview)
    rh_tree.configure(yscrollcommand=rh_ys.set, xscrollcommand=rh_xs.set)

    # 컬럼 widths (전부 stretch=False)
    widths = (40, 50, 50, 80, 80, 80, 80, 80, 105, 60, 60, 80)
    for c, w in zip(cols, widths):
        rh_tree.heading(c, text=c)
        rh_tree.column(c, width=w, minwidth=40, anchor="center", stretch=False)

    # grid 배치(테이블 프레임 내부)
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

    # Enter로 Apply
    rh_dchi_entry.bind("<Return>", lambda e: _apply_dchi_rh())

    # Apply 버튼
    ttk.Button(rh_top, text="Update", command=_apply_dchi_rh).pack(side="left", padx=(6, 12))

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
        if state.get("rh_calc_enabled", False):
            calculate_tensor_components_ui_ax_rh(
                chi_mol_entry=state["chi_mol_entry"],  # entry
                molar_value_label=state["molar_value_label"],
                rh_dchi_entry=rh_dchi_entry,  # Rh tab entry
                tensor_xx_label=state["tensor_xx_label"],
                tensor_yy_label=state["tensor_yy_label"],
                tensor_zz_label=state["tensor_zz_label"],
                messagebox=state["messagebox"],
            )
        else:
            calculate_tensor_components_ui(
                state["chi_mol_entry"],
                state["molar_value_label"],
                state["tensor_xx_label"],
                state["tensor_yy_label"],
                state["tensor_zz_label"],
                state["messagebox"],
            )

    ttk.Button(rh_top, text="Calc χ(xx,yy,zz)", command=_calc_chi_tensor_from_ui) \
        .pack(side="left", padx=(4, 0))

    # --- Fitting tab UI ---
    fittab = ttk.Frame(plots_nb)
    plots_nb.add(fittab, text="Fitting")

    # left(Settings), right(Protons)
    fit_top = ttk.Frame(fittab)
    fit_top.pack(fill=tk.BOTH, expand=True)

    settings_col = ttk.Frame(fit_top)  # left
    settings_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

    protons_col = ttk.Frame(fit_top)  # right
    protons_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Mode selection (A/B) : inside settings_col
    mode_row = ttk.Frame(settings_col)
    mode_row.pack(fill=tk.X, pady=4)
    state['fit_mode_var'] = tk.StringVar(value='theta_alpha_multi')
    tk.Radiobutton(mode_row, text="[Mode A] θ,α (multi-donor)",
                   variable=state['fit_mode_var'], value='theta_alpha_multi',
                   command=lambda: switch_fit_mode(state),
                   bg="#F5F6FA",
                   activebackground="#F5F6FA",
                   highlightthickness=0,
                   relief="flat"
                   ).pack(side=tk.LEFT)
    tk.Radiobutton(mode_row, text="[Mode B] 3D Euler fitting",
                   variable=state['fit_mode_var'], value='euler_global',
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
    frameA = ttk.LabelFrame(settings_col, text="Settings", width=360, height=230)
    state['fit_frameA'] = frameA
    frameA.pack_propagate(False)
    frameA.pack(fill=tk.X, pady=4, before=state['fit_anchor'])

    labelA = ttk.Label(
        frameA,
        text=("Donor atoms list : define the ligand donor atoms and decide "
              "how to establish the vector used in the fitting."),
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
    frameB = ttk.LabelFrame(settings_col, text="Settings", width=360, height=230)
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
                   bg="#F5F6FA",
                   activebackground="#F5F6FA",
                   highlightthickness=0,
                   relief="flat"
                   ).pack(side=tk.LEFT)
    state['fit_dchi_var'] = tk.BooleanVar(value=True)
    tk.Checkbutton(opts, text="Fit Δχ_ax together",
                   variable=state['fit_dchi_var'],
                   bg="#F5F6FA",
                   activebackground="#F5F6FA",
                   highlightthickness=0,
                   relief="flat"
                   ).pack(side=tk.LEFT, padx=12)

    btns = ttk.Frame(fittab);
    btns.pack(fill=tk.X, pady=4)
    ttk.Button(btns, text="Refresh lists",
               command=lambda: populate_fitting_controls(state)).pack(side=tk.LEFT)
    ttk.Button(btns, text="Run fit",
               command=lambda: run_fit_from_ui(state)).pack(side=tk.LEFT, padx=6)
    ttk.Button(btns, text="Apply to plot",
               command=lambda: apply_fit_to_views(state)).pack(side=tk.RIGHT)

    state['fit_result_box'] = tk.Text(fittab, height=10)
    state['fit_result_box'].pack(fill=tk.BOTH, expand=True, pady=4)

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
                   command=lambda: [update_figsize(state), state['update_graph']()],
                   bg="#F5F6FA",
                   activebackground="#F5F6FA",
                   highlightthickness=0,
                   relief="flat"
                   ).pack(side=tk.LEFT)

    # Update/Reset
    bf = ttk.Frame(input_frame); bf.pack(pady=3)
    ttk.Button(bf, text="Update", command=lambda: state['update_graph']()).pack(side=tk.LEFT, padx=2)
    ttk.Button(bf, text="Reset", command=lambda: reset_values(state)).pack(side=tk.LEFT, padx=2)

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
    # 기본값: 자동 계산 ON
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

    def _maybe_calc_chi(_event=None):
        """Auto-calc가 켜져 있을 때만 χ_xx/yy/zz 계산."""
        if not state["chi_auto_calc_var"].get():
            return
        calculate_tensor_components_ui(
            state['chi_mol_entry'],
            state['molar_value_label'],
            state['tensor_xx_label'],
            state['tensor_yy_label'],
            state['tensor_zz_label'],
            state['messagebox'],
        )

    # Enter / FocusOut 에서 자동 계산 (토글로 on/off)
    chi_mol_entry.bind('<Return>', _maybe_calc_chi)
    chi_mol_entry.bind('<FocusOut>', _maybe_calc_chi)

    _sep(input_frame)

    # File load
    ttk.Button(input_frame,
               text="Load xyz File",
               command=lambda: load_xyz_file(state)
               ).pack(anchor="center", pady=3)

    ttk.Label(
        input_frame,
        text=("The coordinates should align\n"
              "the molecule's rotational axis\n"
              "with the z-axis for proper analysis."),
        font=("Helvetica", 8, "italic"),
        justify="center",
        anchor="center"
    ).pack(pady=0, anchor="center")

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

    # 3D / projection buttons
    opf = ttk.Frame(input_frame); opf.pack(fill=tk.X, padx=0, pady=3)
    ttk.Label(opf, text="Open 3d structure/projection", font=("default",9,"bold")).pack(pady=3)
    ttk.Button(opf, text="mol structure", command=lambda: open_3d_plot_window(state)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
    ttk.Button(
        opf, text="Projection",
        command=lambda: open_mollweide_plot(
            state['atom_data'],
            (state['x0'], state['y0'], state['z0']),
            float(state['angle_x_var'].get()),
            float(state['angle_y_var'].get()),
            0.0,  # non z-rotation
            state['root'],
            state['FigureCanvas']
        )
    ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

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
    ttk.Button(sbf, text="Save plot",  command=lambda: on_save_plot_any(state)).grid( row=1, column=1, sticky="ew", padx=2, pady=1)
    ttk.Button(sbf, text="Save table", command=lambda: on_save_table_any(state)).grid(row=1, column=2, sticky="ew", padx=2, pady=1)

    # Defaults
    state['delta_values'] = {}
    reset_values(state)  # sets defaults and draws initial plots
    # Bindings
    tree.bind("<Double-1>", lambda e: on_delta_entry_change(state, e, state['delta_values'], plot_cartesian_graph))
    tree.bind("<<TreeviewSelect>>", lambda e: None)  # placeholder; selection highlight could be added

    return state

def switch_fit_mode(state):
    m = state['fit_mode_var'].get()
    # 일단 둘 다 빼고
    state['fit_frameA'].pack_forget()
    state['fit_frameB'].pack_forget()

    anchor = state['fit_anchor']
    if m == 'theta_alpha_multi':
        state['fit_frameA'].pack(fill=state['tk'].X, pady=4, before=anchor)
    else:
        state['fit_frameB'].pack(fill=state['tk'].X, pady=4, before=anchor)

    try:
        populate_fitting_controls(state)
    except Exception:
        pass

def run_fit_from_ui(state):
    mode = state['fit_mode_var'].get()

    # general section - protons selection
    psel = state['fit_proton_list'].curselection()
    if not psel:
        state['messagebox'].showerror("Fitting", "Select ≥3 protons (Ref IDs)."); return
    proton_ids = [int(state['fit_proton_list'].get(i)) for i in psel]

    try:
        if mode == 'theta_alpha_multi':
            dsel = state['fit_donor_list'].curselection()
            if not dsel:
                state['messagebox'].showerror("Fitting", "Select ≥1 donor IDs."); return
            donor_ids = [int(state['fit_donor_list'].get(i).split(':')[0]) for i in dsel]

            res = fit_theta_alpha_multi(
                state,
                donor_ids=donor_ids,
                proton_ids=proton_ids,
                axis_mode=state['axis_mode_var'].get(),
                fit_visible_as_group=state['fit_use_visible_var'].get(),
                fit_delta_chi=state['fit_dchi_var'].get()
            )
        else:
            res = fit_euler_global(
                state,
                proton_ids=proton_ids,
                fit_visible_as_group=state['fit_use_visible_var'].get(),
                fit_delta_chi=state['fit_dchi_var'].get()
            )

        state['last_fit_result'] = res
        _show_fit_result(state, res)

    except Exception as e:
        state['messagebox'].showerror("Fitting", str(e))

def _show_fit_result(state, res):
    box = state['fit_result_box']; box.delete("1.0", "end")
    if res.get('mode') == 'theta_alpha_multi':
        box.insert("end", f"[Mode A] axis={res.get('axis_mode')}\n")
        box.insert("end", f"Donors: {res['donor_ids']}\n")
        box.insert("end", f"θ*={res['theta']:.2f}°, α*={res['alpha']:.2f}°\n")
    else:
        box.insert("end", "[Mode B] Euler global\n")
        box.insert("end", f"ax={res['ax']:.2f}°, ay={res['ay']:.2f}°, az={res['az']:.2f}°\n")
    box.insert("end", f"Δχ_ax={res['delta_chi_ax']:.3e} (E-32 m³)\n")
    box.insert("end", f"RMSD={res['rmsd']:.3f} ppm  (N={res['n']})\n\n")
    box.insert("end", "Ref\tδ_exp\tδ_pred\tresid\n")
    for rid, exp, pred, r in res['per_point']:
        box.insert("end", f"{rid}\t{exp:.3f}\t{pred:.3f}\t{r:+.3f}\n")

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
        state['pcs_canvas'].print_figure(fd, dpi=600)
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
        update_table(state, polar_data, rotated_coords, tensor, state['delta_values'])
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

    if 'fit_proton_list' in state:
        try:
            populate_fitting_controls(state)
        except Exception:
            pass

def reset_values(state):
    state['tensor_entry'].delete(0, tk.END); state['tensor_entry'].insert(0, '-2.0')
    state['pcs_min_entry'].delete(0, tk.END); state['pcs_min_entry'].insert(0, '-10')
    state['pcs_max_entry'].delete(0, tk.END); state['pcs_max_entry'].insert(0, '10')
    state['pcs_interval_entry'].delete(0, tk.END); state['pcs_interval_entry'].insert(0, '0.5')
    state['angle_x_var'].set(0); state['angle_y_var'].set(0)

    state.get('delta_values', {}).clear()
    state.pop('last_rotated_coords', None)

    state['atom_data']=[]; state['check_vars']={}
    state['pcs_values']=__import__('numpy').arange(-10,10.5,0.5)
    for w in state['checklist_frame'].winfo_children(): w.destroy()
    for r in state['tree'].get_children(): state['tree'].delete(r)
    if 'theta_values' not in state:
        state['theta_values']=__import__('numpy').linspace(0, 2*__import__("numpy").pi, 500)
    update_graph(state)

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

def load_xyz_file(state):
    path = state['filedialog'].askopenfilename(filetypes=[("XYZ files","*.xyz")])
    if not path:
        return

    # reset
    state.get('delta_values', {}).clear()
    state.pop('last_rotated_coords', None)
    state.pop('row_by_id', None)
    state.pop('current_selected_ids', None)

    atom_data = parse_xyz(path)
    state['atom_data'] = atom_data

    # Ref(ID): atom id fix (1..N)
    state['atom_ids'] = list(range(1, len(atom_data) + 1))

    target_atom = state['simpledialog'].askstring("Input center atom", "Enter the center atom (element) :")
    tgt = None
    for atom, x, y, z in atom_data:
        if atom == target_atom:
            tgt = (x, y, z)
            break
    if tgt is None:
        state['messagebox'].showerror("Error", f"Atom {target_atom} not found in the file.")
        return
    state['x0'], state['y0'], state['z0'] = tgt

    create_checklist(state)
    update_graph(state)
    populate_fitting_controls(state)

    # open 3d plot after xyz file loading
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

    # original coord
    abs_coords = np.array([[x, y, z] for a, x, y, z in state['atom_data']])
    metal = np.array([state['x0'], state['y0'], state['z0']])

    # Fit override 있으면 donor 축 기준 회전 먼저 적용
    fo = state.get('fit_override')
    if fo:
        ids = state.get('atom_ids', [])
        id2idx = {rid: i for i, rid in enumerate(ids)}
        donor_ids = fo.get('donor_ids') or []
        if donor_ids:
            donor_pts = [abs_coords[id2idx[rid]] for rid in donor_ids]
            abs_coords = _angles_to_rotation_multi(
                points=abs_coords,
                metal=metal,
                donor_points=donor_pts,
                theta_deg=fo.get('theta', 0.0),
                alpha_deg=fo.get('alpha', 0.0),
                axis_mode=fo.get('axis_mode', 'bisector')
            )

    # 중심 기준으로 이동
    coords0 = abs_coords - metal

    # 슬라이더 회전(원점 기준 Euler)
    ax = float(state['angle_x_var'].get())
    ay = float(state['angle_y_var'].get())
    az = float(state['angle_z_var'].get())
    rotated = rotate_coordinates(coords0, ax, ay, az, (0, 0, 0))

    polar = []
    rotated_sel = []
    selected_ids = []

    ids = state.get('atom_ids', [])
    for idx, ((atom, *_), (dx, dy, dz)) in enumerate(zip(state['atom_data'], rotated)):
        if atom in sel:
            r = (dx*dx + dy*dy + dz*dz) ** 0.5
            theta = np.arccos(dz / r) if r != 0 else 0.0
            polar.append((atom, r, theta))
            rotated_sel.append((dx, dy, dz))
            selected_ids.append(ids[idx])

    state['current_selected_ids'] = selected_ids
    return polar, rotated_sel

def wire(state):
    state['update_graph'] = lambda : update_graph(state)
    state['plot_cartesian'] = plot_cartesian_graph
    state['filter_atoms'] = filter_atoms
    state['update_table'] = update_table

    return state
