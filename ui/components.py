from ui.theme_palette import workspace_palette, theme_id
from ui.parameter_history import (parameter_edit, attach_parameter_gesture,
                                  commit_parameter_update, unchanged_parameter_input)
# ui/components.py

from ui.widget_style import ui_text_color, ui_label_style, ui_font_name
from ui.style import get_app_fonts
import subprocess, sys, os
import re
from dataclasses import replace
from pathlib import Path
import tkinter as tk
import tkinter.ttk as ttk
import threading
from ui.style import apply_style
from ui.icons import IconManager, TAB_ICON_SIZE, TOOLBAR_ICON_SIZE, BUTTON_ICON_SIZE
from tkinter import filedialog, simpledialog, messagebox
import numpy as np
from scipy.constants import Avogadro as AVOGADRO_CONSTANT
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from logic.command_processor import process_command as _pc
from logic.xyz_loader import load_structure
from logic.func_group_collapse import collapse_methyl_groups, collapse_cf3_groups
from logic.torsional_ensemble import average_coordinates_independent

from logic.plot_pcs import plot_graph
from logic.plot_cartesian import (
    plot_cartesian_graph, export_cartesian_plot, start_lasso_selection, start_rectangle_selection,
    fit_selected_points, clear_cartesian_selection,
)
from logic.table_utils import (
    update_molar_value, update_table, on_delta_entry_change, calculate_tensor_components_ui, calculate_tensor_components_ui_ax_rh,
    export_delta_exp_template, import_delta_exp_file, import_delta_exp_from_clipboard, undo_last_delta_import, clear_delta_exp, classify_residual_tag
)
from logic.rotate_align import rotate_coordinates, rotate_euler
from logic.chem_constants import CPK_COLORS
from logic.visibility import (
    ELEMENT_SELECTION_KEY,
    LIGAND_SELECTION_KEY,
    ligand_visibility_records,
    sync_visibility_state_from_vars,
    visible_ref_id_set,
)
from logic.fitting import (
    populate_fitting_controls,
    fit_theta_alpha_multi, fit_euler_global, fit_full_tensor, _angles_to_rotation_multi
)
from logic.include_rhombic import build_rh_table_rows
from logic.diagnostic import update_diagnostic_panel

from ui.plot_3d_window import open_3d_plot_window
from logic.nmr_delta_data_manager import push_layers_to_nmr_if_open

from ui.projection_window import open_projection_window
from ui.pcs_plot_window import open_pcs_plot_popup
from ui.chemical_map_window import open_chemical_map_window
from logic.selection_context import (
    apply_context_to_tree,
    ensure_selection_hub,
    publish_tree_selection,
)
from logic.pcs_model_state import (
    initialize_pcs_model_state,
    sync_canonical_tensor_from_legacy,
    update_canonical_tensor,
)
from logic.pcs_prediction_model import (
    PredictionModel,
    get_prediction_model,
    install_prediction_model,
)
from logic.pcs_tensor_state import get_canonical_tensor_state
from logic.pcs_forward_api import AngleUnit, EulerConvention
from logic.pcs_orientation import matrix_to_euler, orientation_matrix
from logic.pcs_events import PCSDispatchError
from logic.pcs_fit_result_apply import (
    FitResultApplicationError,
    apply_fit_result_to_canonical_tensor,
    fit_result_application_status,
)
from logic.pcs_fit_result_schema import PCS_FIT_RESULT_SCHEMA, parameter_status
from logic.pcs_fit_weighting import (
    FitWeightingMode,
    fit_weighting_label,
    parse_fit_weighting_mode,
)
from logic.pcs_fit_history import (
    append_fit_history_result,
    clear_fit_history,
    compare_fit_results,
    delete_fit_history_entry,
    find_fit_history_entry,
    fit_history_entries,
    label_fit_history_entry,
    select_fit_history_result,
)
from logic.pcs_fit_history_export import (
    build_fit_history_export,
    build_fit_run_export,
    default_fit_history_basename,
    write_fit_observables_csv,
    write_fit_summary_csv,
    write_json_export,
)
from logic.pcs_fit_history_import import (
    FitRunImportError,
    import_fit_run_export_file,
)
from logic.pcs_fit_result_status import (
    FIT_RESULT_CURRENT,
    evaluate_fit_result_status,
)
from logic.pcs_refresh import (
    subscribe_pcs_refresh_consumers,
    unsubscribe_pcs_refresh_consumers,
)
from logic.pcs_document import (
    active_document,
    active_document_view,
    apply_worker_document_updates,
    capture_active_document,
    capture_worker_identity,
    install_single_document,
    mark_document_mutated,
    worker_completion_matches,
    worker_document_state,
)

from ui.advanced_fitting_tab import build_advanced_fitting_tab
from ui.conformer_search import run_conformer_search_gui
from ui.conformer_presentation import ConformerSearchPresentationAdapter
from ui.conformer_commands import (
    refresh_conformer_commands,
    register_conformer_commands,
)
from ui.scrollable_workspace import VerticalScrolledWorkspace

from ui.plot_3d_pyvista import open_pyvista_field

#settings
from ui.settings_window import open_settings_window
from ui.averaging_settings_window import open_averaging_settings_window  # PCS_PATCH_AVERAGING_SETTINGS_PHASE1
from logic.app_settings import load_app_state, save_app_state, is_reasonable_main_geometry
from ui.about_window import open_about_window
from ui.update_window import check_for_updates_ui
from logic.workspace_io import load_workspace, save_workspace
from ui.drag_drop import enable_main_window_dnd


def _sync_legacy_pcs_tensor(state, *, source):
    """Publish model changes without letting plugin listeners break redraws."""
    try:
        sync_canonical_tensor_from_legacy(state, source=source)
    except PCSDispatchError as exc:
        state["_pcs_model_event_dispatch_failures"] = exc.failures

#plugins
from plugin_system.plugin_api import PluginApp
from plugin_system.plugin_loader import load_enabled_plugins
from ui.module_manager_window import open_module_manager_window
from ui.control_panel import ScrollableControlPanel
from ui.control_panel_style import (
    ControlPanelElementChip,
    ControlPanelForm,
    ControlPanelSlider,
    control_panel_palette,
)
from ui.customize_tools_window import open_customize_tools_window
from ui.ribbon import ContextRibbonController, RibbonBar, RibbonIconService
from ui.commands import CommandRegistry
from ui.document_tabs import DocumentTabStrip
from ui.main_window_commands import register_main_window_commands
from ui.menu_presenter import MenuCommandPresenter
from ui.fitting_commands import refresh_fitting_commands, register_fitting_commands
from ui.gax_rhombicity_commands import (
    refresh_gax_rhombicity_commands,
    register_gax_rhombicity_commands,
)
from ui.pcs_data_commands import (
    refresh_pcs_data_commands,
    register_pcs_data_commands,
)
from ui.tooltips import ToolTip
from plugin_system.tool_registry import ToolLauncher, ensure_tool_registry

def get_cpk_color(atom):
    return CPK_COLORS.get(atom, CPK_COLORS['default'])

def _sep(parent, orient='horizontal', pady=8, fill='x'):
    s = ttk.Separator(parent, orient=orient)
    s.pack(fill=fill, pady=pady)
    return s

def _icon(state, name, size=BUTTON_ICON_SIZE):
    manager = state.get("icons") if isinstance(state, dict) else None
    if manager is None:
        return ""
    try:
        return manager.get(name, size=size)
    except Exception:
        return ""

# slider theme related helper
def _apply_scale_theme(scale_widget, state):
    try:
        palette = control_panel_palette(scale_widget)
    except Exception:
        palette = None

    # The refined ControlPanelSlider owns its thumb/track rendering. Keep the
    # legacy tk.Scale path for any remaining callers outside the inspector.
    if hasattr(scale_widget, "apply_theme"):
        try:
            scale_widget.apply_theme(palette)
        except Exception:
            pass
        return

    try:
        app_bg = (palette or {}).get(
            "body", getattr(state["root"], "_app_bg", "#F5F6FA")
        )
        app_fg = getattr(state["root"], "_app_fg", "#111111")
        trough = (palette or {}).get("slider_track", "#D0D7E0")
        thumb_active = (palette or {}).get("slider_thumb_hover", "#657789")
        scale_widget.configure(
            bg=app_bg,
            fg=app_fg,
            activebackground=thumb_active,
            troughcolor=trough,
            highlightbackground=app_bg,
            highlightcolor=app_bg,
            highlightthickness=0,
            borderwidth=0,
            sliderrelief="raised",
        )
    except Exception:
        pass

# app setting helper - dpi
def _get_export_dpi(state, fallback=600):
    cfg = state.get("app_settings", {}) or {}
    try:
        dpi = int(cfg.get("export_default_dpi", fallback))
        if dpi > 0:
            return dpi
    except Exception:
        pass
    return fallback

# Backward-compatible alias retained for old callers/plugins.
# The legacy NMRSpectrumWindow is no longer launched by the main application.
def open_nmr_window(state):
    return open_nmr_workbench(state)

# Cartesian plot helper
def _on_tree_select_update_cartesian(state):
    """Refresh Cartesian plot highlight when the main table selection changes."""
    try:
        plot_cartesian_graph(state)
    except Exception:
        pass

# conformer search helper
def _refresh_conformer_command_state(state):
    refresh = state.get("refresh_conformer_commands")
    if callable(refresh):
        refresh()


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

    # Legacy main controls are active extrinsic XYZ. The search retains its
    # intrinsic ZYZ interface; convert the representation, never the geometry.
    try:
        euler_deg = (
            float(state["angle_x_var"].get()) if "angle_x_var" in state else 0.0,
            float(state["angle_y_var"].get()) if "angle_y_var" in state else 0.0,
            float(state["angle_z_var"].get()) if "angle_z_var" in state else 0.0,
        )
    except Exception:
        euler_deg = (0.0, 0.0, 0.0)
    from logic.pcs_orientation import convert_euler
    euler_deg = convert_euler(
        euler_deg, EulerConvention.EXTRINSIC_XYZ,
        EulerConvention.INTRINSIC_ZYZ, AngleUnit.DEGREE,
    )

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
        project_adapter = state.get("conformer_search_project_adapter")
        if project_adapter is not None:
            project_adapter.reset_for_sync()
        cs_app.initial_data = new_initial_data
        cs_app._prefill_from_initial_data()
        cs_app._auto_detect_if_ready()
        try:
            state["plots_nb"].select(state["conformer_tab"])
        except Exception:
            pass
        state["conformer_status_var"].set(
            "Update complete — Check Setup tab and run search")
        _refresh_conformer_command_state(state)
        return

    # fallback: embedded app 없을 때만 Toplevel로 열기
    run_conformer_search_gui(
        master=state["root"],
        on_preview_result=lambda payload: _conformer_preview_callback(state, payload),
        initial_data=new_initial_data,
    )

def _conformer_preview_callback(state, payload):
    """Shared result-receipt callback for both embedded and Toplevel modes."""
    identity = payload.get("_document_worker_identity")
    stored_payload = dict(payload)
    stored_payload.pop("_document_worker_identity", None)
    manager = state.get("document_manager")
    origin = (
        manager.document_by_id(identity.document_id)
        if manager is not None and identity is not None
        else None
    )
    updates = {
        "conformer_preview": stored_payload,
        "conformer_preview_coords": payload["coords_opt"],
        "conformer_preview_elements": payload["elements"],
        "conformer_preview_report": payload["report"],
        "conformer_applied": False,
        "last_conformer_result": payload.get("result"),
    }
    if payload.get("search_project") is not None:
        updates["conformer_search_session"] = payload["search_project"]
    if origin is not None and origin is not active_document(state):
        apply_worker_document_updates(
            state, identity, updates, domain="conformer_search_result"
        )
        return
    state.update(updates)
    mark_document_mutated(state, "conformer_search_result")

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
            " — click 'Apply'"
        )
    except Exception:
        pass
    _refresh_conformer_command_state(state)

    try:
        from ui.plot_3d_window import _draw_3d_plot
        _draw_3d_plot(state)
    except Exception:
        pass

def apply_conformer_preview(state):
    owner = state.get("conformer_search_project_adapter")
    if owner is not None and not owner.can_apply():
        state["messagebox"].showwarning("Conformer Preview", "Search inputs changed. Rerun the search before applying.")
        return
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

    from logic.document_edit_history import STRUCTURE_KEYS, begin_edit, finish_edit
    edit_token = begin_edit(state, STRUCTURE_KEYS)
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

    finish_edit(state, edit_token, "Apply conformer")
    mark_document_mutated(state, "conformer_apply")
    capture_search = state.get("capture_conformer_search_state")
    if callable(capture_search):
        capture_search()
    state["messagebox"].showinfo("Conformer Preview", "Preview coordinates applied.")
    _refresh_conformer_command_state(state)

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
    _refresh_conformer_command_state(state)

def revert_conformer_to_original(state):
    raw = state.get("atom_data_original")
    if not raw:
        state["messagebox"].showwarning("Conformer Preview", "No original structure stored.")
        return

    from logic.document_edit_history import STRUCTURE_KEYS, begin_edit, finish_edit
    edit_token = begin_edit(state, STRUCTURE_KEYS)
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

    finish_edit(state, edit_token, "Revert conformer to original")
    mark_document_mutated(state, "conformer_revert")
    capture_search = state.get("capture_conformer_search_state")
    if callable(capture_search):
        capture_search()
    state["messagebox"].showinfo("Conformer Preview", "Reverted to original structure.")
    _refresh_conformer_command_state(state)

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
    try:
        publish_tree_selection(state, source="pcs_table")
    except Exception:
        pass
    _on_tree_select_update_spectrum(state)
    _on_tree_select_update_3d(state)
    _on_tree_select_update_pcs(state)
    _on_tree_select_update_cartesian(state)

def open_pcs_workbench(state):
    from tkinter import messagebox

    try:
        from tools.pcs_workbench import AppWindow
    except Exception as e:
        messagebox.showerror(
            "PCS Workbench",
            f"Could not import PCS Workbench:\n{e}"
        )
        return

    win = state.get("pcs_workbench_window")
    if win is not None:
        try:
            if win.winfo_exists():
                win.lift()
                win.focus_force()
                return
        except Exception:
            pass

    try:
        from ui.orca_tensor_transfer import review_tensor_transfer
        win = AppWindow(master=state["root"],
                        on_send_tensor_callback=lambda payload: review_tensor_transfer(
                            state, payload, state.get("pcs_workbench_window") or state["root"]))
        state["pcs_workbench_window"] = win

        def _on_close():
            try:
                win._on_quit()
            except Exception:
                try:
                    win.destroy()
                except Exception:
                    pass
            finally:
                state["pcs_workbench_window"] = None

        win.protocol("WM_DELETE_WINDOW", _on_close)
        win.lift()
        win.focus_force()

    except Exception as e:
        messagebox.showerror(
            "PCS Workbench",
            f"Could not open PCS Workbench:\n{e}"
        )

def open_nmr_workbench(state):
    """Launch the first-class internal NMR Workbench module."""
    from nmr_workbench.integration import open_nmr_workbench as launch

    launch(state)


def _register_builtin_tools(state):
    """Install stable launcher metadata without changing the Modules menu."""
    registry = ensure_tool_registry(state)
    tools = (
        ToolLauncher("builtin.pcs_plot_2d", "2D PCS Plot", "PCS views", lambda: open_pcs_plot_popup(state), order=10),
        ToolLauncher("builtin.projection", "Projection", "PCS views", lambda: open_projection_window(state), order=20),
        ToolLauncher("builtin.structure_3d", "3D Structure", "Structure", lambda: open_3d_plot_window(state), order=10),
        ToolLauncher("builtin.pcs_field_3d", "3D PCS Plot", "PCS views", lambda: open_pyvista_field(state), order=30),
        ToolLauncher("builtin.nmr_workbench", "NMR Workbench", "Workbenches", lambda: open_nmr_workbench(state), order=10),
        ToolLauncher("builtin.pcs_workbench", "PCS Workbench", "Workbenches", lambda: open_pcs_workbench(state), order=20),
        ToolLauncher("builtin.chemical_map", "Chemical Map", "Structure", lambda: open_chemical_map_window(state), order=20),
    )
    for tool in tools:
        registry.register(tool)

def refresh_plugin_menu(state):
    """
    Built-in module entries such as Module Manager and the workbenches are kept.
    External plugin menu items are removed and reloaded from plugins.json.
    """
    modules_menu = state.get("menus", {}).get("Modules")
    if modules_menu is None:
        return

    start_index = state.get("plugin_menu_start_index")
    if start_index is None:
        return

    # Remove old external plugin menu entries
    try:
        end_index = modules_menu.index("end")
        if end_index is not None and end_index >= start_index:
            modules_menu.delete(start_index, "end")
    except Exception:
        pass

    plugin_app = PluginApp(state)
    state["plugin_app"] = plugin_app
    ensure_tool_registry(state).unregister_plugins()
    state["_module_ribbon_registrations"] = []

    try:
        results = load_enabled_plugins(plugin_app)
        state["plugin_load_results"] = results

        # Console diagnostics
        for r in results:
            if not r.get("ok"):
                print(
                    "[Plugin loading failed]",
                    r.get("plugin_id"),
                    r.get("error", "")
                )

    except Exception as e:
        state["plugin_load_results"] = []
        try:
            state["messagebox"].showwarning(
                "Plugin loading",
                f"Plugins could not be loaded:\n{e}"
            )
        except Exception:
            pass
    refresh_tools = state.get("refresh_control_panel_tools")
    if callable(refresh_tools):
        refresh_tools()
    refresh_ribbon = state.get("refresh_ribbon_modules")
    if callable(refresh_ribbon):
        refresh_ribbon()

def build_app():
    state = {}
    # M1 single-document ownership boundary.  The legacy state and widget tree
    # remain stable; one explicit document captures their document-local data.
    install_single_document(state)
    from ui.document_integrations import refresh_document_integration_titles

    state["refresh_document_integration_titles"] = (
        lambda: refresh_document_integration_titles(state)
    )
    # Shared presentation-command infrastructure. Production callbacks remain
    # on their existing menus and controls until later Ribbon milestones.
    state["command_registry"] = CommandRegistry()
    # initialize: settings
    persisted = load_app_state()

    state["settings_window"] = None
    state["about_window"] = None
    state["update_window"] = None
    state["recent_files"] = list(persisted.get("recent_files", []))
    state["app_settings"] = dict(persisted.get("app_settings", {}))
    state["rebuild_recent_files_menu"] = lambda: rebuild_recent_files_menu(state)
    state["refresh_checklist_ui"] = lambda: create_checklist(state)
    state["apply_scale_theme"] = lambda w: _apply_scale_theme(w, state)
    # Project I/O callbacks used by logic.project_io during .pcsp loading.
    state["create_checklist"] = create_checklist
    state["apply_symavg_to_state"] = apply_symavg_to_state
    state["populate_fitting_controls"] = populate_fitting_controls

    # Tk and common modules
    state['tk'] = tk; state['ttk'] = ttk
    # common state for (delta_EXP, delta_PCS) residual
    state["residual_thr_ok"] = 0.10     #ppm
    state["residual_thr_warn"] = 0.30   #ppm

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
    #root = tk.Tk(); root.title("PCS Analyzer"); root.geometry("1320x930"); state['root'] = root
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
        state["dnd_available"] = True
    except Exception:
        root = tk.Tk()
        state["dnd_available"] = False

    root.title("PCS Analyzer")
    state['root'] = root
    state["icons"] = IconManager(root)
    state["save_app_preferences"] = lambda: save_app_state(state, save_geometry=False)
    _register_builtin_tools(state)

    saved_geo = persisted.get("main_window_geometry")
    default_geo = "1320x930"

    if (
        state["app_settings"].get("remember_window_geometry", True)
        and saved_geo
        and is_reasonable_main_geometry(saved_geo)
    ):
        try:
            root.geometry(saved_geo)
        except Exception:
            root.geometry(default_geo)
    else:
        root.geometry(default_geo)

    #apply_style(root, variant="light", accent="green")  # darkmode : variant="dark"
    apply_style(
        root,
        variant=state["app_settings"]["theme_variant"],
        accent=state["app_settings"]["theme_accent"],
        settings=state["app_settings"],
    )

    def _ask_save_discard_cancel(document):
        result = {"action": "cancel"}
        dialog = tk.Toplevel(root)
        dialog.withdraw()
        dialog.title("Unsaved PCS document")
        dialog.transient(root)
        dialog.resizable(False, False)
        body = ttk.Frame(dialog, padding=(16, 14, 16, 10))
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            body,
            text=f"Save changes to {document.title}?",
        ).pack(anchor=tk.W, pady=(0, 12))
        buttons = ttk.Frame(body)
        buttons.pack(anchor=tk.E)

        def choose(action):
            result["action"] = action
            dialog.destroy()

        ttk.Button(
            buttons, text="Save", command=lambda: choose("save")
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            buttons, text="Discard", command=lambda: choose("discard")
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            buttons, text="Cancel", command=lambda: choose("cancel")
        ).pack(side=tk.LEFT)
        dialog.protocol("WM_DELETE_WINDOW", lambda: choose("cancel"))
        dialog.bind("<Escape>", lambda _event: choose("cancel"))
        dialog.update_idletasks()
        x = root.winfo_rootx() + (root.winfo_width() - dialog.winfo_reqwidth()) // 2
        y = root.winfo_rooty() + (root.winfo_height() - dialog.winfo_reqheight()) // 2
        dialog.geometry(f"+{x}+{y}")
        dialog.deiconify()
        dialog.grab_set()
        dialog.wait_window()
        return result["action"]

    def _project_save_path(document, *, title="Save PCS project"):
        initial = (
            document.title
            if document.title.endswith(".pcsp")
            else f"{document.title}.pcsp"
        )
        path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=".pcsp",
            initialfile=initial,
            filetypes=[("PCS project", "*.pcsp"), ("All files", "*.*")],
            parent=root,
        )
        return path or None

    def _dirty_close_decision(document):
        decision = _ask_save_discard_cancel(document)
        if decision != "save" or document.file_path is not None:
            return decision
        path = _project_save_path(document)
        return ("save", path) if path else "cancel"

    def _remember_recent_path(path):
        if not path:
            return
        add_recent_file(state, str(path))
        rebuild_recent_files_menu(state)
        try:
            save_app_state(state, save_geometry=False)
        except Exception:
            pass

    def _save_pcs_document(document_id, *, save_as=False):
        document = state["document_manager"].document_by_id(document_id)
        if document is None:
            return False
        target = None if save_as else document.file_path
        if target is None:
            target = _project_save_path(document)
            if target is None:
                return False
        try:
            saved_document = state["document_manager"].save_document(document_id, target)
            state.get("refresh_document_tabs", lambda: None)()
            _remember_recent_path(saved_document.file_path)
            return True
        except Exception as exc:
            messagebox.showerror("Save PCS project", str(exc), parent=root)
            return False

    def _save_active_pcs_document(*, save_as=False):
        manager = state.get("document_manager")
        if manager is None:
            return False
        return _save_pcs_document(manager.active_document_id, save_as=save_as)

    def _open_project_file_from_path(path: str):
        if not path:
            return False
        try:
            manager = state.get("document_manager")
            if manager is None:
                from logic.project_io import load_project_file

                load_project_file(path, state)
                _remember_recent_path(path)
                return True
            document = manager.open_document(path, activate=True)
            state.get("refresh_document_tabs", lambda: None)()
            _remember_recent_path(document.file_path or path)
            return document
        except Exception as exc:
            messagebox.showerror(
                "Open PCS Project", f"Failed to open project:\n{exc}", parent=root
            )
            return False

    def _open_project_dialog():
        path = filedialog.askopenfilename(
            title="Open PCS project",
            filetypes=[("PCS project", "*.pcsp"), ("All files", "*.*")],
            parent=root,
        )
        return _open_project_file_from_path(path) if path else False

    def _save_all_documents():
        try:
            saved = state["document_manager"].save_all(
                lambda document: _project_save_path(
                    document, title=f"Save {document.title}"
                )
            )
        except Exception as exc:
            messagebox.showerror("Save All", str(exc), parent=root)
            return False
        state.get("refresh_document_tabs", lambda: None)()
        return saved

    def _workspace_save_path():
        path = filedialog.asksaveasfilename(
            title="Save PCS workspace",
            defaultextension=".pcsw",
            initialfile="pcs_workspace.pcsw",
            filetypes=[("PCS workspace", "*.pcsw"), ("All files", "*.*")],
            parent=root,
        )
        return path or None

    def _save_workspace(*, save_as=False):
        if not _save_all_documents():
            return False
        target = None if save_as else state.get("current_workspace_path")
        target = target or _workspace_save_path()
        if not target:
            return False
        try:
            saved = save_workspace(target, state["document_manager"])
        except Exception as exc:
            messagebox.showerror("Save Workspace", str(exc), parent=root)
            return False
        state["current_workspace_path"] = str(saved)
        _remember_recent_path(saved)
        return True

    def _open_workspace_file_from_path(path: str):
        if not path:
            return False
        try:
            if not state["document_manager"].resolve_dirty_documents(
                _dirty_close_decision
            ):
                return False
            result = load_workspace(path, state, state["document_manager"])
        except Exception as exc:
            messagebox.showerror("Open Workspace", str(exc), parent=root)
            return False
        state["current_workspace_path"] = str(result.path)
        state.get("refresh_document_tabs", lambda: None)()
        _remember_recent_path(result.path)
        if result.missing_paths:
            messagebox.showwarning(
                "Open Workspace",
                "Some referenced projects were not found:\n\n"
                + "\n".join(str(item) for item in result.missing_paths),
                parent=root,
            )
        return True

    def _open_workspace_dialog():
        path = filedialog.askopenfilename(
            title="Open PCS workspace",
            filetypes=[("PCS workspace", "*.pcsw"), ("All files", "*.*")],
            parent=root,
        )
        return _open_workspace_file_from_path(path) if path else False

    def _open_file_by_type(path: str):
        if not path:
            return False
        file_path = Path(path)
        ext = file_path.suffix.lower()
        if ext == ".pcsp":
            return _open_project_file_from_path(str(file_path))
        if ext == ".pcsw":
            return _open_workspace_file_from_path(str(file_path))
        if ext in {".xyz", ".out", ".log"}:
            return _load_xyz_from_path(state, str(file_path))
        messagebox.showwarning(
            "Unsupported file",
            "Supported files are:\n"
            "*.pcsp, *.pcsw\n"
            "*.xyz, *.out, *.log",
            parent=root,
        )
        return False

    def _duplicate_pcs_document(document_id):
        document = state["document_manager"].duplicate_document(document_id)
        state.get("refresh_document_tabs", lambda: None)()
        return document

    def _set_project_name(document_id):
        document = state["document_manager"].document_by_id(document_id)
        if document is None:
            return False
        value = simpledialog.askstring(
            "Project Name",
            "Project name (leave empty to use the filename):",
            initialvalue=document.project_name or "",
            parent=root,
        )
        if value is None:
            return False
        state["document_manager"].set_project_name(document_id, value)
        state.get("refresh_document_tabs", lambda: None)()
        return True

    def _set_tab_color(document_id, color):
        state["document_manager"].set_tab_color(document_id, color)
        state.get("refresh_document_tabs", lambda: None)()
        return True

    def _close_pcs_document(document_id):
        try:
            return state["document_manager"].close_document(
                document_id,
                decision_hook=_dirty_close_decision,
            )
        except Exception as exc:
            messagebox.showerror("Close PCS document", str(exc), parent=root)
            return False

    def _close_other_pcs_documents(document_id):
        try:
            return state["document_manager"].close_others(
                document_id, decision_hook=_dirty_close_decision
            )
        except Exception as exc:
            messagebox.showerror("Close PCS documents", str(exc), parent=root)
            return False

    def _close_all_pcs_documents():
        try:
            return state["document_manager"].close_all(
                decision_hook=_dirty_close_decision
            )
        except Exception as exc:
            messagebox.showerror("Close PCS documents", str(exc), parent=root)
            return False

    # close handler
    def _on_main_close():
        try:
            if not state["document_manager"].resolve_dirty_documents(
                _dirty_close_decision
            ):
                return
        except Exception as exc:
            messagebox.showerror("Exit PCS Analyzer", str(exc), parent=root)
            return
        try:
            save_app_state(state, save_geometry=True)
        except Exception:
            pass
        unsubscribe_pcs_refresh_consumers(state)
        # The viewer child must finish cleanup while the Tk event loop still runs.
        # Dirty-document cancellation above never requests child shutdown.
        if state.get("_isolated_molecule_viewer") is not None:
            from ui.molecule_viewer_host import close_isolated_viewer_before_host_exit
            close_isolated_viewer_before_host_exit(state, root.destroy)
        else:
            root.destroy()

    state["save_all_documents"] = _save_all_documents
    state["close_all_documents"] = _close_all_pcs_documents
    state["request_app_close"] = _on_main_close

    # ============================================================
    # Menu bar
    # ============================================================
    command_registry = state["command_registry"]
    from ui.keyboard_shortcuts import navigate_document, open_shortcuts_window, install_main_shortcuts
    from ui.documentation_help import open_documentation
    main_window_commands = register_main_window_commands(
        command_registry,
        {
            "app.file.new_document": lambda: state["document_tabs"].new_document(),
            "app.file.close_document": lambda: state["document_tabs"].close_document(state["document_manager"].active_document_id),
            "app.file.next_document": lambda: navigate_document(state, 1),
            "app.file.previous_document": lambda: navigate_document(state, -1),
            "app.help.documentation": lambda: open_documentation(state),
            "app.help.shortcuts": lambda: open_shortcuts_window(state),
            "app.file.load_xyz": lambda: load_xyz_file(state),
            "app.file.open_project": _open_project_dialog,
            "app.file.save_project": lambda: _save_active_pcs_document(save_as=False),
            "app.file.save_project_as": lambda: _save_active_pcs_document(save_as=True),
            "app.file.save_all": _save_all_documents,
            "app.file.open_workspace": _open_workspace_dialog,
            "app.file.save_workspace": lambda: _save_workspace(save_as=False),
            "app.file.save_workspace_as": lambda: _save_workspace(save_as=True),
            "app.file.save_xyz": lambda: on_save_visible_xyz(state),
            "app.file.save_plot_data": lambda: on_save_plot_any(state),
            "app.file.save_table": lambda: on_save_table_any(state),
            "app.view.pcs_2d": lambda: open_pcs_plot_popup(state),
            "app.view.projection": lambda: open_projection_window(state),
            "app.view.structure_3d": lambda: open_3d_plot_window(state),
            "app.view.chemical_map": lambda: open_chemical_map_window(state),
            "app.view.pcs_3d": lambda: open_pyvista_field(state),
            "app.view.reset_control_panel": lambda: state["control_panel"].reset_layout(),
            "app.tools.nmr_workbench": lambda: open_nmr_workbench(state),
            "app.tools.pcs_workbench": lambda: open_pcs_workbench(state),
            "app.tools.module_manager": lambda: open_module_manager_window(state),
            "app.tools.settings": lambda: open_settings_window(state),
        },
    )
    state["main_window_commands"] = main_window_commands
    state["menu_command_presenters"] = []

    def _add_menu_command(menu, command_id):
        presenter = MenuCommandPresenter(menu, main_window_commands[command_id])
        state["menu_command_presenters"].append(presenter)
        return presenter

    menubar = tk.Menu(root)
    state["menubar"] = menubar
    from ui.edit_history_bar import EditHistoryMenuBar
    state["edit_history_bar"] = EditHistoryMenuBar(root, state)
    file_menu = state["menus"]["File"]
    view_menu = state["menus"]["View"]
    modules_menu = state["menus"]["Modules"]
    settings_menu = state["menus"]["Settings"]
    help_menu = state["menus"]["Help"]

    menubar.add_cascade(label="File", menu=file_menu)
    menubar.add_cascade(label="View", menu=view_menu)
    menubar.add_cascade(label="Modules", menu=modules_menu)
    menubar.add_cascade(label="Settings", menu=settings_menu)
    menubar.add_cascade(label="Help", menu=help_menu)

    state["menus"] = {
        "File": file_menu,
        "View": view_menu,
        "Modules": modules_menu,
        "Settings": settings_menu,
        "Help": help_menu,
    }

    # ----------------------------
    # File menu
    # ----------------------------
    _add_menu_command(file_menu, "app.file.load_xyz")

    recent_files_menu = tk.Menu(file_menu, tearoff=0)
    state["recent_files_menu"] = recent_files_menu
    file_menu.add_cascade(label="Recent files", menu=recent_files_menu)
    rebuild_recent_files_menu(state)

    file_menu.add_separator()

    _add_menu_command(file_menu, "app.file.open_project")
    _add_menu_command(file_menu, "app.file.save_project")
    _add_menu_command(file_menu, "app.file.save_project_as")
    _add_menu_command(file_menu, "app.file.save_all")

    file_menu.add_separator()
    _add_menu_command(file_menu, "app.file.open_workspace")
    _add_menu_command(file_menu, "app.file.save_workspace")
    _add_menu_command(file_menu, "app.file.save_workspace_as")

    file_menu.add_separator()

    _add_menu_command(file_menu, "app.file.save_xyz")
    _add_menu_command(file_menu, "app.file.save_plot_data")
    _add_menu_command(file_menu, "app.file.save_table")

    file_menu.add_separator()

    file_menu.add_command(
        label="Exit",
        command=_on_main_close
    )

    # ----------------------------
    # View menu
    # ----------------------------
    _add_menu_command(view_menu, "app.view.pcs_2d")
    _add_menu_command(view_menu, "app.view.pcs_3d")
    _add_menu_command(view_menu, "app.view.structure_3d")
    _add_menu_command(view_menu, "app.view.chemical_map")
    _add_menu_command(view_menu, "app.view.projection")
    view_menu.add_separator()
    _add_menu_command(view_menu, "app.file.next_document")
    _add_menu_command(view_menu, "app.file.previous_document")

    # ----------------------------
    # Modules menu
    # ----------------------------
    _add_menu_command(modules_menu, "app.tools.module_manager")

    modules_menu.add_separator()

    _add_menu_command(modules_menu, "app.tools.pcs_workbench")
    _add_menu_command(modules_menu, "app.tools.nmr_workbench")

    # External plugin entries will be inserted below this index.
    end_index = modules_menu.index("end")
    state["plugin_menu_start_index"] = (
        end_index + 1 if end_index is not None else 0
    )

    # ----------------------------
    # Settings menu
    # ----------------------------
    _add_menu_command(settings_menu, "app.tools.settings")

    # ----------------------------
    # Help menu
    # ----------------------------
    _add_menu_command(help_menu, "app.help.documentation")
    _add_menu_command(help_menu, "app.help.shortcuts")
    help_menu.add_command(
        label="Check for Updates...",
        command=lambda: check_for_updates_ui(state)
    )

    help_menu.add_separator()

    help_menu.add_command(
        label="About PCS Analyzer",
        command=lambda: open_about_window(state)
    )

    # ============================================================
    # Compact v2 Ribbon (M1-M7 presentation and shared command surfaces)
    # ============================================================
    # Existing callbacks and scientific state remain authoritative; the Ribbon
    # is a command presentation layer over the shared registry.
    ribbon = RibbonBar(
        root,
        collapsed=bool(state["app_settings"].get("ribbon_collapsed", False)),
    )
    state["edit_history_bar"].pack(side=tk.TOP, fill=tk.X)
    ribbon.pack(side=tk.TOP, fill=tk.X)
    state["ribbon"] = ribbon
    state["ribbon_icons"] = RibbonIconService(state.get("icons"), root)

    def _persist_ribbon_collapsed(_event=None):
        state["app_settings"]["ribbon_collapsed"] = ribbon.collapsed
        save_app_state(state, save_geometry=False)

    ribbon.bind(
        "<<RibbonCollapsedChanged>>",
        _persist_ribbon_collapsed,
        add="+",
    )
    state["ribbon_pages"] = {
        page_id: ribbon.add_page(page_id, label)
        for page_id, label in (
            ("home", "Home"),
            ("data", "Data"),
            ("tools", "Tools"),
        )
    }
    state["ribbon_context_page"] = ribbon.add_page(
        "context", "G_ax vs PCS", contextual=True
    )

    stable_page_groups = {
        "home": (
            ("Project", (
                "app.file.load_xyz",
                "app.file.open_project",
                "app.file.save_project",
            )),
            ("Output", (
                "app.file.save_xyz",
                "app.file.save_plot_data",
                "app.file.save_table",
            )),
        ),
        "tools": (
            ("Workbenches", (
                "app.tools.nmr_workbench",
                "app.tools.pcs_workbench",
            )),
            ("Modules", ("app.tools.module_manager",)),
            ("Application", ("app.tools.settings",)),
        ),
    }
    state["stable_ribbon_buttons"] = {}
    for page_id, group_specs in stable_page_groups.items():
        page = state["ribbon_pages"][page_id]
        for caption, command_ids in group_specs:
            group = page.add_group(caption)
            for command_id in command_ids:
                state["stable_ribbon_buttons"][command_id] = group.add_button(
                    command_model=main_window_commands[command_id],
                    icon_resolver=state["ribbon_icons"].resolve,
                )
            if page_id == "tools" and caption == "Modules":
                from ui.ribbon_modules import RibbonModules
                state["ribbon_modules"] = RibbonModules(
                    state, group, state["stable_ribbon_buttons"]["app.tools.module_manager"]
                )


    state["residual_color_enabled_var"] = tk.BooleanVar(value=False)

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

    def _new_pcs_document():
        return state["document_manager"].new_document(activate=True)

    document_tabs = DocumentTabStrip(
        center_frame,
        state["document_manager"],
        on_new=_new_pcs_document,
        on_close=_close_pcs_document,
        on_save=lambda document_id: _save_pcs_document(document_id),
        on_save_as=lambda document_id: _save_pcs_document(
            document_id, save_as=True
        ),
        on_duplicate=_duplicate_pcs_document,
        on_project_name=_set_project_name,
        on_tab_color=_set_tab_color,
        on_close_others=_close_other_pcs_documents,
        on_close_all=_close_all_pcs_documents,
    )
    document_tabs.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 0))
    state["document_tabs"] = document_tabs
    state["refresh_document_tabs"] = document_tabs.refresh

    # Resizable vertical workspace split.  The document tabs remain fixed above
    # it; only the persistent PCS table/status area and the analysis notebook
    # share the divider.  The visible divider is intentionally a single line;
    # a narrow overlay gutter provides the drag target and cursor.  Near either
    # edge the sash has a short magnetic detent before live collapse.
    split_theme_dark = getattr(root, "_theme_variant", "light") == "dark"
    split_bg = getattr(root, "_app_bg", "#F5F6FA")
    split_line = workspace_palette(theme_id(root))["line"]
    split_hover = workspace_palette(theme_id(root))["hover"]
    split_armed = workspace_palette(theme_id(root))["armed"]

    workspace_split = tk.PanedWindow(
        center_frame,
        orient=tk.VERTICAL,
        borderwidth=0,
        relief="flat",
        showhandle=False,
        sashrelief="flat",
        sashwidth=1,
        sashpad=0,
        opaqueresize=True,
        background=split_bg,
    )
    workspace_split.pack(
        side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(1, 8)
    )

    table_pane = ttk.Frame(workspace_split, padding=(0, 0, 0, 2))
    analysis_pane = ttk.Frame(workspace_split)

    def _normalized_workspace_split_ratio(value=None):
        if value is None:
            value = state.get("app_settings", {}).get(
                "main_workspace_split_ratio", 0.42
            )
        try:
            ratio = float(value)
        except Exception:
            ratio = 0.42
        return min(0.85, max(0.15, ratio))

    initial_split_ratio = _normalized_workspace_split_ratio()
    workspace_split.add(table_pane, minsize=0, stretch="always")
    workspace_split.add(analysis_pane, minsize=0, stretch="always")

    state["main_workspace_split"] = workspace_split
    state["main_workspace_table_pane"] = table_pane
    state["main_workspace_analysis_pane"] = analysis_pane

    # Drag-to-collapse uses hysteresis so the boundary does not flap around a
    # single ambiguous threshold.  The persisted ratio is always the last
    # confirmed *expanded* ratio; collapsed edge positions are transient UI
    # state and are never written to preferences.
    split_state = {
        "restored": False,
        "restore_attempts": 0,
        "dragging": False,
        "hover": False,
        "armed": None,       # None | "table" | "analysis" | "restore"
        "collapsed": None,   # None | "table" | "analysis"
        "ratio": initial_split_ratio,  # last confirmed expanded ratio
        "configure_after": None,
        "drag_origin_collapsed": None,
    }

    # Magnetic detents make collapse feel intentional rather than like a
    # sudden release-time snap.  The sash moves freely until the corresponding
    # pane reaches its detent size, then visibly holds there.  Continuing the
    # pointer another short distance collapses the pane immediately *during*
    # the drag, so the user sees the result before releasing the mouse.
    TABLE_DETENT_EDGE_PX = 90
    ANALYSIS_DETENT_EDGE_PX = 100
    COLLAPSE_OVERDRAG_PX = 20

    # Keep the previous effective collapse proximity (about 70/80 px) while
    # inserting the 20 px magnetic hold before it.
    TABLE_COLLAPSE_TRIGGER_PX = max(0, TABLE_DETENT_EDGE_PX - COLLAPSE_OVERDRAG_PX)
    ANALYSIS_COLLAPSE_TRIGGER_PX = max(0, ANALYSIS_DETENT_EDGE_PX - COLLAPSE_OVERDRAG_PX)

    # Restore uses a deliberately hidden pull distance.  From a collapsed edge
    # the sash itself does not move at all for the first 90/100 px of pointer
    # travel.  The next 20 px acts as the magnetic over-drag; once crossed, the
    # pane re-opens immediately during the drag.
    TABLE_RESTORE_EDGE_PX = 110
    ANALYSIS_RESTORE_EDGE_PX = 120

    # The divider looks like a 1 px separator.  Only this narrow transparent-ish
    # overlay is interactive; the PanedWindow itself never owns a resize cursor.
    SPLIT_HIT_HEIGHT = 7

    split_gutter = tk.Canvas(
        workspace_split,
        height=SPLIT_HIT_HEIGHT,
        highlightthickness=0,
        borderwidth=0,
        background=split_bg,
        cursor="sb_v_double_arrow",
    )
    state["main_workspace_split_gutter"] = split_gutter

    def _workspace_height():
        try:
            return max(1, int(workspace_split.winfo_height()))
        except tk.TclError:
            return 1

    def _workspace_sash_y():
        try:
            _x, y = workspace_split.sash_coord(0)
            return int(y)
        except (tk.TclError, ValueError):
            return int(round(_workspace_height() * split_state["ratio"]))

    def _draw_workspace_split_gutter():
        try:
            split_gutter.delete("all")
            width = max(1, split_gutter.winfo_width())
            center_y = SPLIT_HIT_HEIGHT // 2
            if split_state["armed"] is not None:
                color = split_armed
            elif split_state["hover"] or split_state["dragging"]:
                color = split_hover
            else:
                color = split_line
            split_gutter.create_line(
                0,
                center_y,
                width,
                center_y,
                fill=color,
                width=1,
            )
        except tk.TclError:
            pass

    def _refresh_workspace_split_style():
        nonlocal split_bg, split_line, split_hover, split_armed
        dark = getattr(root, "_theme_variant", "light") == "dark"
        split_bg = root._app_bg
        split_line = workspace_palette(theme_id(root))["line"]
        split_hover = workspace_palette(theme_id(root))["hover"]
        split_armed = workspace_palette(theme_id(root))["armed"]
        workspace_split.configure(background=split_bg)
        split_gutter.configure(background=split_bg)
        _draw_workspace_split_gutter()

    def _position_workspace_split_gutter():
        try:
            height = _workspace_height()
            y = _workspace_sash_y()
            y = max(0, min(height - 1, y))
            top = max(0, min(height - SPLIT_HIT_HEIGHT, y - SPLIT_HIT_HEIGHT // 2))
            split_gutter.place(x=0, y=top, relwidth=1.0, height=SPLIT_HIT_HEIGHT)
            split_gutter.lift()
            _draw_workspace_split_gutter()
        except tk.TclError:
            pass

    def _place_workspace_sash(y):
        try:
            height = _workspace_height()
            y = max(0, min(height, int(round(y))))
            workspace_split.sash_place(0, 0, y)
            workspace_split.after_idle(_position_workspace_split_gutter)
        except tk.TclError:
            pass

    def _apply_workspace_split_state():
        if split_state["dragging"]:
            return
        height = _workspace_height()
        if height <= 1:
            return
        collapsed = split_state["collapsed"]
        if collapsed == "table":
            y = 0
        elif collapsed == "analysis":
            y = height
        else:
            y = int(round(height * split_state["ratio"]))
        _place_workspace_sash(y)

    def _persist_workspace_split_ratio(ratio):
        # Persist only a confirmed, expanded layout.  A collapsed edge can never
        # overwrite the user's last usable split position.
        ratio = _normalized_workspace_split_ratio(ratio)
        split_state["ratio"] = ratio
        state.setdefault("app_settings", {})[
            "main_workspace_split_ratio"
        ] = ratio
        save_preferences = state.get("save_app_preferences")
        if callable(save_preferences):
            try:
                save_preferences()
            except Exception:
                pass

    def _restore_workspace_split():
        if split_state["restored"]:
            return
        split_state["restore_attempts"] += 1
        height = _workspace_height()
        if height < 120 and split_state["restore_attempts"] < 12:
            workspace_split.after(40, _restore_workspace_split)
            return
        if height <= 1:
            return
        split_state["collapsed"] = None
        split_state["ratio"] = _normalized_workspace_split_ratio()
        _apply_workspace_split_state()
        split_state["restored"] = True

    def _expanded_drag_target(y):
        """Return (sash_y, detent, collapse_now) for an expanded drag.

        The pointer is allowed to over-drag beyond a magnetic detent while the
        sash itself remains held at the detent.  Crossing the second threshold
        requests an immediate live collapse.
        """
        height = _workspace_height()
        if height <= 1:
            return 0, None, None

        raw_y = int(y)
        clamped_y = max(0, min(height, raw_y))
        table_height = clamped_y
        analysis_height = height - clamped_y

        if table_height < TABLE_DETENT_EDGE_PX:
            if table_height <= TABLE_COLLAPSE_TRIGGER_PX:
                return 0, "table", "table"
            return TABLE_DETENT_EDGE_PX, "table", None

        if analysis_height < ANALYSIS_DETENT_EDGE_PX:
            if analysis_height <= ANALYSIS_COLLAPSE_TRIGGER_PX:
                return height, "analysis", "analysis"
            return height - ANALYSIS_DETENT_EDGE_PX, "analysis", None

        return clamped_y, None, None

    def _collapsed_drag_target(collapsed, y):
        """Return (sash_y, detent, restore_now) for a collapsed drag.

        A collapsed pane stays visually collapsed while the pointer is pulled
        away from the edge.  The first 90/100 px are intentionally invisible;
        the following 20 px are the magnetic over-drag zone.  Crossing the full
        110/120 px threshold re-opens the pane immediately *during* the drag.
        Releasing anywhere before that keeps the pane collapsed.
        """
        height = _workspace_height()
        if height <= 1:
            return 0, None, False

        raw_y = int(y)
        clamped_y = max(0, min(height, raw_y))
        table_height = clamped_y
        analysis_height = height - clamped_y

        if collapsed == "table":
            edge_y = 0
            if table_height < TABLE_DETENT_EDGE_PX:
                return edge_y, None, False
            if table_height < TABLE_RESTORE_EDGE_PX:
                return edge_y, "restore", False
            return clamped_y, "restore", True

        if collapsed == "analysis":
            edge_y = height
            if analysis_height < ANALYSIS_DETENT_EDGE_PX:
                return edge_y, None, False
            if analysis_height < ANALYSIS_RESTORE_EDGE_PX:
                return edge_y, "restore", False
            return clamped_y, "restore", True

        return clamped_y, None, True

    def _collapsed_restore_ready(collapsed, y):
        """Compatibility helper for same-drag reversal after a live collapse."""
        _sash_y, _detent, restore_now = _collapsed_drag_target(collapsed, y)
        return restore_now

    def _split_gutter_enter(_event=None):
        split_state["hover"] = True
        _draw_workspace_split_gutter()

    def _split_gutter_leave(_event=None):
        if not split_state["dragging"]:
            split_state["hover"] = False
            split_state["armed"] = None
            _draw_workspace_split_gutter()

    def _split_gutter_press(_event):
        split_state["dragging"] = True
        split_state["armed"] = None
        split_state["drag_origin_collapsed"] = split_state["collapsed"]
        try:
            split_gutter.grab_set()
        except tk.TclError:
            pass
        _draw_workspace_split_gutter()

    def _split_gutter_drag(event):
        if not split_state["dragging"]:
            return
        try:
            raw_y = int(event.y_root - workspace_split.winfo_rooty())
        except tk.TclError:
            return

        height = _workspace_height()
        y = max(0, min(height, raw_y))
        origin = split_state.get("drag_origin_collapsed")
        live_collapsed = split_state.get("collapsed")

        if origin is not None:
            # Restore gesture stays visually pinned to the collapsed edge while
            # the pointer travels through the hidden pull distance and magnetic
            # over-drag.  Only the final threshold crossing opens the pane.
            sash_y, detent, restore_now = _collapsed_drag_target(origin, raw_y)
            split_state["armed"] = detent
            if restore_now:
                split_state["collapsed"] = None
                split_state["drag_origin_collapsed"] = None
                split_state["armed"] = None
                _place_workspace_sash(sash_y)
            else:
                _place_workspace_sash(sash_y)
            _draw_workspace_split_gutter()
            return

        if live_collapsed is not None:
            # A collapse already happened during this same mouse drag.  Keep it
            # visibly collapsed until the pointer reverses past the wider restore
            # threshold; then reopen immediately and resume normal tracking.
            if _collapsed_restore_ready(live_collapsed, y):
                split_state["collapsed"] = None
                split_state["armed"] = None
                _place_workspace_sash(y)
            else:
                split_state["armed"] = live_collapsed
                _place_workspace_sash(0 if live_collapsed == "table" else height)
            _draw_workspace_split_gutter()
            return

        sash_y, detent, collapse_now = _expanded_drag_target(raw_y)
        split_state["armed"] = detent

        if collapse_now is not None:
            # Live snap: once the user pushes through the magnetic hold, collapse
            # immediately instead of waiting for ButtonRelease.
            split_state["collapsed"] = collapse_now
            _place_workspace_sash(0 if collapse_now == "table" else height)
        else:
            _place_workspace_sash(sash_y)
        _draw_workspace_split_gutter()

    def _split_gutter_release(event):
        if not split_state["dragging"]:
            return
        try:
            y = int(event.y_root - workspace_split.winfo_rooty())
        except tk.TclError:
            y = _workspace_sash_y()

        height = _workspace_height()
        y = max(0, min(height, y))
        origin = split_state.get("drag_origin_collapsed")
        live_collapsed = split_state.get("collapsed") if origin is None else None
        split_state["dragging"] = False
        split_state["armed"] = None
        split_state["drag_origin_collapsed"] = None

        if origin is not None:
            # A drag that began collapsed but never pushed through the restore
            # detent is only a preview.  Release returns to the same edge.
            split_state["collapsed"] = origin
            _place_workspace_sash(0 if origin == "table" else height)
        elif live_collapsed is not None:
            # The visual collapse already happened during motion.  Releasing just
            # confirms that state; never overwrite last_expanded_ratio with 0/1.
            _place_workspace_sash(0 if live_collapsed == "table" else height)
        else:
            # If release happens while sitting on the magnetic detent, keep the
            # pane expanded at the detent rather than interpreting release itself
            # as a collapse command.
            sash_y, _detent, _collapse_now = _expanded_drag_target(y)
            sash_y = max(0, min(height, sash_y))
            split_state["collapsed"] = None
            ratio = _normalized_workspace_split_ratio(
                sash_y / float(max(1, height))
            )
            _place_workspace_sash(height * ratio)
            _persist_workspace_split_ratio(ratio)

        try:
            split_gutter.grab_release()
        except tk.TclError:
            pass
        split_state["hover"] = True
        workspace_split.after_idle(_position_workspace_split_gutter)

    def _workspace_split_configured(_event=None):
        # Debounce geometry bursts so window resize preserves the remembered
        # expanded ratio instead of letting Tk's native stretch policy drift it.
        after_id = split_state.get("configure_after")
        if after_id:
            try:
                workspace_split.after_cancel(after_id)
            except tk.TclError:
                pass
        split_state["configure_after"] = workspace_split.after(
            12, _apply_workspace_split_state
        )

    split_gutter.bind("<Enter>", _split_gutter_enter, add="+")
    split_gutter.bind("<Leave>", _split_gutter_leave, add="+")
    split_gutter.bind("<ButtonPress-1>", _split_gutter_press, add="+")
    split_gutter.bind("<B1-Motion>", _split_gutter_drag, add="+")
    split_gutter.bind("<ButtonRelease-1>", _split_gutter_release, add="+")
    split_gutter.bind("<Configure>", lambda _event: _draw_workspace_split_gutter(), add="+")
    workspace_split.bind("<Configure>", _workspace_split_configured, add="+")
    workspace_split.after_idle(_restore_workspace_split)

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
    table_pane.columnconfigure(0, weight=1)
    table_pane.rowconfigure(0, weight=1)
    table_pane.rowconfigure(1, weight=0)

    table_frame = ttk.Frame(table_pane, padding=(0, 0, 0, 2))
    table_frame.grid(row=0, column=0, sticky="nsew")
    columns = ('Ref','Atom','X','Y','Z','G_ax','δ_PCS','δ_Exp');
    tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=12)
    for col in columns: tree.heading(col, text=col)
    widths = {'Ref':10,'Atom':10,'X':20,'Y':20,'Z':20,'G_ax':30,'δ_PCS':20,'δ_Exp':20}
    for c,w in widths.items(): tree.column(c, width=w)
    scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set); scrollbar.pack(side=tk.RIGHT, fill=tk.Y); tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    state['tree']=tree

    # The table owns the fixed Data command surface. Only meaningful user
    # interaction activates it: programmatic selection restoration/rebuilds
    # may emit <<TreeviewSelect>> but never arm this gate.
    table_activation = {"ready": False, "user_pending": False}

    def _activate_table_data_page(_event=None):
        if table_activation["ready"]:
            ribbon.select_page("data")

    def _clear_table_user_pending():
        table_activation["user_pending"] = False

    def _on_table_pointer(_event=None):
        if not table_activation["ready"]:
            return
        table_activation["user_pending"] = True
        ribbon.select_page("data")
        tree.after_idle(_clear_table_user_pending)

    def _on_table_key(event):
        if getattr(event, "keysym", "") not in {
            "Up", "Down", "Home", "End", "Prior", "Next", "space",
        }:
            return
        _on_table_pointer(event)

    def _on_table_selection(_event=None):
        if table_activation["ready"] and table_activation["user_pending"]:
            ribbon.select_page("data")

    tree.bind("<Button-1>", _on_table_pointer, add="+")
    tree.bind("<FocusIn>", _activate_table_data_page, add="+")
    tree.bind("<KeyPress>", _on_table_key, add="+")
    tree.bind("<<TreeviewSelect>>", _on_table_selection, add="+")
    tree.after_idle(lambda: table_activation.__setitem__("ready", True))
    state["pcs_table_ribbon_activation"] = table_activation

    # Residual Thresholds related
    tree.tag_configure("ok",   foreground="#2ca02c")
    tree.tag_configure("warn", foreground="#ff7f0e")
    tree.tag_configure("bad",  foreground="#d62728")
    tree.tag_configure("none", foreground="#888888")

    # Persistent PCS-data commands now have one production presentation in the
    # Data Ribbon page. Their callbacks remain the established table_utils
    # functions; the table/status strip below remains persistent.
    pcs_data_commands = register_pcs_data_commands(
        command_registry,
        state,
        {
            "pcs.experimental.export_template": (
                lambda: export_delta_exp_template(state)
            ),
            "pcs.experimental.import": (
                lambda: import_delta_exp_file(state, plot_cartesian_graph)
            ),
            "pcs.experimental.paste": (
                lambda: import_delta_exp_from_clipboard(state, plot_cartesian_graph)
            ),
            "pcs.experimental.undo": (
                lambda: undo_last_delta_import(state, plot_cartesian_graph)
            ),
            "pcs.experimental.clear": (
                lambda: clear_delta_exp(state, plot_cartesian_graph)
            ),
        },
    )
    state["pcs_data_commands"] = pcs_data_commands
    state["refresh_pcs_data_commands"] = (
        lambda: refresh_pcs_data_commands(command_registry)
    )

    pcs_data_group = state["ribbon_pages"]["data"].add_group("Experimental PCS")
    pcs_data_ribbon_order = (
        "pcs.experimental.import",
        "pcs.experimental.export_template",
        "pcs.experimental.paste",
        "pcs.experimental.undo",
        "pcs.experimental.clear",
    )
    state["pcs_data_ribbon_buttons"] = {
        command_id: pcs_data_group.add_button(
            command_model=pcs_data_commands[command_id],
            icon_resolver=state["ribbon_icons"].resolve,
        )
        for command_id in pcs_data_ribbon_order
    }
    # View commands now share the persistent Data page with the PCS table.
    # These stable groups are constructed once, after Experimental PCS, so the
    # page follows the data workflow from import through visualization.
    for caption, command_ids in (
        ("PCS Views", (
            "app.view.pcs_2d",
            "app.view.pcs_3d",
            "app.view.projection",
        )),
        ("Structure", (
            "app.view.structure_3d",
            "app.view.chemical_map",
        )),
    ):
        group = state["ribbon_pages"]["data"].add_group(caption)
        for command_id in command_ids:
            state["stable_ribbon_buttons"][command_id] = group.add_button(
                command_model=main_window_commands[command_id],
                icon_resolver=state["ribbon_icons"].resolve,
            )
    # Home is now a complete, low-risk entry surface; Data remains the
    # persistent PCS-table workflow page immediately beside it.
    ribbon.select_page("home")

    # Persistent table status strip: residual threshold legend and display
    # toggle stay adjacent to the data they describe.
    table_status = ttk.Frame(table_pane, padding=(6, 0, 6, 0))
    table_status.grid(row=1, column=0, sticky="ew", pady=(2, 2))

    thr_box = ttk.Frame(table_status, padding=(6, 0, 6, 0))
    thr_box.pack(side=tk.RIGHT, padx=(2, 0))

    from ui.residual_controls import ResidualControls, install_rhombicity_control_layout
    ResidualControls(thr_box, state).pack(side=tk.RIGHT)


    # --- tab ---
    # The one-line splitter above adjusts only the Treeview height; the
    # residual/status strip stays fixed until the whole table pane is
    # intentionally snap-collapsed.
    plots_nb = ttk.Notebook(analysis_pane)
    # Keep a tiny visual breath below the one-line splitter so the analysis
    # tab border does not appear glued directly to the separator.
    plots_nb.pack(fill=tk.BOTH, expand=True, pady=(3, 0))
    state['plots_nb'] = plots_nb

    # -------------------------
    # --- Cartesian tab UI ---
    # -------------------------
    cartesian_tab = ttk.Frame(plots_nb)
    plots_nb.add(cartesian_tab, text="G_ax vs PCS", image=_icon(state, "point_select", TAB_ICON_SIZE), compound=tk.LEFT)

    plot_body = ttk.Frame(cartesian_tab)
    plot_body.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    plot_body.columnconfigure(0, weight=5)
    plot_body.columnconfigure(1, weight=2)
    plot_body.rowconfigure(0, weight=1)

    plot_left = ttk.Frame(plot_body)
    plot_left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

    plot_right = ttk.Frame(plot_body)
    plot_right.grid(row=0, column=1, sticky="nsew")

    # Keep axis-label margins current when the Tk canvas is resized.
    cartesian_figure = plt.Figure(figsize=(4, 3), dpi=100, layout="tight")
    state['cartesian_figure'] = cartesian_figure

    cartesian_canvas = FigureCanvasTkAgg(cartesian_figure, master=plot_left)
    cartesian_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    state['cartesian_canvas'] = cartesian_canvas

    state["cartesian_click_cid"] = None
    state["cartesian_plot_rows"] = []

    # -------------------------
    # Plot control panel
    # -------------------------
    plot_ctrl = ttk.Frame(plot_right)
    plot_ctrl.pack(fill=tk.X, pady=(0, 4))

    ttk.Label(
        plot_ctrl,
        text="Plot Results",
        font=ui_font_name(plot_ctrl, "ui_bold"),
    ).pack(side=tk.LEFT)

    state["cartesian_selection_status_var"] = tk.StringVar(value="All points")
    ttk.Label(
        plot_ctrl,
        textvariable=state["cartesian_selection_status_var"],
        font=ui_font_name(plot_ctrl, "ui_small"),
        style=ui_label_style(plot_ctrl, "muted"),
    ).pack(side=tk.RIGHT)

    # Regression assumption is owned by the existing variable and presented
    # by the contextual Ribbon beside Fit.
    state["plot_force_origin_var"] = tk.BooleanVar(value=False)

    state["cartesian_selection_mode"] = "all"
    state["cartesian_selected_ids"] = set()
    state["cartesian_selector"] = None
    state["cartesian_selector_kind"] = None
    state["cartesian_selector_active"] = False

    plot_result_box = tk.Text(
        plot_right,
        height=12,
        width=34,
        wrap="word",
        font=ui_font_name(plot_right, "report"),
        padx=10,
        pady=8,
    )
    plot_result_scrollbar = ttk.Scrollbar(
        plot_right, orient=tk.VERTICAL, command=plot_result_box.yview,
    )
    plot_result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    plot_result_box.configure(yscrollcommand=plot_result_scrollbar.set)
    plot_result_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    plot_result_box.insert("1.0", "Plot analysis will appear here.")
    plot_result_box.configure(state="disabled")

    state["plot_result_box"] = plot_result_box

    # --------------------------
    # --- Diagnostic tab UI ---
    # --------------------------
    diagtab = ttk.Frame(plots_nb)
    plots_nb.add(diagtab, text="Rhombicity Check", image=_icon(state, "eye", TAB_ICON_SIZE), compound=tk.LEFT)
    state["diagtab"] = diagtab

    # Diagnostic assumption remains owned here and is presented by the
    # contextual Ribbon beside Update.
    state["diag_fit_intercept_var"] = tk.BooleanVar(value=False)

    from ui.rhombicity_panel import build_rhombicity_panel
    build_rhombicity_panel(diagtab, state)

    # --------------------------
    # --- Rhombicity tab UI ----
    # --------------------------
    rhtab = ttk.Frame(plots_nb, width=560)  # Fixed width
    plots_nb.add(rhtab, text="Rhombicity Table", image=_icon(state, "data_table", TAB_ICON_SIZE), compound=tk.LEFT)
    state["rhtab"] = rhtab

    # Prevent the tab frame from resizing to the requested size of its children.
    rhtab.grid_propagate(False)

    # Use grid layout only inside the Rhombicity tab.
    rhtab.rowconfigure(0, weight=0)  # top bar
    rhtab.rowconfigure(1, weight=0)  # z-rotation bar
    rhtab.rowconfigure(2, weight=1)  # table area
    rhtab.columnconfigure(0, weight=1)

    # Top control row
    rh_top = ttk.Frame(rhtab, padding=0)
    rh_top.grid(row=0, column=0, sticky="ew", padx=8, pady=(4, 6))
    rh_inputs = ttk.Frame(rh_top, padding=0)
    rh_inputs.grid(row=0, column=0, sticky="w")
    rh_status = ttk.Frame(rh_top, padding=0)
    rh_status.grid(row=0, column=2, sticky="e")

    # Δχ_rh input (unit: 1e-32 m^3)
    ttk.Label(rh_inputs, text="Δχ_rh (E-32 m³):").pack(side="left", padx=(0, 6))

    # Default state value
    state.setdefault("rh_dchi_rh", 0.0)

    # Entry variable
    state["rh_dchi_rh_var"] = tk.StringVar(value=f"{state.get('rh_dchi_rh', 0.0):g}")
    rh_dchi_entry = ttk.Entry(rh_inputs, textvariable=state["rh_dchi_rh_var"], width=8, justify="right")
    rh_dchi_entry.pack(side="left")
    state["rh_dchi_entry"] = rh_dchi_entry

    # Rhombicity tab table
    cols = ("Ref", "Atom", "r", "theta(deg)", "phi(deg)",
            "Gi_ax", "Gi_rh", "δ_PCS(ax)", "δ_PCS(ax+rh)", "δ_Exp", "res(ax)", "res(ax+rh)")

    # ---------------------------
    # Z-rotation row (between top and table)
    # ---------------------------
    rh_zrow = ttk.Frame(rh_top, padding=0)
    rh_zrow.grid(row=0, column=1, sticky="ew", padx=8)
    state["rh_zrow"] = rh_zrow

    ttk.Label(rh_zrow, text="Rotate around Z-axis (°):").pack(side="left", padx=(0, 6))

    # z state vars
    state.setdefault("angle_z_var", tk.DoubleVar(value=0.0))
    # on_angle_slider / on_angle_entry_commit must support axis 'z'
    azf = ttk.Frame(rh_zrow, padding=0)
    azf.pack(side="left", fill="x", expand=True)

    app_bg = getattr(state["root"], "_app_bg", "#F5F6FA")

    angle_z_slider = ControlPanelSlider(
        azf,
        from_=-180,
        to=180,
        variable=state["angle_z_var"],
        resolution=0.1,
        length=120,
        command=lambda v: on_angle_slider(state, 'z', v),
    )
    _apply_scale_theme(angle_z_slider, state)
    state["rh_angle_z_slider"] = angle_z_slider
    attach_parameter_gesture(angle_z_slider, state, "Rotate structure · Z")
    angle_z_slider.pack(side="left", fill="x", expand=True)

    angle_z_entry = ttk.Entry(azf, width=6)
    angle_z_entry.pack(side="left", padx=(6, 0))
    angle_z_entry.delete(0, tk.END)
    angle_z_entry.insert(0, f"{float(state['angle_z_var'].get()):.1f}")

    state["angle_z_entry"] = angle_z_entry
    angle_z_entry.bind('<Return>', lambda e: on_angle_entry_commit(state, 'z'))
    angle_z_entry.bind('<FocusOut>', lambda e: on_angle_entry_commit(state, 'z'))

    rh_euler_summary_frame = ttk.Frame(rh_top, padding=0)
    state["rh_euler_summary_frame"] = rh_euler_summary_frame
    state["rh_euler_summary_var"] = tk.StringVar(value="α 0.000°   β 0.000°   γ 0.000°")
    ttk.Label(rh_euler_summary_frame, text="Tensor orientation (ZYZ):").pack(side=tk.LEFT)
    ttk.Label(rh_euler_summary_frame, textvariable=state["rh_euler_summary_var"]).pack(side=tk.LEFT, padx=(6, 10))
    def _edit_orientation_in_control_panel():
        state["orientation_mode_var"].set("tensor")
        state["control_panel"].sections["orientation"].set_expanded(True)
    rh_euler_summary_frame.grid(row=0, column=1, sticky="ew", padx=8)
    rh_euler_summary_frame.grid_remove()

    # ---------------------------
    # Use grid layout for the table frame as well(CHANGED row: 1 -> 2)
    # ---------------------------
    rh_table_frame = ttk.Frame(rhtab)
    rh_table_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))

    rh_tree = ttk.Treeview(rh_table_frame, columns=cols, show="headings", height=18)

    # Rhombicity table residual thresholds
    rh_tree.tag_configure("ok",   foreground="#2ca02c")
    rh_tree.tag_configure("warn", foreground="#ff7f0e")
    rh_tree.tag_configure("bad",  foreground="#d62728")
    rh_tree.tag_configure("none", foreground="#888888")

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
        state["_rh_row_residuals"] = {}
        for it in rh_tree.get_children():
            rh_tree.delete(it)
        if not state.get("atom_data"):
            return
        try:
            rows = build_rh_table_rows(state, filter_atoms)
        except Exception as e:
            state["messagebox"].showerror("Rhombicity", str(e))
            return

        use_axrh = bool(state.get("rh_calc_enabled", False))

        for row in rows:
            residual = None
            try:
                # row layout:
                # 0 Ref, 1 Atom, 2 r, 3 theta, 4 phi, 5 Gi_ax, 6 Gi_rh,
                # 7 δ_PCS(ax), 8 δ_PCS(ax+rh), 9 δ_Exp, 10 res(ax), 11 res(ax+rh)
                res_val = row[11] if use_axrh else row[10]
                s = str(res_val).strip()
                if s != "":
                    residual = float(s)
            except Exception:
                residual = None

            tags = classify_residual_tag(state, residual)
            item = rh_tree.insert("", "end", values=row, tags=tags)
            state["_rh_row_residuals"][item] = residual

        if hasattr(state["root"], "_stripe_treeview"):
            state["root"]._stripe_treeview(rh_tree)

    state["rh_refresh_table"] = _rh_refresh_table

    @parameter_edit("Edit rhombic tensor", state)
    def _apply_dchi_rh():
        s = state["rh_dchi_rh_var"].get().strip()
        try:
            v = float(s) if s else 0.0
        except Exception:
            state["messagebox"].showerror("Rhombicity", "Invalid Δχ_rh value.")
            return False

        # Δχ_rh 저장 (입력값은 '1e-32 m^3' 스케일)
        state["rh_dchi_rh"] = v
        canonical = get_canonical_tensor_state(state)
        try:
            current_axial = float(state["tensor_entry"].get())
        except (TypeError, ValueError):
            current_axial = canonical.delta_chi_ax
        try:
            changed = update_canonical_tensor(
                state,
                replace(
                    canonical,
                    delta_chi_ax=current_axial,
                    delta_chi_rh=v,
                    rhombic_enabled=bool(state.get("rh_calc_enabled", False)),
                ),
                source="rhombicity:apply",
                legacy_callback=state.get("update_graph"),
            )
        except PCSDispatchError as exc:
            state["_pcs_model_event_dispatch_failures"] = exc.failures
            changed = True

        # A no-op command still refreshes the explicitly requested diagnostic.
        if not changed:
            _rh_refresh_table()

    # Apply on Enter
    rh_dchi_entry.bind("<Return>", lambda e: _apply_dchi_rh())

    # Rhombicity ON/OFF toggle checkbox
    state.setdefault("rh_calc_enabled", False)
    state["rh_calc_enabled_var"] = tk.BooleanVar(value=state["rh_calc_enabled"])

    @parameter_edit("Toggle rhombic tensor", state)
    def _on_toggle_rh_calc():
        state["rh_calc_enabled"] = bool(state["rh_calc_enabled_var"].get())
        try:
            _apply_dchi_rh()
        except Exception:
            pass
        refresh_orientation = state.get("refresh_orientation_editor")
        if callable(refresh_orientation):
            refresh_orientation()

    ttk.Checkbutton(
        rh_inputs,
        text="Enable Δχ_rh",
        variable=state["rh_calc_enabled_var"],
        command=_on_toggle_rh_calc
    ).pack(side="right", padx=(10, 0))

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
        refresh_susceptibility = state.get("refresh_susceptibility_display")
        if callable(refresh_susceptibility):
            refresh_susceptibility(capture=True)

    ResidualControls(rh_status, state).pack(side=tk.RIGHT)
    install_rhombicity_control_layout(state, rh_top, rh_inputs, rh_zrow,
                                    rh_euler_summary_frame, rh_status)

    # M4 contextual command surfaces. Groups and widgets are built once; the
    # active outer analysis tab only changes which stable groups are packed.
    command_registry = state["command_registry"]
    def _toggle_boolean_var(key):
        variable = state[key]
        variable.set(not bool(variable.get()))

    def _toggle_plot_force_origin():
        _toggle_boolean_var("plot_force_origin_var")
        plot_cartesian_graph(state)

    analysis_commands = register_gax_rhombicity_commands(
        command_registry,
        state,
        {
            "analysis.gax.selection.lasso": (
                lambda: state["cartesian_start_lasso"](state)
            ),
            "analysis.gax.selection.box": (
                lambda: state["cartesian_start_rectangle"](state)
            ),
            "analysis.gax.regression.fit_selected": (
                lambda: state["cartesian_fit_selected"](state)
            ),
            "analysis.gax.regression.force_origin": _toggle_plot_force_origin,
            "analysis.gax.selection.clear": (
                lambda: state["cartesian_clear_selection"](state)
            ),
            "analysis.gax.output.export": lambda: export_cartesian_plot(state),
            "analysis.rhombicity_check.update": (
                lambda: update_diagnostic_panel(state)
            ),
            "analysis.rhombicity_check.force_origin": (
                lambda: _toggle_boolean_var("diag_fit_intercept_var")
            ),
            "analysis.rhombicity_table.update": _apply_dchi_rh,
            "analysis.rhombicity_table.calculate_tensor": _calc_chi_tensor_from_ui,
            "analysis.rhombicity_table.edit_orientation": (
                _edit_orientation_in_control_panel
            ),
        },
    )
    state["gax_rhombicity_commands"] = analysis_commands
    state["refresh_gax_rhombicity_commands"] = (
        lambda: refresh_gax_rhombicity_commands(command_registry)
    )

    context_page = state["ribbon_context_page"]
    gax_selection_group = context_page.add_group("Selection")
    gax_regression_group = context_page.add_group("Regression")
    gax_output_group = context_page.add_group("Output")
    rh_check_group = context_page.add_group("Analysis")
    rh_table_group = context_page.add_group("Table")
    rh_view_group = context_page.add_group("View")

    icon_resolver = state["ribbon_icons"].resolve
    gax_selection_group.add_toggle_button(
        command_model=analysis_commands["analysis.gax.selection.lasso"],
        icon_resolver=icon_resolver,
    )
    gax_selection_group.add_toggle_button(
        command_model=analysis_commands["analysis.gax.selection.box"],
        icon_resolver=icon_resolver,
    )
    gax_selection_group.add_button(
        command_model=analysis_commands["analysis.gax.selection.clear"],
        icon_resolver=icon_resolver,
    )
    gax_regression_group.add_button(
        command_model=analysis_commands["analysis.gax.regression.fit_selected"],
        icon_resolver=icon_resolver,
    )
    state["gax_force_origin_check"] = gax_regression_group.add_checkbutton(
        command_model=analysis_commands["analysis.gax.regression.force_origin"],
        variable=state["plot_force_origin_var"],
        on_change=lambda: plot_cartesian_graph(state),
    )
    gax_output_group.add_button(
        command_model=analysis_commands["analysis.gax.output.export"],
        icon_resolver=icon_resolver,
    )
    rh_check_group.add_button(
        command_model=analysis_commands["analysis.rhombicity_check.update"],
        icon_resolver=icon_resolver,
    )
    state["rhombicity_force_origin_check"] = rh_check_group.add_checkbutton(
        command_model=(
            analysis_commands["analysis.rhombicity_check.force_origin"]
        ),
        variable=state["diag_fit_intercept_var"],
    )
    rh_table_group.add_button(
        command_model=analysis_commands["analysis.rhombicity_table.update"],
        icon_resolver=icon_resolver,
    )
    rh_table_group.add_button(
        command_model=(
            analysis_commands["analysis.rhombicity_table.calculate_tensor"]
        ),
        icon_resolver=icon_resolver,
    )
    rh_view_group.add_button(
        command_model=(
            analysis_commands["analysis.rhombicity_table.edit_orientation"]
        ),
        icon_resolver=icon_resolver,
    )
    state["gax_rhombicity_context_groups"] = {
        "G_ax vs PCS": (
            gax_selection_group,
            gax_regression_group,
            gax_output_group,
        ),
        "Rhombicity Check": (rh_check_group,),
        "Rhombicity Table": (rh_table_group, rh_view_group),
    }


    # --------------------------
    # --- Fitting tab UI -------
    # --------------------------
    fittab = ttk.Frame(plots_nb)
    plots_nb.add(fittab, text="Fitting", image=_icon(state, "run_fit", TAB_ICON_SIZE), compound=tk.LEFT)

    from ui.fitting_presentation import (
        build_fitting_workspace, build_fitting_result_split, show_fit_empty_guide,
    )
    fitting_workspace, fit_output = build_fitting_workspace(fittab, state)
    state["fitting_workspace_scroll"] = fitting_workspace
    fit_content = fitting_workspace

    fit_config = ttk.Frame(fit_content)
    fit_config.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 2))
    try:
        fitting_section_height = max(
            190, round(158 * float(root.tk.call("tk", "scaling")))
        )
    except (tk.TclError, TypeError, ValueError):
        fitting_section_height = 210
    fit_config.configure(height=fitting_section_height)
    fit_config.grid_propagate(False)
    fit_config.columnconfigure(
        0, weight=1, uniform="fitting_model_data", minsize=300
    )
    fit_config.columnconfigure(
        1, weight=1, uniform="fitting_model_data", minsize=300
    )
    fit_config.rowconfigure(1, weight=1)

    model_section = ttk.LabelFrame(fit_config, text="Mode details", padding=6)
    model_section.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
    data_section = ttk.LabelFrame(fit_config, text="Data", padding=6)
    data_section.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
    state["fit_config_frame"] = fit_config
    state["fit_model_section"] = model_section
    state["fit_data_section"] = data_section

    mode_row = ttk.Frame(fit_config)
    mode_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
    state['fit_mode_var'] = tk.StringVar(value='theta_alpha_multi')
    ttk.Radiobutton(
        mode_row,
        text="Mode A · Ligand-donor",
        variable=state['fit_mode_var'],
        value='theta_alpha_multi',
        command=lambda: switch_fit_mode(state),
    ).pack(side=tk.LEFT)

    ttk.Radiobutton(
        mode_row,
        text="Mode B · Euler",
        variable=state['fit_mode_var'],
        value='euler_global',
        command=lambda: switch_fit_mode(state),
    ).pack(side=tk.LEFT, padx=(10, 0))

    ttk.Radiobutton(
        mode_row,
        text="Mode C · Full tensor + metal",
        variable=state['fit_mode_var'],
        value='full_tensor',
        command=lambda: switch_fit_mode(state),
    ).pack(side=tk.LEFT, padx=(10, 0))

    # The mode frames share one parent so switch_fit_mode can expose only the
    # controls that are scientifically relevant to the selected model.
    state['fit_geometry_label_var'] = tk.StringVar(value="Donor atoms")
    state['fit_geometry_label'] = ttk.Label(
        model_section, textvariable=state['fit_geometry_label_var'])
    state['fit_geometry_label'].pack(anchor='w')
    state['fit_anchor'] = ttk.Frame(model_section)

    # --- Mode A (Settings) ---
    frameA = ttk.Frame(model_section)
    state['fit_frameA'] = frameA
    frameA.pack(fill=tk.BOTH, expand=True)

    _donor_box = ttk.Frame(frameA)
    _donor_box.pack(fill=tk.BOTH, expand=True)

    state['fit_donor_list'] = tk.Listbox(_donor_box, selectmode=tk.EXTENDED,
                                         exportselection=False, height=6)
    state['fit_donor_list'].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    _donor_scroll = ttk.Scrollbar(_donor_box, orient=tk.VERTICAL,
                                  command=state['fit_donor_list'].yview)
    _donor_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    state['fit_donor_list'].configure(yscrollcommand=_donor_scroll.set)
    state['fit_donor_list'].bind(
        "<<ListboxSelect>>",
        lambda _event: (
            state.get("refresh_fitting_commands", lambda: None)()
        ),
        add="+",
    )

    rowA = ttk.Frame(frameA);
    rowA.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0), before=_donor_box)
    state['fit_axis_row'] = rowA
    ttk.Label(rowA, text="Axis vector mode:").pack(side=tk.LEFT)
    state['axis_mode_var'] = tk.StringVar(value='bisector')
    ttk.Combobox(rowA, textvariable=state['axis_mode_var'],
                 values=['bisector', 'normal', 'pca', 'centroid', 'average', 'first'],
                 width=12, state="readonly").pack(side=tk.LEFT, padx=6)

    # --- Mode B (Settings) ---
    frameB = ttk.Frame(model_section)
    state['fit_frameB'] = frameB

    labelB = ttk.Label(
        frameB,
        text=("Fits a rigid-body XYZ orientation for the selected PCS "
              "observations. Donor selection is not used."),
        justify="left"
    )
    labelB.pack(anchor='w', pady=(4, 0), fill="x")

    def _adjust_wrapB(event):
        labelB.configure(wraplength=event.width - 12)

    frameB.bind("<Configure>", _adjust_wrapB)

    # initiate- hide mode B
    frameB.pack_forget()

    # --- Mode C (Settings) ---
    frameC = ttk.Frame(model_section)
    state['fit_frameC'] = frameC

    labelC = ttk.Label(
        frameC,
        text=("Fits rigid-body XYZ orientation and metal position for the "
              "selected PCS observations. Tensor terms follow the Advanced "
              "fit flags."),
        justify="left"
    )
    labelC.pack(anchor='w', pady=(4, 0), fill="x")

    def _adjust_wrapC(event):
        labelC.configure(wraplength=event.width - 12)

    frameC.bind("<Configure>", _adjust_wrapC)
    # initiate- hide mode C
    frameC.pack_forget()

    # --- PCS observations to fit ---
    state['fit_protons_label'] = ttk.Label(data_section, text="PCS observations")
    state['fit_protons_label'].pack(anchor='w')

    _proton_box = ttk.Frame(data_section)
    _proton_box.pack(fill=tk.BOTH, expand=True)

    state['fit_proton_list'] = tk.Listbox(_proton_box, selectmode=tk.EXTENDED,
                                          exportselection=False, height=6)
    state['fit_proton_list'].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    _proton_scroll = ttk.Scrollbar(_proton_box, orient=tk.VERTICAL,
                                   command=state['fit_proton_list'].yview)
    _proton_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    state['fit_proton_list'].configure(yscrollcommand=_proton_scroll.set)
    state['fit_proton_list'].bind(
        "<<ListboxSelect>>",
        lambda _event: (
            state.get("refresh_fitting_commands", lambda: None)()
        ),
        add="+",
    )

    weighting_opts = ttk.Frame(data_section)
    state['fit_weighting_row'] = weighting_opts
    weighting_opts.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0), before=_proton_box)
    ttk.Label(weighting_opts, text="Weighting").pack(side=tk.LEFT, padx=(0, 6))
    state['fit_weighting_var'] = tk.StringVar(
        value=fit_weighting_label(FitWeightingMode.UNWEIGHTED)
    )
    ttk.Combobox(
        weighting_opts,
        textvariable=state['fit_weighting_var'],
        values=(
            fit_weighting_label(FitWeightingMode.UNWEIGHTED),
            fit_weighting_label(FitWeightingMode.EXPERIMENTAL_UNCERTAINTY),
        ),
        state="readonly",
        width=25,
    ).pack(side=tk.LEFT)

    state['fit_mode_var'].trace_add("write", state["refresh_fit_settings_summary"])
    state['fit_mode_var'].trace_add("write", lambda *_: _sync_fit_mode_details(state))
    for listing in (state['fit_donor_list'], state['fit_proton_list']):
        listing.bind("<<ListboxSelect>>", state["refresh_fit_settings_summary"], add="+")
    state["refresh_fit_settings_summary"]()

    # Retained for fitting/project compatibility. The permanently disabled WIP
    # control is intentionally not rendered in the routine workflow.
    state['fit_use_visible_var'] = tk.BooleanVar(value=False)

    # Initialization options remain authoritative Tk variables but their sole
    # production presentation is now the contextual Ribbon.
    state['fit_use_current_tensor_var'] = tk.BooleanVar(value=False)
    state['fit_use_current_tensor_hint_var'] = tk.StringVar(
        value="Available for Mode B and Mode C only."
    )
    state['refresh_fit_initial_guess_ui'] = (
        lambda: _update_current_tensor_guess_availability(state)
    )
    _update_current_tensor_guess_availability(state)

    state['fit_dchi_var'] = tk.BooleanVar(value=True)
    state['fit_dchi_rh_var'] = tk.BooleanVar(value=False)
    state['fit_global_search_var'] = tk.BooleanVar(value=False)

    state['fit_status_var'] = tk.StringVar(value="Ready.")
    state['fit_is_running'] = False

    def _fit_progress_cb(msg):
        state['fit_status_var'].set(msg)
        try:
            state['root'].update_idletasks()
        except Exception:
            pass

    state['fit_progress_cb'] = _fit_progress_cb

    # Result is the immediate output of Run Fit and therefore precedes all
    # history-management controls.
    result_section = ttk.LabelFrame(fit_output, text="Result", padding=6)
    result_section.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 4))

    result_header = ttk.Frame(result_section)
    result_header.pack(fill=tk.X, pady=(0, 4))
    state['fit_result_summary_var'] = tk.StringVar(value="No fit result.")
    ttk.Label(
        result_header,
        textvariable=state['fit_result_summary_var'],
    ).pack(side=tk.LEFT)
    state['fit_result_validity_var'] = tk.StringVar(value="Unknown")
    state['fit_result_validity_label'] = ttk.Label(
        result_header,
        textvariable=state['fit_result_validity_var'],
        style=ui_label_style(result_header, "muted"),
    )
    state['fit_result_validity_label'].pack(side=tk.RIGHT)

    # Keep transient run feedback visually associated with the fit summary,
    # while the result-validity indicator ("Current", "Stale", ...) remains
    # isolated at the far right.
    state['fit_status_label'] = ttk.Label(
        result_header,
        textvariable=state['fit_status_var'],
        font=ui_font_name(result_header, "ui_small"),
        style=ui_label_style(result_header, "muted"),
        width=26,
        anchor=tk.W,
    )
    state['fit_status_label'].pack(side=tk.LEFT, padx=(16, 0))

    state['fit_result_validity_hint_var'] = tk.StringVar(value="")
    ttk.Label(
        result_section,
        textvariable=state['fit_result_validity_hint_var'],
        style=ui_label_style(result_section, "muted"),
    ).pack(fill=tk.X, pady=(0, 3))

    fit_left, fit_right = build_fitting_result_split(result_section, state)
    fit_left.rowconfigure(0, weight=1)
    fit_left.columnconfigure(0, weight=1)

    state['fit_result_box'] = tk.Text(
        fit_left, height=8, width=1, font=ui_font_name(fit_left, "report"),
        wrap="none", padx=10, pady=8
    )
    state['fit_result_box'].grid(row=0, column=0, sticky="nsew")
    fit_report_y = ttk.Scrollbar(fit_left, orient=tk.VERTICAL,
                                 command=state['fit_result_box'].yview)
    fit_report_y.grid(row=0, column=1, sticky="ns")
    fit_report_x = ttk.Scrollbar(fit_left, orient=tk.HORIZONTAL,
                                 command=state['fit_result_box'].xview)
    fit_report_x.grid(row=1, column=0, sticky="ew")
    state['fit_result_box'].configure(yscrollcommand=fit_report_y.set,
                                     xscrollcommand=fit_report_x.set)

    fit_fig = Figure(figsize=(6, 3), dpi=100, layout="tight")
    fit_ax = None
    show_fit_empty_guide(fit_fig, state)

    fit_canvas = FigureCanvasTkAgg(fit_fig, master=fit_right)
    fit_canvas.get_tk_widget().configure(width=1, height=1)
    fit_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    state['fit_corr_fig'] = fit_fig
    state['fit_corr_ax'] = fit_ax
    state['fit_corr_canvas'] = fit_canvas

    state['fit_apply_tensor_hint_var'] = tk.StringVar(
        value="Run a valid Mode B or Mode C fit first."
    )
    ttk.Label(
        result_section,
        textvariable=state['fit_apply_tensor_hint_var'],
        style=ui_label_style(result_section, "muted"),
    ).pack(fill=tk.X, pady=(3, 0))
    ttk.Label(
        result_section,
        text=("Preview uses temporary geometry; Mode C also uses fitted Δχ values. "
              "Clear Preview removes geometry and keeps current tensor settings."),
        style=ui_label_style(result_section, "muted"),
        wraplength=780,
        justify="left",
    ).pack(fill=tk.X, pady=(2, 0))

    state['refresh_fit_result_apply_ui'] = (
        lambda: _refresh_fit_result_apply_ui(state)
    )
    state['refresh_fit_result_validity_ui'] = (
        lambda: _refresh_fit_result_validity_ui(state)
    )

    history_frame = ttk.LabelFrame(fit_output, text="Fit History", padding=6)
    history_frame.pack(fill=tk.X, padx=6, pady=(0, 4))
    state['fit_latest_run_summary_var'] = tk.StringVar(value="No fit runs yet.")
    ttk.Label(
        history_frame,
        textvariable=state['fit_latest_run_summary_var'],
        style=ui_label_style(history_frame, "muted"),
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)
    state['fit_history_tree'] = None
    state['fit_history_window'] = None
    state['refresh_fit_history_ui'] = lambda: _refresh_fit_history_ui(state)
    state['refresh_last_fit_result_ui'] = (
        lambda: _refresh_last_fit_result_ui(state)
    )

    # -------------------------------
    # --- Advanced Fitting tab UI ---
    # -------------------------------
    build_advanced_fitting_tab(state, plots_nb)

    # M5 command adapters keep validity decisions with the existing fitting
    # result services and callbacks with their current workspace owners.
    state["can_apply_fit_result"] = lambda: fit_result_application_status(
        state.get("last_fit_result"),
        state,
    )[0]
    state["can_preview_fit_result"] = lambda: bool(
        state.get("last_fit_result")
        and evaluate_fit_result_status(
            state.get("last_fit_result"), state
        ).get("status") != "stale"
    )
    from ui.geometry_preview_history import geometry_preview_command
    fitting_callbacks = {
        "analysis.fitting.run": lambda: run_fit_from_ui(state),
        "analysis.fitting.initialization.use_current_tensor": (
            lambda: _toggle_boolean_var("fit_use_current_tensor_var")
        ),
        "analysis.fitting.initialization.fit_dchi_ax": (
            lambda: _toggle_boolean_var("fit_dchi_var")
        ),
        "analysis.fitting.initialization.fit_dchi_rh": (
            lambda: _toggle_boolean_var("fit_dchi_rh_var")
        ),
        "analysis.fitting.initialization.global_search": (
            lambda: _toggle_boolean_var("fit_global_search_var")
        ),
        "analysis.fitting.data.refresh": (
            lambda: (populate_fitting_controls(state),
                     state["refresh_fit_settings_summary"]())
        ),
        "analysis.fitting.results.apply_tensor": (
            lambda: apply_fit_result_to_current_tensor_ui(state)
        ),
        "analysis.fitting.results.preview_geometry": (
            lambda: geometry_preview_command(state)
        ),
        "analysis.fitting.results.clear_geometry": (
            lambda: geometry_preview_command(state, clear=True)
        ),
        "analysis.fitting.results.history": (
            lambda: _open_fit_history_window(state)
        ),
        "analysis.fitting.output.export_plot": lambda: export_fit_plot(state),
    }
    fitting_callbacks.update(state["advanced_fitting_callbacks"])
    fitting_commands = register_fitting_commands(
        state["command_registry"],
        state,
        fitting_callbacks,
    )
    state["fitting_commands"] = fitting_commands
    state["refresh_fitting_commands"] = (
        lambda: refresh_fitting_commands(state["command_registry"])
    )

    fitting_checkbox_variables = {
        "analysis.fitting.initialization.use_current_tensor": (
            state["fit_use_current_tensor_var"]
        ),
        "analysis.fitting.initialization.fit_dchi_ax": state["fit_dchi_var"],
        "analysis.fitting.initialization.fit_dchi_rh": (
            state["fit_dchi_rh_var"]
        ),
        "analysis.fitting.initialization.global_search": (
            state["fit_global_search_var"]
        ),
    }
    state["fitting_initialization_checkboxes"] = {}

    def _fitting_ribbon_group(caption, *command_ids):
        group = state["ribbon_context_page"].add_group(caption)
        for command_id in command_ids:
            command = fitting_commands[command_id]
            if command_id in fitting_checkbox_variables:
                checkbox = group.add_checkbutton(
                    command_model=command,
                    variable=fitting_checkbox_variables[command_id],
                )
                state["fitting_initialization_checkboxes"][command_id] = checkbox
            elif command_id in (
                "analysis.advanced.multiconformer.parameters.fit_ax",
                "analysis.advanced.multiconformer.parameters.fit_rh",
            ):
                group.add_checkbutton(command_model=command)
            else:
                presenter = (
                    group.add_toggle_button
                    if command.is_toggle
                    else group.add_button
                )
                presenter(
                    command_model=command,
                    icon_resolver=state["ribbon_icons"].resolve,
                )
        return group

    fitting_groups = (
        _fitting_ribbon_group("Fitting", "analysis.fitting.run"),
        _fitting_ribbon_group(
            "Initialization",
            "analysis.fitting.initialization.use_current_tensor",
            "analysis.fitting.initialization.fit_dchi_ax",
            "analysis.fitting.initialization.fit_dchi_rh",
            "analysis.fitting.initialization.global_search",
        ),
        _fitting_ribbon_group("Data", "analysis.fitting.data.refresh"),
        _fitting_ribbon_group(
            "Results",
            "analysis.fitting.results.apply_tensor",
            "analysis.fitting.results.preview_geometry",
            "analysis.fitting.results.clear_geometry",
            "analysis.fitting.results.history",
        ),
        _fitting_ribbon_group(
            "Output", "analysis.fitting.output.export_plot"
        ),
    )
    state["gax_rhombicity_context_groups"]["Fitting"] = fitting_groups

    state["advanced_fitting_context_groups"] = {
        "Multi-Conformer": (
            _fitting_ribbon_group(
                "Fitting", "analysis.advanced.multiconformer.run"
            ),
            _fitting_ribbon_group(
                "Input",
                "analysis.advanced.multiconformer.input.use_current",
                "analysis.advanced.multiconformer.input.load_xyz",
                "analysis.advanced.multiconformer.input.clear",
                "analysis.advanced.multiconformer.input.remove",
                "analysis.advanced.multiconformer.input.configure",
            ),
            _fitting_ribbon_group(
                "Parameters",
                "analysis.advanced.multiconformer.parameters.settings",
                "analysis.advanced.multiconformer.parameters.fit_ax",
                "analysis.advanced.multiconformer.parameters.fit_rh",
            ),
            _fitting_ribbon_group(
                "Data",
                "analysis.advanced.multiconformer.input.observations",
                "analysis.advanced.multiconformer.input.validate",
            ),
            _fitting_ribbon_group(
                "Results",
                "analysis.advanced.multiconformer.results.apply",
                "analysis.advanced.multiconformer.results.reliability",
            ),
            _fitting_ribbon_group(
                "Output", "analysis.advanced.multiconformer.output.export",
            ),
        ),
        "Joint PCS+RDC": (
            _fitting_ribbon_group("Fitting", "analysis.advanced.rdc.run"),
            _fitting_ribbon_group("Parameters", "analysis.advanced.rdc.parameters.settings"),
            _fitting_ribbon_group(
                "Input", "analysis.advanced.rdc.input.observations", "analysis.advanced.rdc.input.load_csv"
            ),
            _fitting_ribbon_group("Validation", "analysis.advanced.rdc.input.validate"),
            _fitting_ribbon_group(
                "Results", "analysis.advanced.rdc.results.apply"
            ),
            _fitting_ribbon_group(
                "Output",
                "analysis.advanced.rdc.output.export",
                "analysis.advanced.rdc.output.export_template",
            ),
        ),
        "PRE": (
            _fitting_ribbon_group("Fitting", "analysis.advanced.pre.run"),
            _fitting_ribbon_group("Input", "analysis.advanced.pre.input.load_csv"),
            _fitting_ribbon_group("Validation", "analysis.advanced.pre.input.validate"),
            _fitting_ribbon_group("Output", "analysis.advanced.pre.output.export",
                                   "analysis.advanced.pre.output.export_template"),
        ),
        "Multi-Lanthanide": (
            _fitting_ribbon_group("Run", "analysis.advanced.multilanthanide.run"),
            _fitting_ribbon_group("Analysis"),
            _fitting_ribbon_group("Input", "analysis.advanced.multilanthanide.input.use_current",
                                   "analysis.advanced.multilanthanide.input.add_csv",
                                   "analysis.advanced.multilanthanide.input.remove",
                                   "analysis.advanced.multilanthanide.input.clear"),
            _fitting_ribbon_group("Validation", "analysis.advanced.multilanthanide.input.validate"),
            _fitting_ribbon_group("Results", "analysis.advanced.multilanthanide.results.transfer"),
            _fitting_ribbon_group("Output", "analysis.advanced.multilanthanide.output.export",
                                    "analysis.advanced.multilanthanide.output.template"),
        ),
    }

    # -------------------------------
    state["ml_workspace"].attach_selector(state["advanced_fitting_context_groups"]["Multi-Lanthanide"][1])

    # --- Conformer search tab UI ---
    # -------------------------------
    cstab = ttk.Frame(plots_nb)
    plots_nb.add(cstab, text="Conformer Search", image=_icon(state, "search", TAB_ICON_SIZE), compound=tk.LEFT)
    state["conformer_tab"] = cstab

    # Candidate/search actions are presented by the contextual Ribbon. The
    # workspace keeps status, scientific parameters, results, and navigation.
    state["conformer_status_var"] = tk.StringVar(
        value="Load XYZ and δ_Exp, then choose 'Sync from Main' in the Ribbon.")

    # ── 4개 서브탭 직접 embed ────────────────────────────────────────────────
    cs_embed_frame = ttk.Frame(cstab)
    cs_embed_frame.pack(fill=tk.BOTH, expand=True)

    cs_app = run_conformer_search_gui(
        embed_in=cs_embed_frame,
        on_preview_result=lambda payload: _conformer_preview_callback(state, payload),
        status_var=state["conformer_status_var"],
    )

    # The embedded Conformer Search renders integration status inside its
    # existing workflow-page headers.  Avoid floating widgets over the native
    # ttk.Notebook tab strip because they clip and can overlap tab hit targets.
    status_labels = getattr(cs_app, "_external_status_labels", ())
    state["conformer_status_label"] = status_labels[0] if status_labels else None
    cs_app._pcs_document_identity_provider = (
        lambda: capture_worker_identity(state)
    )
    cs_app._pcs_document_completion_validator = (
        lambda identity: worker_completion_matches(state, identity)
    )
    cs_app._pcs_document_active_validator = (
        lambda identity: identity is None
        or (
            active_document(state) is not None
            and active_document(state).document_id == identity.document_id
        )
    )
    state["conformer_search_app"] = cs_app
    from ui.conformer_search_persistence import ConformerSearchProjectAdapter
    search_project_adapter = ConformerSearchProjectAdapter(state, cs_app)
    cs_app._project_adapter = search_project_adapter
    state["conformer_search_project_adapter"] = search_project_adapter
    state["capture_conformer_search_state"] = search_project_adapter.capture
    state["refresh_conformer_search_ui"] = search_project_adapter.restore
    # _result_box는 ScrolledText(disabled) — _show_result()가 직접 씀
    state["conformer_result_box"] = cs_app._result_box

    conformer_adapter = ConformerSearchPresentationAdapter(cs_app)
    state["conformer_presentation"] = conformer_adapter
    conformer_callbacks = conformer_adapter.callbacks()
    conformer_callbacks.update({
        "analysis.conformer.setup.sync_main": (
            lambda: open_conformer_search(state)
        ),
        "analysis.conformer.candidate.apply": (
            lambda: apply_conformer_preview(state)
        ),
        "analysis.conformer.candidate.revert": (
            lambda: revert_conformer_to_original(state)
        ),
        "analysis.conformer.candidate.discard": (
            lambda: discard_conformer_preview(state)
        ),
    })
    conformer_commands = register_conformer_commands(
        command_registry,
        state,
        conformer_adapter,
        conformer_callbacks,
    )
    state["conformer_commands"] = conformer_commands
    state["refresh_conformer_commands"] = (
        lambda: refresh_conformer_commands(command_registry)
    )
    conformer_adapter.set_state_changed_callback(
        lambda: (search_project_adapter.capture(), state["refresh_conformer_commands"]())
    )

    def _conformer_ribbon_group(caption, *command_ids):
        group = state["ribbon_context_page"].add_group(caption)
        for command_id in command_ids:
            group.add_button(
                command_model=conformer_commands[command_id],
                icon_resolver=state["ribbon_icons"].resolve,
            )
        return group

    conformer_candidate_group = _conformer_ribbon_group(
        "Candidate",
        "analysis.conformer.results.select_candidate",
        "analysis.conformer.candidate.apply",
        "analysis.conformer.candidate.revert",
        "analysis.conformer.candidate.discard",
    )
    state["conformer_context_groups"] = {
        "①  Fixed atoms": (
            _conformer_ribbon_group(
                "Setup", "analysis.conformer.setup.sync_main"
            ),
            _conformer_ribbon_group(
                "Selection",
                "analysis.conformer.fixed.fix_all_shown",
                "analysis.conformer.fixed.unfix_all",
                "analysis.conformer.fixed.invert",
            ),
            _conformer_ribbon_group(
                "Geometry",
                "analysis.conformer.fixed.apply_locked_bonds",
            ),
            conformer_candidate_group,
        ),
        "②  Rotatable bonds": (
            _conformer_ribbon_group(
                "Selection",
                "analysis.conformer.bonds.select_all_free",
                "analysis.conformer.bonds.select_none",
                "analysis.conformer.bonds.invert",
            ),
            _conformer_ribbon_group(
                "Geometry",
                "analysis.conformer.bonds.skipped_contacts",
                "analysis.conformer.bonds.add",
                "analysis.conformer.bonds.remove",
            ),
            conformer_candidate_group,
        ),
        "③  Run & Options": (
            _conformer_ribbon_group(
                "Search", "analysis.conformer.search.run"
            ),
            conformer_candidate_group,
        ),
        "④  Results": (
            _conformer_ribbon_group(
                "Results",
                "analysis.conformer.results.view_structure",
            ),
            _conformer_ribbon_group(
                "Display",
                "analysis.conformer.results.correlation_plot",
            ),
            _conformer_ribbon_group(
                "Output",
                "analysis.conformer.output.save_xyz",
                "analysis.conformer.output.save_report",
            ),
            conformer_candidate_group,
        ),
    }

    # The outer analysis notebook owns contextual navigation. Initial binding
    # refreshes presentation only; a later meaningful outer/nested tab change
    # selects the existing contextual page after its stable groups are updated.
    ribbon.bind_context_notebook(
        plots_nb,
        labels={
            "G_ax vs PCS": "G_ax vs PCS",
            "Rhombicity Check": "Rhombicity Check",
            "Rhombicity Table": "Rhombicity Table",
            "Fitting": "Fitting",
            "Advanced Fitting": "Advanced Fitting",
            "Conformer Search": "Conformer Search",
        },
    )
    state["analysis_ribbon_controller"] = ContextRibbonController(
        plots_nb,
        state["gax_rhombicity_context_groups"],
        nested_contexts={
            "Advanced Fitting": (
                state["advanced_fitting_nb"],
                state["advanced_fitting_context_groups"],
            ),
            "Conformer Search": (
                conformer_adapter.notebook,
                state["conformer_context_groups"],
            ),
        },
        on_context_changed=lambda _label: (
            refresh_gax_rhombicity_commands(command_registry),
            refresh_fitting_commands(command_registry),
            refresh_conformer_commands(command_registry),
        ),
        on_context_activated=lambda _label: ribbon.select_context_page(),
    )

    # Right Control Panel: scrollable, independently collapsible sections.
    control_panel = ScrollableControlPanel(right_frame, state)
    state["control_panel"] = control_panel
    input_frame = control_panel.frame
    state['input_frame'] = input_frame
    tensor_section = control_panel.section("tensor")
    plot_section = control_panel.section("plot_display")
    orientation_section = control_panel.section("orientation")
    susceptibility_section = control_panel.section("susceptibility")
    symmetry_section = control_panel.section("symmetry")
    tools_section = control_panel.section("tools")
    elements_section = control_panel.section("elements")

    panel_metrics = control_panel.metrics
    tensor_form = ControlPanelForm(tensor_section, panel_metrics)

    tensor_entry, dchi_ax_label, _ = tensor_form.numeric(
        "Δχ_ax", unit="×10⁻³² m³", width=8
    )
    state['tensor_entry'] = tensor_entry
    def _confirm_axial_input(_event):
        commit_parameter_update(state)
        return "break"
    for sequence in ("<Return>", "<KP_Enter>"):
        tensor_entry.bind(sequence, _confirm_axial_input)
    tensor_entry.bind("<FocusOut>", lambda e: commit_parameter_update(state)
                      if not unchanged_parameter_input(state, "tensor_entry") else None)

    tensor_status_var = tk.StringVar(value="Axial")
    state["tensor_status_var"] = tensor_status_var
    def _refresh_tensor_state_label():
        from logic.orca_tensor_transfer import tensor_state_label
        tensor_status_var.set(tensor_state_label(state, get_canonical_tensor_state(state)))
    state["refresh_tensor_state_label"] = _refresh_tensor_state_label
    def _show_tensor_details():
        from ui.orca_tensor_transfer import show_tensor_details
        show_tensor_details(state)
    tensor_status_label, tensor_state_name, tensor_details_button = tensor_form.status_action(
        "Tensor state",
        textvariable=tensor_status_var,
        action_text="Details…",
        command=_show_tensor_details,
    )
    state["tensor_status_label"] = tensor_status_label
    # Keep the existing status slot compact when optional source text is shown.
    import tkinter.font as tkfont
    tensor_status_label.configure(wraplength=tkfont.Font(
        root=tensor_status_label, font=get_app_fonts(state)["ui"]).measure("Axial + Rhombic"))
    state["tensor_details_button"] = tensor_details_button

    prediction_labels = {
        PredictionModel.AXIAL: "Axial",
        PredictionModel.FULL_TENSOR: "Full tensor (Axial + Rhombic)",
    }
    prediction_by_label = {
        label: model for model, label in prediction_labels.items()
    }
    prediction_model_var = tk.StringVar(
        value=prediction_labels[get_prediction_model(state)]
    )
    state["pcs_prediction_model_var"] = prediction_model_var
    prediction_combo = ttk.Combobox(
        tensor_form.frame,
        textvariable=prediction_model_var,
        values=tuple(prediction_by_label),
        state="readonly",
    )
    tensor_form.add_control("Prediction model", prediction_combo)

    def _refresh_prediction_model_ui():
        prediction_model_var.set(
            prediction_labels[get_prediction_model(state)]
        )

    def _on_prediction_model_selected(_event=None):
        requested = prediction_by_label.get(
            prediction_model_var.get(),
            PredictionModel.AXIAL,
        )
        try:
            from ui.prediction_history import select_prediction_model
            select_prediction_model(state, requested)
        except PCSDispatchError as exc:
            state["_pcs_model_event_dispatch_failures"] = exc.failures
        except ValueError as exc:
            state["messagebox"].showerror("Prediction model", str(exc))
        finally:
            _refresh_prediction_model_ui()

    prediction_combo.bind("<<ComboboxSelected>>", _on_prediction_model_selected)
    state["refresh_prediction_model_ui"] = _refresh_prediction_model_ui
    ToolTip(
        prediction_combo,
        "Select the model used by Main PCS, NMR predictions, Projection, "
        "and the 3D PCS field. Full tensor includes the rhombic term only "
        "when that tensor term is enabled.",
    )

    dchi_ax_tooltip = (
        "Set the axial magnetic susceptibility anisotropy used for PCS calculation.\n"
        "Press Enter (including keypad Enter) to apply."
    )
    ToolTip(dchi_ax_label, dchi_ax_tooltip)
    ToolTip(tensor_entry, dchi_ax_tooltip)

    plot_form = ControlPanelForm(plot_section, panel_metrics)
    plot_form.heading("2D PCS contours")

    range_controls = ttk.Frame(plot_form.frame, style="ControlPanel.Form.TFrame")
    range_controls.columnconfigure(0, weight=1)
    range_controls.columnconfigure(2, weight=1)
    pcs_min_entry = ttk.Entry(range_controls, width=7, justify="right")
    pcs_min_entry.grid(row=0, column=0, sticky="ew")
    ttk.Label(
        range_controls, text="–", style="ControlPanel.Unit.TLabel"
    ).grid(row=0, column=1, padx=panel_metrics.column_gap)
    pcs_max_entry = ttk.Entry(range_controls, width=7, justify="right")
    pcs_max_entry.grid(row=0, column=2, sticky="ew")
    pcs_range_label, _ = plot_form.add_control("PCS range", range_controls, "ppm")
    pcs_min_label = pcs_range_label
    pcs_max_label = pcs_range_label

    state['pcs_min_entry'] = pcs_min_entry
    state['pcs_max_entry'] = pcs_max_entry

    pcs_range_tooltip = (
        "Set the minimum and maximum PCS contour values shown in the 2D polar plot."
    )
    ToolTip(pcs_range_label, pcs_range_tooltip)
    ToolTip(pcs_min_label, pcs_range_tooltip)
    ToolTip(pcs_min_entry, pcs_range_tooltip)
    ToolTip(pcs_max_label, pcs_range_tooltip)
    ToolTip(pcs_max_entry, pcs_range_tooltip)

    pcs_interval_entry, pcs_interval_label, _ = plot_form.numeric(
        "Interval", unit="ppm", width=8
    )
    state['pcs_interval_entry'] = pcs_interval_entry

    pcs_interval_tooltip = (
        "Set the spacing between PCS contour lines in the 2D polar plot."
    )
    ToolTip(pcs_interval_label, pcs_interval_tooltip)
    ToolTip(pcs_interval_entry, pcs_interval_tooltip)

    plot_90_var = tk.BooleanVar(value=False)
    state['plot_90_var'] = plot_90_var
    plot_toggle_check = plot_form.checkbox(
        "Half / Quarter (0–90°)",
        variable=plot_90_var,
        command=lambda: state['update_graph']()
    )
    plot_toggle_label = plot_toggle_check

    plot_toggle_tooltip = (
        "Toggle the 2D polar plot range between 0–180° and 0–90°."
    )
    ToolTip(plot_toggle_label, plot_toggle_tooltip)
    ToolTip(plot_toggle_check, plot_toggle_tooltip)

    plot_form.heading("Structure rendering")

    atom_scale_var = tk.BooleanVar(value=False)
    state['atom_scale_var'] = atom_scale_var
    show_bonds_var = tk.BooleanVar(value=False)
    state['show_bonds_var'] = show_bonds_var
    atom_radii_check = plot_form.checkbox(
        "Use atom radii",
        variable=atom_scale_var,
        command=lambda: state['update_graph']()
    )
    atom_radii_label = atom_radii_check

    atom_scale_entry, atom_scale_label, _ = plot_form.numeric(
        "Atom scale", width=8
    )
    atom_scale_entry.insert(0, "1.0")
    state['atom_scale_entry'] = atom_scale_entry

    atom_radii_tooltip = (
        "Show atom-size-scaled scatter points in the 2D polar plot."
    )
    atom_scale_tooltip = (
        "Adjust the atom-radius scaling factor in the 2D polar plot."
    )

    ToolTip(atom_radii_label, atom_radii_tooltip)
    ToolTip(atom_radii_check, atom_radii_tooltip)
    ToolTip(atom_scale_label, atom_scale_tooltip)
    ToolTip(atom_scale_entry, atom_scale_tooltip)

    bonds_check = plot_form.checkbox(
        "Show bonds",
        variable=show_bonds_var,
        command=lambda: state['update_graph']()
    )
    bonds_label = bonds_check

    bond_width_entry, width_label, _ = plot_form.numeric(
        "Bond width", width=8
    )
    bond_width_entry.insert(0, "1.1")
    state['bond_width_entry'] = bond_width_entry

    bond_tol_entry, tol_label, _ = plot_form.numeric(
        "Tolerance", width=8
    )
    bond_tol_entry.insert(0, "0.03")
    state['bond_tol_entry'] = bond_tol_entry

    bond_tooltip = (
        "Display guide lines between nearby atoms. "
        "These are visual aids, not confirmed bonding paths."
    )
    width_tooltip = (
        "Set the thickness of the displayed bond lines."
    )
    tol_tooltip = (
        "Set the extra margin for bond detection. "
        "Cutoff = summed covalent radii × (1.00 + tol.)."
    )

    ToolTip(bonds_label, bond_tooltip)
    ToolTip(bonds_check, bond_tooltip)
    ToolTip(width_label, width_tooltip)
    ToolTip(bond_width_entry, width_tooltip)
    ToolTip(tol_label, tol_tooltip)
    ToolTip(bond_tol_entry, tol_tooltip)

    plot_actions = plot_form.actions()
    update_btn = ttk.Button(
        plot_actions,
        text="Update",
        image=_icon(state, "update", TOOLBAR_ICON_SIZE),
        compound=tk.LEFT,
        style="ControlPanel.Action.TButton",
        command=lambda: commit_parameter_update(state),
    )
    update_btn.pack(side=tk.LEFT, padx=(0, panel_metrics.action_gap))
    reset_btn = ttk.Button(
        plot_actions,
        text="Reset",
        image=_icon(state, "reset", TOOLBAR_ICON_SIZE),
        compound=tk.LEFT,
        style="ControlPanel.Danger.TButton",
        command=lambda: reset_values(state),
    )
    reset_btn.pack(side=tk.LEFT)
    ToolTip(
        update_btn,
        "Update the PCS values, table, and plots using the current settings."
    )
    ToolTip(
        reset_btn,
        "Reset all PCS Analyzer data and controls to their defaults."
    )

    susceptibility_form = ControlPanelForm(susceptibility_section, panel_metrics)
    chi_mol_entry, chi_mol_label, _ = susceptibility_form.numeric(
        "χ_mol", unit="m³/mol", width=10
    )
    state['chi_mol_entry'] = chi_mol_entry

    chi_mol_tooltip = (
        "Enter the experimental molar susceptibility. "
        "Leave blank to use a traceless tensor approximation."
    )
    ToolTip(chi_mol_label, chi_mol_tooltip)
    ToolTip(chi_mol_entry, chi_mol_tooltip)

    state.setdefault("chi_auto_calc_var", tk.BooleanVar(value=True))
    auto_calc_check = susceptibility_form.checkbox(
        "Auto-calculate",
        variable=state["chi_auto_calc_var"],
    )
    ToolTip(
        auto_calc_check,
        "Automatically update χ tensor components when related input values change."
    )

    # Preserve the existing table_utils label side-effects as hidden compatibility
    # widgets. The visible rows below are presentation-only and can change units
    # without altering calculation semantics or the legacy label contract.
    state['molar_value_label'] = ttk.Label(
        susceptibility_section, text="Δχ_mol_ax : N/A m³/mol"
    )
    state['tensor_xx_label'] = ttk.Label(
        susceptibility_section, text="χ_xx: N/A m³/mol"
    )
    state['tensor_yy_label'] = ttk.Label(
        susceptibility_section, text="χ_yy: N/A m³/mol"
    )
    state['tensor_zz_label'] = ttk.Label(
        susceptibility_section, text="χ_zz: N/A m³/mol"
    )

    display_mode = state.get("app_settings", {}).get(
        "susceptibility_display_unit", "molar"
    )
    if display_mode not in {"molar", "molecular"}:
        display_mode = "molar"
    susceptibility_unit_mode_var = tk.StringVar(value=display_mode)
    state["susceptibility_unit_mode_var"] = susceptibility_unit_mode_var
    susceptibility_unit_button_var = tk.StringVar(
        value="Molar" if display_mode == "molar" else "Molecular"
    )

    # Keep the Derived result grid separate from the χ_mol input grid.
    # This prevents the longer molecular unit label from changing the input
    # column geometry when the display mode is toggled.
    derived_header = ttk.Frame(
        susceptibility_section, style="ControlPanel.Form.TFrame"
    )
    derived_header.pack(
        fill=tk.X,
        pady=(panel_metrics.row_y + max(2, int(round(2 * panel_metrics.scale))),
              panel_metrics.row_y),
    )
    ttk.Label(
        derived_header,
        text="Derived",
        style="ControlPanel.Subheading.TLabel",
        anchor="w",
    ).pack(side=tk.LEFT)
    unit_toggle = ttk.Button(
        derived_header,
        textvariable=susceptibility_unit_button_var,
        style="ControlPanel.Secondary.TButton",
    )
    unit_toggle.pack(side=tk.RIGHT)
    ToolTip(
        unit_toggle,
        "Switch derived susceptibility display between molar (m³/mol) and "
        "molecular (×10⁻³² m³) units. The scientific input is unchanged."
    )

    derived_body = ttk.Frame(
        susceptibility_section, style="ControlPanel.Form.TFrame"
    )
    derived_body.pack(fill=tk.X)
    derived_form = ControlPanelForm(derived_body, panel_metrics)
    # Reserve the longest unit width so only the text changes, not the columns.
    derived_form.frame.columnconfigure(
        2, minsize=max(1, int(round(76 * panel_metrics.scale)))
    )

    derived_unit_var = tk.StringVar(
        value="m³/mol" if display_mode == "molar" else "×10⁻³² m³"
    )
    state["susceptibility_derived_unit_var"] = derived_unit_var
    derived_value_vars = {
        "delta_chi_ax": tk.StringVar(value="N/A"),
        "chi_xx": tk.StringVar(value="N/A"),
        "chi_yy": tk.StringVar(value="N/A"),
        "chi_zz": tk.StringVar(value="N/A"),
    }
    state["susceptibility_derived_value_vars"] = derived_value_vars

    molar_display_label, molar_display_name, _ = derived_form.result(
        "Δχ_mol_ax",
        textvariable=derived_value_vars["delta_chi_ax"],
        unitvariable=derived_unit_var,
    )
    xx_display_label, xx_display_name, _ = derived_form.result(
        "χ_xx",
        textvariable=derived_value_vars["chi_xx"],
        unitvariable=derived_unit_var,
    )
    yy_display_label, yy_display_name, _ = derived_form.result(
        "χ_yy",
        textvariable=derived_value_vars["chi_yy"],
        unitvariable=derived_unit_var,
    )
    zz_display_label, zz_display_name, _ = derived_form.result(
        "χ_zz",
        textvariable=derived_value_vars["chi_zz"],
        unitvariable=derived_unit_var,
    )

    molar_value_tooltip = (
        "Molar axial susceptibility anisotropy converted from the molecular Δχ_ax value."
    )
    for widget in (molar_display_name, molar_display_label):
        ToolTip(widget, molar_value_tooltip)
    tensor_component_tooltip = (
        "Calculated tensor components derived from the current Δχ settings."
    )
    for widget in (
        xx_display_name, xx_display_label,
        yy_display_name, yy_display_label,
        zz_display_name, zz_display_label,
    ):
        ToolTip(widget, tensor_component_tooltip)

    _susceptibility_number_pattern = re.compile(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    )

    def _legacy_label_number(widget):
        try:
            text = str(widget.cget("text") or "")
        except Exception:
            return None
        if "N/A" in text.upper():
            return None
        match = _susceptibility_number_pattern.search(text.replace("−", "-"))
        if match is None:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _capture_susceptibility_molar_values():
        values = {
            "delta_chi_ax": _legacy_label_number(state["molar_value_label"]),
            "chi_xx": _legacy_label_number(state["tensor_xx_label"]),
            "chi_yy": _legacy_label_number(state["tensor_yy_label"]),
            "chi_zz": _legacy_label_number(state["tensor_zz_label"]),
        }
        state["susceptibility_molar_values"] = values
        return values

    def _format_susceptibility_value(value, mode):
        if value is None or not np.isfinite(value):
            return "N/A"
        if mode == "molecular":
            return f"{(value / AVOGADRO_CONSTANT) * 1.0e32:.6g}"
        return f"{value:.6g}"

    def _refresh_susceptibility_display(*, capture=False):
        values = (
            _capture_susceptibility_molar_values()
            if capture
            else state.get("susceptibility_molar_values")
        )
        if values is None:
            values = _capture_susceptibility_molar_values()
        mode = susceptibility_unit_mode_var.get()
        if mode not in {"molar", "molecular"}:
            mode = "molar"
            susceptibility_unit_mode_var.set(mode)
        derived_unit_var.set("m³/mol" if mode == "molar" else "×10⁻³² m³")
        susceptibility_unit_button_var.set(
            "Molar" if mode == "molar" else "Molecular"
        )
        for key, variable in derived_value_vars.items():
            variable.set(_format_susceptibility_value(values.get(key), mode))

    def _toggle_susceptibility_display_unit():
        new_mode = (
            "molecular"
            if susceptibility_unit_mode_var.get() == "molar"
            else "molar"
        )
        susceptibility_unit_mode_var.set(new_mode)
        state.setdefault("app_settings", {})["susceptibility_display_unit"] = new_mode
        callback = state.get("save_app_preferences")
        if callable(callback):
            callback()
        _refresh_susceptibility_display(capture=False)

    unit_toggle.configure(command=_toggle_susceptibility_display_unit)
    state["refresh_susceptibility_display"] = _refresh_susceptibility_display
    state["capture_susceptibility_molar_values"] = _capture_susceptibility_molar_values
    _refresh_susceptibility_display(capture=True)

    # ---- Placeholder (blank -> traceless) ----
    CHI_PLACEHOLDER = "blank → χ_iso=0"
    state["chi_placeholder_text"] = CHI_PLACEHOLDER
    state["chi_placeholder_active"] = False

    def _set_placeholder():
        if (chi_mol_entry.get() or "").strip() == "":
            chi_mol_entry.delete(0, tk.END)
            chi_mol_entry.insert(0, CHI_PLACEHOLDER)
            chi_mol_entry.configure(style="ControlPanel.Placeholder.TEntry")
            state["chi_placeholder_active"] = True

    def _clear_placeholder():
        if state.get("chi_placeholder_active", False):
            chi_mol_entry.delete(0, tk.END)
            chi_mol_entry.configure(style="TEntry")
            state["chi_placeholder_active"] = False

    _set_placeholder()

    chi_mol_entry.bind("<FocusIn>", lambda e: _clear_placeholder())
    chi_mol_entry.bind(
        "<FocusOut>",
        lambda e: (
            _set_placeholder(),
            state.get("schedule_chi_autocalc", lambda: None)(),
        ),
    )

    # ---- Auto-calc core (NO popup on typing) ----
    state["_chi_autocalc_after_id"] = None

    def _maybe_calc_chi(_event=None, *, quiet=True):
        """Auto-calc가 켜져 있을 때만 χ_xx/yy/zz 계산."""
        if not state["chi_auto_calc_var"].get():
            return

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
            _refresh_susceptibility_display(capture=True)
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
        state["_chi_autocalc_after_id"] = state["root"].after(
            200, lambda: _maybe_calc_chi(None, quiet=True)
        )

    state["schedule_chi_autocalc"] = _schedule_chi_autocalc

    chi_mol_entry.bind("<Return>", lambda e: _maybe_calc_chi(e, quiet=True))
    chi_mol_entry.bind("<KeyRelease>", lambda e: _schedule_chi_autocalc())

    # # File load
    # ttk.Button(input_frame,
    #            text="📂 Load xyz File",
    #            command=lambda: load_xyz_file(state)
    #            ).pack(anchor="center", pady=3)

    # ---- Averaging settings ----  # PCS_PATCH_AVERAGING_SETTINGS_PHASE1
    # Compact main-panel summary; detailed controls live in Averaging Settings.
    # PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE2
    state.setdefault("symavg_enabled_var", tk.BooleanVar(value=False))
    state.setdefault("symavg_methyl_enabled_var", tk.BooleanVar(value=True))
    state.setdefault("symavg_cf3_enabled_var", tk.BooleanVar(value=True))
    state.setdefault("symavg_keep_original_var", tk.BooleanVar(value=False))
    state.setdefault("torsion_avg_enabled_var", tk.BooleanVar(value=False))
    state.setdefault("torsion_avg_keep_reference_var", tk.BooleanVar(value=False))
    state.setdefault("torsion_avg_detection_mode_var", tk.StringVar(value="auto"))
    state.setdefault("torsion_avg_planarity_var", tk.DoubleVar(value=0.10))
    state.setdefault("torsion_avg_groups", [])
    state.setdefault("torsion_avg_phase", 2)
    state.setdefault("averaging_settings_window", None)

    averaging_status_var = tk.StringVar(value="No averaging applied")
    state["averaging_status_var"] = averaging_status_var

    def _refresh_averaging_status():
        active = []

        if bool(state["symavg_enabled_var"].get()):
            if bool(state["symavg_methyl_enabled_var"].get()):
                active.append("CH₃")
            if bool(state["symavg_cf3_enabled_var"].get()):
                active.append("CF₃")

        if bool(state["torsion_avg_enabled_var"].get()):
            n_rot = sum(
                bool(g.get("enabled", True))
                for g in (state.get("torsion_avg_groups", []) or [])
            )
            active.append(f"Rot×{n_rot}")

        if active:
            averaging_status_var.set(" · ".join(active))
            status_label.configure(style="ControlPanel.TLabel")
        else:
            averaging_status_var.set("No averaging applied")
            status_label.configure(style="ControlPanel.Muted.TLabel")

    def _apply_averaging_settings():
        # Phase 2 keeps the established local-symmetry calculation unchanged.
        # Rotational groups are configured/detected here and integrated into G
        # calculations by the following patch phase.
        try:
            apply_symavg_to_state(state)
        except Exception as e:
            state["messagebox"].showwarning("Averaging", f"Failed:\n{e}")
            return

        tree = state.get("tree")
        if tree is not None:
            try:
                tree.selection_remove(tree.selection())
            except Exception:
                pass

        _refresh_averaging_status()
        update_graph(state)
        try:
            populate_fitting_controls(state)
        except Exception:
            pass

    state["refresh_averaging_status"] = _refresh_averaging_status
    state["apply_averaging_settings"] = _apply_averaging_settings

    symmetry_form = ControlPanelForm(symmetry_section, panel_metrics)
    symmetry_form.heading("CH₃, CF₃, phenyl, and rotational groups")
    avg_row = ttk.Frame(symmetry_form.frame, style="ControlPanel.Form.TFrame")
    status_label = ttk.Label(
        avg_row,
        textvariable=averaging_status_var,
        anchor="w",
        style="ControlPanel.Muted.TLabel",
    )
    status_label.pack(side="left", fill="x", expand=True)

    ttk.Button(
        avg_row,
        text="Options...",
        style="ControlPanel.Secondary.TButton",
        command=lambda: open_averaging_settings_window(state),
    ).pack(side="right")
    symmetry_form.add_control("Status", avg_row)

    _refresh_averaging_status()

    # Two UI representations of one canonical molecular->tensor orientation.
    orientation_mode_var = tk.StringVar(value="structure")
    state["orientation_mode_var"] = orientation_mode_var
    mode_row = ttk.Frame(orientation_section, style="ControlPanel.Form.TFrame")
    mode_row.pack(fill=tk.X, pady=(0, panel_metrics.row_y))
    ttk.Radiobutton(
        mode_row,
        text="Structure",
        value="structure",
        variable=orientation_mode_var,
        style="ControlPanel.TRadiobutton",
    ).pack(side=tk.LEFT)
    ttk.Radiobutton(
        mode_row,
        text="Tensor Euler (ZYZ)",
        value="tensor",
        variable=orientation_mode_var,
        style="ControlPanel.TRadiobutton",
    ).pack(side=tk.LEFT, padx=(panel_metrics.column_gap, 0))

    structure_orientation_frame = ttk.Frame(
        orientation_section, style="ControlPanel.Form.TFrame"
    )
    tensor_orientation_frame = ttk.Frame(
        orientation_section, style="ControlPanel.Form.TFrame"
    )
    state["structure_orientation_frame"] = structure_orientation_frame
    state["tensor_orientation_frame"] = tensor_orientation_frame

    structure_form = ControlPanelForm(structure_orientation_frame, panel_metrics)
    angle_x_var = tk.DoubleVar()
    state['angle_x_var'] = angle_x_var
    angle_x_slider, angle_x_entry, _, _ = structure_form.slider(
        "X-axis",
        from_=-180,
        to=180,
        variable=angle_x_var,
        command=lambda v: on_angle_slider(state, 'x', v),
    )
    _apply_scale_theme(angle_x_slider, state)
    state["angle_x_slider"] = angle_x_slider
    attach_parameter_gesture(angle_x_slider, state, "Rotate structure · X")
    angle_x_entry.insert(0, '0.0')
    state['angle_x_entry'] = angle_x_entry
    angle_x_entry.bind('<Return>', lambda e: on_angle_entry_commit(state, 'x'))
    angle_x_entry.bind('<FocusOut>', lambda e: on_angle_entry_commit(state, 'x'))

    angle_y_var = tk.DoubleVar()
    state['angle_y_var'] = angle_y_var
    angle_y_slider, angle_y_entry, _, _ = structure_form.slider(
        "Y-axis",
        from_=-180,
        to=180,
        variable=angle_y_var,
        command=lambda v: on_angle_slider(state, 'y', v),
    )
    _apply_scale_theme(angle_y_slider, state)
    state["angle_y_slider"] = angle_y_slider
    attach_parameter_gesture(angle_y_slider, state, "Rotate structure · Y")
    angle_y_entry.insert(0, '0.0')
    state['angle_y_entry'] = angle_y_entry
    angle_y_entry.bind('<Return>', lambda e: on_angle_entry_commit(state, 'y'))
    angle_y_entry.bind('<FocusOut>', lambda e: on_angle_entry_commit(state, 'y'))

    tensor_form = ControlPanelForm(tensor_orientation_frame, panel_metrics)
    tensor_form.help("Active ZYZ · molecular frame → tensor frame")
    tensor_euler_vars = {}
    tensor_euler_slider_vars = {}
    tensor_euler_entries = {}
    tensor_euler_sliders = {}
    slider_ranges = {
        "alpha": (-180.0, 180.0),
        "beta": (0.0, 180.0),
        "gamma": (-180.0, 180.0),
    }

    def _on_tensor_euler_slider(symbol, value):
        if state.get("_syncing_tensor_euler_controls", False):
            return
        if symbol == "alpha" and not get_canonical_tensor_state(state).rhombic_enabled:
            return
        tensor_euler_vars[symbol].set(f"{float(value):.6g}")
        _commit_tensor_euler()

    for symbol, label in (("alpha", "α"), ("beta", "β"), ("gamma", "γ")):
        lo, hi = slider_ranges[symbol]
        slider_var = tk.DoubleVar(value=0.0)
        variable = tk.StringVar(value="0")
        slider, entry, _, _ = tensor_form.slider(
            label,
            from_=lo,
            to=hi,
            variable=slider_var,
            entry_textvariable=variable,
            command=lambda value, key=symbol: _on_tensor_euler_slider(key, value),
        )
        _apply_scale_theme(slider, state)
        tensor_euler_vars[symbol] = variable
        tensor_euler_slider_vars[symbol] = slider_var
        tensor_euler_entries[symbol] = entry
        tensor_euler_sliders[symbol] = slider
        attach_parameter_gesture(slider, state, f"Rotate tensor · {label}")
    tensor_form.help("α is retained but read-only for an axial tensor.")
    state["tensor_euler_vars"] = tensor_euler_vars
    state["tensor_euler_slider_vars"] = tensor_euler_slider_vars
    state["tensor_euler_entries"] = tensor_euler_entries
    state["tensor_euler_sliders"] = tensor_euler_sliders

    def _refresh_orientation_editor():
        canonical = get_canonical_tensor_state(state)
        zyz_angles = matrix_to_euler(
            orientation_matrix(
                canonical.euler_angles,
                canonical.frame_convention,
                canonical.angle_unit,
            ),
            EulerConvention.INTRINSIC_ZYZ,
            AngleUnit.DEGREE,
        )
        state["_syncing_tensor_euler_controls"] = True
        try:
            for key, value in zip(("alpha", "beta", "gamma"), zyz_angles):
                tensor_euler_vars[key].set(f"{float(value):.8g}")
                tensor_euler_slider_vars[key].set(float(value))
        finally:
            state["_syncing_tensor_euler_controls"] = False
        alpha_state = "normal" if canonical.rhombic_enabled else "readonly"
        tensor_euler_entries["alpha"].configure(state=alpha_state)
        tensor_euler_sliders["alpha"].configure(
            state=tk.NORMAL if canonical.rhombic_enabled else tk.DISABLED
        )
        if state.get("tensor_status_var") is not None:
            _refresh_tensor_state_label()
        summary = state.get("rh_euler_summary_var")
        if summary is not None:
            summary.set(
                f"α {zyz_angles[0]:.3f}°   β {zyz_angles[1]:.3f}°   γ {zyz_angles[2]:.3f}°"
            )

    @parameter_edit("Edit tensor Euler angles", state)
    def _commit_tensor_euler(_event=None):
        if all(unchanged_parameter_input(state, f"tensor_euler_{key}")
               for key in ("alpha", "beta", "gamma")):
            return
        try:
            values = tuple(float(tensor_euler_vars[key].get()) for key in ("alpha", "beta", "gamma"))
        except ValueError:
            state["messagebox"].showerror("Orientation", "Enter valid Euler angles.")
            _refresh_orientation_editor()
            return False
        canonical = get_canonical_tensor_state(state)
        updated = replace(
            canonical,
            alpha=values[0], beta=values[1], gamma=values[2],
            frame_convention=EulerConvention.INTRINSIC_ZYZ,
            angle_unit=AngleUnit.DEGREE,
        )
        try:
            update_canonical_tensor(
                state,
                updated,
                source="control_panel:tensor_euler",
                legacy_callback=state.get("update_graph"),
            )
        except PCSDispatchError as exc:
            state["_pcs_model_event_dispatch_failures"] = exc.failures
        _refresh_orientation_editor()

    for entry in tensor_euler_entries.values():
        entry.bind("<Return>", _commit_tensor_euler)
        entry.bind("<FocusOut>", _commit_tensor_euler)

    def _set_orientation_mode(*_args):
        mode = orientation_mode_var.get()
        structure_orientation_frame.pack_forget()
        tensor_orientation_frame.pack_forget()
        if mode == "tensor":
            _refresh_orientation_editor()
            tensor_orientation_frame.pack(fill=tk.X)
        else:
            structure_orientation_frame.pack(fill=tk.X)
        rh_zrow = state.get("rh_zrow")
        rh_summary = state.get("rh_euler_summary_frame")
        if rh_zrow is not None and rh_summary is not None:
            if mode == "tensor":
                rh_zrow.grid_remove()
                rh_summary.grid()
            else:
                rh_summary.grid_remove()
                rh_zrow.grid()
            layout_controls = state.get("layout_rhombicity_controls")
            if callable(layout_controls):
                layout_controls()

    state["refresh_orientation_editor"] = _refresh_orientation_editor
    state["set_orientation_mode"] = _set_orientation_mode
    orientation_mode_var.trace_add("write", _set_orientation_mode)
    structure_orientation_frame.pack(fill=tk.X)

    def _refresh_control_panel_theme():
        control_panel.refresh_styles()
        document_tabs.refresh_styles()
        _refresh_workspace_split_style()
        for scale in (
            state.get("angle_x_slider"),
            state.get("angle_y_slider"),
            state.get("rh_angle_z_slider"),
            *state.get("tensor_euler_sliders", {}).values(),
        ):
            if scale is not None:
                _apply_scale_theme(scale, state)

    state["refresh_control_panel_theme"] = _refresh_control_panel_theme

    # User-configurable quick-access launchers. All built-ins remain available
    # through Customize Tools and the existing menus.
    opf = ttk.Frame(tools_section, style="ControlPanel.Form.TFrame")
    opf.pack(fill=tk.X, padx=0, pady=0)
    opf.columnconfigure(0, weight=1)
    opf.columnconfigure(1, weight=1)

    def _refresh_control_panel_tools():
        for child in opf.winfo_children():
            child.destroy()
        preferences = state["app_settings"]["control_panel"]
        tools = ensure_tool_registry(state).pinned(preferences["pinned_tools"])
        if not tools:
            ttk.Label(
                opf,
                text="No pinned tools",
                style="ControlPanel.Muted.TLabel",
            ).grid(row=0, column=0, columnspan=2, pady=panel_metrics.row_y)
            return
        for index, tool in enumerate(tools):
            button = ttk.Button(
                opf,
                text=tool.label,
                style="ControlPanel.Secondary.TButton",
                command=tool.launch,
                state="normal" if tool.is_enabled() else "disabled",
            )
            button.grid(
                row=index // 2,
                column=index % 2,
                sticky="ew",
                padx=panel_metrics.row_y,
                pady=panel_metrics.row_y,
            )
            if tool.tooltip:
                ToolTip(button, tool.tooltip)

    state["refresh_control_panel_tools"] = _refresh_control_panel_tools
    ttk.Button(
        control_panel.section_actions("tools"),
        image=_icon(state, "settings", TOOLBAR_ICON_SIZE),
        style="ControlPanel.HeaderUtility.TButton",
        command=lambda: open_customize_tools_window(state),
    ).pack(side=tk.RIGHT)
    _refresh_control_panel_tools()

    # Checklist
    checklist_frame = ttk.Frame(elements_section, style="ControlPanel.Form.TFrame")
    checklist_frame.pack(fill=tk.X)
    state['checklist_frame'] = checklist_frame
    _show_empty_element_visibility(state)
    # Deleted message
    # ttk.Label(input_frame, text="Rotation and atom selection can be\n applied after updating PCS information.", font=("Helvetica",8,"italic"), justify="center", anchor="center").pack(pady=0)

    _sep(root, orient='horizontal', pady=0, fill='x')

    # # Command bar
    # cmdf = ttk.Frame(root)
    # cmdf.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
    # ttk.Label(cmdf, text="Command :").pack(side=tk.LEFT)
    # command_entry = tk.Entry(cmdf, width=50); command_entry.pack(side=tk.LEFT, padx=5); state['command_entry']=command_entry
    # ttk.Button(cmdf, text="Run", command=lambda: _pc(state)).pack(side=tk.LEFT, padx=5)
    # ttk.Button(cmdf, text="Exit", command=lambda: root.destroy()).pack(side=tk.RIGHT, padx=0)
    #
    # # Export buttons
    # sbf = ttk.Frame(cmdf); sbf.pack(side=tk.RIGHT, padx=15)
    # ttk.Label(sbf, text="Export data ", font=("default",9,"bold")).grid(row=1, column=0, sticky="ew", pady=3)
    # ttk.Button(sbf, text="💾 Save XYZ", command=lambda: on_save_visible_xyz(state)).grid(row=1, column=1, sticky="ew", padx=2, pady=1)
    # ttk.Button(sbf, text="💾 Save plot",  command=lambda: on_save_plot_any(state)).grid( row=1, column=2, sticky="ew", padx=2, pady=1)
    # ttk.Button(sbf, text="💾 Save table", command=lambda: on_save_table_any(state)).grid(row=1, column=3, sticky="ew", padx=2, pady=1)

    # Defaults
    state['delta_exp_values'] = {}
    reset_values(state, record_history=False)  # startup defaults are not a user edit
    # Bindings
    tree.bind("<Double-1>", lambda e: on_delta_entry_change(state, e, state['delta_exp_values'], plot_cartesian_graph))
    from ui.parameter_history import install_center_menu
    install_center_menu(tree, state)
    tree.bind("<<TreeviewSelect>>", lambda e: _on_tree_select_update_views(state))

    selection_hub = ensure_selection_hub(state)
    def _selection_to_main_table(context):
        if context.source == "pcs_table":
            return
        apply_context_to_tree(state, context)
    state["_selection_tree_bridge_token"] = selection_hub.subscribe(
        _selection_to_main_table
    )

    # state['root'].after(150, lambda: open_pcs_plot_popup(state))
    if state["app_settings"].get("open_2d_plot_on_start", True):
        state['root'].after(150, lambda: open_pcs_plot_popup(state))

    state['root'].after(250, lambda: state['update_graph']())

    root.protocol("WM_DELETE_WINDOW", _on_main_close)

    # ============================================================
    # Unified file opening and drag-and-drop support
    # ============================================================
    state["load_structure_file"] = lambda path: _load_xyz_from_path(state, path)
    state["open_project_file"] = _open_project_file_from_path
    state["open_workspace_file"] = _open_workspace_file_from_path
    state["open_file_by_type"] = _open_file_by_type

    try:
        state["drag_drop_enabled"] = enable_main_window_dnd(state)
    except Exception:
        state["drag_drop_enabled"] = False

    # ============================================================
    # Load enabled plugins
    # ============================================================
    state["refresh_plugin_menu"] = lambda: refresh_plugin_menu(state)
    refresh_plugin_menu(state)

    # Establish the clean baseline only after defaults, widgets, variables,
    # and derived single-document state have all been constructed.
    document = capture_active_document(state)
    document.dirty = False
    document.revision = 0
    state["refresh_active_document_commands"]()
    # The menu/history bar is constructed before the parameter widgets. Its
    # initial projection is empty; defaults created later are not pending edits.
    state["edit_history_recorded"]()

    install_main_shortcuts(state)
    return state

def _update_current_tensor_guess_availability(state):
    """Enable canonical initial guesses only for matrix-based fit modes."""
    mode = state['fit_mode_var'].get()
    supported = mode in ('euler_global', 'full_tensor')
    check = state.get('fit_use_current_tensor_check')
    if check is not None:
        check.configure(state='normal' if supported else 'disabled')
        if hasattr(check, 'pack') and hasattr(check, 'pack_forget'):
            if supported:
                check.pack(side=state['tk'].LEFT, padx=3)
            else:
                check.pack_forget()
    hint = state.get('fit_use_current_tensor_hint_var')
    if hint is not None:
        hint.set(
            '' if supported
            else 'Current tensor initialization is available for Mode B and Mode C only.'
        )
    _refresh_fitting_command_state(state)


def _refresh_fitting_command_state(state):
    summary = state.get("refresh_fit_settings_summary")
    if callable(summary):
        summary()
    refresh = state.get("refresh_fitting_commands")
    if callable(refresh):
        refresh()


def _set_standard_fit_running(state, running):
    state["fit_is_running"] = bool(running)
    _refresh_fitting_command_state(state)


def _refresh_fit_result_apply_ui(state):
    """Expose canonical Apply only for complete Mode B/C results."""
    available, message = fit_result_application_status(
        state.get('last_fit_result'),
        state,
    )
    button = state.get('fit_apply_tensor_button')
    if button is not None:
        button.configure(state='normal' if available else 'disabled')
    hint = state.get('fit_apply_tensor_hint_var')
    if hint is not None:
        status = evaluate_fit_result_status(state.get('last_fit_result'), state)
        # Stale/Unknown detail belongs to the Result status area. Keep this
        # helper for action-specific limitations such as Mode A and Mode C.
        hint.set('' if message == status.get('reason') else message)
    _refresh_fitting_command_state(state)


def _open_fit_history_window(state):
    """Open the dedicated non-modal Fit History manager."""
    existing = state.get('fit_history_window')
    if existing is not None:
        try:
            if existing.winfo_exists():
                existing.deiconify()
                existing.lift()
                existing.focus_set()
                _refresh_fit_history_ui(state)
                return existing
        except tk.TclError:
            pass

    window = tk.Toplevel(state['root'])
    window.title("Fit History")
    window.geometry("920x430")
    window.minsize(720, 300)
    state['fit_history_window'] = window

    outer = ttk.Frame(window, padding=8)
    outer.pack(fill=tk.BOTH, expand=True)

    toolbar = ttk.Frame(outer)
    toolbar.pack(fill=tk.X, pady=(0, 6))
    inspect_button = ttk.Button(
        toolbar,
        text="Inspect",
        command=lambda: _on_fit_history_selected(state),
        state="disabled",
    )
    inspect_button.pack(side=tk.LEFT)
    compare_button = ttk.Button(
        toolbar,
        text="Compare",
        command=lambda: _compare_selected_fit_history_ui(state),
        state="disabled",
    )
    compare_button.pack(side=tk.LEFT, padx=(6, 0))

    more_button = ttk.Menubutton(toolbar, text="More ▾")
    more_menu = tk.Menu(more_button, tearoff=False)
    more_menu.add_command(
        label="Label…", command=lambda: _label_selected_fit_history_ui(state)
    )
    more_menu.add_separator()
    more_menu.add_command(
        label="Import Fit Run(s)…", command=lambda: _import_fit_history_runs_ui(state)
    )
    more_menu.add_command(
        label="Export selected as JSON…",
        command=lambda: _export_selected_fit_history_json_ui(state),
    )
    more_menu.add_command(
        label="Export selected as CSV…",
        command=lambda: _export_selected_fit_history_csv_ui(state),
    )
    more_menu.add_separator()
    more_menu.add_command(
        label="Delete…", command=lambda: _delete_selected_fit_history_ui(state)
    )
    more_menu.add_command(
        label="Refresh", command=lambda: _refresh_fit_history_ui(state)
    )
    more_button.configure(menu=more_menu)
    more_button.pack(side=tk.LEFT, padx=(6, 0))

    columns = ("run", "label", "mode", "weighting", "status", "rmsd", "created")
    tree_frame = ttk.Frame(outer)
    tree_frame.pack(fill=tk.BOTH, expand=True)
    tree = ttk.Treeview(
        tree_frame,
        columns=columns,
        show="headings",
        selectmode="extended",
    )
    headings = {
        "run": "Run",
        "label": "Label",
        "mode": "Mode",
        "weighting": "Weighting",
        "status": "Status",
        "rmsd": "RMSD",
        "created": "Created",
    }
    widths = {
        "run": 55,
        "label": 180,
        "mode": 65,
        "weighting": 120,
        "status": 80,
        "rmsd": 80,
        "created": 165,
    }
    for column in columns:
        tree.heading(column, text=headings[column])
        tree.column(
            column, width=widths[column],
            stretch=column in {"label", "created"},
        )
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scrollbar.set)

    state['fit_history_tree'] = tree
    state['fit_history_inspect_button'] = inspect_button
    state['fit_history_compare_button'] = compare_button

    tree.bind(
        "<<TreeviewSelect>>",
        lambda _event: _update_fit_history_window_actions(state),
    )
    tree.bind(
        "<Double-1>",
        lambda _event: _on_fit_history_selected(state),
    )

    def _close():
        state['fit_history_tree'] = None
        state['fit_history_window'] = None
        state.pop('fit_history_inspect_button', None)
        state.pop('fit_history_compare_button', None)
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", _close)
    _refresh_fit_history_ui(state)
    return window


def _update_fit_history_window_actions(state):
    count = len(_fit_history_selected_ids(state))
    inspect = state.get('fit_history_inspect_button')
    compare = state.get('fit_history_compare_button')
    if inspect is not None:
        inspect.configure(state='normal' if count >= 1 else 'disabled')
    if compare is not None:
        compare.configure(state='normal' if count == 2 else 'disabled')


def _fit_history_selected_ids(state):
    tree = state.get('fit_history_tree')
    if tree is None:
        return ()
    return tuple(tree.selection())


def _selected_fit_history_entries(state):
    return [
        entry
        for result_id in _fit_history_selected_ids(state)
        if (entry := find_fit_history_entry(state, result_id)) is not None
    ]


def _refresh_fit_history_ui(state):
    entries = sorted(
        fit_history_entries(state), key=lambda item: item.sequence, reverse=True
    )

    summary = state.get('fit_latest_run_summary_var')
    if summary is not None:
        if entries:
            latest = entries[0]
            result = latest.result()
            mode = {
                'theta_alpha_multi': 'Mode A',
                'euler_global': 'Mode B',
                'full_tensor': 'Mode C',
            }.get(result.get('mode'), 'Unknown mode')
            try:
                status = evaluate_fit_result_status(result, state)['status'].capitalize()
            except Exception:
                status = 'Unknown'
            try:
                rmsd = f"{float(result.get('rmsd')):.4f} ppm"
            except (TypeError, ValueError):
                rmsd = "RMSD unavailable"
            label = f" · {latest.label}" if latest.label else ""
            summary.set(
                f"Latest: Run #{latest.sequence}{label} · {mode} · {status} · RMSD {rmsd}"
            )
        else:
            summary.set("No fit runs yet.")

    tree = state.get('fit_history_tree')
    if tree is None:
        return
    try:
        children = tree.get_children()
    except tk.TclError:
        state['fit_history_tree'] = None
        return
    for item_id in children:
        tree.delete(item_id)

    for entry in entries:
        result = entry.result()
        try:
            status = evaluate_fit_result_status(result, state)['status'].capitalize()
        except Exception:
            status = 'Unknown'
        metadata = result.get('fit_metadata') or {}
        weighting = fit_weighting_label(metadata.get('weighting_mode', 'unweighted'))
        rmsd = result.get('rmsd')
        try:
            rmsd_text = f"{float(rmsd):.4f}"
        except (TypeError, ValueError):
            rmsd_text = "—"
        mode = {'theta_alpha_multi': 'A', 'euler_global': 'B', 'full_tensor': 'C'}.get(
            result.get('mode'), '?'
        )
        created = str(entry.created_at).replace('T', ' ').replace('Z', ' UTC')
        tree.insert(
            '', 'end', iid=entry.result_id,
            values=(
                f"#{entry.sequence}", entry.label, mode, weighting,
                status, rmsd_text, created,
            ),
        )

    active = state.get('active_fit_result_id')
    if active and tree.exists(active):
        tree.selection_set(active)
        tree.focus(active)
        tree.see(active)
    _update_fit_history_window_actions(state)


def _on_fit_history_selected(state):
    tree = state.get('fit_history_tree')
    selected = _fit_history_selected_ids(state)
    if tree is None or not selected:
        return
    focused = tree.focus()
    result_id = focused if focused in selected else selected[0]
    try:
        select_fit_history_result(state, result_id)
    except KeyError:
        return
    state.pop('_last_fit_result_restored', None)
    _refresh_last_fit_result_ui(state)


def _label_selected_fit_history_ui(state):
    selected = _fit_history_selected_ids(state)
    if len(selected) != 1:
        state['messagebox'].showinfo("Fit History", "Select one fit run to label.")
        return False
    entry = find_fit_history_entry(state, selected[0])
    if entry is None:
        return False
    label = simpledialog.askstring(
        "Fit History label",
        "Optional label:",
        initialvalue=entry.label,
        parent=state.get('root'),
    )
    if label is None:
        return False
    label_fit_history_entry(state, entry.result_id, label.strip())
    _refresh_fit_history_ui(state)
    return True


def _delete_selected_fit_history_ui(state):
    selected = _fit_history_selected_ids(state)
    if len(selected) != 1:
        state['messagebox'].showinfo("Fit History", "Select one fit run to delete.")
        return False
    if not state['messagebox'].askyesno(
        "Delete fit run", "Delete the selected fit-history entry?"
    ):
        return False
    if not delete_fit_history_entry(state, selected[0]):
        return False
    _refresh_fit_history_ui(state)
    _refresh_last_fit_result_ui(state)
    return True


def _comparison_text(left_entry, right_entry, comparison):
    lines = [
        f"Run A: #{left_entry.sequence} {left_entry.label}".rstrip(),
        f"Run B: #{right_entry.sequence} {right_entry.label}".rstrip(),
        "",
    ]
    for warning in comparison['warnings']:
        lines.append(f"Warning: {warning}")
    if comparison['warnings']:
        lines.append("")
    lines.append(f"{'Field':<24} {'Run A':<28} {'Run B':<28} Difference")
    lines.append("-" * 100)
    for name, (left, right) in comparison['fields'].items():
        difference = comparison['differences'].get(name)
        difference_text = "" if difference is None else f"{difference:+.6g}"
        lines.append(f"{name:<24} {str(left):<28.28} {str(right):<28.28} {difference_text}")
    return "\n".join(lines)


def _compare_selected_fit_history_ui(state):
    selected = _fit_history_selected_ids(state)
    if len(selected) != 2:
        state['messagebox'].showinfo("Compare fit runs", "Select exactly two fit runs.")
        return False
    left_entry = find_fit_history_entry(state, selected[0])
    right_entry = find_fit_history_entry(state, selected[1])
    if left_entry is None or right_entry is None:
        return False
    comparison = compare_fit_results(left_entry.result(), right_entry.result(), state=state)
    window = tk.Toplevel(state['root'])
    window.title("Fit Result Comparison")
    window.geometry("980x520")
    text = tk.Text(window, wrap="none", font=ui_font_name(window, "report"))
    text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
    text.insert("1.0", _comparison_text(left_entry, right_entry, comparison))
    text.configure(state="disabled")
    return True


def _export_selected_fit_history_json_ui(state):
    entries = _selected_fit_history_entries(state)
    if not entries:
        state['messagebox'].showinfo("Export fit history", "Select one or more fit runs.")
        return False
    basename = default_fit_history_basename(entries)
    path = state['filedialog'].asksaveasfilename(
        title="Export selected fit run(s) as JSON",
        defaultextension=".json",
        initialfile=f"{basename}.json",
        filetypes=[("JSON", "*.json"), ("All files", "*.*")],
    )
    if not path:
        return False
    payload = (
        build_fit_run_export(entries[0], state)
        if len(entries) == 1
        else build_fit_history_export(entries, state)
    )
    try:
        write_json_export(path, payload)
    except (OSError, TypeError, ValueError) as exc:
        state['messagebox'].showerror("Export fit history", str(exc))
        return False
    state['messagebox'].showinfo("Export fit history", f"Saved JSON:\n{path}")
    return True


def _export_selected_fit_history_csv_ui(state):
    entries = _selected_fit_history_entries(state)
    if not entries:
        state['messagebox'].showinfo("Export fit history", "Select one or more fit runs.")
        return False
    basename = default_fit_history_basename(entries)
    chosen = state['filedialog'].asksaveasfilename(
        title="Export selected fit run(s) as CSV",
        defaultextension=".csv",
        initialfile=f"{basename}.csv",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
    )
    if not chosen:
        return False
    base, _extension = os.path.splitext(chosen)
    summary_path = f"{base}_summary.csv"
    observable_path = f"{base}_observables.csv"
    try:
        write_fit_summary_csv(summary_path, entries, state)
        if len(entries) == 1:
            write_fit_observables_csv(observable_path, entries[0])
    except (OSError, TypeError, ValueError) as exc:
        state['messagebox'].showerror("Export fit history", str(exc))
        return False
    paths = summary_path
    if len(entries) == 1:
        paths += f"\n{observable_path}"
    state['messagebox'].showinfo("Export fit history", f"Saved CSV:\n{paths}")
    return True


def _import_fit_history_runs_ui(state):
    path = state['filedialog'].askopenfilename(
        title="Import Fit Run(s) into History",
        filetypes=[("Fit run JSON", "*.json"), ("All files", "*.*")],
    )
    if not path:
        return False
    try:
        report = import_fit_run_export_file(state, path)
    except (FitRunImportError, OSError, TypeError, ValueError) as exc:
        state['messagebox'].showerror("Import Fit Run(s)", str(exc))
        return False

    if report.imported:
        _refresh_fit_history_ui(state)
        _refresh_last_fit_result_ui(state)
    message = report.summary()
    if report.rejected:
        details = "\n".join(
            f"Run {item.index + 1}: {item.reason}"
            for item in report.rejected[:5]
        )
        if len(report.rejected) > 5:
            details += f"\n… and {len(report.rejected) - 5} more"
        message += f"\n\n{details}"
        show = getattr(state['messagebox'], 'showwarning', state['messagebox'].showinfo)
        show("Import Fit Run(s)", message)
    else:
        state['messagebox'].showinfo("Import Fit Run(s)", message)
    return bool(report.imported or report.skipped_duplicate_ids)


def _refresh_fit_result_validity_ui(state):
    result = state.get('last_fit_result')
    status = evaluate_fit_result_status(result, state)
    label = status["status"].capitalize()
    value = state.get('fit_result_validity_var')
    if value is not None:
        value.set(label)
    status_label = state.get('fit_result_validity_label')
    if status_label is not None:
        color = {
            'current': '#2e7d32',
            'stale': '#a15c00',
            'unknown': '#666666',
        }.get(status['status'], '#666666')
        status_label.configure(foreground=color)
    hint = state.get('fit_result_validity_hint_var')
    if hint is not None:
        hint.set("" if status["status"] == FIT_RESULT_CURRENT else status["reason"])
    state['_last_fit_result_status'] = status
    return status


def apply_fit_result_to_current_tensor_ui(state):
    """Apply the inspected fit result through one canonical tensor command."""
    dispatch_warning = False
    try:
        changed = apply_fit_result_to_canonical_tensor(state)
    except FitResultApplicationError as exc:
        state['fit_status_var'].set("Fit result was not applied.")
        state['messagebox'].showerror("Fitting", str(exc))
        _refresh_fit_result_apply_ui(state)
        return False
    except PCSDispatchError as exc:
        # update_canonical_tensor installs the new tensor before dispatching.
        state["_pcs_model_event_dispatch_failures"] = exc.failures
        changed = True
        dispatch_warning = True

    refresh_orientation = state.get("refresh_orientation_editor")
    if callable(refresh_orientation):
        refresh_orientation()
    if dispatch_warning:
        state['fit_status_var'].set("Fit result applied with refresh warnings.")
    elif changed:
        state['fit_status_var'].set("Fit result applied to current tensor.")
    else:
        state['fit_status_var'].set("Current tensor already matches the fit result.")
    _refresh_fit_result_apply_ui(state)
    return changed


def _refresh_last_fit_result_ui(state):
    """Present a fresh or restored result without rerunning fitting."""
    result = state.get('last_fit_result')
    summary = state.get('fit_result_summary_var')
    if result:
        if summary is not None:
            mode = {
                'theta_alpha_multi': 'Mode A',
                'euler_global': 'Mode B',
                'full_tensor': 'Mode C',
            }.get(result.get('mode'), 'Unknown mode')
            parts = [mode, f"N={result.get('n', 0)}"]
            try:
                parts.append(f"RMSD {float(result.get('rmsd')):.4f} ppm")
            except (TypeError, ValueError):
                pass
            try:
                parts.append(f"R² {float(result.get('r2')):.4f}")
            except (TypeError, ValueError):
                pass
            summary.set(" · ".join(parts))
        _show_fit_result(state, result)
        update_fit_correlation_plot_from_result(state, result)
    else:
        if summary is not None:
            summary.set("No fit result.")
        box = state.get('fit_result_box')
        if box is not None:
            box.delete("1.0", "end")
        update_fit_correlation_plot_from_result(state, {})
    _refresh_fit_result_validity_ui(state)
    _refresh_fit_result_apply_ui(state)

    status = state.get('fit_status_var')
    if status is not None:
        if state.get('_last_fit_result_restored'):
            status.set("Restored fit result.")
        elif state.get('_last_fit_result_restore_error'):
            status.set("Saved fit result could not be restored.")
        elif state.get('_last_fit_result_load_processed'):
            status.set("No saved fit result.")


def _sync_fit_mode_details(state):
    """Reflect radio/project/document mode changes without repopulating inputs."""
    m = state['fit_mode_var'].get()
    state['fit_geometry_label_var'].set({
        'theta_alpha_multi': 'Donor atoms',
        'euler_global': 'Euler orientation',
        'full_tensor': 'Orientation and metal position',
    }[m])

    # Traces also run on document activation. Leave the already visible frame
    # in place to avoid redundant geometry work when the radio is invoked.
    if m != 'theta_alpha_multi':
        state['fit_frameA'].pack_forget()
    if m != 'euler_global':
        state['fit_frameB'].pack_forget()
    if m != 'full_tensor':
        state['fit_frameC'].pack_forget()
    if m == 'theta_alpha_multi' and not state['fit_frameA'].winfo_manager():
        state['fit_frameA'].pack(fill=state['tk'].BOTH, expand=True)
    elif m == 'euler_global' and not state['fit_frameB'].winfo_manager():
        state['fit_frameB'].pack(fill=state['tk'].X, pady=4)
    elif m == 'full_tensor' and not state['fit_frameC'].winfo_manager():
        state['fit_frameC'].pack(fill=state['tk'].X, pady=4)


def switch_fit_mode(state):
    _sync_fit_mode_details(state)
    _update_current_tensor_guess_availability(state)

    try:
        populate_fitting_controls(state)
    except Exception:
        pass
    _refresh_fitting_command_state(state)

def run_fit_from_ui(state):
    worker_state = worker_document_state(state)
    worker_identity = capture_worker_identity(state)
    state["standard_fit_worker_identity"] = worker_identity

    def _worker_result_is_current():
        return worker_identity is None or worker_completion_matches(
            state, worker_identity
        )

    mode = state['fit_mode_var'].get()
    weighting_var = state.get('fit_weighting_var')
    weighting_mode = parse_fit_weighting_mode(
        weighting_var.get() if weighting_var is not None else None
    ).value
    use_current_tensor_var = state.get('fit_use_current_tensor_var')
    use_current_tensor = bool(
        use_current_tensor_var.get()
        if use_current_tensor_var is not None
        else False
    )
    axis_mode = state['axis_mode_var'].get()
    fit_visible_as_group = state['fit_use_visible_var'].get()
    fit_delta_chi = state['fit_dchi_var'].get()
    fit_delta_chi_rh = state['fit_dchi_rh_var'].get()
    use_global_search = state['fit_global_search_var'].get()

    psel = state['fit_proton_list'].curselection()
    if not psel:
        state['messagebox'].showerror("Fitting", "Select PCS observations to fit.")
        return
    selectors = state.get('fit_observation_selectors')
    if selectors is None:
        proton_ids = [int(state['fit_proton_list'].get(i)) for i in psel]
        observation_ids = None
    else:
        proton_ids = []
        observation_ids = [selectors[i] for i in psel]

    if mode == 'theta_alpha_multi':
        dsel = state['fit_donor_list'].curselection()
        if not dsel:
            state['messagebox'].showerror("Fitting", "Select ≥1 donor IDs.")
            return
        donor_ids = [int(state['fit_donor_list'].get(i).split(':')[0]) for i in dsel]
    else:
        donor_ids = []

    def _finish_success(res):
        _set_standard_fit_running(state, False)
        if not _worker_result_is_current():
            state['fit_status_var'].set(
                "Fit result discarded because its document changed."
            )
            return
        origin = (
            state["document_manager"].document_by_id(worker_identity.document_id)
            if worker_identity is not None and state.get("document_manager") is not None
            else None
        )
        if origin is not None and origin is not active_document(state):
            values = origin.snapshot()
            if res.get('result_schema') == PCS_FIT_RESULT_SCHEMA:
                append_fit_history_result(values, res)
            else:
                values['last_fit_result'] = res
            for key in (
                '_last_fit_result_restored', '_last_fit_result_restore_error',
                '_last_fit_result_load_processed',
            ):
                values.pop(key, None)
            apply_worker_document_updates(
                state, worker_identity, values, domain="standard_fit_result"
            )
            return
        try:
            document_state = active_document_view(state)
        except RuntimeError:
            # Keep the fitting presenter usable for isolated legacy callers
            # that have not installed a PCSDocument yet.
            document_state = state
        if res.get('result_schema') == PCS_FIT_RESULT_SCHEMA:
            append_fit_history_result(state, res)
        else:
            # Compatibility for external/legacy fit producers that do not yet
            # emit a complete persistable pcs_fit_result_v1 payload.
            document_state['last_fit_result'] = res
        document_state.pop('_last_fit_result_restored', None)
        document_state.pop('_last_fit_result_restore_error', None)
        document_state.pop('_last_fit_result_load_processed', None)
        mark_document_mutated(state, "standard_fit_result")
        _refresh_fit_history_ui(state)
        _refresh_last_fit_result_ui(state)
        state['fit_status_var'].set("Fit completed.")

    def _finish_error(msg):
        _set_standard_fit_running(state, False)
        if not _worker_result_is_current():
            state['fit_status_var'].set(
                "Fit stopped because its document changed."
            )
            return
        state['fit_status_var'].set("Fit failed.")
        state['messagebox'].showerror("Fitting", msg)

    def _worker():
        try:
            state['root'].after(
                0,
                lambda: (
                    state['fit_status_var'].set("Collecting atoms...")
                    if _worker_result_is_current()
                    else None
                ),
            )

            if mode == 'theta_alpha_multi':
                res = fit_theta_alpha_multi(
                    worker_state,
                    donor_ids=donor_ids,
                    proton_ids=proton_ids,
                    observation_ids=observation_ids,
                    axis_mode=axis_mode,
                    fit_visible_as_group=fit_visible_as_group,
                    fit_delta_chi=fit_delta_chi,
                    fit_delta_chi_rh=fit_delta_chi_rh,
                    use_global_search=use_global_search,
                    weighting_mode=weighting_mode,
                    progress_cb=state.get('fit_progress_cb'),
                )

            elif mode == 'euler_global':
                res = fit_euler_global(
                    worker_state,
                    proton_ids=proton_ids,
                    observation_ids=observation_ids,
                    fit_visible_as_group=fit_visible_as_group,
                    fit_delta_chi=fit_delta_chi,
                    fit_delta_chi_rh=fit_delta_chi_rh,
                    use_global_search=use_global_search,
                    use_current_tensor=use_current_tensor,
                    weighting_mode=weighting_mode,
                    progress_cb=state.get('fit_progress_cb'),
                )

            elif mode == 'full_tensor':
                res = fit_full_tensor(
                    worker_state,
                    proton_ids=proton_ids,
                    observation_ids=observation_ids,
                    fit_delta_chi=fit_delta_chi,
                    fit_delta_chi_rh=fit_delta_chi_rh,
                    use_global_search=use_global_search,
                    use_current_tensor=use_current_tensor,
                    weighting_mode=weighting_mode,
                    progress_cb=state.get('fit_progress_cb'),
                )

            else:
                raise ValueError(f"Unknown fitting mode: {mode}")

            state['root'].after(0, lambda r=res: _finish_success(r))

        except Exception as e:
            state['root'].after(0, lambda m=str(e): _finish_error(m))

    state['fit_status_var'].set("Preparing fit...")
    _set_standard_fit_running(state, True)
    try:
        threading.Thread(target=_worker, daemon=True).start()
    except Exception:
        _set_standard_fit_running(state, False)
        raise

def _show_fit_result(state, res):
    import numpy as np

    box = state['fit_result_box']
    box.delete("1.0", "end")
    from ui.fitting_presentation import format_fit_conditions
    box.insert("end", format_fit_conditions(res))

    state.setdefault("rh_ratio_warn_threshold", 0.5)

    mode = res.get('mode')
    per_point = list(res.get('per_point', []) or [])

    atom_by_id = state.get("atom_by_id", {}) or {}
    label_overrides = state.get("ref_label_overrides", {}) or {}

    def atom_label(rid):
        return label_overrides.get(rid, atom_by_id.get(rid, ""))

    def add(line=""):
        box.insert("end", line + "\n")

    def sec(title, char="="):
        bar = char * 78
        add(bar)
        add(title)
        add(bar)

    def fmt_float(v, nd=3, signed=False):
        try:
            v = float(v)
            if np.isnan(v):
                return "nan"
            return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
        except Exception:
            return "n/a"

    def fmt_sci(v, nd=3):
        try:
            v = float(v)
            if np.isnan(v):
                return "nan"
            return f"{v:.{nd}e}"
        except Exception:
            return "n/a"

    def fmt_list(vals):
        if not vals:
            return "-"
        return ", ".join(str(v) for v in vals)

    def status_suffix(name):
        status = parameter_status(res, name)
        return f"  [{status}]" if status else ""

    ref_w = max(3, max((len(str(rid)) for rid, *_ in per_point), default=3))
    atom_w = max(4, min(18, max((len(str(atom_label(rid))) for rid, *_ in per_point), default=4)))

    top_res = sorted(per_point, key=lambda x: abs(float(x[3])), reverse=True)[:5]

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    if mode == 'theta_alpha_multi':
        title = "Mode A  |  Ligand-donor axis fit"
        subtitle = f"Axis mode: {res.get('axis_mode', 'n/a')}"
    elif mode == 'euler_global':
        title = "Mode B  |  Global Euler fit"
        subtitle = "Rigid-body rotation in the global frame"
    elif mode == 'full_tensor':
        title = "Mode C  |  Full tensor / metal-position fit"
        subtitle = "Euler + metal position + tensor parameters"
    else:
        title = f"Unknown mode  |  {mode}"
        subtitle = ""

    sec(title, "=")
    if subtitle:
        add(subtitle)
    add()

    # ------------------------------------------------------------------
    # Section 1: Optimized parameters
    # ------------------------------------------------------------------
    sec("Optimized parameters", "-")

    if mode == 'theta_alpha_multi':
        add(f"Donor IDs               : {fmt_list(res.get('donor_ids', []))}")
        add(f"θ*                      : {fmt_float(res.get('theta'), 2)}°{status_suffix('theta')}")
        add(f"α*                      : {fmt_float(res.get('alpha'), 2)}°{status_suffix('alpha')}")

    elif mode == 'euler_global':
        orientation_status = status_suffix('orientation')
        add(f"ax                      : {fmt_float(res.get('ax'), 2)}°{orientation_status}")
        add(f"ay                      : {fmt_float(res.get('ay'), 2)}°{orientation_status}")
        add(f"az                      : {fmt_float(res.get('az'), 2)}°{orientation_status}")

    elif mode == 'full_tensor':
        if 'metal_pos' in res:
            mx, my, mz = res['metal_pos']
            add(f"Metal position          : ({mx:.3f}, {my:.3f}, {mz:.3f}){status_suffix('metal_pos')}")
        if 'euler_deg' in res:
            ax, ay, az = res['euler_deg']
            add(f"XYZ (ax, ay, az)        : ({ax:.2f}°, {ay:.2f}°, {az:.2f}°){status_suffix('orientation')}")

    add(f"Δχ_ax                   : {fmt_sci(res.get('delta_chi_ax'))}  (E-32 m³){status_suffix('delta_chi_ax')}")

    if 'delta_chi_rh' in res:
        dchi_ax = float(res.get('delta_chi_ax', np.nan))
        dchi_rh = float(res.get('delta_chi_rh', np.nan))
        add(f"Δχ_rh                   : {fmt_sci(dchi_rh)}  (E-32 m³){status_suffix('delta_chi_rh')}")

        if np.isfinite(dchi_ax) and abs(dchi_ax) > 0:
            ratio = abs(dchi_rh / dchi_ax)
            add(f"|Δχ_rh / Δχ_ax|         : {ratio:.3f}")

            thr = float(state.get("rh_ratio_warn_threshold", 0.5))
            if ratio > thr:
                add(f"Warning                 : |Δχ_rh/Δχ_ax| > {thr:.2f}")
                add("                          Rhombic fit may be unstable or")
                add("                          the φ-reference may be ill-defined.")
        else:
            add("|Δχ_rh / Δχ_ax|         : undefined (Δχ_ax = 0)")

    add()

    # ------------------------------------------------------------------
    # Section 2: Reference / frame information
    # ------------------------------------------------------------------
    sec("Reference / frame information", "-")

    if mode == 'theta_alpha_multi':
        try:
            az_ref = float(state.get('angle_z_var', 0.0).get())
            add(f"φ reference             : user z-rotation = {az_ref:.1f}°")
        except Exception:
            add("φ reference             : n/a")

    elif mode == 'euler_global':
        add("φ reference             : Euler convention")
        add(f"z-angle reference       : az = {fmt_float(res.get('az'), 2)}°")

    elif mode == 'full_tensor':
        if 'euler_deg' in res:
            _, _, eg = res['euler_deg']
            add("φ reference             : tensor-frame Euler convention")
            add(f"z-angle reference       : γ = {eg:.2f}°")
        else:
            add("φ reference             : tensor-frame convention")

    else:
        add("φ reference             : n/a")

    add()

    # ------------------------------------------------------------------
    # Section 3: Fit quality
    # ------------------------------------------------------------------
    sec("Fit quality", "-")

    add(f"N points                : {res.get('n', 0)}")
    metadata = res.get('fit_metadata') or {}
    weighting_mode = metadata.get('weighting_mode', FitWeightingMode.UNWEIGHTED.value)
    try:
        weighting_text = fit_weighting_label(weighting_mode)
    except ValueError:
        weighting_text = str(weighting_mode)
    add(f"Residual weighting      : {weighting_text}")
    add(f"RMSD                    : {fmt_float(res.get('rmsd'), 4)} ppm")

    def finite_metric(value):
        try:
            return np.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    r2 = res.get('r2')
    if finite_metric(r2):
        add(f"R²                      : {r2:.4f}")

    qf = res.get('q_factor')
    if finite_metric(qf):
        add(f"Q-factor                : {qf:.4f}")

    chi2_n = res.get('chi2_n')
    if finite_metric(chi2_n):
        add(f"Raw SSE/N               : {chi2_n:.4f} ppm²")

    chi_square = res.get('chi_square')
    if finite_metric(chi_square):
        add(f"χ² = Σ(resid/σ)²      : {chi_square:.4f}")

    chi_square_per_datum = res.get('chi_square_per_datum')
    if finite_metric(chi_square_per_datum):
        add(f"χ²/N (weighted)         : {chi_square_per_datum:.4f}")

    add()

    # ------------------------------------------------------------------
    # Section 4: Largest residuals
    # ------------------------------------------------------------------
    sec("Largest residuals (top 5 by |resid|)", "-")

    if top_res:
        add(
            f"{'Ref':>{ref_w}}   "
            f"{'Atom':<{atom_w}}   "
            f"{'δ_exp':>10}   "
            f"{'δ_pred':>10}   "
            f"{'resid':>10}"
        )
        add("-" * (ref_w + atom_w + 39))

        for rid, exp, pred, r in top_res:
            add(
                f"{rid:>{ref_w}}   "
                f"{str(atom_label(rid)):<{atom_w}.{atom_w}}   "
                f"{exp:>10.3f}   "
                f"{pred:>10.3f}   "
                f"{r:>+10.3f}"
            )
    else:
        add("No fitted points.")

    add()

    # ------------------------------------------------------------------
    # Section 5: Full per-point results
    # ------------------------------------------------------------------
    sec("Per-point results", "-")

    if per_point:
        add(
            f"{'Ref':>{ref_w}}   "
            f"{'Atom':<{atom_w}}   "
            f"{'δ_exp':>10}   "
            f"{'δ_pred':>10}   "
            f"{'resid':>10}"
        )
        add("-" * (ref_w + atom_w + 39))

        for rid, exp, pred, r in per_point:
            add(
                f"{rid:>{ref_w}}   "
                f"{str(atom_label(rid)):<{atom_w}.{atom_w}}   "
                f"{exp:>10.3f}   "
                f"{pred:>10.3f}   "
                f"{r:>+10.3f}"
            )
    else:
        add("No fitted points.")

# Fitting function - correlation plot
def update_fit_correlation_plot_from_result(state, res):
    fig = state.get('fit_corr_fig')
    canvas = state.get('fit_corr_canvas')
    if fig is None or canvas is None:
        return

    per_point = res.get('per_point', [])
    per_observable = res.get('per_observable', [])
    fig.clear()

    if not per_point:
        from ui.fitting_presentation import show_fit_empty_guide
        show_fit_empty_guide(fig, state)
        canvas.draw_idle()
        return

    if per_observable and len(per_observable) == len(per_point):
        exp_values = np.array([row['observed'] for row in per_observable], float)
        pred_values = np.array([row['predicted'] for row in per_observable], float)
        resid_values = np.array([row['residual'] for row in per_observable], float)
        ref_ids = [
            row.get('assignment') or (
                row.get('primary_ref_id')
                if str(row.get('observable_id', '')).startswith('legacy-ref:')
                else row.get('observable_id')
            ) or row.get('primary_ref_id')
            for row in per_observable
        ]
    else:
        exp_values = np.array([row[1] for row in per_point], float)
        pred_values = np.array([row[2] for row in per_point], float)
        resid_values = np.array([row[3] for row in per_point], float)
        ref_ids = [row[0] for row in per_point]

    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.0])
    fonts = state.get("fonts", {}) or {}
    title_size = fonts.get("plot_title", 10)
    label_size = fonts.get("plot_label", 9)
    tick_size = fonts.get("plot_tick", 8)

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
            fontsize=tick_size,
            color='dimgray'
        )

    ax_corr.set_xlim(lo, hi)
    ax_corr.set_ylim(lo, hi)
    ax_corr.set_xlabel("Experimental shift\n" + r"$\delta_{Exp}$ (ppm)", fontsize=label_size)
    ax_corr.set_ylabel("Predicted shift\n" + r"$\delta_{Pred}$ (ppm)", fontsize=label_size)
    ax_corr.set_title("Fit correlation", fontsize=title_size)
    ax_corr.tick_params(labelsize=tick_size)
    ax_corr.grid(alpha=0.2)

    cbar = fig.colorbar(sc, ax=ax_corr, fraction=0.04, pad=0.02)
    cbar.set_label("|Residual| (ppm)", fontsize=label_size)
    cbar.ax.tick_params(labelsize=tick_size)

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
    ax_res.set_ylabel("Residual (ppm)", fontsize=label_size)
    ax_res.set_xlabel("Ref / observation ID", fontsize=label_size)
    ax_res.tick_params(labelsize=tick_size)
    ax_res.set_xticks(np.arange(len(ref_ids)))
    ax_res.set_xticklabels([str(r) for r in ref_ids], rotation=45, ha='right', fontsize=tick_size)
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

    dpi = _get_export_dpi(state, fallback=600)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    state['messagebox'].showinfo("Export", f"Saved fit plot ({dpi} dpi):\n{path}")

def _current_export_geom_values(state):
    """Return displayed G values in current effective-atom order."""
    ids = state.get("current_selected_ids", []) or []
    geom_by_id = state.get("geom_by_id", {}) or {}
    return [geom_by_id.get(rid, np.nan) for rid in ids]

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
        pcs_values, theta_values, tensor, polar_data, _ = recompute_plot_inputs(state)
        try:
            from logic.export_utils import save_to_excel
            save_to_excel(pcs_values, theta_values, tensor, fd, polar_data, geom_values=_current_export_geom_values(state))
        except ModuleNotFoundError as e:
            if "openpyxl" in str(e):
                state['messagebox'].showerror(
                    "Export",
                    "Excel export requires the 'openpyxl' package.\n\n"
                    "Install it with:\npython -m pip install openpyxl"
                )
                return
            raise
        state['messagebox'].showinfo("Export", f"Saved Excel:\n{fd}")

    elif ext == ".csv":
        pcs_values, theta_values, tensor, polar_data, _ = recompute_plot_inputs(state)
        from logic.export_utils import save_to_csv
        pcs_path, atoms_path = save_to_csv(pcs_values, theta_values, tensor, fd, polar_data, geom_values=_current_export_geom_values(state))
        state['messagebox'].showinfo("Export", f"Saved CSV:\n{pcs_path}\n{atoms_path}")

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
        dpi = _get_export_dpi(state, fallback=600)
        canvas.print_figure(fd, dpi=dpi)
        state['messagebox'].showinfo("Export", f"Saved PNG ({dpi} dpi):\n{fd}")

    else:
        # 확장자 없으면 .xlsx로 저장
        fd_x = base + ".xlsx"
        pcs_values, theta_values, tensor, polar_data, _ = recompute_plot_inputs(state)
        try:
            from logic.export_utils import save_to_excel
            save_to_excel(pcs_values, theta_values, tensor, fd_x, polar_data, geom_values=_current_export_geom_values(state))
        except ModuleNotFoundError as e:
            if "openpyxl" in str(e):
                state['messagebox'].showerror(
                    "Export",
                    "Excel export requires the 'openpyxl' package.\n\n"
                    "Install it with:\n"
                    "python -m pip install openpyxl"
                )
                return
            raise
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
        try:
            from logic.export_utils import export_tables_to_excel
            export_tables_to_excel(state['tree'], fd, rh_tree=rh_tree)
        except ModuleNotFoundError as e:
            if "openpyxl" in str(e):
                state['messagebox'].showerror(
                    "Export",
                    "Excel export requires the 'openpyxl' package.\n\n"
                    "Install it with:\npython -m pip install openpyxl"
                )
                return
            raise
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
        try:
            from logic.export_utils import export_tables_to_excel
            export_tables_to_excel(state['tree'], fd_x, rh_tree=rh_tree)
        except ModuleNotFoundError as e:
            if "openpyxl" in str(e):
                state['messagebox'].showerror(
                    "Export",
                    "Excel export requires the 'openpyxl' package.\n\n"
                    "Install it with:\npython -m pip install openpyxl"
                )
                return
            raise
        state['messagebox'].showinfo("Export", f"No/unknown extension. Saved as Excel:\n{fd_x}")

# Save .xyz file related helper
def get_visible_xyz_rows(state):
    """
    Return the currently visible structure in the same metal-centered,
    rotated coordinate frame used by the 2D/table pipeline.

    Returns:
        list of (element, x, y, z)
    """
    atom_data = state.get("atom_data_eff") or state.get("atom_data") or []
    ids = state.get("atom_ids_eff") or state.get("atom_ids") or list(range(1, len(atom_data) + 1))
    visible_ids = visible_ref_id_set(state, atom_data=atom_data, ref_ids=ids)

    if not atom_data:
        return []

    abs_coords = np.array([[x, y, z] for a, x, y, z in atom_data], dtype=float)
    metal = np.array([state['x0'], state['y0'], state['z0']], dtype=float)

    fo = state.get('fit_override')
    if fo:
        mode = (fo.get('mode') or '').lower()

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

        elif mode == 'euler_global':
            ax = float(fo.get('ax', 0.0))
            ay = float(fo.get('ay', 0.0))
            az = float(fo.get('az', 0.0))
            coords0 = abs_coords - metal
            rot0 = rotate_euler(coords0, ax, ay, az)
            abs_coords = rot0 + metal

        elif mode == 'full_tensor':
            euler_deg = fo.get('euler_deg', (0.0, 0.0, 0.0))
            ax_e, ay_e, az_e = euler_deg

            metal_pos = fo.get('metal_pos')
            if metal_pos is not None:
                metal = np.array(metal_pos, dtype=float)

            coords0 = abs_coords - metal
            rot0 = rotate_euler(coords0, ax_e, ay_e, az_e)
            abs_coords = rot0 + metal

    # metal-centered coordinates
    coords0 = abs_coords - metal

    ax = float(state['angle_x_var'].get())
    ay = float(state['angle_y_var'].get())
    az = float(state['angle_z_var'].get())
    rotated = rotate_coordinates(coords0, ax, ay, az, (0, 0, 0))

    rows = []
    for rid, (atom, *_), (x, y, z) in zip(ids, atom_data, rotated):
        if int(rid) in visible_ids:
            rows.append((atom, float(x), float(y), float(z)))

    return rows

def on_save_visible_xyz(state):
    rows = get_visible_xyz_rows(state)
    if not rows:
        state['messagebox'].showwarning("Export XYZ", "No visible structure data to save.")
        return

    fd = state['filedialog'].asksaveasfilename(
        title="Save visible structure as XYZ",
        defaultextension=".xyz",
        filetypes=[("XYZ file", "*.xyz"), ("All files", "*.*")]
    )
    if not fd:
        return

    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(f"{len(rows)}\n")
            f.write("Visible structure from PCS Analyzer (metal-centered, rotated view)\n")
            for atom, x, y, z in rows:
                f.write(f"{atom:2s} {x: .8f} {y: .8f} {z: .8f}\n")

        state['messagebox'].showinfo("Export XYZ", f"Saved visible XYZ:\n{fd}")
    except Exception as e:
        state['messagebox'].showerror("Export XYZ", f"Failed to save XYZ:\n{e}")

def recompute_plot_inputs(state):
    tensor = state['tensor_entry'].get()
    tensor = float(tensor) if tensor else 1.0
    pcs_min = float(state['pcs_min_entry'].get()); pcs_max = float(state['pcs_max_entry'].get()); pcs_interval = float(state['pcs_interval_entry'].get())
    pcs_values = np.arange(pcs_min, pcs_max + pcs_interval, pcs_interval)
    theta_values = state.get('theta_values')
    if theta_values is None:
        theta_values = np.linspace(0, 2*np.pi, 500); state['theta_values']=theta_values
    polar_data, rotated_coords = filter_atoms(state)
    return pcs_values, theta_values, tensor, polar_data, rotated_coords

def update_graph(state):
    tensor = state['tensor_entry'].get()
    tensor = float(tensor) if tensor else 1.0
    pcs_min = float(state['pcs_min_entry'].get())
    pcs_max = float(state['pcs_max_entry'].get())
    pcs_interval = float(state['pcs_interval_entry'].get())
    if not np.all(np.isfinite((pcs_min, pcs_max, pcs_interval))):
        raise ValueError("PCS range and interval must be finite numbers.")
    if pcs_interval <= 0:
        raise ValueError("PCS interval must be greater than 0.")
    pcs_values = np.arange(pcs_min, pcs_max + pcs_interval, pcs_interval)
    # Reject invalid display inputs before committing the pending tensor edit.
    _sync_legacy_pcs_tensor(state, source="legacy:update_graph")
    refresh_tensor_label = state.get("refresh_tensor_state_label")
    if callable(refresh_tensor_label):
        refresh_tensor_label()
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
        rotated_coords = None

    theta_values = state.get('theta_values')
    if theta_values is None:
        theta_values = np.linspace(0, 2*np.pi, 500)
    state['theta_values'] = theta_values

    plot_graph(state, pcs_values, theta_values, tensor, polar_data=polar_data, rotated_coords=rotated_coords)
    plot_cartesian_graph(state)
    update_molar_value(state, tensor)
    try:
        refresh_susceptibility = state.get("refresh_susceptibility_display")
        if callable(refresh_susceptibility):
            refresh_susceptibility(capture=True)
    except Exception:
        pass
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

    if state.get("_isolated_molecule_viewer") is not None:
        from ui.molecule_viewer_host import request_isolated_viewer_refresh
        request_isolated_viewer_refresh(state)

    # Interactive PCS scalar-field viewer. Its refresh adapter preserves the
    # current camera, so model changes do not reset the user's 3D view.
    try:
        if state.get("pyvista_field_plotter") is not None:
            from ui.plot_3d_pyvista import refresh_pcs_viewer

            refresh_pcs_viewer(
                state,
                **dict(state.get("pcs_scene_kwargs", {}) or {}),
            )
    except Exception as exc:
        state["_pcs_pyvista_refresh_error"] = exc

    # Rhombicity remains the authoritative side-by-side axial/full diagnostic.
    # Main pcs_by_id and NMR use the explicit active prediction-model policy.
    try:
        refresh_rhombicity = state.get("rh_refresh_table")
        if callable(refresh_rhombicity) and state.get("atom_data"):
            refresh_rhombicity()
    except Exception:
        pass

    # Presentation-only notification: PCS-data commands derive readiness from
    # the existing table, δ_Exp mapping, and undo snapshot.
    refresh_data_commands = state.get("refresh_pcs_data_commands")
    if callable(refresh_data_commands):
        refresh_data_commands()

    refresh_analysis_commands = state.get("refresh_gax_rhombicity_commands")
    if callable(refresh_analysis_commands):
        refresh_analysis_commands()

    refresh_fit_commands = state.get("refresh_fitting_commands")
    if callable(refresh_fit_commands):
        refresh_fit_commands()

    refresh_conformer = state.get("refresh_conformer_commands")
    if callable(refresh_conformer):
        refresh_conformer()

def reset_values(state, *, record_history=True):
    """Reset the main working model, retaining a scoped before/after record."""
    from logic.document_edit_history import begin_edit, finish_edit, install
    from logic.reset_edits import RESET_KEYS, RESET_ENTRY_KEYS, reset_document_values
    from logic.pcs_document import capture_active_document, active_document, suspend_document_tracking
    from logic.pcs_tensor_state import apply_canonical_tensor_to_legacy
    from ui.structure_history import (capture_structure_controls, project_structure_controls,
                                      prepare_structure_views)
    from logic.nmr_delta_data_manager import recompute_delta_para, push_layers_to_nmr_if_open

    try:
        values, controls_after = reset_document_values(state.get("app_settings", {}))
    except (TypeError, ValueError) as exc:
        state["messagebox"].showerror("Reset", str(exc))
        return False
    finish_gesture = state.get("finish_parameter_gesture")
    if callable(finish_gesture):
        finish_gesture()
    controls_before = capture_structure_controls(state)
    for key in RESET_ENTRY_KEYS:
        widget = state.get(key)
        if widget is not None:
            controls_before[key] = widget.get()
    controls_after.update({key: False for key in controls_before if key not in RESET_ENTRY_KEYS})
    token = begin_edit(state, RESET_KEYS) if record_history else None
    previous_tensor = state.get("pcs_tensor_state")
    install(state, RESET_KEYS, values)
    errors = []
    with suspend_document_tracking(state):
        project_structure_controls(state, controls_after)
        apply_canonical_tensor_to_legacy(state, values["pcs_tensor_state"])
        install_prediction_model(state, PredictionModel.AXIAL)
        try:
            prepare_structure_views(state)
        except Exception as exc:
            errors.append(f"Structure presentation: {exc}")
        recompute_delta_para(state)
        # Recorded result presenters read document-owned data, not widgets.
        if active_document(state) is not None:
            capture_active_document(state)
        redraw = state.get("update_graph") or (lambda: update_graph(state))
        try:
            if record_history and previous_tensor is not None:
                state["pcs_tensor_state"] = previous_tensor
                changed = update_canonical_tensor(state, values["pcs_tensor_state"],
                                                   source="reset:document", legacy_callback=redraw)
                if not changed:
                    apply_canonical_tensor_to_legacy(state, values["pcs_tensor_state"])
                    redraw()
            else:
                redraw()
        except Exception as exc:
            apply_canonical_tensor_to_legacy(state, values["pcs_tensor_state"])
            errors.append(f"update_graph: {exc}")
        for key, args in (("populate_fitting_controls", (state,)),
                          ("refresh_orientation_editor", ()), ("refresh_prediction_model_ui", ()),
                          ("rh_refresh_table", ()), ("refresh_fit_history_ui", ()),
                          ("refresh_last_fit_result_ui", ()), ("refresh_conformer_commands", ())):
            callback = state.get(key)
            if callable(callback):
                try:
                    callback(*args)
                except Exception as exc:
                    errors.append(f"{key}: {exc}")
    changed = finish_edit(state, token, "Reset document", controls_before=controls_before,
                          controls_after=controls_after, guard_keys=("fit_history",))
    if active_document(state) is not None:
        if record_history and changed:
            mark_document_mutated(state, "reset")
        else:
            capture_active_document(state)
    try:
        push_layers_to_nmr_if_open(state)
    except Exception as exc:
        errors.append(f"NMR compatibility: {exc}")
    if errors:
        state["messagebox"].showwarning("Reset", "Values reset with refresh warnings:\n" + "\n".join(errors))
    return True

@parameter_edit("Rotate structure")
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

@parameter_edit("Edit structure orientation")
def on_angle_entry_commit(state, axis):
    if unchanged_parameter_input(state, f"angle_{axis}_entry"):
        return
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
        return False
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
        state["symavg_members_by_pseudo_id"] = {}
        state["ref_label_overrides"] = {}
        state["torsion_avg_display_atom_data"] = []
        return

    # PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE3
    # Build representative ensemble-averaged coordinates for display.  PCS G
    # factors are NOT evaluated at these mean coordinates; they are averaged
    # over sampled conformers later in symmetry_geometry.effective_geometry_factors().
    raw_ids0 = list(state.get("atom_ids_raw") or range(1, len(raw) + 1))
    display_raw = list(raw)
    torsion_var = state.get("torsion_avg_enabled_var")
    torsion_enabled = bool(torsion_var.get()) if torsion_var is not None else False
    torsion_groups = state.get("torsion_avg_groups", []) or []
    active_torsions = [g for g in torsion_groups if bool(g.get("enabled", True))]
    if torsion_enabled and active_torsions:
        try:
            raw_xyz = np.asarray([[x, y, z] for _, x, y, z in raw], dtype=float)
            avg_xyz = average_coordinates_independent(raw_xyz, raw_ids0, active_torsions)
            display_raw = [
                (atom, float(avg_xyz[i, 0]), float(avg_xyz[i, 1]), float(avg_xyz[i, 2]))
                for i, (atom, *_xyz) in enumerate(raw)
            ]
        except Exception as exc:
            print(f"[Averaging] rotational display average failed: {exc}")
            display_raw = list(raw)
    state["torsion_avg_display_atom_data"] = list(display_raw)

    enabled_var = state.get("symavg_enabled_var")
    enabled = bool(enabled_var.get()) if enabled_var is not None else False

    # --- symmetry average OFF → raw as it is ---
    if not enabled:
        state["atom_data_eff"] = list(display_raw)
        state["atom_ids_eff"] = list(state.get("atom_ids_raw") or range(1, len(raw) + 1))
        state["symavg_pseudo_ref_ids"] = set()
        state["symavg_records"] = []
        state["symavg_members_by_pseudo_id"] = {}
        state["ref_label_overrides"] = {}
        return

    # --- keep original atoms  ---
    keep_var = state.get("symavg_keep_original_var")
    keep_original = bool(keep_var.get()) if keep_var is not None else False

    # collapse mode
    mode = "mask" if keep_original else "drop"

    # --- 시작 구조 ---
    atom_data = list(display_raw)
    ids = list(raw_ids0)

    records_all = []
    symavg_members = {}
    # Ref-ID -> display label override for table (pseudo atoms)
    # Example: { 153: "MeH@C12", 154: "CF3F@C7" }
    label_overrides = {}


    def _apply_collapse_with_ids(atom_data_in, ids_in, collapse_fn):
        """Apply collapse while preserving Ref-ID/member relationships.

        The returned member map links each pseudo Ref-ID to the Ref-IDs of the
        original symmetry-equivalent atoms.  This is later used to average
        G_ax/G_rh rather than evaluating G at the Cartesian centroid.
        """
        out_atoms, records, masked = collapse_fn(atom_data_in)

        next_id = (max(ids_in) if ids_in else 0) + 1
        pseudo_ids = list(range(next_id, next_id + len(records)))

        if mode == "mask":
            out_ids = list(ids_in) + pseudo_ids
        else:
            kept_ids = [rid for i, rid in enumerate(ids_in) if i not in masked]
            out_ids = kept_ids + pseudo_ids

        member_map = {}
        for pseudo_id, rec in zip(pseudo_ids, records):
            try:
                member_map[pseudo_id] = tuple(
                    ids_in[i] for i in rec.member_indices_original
                )
            except Exception:
                # Keep the pseudo atom usable for visualisation even if legacy
                # collapse metadata is incomplete.
                pass

        return out_atoms, out_ids, records, set(pseudo_ids), member_map

    # 1) Methyl collapse  # PCS_PATCH_AVERAGING_SETTINGS_PHASE1
    methyl_var = state.get("symavg_methyl_enabled_var")
    methyl_enabled = bool(methyl_var.get()) if methyl_var is not None else True
    rec_me = []
    pseudo_me = set()
    members_me = {}

    if methyl_enabled:
        def _do_me(atoms):
            return collapse_methyl_groups(
                atoms,
                mode=mode,
                pseudo_element="H",
                require_carbon_substituent_count=1,
            )

        atom_data, ids, rec_me, pseudo_me, members_me = _apply_collapse_with_ids(
            atom_data, ids, _do_me
        )
        records_all.extend(rec_me)
        symavg_members.update(members_me)

        # Map pseudo Ref IDs to human-readable labels for table display
        # NOTE: record.pseudo_index must refer to the index in the returned out_atoms list
        for rec in rec_me:
            try:
                rid = ids[rec.pseudo_index]
                label_overrides[rid] = rec.label
            except Exception:
                pass

    # 2) CF3 collapse
    cf3_var = state.get("symavg_cf3_enabled_var")
    cf3_enabled = bool(cf3_var.get()) if cf3_var is not None else True
    rec_cf = []
    pseudo_cf = set()
    members_cf = {}

    if cf3_enabled:
        def _do_cf(atoms):
            return collapse_cf3_groups(
                atoms,
                mode=mode,
                pseudo_element="F",
                require_carbon_substituent_count=1,
            )

        atom_data, ids, rec_cf, pseudo_cf, members_cf = _apply_collapse_with_ids(
            atom_data, ids, _do_cf
        )
        records_all.extend(rec_cf)
        symavg_members.update(members_cf)

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
    state["symavg_members_by_pseudo_id"] = symavg_members
    state["ref_label_overrides"] = label_overrides

# recent files helper
def add_recent_file(state, path: str):
    """
    Add a file path to the recent-files list.
    - newest first
    - no duplicates
    - limited by max_recent_files
    """
    if not path:
        return

    cfg = state.get("app_settings", {}) or {}
    if not cfg.get("remember_recent_files", True):
        return

    p = str(Path(path).resolve())

    recents = list(state.get("recent_files", []) or [])
    recents = [x for x in recents if str(x).strip() and str(x) != p]
    recents.insert(0, p)

    max_n = int(cfg.get("max_recent_files", 8) or 8)
    if max_n < 1:
        max_n = 1

    state["recent_files"] = recents[:max_n]

def rebuild_recent_files_menu(state):
    menu = state.get("recent_files_menu")
    if menu is None:
        return

    menu.delete(0, "end")

    cfg = state.get("app_settings", {}) or {}

    if not cfg.get("remember_recent_files", True):
        menu.add_command(label="(disabled in Settings)", state="disabled")
        return

    recents = list(state.get("recent_files", []) or [])
    recents = [p for p in recents if str(p).strip()]

    # remove missing files automatically
    recents = [p for p in recents if Path(p).exists()]

    max_n = int(cfg.get("max_recent_files", 8) or 8)
    if max_n < 1:
        max_n = 1

    recents = recents[:max_n]
    state["recent_files"] = recents

    try:
        save_app_state(state, save_geometry=False)
    except Exception:
        pass

    if not recents:
        menu.add_command(label="(empty)", state="disabled")
        return

    def _open_recent(path: str):
        if not Path(path).exists():
            state["messagebox"].showwarning("Recent files", f"File not found:\n{path}")
            state["recent_files"] = [p for p in state.get("recent_files", []) if p != path]
            try:
                save_app_state(state, save_geometry=False)
            except Exception:
                pass
            rebuild_recent_files_menu(state)
            return

        opener = state.get("open_file_by_type")
        if callable(opener):
            opener(path)
        else:
            _load_xyz_from_path(state, path)

    for path in recents:
        p = Path(path)
        name = p.name
        parent = p.parent.name if p.parent.name else str(p.parent)
        label = f"{name}  —  {parent}"

        menu.add_command(
            label=label,
            command=lambda p=path: _open_recent(p)
        )

    menu.add_separator()
    menu.add_command(label="Clear recent files", command=lambda: _clear_recent_files(state))

def _clear_recent_files(state):
    state["recent_files"] = []
    try:
        save_app_state(state, save_geometry=False)
    except Exception:
        pass
    rebuild_recent_files_menu(state)

def _load_xyz_from_path(state, path: str):
    """Accept a detached structure/centre before replacing the active document."""
    if not path:
        return False
    from logic.pcs_document import active_document_id, suspend_document_tracking
    from logic.document_edit_history import begin_edit, finish_edit, install
    from logic.structure_edits import REPLACEMENT_KEYS, replacement_values, validate_structure_atoms
    from ui.structure_history import (capture_structure_controls, project_structure_controls,
                                      prepare_structure_views)

    finish_gesture = state.get("finish_parameter_gesture")
    if callable(finish_gesture):
        finish_gesture()
    source_document = active_document_id(state)
    try:
        check = state.get("check_edit_history_pending_inputs")
        if callable(check):
            check()
        atom_data = validate_structure_atoms(load_structure(path))
    except (OSError, ValueError, TypeError) as exc:
        state["messagebox"].showerror("Load structure", str(exc))
        return False
    target_atom = state['simpledialog'].askstring(
        "Input center atom", "Enter the center atom (element) :", parent=state.get("root"))
    if not target_atom:
        return False
    if active_document_id(state) != source_document:
        state["messagebox"].showwarning("Load structure", "The active document changed; structure was not replaced.")
        return False
    try:
        values = replacement_values(atom_data, target_atom, path)
        # FocusOut during the modal prompt can confirm an input; recheck any
        # remaining edits before taking the actual before-snapshot.
        if callable(check):
            check()
    except ValueError as exc:
        state["messagebox"].showerror("Load structure", str(exc))
        return False
    values["torsion_avg_phase"] = state.get("torsion_avg_phase", 2)
    controls_before = capture_structure_controls(state)
    controls_after = {key: False for key in controls_before}
    token = begin_edit(state, REPLACEMENT_KEYS)
    install(state, REPLACEMENT_KEYS, values)
    # Finish recording even if a view fails: accepted scientific values must
    # remain recoverable. Redo restores this snapshot without file I/O/dialogs.
    errors = []
    try:
        with suspend_document_tracking(state):
            project_structure_controls(state, controls_after)
            prepare_structure_views(state)
    except Exception as exc:
        errors.append(str(exc))
    finally:
        finish_edit(state, token, f"Load structure · {Path(path).name}",
                    controls_before=controls_before, controls_after=controls_after)
    # Reuse the restore presenter for plots, readiness, NMR and bound views.
    # A zero-step history move is not appropriate; refresh the accepted state
    # directly through the existing owners instead.
    from logic.nmr_delta_data_manager import recompute_delta_para, push_layers_to_nmr_if_open
    recompute_delta_para(state)
    with suspend_document_tracking(state):
        for key, args in (("update_graph", ()), ("populate_fitting_controls", (state,)),
                          ("refresh_orientation_editor", ()), ("refresh_fit_result_validity_ui", ()),
                          ("refresh_conformer_commands", ())):
            callback = state.get(key)
            if callable(callback):
                try:
                    callback(*args)
                except Exception as exc:
                    errors.append(f"{key}: {exc}")
    try:
        push_layers_to_nmr_if_open(state)
    except Exception as exc:
        errors.append(f"NMR compatibility: {exc}")
    mark_document_mutated(state, "structure")
    from logic.pcs_document import refresh_follow_active_surfaces
    refresh_follow_active_surfaces(state, include_legacy=False)
    if errors:
        state["messagebox"].showwarning("Load structure", "Structure accepted with refresh warnings:\n" + "\n".join(errors))
    add_recent_file(state, path)
    rebuild_recent_files_menu(state)
    if state.get("app_settings", {}).get("auto_open_3d_on_load", True):
        # The delayed launch belongs to this document, not whichever tab becomes
        # active while the event loop is waiting.
        def open_loaded_viewer():
            if active_document_id(state) == source_document:
                open_3d_plot_window(state)
        state['root'].after(50, open_loaded_viewer)
    return True

def load_xyz_file(state):
    path = state['filedialog'].askopenfilename(filetypes=[
        ("Structure files", "*.xyz *.out *.log"),
        ("XYZ", "*.xyz"),
        ("ORCA output", "*.out *.log"),
        ("All files", "*.*"),
    ])
    if not path:
        return

    _load_xyz_from_path(state, path)

def _show_empty_element_visibility(state, text="No structure loaded."):
    checklist = state.get('checklist_frame')
    if checklist is None:
        return

    state["element_visibility_chips"] = []
    state["ligand_visibility_chips"] = []
    state["refresh_element_visibility_styles"] = lambda: None
    state["refresh_element_visibility_summary"] = lambda: None

    ttk.Label(
        checklist,
        text=text,
        style="ControlPanel.TLabel",
        anchor="w",
        justify="left",
    ).pack(anchor="w")


def create_checklist(state):
    """Build document-owned Element + Ligand visibility controls.

    ``check_vars`` remains as the legacy Element UI adapter, while the plain
    selections stored in ``_visibility_selected_*`` are the document-owned
    source captured by :mod:`logic.pcs_document`.
    """
    checklist = state['checklist_frame']

    previous_elements = {}
    for element, variable in (state.get("check_vars") or {}).items():
        try:
            previous_elements[str(element)] = bool(variable.get())
        except Exception:
            pass
    previous_ligands = {}
    for key, variable in (state.get("ligand_check_vars") or {}).items():
        try:
            previous_ligands[str(key)] = bool(variable.get())
        except Exception:
            pass

    explicit_elements_value = state.get(ELEMENT_SELECTION_KEY)
    explicit_elements = (
        {str(value) for value in explicit_elements_value}
        if isinstance(explicit_elements_value, (list, tuple, set))
        else None
    )
    explicit_ligands_value = state.get(LIGAND_SELECTION_KEY)
    explicit_ligands = (
        {str(value) for value in explicit_ligands_value}
        if isinstance(explicit_ligands_value, (list, tuple, set))
        else None
    )

    for widget in checklist.winfo_children():
        widget.destroy()

    atom_data = state.get('atom_data') or []
    raw_ids = state.get("atom_ids_raw") or list(range(1, len(atom_data) + 1))
    if not atom_data:
        state['check_vars'] = {}
        state['ligand_check_vars'] = {}
        _show_empty_element_visibility(state)
        try:
            populate_fitting_controls(state)
        except Exception:
            pass
        return

    element_counts = {}
    for atom, *_coords in atom_data:
        atom = str(atom)
        element_counts[atom] = element_counts.get(atom, 0) + 1
    atom_types = sorted(element_counts)
    if not atom_types:
        state['check_vars'] = {}
        state['ligand_check_vars'] = {}
        _show_empty_element_visibility(state, "No element types available.")
        return

    ligand_records = ligand_visibility_records(state)
    metrics = state.get("control_panel").metrics if state.get("control_panel") is not None else None
    palette = control_panel_palette(checklist)

    # Combined summary reflects the actual Element ∩ Ligand result by Ref ID.
    summary = ttk.Frame(checklist, style="ControlPanel.Form.TFrame")
    summary.pack(fill=tk.X, pady=(0, 5))
    total_atoms = len(atom_data)
    total_var = tk.StringVar(value=f"{total_atoms} atoms")
    element_count_var = tk.StringVar(value=f"{len(atom_types)} elements")
    visible_count_var = tk.StringVar(value=f"{total_atoms} visible")
    ttk.Label(summary, textvariable=total_var, style="ControlPanel.ElementSummaryStrong.TLabel").pack(side=tk.LEFT)
    ttk.Label(summary, text=" · ", style="ControlPanel.ElementSummary.TLabel").pack(side=tk.LEFT)
    ttk.Label(summary, textvariable=element_count_var, style="ControlPanel.ElementSummary.TLabel").pack(side=tk.LEFT)
    ttk.Label(summary, text=" · ", style="ControlPanel.ElementSummary.TLabel").pack(side=tk.LEFT)
    ttk.Label(summary, textvariable=visible_count_var, style="ControlPanel.ElementSummary.TLabel").pack(side=tk.LEFT)

    # ---- Elements ---------------------------------------------------------
    ttk.Label(
        checklist,
        text="Elements",
        style="ControlPanel.Subheading.TLabel",
    ).pack(anchor="w", pady=(1, 2))

    element_grid = ttk.Frame(checklist, style="ControlPanel.Form.TFrame")
    element_grid.pack(fill=tk.X)
    element_columns = min(4, max(1, len(atom_types)))
    for column in range(element_columns):
        element_grid.columnconfigure(column, weight=1, uniform="element_chips")

    check_vars = {}
    element_chips = []
    for index, element in enumerate(atom_types):
        if explicit_elements is not None:
            initially_visible = element in explicit_elements
        else:
            initially_visible = previous_elements.get(element, True)
        variable = tk.BooleanVar(value=initially_visible)
        check_vars[element] = variable
        chip = ControlPanelElementChip(
            element_grid,
            element=element,
            count=element_counts[element],
            variable=variable,
            cpk_color=CPK_COLORS.get(element, CPK_COLORS["default"]),
            metrics=metrics,
        )
        row = index // element_columns
        column = index % element_columns
        chip.grid(
            row=row,
            column=column,
            sticky="ew",
            padx=(0 if column == 0 else 2, 0 if column == element_columns - 1 else 2),
            pady=2,
        )
        element_chips.append(chip)

    element_utilities = ttk.Frame(checklist, style="ControlPanel.Form.TFrame")
    element_utilities.pack(fill=tk.X, pady=(3, 0))

    # ---- Ligands ----------------------------------------------------------
    ttk.Label(
        checklist,
        text="Ligands",
        style="ControlPanel.Subheading.TLabel",
    ).pack(anchor="w", pady=(7, 2))

    ligand_vars = {}
    ligand_chips = []
    if ligand_records:
        ligand_grid = ttk.Frame(checklist, style="ControlPanel.Form.TFrame")
        ligand_grid.pack(fill=tk.X)
        ligand_columns = min(4, max(1, len(ligand_records)))
        for column in range(ligand_columns):
            ligand_grid.columnconfigure(column, weight=1, uniform="ligand_chips")

        for index, record in enumerate(ligand_records):
            if explicit_ligands is not None:
                initially_visible = record.key in explicit_ligands
            else:
                initially_visible = previous_ligands.get(record.key, True)
            variable = tk.BooleanVar(value=initially_visible)
            ligand_vars[record.key] = variable
            chip = ControlPanelElementChip(
                ligand_grid,
                element=record.label,
                count=record.atom_count,
                variable=variable,
                cpk_color=palette["primary"],
                metrics=metrics,
            )
            row = index // ligand_columns
            column = index % ligand_columns
            chip.grid(
                row=row,
                column=column,
                sticky="ew",
                padx=(0 if column == 0 else 2, 0 if column == ligand_columns - 1 else 2),
                pady=2,
            )
            tooltip = f"Ligand {index + 1}"
            if record.formula:
                tooltip += f" · {record.formula}"
            tooltip += " · Ref " + ", ".join(str(value) for value in record.atom_ref_ids)
            ToolTip(chip, tooltip)
            ligand_chips.append(chip)

        ligand_utilities = ttk.Frame(checklist, style="ControlPanel.Form.TFrame")
        ligand_utilities.pack(fill=tk.X, pady=(3, 0))
    else:
        ligand_utilities = None
        message = "No coordinating ligands detected."
        if state.get("ligand_visibility_error") is not None:
            message = "Ligand detection unavailable."
        ttk.Label(
            checklist,
            text=message,
            style="ControlPanel.Muted.TLabel",
        ).pack(anchor="w", pady=(0, 1))

    state['check_vars'] = check_vars
    state['ligand_check_vars'] = ligand_vars
    state["element_visibility_chips"] = element_chips
    state["ligand_visibility_chips"] = ligand_chips

    # Initialize plain state from the newly built adapters without marking a
    # document dirty.  This also turns a default ``None`` selection into the
    # explicit all-visible set for a loaded structure.
    sync_visibility_state_from_vars(state)

    def _refresh_summary():
        visible = len(
            visible_ref_id_set(
                state,
                atom_data=atom_data,
                ref_ids=raw_ids,
            )
        )
        visible_count_var.set(f"{visible} visible")

    def _visibility_changed():
        sync_visibility_state_from_vars(state)
        _refresh_summary()
        mark_document_mutated(state, "visibility")
        try:
            update_graph(state)
        except Exception:
            pass

    for chip in element_chips:
        chip.command = _visibility_changed
    for chip in ligand_chips:
        chip.command = _visibility_changed

    def _set_all(variables, value: bool):
        for variable in variables.values():
            variable.set(bool(value))
        _visibility_changed()

    ttk.Button(
        element_utilities,
        text="None",
        style="ControlPanel.ElementUtility.TButton",
        command=lambda: _set_all(check_vars, False),
    ).pack(side=tk.RIGHT)
    ttk.Button(
        element_utilities,
        text="All",
        style="ControlPanel.ElementUtility.TButton",
        command=lambda: _set_all(check_vars, True),
    ).pack(side=tk.RIGHT, padx=(0, 2))

    if ligand_utilities is not None:
        ttk.Button(
            ligand_utilities,
            text="None",
            style="ControlPanel.ElementUtility.TButton",
            command=lambda: _set_all(ligand_vars, False),
        ).pack(side=tk.RIGHT)
        ttk.Button(
            ligand_utilities,
            text="All",
            style="ControlPanel.ElementUtility.TButton",
            command=lambda: _set_all(ligand_vars, True),
        ).pack(side=tk.RIGHT, padx=(0, 2))

    state["element_visibility_summary_vars"] = {
        "total": total_var,
        "elements": element_count_var,
        "visible": visible_count_var,
    }
    state["refresh_element_visibility_summary"] = _refresh_summary

    def _refresh_element_visibility_styles():
        refreshed_palette = control_panel_palette(checklist)
        for chip in tuple(state.get("element_visibility_chips") or ()) + tuple(state.get("ligand_visibility_chips") or ()):
            try:
                if chip.winfo_exists():
                    if chip in state.get("ligand_visibility_chips", ()):
                        chip.cpk_color = refreshed_palette["primary"]
                    chip.apply_theme(refreshed_palette)
            except Exception:
                pass

    state["refresh_element_visibility_styles"] = _refresh_element_visibility_styles

    _refresh_summary()
    populate_fitting_controls(state)

def filter_atoms(state):
    # Use effective coords for 2D/table, keep raw for 3D separately. Visibility
    # is resolved by authoritative Ref ID; filtering never renumbers atoms.
    atom_data = state.get("atom_data_eff") or state.get("atom_data") or []
    ids = state.get("atom_ids_eff") or state.get("atom_ids") or list(range(1, len(atom_data) + 1))
    visible_ids = visible_ref_id_set(state, atom_data=atom_data, ref_ids=ids)

    abs_coords = np.array([[x, y, z] for a, x, y, z in atom_data])
    metal = np.array([state['x0'], state['y0'], state['z0']])

    # Raw coordinates are transformed in parallel.  They are not displayed in
    # the 2D/table effective structure, but are required for physically correct
    # G-factor averaging of CH3/CF3 pseudo atoms.
    raw_data = state.get("atom_data_raw") or state.get("atom_data") or []
    raw_ids = state.get("atom_ids_raw") or list(range(1, len(raw_data) + 1))
    raw_visible_ids = visible_ref_id_set(state, atom_data=raw_data, ref_ids=raw_ids)
    raw_abs_coords = np.array([[x, y, z] for _, x, y, z in raw_data], dtype=float)

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
                if len(raw_abs_coords):
                    raw_abs_coords = _angles_to_rotation_multi(
                        points=raw_abs_coords,
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
            if len(raw_abs_coords):
                raw0 = raw_abs_coords - metal
                raw_abs_coords = rotate_euler(raw0, ax, ay, az) + metal

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
            if len(raw_abs_coords):
                raw0 = raw_abs_coords - metal
                raw_abs_coords = rotate_euler(raw0, ax_e, ay_e, az_e) + metal

            if 'dchi_ax' in fo:
                try:
                    state['tensor_entry'].delete(0, tk.END)
                    state['tensor_entry'].insert(0, str(float(fo['dchi_ax'])))
                except Exception:
                    pass

    coords0 = abs_coords - metal

    ax = float(state['angle_x_var'].get())
    ay = float(state['angle_y_var'].get())
    az = float(state['angle_z_var'].get())
    rotated = rotate_coordinates(coords0, ax, ay, az, (0, 0, 0))

    if len(raw_abs_coords):
        raw0 = raw_abs_coords - metal
        raw_rotated = rotate_coordinates(raw0, ax, ay, az, (0, 0, 0))
        state["last_rotated_raw_by_id"] = {
            rid: np.asarray(coord, dtype=float)
            for rid, coord in zip(raw_ids, raw_rotated)
        }
    else:
        state["last_rotated_raw_by_id"] = {}

    # Reference coordinates for optional torsional overlay in structural plots.
    state["torsion_reference_polar_data"] = []
    state["torsion_reference_rotated_coords"] = []
    state["torsion_reference_ref_ids"] = []
    keep_ref_var = state.get("torsion_avg_keep_reference_var")
    keep_reference = bool(keep_ref_var.get()) if keep_ref_var is not None else False
    torsion_var = state.get("torsion_avg_enabled_var")
    torsion_enabled = bool(torsion_var.get()) if torsion_var is not None else False
    if torsion_enabled and keep_reference and len(raw_abs_coords):
        moving_refs = set()
        for g in (state.get("torsion_avg_groups", []) or []):
            if bool(g.get("enabled", True)):
                moving_refs.update(int(r) for r in g.get("rotating_atoms", ()))
        ref_polar, ref_xyz, ref_ids_out = [], [], []
        raw_labels = [a for a, *_ in raw_data]
        for rid, atom, coord in zip(raw_ids, raw_labels, raw_rotated):
            if rid not in moving_refs or int(rid) not in raw_visible_ids:
                continue
            rr = float(np.linalg.norm(coord))
            th = float(np.arccos(np.clip(coord[2] / rr, -1.0, 1.0))) if rr else 0.0
            ref_polar.append((atom, rr, th))
            ref_xyz.append(tuple(float(x) for x in coord))
            ref_ids_out.append(int(rid))
        state["torsion_reference_polar_data"] = ref_polar
        state["torsion_reference_rotated_coords"] = ref_xyz
        state["torsion_reference_ref_ids"] = ref_ids_out

    polar = []
    rotated_sel = []
    selected_ids = []

    label_overrides = state.get("ref_label_overrides", {}) or {}

    for idx, ((atom, *_), (dx, dy, dz)) in enumerate(zip(atom_data, rotated)):
        ref_id = ids[idx]
        if int(ref_id) in visible_ids:
            r = (dx*dx + dy*dy + dz*dz) ** 0.5
            theta = np.arccos(dz / r) if r != 0 else 0.0

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
    #-- plot selection interaction ---
    state["cartesian_start_lasso"] = start_lasso_selection
    state["cartesian_start_rectangle"] = start_rectangle_selection
    state["cartesian_fit_selected"] = fit_selected_points
    state["cartesian_clear_selection"] = clear_cartesian_selection

    # Establish the application-level PCS model boundary after all legacy UI
    # inputs exist. Initial synchronization is deliberately event-free.
    initialize_pcs_model_state(state)
    subscribe_pcs_refresh_consumers(state)

    return state
