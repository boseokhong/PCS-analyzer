# logic/plot_cartesian.py

import os
import csv

from scipy import stats
import numpy as np
from matplotlib.widgets import LassoSelector, RectangleSelector
from matplotlib.path import Path

def _safe_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def _get_selected_ref_id(state):
    tree = state.get("tree")
    if tree is None:
        return None
    sel = tree.selection()
    if not sel:
        return None
    try:
        return int(tree.item(sel[0], "values")[0])
    except Exception:
        return None

def _set_result_box_text(state, text):
    box = state.get("plot_result_box")
    if box is None:
        return
    try:
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)
        box.configure(state="disabled")
    except Exception:
        pass

def _build_plot_rows(state):
    tree = state.get("tree")
    if tree is None:
        return []

    rows = []
    for item in tree.get_children():
        vals = tree.item(item, "values")
        if not vals or len(vals) < 8:
            continue

        ref_id = _safe_float(vals[0])
        gi = _safe_float(vals[5])
        dpcs = _safe_float(vals[6])
        dexp = _safe_float(vals[7])
        atom = str(vals[1]) if len(vals) > 1 else ""

        if ref_id is None or gi is None or dexp is None:
            continue

        rows.append({
            "ref_id": int(ref_id),
            "atom": atom,
            "gi": float(gi),
            "dexp": float(dexp),
            "dpcs": float(dpcs) if dpcs is not None else None,
            "residual": (float(dexp) - float(dpcs)) if (dpcs is not None) else None,
        })
    return rows

def _get_active_rows_for_fit(state, rows):
    """
    Return rows used for regression depending on the current selection mode.
    """
    mode = state.get("cartesian_selection_mode", "all")
    selected = state.get("cartesian_selected_ids", set()) or set()

    if mode == "selected" and selected:
        return [r for r in rows if r["ref_id"] in selected]

    return list(rows)

def _update_selection_status(state, rows=None):
    """
    Update the compact selection status label in the Cartesian plot panel.
    """
    var = state.get("cartesian_selection_status_var")
    if var is None:
        return

    if rows is None:
        rows = state.get("cartesian_plot_rows", []) or []

    selected = state.get("cartesian_selected_ids", set()) or set()
    active = bool(state.get("cartesian_selector_active", False))

    if active:
        kind = state.get("cartesian_selector_kind") or "selection"
        if kind == "rectangle":
            kind = "box"
        var.set(f"{kind} active")
    elif selected:
        var.set(f"Selected: {len(selected)} / {len(rows)}")
    else:
        var.set("All points")

def _format_result_text(rows, selected_ref=None, force_origin=False, fit_scope="all"):
    if not rows:
        return "No assigned δ_Exp values.\n\nImport or enter δ_Exp values to populate the plot."

    x = np.asarray([r["gi"] for r in rows], dtype=float)
    y = np.asarray([r["dexp"] for r in rows], dtype=float)

    fit = _linear_fit(x, y, force_origin=force_origin, ci_level=0.95)

    slope = fit["slope"]
    intercept = fit["intercept"]
    r2 = fit["r2"]
    slope_se = fit["slope_se"]
    intercept_se = fit["intercept_se"]
    resid_sd = fit["resid_sd"]
    rmse = fit["rmse"]
    slope_ci = fit["slope_ci"]
    intercept_ci = fit["intercept_ci"]

    scale = (12.0 * np.pi) / 1e4
    dchi_ax = slope * scale if np.isfinite(slope) else np.nan
    dchi_ax_ci = (
        slope_ci[0] * scale if np.isfinite(slope_ci[0]) else np.nan,
        slope_ci[1] * scale if np.isfinite(slope_ci[1]) else np.nan,
    )

    lines = []
    lines.append("G_i vs δ_Exp analysis")
    lines.append("=" * 30)
    lines.append(f"n points        : {len(rows)}")
    lines.append(f"fit mode        : {'through origin' if force_origin else 'with intercept'}")
    lines.append(f"fit scope       : {fit_scope}")

    if np.isfinite(slope):
        lines.append("")
        lines.append("Regression")
        lines.append("-" * 30)
        lines.append(f"slope(raw)      : {slope:.6e}")
        lines.append(f"slope SE        : {slope_se:.6e}" if np.isfinite(slope_se) else "slope SE        : N/A")
        lines.append(
            f"slope 95% CI    : [{slope_ci[0]:.6e}, {slope_ci[1]:.6e}]"
            if np.isfinite(slope_ci[0]) and np.isfinite(slope_ci[1]) else
            "slope 95% CI    : N/A"
        )

        lines.append(f"intercept       : {intercept:.6g}")
        lines.append(
            f"intercept SE    : {intercept_se:.6e}"
            if np.isfinite(intercept_se) else
            "intercept SE    : N/A"
        )
        lines.append(
            f"intercept 95% CI: [{intercept_ci[0]:.6g}, {intercept_ci[1]:.6g}]"
            if np.isfinite(intercept_ci[0]) and np.isfinite(intercept_ci[1]) else
            "intercept 95% CI: N/A"
        )

        lines.append(f"R²              : {r2:.4f}" if np.isfinite(r2) else "R²              : N/A")
        lines.append(
            f"residual SD     : {resid_sd:.6g}"
            if np.isfinite(resid_sd) else
            "residual SD     : N/A"
        )
        lines.append(
            f"RMSE            : {rmse:.6g}"
            if np.isfinite(rmse) else
            "RMSE            : N/A"
        )

        lines.append("")
        lines.append("Converted tensor")
        lines.append("-" * 30)
        lines.append(f"Δχ_ax           : {dchi_ax:.6g} E-32 m³")
        lines.append(
            f"Δχ_ax 95% CI    : [{dchi_ax_ci[0]:.6g}, {dchi_ax_ci[1]:.6g}] E-32 m³"
            if np.isfinite(dchi_ax_ci[0]) and np.isfinite(dchi_ax_ci[1]) else
            "Δχ_ax 95% CI    : N/A"
        )
    else:
        lines.append("Need at least 2 points for regression.")

    if selected_ref is not None:
        sel = next((r for r in rows if r["ref_id"] == selected_ref), None)
        if sel is not None:
            lines.append("")
            lines.append("Selected point")
            lines.append("-" * 30)
            lines.append(f"Ref             : {sel['ref_id']}")
            lines.append(f"Atom            : {sel['atom']}")
            lines.append(f"G_i             : {sel['gi']:.2e}")
            lines.append(f"δ_Exp           : {sel['dexp']:.6g}")
            if sel["dpcs"] is not None:
                lines.append(f"δ_PCS           : {sel['dpcs']:.6g}")
            if sel["residual"] is not None:
                lines.append(f"Residual        : {sel['residual']:.6g}")

    return "\n".join(lines)

def _install_pick_handler(state, ax, rows, artists):
    canvas = state.get("cartesian_canvas")
    if canvas is None:
        return

    old_cid = state.get("cartesian_click_cid")
    if old_cid is not None:
        try:
            canvas.mpl_disconnect(old_cid)
        except Exception:
            pass

    row_by_id = state.get("row_by_id", {})
    tree = state.get("tree")

    def _on_click(event):
        # During lasso/box selection, do not treat mouse press as point picking.
        # Otherwise Treeview selection can trigger plot redraw while the selector is active.
        if state.get("cartesian_selector_active", False):
            return

        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        best = None
        best_dist = None

        for row, art in zip(rows, artists):
            offsets = art.get_offsets()
            if len(offsets) == 0:
                continue
            px, py = offsets[0]
            dist = (px - event.xdata) ** 2 + (py - event.ydata) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = row

        if best is None:
            return

        ref_id = best["ref_id"]
        if tree is not None and ref_id in row_by_id:
            item = row_by_id[ref_id]
            tree.selection_set(item)
            tree.focus(item)
            tree.see(item)

    state["cartesian_click_cid"] = canvas.mpl_connect("button_press_event", _on_click)

def _linear_fit(x, y, force_origin=False, ci_level=0.95):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(x)
    if n < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "yhat": np.full_like(y, np.nan),
            "slope_se": np.nan,
            "intercept_se": np.nan,
            "resid_sd": np.nan,
            "rmse": np.nan,
            "slope_ci": (np.nan, np.nan),
            "intercept_ci": (np.nan, np.nan),
        }

    alpha = 1.0 - float(ci_level)

    if force_origin:
        xx = float(np.dot(x, x))
        if xx == 0.0:
            return {
                "slope": np.nan,
                "intercept": 0.0,
                "r2": np.nan,
                "yhat": np.full_like(y, np.nan),
                "slope_se": np.nan,
                "intercept_se": np.nan,
                "resid_sd": np.nan,
                "rmse": np.nan,
                "slope_ci": (np.nan, np.nan),
                "intercept_ci": (0.0, 0.0),
            }

        slope = float(np.dot(x, y) / xx)
        intercept = 0.0
        yhat = slope * x
        resid = y - yhat

        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = (1.0 - ss_res / ss_tot) if ss_tot != 0.0 else np.nan
        rmse = float(np.sqrt(np.mean(resid ** 2)))

        # through-origin 회귀에서는 자유도 n-1
        dof = n - 1
        if dof > 0:
            resid_sd = float(np.sqrt(ss_res / dof))
            slope_se = float(np.sqrt((ss_res / dof) / xx))
            tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, dof))
            slope_ci = (slope - tcrit * slope_se, slope + tcrit * slope_se)
        else:
            resid_sd = np.nan
            slope_se = np.nan
            slope_ci = (np.nan, np.nan)

        return {
            "slope": slope,
            "intercept": intercept,
            "r2": float(r2),
            "yhat": yhat,
            "slope_se": slope_se,
            "intercept_se": np.nan,
            "resid_sd": resid_sd,
            "rmse": rmse,
            "slope_ci": slope_ci,
            "intercept_ci": (0.0, 0.0),
        }

    # with intercept
    res = stats.linregress(x, y)
    slope = float(res.slope)
    intercept = float(res.intercept)
    yhat = slope * x + intercept
    resid = y - yhat

    ss_res = float(np.sum(resid ** 2))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    r2 = float(res.rvalue ** 2)

    dof = n - 2
    resid_sd = float(np.sqrt(ss_res / dof)) if dof > 0 else np.nan

    slope_se = float(getattr(res, "stderr", np.nan))
    intercept_se = float(getattr(res, "intercept_stderr", np.nan))

    if dof > 0 and np.isfinite(slope_se):
        tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, dof))
        slope_ci = (slope - tcrit * slope_se, slope + tcrit * slope_se)
    else:
        slope_ci = (np.nan, np.nan)

    if dof > 0 and np.isfinite(intercept_se):
        tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, dof))
        intercept_ci = (
            intercept - tcrit * intercept_se,
            intercept + tcrit * intercept_se,
        )
    else:
        intercept_ci = (np.nan, np.nan)

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "yhat": yhat,
        "slope_se": slope_se,
        "intercept_se": intercept_se,
        "resid_sd": resid_sd,
        "rmse": rmse,
        "slope_ci": slope_ci,
        "intercept_ci": intercept_ci,
    }

def _point_color_from_dpcs(dpcs):
    if dpcs is None:
        return "#A0A0A0"
    if dpcs > 0:
        return "#C96A6A"
    if dpcs < 0:
        return "#5B7DB1"
    return "#A0A0A0"

def _row_tooltip_text(row):
    lines = [
        f"Ref: {row['ref_id']}",
        f"Atom: {row['atom']}",
        f"G_i: {row['gi']:.2e}",
        f"δ_Exp: {row['dexp']:.6g}",
    ]
    if row["dpcs"] is not None:
        lines.append(f"δ_PCS: {row['dpcs']:.6g}")
    if row["residual"] is not None:
        lines.append(f"Residual: {row['residual']:.6g}")
    return "\n".join(lines)

def _install_hover_handler(state, ax, rows, artists):
    canvas = state.get("cartesian_canvas")
    if canvas is None:
        return

    old_cid = state.get("cartesian_hover_cid")
    if old_cid is not None:
        try:
            canvas.mpl_disconnect(old_cid)
        except Exception:
            pass

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="0.3"),
        zorder=20,
    )
    annot.set_visible(False)

    x_vals = np.asarray([r["gi"] for r in rows], dtype=float)
    y_vals = np.asarray([r["dexp"] for r in rows], dtype=float)

    x_span = max(float(np.max(x_vals) - np.min(x_vals)), 1e-12)
    y_span = max(float(np.max(y_vals) - np.min(y_vals)), 1e-12)

    # 데이터 범위에 따라 hover 허용 반경 설정
    hover_thr2 = (0.03 ** 2 + 0.03 ** 2)  # normalized distance^2

    def _on_move(event):
        # Disable hover tooltip while interactive selection is active.
        if state.get("cartesian_selector_active", False):
            if annot.get_visible():
                annot.set_visible(False)
                canvas.draw_idle()
            return

        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            if annot.get_visible():
                annot.set_visible(False)
                canvas.draw_idle()
            return

        best = None
        best_d2 = None

        for row, art in zip(rows, artists):
            offsets = art.get_offsets()
            if len(offsets) == 0:
                continue

            px, py = offsets[0]
            dx = (px - event.xdata) / x_span
            dy = (py - event.ydata) / y_span
            d2 = dx * dx + dy * dy

            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = (row, px, py)

        if best is None or best_d2 is None or best_d2 > hover_thr2:
            if annot.get_visible():
                annot.set_visible(False)
                canvas.draw_idle()
            return

        row, px, py = best
        annot.xy = (px, py)
        annot.set_text(_row_tooltip_text(row))
        annot.set_visible(True)
        canvas.draw_idle()

    state["cartesian_hover_cid"] = canvas.mpl_connect("motion_notify_event", _on_move)

def _disconnect_plot_callbacks(state):
    canvas = state.get("cartesian_canvas")
    if canvas is None:
        return

    for key in ("cartesian_click_cid", "cartesian_hover_cid"):
        cid = state.get(key)
        if cid is not None:
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass
            state[key] = None

# export
def export_cartesian_plot(state):
    fig = state.get("cartesian_figure")
    if fig is None:
        state["messagebox"].showerror("Export Plot", "No plot figure available.")
        return

    fd = state["filedialog"].asksaveasfilename(
        title="Export Plot Figure",
        defaultextension=".png",
        filetypes=[
            ("PNG image", "*.png"),
            ("SVG vector", "*.svg"),
            ("PDF file", "*.pdf"),
            ("All files", "*.*"),
        ],
    )
    if not fd:
        return

    base, ext = os.path.splitext(fd)
    ext = ext.lower() or ".png"
    fig_path = fd
    summary_path = base + "_summary.txt"
    points_path = base + "_points.csv"

    try:
        fig.savefig(fig_path, dpi=600, bbox_inches="tight", facecolor="white")
        _write_plot_summary_txt(state, summary_path)
        _write_plot_points_csv(state, points_path)

        state["messagebox"].showinfo(
            "Export Plot",
            "Saved:\n"
            f"{fig_path}\n"
            f"{summary_path}\n"
            f"{points_path}"
        )
    except Exception as e:
        state["messagebox"].showerror("Export Plot", f"Export failed:\n{e}")

def _current_force_origin(state):
    var = state.get("plot_force_origin_var")
    return bool(var.get()) if var is not None else False

def _write_plot_summary_txt(state, path):
    rows = state.get("cartesian_plot_rows", []) or _build_plot_rows(state)
    selected_ref = _get_selected_ref_id(state)
    force_origin = _current_force_origin(state)

    text = _format_result_text(
        rows,
        selected_ref=selected_ref,
        force_origin=force_origin,
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")

def _write_plot_points_csv(state, path):
    rows = state.get("cartesian_plot_rows", []) or _build_plot_rows(state)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Ref", "Atom", "G_i", "δ_Exp", "δ_PCS", "Residual"])
        for r in rows:
            writer.writerow([
                r.get("ref_id"),
                r.get("atom"),
                r.get("gi"),
                r.get("dexp"),
                r.get("dpcs"),
                r.get("residual"),
            ])

# plot graph
def plot_cartesian_graph(state):
    fig = state["cartesian_figure"]
    canvas = state["cartesian_canvas"]

    rows = _build_plot_rows(state)
    state["cartesian_plot_rows"] = rows
    _update_selection_status(state, rows)

    force_origin = bool(state.get("plot_force_origin_var").get()) if state.get(
        "plot_force_origin_var") is not None else False

    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)

    if not rows:
        ax.set_xlabel("Gᵢ")
        ax.set_ylabel("δ (ppm)")
        ax.set_title("Geometrical factor (Gᵢ) vs Chemical shift (δ_Exp)")
        ax.grid(True)
        ax.axhline(0, color="#444444", linewidth=1.0, alpha=0.9, zorder=1)
        ax.axvline(0, color="#444444", linewidth=1.0, alpha=0.9, zorder=1)
        _set_result_box_text(
            state,
            _format_result_text(
                rows,
                selected_ref=None,
                force_origin=force_origin,
                fit_scope="all points",
            )
        )
        _disconnect_plot_callbacks(state)
        canvas.draw()
        return

    selected_ref = _get_selected_ref_id(state)

    fit_rows = _get_active_rows_for_fit(state, rows)
    fit_scope = "selected points" if (
            state.get("cartesian_selection_mode") == "selected"
            and state.get("cartesian_selected_ids")
    ) else "all points"

    x = np.asarray([r["gi"] for r in fit_rows], dtype=float)
    y = np.asarray([r["dexp"] for r in fit_rows], dtype=float)

    selected_ids = state.get("cartesian_selected_ids", set()) or set()
    has_multi_selection = bool(selected_ids)

    artists = []

    for row in rows:
        ref_id = row["ref_id"]

        # Current single selection from the main Treeview
        is_focused = (selected_ref == ref_id)

        # Multi-point selection from Lasso/Rectangle
        is_in_subset = ref_id in selected_ids

        if has_multi_selection:
            if is_in_subset:
                # Included in the current regression subset
                alpha = 0.90
                size = 46
                edge = "#333333"
                lw = 0.75
                zorder = 5
            else:
                # Visible but excluded from the current regression subset
                alpha = 0.18
                size = 26
                edge = "#BBBBBB"
                lw = 0.25
                zorder = 2

            # Focused point should remain recognizable even in subset mode
            if is_focused:
                size = max(size, 64)
                edge = "gold"
                lw = 1.4
                zorder = 7

        else:
            # Normal mode: only the currently focused Treeview point is highlighted
            alpha = 0.90
            size = 76 if is_focused else 38
            edge = "gold" if is_focused else "#444444"
            lw = 1.8 if is_focused else 0.55
            zorder = 4 if is_focused else 3

        art = ax.scatter(
            [row["gi"]],
            [row["dexp"]],
            s=size,
            marker="o",
            color=_point_color_from_dpcs(row["dpcs"]),
            alpha=alpha,
            zorder=zorder,
            edgecolors=edge,
            linewidths=lw,
        )
        artists.append(art)

    if len(fit_rows) >= 2:
        fit = _linear_fit(x, y, force_origin=force_origin, ci_level=0.95)
        slope = fit["slope"]
        intercept = fit["intercept"]

        if np.isfinite(slope):
            xx = np.linspace(np.min(x), np.max(x), 200)
            yy = slope * xx + intercept
            label = "Linear fit selected (b=0)" if force_origin else "Linear fit selected"
            if fit_scope == "all points":
                label = "Linear fit (b=0)" if force_origin else "Linear fit"

            ax.plot(xx, yy, color="#222222", linewidth=1.6, label=label)

    if selected_ref is not None:
        sel = next((r for r in rows if r["ref_id"] == selected_ref), None)
        if sel is not None:
            ann_text = f"{sel['atom']} (Ref {sel['ref_id']})\nδ = {sel['dexp']:.4g}"

            ax.annotate(
                ann_text,
                xy=(sel["gi"], sel["dexp"]),
                xytext=(14, 14),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="gold", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="goldenrod"),
                zorder=15,
            )

    ax.set_xlabel("Gᵢ")
    ax.set_ylabel("δ (ppm)")
    ax.set_title("Geometrical factor (Gᵢ) vs Chemical shift (δ_Exp)")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    # origin axes
    ax.axhline(0, color="#444444", linewidth=1.0, alpha=0.9, zorder=1)
    ax.axvline(0, color="#444444", linewidth=1.0, alpha=0.9, zorder=1)


    if len(rows) >= 2:
        ax.legend(fontsize=8)
        ax.text(
            0.98, 0.02,
            "red: δ_PCS > 0\nblue: δ_PCS < 0",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=7,
            color="#555555",
        )

    if not state.get("cartesian_selector_active", False):
        _install_pick_handler(state, ax, rows, artists)
        _install_hover_handler(state, ax, rows, artists)

    _set_result_box_text(
        state,
        _format_result_text(
            fit_rows,
            selected_ref=selected_ref,
            force_origin=force_origin,
            fit_scope=fit_scope,
        )
    )
    fig.tight_layout()
    canvas.draw()

# Lasso/Rectangle functions
def _clear_existing_selector(state):
    selector = state.get("cartesian_selector")
    if selector is not None:
        try:
            selector.disconnect_events()
        except Exception:
            pass

    state["cartesian_selector"] = None
    state["cartesian_selector_kind"] = None
    state["cartesian_selector_active"] = False
    _update_selection_status(state)

def _select_rows_by_indices(state, rows, indices):
    selected_ids = {rows[i]["ref_id"] for i in indices if 0 <= i < len(rows)}
    state["cartesian_selected_ids"] = selected_ids
    state["cartesian_selection_mode"] = "selected" if selected_ids else "all"

    # Reflect selection in the main Treeview
    tree = state.get("tree")
    row_by_id = state.get("row_by_id", {})
    if tree is not None and row_by_id:
        items = [row_by_id[rid] for rid in selected_ids if rid in row_by_id]
        if items:
            tree.selection_set(items)
            tree.focus(items[0])
            tree.see(items[0])

    plot_cartesian_graph(state)


def start_lasso_selection(state):
    fig = state.get("cartesian_figure")
    canvas = state.get("cartesian_canvas")
    rows = state.get("cartesian_plot_rows", []) or _build_plot_rows(state)

    if fig is None or canvas is None or not rows:
        return

    # Remove previous selector and normal plot callbacks.
    # This prevents point-click / hover events from redrawing the plot
    # while the lasso selector is active.
    _clear_existing_selector(state)
    _disconnect_plot_callbacks(state)

    ax = fig.axes[0] if fig.axes else None
    if ax is None:
        state["cartesian_selector_active"] = False
        return

    state["cartesian_selector_kind"] = "lasso"
    state["cartesian_selector_active"] = True
    _update_selection_status(state, rows)

    points = np.asarray([[r["gi"], r["dexp"]] for r in rows], dtype=float)

    def _on_select(verts):
        indices = []
        try:
            if verts is None or len(verts) < 3:
                return

            path = Path(verts)
            mask = path.contains_points(points)
            indices = np.nonzero(mask)[0].tolist()

        finally:
            _clear_existing_selector(state)

        _select_rows_by_indices(state, rows, indices)

    selector = LassoSelector(ax, onselect=_on_select)
    state["cartesian_selector"] = selector


def start_rectangle_selection(state):
    fig = state.get("cartesian_figure")
    canvas = state.get("cartesian_canvas")
    rows = state.get("cartesian_plot_rows", []) or _build_plot_rows(state)

    if fig is None or canvas is None or not rows:
        return

    # Remove previous selector and normal plot callbacks.
    # This prevents point-click / hover events from redrawing the plot
    # while the rectangle selector is active.
    _clear_existing_selector(state)
    _disconnect_plot_callbacks(state)

    ax = fig.axes[0] if fig.axes else None
    if ax is None:
        state["cartesian_selector_active"] = False
        return

    state["cartesian_selector_kind"] = "rectangle"
    state["cartesian_selector_active"] = True
    _update_selection_status(state, rows)

    points = np.asarray([[r["gi"], r["dexp"]] for r in rows], dtype=float)

    def _on_select(eclick, erelease):
        indices = []
        try:
            if eclick.xdata is None or eclick.ydata is None:
                return
            if erelease.xdata is None or erelease.ydata is None:
                return

            x0, x1 = sorted([eclick.xdata, erelease.xdata])
            y0, y1 = sorted([eclick.ydata, erelease.ydata])

            mask = (
                (points[:, 0] >= x0) &
                (points[:, 0] <= x1) &
                (points[:, 1] >= y0) &
                (points[:, 1] <= y1)
            )

            indices = np.nonzero(mask)[0].tolist()

        finally:
            _clear_existing_selector(state)

        _select_rows_by_indices(state, rows, indices)

    selector = RectangleSelector(
        ax,
        _on_select,
        useblit=False,
        button=[1],
        interactive=False,
        props=dict(
            facecolor="gray",
            edgecolor="black",
            alpha=0.15,
            fill=True,
        ),
    )

    state["cartesian_selector"] = selector


def fit_selected_points(state):
    selected = state.get("cartesian_selected_ids", set()) or set()
    if selected:
        state["cartesian_selection_mode"] = "selected"
    else:
        state["cartesian_selection_mode"] = "all"

    plot_cartesian_graph(state)


def clear_cartesian_selection(state):
    _clear_existing_selector(state)
    state["cartesian_selector_active"] = False
    state["cartesian_selected_ids"] = set()
    state["cartesian_selection_mode"] = "all"
    _update_selection_status(state)

    tree = state.get("tree")
    if tree is not None:
        try:
            tree.selection_remove(tree.selection())
        except Exception:
            pass

    plot_cartesian_graph(state)