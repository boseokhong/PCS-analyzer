# logic/plot_cartesian.py

import os
import csv

from scipy import stats
import numpy as np

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

def _format_result_text(rows, selected_ref=None, force_origin=False):
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
        _set_result_box_text(
            state,
            _format_result_text(rows, selected_ref=None, force_origin=force_origin)
        )
        _disconnect_plot_callbacks(state)
        canvas.draw()
        return

    selected_ref = _get_selected_ref_id(state)

    x = np.asarray([r["gi"] for r in rows], dtype=float)
    y = np.asarray([r["dexp"] for r in rows], dtype=float)

    artists = []
    for row in rows:
        is_selected = (selected_ref == row["ref_id"])
        art = ax.scatter(
            [row["gi"]],
            [row["dexp"]],
            s=(76 if is_selected else 38),
            marker="o",
            color=_point_color_from_dpcs(row["dpcs"]),
            alpha=0.90,
            zorder=(4 if is_selected else 3),
            edgecolors=("gold" if is_selected else "#444444"),
            linewidths=(1.8 if is_selected else 0.55),
        )
        artists.append(art)

    if len(rows) >= 2:
        fit = _linear_fit(x, y, force_origin=force_origin, ci_level=0.95)
        slope = fit["slope"]
        intercept = fit["intercept"]

        if np.isfinite(slope):
            xx = np.linspace(np.min(x), np.max(x), 200)
            yy = slope * xx + intercept
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

    _install_pick_handler(state, ax, rows, artists)
    _install_hover_handler(state, ax, rows, artists)
    _set_result_box_text(
        state,
        _format_result_text(rows, selected_ref=selected_ref, force_origin=force_origin)
    )
    fig.tight_layout()
    canvas.draw()