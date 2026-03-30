# logic/plot_cartesian.py

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

def _format_result_text(rows, selected_ref=None):
    if not rows:
        return "No assigned δ_Exp values.\n\nImport or enter δ_Exp values to populate the plot."

    x = np.asarray([r["gi"] for r in rows], dtype=float)
    y = np.asarray([r["dexp"] for r in rows], dtype=float)

    if len(rows) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r2 = r_value ** 2
        dchi_ax = slope * (12.0 * np.pi) / 1e4
    else:
        slope = intercept = r_value = r2 = dchi_ax = np.nan

    lines = []
    lines.append("G_i vs δ_Exp analysis")
    lines.append("=" * 24)
    lines.append(f"n points   : {len(rows)}")
    if np.isfinite(slope):
        lines.append(f"slope(raw) : {slope:.6e}")
        lines.append(f"intercept  : {intercept:.6g}")
        lines.append(f"R          : {r_value:.4f}")
        lines.append(f"R²         : {r2:.4f}")
        lines.append(f"Δχ_ax      : {dchi_ax:.6g} E-32 m³")
    else:
        lines.append("Need at least 2 points for regression.")

    if selected_ref is not None:
        sel = next((r for r in rows if r["ref_id"] == selected_ref), None)
        if sel is not None:
            lines.append("")
            lines.append("Selected point")
            lines.append("-" * 24)
            lines.append(f"Ref        : {sel['ref_id']}")
            lines.append(f"Atom       : {sel['atom']}")
            lines.append(f"G_i        : {sel['gi']:.2e}")
            lines.append(f"δ_Exp      : {sel['dexp']:.6g}")
            if sel["dpcs"] is not None:
                lines.append(f"δ_PCS      : {sel['dpcs']:.6g}")
            if sel["residual"] is not None:
                lines.append(f"Residual   : {sel['residual']:.6g}")

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

def _point_color_from_dpcs(dpcs):
    if dpcs is None:
        return "#9E9E9E"
    if dpcs > 0:
        return "#D55E5E"
    if dpcs < 0:
        return "#4C72B0"
    return "#9E9E9E"

def plot_cartesian_graph(state):
    fig = state["cartesian_figure"]
    canvas = state["cartesian_canvas"]

    rows = _build_plot_rows(state)
    state["cartesian_plot_rows"] = rows

    fig.clear()
    ax = fig.add_subplot(111)

    if not rows:
        ax.set_xlabel("G_i")
        ax.set_ylabel("δ (ppm)")
        ax.set_title("Geometrical factor (G_i) vs Chemical shift (δ_Exp)")
        ax.grid(True)
        _set_result_box_text(state, _format_result_text(rows, selected_ref=None))
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
            s=(80 if is_selected else 32),
            marker="o",
            color=_point_color_from_dpcs(row["dpcs"]),
            zorder=(4 if is_selected else 3),
            edgecolors=("gold" if is_selected else "black"),
            linewidths=(1.8 if is_selected else 0.5),
        )
        artists.append(art)

    if len(rows) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        xx = np.linspace(np.min(x), np.max(x), 200)
        yy = slope * xx + intercept
        ax.plot(xx, yy, linewidth=1.2, label="Linear fit")

    ax.set_xlabel("G_i")
    ax.set_ylabel("δ (ppm)")
    ax.set_title("Geometrical factor (G_i) vs Chemical shift (δ_Exp)")
    ax.grid(True)

    if len(rows) >= 2:
        ax.legend(fontsize=8)

    _install_pick_handler(state, ax, rows, artists)
    _set_result_box_text(state, _format_result_text(rows, selected_ref=selected_ref))
    canvas.draw()