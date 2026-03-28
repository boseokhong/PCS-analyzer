# logic/plot_pcs.py

import numpy as np
from logic.chem_constants import CPK_COLORS


def get_cpk_color(atom):
    return CPK_COLORS.get(atom, CPK_COLORS["default"])


def calculate_r(pcs_value, theta, tensor):
    """
    r = cbrt( denom / (12*pi*pcs) ),  denom = 1e4 * tensor * (3*cos^2(theta) - 1)
    - pcs == 0 또는 denom == 0 → NaN
    - 물리적으로 의미없는 음수 비율 → NaN
    - np.cbrt 사용으로 음수의 3제곱근도 안전 (하지만 r은 거리이므로 양수 비율만 사용)
    """
    denom = 1e4 * tensor * (3 * (np.cos(theta)) ** 2 - 1)

    # 분자/분모 0 회피
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = denom / (12 * np.pi * pcs_value)

    # 물리적으로 의미 있는 양수만 남기고 나머진 NaN
    frac = np.where(frac > 0, frac, np.nan)

    # 3제곱근 (양수에 대해 안전)
    r = np.cbrt(frac)

    # 비정상 값 정리
    r[~np.isfinite(r)] = np.nan
    return r


def _draw_single_pcs_plot(fig, canvas, state, pcs_values, theta_values, tensor, polar_data=None, store_click_key=None):
    """
    Draw one PCS polar plot onto the given figure/canvas pair.

    store_click_key:
        state key used to store the mpl click callback id
        e.g. "pcs_click_cid" or "pcs_popup_click_cid"
    """
    if fig is None or canvas is None:
        return

    plot_90 = state["plot_90_var"].get()

    fig.clear()
    ax = fig.add_subplot(1, 1, 1, projection="polar")

    pcs_min = float(np.min(pcs_values))
    pcs_max = float(np.max(pcs_values))

    if plot_90:
        theta_range = theta_values[theta_values <= np.pi / 2]
        ax.set_position([0.2, 0.1, 0.4, 0.8])
        ax.set_thetamax(90)
        ax.set_xticks([0, np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 3, 5 * np.pi / 12, np.pi / 2])
        ax.set_xticklabels(["0°", "15°", "30°", "45°", "60°", "75°", "90°"], fontsize=8)
        bbox = (1.3, 0.5)
    else:
        theta_range = theta_values
        ax.set_position([0.0, 0.1, 0.8, 0.8])
        ax.set_thetamax(180)
        ax.set_xticks([0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6, np.pi])
        ax.set_xticklabels(["0°", "30°", "60°", "90°", "120°", "150°", "180°"], fontsize=8)
        bbox = (0.9, 0.5)

    # PCS contour lines
    for pcs_value in pcs_values:
        rvals = calculate_r(pcs_value, theta_range, tensor)

        if np.isclose(pcs_value, 0):
            color = "white"
        elif pcs_value > 0:
            intensity = max(0, 1 - pcs_value / pcs_max)
            color = (1, intensity, intensity)
        else:
            intensity = max(0, 1 + pcs_value / abs(pcs_min))
            color = (intensity, intensity, 1)

        if plot_90:
            mirrored_theta = np.pi - theta_range
            ax.plot(
                np.concatenate([theta_range, mirrored_theta]),
                np.concatenate([rvals, rvals]),
                color=color,
                label=f"{pcs_value: .1f}",
            )
        else:
            ax.plot(theta_range, rvals, color=color, label=f"{pcs_value: .1f}")

    click_pairs = []

    ids = state.get("current_selected_ids", []) or []

    pseudo_ref_ids = state.get("symavg_pseudo_ref_ids", None)
    if pseudo_ref_ids is None:
        pseudo_ref_ids = set()
    else:
        pseudo_ref_ids = set(pseudo_ref_ids)

    if polar_data:
        n = min(len(polar_data), len(ids))

        for i in range(n):
            atom, r, theta = polar_data[i]
            ref_id = ids[i]

            if plot_90 and theta > np.pi / 2:
                theta = np.pi - theta

            is_pseudo = ref_id in pseudo_ref_ids

            marker = "x" if is_pseudo else "o"
            size = 30 if is_pseudo else 15
            lw = 1.3 if is_pseudo else 0.8

            pt = ax.scatter(
                theta,
                r,
                color=get_cpk_color(atom),
                zorder=6 if is_pseudo else 5,
                s=size,
                marker=marker,
                linewidths=lw,
            )
            click_pairs.append((pt, ref_id))

    r_ticks = [0, 2, 4, 6, 8, 10]
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{r} Å" for r in r_ticks])
    ax.tick_params(axis="y", labelsize=8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 10)
    ax.set_thetamax(90 if plot_90 else 180)

    leg = ax.legend(
        fontsize=6,
        bbox_to_anchor=bbox,
        loc="center left",
        ncol=2,
        frameon=False,
        handletextpad=0.8,
        labelspacing=0.5,
        columnspacing=1.0,
        borderpad=0,
        borderaxespad=0.5,
        handlelength=1.0,
        handleheight=1.0,
    )
    leg.set_title("PCS legend (ppm)", prop={"size": 6, "weight": "bold"})

    # Disconnect previous click handler for this specific figure
    if store_click_key:
        old_cid = state.get(store_click_key)
        if old_cid is not None:
            try:
                fig.canvas.mpl_disconnect(old_cid)
            except Exception:
                pass
            state[store_click_key] = None

    def on_click(event):
        if event.inaxes != ax:
            return
        row_by_id = state.get("row_by_id", {})
        tree = state["tree"]
        for artist, ref_id in click_pairs:
            contains, _ = artist.contains(event)
            if contains:
                item = row_by_id.get(ref_id)
                if item:
                    tree.selection_set(item)
                    tree.see(item)
                break

    if store_click_key:
        state[store_click_key] = fig.canvas.mpl_connect("button_press_event", on_click)

    canvas.draw()


def plot_graph(state, pcs_values, theta_values, tensor, polar_data=None):
    """
    Draw the embedded PCS plot and, if open, the popup PCS plot as well.
    """
    # Main embedded PCS plot
    _draw_single_pcs_plot(
        fig=state.get("pcs_figure"),
        canvas=state.get("pcs_canvas"),
        state=state,
        pcs_values=pcs_values,
        theta_values=theta_values,
        tensor=tensor,
        polar_data=polar_data,
        store_click_key="pcs_click_cid",
    )

    # Popup PCS plot (same rendering)
    _draw_single_pcs_plot(
        fig=state.get("pcs_figure_popup"),
        canvas=state.get("pcs_canvas_popup"),
        state=state,
        pcs_values=pcs_values,
        theta_values=theta_values,
        tensor=tensor,
        polar_data=polar_data,
        store_click_key="pcs_popup_click_cid",
    )


def update_figsize(state):
    canvas = state.get("pcs_canvas")
    fig = state.get("pcs_figure")

    if canvas is None or fig is None:
        return

    canvas_widget = canvas.get_tk_widget()
    parent = canvas_widget.master
    canvas_widget.destroy()

    if state["plot_90_var"].get():
        fig.set_size_inches(4, 2, forward=True)
    else:
        fig.set_size_inches(4, 4, forward=True)

    state["pcs_canvas"] = state["FigureCanvas"](fig, master=parent)
    state["pcs_canvas"].get_tk_widget().pack(fill=state["tk"].BOTH, expand=True)
    state["pcs_click_cid"] = None