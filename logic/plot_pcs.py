# logic/plot_pcs.py

import numpy as np
from logic.chem_constants import CPK_COLORS

def get_cpk_color(atom): 
    return CPK_COLORS.get(atom, CPK_COLORS['default'])

def calculate_r(pcs_value, theta, tensor):
    """
    r = cbrt( denom / (12*pi*pcs) ),  denom = 1e4 * tensor * (3*cos^2(theta) - 1)
    - pcs == 0 또는 denom == 0 → NaN
    - 물리적으로 의미없는 음수 비율 → NaN
    - np.cbrt 사용으로 음수의 3제곱근도 안전 (하지만 r은 거리이므로 양수 비율만 사용)
    """
    import numpy as np
    denom = 1e4 * tensor * (3 * (np.cos(theta))**2 - 1)

    # 분자/분모 0 회피
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = denom / (12 * np.pi * pcs_value)

    # 물리적으로 의미 있는 양수만 남기고 나머진 NaN
    frac = np.where(frac > 0, frac, np.nan)

    # 3제곱근 (양수에 대해 안전)
    r = np.cbrt(frac)

    # 비정상 값 정리
    r[~np.isfinite(r)] = np.nan
    return r

def plot_graph(state, pcs_values, theta_values, tensor, polar_data=None):
    fig = state['pcs_figure']; canvas = state['pcs_canvas']; plot_90 = state['plot_90_var'].get()
    fig.clear()
    ax = fig.add_subplot(1,1,1, projection='polar')

    pcs_min = float(np.min(pcs_values)); pcs_max = float(np.max(pcs_values))
    if plot_90:
        theta_range = theta_values[theta_values <= np.pi/2]
        ax.set_position([0.2,0.1,0.4,0.8]); ax.set_thetamax(90)
        ax.set_xticks([0, np.pi/12, np.pi/6, np.pi/4, np.pi/3, 5*np.pi/12, np.pi/2])
        ax.set_xticklabels(["0°","15°","30°","45°","60°","75°","90°"], fontsize=8)
        bbox = (1.3, 0.5)
    else:
        theta_range = theta_values
        ax.set_position([0.0,0.1,0.8,0.8]); ax.set_thetamax(180)
        ax.set_xticks([0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi])
        ax.set_xticklabels(["0°","30°","60°","90°","120°","150°","180°"], fontsize=8)
        bbox = (0.9, 0.5)

    for pcs_value in pcs_values:
        rvals = calculate_r(pcs_value, theta_range, tensor)
        if np.isclose(pcs_value, 0): color = 'white'
        elif pcs_value > 0:
            intensity = max(0, 1 - pcs_value/pcs_max); color = (1, intensity, intensity)
        else:
            intensity = max(0, 1 + pcs_value/abs(pcs_min)); color = (intensity, intensity, 1)
        if plot_90:
            mirrored_theta = np.pi - theta_range
            ax.plot(np.concatenate([theta_range, mirrored_theta]), np.concatenate([rvals, rvals]), color=color, label=f"{pcs_value: .1f}")
        else:
            ax.plot(theta_range, rvals, color=color, label=f"{pcs_value: .1f}")

    click_pairs = []  # [(PathCollection, ref_id), ...]
    ids = state.get('current_selected_ids', [])

    if polar_data:
        for i, (atom, r, theta) in enumerate(polar_data):
            if plot_90 and theta > np.pi / 2:
                theta = np.pi - theta
            pt = ax.scatter(theta, r, color=get_cpk_color(atom), zorder=5, s=15)
            ref_id = state['current_selected_ids'][i]  # Ref(ID)
            click_pairs.append((pt, ref_id))

    # radius ticks
    r_ticks = [0,2,4,6,8,10]
    ax.set_yticks(r_ticks); ax.set_yticklabels([f"{r} Å" for r in r_ticks]); ax.tick_params(axis='y', labelsize=8)
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1); ax.set_ylim(0,10)
    if plot_90: ax.set_thetamax(90)
    else: ax.set_thetamax(180)

    leg = ax.legend(fontsize=6, bbox_to_anchor=bbox, loc="center left", ncol=2, frameon=False,
                    handletextpad=0.8, labelspacing=0.5, columnspacing=1.0, borderpad=0, borderaxespad=0.5, handlelength=1.0, handleheight=1.0)
    leg.set_title("PCS legend (ppm)", prop={'size':6, 'weight':'bold'})

    if state.get('pcs_click_cid') is not None:
        try:
            fig.canvas.mpl_disconnect(state['pcs_click_cid'])
        except Exception:
            pass
        state['pcs_click_cid'] = None

    #점을 클릭하면 해당 Ref(ID)로 테이블 선택
    def on_click(event):
        if event.inaxes != ax:
            return
        row_by_id = state.get('row_by_id', {})  # update_table에서 채움
        tree = state['tree']
        for artist, ref_id in click_pairs:
            contains, _ = artist.contains(event)
            if contains:
                item = row_by_id.get(ref_id)
                if item:
                    tree.selection_set(item)
                    tree.see(item)
                break

    state['pcs_click_cid'] = fig.canvas.mpl_connect('button_press_event', on_click)
    canvas.draw()

def update_figsize(state):
    canvas_widget = state['pcs_canvas'].get_tk_widget()
    parent = canvas_widget.master
    canvas_widget.destroy()
    fig = state['pcs_figure']
    if state['plot_90_var'].get(): fig.set_size_inches(4,2, forward=True)
    else: fig.set_size_inches(4,4, forward=True)
    state['pcs_canvas'] = state['FigureCanvas'](fig, master=parent)
    state['pcs_canvas'].get_tk_widget().pack(fill=state['tk'].BOTH, expand=True)
    state['pcs_click_cid'] = None