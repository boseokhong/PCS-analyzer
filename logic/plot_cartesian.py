# logic/plot_cartesian.py

from scipy import stats

class DraggableAnnotation:
    def __init__(self, annotation):
        import numpy as np
        self.annotation = annotation
        self.press = None
        self.annotation.set_picker(True)
        self.annotation.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.annotation.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.annotation.figure.canvas.mpl_connect('button_release_event', self.on_release)
    def on_pick(self, event):
        if event.artist == self.annotation:
            self.press = (self.annotation.xy, event.mouseevent.xdata, event.mouseevent.ydata)
    def on_motion(self, event):
        if self.press is None: return
        xy, xpress, ypress = self.press
        dx = event.xdata - xpress; dy = event.ydata - ypress
        self.annotation.xy = (xy[0] + dx, xy[1] + dy)
        self.annotation.figure.canvas.draw()
    def on_release(self, event):
        self.press = None; self.annotation.figure.canvas.draw()

def plot_cartesian_graph(state):
    tree = state['tree']
    geom_values, delta_values, scaled_x = [], [], []
    import numpy as np
    for item in tree.get_children():
        gv = float(tree.item(item, "values")[5])  # G_i
        dv_raw = tree.item(item, "values")[7]
        if dv_raw != "":
            dv = float(dv_raw)
            geom_values.append(gv); delta_values.append(dv)
            scaled_x.append((gv*1e4)/(12*np.pi))
    fig = state['cartesian_figure']; canvas = state['cartesian_canvas']
    fig.clear(); ax = fig.add_subplot(111)
    if geom_values and delta_values:
        ax.scatter(geom_values, delta_values, color='blue', marker='o', label='δ vs G_i')
        slope, intercept, r_value, p_value, std_err = stats.linregress(geom_values, delta_values)
        reg = [slope*x+intercept for x in geom_values]; ax.plot(geom_values, reg, color='blue', label='Linear Fit (Blue)')
        txt = f"Blue Scatter\nSlope: {slope:.2e}\nIntercept: {intercept:.2e}\nR-value: {r_value:.3f}"
        ann = ax.text(0.05,0.95, txt, transform=ax.transAxes, fontsize=8, va='top',
                      bbox=dict(boxstyle='round,pad=0.3', edgecolor='blue', facecolor=(1,1,1,0.7)))
        DraggableAnnotation(ann)
        ax.scatter(scaled_x, delta_values, color='red', marker='o', label='δ vs Scaled G_i')
        slope_r, inter_r, r_r, p_r, se_r = stats.linregress(scaled_x, delta_values)
        reg_r = [slope_r*x+inter_r for x in scaled_x]; ax.plot(scaled_x, reg_r, color='red', label='Linear Fit (Red)')
        txt_r = f"Red Scatter\nSlope: {slope_r:.2e}\nIntercept: {inter_r:.2e}\nR-value: {r_r:.3f}"
        ann_r = ax.text(0.4,0.95, txt_r, transform=ax.transAxes, fontsize=8, fontweight='bold', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', edgecolor='red', facecolor=(1,1,1,0.7)))
        DraggableAnnotation(ann_r)
        ax.legend(fontsize=8)
    ax.set_xlabel('G_i'); ax.set_ylabel('δ (ppm)')
    ax.set_title('Geometrical factor (G_i) vs Chemical shift (δ_Exp)'); ax.grid(True)
    canvas.draw()
