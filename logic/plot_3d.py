# logic/plot_3d.py

import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from logic.chem_constants import CPK_COLORS, covalent_radii
from logic.rotate_align import rotate_coordinates

class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0,0),(0,0), *args, **kwargs)
        self._xyz = (x,y,z); self._dxdydz = (dx,dy,dz)
    def draw(self, renderer):
        x1,y1,z1 = self._xyz; dx,dy,dz = self._dxdydz
        x2,y2,z2 = (x1+dx, y1+dy, z1+dz)
        xs,ys,zs = proj3d.proj_transform((x1,x2), (y1,y2), (z1,z2), self.axes.M)
        self.set_positions((xs[0],ys[0]), (xs[1],ys[1]))
        super().draw(renderer)
    def do_3d_projection(self): return min(self._xyz[2], self._xyz[2])

def add_arrow3D(ax, x,y,z, dx,dy,dz, *args, **kwargs):
    arrow = Arrow3D(x,y,z, dx,dy,dz, *args, **kwargs)
    ax.add_artist(arrow)

def get_cpk_color(atom): return CPK_COLORS.get(atom, CPK_COLORS['default'])

def set_axes_equal(ax):
    xlim=ax.get_xlim3d(); ylim=ax.get_ylim3d(); zlim=ax.get_zlim3d()
    xr=abs(xlim[1]-xlim[0]); xm=np.mean(xlim); yr=abs(ylim[1]-ylim[0]); ym=np.mean(ylim); zr=abs(zlim[1]-zlim[0]); zm=np.mean(zlim)
    R=0.5*max([xr,yr,zr]); ax.set_xlim3d([xm-R,xm+R]); ax.set_ylim3d([ym-R,ym+R]); ax.set_zlim3d([zm-R,zm+R])

def calculate_bonds(atom_coords, atom_elements):
    from scipy.spatial.distance import pdist, squareform
    bonds=[]; D = squareform(pdist(atom_coords))
    for i in range(len(atom_coords)):
        for j in range(i+1, len(atom_coords)):
            rsum = covalent_radii.get(atom_elements[i],0)+covalent_radii.get(atom_elements[j],0)
            if D[i,j] <= rsum*1.1: bonds.append((i,j))
    return bonds

def plot_3d_molecule_with_bonds(rotated_coords, atoms, atom_elements, master_window, FigureCanvas, NavigationToolbar2Tk):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4,4)); ax = fig.add_subplot(111, projection='3d')
    cx,cy,cz = rotated_coords[0]; shifted = [(x-cx,y-cy,z-cz) for x,y,z in rotated_coords]
    for atom, (x,y,z), el in zip(atoms, shifted, atom_elements):
        radius = covalent_radii.get(el, 1.0); size = radius * 50
        ax.scatter(x,y,z, color=get_cpk_color(atom), s=size, alpha=0.7)
    for i,j in calculate_bonds(np.array(shifted), atom_elements):
        ax.plot([shifted[i][0], shifted[j][0]], [shifted[i][1], shifted[j][1]], [shifted[i][2], shifted[j][2]], color='gray', lw=2.0, alpha=1.0)
    L=8
    add_arrow3D(ax,0,0,0, L,0,0, mutation_scale=20, ec='black', fc='blue')
    add_arrow3D(ax,0,0,0, 0,L,0, mutation_scale=20, ec='black', fc='green')
    add_arrow3D(ax,0,0,0, 0,0,L, mutation_scale=20, ec='black', fc='red')
    ax.text(L,0,0,'X', color='blue', fontsize=12, weight='bold')
    ax.text(0,L,0,'Y', color='green', fontsize=12, weight='bold')
    ax.text(0,0,L,'Z', color='red', fontsize=12, weight='bold')
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off(); set_axes_equal(ax); fig.tight_layout()
    canvas = FigureCanvas(fig, master=master_window)
    toolbar = NavigationToolbar2Tk(canvas, master_window)
    toolbar.update()
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

def open_3d_plot_window(state):
    """state에서 현재 슬라이더 각도/중심원자/좌표를 읽어 3D 창을 띄웁니다."""
    if not state.get('atom_data'):
        state['messagebox'].showerror("Error", "No atom data loaded")
        return

    import tkinter as tk
    coords = np.array([[x, y, z] for _, x, y, z in state['atom_data']])
    atoms = [a for a, *_ in state['atom_data']]
    atom_elements = atoms  # 요소 심볼 리스트 (가독성 위해 이름 분리)

    # 슬라이더 각도 읽기
    ax_deg = float(state['angle_x_var'].get())
    ay_deg = float(state['angle_y_var'].get())
    # Z는 현재 슬라이더가 없으니 0.0
    rot = rotate_coordinates(
        coords,
        ax_deg, ay_deg, 0.0,
        (state['x0'], state['y0'], state['z0'])
    )

    # 창 생성 및 그리기
    win = tk.Toplevel(state['root'])
    win.title("3D Molecular Structure")
    win.geometry("600x600")

    plot_3d_molecule_with_bonds(
        rot, atoms, atom_elements,
        win,
        state['FigureCanvas'],
        state['NavigationToolbar2Tk']
    )