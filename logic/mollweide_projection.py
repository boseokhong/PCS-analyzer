# logic/mollweide_projection.py

import numpy as np
from logic.rotate_align import rotate_coordinates
from logic.chem_constants import CPK_COLORS

def cartesian_to_spherical_with_colors(coords, elements):
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    r = np.sqrt(x*x + y*y + z*z)
    phi = np.arcsin(z / r)      # polar
    theta = np.arctan2(y, x)    # azimuth
    colors = [CPK_COLORS.get(el, CPK_COLORS['default']) for el in elements]
    return theta, phi, colors

def plot_theta_phi_scatter(root, theta, phi, elements, colors, FigureCanvasTkAgg):
    import matplotlib.pyplot as plt
    import tkinter as tk
    win = tk.Toplevel(root)
    win.title("Mollweide projection plot")
    win.geometry("800x400")

    fig, ax = plt.subplots(subplot_kw={'projection':'mollweide'}, figsize=(8,6))
    ax.grid(True)
    for t, p, c in zip(theta, phi, colors):
        ax.scatter(t, p, color=c, s=25, alpha=0.8)
    ax.set_xlabel("Azimuthal angle (φ)", fontsize=8)
    ax.set_ylabel("Polar angle (θ)", fontsize=8)
    ax.axhline(y=0, color='black', linewidth=1)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

def open_theta_phi_plot(atom_data, center_xyz, ax_deg, ay_deg, az_deg, root, FigureCanvasTkAgg):
    if not atom_data:
        return
    import numpy as np
    coords = np.array([[x,y,z] for _,x,y,z in atom_data])
    cx, cy, cz = center_xyz
    # 슬라이더 각도로 중심 기준 회전 (3D와 동일)
    rot = rotate_coordinates(coords, ax_deg, ay_deg, az_deg, (cx, cy, cz))
    # 투영은 중심 기준 좌표로
    shifted = rot - np.array([cx,cy,cz])
    elements = [a for a, *_ in atom_data]
    theta, phi, colors = cartesian_to_spherical_with_colors(shifted, elements)
    plot_theta_phi_scatter(root, theta, phi, elements, colors, FigureCanvasTkAgg)
