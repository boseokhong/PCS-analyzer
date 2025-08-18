'''
PCS plot analyzer

The following code requires the numpy, matplotlib, pandas, and openpyxl module packages to run.
It is designed for PCS analysis and 2d polar contour plot visualization (using only axiality of magnetic susceptibility tensor).
When exporting to an Excel file, the file extension ".xlsx" should be included. e.g.[filename.xlsx]


v.0.1 updates
-Excel export: x, y cartesian coordinates are calculated, and each column are separated by PCS values.

v.0.2 updates
-tensor values, PCS range, and intervals can now be input for calculations.
-molar susceptibility tensor calculation

v.0.3 updates
-XYZ file import and plot, coordinate rotation, atom coordinate table added, geometrical parameter analysis plot

v.0.3.1 updates
-clicking on a atom point on the PCS plot to highlight the corresponding table entry, enhancing data visualization and interaction.

v.0.4 updates
-Added 3d molecule scatter plot
-The code from the following project was referred to for implementing the 3D scatter plot.
 This file includes portions of code derived from work by Sebastian Dechert, 
 licensed under the BSD 3-Clause License. See LICENSE or the project's README.md for full license details. The original code can be found at:
 <https://github.com/radi0sus/xyz2tab>

v.0.5 updates
-Reorganized GUI
-Added Mollweide projection plot

v.0.6 updates
-PCS plot: Half/Quarter view toggle function


2024.12. Boseok Hong [Department of Chemistry of the f-elements, Institute of Resource Ecology, HZDR]
<bshong66@gmail.com> / <b.hong@hzdr.de>
<https://github.com/boseokhong/PCS-analyzer>
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, simpledialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from mpl_toolkits.mplot3d import proj3d
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Constants and initial settings
AVOGADRO_CONSTANT = 6.0221408e23
delta_values = {}


# CPK Colors dictionary
CPK_COLORS = {
    'H': '#656565', 'C': '#111111', 'N': '#0000FF', 'O': '#FF0000',
    'F': '#90E050', 'Cl': '#1FF01F', 'Br': '#A62929', 'I': '#940094',
    'He': '#D9FFFF', 'Ne': '#B3E3F5', 'Ar': '#80D1E3', 'Xe': '#67D3E0',
    'Kr': '#5CB8D1', 'P': '#FF8000', 'S': '#FFFF30', 'B': '#FFB5B5',
    'Li': '#CC80FF', 'Be': '#C2FF00', 'Na': '#AB5CF2', 'Mg': '#8AFF00',
    'Al': '#BFA6A6', 'Si': '#F0C8A0', 'K': '#8F40D4', 'Ca': '#3DFF00',
    'Fe': '#E06633', 'Pt': '#E06633', 'Pd': '#E06633', 'default': '#70ABFA'
}

# https://github.com/radi0sus/xyz2tab
# Covalent radii from Alvarez (2008)
covalent_radii = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28,
    'Be': 0.96, 'B': 0.84, 'C': 0.76, 
    'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 
    'V': 1.53, 'Cr': 1.39, 'Mn': 1.61, 'Fe': 1.52, 
    'Co': 1.50, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 
    'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 
    'Br': 1.20, 'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95,
    'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
    'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39,
    'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39,
    'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40,
    'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04,
    'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98,
    'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92,
    'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90, 'Yb': 1.87,
    'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.70, 'W': 1.62,
    'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36,
    'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46,
    'Bi': 1.48, 'Po': 1.40, 'At': 1.50, 'Rn': 1.50, 
    'Fr': 2.60, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06,
    'Pa': 2.00, 'U': 1.96, 'Np': 1.90, 'Pu': 1.87,
    'Am': 1.80, 'Cm': 1.69
}

# Helper class to add 3D arrows
class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        # Get the coordinates of the arrow base and tip
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        # Project 3D coordinates into 2D
        xs, ys, zs = proj3d.proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)

        # Set the 2D positions of the arrow base and tip
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        # Call the original draw method to draw the arrow
        super().draw(renderer)

    def do_3d_projection(self):
        return np.min(self._xyz[2])

# Helper function to add 3D arrows to an Axes3D plot
def add_arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

# Add arrow3D as a method to Axes3D using setattr
setattr(Axes3D, 'arrow3D', add_arrow3D)

# DraggableAnnotation class
class DraggableAnnotation:
    def __init__(self, annotation):
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
        if self.press is None:
            return
        
        xy, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        
        self.annotation.xy = (xy[0] + dx, xy[1] + dy)
        self.annotation.figure.canvas.draw()
    
    def on_release(self, event):
        self.press = None
        self.annotation.figure.canvas.draw()


def get_cpk_color(atom):
    return CPK_COLORS.get(atom, CPK_COLORS['default'])

def calculate_r(pcs_value, theta, tensor):
    denominator = 1e4 * tensor * (3 * (np.cos(theta)) ** 2 - 1)
    valid_denominator = np.where(denominator != 0, denominator, np.nan)
    r_values = (1 / (12 * np.pi * (pcs_value / valid_denominator))) ** (1 / 3)
    return r_values

def rotate_coordinates(coords, angle_x, angle_y, angle_z):
    angle_x = np.radians(angle_x)
    angle_y = np.radians(angle_y)
    angle_z = np.radians(angle_z)
    
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, np.cos(angle_x), -np.sin(angle_x)],
                                  [0, np.sin(angle_x), np.cos(angle_x)]])
    
    rotation_matrix_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                                  [0, 1, 0],
                                  [-np.sin(angle_y), 0, np.cos(angle_y)]])
    
    rotation_matrix_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                                  [np.sin(angle_z), np.cos(angle_z), 0],
                                  [0, 0, 1]])
    
    rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x
    rotated_coords = np.dot(coords, rotation_matrix.T)
    return rotated_coords

def update_angle_x_slider(value):
    angle_x_entry.delete(0, tk.END)
    angle_x_entry.insert(0, f"{float(value):.1f}")
    update_graph()

def update_angle_y_slider(value):
    angle_y_entry.delete(0, tk.END)
    angle_y_entry.insert(0, f"{float(value):.1f}")
    update_graph()

def on_angle_x_entry_change(*args):
    try:
        value = float(angle_x_entry.get())
        angle_x_slider.set(value)
    except ValueError:
        pass

def on_angle_y_entry_change(*args):
    try:
        value = float(angle_y_entry.get())
        angle_y_slider.set(value)
    except ValueError:
        pass

def toggle_plot_range():
    """
    Function to toggle between 0-180 degrees and 0-90 degrees plot range.
    """
    update_graph()

def plot_graph(pcs_values, theta_values, tensor, canvas, figure, polar_data=None):
    """
    Function to create the PCS polar plot with toggle for 0-180 or 0-90 degrees.
    """
    figure.clear()  # Clear the previous figure to avoid overlay
    ax = figure.add_subplot(1, 1, 1, projection='polar')
    #ax.set_position([0, 0.1, 0.8, 0.8])

    pcs_min = min(pcs_values)
    pcs_max = max(pcs_values)

    # Initialize theta range based on toggle
    if plot_90_degrees.get():
        theta_range = theta_values[theta_values <= np.pi / 2]  # 0~90 degree
        ax.set_position([0.2, 0.1, 0.4, 0.8])  # x, y, width, height
        ax.set_thetamax(90)  # limit 0~90 degree
        ax.set_xticks([0, np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 3, 5 * np.pi / 12, np.pi / 2])  # 0, 15, 30, 45, 60, 75, 90° label
        ax.set_xticklabels(["0°", "15°", "30°", "45°", "60°", "75°", "90°"], fontsize=8)  # label font size
        bbox = (1.3, 0.5)  
 
    else:
        theta_range = theta_values  # 0~180 degree
        ax.set_position([0.0, 0.1, 0.8, 0.8])
        ax.set_thetamax(180)  # limit 0~180 degree
        ax.set_xticks([0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6, np.pi])
        ax.set_xticklabels(["0°", "30°", "60°", "90°", "120°", "150°", "180°"], fontsize=8)
        bbox = (0.9, 0.5)

    scatter_points = []  # For handling polar_data

    for pcs_value in pcs_values:
        r_values = calculate_r(pcs_value, theta_range, tensor)

        # Handle colors
        if np.isclose(pcs_value, 0):
            color = 'white'
        elif pcs_value > 0:
            intensity = max(0, 1 - pcs_value / pcs_max)
            color = (1, intensity, intensity)
        else:
            intensity = max(0, 1 + pcs_value / abs(pcs_min))
            color = (intensity, intensity, 1)

        # Mirror data for 0~90 degrees if enabled
        if plot_90_degrees.get():
            mirrored_theta = np.pi - theta_range  # Reflect theta to 90~180 mirrored as 0~90
            combined_theta = np.concatenate((theta_range, mirrored_theta))
            combined_r = np.concatenate((r_values, r_values))
            ax.plot(combined_theta, combined_r, label=f'{pcs_value: .1f}', color=color)
        else:
            ax.plot(theta_range, r_values, label=f'{pcs_value: .1f}', color=color)

    # Plot polar_data
    if polar_data:
        for i, (atom, r, theta) in enumerate(polar_data):
            # Reflect data if 0~90 toggle is on
            if plot_90_degrees.get() and theta > np.pi / 2:
                theta = np.pi - theta  # Reflect to 0~90
            point = ax.scatter(theta, r, color=get_cpk_color(atom), zorder=5, s=15)
            scatter_points.append((point, i + 1))

    # Set radius ticks and labels
    r_ticks = [0, 2, 4, 6, 8, 10]  # r ticks in angstrom
    r_labels = [f"{r} Å" for r in r_ticks]
    ax.set_yticks(r_ticks)
    ax.set_yticklabels(r_labels)
    ax.tick_params(axis='y', labelsize=8)

    def on_click(event):
        for point, idx in scatter_points:
            contains, _ = point.contains(event)
            if contains:
                # Highlight corresponding row in table
                tree.selection_set(tree.get_children()[idx - 1])
                tree.see(tree.get_children()[idx - 1])
                break
    
    canvas.mpl_connect('button_press_event', on_click)
    
    # Set angle limits
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 10)  # r limit
    if plot_90_degrees.get():
        ax.set_thetamax(90)  # 0~90 degree
    else:
        ax.set_thetamax(180)  # 0~180 degree
    
    # Plot PCS legend (두 줄로 고정)
    legend = ax.legend(
        fontsize=6,
        bbox_to_anchor=bbox,
        loc="center left",
        ncol=2,  # 두 줄로 고정
        frameon=False,
        handletextpad=0.8,
        labelspacing=0.5,
        columnspacing=1.0,
        borderpad=0,
        borderaxespad=0.5,
        handlelength=1.0,
        handleheight=1.0,
    )
    legend.set_title("PCS legend (ppm)", prop={'size': 6, 'weight': 'bold'})  # Legend 타이틀 설정
    canvas.draw()

def update_figsize():
    """
    Dynamically recreate the canvas to ensure proper resizing.
    """
    global canvas, pcs_figure  # Ensure we refer to the global variables

    # Remove the existing canvas
    canvas.get_tk_widget().destroy()

    # Adjust figure size based on the toggle state
    if plot_90_degrees.get():  # 0~90 degree display option
        pcs_figure.set_size_inches(4, 2, forward=True)
    else:  # 0~180 degree display option
        pcs_figure.set_size_inches(4, 4, forward=True)

    # Recreate the canvas
    canvas = FigureCanvasTkAgg(pcs_figure, master=left_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Redraw the plot
    update_graph()


# Generate data to export in xlsx file
def save_to_excel(pcs_values, theta_values, tensor, file_name, polar_data):
    pcs_df = pd.DataFrame({'Theta': theta_values})

    for i, pcs_value in enumerate(pcs_values):
        r_values = calculate_r(pcs_value, theta_values, tensor)
        x_cartesian_values = r_values * np.sin(theta_values)
        y_cartesian_values = r_values * np.cos(theta_values)

        pcs_df[f'PCS (ppm) {i + 1}'] = [pcs_value] * len(theta_values)
        pcs_df[f'R {i + 1}'] = r_values
        pcs_df[f'X Cartesian {i + 1}'] = x_cartesian_values
        pcs_df[f'Y Cartesian {i + 1}'] = y_cartesian_values

    pcs_columns = ['Theta']
    for i in range(len(pcs_values)):
        pcs_columns.extend([f'PCS (ppm) {i + 1}', f'R {i + 1}', f'X Cartesian {i + 1}', f'Y Cartesian {i + 1}'])

    pcs_df = pcs_df[pcs_columns]

    polar_df = pd.DataFrame(polar_data, columns=['Atom', 'R', 'Theta'])
    polar_df['Geom Param'] = (3 * (np.cos(polar_df['Theta']))**2 - 1) / polar_df['R']**3

    with pd.ExcelWriter(file_name) as writer:
        pcs_df.to_excel(writer, sheet_name='PCS Data', index=False)
        polar_df.to_excel(writer, sheet_name='Atom Coordinates', index=False)

def update_graph(*args):
    try:
        tensor = tensor_entry.get()
        if not tensor:
            tensor = 1.0
        tensor = float(tensor)  # Δχ_ax tensor values

        # Get PCS range values from the entry fields
        pcs_min = float(pcs_min_entry.get())
        pcs_max = float(pcs_max_entry.get())
        pcs_interval = float(pcs_interval_entry.get())

        global pcs_values
        pcs_values = np.arange(pcs_min, pcs_max + pcs_interval, pcs_interval)

        # if there atom data exist
        if atom_data:
            polar_data, rotated_coords = filter_atoms()  # Rotate coordinates in filter_atoms
            update_table(polar_data, rotated_coords)  # Update the table

            # Extract atom list from atom_data
            atoms = [atom for atom, _, _, _ in atom_data]
            atom_elements = [atom for atom, _, _, _ in atom_data]  # Extract element symbols from atom_data
        else:
            polar_data = None
            rotated_coords = None
            atoms = None
            atom_elements = None

        # Plot PCS graph (update 2D PCS plot)
        plot_graph(pcs_values, theta_values, tensor, canvas, pcs_figure, polar_data=polar_data)

        # Plot Cartesian graph (update Cartesian plot)
        plot_cartesian_graph()

        # Update molar value
        update_molar_value(tensor)  # Update Δχ_ax_molar value

        # 3D plot update
        if rotated_coords and atoms and top_3d_window is not None and tk.Toplevel.winfo_exists(top_3d_window):
            plot_3d_molecule_with_bonds(rotated_coords, atoms, atom_elements, top_3d_window)

        # Forced UI update
        root.update_idletasks()
        print("[DEBUG] Graphs, table, and 3D plot successfully updated.")
    
    except ValueError as ve:
        print(f"[DEBUG] ValueError in update_graph: {ve}")
        pass  # If input is invalid, catch error and ignore update steps

    except Exception as e:
        print(f"[DEBUG] Unexpected error in update_graph: {e}")


def update_molar_value(tensor):
    molar_value = tensor * AVOGADRO_CONSTANT * 1e-32
    molar_value_label.config(text=f"Δχ_mol_ax: {molar_value:.2e} m³/mol")

# Using Δχ_mol_ax to calculate tensor properties
def calculate_tensor_components():
    try:
        # Enter χ_mol value (in m³/mol)
        chi_mol = float(chi_mol_entry.get())

        # Bring Δχ_mol_ax value
        delta_chi_ax = float(molar_value_label['text'].split(':')[1].strip().split()[0])

        # χ_perp and χ_parallel calculation
        chi_perp = chi_mol - delta_chi_ax / 3
        chi_parallel = (2 / 3) * delta_chi_ax + chi_mol

        # χ_xx, χ_yy, χ_zz calculation (χ_xx = χ_yy = χ_perp, χ_zz = χ_parallel)
        chi_xx = chi_perp
        chi_yy = chi_perp
        chi_zz = chi_parallel

        # Print results
        tensor_xx_label.config(text=f"χ_xx: {chi_xx:.2e} m³/mol")
        tensor_yy_label.config(text=f"χ_yy: {chi_yy:.2e} m³/mol")
        tensor_zz_label.config(text=f"χ_zz: {chi_zz:.2e} m³/mol")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for χ_mol.")

# χ_mol calculation when pressing Enter
def on_enter_key(event):
    calculate_tensor_components()


def on_delta_entry_change(event):
    selected_item = tree.selection()
    if selected_item:
        item = selected_item[0]
        column = tree.identify_column(event.x)
        if column == '#8':  # delta (ppm) column
            current_value = tree.set(item, column)
            new_value = simpledialog.askstring("Input", "Enter experimental chemical shift value (ppm):", initialvalue=current_value)
            if new_value is not None:
                try:
                    new_value_float = float(new_value)
                    tree.set(item, column, new_value)
                    i = int(tree.item(item, "values")[0])
                    delta_values[i] = new_value_float
                    plot_cartesian_graph()  # Update Cartesian plot with user input
                except ValueError:
                    messagebox.showerror("Invalid input", "Please enter a valid number for δ_Exp.")

# Table update functions
def update_table(polar_data, rotated_coords):
    print(f"[DEBUG] Updating table with {len(polar_data)} rows.")
    for row in tree.get_children():
        tree.delete(row)

    try:
        tensor = float(tensor_entry.get())  # Get the Δχ_ax value from the user input
    except ValueError:
        tensor = 1.0  # Default value if input is invalid

    for i, ((atom, r, theta), (dx, dy, dz)) in enumerate(zip(polar_data, rotated_coords), start=1):
        geom_param = (3 * (np.cos(theta)) ** 2 - 1) / (r ** 3)
        delta_value = delta_values.get(i, "")
        geom_value = geom_param
        delta_pcs = (tensor * (geom_value * 1e4)) / (12 * np.pi)  # Automatic calculation of delta_PCS
        tree.insert("", tk.END, values=(i, atom, f"{dx:.3f}", f"{dy:.3f}", f"{dz:.3f}", f"{geom_param:.5f}", f"{delta_pcs: .2f}", delta_value))

    print(f"[DEBUG] Table successfully updated with {len(polar_data)} rows.")
    root.update_idletasks()  # forced UI update

# Function to update the Cartesian plot in real-time and add a linear regression line
def plot_cartesian_graph():
    geom_values = []
    delta_values_list = []
    scaled_x_values = []

    # Get data from the table (tree)
    for item in tree.get_children():
        geom_value = float(tree.item(item, "values")[5])  # G_i value
        delta_value = tree.item(item, "values")[7]  # delta (ppm) value is in the 8th column
        if delta_value:
            delta_value = float(delta_value)
            geom_values.append(geom_value)
            delta_values_list.append(delta_value)
            scaled_x = (geom_value * 1e4) / (12 * np.pi)
            scaled_x_values.append(scaled_x)

    # Clear the existing figure
    cartesian_figure.clear()
    ax = cartesian_figure.add_subplot(111)

    if geom_values and delta_values_list:
        # Blue scatter plot and linear fit
        ax.scatter(geom_values, delta_values_list, color='blue', marker='o', label='δ vs G_i')
        slope, intercept, r_value, p_value, std_err = stats.linregress(geom_values, delta_values_list)
        regression_line = [slope * x + intercept for x in geom_values]
        ax.plot(geom_values, regression_line, color='blue', label='Linear Fit (Blue)')

        # Add annotation for blue plot
        stats_text = f"Blue Scatter\nSlope: {slope:.2e}\nIntercept: {intercept:.2e}\nR-value: {r_value:.3f}"
        annotation = ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='blue', facecolor=(1, 1, 1, 0.7)))
        DraggableAnnotation(annotation)

        # Red scatter plot and linear fit
        ax.scatter(scaled_x_values, delta_values_list, color='red', marker='o', label='δ vs Scaled G_i')
        slope_red, intercept_red, r_value_red, p_value_red, std_err_red = stats.linregress(scaled_x_values, delta_values_list)
        regression_line_red = [slope_red * x + intercept_red for x in scaled_x_values]
        ax.plot(scaled_x_values, regression_line_red, color='red', label='Linear Fit (Red)')

        # Add annotation for red plot
        stats_text_red = f"Red Scatter\nSlope: {slope_red:.2e}\nIntercept: {intercept_red:.2e}\nR-value: {r_value_red:.3f}"
        annotation_red = ax.text(0.4, 0.95, stats_text_red, transform=ax.transAxes, fontsize=8, fontweight='bold',
                                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor=(1, 1, 1, 0.7)))
        DraggableAnnotation(annotation_red)

        # Add legend
        ax.legend(fontsize=8)

    ax.set_xlabel('G_i')
    ax.set_ylabel('δ (ppm)')
    ax.set_title('Geometrical factor (G_i) vs Chemical shift (δ_Exp)')
    ax.grid(True)

    # Redraw the canvas to reflect updates
    cartesian_canvas.draw()
    root.update_idletasks()  # Force UI update

    #print("[DEBUG] Cartesian plot updated and UI refreshed.")

# Reset/Initialize to default values
def reset_values():
    tensor_entry.delete(0, tk.END)
    tensor_entry.insert(0, '-2.0')
    
    pcs_min_entry.delete(0, tk.END)
    pcs_min_entry.insert(0, '-10')
    
    pcs_max_entry.delete(0, tk.END)
    pcs_max_entry.insert(0, '10')
    
    pcs_interval_entry.delete(0, tk.END)
    pcs_interval_entry.insert(0, '0.5')

    angle_x.set(0)
    angle_y.set(0)
    angle_z.set(0)

    global atom_data, check_vars, pcs_values
    atom_data = []
    check_vars = {}
    pcs_values = np.arange(-10, 10.5, 0.5)
    
    for widget in checklist_frame.winfo_children():
        widget.destroy()

    for row in tree.get_children():
        tree.delete(row)

    plot_graph(pcs_values, theta_values, -2.0, canvas, pcs_figure)
    plot_cartesian_graph()

def save_graph_to_excel():
    tensor = tensor_entry.get()
    if not tensor:
        tensor = -2.0
    try:
        tensor = float(tensor)
        pcs_min = float(pcs_min_entry.get())
        pcs_max = float(pcs_max_entry.get())
        pcs_interval = float(pcs_interval_entry.get())
        global pcs_values
        pcs_values = np.arange(pcs_min, pcs_max + pcs_interval, pcs_interval)
        
        if atom_data:
            polar_data, _ = filter_atoms()
        else:
            polar_data = []
        
        file_path = filedialog.asksaveasfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            save_to_excel(pcs_values, theta_values, tensor, file_path, polar_data)
    except ValueError:
        pass


def save_table_to_excel():
    # Open the file save dialog to get the file name and path
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    
    if not file_path:
        return  # Exit the function if the user cancels

    # Initialize a DataFrame to store the table data
    table_data = []
    columns = ('Ref', 'Atom', 'X', 'Y', 'Z', 'G_i', 'δ_PCS', 'δ_Exp')
    
    # Read the data from the treeview (table) and store it in a list
    for item in tree.get_children():
        values = tree.item(item)["values"]
        table_data.append(values)
    
    # Convert the list to a DataFrame
    df = pd.DataFrame(table_data, columns=columns)
    # Save the DataFrame to an Excel file
    df.to_excel(file_path, index=False)

    messagebox.showinfo("Export Successful", f"Table data successfully exported to {file_path}")

def load_xyz_file():
    file_path = filedialog.askopenfilename(filetypes=[("XYZ files", "*.xyz")])
    if file_path:
        process_xyz_file(file_path)
        create_checklist()
        update_graph()  # Update PCS plot
        polar_data, rotated_coords = filter_atoms()
        update_table(polar_data, rotated_coords)
        plot_cartesian_graph()  # Update Cartesian plot

def process_xyz_file(file_path):
    global atom_data
    atom_data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines[2:]:  # Ignore the first two lines
        parts = line.split()
        atom_data.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))

    target_atom = simpledialog.askstring("Input center atom", "Enter the center atom (element) :")
    target_coords = None

    for atom, x, y, z in atom_data:
        if atom == target_atom:
            target_coords = (x, y, z)
            break

    if target_coords is None:
        messagebox.showerror("Error", f"Atom {target_atom} not found in the file.")
        return

    global x0, y0, z0
    x0, y0, z0 = target_coords

    # Update 2D graph after loading the file
    update_graph()    
    # Show 3D plot in a new window
    open_3d_plot_window()

# Calculate bonds between atoms based on covalent radii
def calculate_bonds(atom_coords, atom_elements):
    bond_pairs = []
    distances = pdist(atom_coords)  # Compute pairwise distances
    dist_matrix = squareform(distances)  # Convert to a distance matrix
    
    for i in range(len(atom_coords)):
        for j in range(i+1, len(atom_coords)):
            # Get the covalent radii of the two atoms
            radius_sum = covalent_radii.get(atom_elements[i], 0) + covalent_radii.get(atom_elements[j], 0)
            # Check if the distance is within bond range
            if dist_matrix[i, j] <= radius_sum * 1.1:  # Tolerance for bond distances (default is 10%), Adjust this value if chemical bonds are not displayed correctly.
                bond_pairs.append((i, j))
    
    return bond_pairs

# https://github.com/radi0sus/xyz2tab
# Function to visualize the 3D molecule
def plot_3d_molecule_with_bonds(rotated_coords, atoms, atom_elements, master_window):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Find the center atom's coordinates (assumed to be the first one in the list)
    center_atom_coords = rotated_coords[0]  # Assuming the first atom is the center
    cx, cy, cz = center_atom_coords

    # Shift all atom coordinates so that the center atom is at the origin (0, 0, 0)
    shifted_coords = [(x - cx, y - cy, z - cz) for x, y, z in rotated_coords]

    # Plot each atom with size proportional to covalent radii
    for atom, (x, y, z), element in zip(atoms, shifted_coords, atom_elements):
        # Get the covalent radius for the element, use default if element is missing
        radius = covalent_radii.get(element, 1.0)  # Set default radius to 1.0
        size = radius * 50  # Set point size proportional to radius, adjust scale appropriately

        # Plot the atom as a scatter point with size proportional to covalent radius
        ax.scatter(x, y, z, color=get_cpk_color(atom), s=size, alpha=0.7)
        #ax.text(x + 0.2, y + 0.2, z + 0.2, atom, fontsize=8)  # Uncomment if you want to add atom labels

    # Calculate bonds based on covalent radii and plot them
    bond_pairs = calculate_bonds(np.array(shifted_coords), atom_elements)
    for i, j in bond_pairs:
        ax.plot([shifted_coords[i][0], shifted_coords[j][0]],
                [shifted_coords[i][1], shifted_coords[j][1]],
                [shifted_coords[i][2], shifted_coords[j][2]], color='gray', lw=2.0, alpha=1.0) # bond display options

    # Add XYZ axis arrows
    arrow_length = 8
    ax.arrow3D(0, 0, 0, arrow_length, 0, 0, mutation_scale=20, ec='black', fc='blue')   # X axis arrow option
    ax.arrow3D(0, 0, 0, 0, arrow_length, 0, mutation_scale=20, ec='black', fc='green')  # Y axis arrow option
    ax.arrow3D(0, 0, 0, 0, 0, arrow_length, mutation_scale=20, ec='black', fc='red')    # Z axis arrow option

    # Add labels to the ends of the arrows
    ax.text(arrow_length, 0, 0, 'X', color='blue', fontsize=12, weight='bold')
    ax.text(0, arrow_length, 0, 'Y', color='green', fontsize=12, weight='bold')
    ax.text(0, 0, arrow_length, 'Z', color='red', fontsize=12, weight='bold')

    # Remove grid, ticks, and background axes
    ax.grid(False)
    ax.set_xticks([])            
    ax.set_yticks([])            
    ax.set_zticks([])            
    ax.set_xlabel('')            
    ax.set_ylabel('')            
    ax.set_zlabel('')            
    ax.set_axis_off()

    # Set equal axis limits
    set_axes_equal(ax)

    plt.tight_layout()

    # Update the canvas for the 3D plot
    canvas = FigureCanvasTkAgg(fig, master=master_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Add toolbar for navigation (zoom, pan, save)
    toolbar = NavigationToolbar2Tk(canvas, master_window)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Manage the 3D window as a global variable
top_3d_window = None

def visualize_molecule_3d(root, coords, atoms):
    global top_3d_window

    new_window = tk.Toplevel(root)
    new_window.title("3D Molecular Structure")
    new_window.geometry("600x600")

    # Rotate coordinates based on current slider values
    rotated_coords = rotate_coordinates(coords, angle_x.get(), angle_y.get(), angle_z.get())
    
    # Extract element symbols from atoms
    atom_elements = [atom for atom, _, _, _ in atom_data]  # Extract element symbols from atom_data

    # Call plot_3d_molecule_with_bonds to draw or update the 3D plot
    plot_3d_molecule_with_bonds(rotated_coords, atoms, atom_elements, new_window)

# set equal axes aspect ratio for 3D plots
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Add functionality to load 3D scatter plot window
def open_3d_plot_window():
    if atom_data:
        coords = np.array([[x, y, z] for atom, x, y, z in atom_data])
        atoms = [atom for atom, _, _, _ in atom_data]
        visualize_molecule_3d(root, coords, atoms)
    else:
        messagebox.showerror("Error", "No atom data loaded")

# Mollweide projection plot
def plot_theta_phi_scatter(theta, phi, elements, colors):
    """
    Creates a scatter plot with azimuthal (theta) and polar (phi) angles using element colors.
    """
    new_window = tk.Toplevel(root)
    new_window.title("Mollweide projection plot")
    new_window.geometry("800x400")
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'mollweide'}, figsize=(8, 6))
    ax.grid(True)
    
    # Plot scatter points with corresponding colors
    for t, p, c, e in zip(theta, phi, colors, elements):
        ax.scatter(t, p, color=c, label=e, s=25, alpha=0.8)
    
    ax.set_xlabel("Azimuthal angle (φ)", fontsize=8)
    ax.set_ylabel("Polar angle (θ)", fontsize=8)
    #ax.legend(fontsize=8, loc='upper left', title="Elements")
    #ax.set_title("Mollweide projection plot", fontsize=8)  #uncomment this if you need a title for the plot
    
    # Add a bold horizontal line for theta = 0 (Polar angle = 0)
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    
    # Add the plot to the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def cartesian_to_spherical_with_colors(coords, elements):
    """
    Convert Cartesian coordinates to spherical (azimuthal and polar angles) and retrieve corresponding colors.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    
    r = np.sqrt(x**2 + y**2 + z**2)  # Radius
    phi = np.arcsin(z / r)          # Polar angle
    theta = np.arctan2(y, x)        # Azimuthal angle
    
    colors = [get_cpk_color(el) for el in elements]
    return theta, phi, colors

def open_theta_phi_plot():
    """
    Open a new plot window with azimuthal and polar angles.
    """
    try:
        # Use the latest rotated coordinates if available
        if last_rotated_coords is not None:
            coords = last_rotated_coords
        else:
            coords = np.array([[x, y, z] for _, x, y, z in atom_data])
        
        # Translate to origin and rotate
        center_coords = np.array([x0, y0, z0])  # Center atom coordinates
        shifted_coords = coords - center_coords  # Translate to origin
        rotated_coords = rotate_coordinates(
            shifted_coords,
            angle_x.get(),
            angle_y.get(),
            angle_z.get()
        )
        
        # Extract element types
        elements = [atom[0] for atom in atom_data]
        
        # Convert to spherical coordinates and get colors
        theta, phi, colors = cartesian_to_spherical_with_colors(rotated_coords, elements)
        
        # Plot the scatter plot
        plot_theta_phi_scatter(theta, phi, elements, colors)
        
    except Exception as e:
        print(f"[DEBUG] Error in open_theta_phi_plot: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")

# atom element selection checkbox
def create_checklist():
    for widget in checklist_frame.winfo_children():
        widget.destroy()

    global check_vars
    check_vars = {}
    atom_types = sorted(set([atom[0] for atom in atom_data]))

    canvas = tk.Canvas(checklist_frame, width=100, height=60)
    scrollbar = tk.Scrollbar(checklist_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    for i, atom_type in enumerate(atom_types):
        var = tk.BooleanVar(value=True)
        chk = tk.Checkbutton(scrollable_frame, text=atom_type, variable=var, command=update_graph)
        row = i // 3
        col = i % 3
        chk.grid(row=row, column=col, sticky='w', padx=5, pady=2)
        check_vars[atom_type] = var

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    update_graph()

def on_table_select(event):
    selected_item = tree.selection()
    if selected_item:
        item = selected_item[0]
        values = tree.item(item, 'values')
        atom = values[1]
        dx = float(values[2])
        dy = float(values[3])
        dz = float(values[4])

        for atom_idx, (a, r, theta) in enumerate(polar_data):
            if a == atom and np.isclose([dx, dy, dz], [r * np.sin(theta), r * np.cos(theta), dz]).all():
                plot_highlighted_point(atom_idx)
                break

def plot_highlighted_point(atom_idx):
    plot_graph(pcs_values, theta_values, float(tensor_entry.get()), canvas, pcs_figure, polar_data)
    ax = pcs_figure.gca(projection='polar')
    atom, r, theta = polar_data[atom_idx]
    ax.scatter(theta, r, color='yellow', zorder=10, s=50, edgecolor='black')  # highlight effect
    canvas.draw()

def filter_atoms():
    global atom_data, check_vars, x0, y0, z0
    selected_atoms = [atom for atom, var in check_vars.items() if var.get()]
    polar_data = []

    coords = np.array([[x - x0, y - y0, z - z0] for atom, x, y, z in atom_data])
    rotated_coords = rotate_coordinates(coords, angle_x.get(), angle_y.get(), angle_z.get())

    for (atom, _, _, _), (dx, dy, dz) in zip(atom_data, rotated_coords):
        if atom in selected_atoms:
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            theta = np.arccos(dz / r) if r != 0 else 0
            polar_data.append((atom, r, theta))

    return polar_data, rotated_coords

def close_window():
    root.destroy()

def save_graph():
    file_path = filedialog.asksaveasfilename(filetypes=[("PNG Image", "*.png")])
    if file_path:
        canvas.print_figure(file_path, dpi=600)

root = tk.Tk()
root.title("PCS Analyzer")
root.geometry("1280x800")

# New global variable to manage the toggle state
plot_90_degrees = tk.BooleanVar(value=False)

'''
----------------------↓↓↓↓↓↓↓↓Working in progress for the next update↓↓↓↓↓↓↓↓-------------------------------------
'''
# 좌표 변환 상태를 저장할 변수
last_rotated_coords = None

# Helper function: Move coordinates relative to a center atom (to origin)
def move_to_origin(coords, center_atom):
    # 중심 원자 좌표를 원점으로 이동
    return coords - np.array(center_atom)

def restore_from_origin(coords, center_atom):
    # 원점에서 중심 원자로 좌표를 복귀
    return coords + np.array(center_atom)

# Rotate coordinates around the center atom
def rotate_coordinates(coords, angle_x, angle_y, angle_z, center_atom=(0, 0, 0)):
    angle_x = np.radians(angle_x)
    angle_y = np.radians(angle_y)
    angle_z = np.radians(angle_z)

    center_atom_coords = coords[0]  # 첫 번째 원자(중심 원자)의 좌표
    coords = translate_to_origin(coords, center_atom_coords)  # 중심 원자를 원점으로 이동

    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, np.cos(angle_x), -np.sin(angle_x)],
                                  [0, np.sin(angle_x), np.cos(angle_x)]])
    
    rotation_matrix_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                                  [0, 1, 0],
                                  [-np.sin(angle_y), 0, np.cos(angle_y)]])
    
    rotation_matrix_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                                  [np.sin(angle_z), np.cos(angle_z), 0],
                                  [0, 0, 1]])

    rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x    
    rotated_coords = np.dot(coords, rotation_matrix.T)
    
    # 중심 원자의 좌표를 원점에 고정
    rotated_coords -= rotated_coords[0]

    # 변환된 좌표를 중심 원자 좌표로 복귀
    result = translate_from_origin(rotated_coords, center_atom_coords)
    return result

# Helper function: Translate all coordinates so that the center atom is at the origin
def translate_to_origin(coords, center_atom_coords):
    return coords - center_atom_coords

# 원점에서 중심 좌표로 복귀 (이미 중심 원자가 원점에 위치하므로, 이 부분에서 오류 발생 가능성)
def translate_from_origin(coords, center_atom_coords):
    # 중심 원자가 원점에 있으면, 다시 이동할 필요가 없음
    return coords + center_atom_coords

# Align coordinates using two vectors and translate to ensure the center atom remains at the origin
def align_xyz(vec1, vec2, coords, center_atom):
    # 중심 원자를 원점으로 이동
    center_atom_coords = coords[0]  # 첫 번째 원자(중심 원자)의 좌표
    print(f"[DEBUG] Initial center atom coordinates: {center_atom_coords}")
    
    coords = translate_to_origin(coords, center_atom_coords)  # 중심 원자를 원점으로 이동
    print(f"[DEBUG] Coordinates after translating to origin:\n{coords[:5]}...")  # 첫 5개만 출력
    
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        print("[DEBUG] Zero vector found in alignment, returning original coordinates")
        return translate_from_origin(coords, center_atom_coords)  # 원래 좌표로 복귀

    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    cross_prod = np.cross(vec1, vec2)
    dot_prod = np.dot(vec1, vec2)

    print(f"[DEBUG] Cross product: {cross_prod}, Dot product: {dot_prod}")

    if np.linalg.norm(cross_prod) < 1e-12:
        # Vectors are already aligned or anti-parallel
        print("[DEBUG] Vectors are already aligned or anti-parallel")
        if dot_prod < 0:
            coords = -coords  # 180-degree 회전
        return translate_from_origin(coords, center_atom_coords)  # 원래 좌표로 복귀

    # 회전 행렬을 생성
    skew_symmetric = np.array([[0, -cross_prod[2], cross_prod[1]],
                               [cross_prod[2], 0, -cross_prod[0]],
                               [-cross_prod[1], cross_prod[0], 0]])

    identity_matrix = np.eye(3)
    rotation_matrix = identity_matrix + skew_symmetric + \
                      (np.dot(skew_symmetric, skew_symmetric)) * ((1 - dot_prod) / (np.linalg.norm(cross_prod) ** 2))

    print(f"[DEBUG] Rotation matrix:\n{rotation_matrix}")

    rotated_coords = np.dot(coords, rotation_matrix.T)
    print(f"[DEBUG] Coordinates after rotation:\n{rotated_coords[:5]}...")  # 첫 5개만 출력

    # 중심 원자의 좌표를 원점에 고정
    rotated_coords -= rotated_coords[0]  # 중심 좌표에 따라 모든 좌표가 이동
    print(f"[DEBUG] Coordinates after adjusting to center at origin:\n{rotated_coords[:5]}...")

    # 이 시점에서 중심 원자가 원점에 있으므로, translate_from_origin을 호출할 때 중심 원자 좌표를 다시 잘못 적용하지 않도록 주의
    result = translate_from_origin(rotated_coords, np.array([0, 0, 0]))  # 중심 좌표가 원점에 있으므로 이동 필요 없음
    print(f"[DEBUG] Final aligned coordinates:\n{result[:5]}...")
    return result

# 명령어 처리 함수
def process_command():
    global last_rotated_coords, x0, y0, z0
    command = command_entry.get()  # 명령어 입력창에서 입력된 내용 가져오기
    args = command.split()

    if not args:
        messagebox.showerror("Error", "No command entered.")
        return

    # Update the angle sliders after the rotation command is executed
    def update_angle_sliders(angle_x_cmd, angle_y_cmd, angle_z_cmd):
        angle_x.set(angle_x_cmd)
        angle_y.set(angle_y_cmd)
        angle_z.set(angle_z_cmd)

        # Update the corresponding entry widgets for each slider
        angle_x_entry.delete(0, tk.END)
        angle_x_entry.insert(0, f"{float(angle_x_cmd):.1f}")
        angle_y_entry.delete(0, tk.END)
        angle_y_entry.insert(0, f"{float(angle_y_cmd):.1f}")

    try:
        if args[0] == 'rotate':  # 회전 명령어 처리
            if len(args) != 4:
                messagebox.showerror("Error", "Usage: rotate <angle_x> <angle_y> <angle_z>")
                return
            angle_x_cmd, angle_y_cmd, angle_z_cmd = float(args[1]), float(args[2]), float(args[3])
            coords = last_rotated_coords if last_rotated_coords is not None else np.array([[x, y, z] for _, x, y, z in atom_data])
            center_atom = (x0, y0, z0)  # 중심 원자 좌표
            rotated_coords = rotate_coordinates(coords, angle_x_cmd, angle_y_cmd, angle_z_cmd, center_atom)  # 중심 원자를 기준으로 회전
            last_rotated_coords = rotated_coords  # 좌표 갱신

            # 슬라이더 업데이트
            update_angle_sliders(angle_x_cmd, angle_y_cmd, angle_z_cmd)
            print("[DEBUG] Rotated coordinates updated and sliders adjusted.")

        elif args[0] == 'align':  # 정렬 명령어 처리
            if len(args) != 7:
                messagebox.showerror("Error", "Usage: align <vec1_x> <vec1_y> <vec1_z> <vec2_x> <vec2_y> <vec2_z>")
                return
            vec1 = [float(args[1]), float(args[2]), float(args[3])]
            vec2 = [float(args[4]), float(args[5]), float(args[6])]
            print(f"[DEBUG] Align command received with vec1: {vec1}, vec2: {vec2}")

            coords = last_rotated_coords if last_rotated_coords is not None else np.array([[x, y, z] for _, x, y, z in atom_data])
            center_atom = (x0, y0, z0)  # 중심 원자 좌표
            
            # 좌표를 중심 원자로부터 이동 후, 정렬
            aligned_coords = align_xyz(vec1, vec2, coords, center_atom)  
            print(f"[DEBUG] Aligned coordinates: {aligned_coords}")
            last_rotated_coords = aligned_coords  # 좌표 갱신

            # 슬라이더를 변환된 상태로 업데이트
            update_angle_sliders(0, 0, 0)
            print("[DEBUG] Sliders reset after align.")

        else:
            messagebox.showerror("Error", f"Unknown command: {args[0]}")

        # 좌표 변환 후 2D 플롯 및 테이블을 실시간으로 업데이트
        print("[DEBUG] Calling update_all_plots_and_tables.")
        update_all_plots_and_tables()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        print(f"[DEBUG] Error in process_command: {e}")

# Update all plots and tables
def update_all_plots_and_tables():
    if last_rotated_coords is not None:
        print("[DEBUG] Updating all plots and tables with new coordinates.")
        update_graph()  # 2D PCS 업데이트
        plot_cartesian_graph()  # Cartesian plot 업데이트
        polar_data, rotated_coords = filter_atoms()  # 필터된 원자 데이터와 회전된 좌표
        print("[DEBUG] Filtered atoms and rotated coordinates obtained.")

        # 필터링된 데이터와 변환된 좌표로 테이블을 업데이트
        update_table(polar_data, rotated_coords)  # 테이블 업데이트
        print("[DEBUG] Table updated with rotated coordinates.")

# Command input GUI
def add_command_input_gui(main_frame):
    command_frame = tk.Frame(main_frame)
    command_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)


    # Command label
    command_label = tk.Label(command_frame, text="Command :")
    command_label.pack(side=tk.LEFT)

    global command_entry
    command_entry = tk.Entry(command_frame, width=50)
    command_entry.pack(side=tk.LEFT, padx=5)

    # Run command button
    command_button = tk.Button(command_frame, text="Run", command=process_command)
    command_button.pack(side=tk.LEFT, padx=5)


    # Exit button
    close_button = tk.Button(command_frame, text="Exit", command=close_window)
    close_button.pack(side=tk.RIGHT, padx=0)
    
    # Export data button
    save_buttons_frame = tk.Frame(command_frame)
    save_buttons_frame.pack(side=tk.RIGHT, padx=15)

    # Export data label
    save_label = tk.Label(save_buttons_frame, text="Export data", font=("default", 9, "bold"))
    save_label.grid(row=1, column=0,sticky="ew", pady=3)

    # Save plot as Excel file button
    save_button = tk.Button(save_buttons_frame, text="plot(xlsx)", command=save_graph_to_excel)
    save_button.grid(row=1, column=1, sticky="ew", padx=0, pady=1)

    # Save plot as PNG button
    save_png_button = tk.Button(save_buttons_frame, text="plot(png)", command=save_graph)
    save_png_button.grid(row=1, column=2, sticky="ew", padx=0, pady=1)

    # Save table as Excel file button
    save_table_button = tk.Button(save_buttons_frame, text="table(xlsx)", command=save_table_to_excel)
    save_table_button.grid(row=1, column=3, sticky="ew", padx=0, pady=1)
    
'''
----------------------↑↑↑↑↑↑↑Working in progress for the next update↑↑↑↑↑↑↑-------------------------------------
'''

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# add command input gui
add_command_input_gui(main_frame)

left_frame = tk.Frame(main_frame)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

center_frame = tk.Frame(main_frame)
center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

right_frame = tk.Frame(main_frame)
right_frame.pack(side=tk.LEFT, fill=tk.Y)

# PCS Plot figure
pcs_figure = plt.figure(figsize=(4, 4), dpi=150)
canvas = FigureCanvasTkAgg(pcs_figure, master=left_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Table options
table_frame = tk.Frame(center_frame)
table_frame.pack(side=tk.TOP, fill=tk.X, padx=3, pady=0)

columns = ('Ref', 'Atom', 'X', 'Y', 'Z', 'G_i', 'δ_PCS', 'δ_Exp')
tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15) #display 15 rows

for col in columns:
    tree.heading(col, text=col)

# Column width
tree.column('Ref', width=45)
tree.column('Atom', width=45)
tree.column('X', width=50)  
tree.column('Y', width=50)  
tree.column('Z', width=50)  
tree.column('G_i', width=70)
tree.column('δ_PCS', width=60) 
tree.column('δ_Exp', width=60)

scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

tree.bind("<Double-1>", on_delta_entry_change)
# Bind event when clicking table item
tree.bind("<<TreeviewSelect>>", on_table_select)

# G_i vs δ plot
cartesian_frame = tk.Frame(center_frame)
cartesian_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=3, pady=0)

cartesian_figure = plt.Figure(figsize=(4, 3), dpi=100)
cartesian_canvas = FigureCanvasTkAgg(cartesian_figure, master=cartesian_frame)
cartesian_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Control UI
input_frame = tk.Frame(right_frame)
input_frame.pack(fill=tk.Y, padx=10, pady=10)

# Frame for Δχ_ax input
tensor_frame = tk.Frame(input_frame)
tensor_frame.pack(fill=tk.X, pady=3)

tensor_label = tk.Label(tensor_frame, text="Δχ_ax values (E-32 m³):", font=("default", 9, "bold"))
tensor_label.pack(side=tk.LEFT)

# Define and initialize tensor_entry
tensor_entry = tk.Entry(tensor_frame, width=5)
tensor_entry.pack(side=tk.LEFT, padx=5)

# Frame for Min and Max PCS
pcs_range_frame = tk.Frame(input_frame)
pcs_range_frame.pack(fill=tk.X, pady=3)

# Label for PCS range title
pcs_min_max_label = tk.Label(pcs_range_frame, text="PCS plot range (ppm)", font=("default", 9, "bold"))
pcs_min_max_label.pack(side=tk.TOP, pady=0)

pcs_entry_frame = tk.Frame(pcs_range_frame)
pcs_entry_frame.pack(side=tk.TOP, pady=0)

# Label Min Max Entry
pcs_min_label = tk.Label(pcs_entry_frame, text="Min:")
pcs_min_label.pack(side=tk.LEFT)

pcs_min_entry = tk.Entry(pcs_entry_frame, width=5)
pcs_min_entry.pack(side=tk.LEFT, padx=5)

separator_label = tk.Label(pcs_entry_frame, text="/")
separator_label.pack(side=tk.LEFT, padx=0)

pcs_max_label = tk.Label(pcs_entry_frame, text="Max:")
pcs_max_label.pack(side=tk.LEFT)

pcs_max_entry = tk.Entry(pcs_entry_frame, width=5)
pcs_max_entry.pack(side=tk.LEFT, padx=5)

# Frame for PCS plot interval input
pcs_interval_frame = tk.Frame(input_frame)
pcs_interval_frame.pack(fill=tk.X, pady=3)

# Toggle PCS plot range
plot_range_frame = tk.Frame(input_frame)
plot_range_frame.pack(fill=tk.X, pady=0)

plot_range_label = tk.Label(plot_range_frame, text="Half/Quarter plot toggle", font=("default", 9, "bold"))
plot_range_label.pack(side=tk.LEFT)

toggle_plot_range_button = tk.Checkbutton(
    plot_range_frame,
    variable=plot_90_degrees,
    command=lambda: [update_figsize(), update_graph()]
)
toggle_plot_range_button.pack(side=tk.LEFT)


pcs_interval_label = tk.Label(pcs_interval_frame, text="PCS plot interval (ppm):", font=("default", 9, "bold"))
pcs_interval_label.pack(side=tk.LEFT)

pcs_interval_entry = tk.Entry(pcs_interval_frame, width=5)
pcs_interval_entry.pack(side=tk.LEFT, padx=5)

# Update and Reset Buttons
button_frame = tk.Frame(input_frame)
button_frame.pack(pady=3)

update_button = tk.Button(button_frame, text="Update", command=update_graph)
update_button.pack(side=tk.LEFT, padx=5)

reset_button = tk.Button(button_frame, text="Reset", command=reset_values)
reset_button.pack(side=tk.LEFT, padx=5)

'''
------------------------------------------------------------------
'''
separator = ttk.Separator(input_frame, orient='horizontal')
separator.pack(fill='x', pady=8)
'''
------------------------------------------------------------------
'''

molar_value_label = tk.Label(input_frame, text="Δχ_mol_ax : N/A m³/mol")
molar_value_label.pack(pady=0)

# χ_mol from experiment
chi_mol_label = tk.Label(input_frame, text="χ_mol from Exp. (m³/mol):")
chi_mol_label.pack()

# χ_mol entry field
chi_mol_entry = tk.Entry(input_frame)
chi_mol_entry.pack()

# call calculate_tensor_components functions after press Enter key
chi_mol_entry.bind('<Return>', on_enter_key)

# χ_xx, χ_yy, χ_zz label
tensor_xx_label = tk.Label(input_frame, text="χ_xx: N/A m³/mol")
tensor_xx_label.pack()

tensor_yy_label = tk.Label(input_frame, text="χ_yy: N/A m³/mol")
tensor_yy_label.pack()

tensor_zz_label = tk.Label(input_frame, text="χ_zz: N/A m³/mol")
tensor_zz_label.pack()

'''
------------------------------------------------------------------
'''
separator = ttk.Separator(input_frame, orient='horizontal')
separator.pack(fill='x', pady=8)
'''
------------------------------------------------------------------
'''

load_xyz_button = tk.Button(input_frame, text="Load xyz File", command=load_xyz_file)
load_xyz_button.pack()

instruction_label = tk.Label(input_frame, text="The coordinates should align\n the molecule's rotational axis\n with the z-axis for proper analysis.", font=("Helvetica", 8, "italic"))
instruction_label.pack(pady=0)

'''
------------------------------------------------------------------
'''
separator = ttk.Separator(input_frame, orient='horizontal')
separator.pack(fill='x', pady=8)
'''
------------------------------------------------------------------
'''

angle_x_label = tk.Label(input_frame, text="Rotate around X-axis (degrees):")
angle_x_label.pack()

angle_x_frame = tk.Frame(input_frame)
angle_x_frame.pack(fill=tk.X)

angle_x = tk.DoubleVar()
angle_x_slider = tk.Scale(angle_x_frame, from_=-180, to=180, orient=tk.HORIZONTAL, variable=angle_x, command=update_angle_x_slider, resolution=0.1)
angle_x_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

angle_x_entry = tk.Entry(angle_x_frame, width=6)
angle_x_entry.pack(side=tk.RIGHT, padx=5)
angle_x_entry.insert(0, '0.0')
angle_x_entry.bind('<Return>', lambda event: on_angle_x_entry_change())

angle_y_label = tk.Label(input_frame, text="Rotate around Y-axis (degrees):")
angle_y_label.pack()

angle_y_frame = tk.Frame(input_frame)
angle_y_frame.pack(fill=tk.X)

angle_y = tk.DoubleVar()
angle_y_slider = tk.Scale(angle_y_frame, from_=-180, to=180, orient=tk.HORIZONTAL, variable=angle_y, command=update_angle_y_slider, resolution=0.1)
angle_y_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

angle_y_entry = tk.Entry(angle_y_frame, width=6)
angle_y_entry.pack(side=tk.RIGHT, padx=5)
angle_y_entry.insert(0, '0.0')
angle_y_entry.bind('<Return>', lambda event: on_angle_y_entry_change())

angle_z = tk.DoubleVar()

# Frame for plot buttons
open_plot_frame = tk.Frame(input_frame)
open_plot_frame.pack(fill=tk.X, padx=0, pady=3)

plot_label = tk.Label(open_plot_frame, text="Open 3d structure/projection", font=("default", 9, "bold"))
plot_label.pack(pady=3)

# Add button to main interface to open 3D plot window
load_3d_plot_button = tk.Button(open_plot_frame, text="mol structure", command=open_3d_plot_window)
load_3d_plot_button.pack(side=tk.LEFT, fill=tk.X,expand=True, padx=1)

# Add button to main interface to open theta-phi plot window
load_theta_phi_button = tk.Button(open_plot_frame, text="Projection", command=open_theta_phi_plot)
load_theta_phi_button.pack(side=tk.LEFT, fill=tk.X,expand=True, padx=1)

'''
------------------------------------------------------------------
'''
separator = ttk.Separator(input_frame, orient='horizontal')
separator.pack(fill='x', pady=8)
'''
------------------------------------------------------------------
'''

instruction_label = tk.Label(input_frame, text="Select elements to display", font=("default", 9, "bold"))
instruction_label.pack(pady=0)

checklist_frame = tk.Frame(input_frame)
checklist_frame.pack(pady=3, fill=tk.BOTH, expand=True)

instruction_label = tk.Label(input_frame, text="Rotation and atom selection can be\n applied after updating PCS information.", font=("Helvetica", 8, "italic"))
instruction_label.pack(pady=0)

'''
------------------------------------------------------------------
'''
separator = ttk.Separator(input_frame, orient='horizontal')
separator.pack(fill='x', pady=8)
'''
------------------------------------------------------------------
'''

polar_data = []
atom_data = []
check_vars = {}

# default and initial values
pcs_values = np.arange(-10, 10.5, 0.5)
theta_values = np.linspace(0, 2 * np.pi, 500)
plot_graph(pcs_values, theta_values, -2.0, canvas, pcs_figure)
plot_cartesian_graph()

root.mainloop()



'''
When we try to pick out anything by itself, we find it hitched to everything else in the universe.

—John Muir, My First Summer in the Sierra
'''
