'''
2D PCS plot analyzer v3.2

The following code requires the numpy, matplotlib, pandas, and openpyxl module packages to run.
It is designed for 2d PCS plot visualization (using only axiality of magnetic susceptibility tensor).
When exporting to an Excel file, the file extension should be included. e.g.[filename.xlsx]

v1.1 updates
Excel export: x, y cartesian coordinates are calculated, and each column are separated by PCS values.

v.1.2 updates
-tensor values, PCS range, and intervals can now be input for calculations.
-molar susceptibility tensor calculation

...

v.2.7 updates
-XYZ file import and plot, coordinate rotation, atom coordinate table added, geometrical parameter analysis plot

v.2.7.1 updates
-clicking on a atom point on the PCS plot to highlight the corresponding table entry, enhancing data visualization and interaction.

2024.08. Boseok Hong [Department of Chemistry of the f-elements, Institute of Resource Ecology, HZDR]
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, simpledialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats

# Constants and initial settings
AVOGADRO_CONSTANT = 6.0221408e23
delta_values = {}

# CPK Colors dictionary
CPK_COLORS = {
    'H': '#656565',
    'C': '#111111',
    'N': '#0000FF',
    'O': '#FF0000',
    'F': '#90E050',
    'Cl': '#1FF01F',
    'Br': '#A62929',
    'I': '#940094',
    'He': '#D9FFFF',
    'Ne': '#B3E3F5',
    'Ar': '#80D1E3',
    'Xe': '#67D3E0',
    'Kr': '#5CB8D1',
    'P': '#FF8000',
    'S': '#FFFF30',
    'B': '#FFB5B5',
    'Li': '#CC80FF',
    'Be': '#C2FF00',
    'Na': '#AB5CF2',
    'Mg': '#8AFF00',
    'Al': '#BFA6A6',
    'Si': '#F0C8A0',
    'K': '#8F40D4',
    'Ca': '#3DFF00',
    'Fe': '#E06633',
    'default': '#70ABFA'
}

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

def plot_graph(pcs_values, theta_values, tensor, canvas, figure, polar_data=None):
    figure.clear()
    ax = figure.add_subplot(1, 1, 1, projection='polar')
    ax.set_position([0.1, 0.1, 0.6, 0.8])
    
    ax.tick_params(axis='both', which='major', labelsize=8)
    pcs_min = min(pcs_values)
    pcs_max = max(pcs_values)
    
    scatter_points = []
    
    for pcs_value in pcs_values:
        r_values = calculate_r(pcs_value, theta_values, tensor)
        if np.isclose(pcs_value, 0):
            color = 'white'
        elif pcs_value > 0:
            intensity = max(0, 1 - pcs_value / pcs_max)
            color = (1, intensity, intensity)
        else:
            intensity = max(0, 1 + pcs_value / abs(pcs_min))
            color = (intensity, intensity, 1)

        ax.plot(theta_values, r_values, label=f'PCS={pcs_value}', color=color)
    
    if polar_data:
        for i, (atom, r, theta) in enumerate(polar_data):
            point = ax.scatter(theta, r, color=get_cpk_color(atom), zorder=5, s=15)
            scatter_points.append((point, i + 1))  # 점과 테이블의 인덱스(i+1)을 연결
    
    def on_click(event):
        for point, idx in scatter_points:
            contains, _ = point.contains(event)
            if contains:
                # 테이블의 해당 행을 하이라이트
                tree.selection_set(tree.get_children()[idx - 1])
                tree.see(tree.get_children()[idx - 1])
                break
    
    canvas.mpl_connect('button_press_event', on_click)
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 10)
    ax.grid(True)
    ax.legend(fontsize=6, bbox_to_anchor=(1.15, 0.5), loc="center left", ncol=1, frameon=False)

    canvas.draw()

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
    tensor = tensor_entry.get()
    if not tensor:
        tensor = 1.0
    try:
        tensor = float(tensor)
        pcs_min = float(pcs_min_entry.get())
        pcs_max = float(pcs_max_entry.get())
        pcs_interval = float(pcs_interval_entry.get())
        global pcs_values
        pcs_values = np.arange(pcs_min, pcs_max + pcs_interval, pcs_interval)

        if atom_data:
            polar_data, rotated_coords = filter_atoms()
            update_table(polar_data, rotated_coords)  # Update the table
        else:
            polar_data = None
        
        plot_graph(pcs_values, theta_values, tensor, canvas, pcs_figure, polar_data=polar_data)  # Update PCS plot
        plot_cartesian_graph()  # Update Cartesian plot
        update_molar_value(tensor)  # Update Δχ_ax_molar value
    except ValueError:
        pass  # Handle invalid input

def update_molar_value(tensor):
    molar_value = tensor * AVOGADRO_CONSTANT * 1e-32
    molar_value_label.config(text=f"Δχ_mol_ax: {molar_value:.2e} m³/mol")

def on_delta_entry_change(event):
    selected_item = tree.selection()
    if selected_item:
        item = selected_item[0]
        column = tree.identify_column(event.x)
        if column == '#8':  # delta (ppm) column
            current_value = tree.set(item, column)
            new_value = simpledialog.askstring("Input", "Enter δ (ppm) value:", initialvalue=current_value)
            if new_value is not None:
                try:
                    new_value_float = float(new_value)
                    tree.set(item, column, new_value)
                    i = int(tree.item(item, "values")[0])
                    delta_values[i] = new_value_float
                    plot_cartesian_graph()  # Update Cartesian plot with user input
                except ValueError:
                    messagebox.showerror("Invalid input", "Please enter a valid number for δ (ppm).")

def update_table(polar_data, rotated_coords):
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

# Function to update the Cartesian plot in real-time and add a linear regression line
def plot_cartesian_graph():
    geom_values = []
    delta_values_list = []
    scaled_x_values = []

    for item in tree.get_children():
        geom_value = float(tree.item(item, "values")[5])  # G_i value
        delta_value = tree.item(item, "values")[7]  # delta (ppm) value is in the 8th column
        if delta_value:
            delta_value = float(delta_value)
            geom_values.append(geom_value)
            delta_values_list.append(delta_value)
            scaled_x = (geom_value * 1e4) / (12 * np.pi)
            scaled_x_values.append(scaled_x)

    cartesian_figure.clear()
    ax = cartesian_figure.add_subplot(111)

    if geom_values and delta_values_list:
        # Existing blue scatter plot and regression line
        ax.scatter(geom_values, delta_values_list, color='blue', marker='o', label='G_i vs δ (ppm)')
        slope, intercept, r_value, p_value, std_err = stats.linregress(geom_values, delta_values_list)
        regression_line = [slope * x + intercept for x in geom_values]
        ax.plot(geom_values, regression_line, color='blue', label='Linear Fit (Blue)')

        stats_text = f"Blue Scatter\nSlope: {slope:.2e}\nIntercept: {intercept:.2e}\nR-value: {r_value:.3f}"
        annotation = ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=6,
                             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor=(1, 1, 1, 0.5)))
        DraggableAnnotation(annotation)

        # New red scatter plot and regression line
        ax.scatter(scaled_x_values, delta_values_list, color='red', marker='o', label='Scaled G_i vs δ (ppm)')
        slope_red, intercept_red, r_value_red, p_value_red, std_err_red = stats.linregress(scaled_x_values, delta_values_list)
        regression_line_red = [slope_red * x + intercept_red for x in scaled_x_values]
        ax.plot(scaled_x_values, regression_line_red, color='red', label='Linear Fit (Red)')

        stats_text_red = f"Red Scatter\nSlope: {slope_red:.2e}\nIntercept: {intercept_red:.2e}\nR-value: {r_value_red:.3f}"
        annotation_red = ax.text(0.05, 0.75, stats_text_red, transform=ax.transAxes, fontsize=6,
                                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.1", edgecolor='black', facecolor=(1, 1, 1, 0.5)))
        DraggableAnnotation(annotation_red)

        # Add legend
        ax.legend(fontsize=6)

    ax.set_xlabel('G_i')
    ax.set_ylabel('δ (ppm)')
    ax.set_title('G_i vs δ (ppm)')
    ax.grid(True)
    plt.show()

    cartesian_canvas.draw()

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
    columns = ('Ref', 'Atom', 'X', 'Y', 'Z', 'G_i', 'δ_PCS', 'δ (ppm)')
    
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

    target_atom = simpledialog.askstring("Input", "Enter the center atom:")
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
    update_graph()  # Refresh the graph after loading the file

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
    ax.scatter(theta, r, color='yellow', zorder=10, s=50, edgecolor='black')  # hightlight effect
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
root.title("2D PCS Analyzer v2.7.1")
root.geometry("1500x800")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

left_frame = tk.Frame(main_frame)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

center_frame = tk.Frame(main_frame)
center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

right_frame = tk.Frame(main_frame)
right_frame.pack(side=tk.LEFT, fill=tk.Y)

# PCS Plot
pcs_figure = plt.figure(figsize=(5, 4), dpi=150)
canvas = FigureCanvasTkAgg(pcs_figure, master=left_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Table
table_frame = tk.Frame(center_frame)
table_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

columns = ('Ref', 'Atom', 'X', 'Y', 'Z', 'G_i', 'δ_PCS', 'δ (ppm)')
tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=20)

for col in columns:
    tree.heading(col, text=col)

tree.column('Ref', width=50)
tree.column('Atom', width=50)
tree.column('X', width=50)  
tree.column('Y', width=50)  
tree.column('Z', width=50)  
tree.column('G_i', width=80)
tree.column('δ_PCS', width=70) 
tree.column('δ (ppm)', width=70)

scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

tree.bind("<Double-1>", on_delta_entry_change)
# 테이블 항목 클릭 시 이벤트 연결
tree.bind("<<TreeviewSelect>>", on_table_select)

# Cartesian Plot
cartesian_frame = tk.Frame(center_frame)
cartesian_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

cartesian_figure = plt.Figure(figsize=(5, 4), dpi=100)
cartesian_canvas = FigureCanvasTkAgg(cartesian_figure, master=cartesian_frame)
cartesian_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Control UI
input_frame = tk.Frame(right_frame)
input_frame.pack(fill=tk.Y, padx=10, pady=10)

tensor_label = tk.Label(input_frame, text="Δχ_ax values (*10^-32 m³):")
tensor_label.pack()

# Define and initialize tensor_entry
tensor_entry = tk.Entry(input_frame)
tensor_entry.pack()

pcs_min_label = tk.Label(input_frame, text="Min PCS values range (ppm):")
pcs_min_label.pack()

pcs_min_entry = tk.Entry(input_frame)
pcs_min_entry.pack()

pcs_max_label = tk.Label(input_frame, text="Max PCS values range (ppm):")
pcs_max_label.pack()

pcs_max_entry = tk.Entry(input_frame)
pcs_max_entry.pack()

pcs_interval_label = tk.Label(input_frame, text="PCS values interval (ppm):")
pcs_interval_label.pack()

pcs_interval_entry = tk.Entry(input_frame)
pcs_interval_entry.pack()

button_frame = tk.Frame(input_frame)
button_frame.pack(pady=5)

update_button = tk.Button(button_frame, text="Update", command=update_graph)
update_button.pack(side=tk.LEFT, padx=5)

reset_button = tk.Button(button_frame, text="Reset", command=reset_values)
reset_button.pack(side=tk.LEFT, padx=5)

molar_value_label = tk.Label(input_frame, text="Δχ_mol_ax : N/A m³/mol")
molar_value_label.pack(pady=5)

separator = ttk.Separator(input_frame, orient='horizontal')
separator.pack(fill='x', pady=10)

load_xyz_button = tk.Button(input_frame, text="Load xyz File", command=load_xyz_file)
load_xyz_button.pack()

instruction_label = tk.Label(input_frame, text="The coordinates should align\n the molecule's rotational axis\n with the z-axis for proper analysis.", font=("Helvetica", 8, "italic"))
instruction_label.pack(pady=0)

separator = ttk.Separator(input_frame, orient='horizontal')
separator.pack(fill='x', pady=10)

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

separator = ttk.Separator(input_frame, orient='horizontal')
separator.pack(fill='x', pady=10)

instruction_label = tk.Label(input_frame, text="Atom list")
instruction_label.pack(pady=0)

checklist_frame = tk.Frame(input_frame)
checklist_frame.pack(pady=5, fill=tk.BOTH, expand=True)

instruction_label = tk.Label(input_frame, text="Rotation and atom selection can be\n applied after updating PCS information.", font=("Helvetica", 8, "italic"))
instruction_label.pack(pady=0)

separator = ttk.Separator(input_frame, orient='horizontal')
separator.pack(fill='x', pady=10)

close_button = tk.Button(input_frame, text="Close", command=close_window)
close_button.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=10)

save_button = tk.Button(input_frame, text="Save plot as xlsx", command=save_graph_to_excel)
save_button.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=0)

save_png_button = tk.Button(input_frame, text="Save plot as png", command=save_graph)
save_png_button.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=0)

# Save table as Excel button
save_table_button = tk.Button(input_frame, text="Save table as xlsx", command=save_table_to_excel)
save_table_button.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=5)


polar_data = []
atom_data = []
check_vars = {}

pcs_values = np.arange(-10, 10.5, 0.5)
theta_values = np.linspace(0, 2 * np.pi, 500)
plot_graph(pcs_values, theta_values, -2.0, canvas, pcs_figure)
plot_cartesian_graph()

root.mainloop()
