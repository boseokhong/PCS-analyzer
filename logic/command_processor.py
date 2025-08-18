# logic/command_processor.py

import numpy as np
from logic.rotate_align import rotate_about_center, align_xyz

def process_command(state):
    cmd = state['command_entry'].get().strip()
    if not cmd:
        state['messagebox'].showerror("Error", "No command entered."); return
    args = cmd.split()
    def set_sliders(ax, ay):
        state['angle_x_var'].set(ax); state['angle_y_var'].set(ay)
        state['angle_x_entry'].delete(0, state['tk'].END); state['angle_x_entry'].insert(0, f"{float(ax):.1f}")
        state['angle_y_entry'].delete(0, state['tk'].END); state['angle_y_entry'].insert(0, f"{float(ay):.1f}")
    try:
        if args[0]=='rotate' and len(args)==4:
            ax, ay, az = float(args[1]), float(args[2]), float(args[3])
            coords = state['last_rotated_coords'] if state['last_rotated_coords'] is not None else np.array([[x,y,z] for _,x,y,z in state['atom_data']])
            center = (state['x0'], state['y0'], state['z0'])
            rot = rotate_about_center(coords, ax, ay, az, center)
            state['last_rotated_coords'] = rot; set_sliders(ax, ay)
        elif args[0]=='align' and len(args)==7:
            v1 = [float(args[1]), float(args[2]), float(args[3])]
            v2 = [float(args[4]), float(args[5]), float(args[6])]
            coords = state['last_rotated_coords'] if state['last_rotated_coords'] is not None else np.array([[x,y,z] for _,x,y,z in state['atom_data']])
            rot = align_xyz(v1, v2, coords)
            state['last_rotated_coords'] = rot; set_sliders(0,0)
        else:
            state['messagebox'].showerror("Error", f"Unknown/invalid command: {' '.join(args)}"); return
        update_all(state)
    except Exception as e:
        state['messagebox'].showerror("Error", f"An error occurred: {e}")

def update_all(state):
    # Recompute polar data and redraw everything
    state['update_graph']()
    state['plot_cartesian'](state)
    polar_data, rotated_coords = state['filter_atoms'](state)
    state['update_table'](state, polar_data, rotated_coords, float(state['tensor_entry'].get() or 1.0), state['delta_values'])
