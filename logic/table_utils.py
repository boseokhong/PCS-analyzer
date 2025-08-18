# logic/table_utils.py

import numpy as np
from logic.chem_constants import AVOGADRO_CONSTANT

def update_molar_value(state, tensor):
    val = tensor * AVOGADRO_CONSTANT * 1e-32
    state['molar_value_label'].config(text=f"Δχ_mol_ax: {val:.2e} m³/mol")

def update_table(state, polar_data, rotated_coords, tensor, delta_values):
    tree = state['tree']
    # 모든 행 비우기
    for r in tree.get_children():
        tree.delete(r)

    # 현재 표시 중인 행의 Ref(ID) 목록
    ids = state.get('current_selected_ids', [])
    state['row_by_id'] = {}  # 클릭/하이라이트 용 역방향 매핑

    for i, ((atom, r, theta), (dx, dy, dz)) in enumerate(zip(polar_data, rotated_coords)):
        ref_id = ids[i] if i < len(ids) else (i + 1)  # 안전장치
        geom_param = (3 * (np.cos(theta))**2 - 1) / (r**3) if r != 0 else 0.0
        geom_value = geom_param
        delta_pcs = (tensor * (geom_value * 1e4)) / (12 * np.pi)
        # Ref ID 로 읽고/쓴다
        delta_exp_str = ""
        if ref_id in delta_values:
            v = delta_values[ref_id]
            delta_exp_str = f"{v:g}"

        item = tree.insert("", state['tk'].END, values=(
            ref_id, atom, f"{dx:.3f}", f"{dy:.3f}", f"{dz:.3f}",
            f"{geom_param:.5f}", f"{delta_pcs: .2f}", delta_exp_str
        ))
        state['row_by_id'][ref_id] = item

def on_delta_entry_change(state, event, delta_values, plot_cartesian_graph_fn):
    tree = state['tree']
    sel = tree.selection()
    if not sel: return
    item = sel[0]
    col = tree.identify_column(event.x)
    if col != '#8':  # δ_Exp 컬럼
        return

    cur = tree.set(item, col)
    prompt = "Enter experimental chemical shift value (ppm):\n(leave blank to clear)"
    new = state['simpledialog'].askstring("Input", prompt, initialvalue=cur)
    if new is None:
        return

    # Ref(ID)
    try:
        ref_id = int(tree.item(item, "values")[0])
    except Exception:
        return

    new_str = new.strip()
    if new_str == "":
        tree.set(item, col, "")
        delta_values.pop(ref_id, None)
        plot_cartesian_graph_fn(state)
        return

    try:
        val = float(new_str)
        tree.set(item, col, new_str)
        delta_values[ref_id] = val
        plot_cartesian_graph_fn(state)
    except ValueError:
        state['messagebox'].showerror(
            "Invalid input",
            "Please enter a valid number for δ_Exp or leave empty to clear."
        )

def calculate_tensor_components_ui(chi_mol_entry, molar_value_label, tensor_xx_label, tensor_yy_label, tensor_zz_label, messagebox):
    """
    chi_mol_entry: 실험 χ_mol (m^3/mol)
    molar_value_label: 'Δχ_mol_ax : ... m³/mol' 형식의 라벨
    tensor_*_label: 결과 출력 라벨들
    """
    try:
        chi_mol = float(chi_mol_entry.get())
        # 라벨에서 Δχ_mol_ax 수치만 파싱
        txt = molar_value_label['text']  # 예: 'Δχ_mol_ax : 1.23e-28 m³/mol'
        delta_chi_ax = float(txt.split(':')[1].strip().split()[0])

        chi_perp = chi_mol - delta_chi_ax/3
        chi_parallel = (2/3)*delta_chi_ax + chi_mol

        tensor_xx_label.config(text=f"χ_xx: {chi_perp:.2e} m³/mol")
        tensor_yy_label.config(text=f"χ_yy: {chi_perp:.2e} m³/mol")
        tensor_zz_label.config(text=f"χ_zz: {chi_parallel:.2e} m³/mol")
    except Exception:
        messagebox.showerror("Input Error", "Please enter valid numerical values for χ_mol.")