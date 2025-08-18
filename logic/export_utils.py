# logic/export_utils.py

import numpy as np
import pandas as pd

def _safe_r(pcs_value, theta, tensor):
    denom = 1e4 * tensor * (3 * (np.cos(theta))**2 - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = denom / (12 * np.pi * pcs_value)
    frac = np.where(frac > 0, frac, np.nan)  # 양수만 유효
    r = np.cbrt(frac)
    r[~np.isfinite(r)] = np.nan
    return r

def build_pcs_frames(pcs_values, theta_values, tensor, polar_data):
    """공통 계산 → (pcs_df, polar_df) 반환. 엑셀/CSV가 이걸 재사용."""
    theta_values = np.asarray(theta_values, float)

    cols = {'Theta': theta_values}
    for i, pcs_value in enumerate(pcs_values, start=1):
        r_vals = _safe_r(pcs_value, theta_values, tensor)
        x_cart = r_vals * np.sin(theta_values)
        y_cart = r_vals * np.cos(theta_values)
        cols[f'PCS (ppm) {i}'] = np.full(theta_values.shape, float(pcs_value), dtype=float)
        cols[f'R {i}'] = r_vals
        cols[f'X Cartesian {i}'] = x_cart
        cols[f'Y Cartesian {i}'] = y_cart
    pcs_df = pd.DataFrame(cols)

    if polar_data:
        atoms, r_list, th_list = zip(*polar_data)
        r_arr = np.asarray(r_list, float)
        th_arr = np.asarray(th_list, float)
    else:
        atoms, r_arr, th_arr = [], np.array([], float), np.array([], float)

    with np.errstate(divide='ignore', invalid='ignore'):
        Gi = (3 * np.cos(th_arr)**2 - 1) / np.where(r_arr == 0, np.nan, r_arr**3)

    polar_df = pd.DataFrame({
        'Atom': atoms,
        'R': r_arr,
        'Theta': th_arr,
        'Geom Param': Gi
    })
    return pcs_df, polar_df

def save_to_excel(pcs_values, theta_values, tensor, file_name, polar_data):
    pcs_df, polar_df = build_pcs_frames(pcs_values, theta_values, tensor, polar_data)
    with pd.ExcelWriter(file_name) as writer:
        pcs_df.to_excel(writer, sheet_name='PCS Data', index=False)
        polar_df.to_excel(writer, sheet_name='Atom Coordinates', index=False)

def save_to_csv(pcs_values, theta_values, tensor, base_path, polar_data):
    """
    base_path: 사용자가 고른 csv 경로(예: C:/out.csv).
    실제 저장은 out_pcs.csv, out_atoms.csv 두 파일로 만듭니다.
    반환: (pcs_path, atoms_path)
    """
    import os
    pcs_df, polar_df = build_pcs_frames(pcs_values, theta_values, tensor, polar_data)
    base, _ = os.path.splitext(base_path)
    pcs_path = base + "_pcs.csv"
    atoms_path = base + "_atoms.csv"
    pcs_df.to_csv(pcs_path, index=False)
    polar_df.to_csv(atoms_path, index=False)
    return pcs_path, atoms_path

def export_table_to_excel(tree, file_path):
    cols = ('Ref', 'Atom', 'X', 'Y', 'Z', 'G_i', 'δ_PCS', 'δ_Exp')
    data = [tree.item(i)["values"] for i in tree.get_children()]
    pd.DataFrame(data, columns=cols).to_excel(file_path, index=False)

def export_table_to_csv(tree, file_path):
    cols = ('Ref', 'Atom', 'X', 'Y', 'Z', 'G_i', 'δ_PCS', 'δ_Exp')
    data = [tree.item(i)["values"] for i in tree.get_children()]
    pd.DataFrame(data, columns=cols).to_csv(file_path, index=False)
