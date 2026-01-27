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
    pcs_df.to_csv(pcs_path, index=False, encoding="utf-8-sig")
    polar_df.to_csv(atoms_path, index=False, encoding="utf-8-sig")
    return pcs_path, atoms_path

def export_table_to_excel(tree, file_path):
    cols = ('Ref', 'Atom', 'X', 'Y', 'Z', 'G_i', 'δ_PCS', 'δ_Exp')
    data = [tree.item(i)["values"] for i in tree.get_children()]
    pd.DataFrame(data, columns=cols).to_excel(file_path, index=False)

def export_table_to_csv(tree, file_path):
    cols = ('Ref', 'Atom', 'X', 'Y', 'Z', 'G_i', 'δ_PCS', 'δ_Exp')
    data = [tree.item(i)["values"] for i in tree.get_children()]
    pd.DataFrame(data, columns=cols).to_csv(file_path, index=False, encoding="utf-8-sig")

def _tree_to_df(tree, cols):
    """Treeview -> DataFrame (values 그대로)."""
    data = [tree.item(i)["values"] for i in tree.get_children()]
    return pd.DataFrame(data, columns=cols)

# Rhombicity 탭 테이블 컬럼 (Important! components.py의 cols names)
_RH_COLS = ("Ref", "Atom", "r", "theta(deg)", "phi(deg)",
            "Gi_ax", "Gi_rh", "δ_PCS(ax)", "δ_PCS(ax+rh)", "δ_Exp", "res(ax)", "res(ax+rh)")

def export_tables_to_excel(main_tree, file_path, rh_tree=None):
    """
    Save table 버튼용:
    - 메인 테이블은 항상 저장 (sheet: 'Table')
    - rh_tree가 있고 데이터가 있으면 Rhombicity 시트를 추가 저장
    """
    main_cols = ('Ref', 'Atom', 'X', 'Y', 'Z', 'G_i', 'δ_PCS', 'δ_Exp')
    main_df = _tree_to_df(main_tree, main_cols)

    with pd.ExcelWriter(file_path) as writer:
        main_df.to_excel(writer, sheet_name='Table', index=False)

        if rh_tree is not None and len(rh_tree.get_children()) > 0:
            rh_df = _tree_to_df(rh_tree, _RH_COLS)
            rh_df.to_excel(writer, sheet_name='Rhombicity', index=False)

def export_tables_to_csv(main_tree, file_path, rh_tree=None):
    """
    Save table 버튼용 (CSV):
    - <base>_table.csv 는 항상 저장
    - rh_tree가 있고 데이터가 있으면 <base>_rhombicity.csv 도 추가 저장
    반환: (table_path, rh_path_or_None)
    """
    import os
    base, _ = os.path.splitext(file_path)

    main_cols = ('Ref', 'Atom', 'X', 'Y', 'Z', 'G_i', 'δ_PCS', 'δ_Exp')
    main_df = _tree_to_df(main_tree, main_cols)
    table_path = base + "_table.csv"
    main_df.to_csv(table_path, index=False, encoding="utf-8-sig")

    rh_path = None
    if rh_tree is not None and len(rh_tree.get_children()) > 0:
        rh_df = _tree_to_df(rh_tree, _RH_COLS)
        rh_path = base + "_rhombicity.csv"
        rh_df.to_csv(rh_path, index=False, encoding="utf-8-sig")

    return table_path, rh_path
