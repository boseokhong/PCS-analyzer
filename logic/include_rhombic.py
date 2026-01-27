import numpy as np

def geom_factors_ax_rh(coords, metal):
    """
    coords : (N,3) array, already rotated into tensor frame (z = principal axis)
    metal  : (3,) metal center

    Returns:
        r, theta, phi, Gax, Grh
    """
    coords = np.asarray(coords, float)
    metal  = np.asarray(metal, float)

    vecs = coords - metal
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]

    r = np.linalg.norm(vecs, axis=1)
    r_safe = np.where(r == 0.0, np.inf, r)

    cos_theta = np.clip(z / r_safe, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    phi   = np.arctan2(y, x)   # azimuth in tensor frame

    Gax = (3.0 * np.cos(theta)**2 - 1.0) / (r_safe**3)
    Grh = (1.5 * np.sin(theta)**2 * np.cos(2.0 * phi)) / (r_safe**3)

    return r, theta, phi, Gax, Grh


def pcs_ax_only(Gax, dchi_ax):
    """axial-only PCS"""
    return (dchi_ax * Gax * 1e4) / (12.0 * np.pi)


def pcs_ax_rh(Gax, Grh, dchi_ax, dchi_rh):
    """axial + rhombic PCS"""
    return ((dchi_ax * Gax + dchi_rh * Grh) * 1e4) / (12.0 * np.pi)


def build_rh_table_rows(state, filter_atoms_fn):
    """
    Rhombicity 탭 전용 테이블에 넣을 rows 생성.
    반환: list[tuple]  (Treeview values로 바로 넣을 수 있게)
    """
    dv = state.get("delta_values", {})

    # Δχ_ax(tensor) 읽기: state["tensor"]가 아니라 Entry에서 읽어야 함
    tensor = 0.0
    try:
        if "tensor_entry" in state and state["tensor_entry"] is not None:
            s = str(state["tensor_entry"].get()).strip()
            tensor = float(s) if s else 0.0
        else:
            tensor = float(state.get("tensor", 0.0))
    except Exception:
        tensor = 0.0

    dchi_rh = float(state.get("rh_dchi_rh", 0.0))

    polar_data, rotated_coords = filter_atoms_fn(state)

    ids = state.get("current_selected_ids", [])
    if not ids:
        ids = list(range(1, len(rotated_coords) + 1))

    metal = np.array([state["x0"], state["y0"], state["z0"]], float)

    r_arr, theta_arr, phi_arr, Gax_arr, Grh_arr = geom_factors_ax_rh(rotated_coords, metal)

    rows = []
    for i, (atom, _, _) in enumerate(polar_data):
        ref_id = ids[i] if i < len(ids) else (i + 1)

        r_val = float(r_arr[i])
        theta_deg = float(theta_arr[i] * 180.0 / np.pi)
        phi_deg   = float(phi_arr[i]   * 180.0 / np.pi)

        gax = float(Gax_arr[i])
        grh = float(Grh_arr[i])

        d_ax   = float(pcs_ax_only(gax, tensor))
        d_axrh = float(pcs_ax_rh(gax, grh, tensor, dchi_rh))

        dexp = dv.get(ref_id, None)
        if dexp is None:
            dexp_str = ""
            resid_ax = ""
            resid_rh = ""
        else:
            dexp = float(dexp)
            dexp_str = f"{dexp:g}"
            resid_ax = f"{(dexp - d_ax): .2f}"
            resid_rh = f"{(dexp - d_axrh): .2f}"

        # ✅ 컬럼 추가: r, theta, phi
        rows.append((
            ref_id,
            atom,
            f"{r_val:.2f}",
            f"{theta_deg:.2f}",
            f"{phi_deg:.2f}",
            f"{gax:.4e}",
            f"{grh:.4e}",
            f"{d_ax: .2f}",
            f"{d_axrh: .2f}",
            dexp_str,
            resid_ax,
            resid_rh,
        ))
    return rows
