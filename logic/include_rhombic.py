# logic/include_rhombic.py
# PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE3

import numpy as np
from logic.symmetry_geometry import effective_geometry_factors

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
    dv = state.get("delta_exp_values", {})

    # NEW: pseudo label overrides (Ref-ID -> "MeH@Cxx" etc.)
    label_overrides = state.get("ref_label_overrides", {}) or {}

    # PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE4
    torsion_var = state.get("torsion_avg_enabled_var")
    torsion_enabled = bool(torsion_var.get()) if torsion_var is not None else False
    torsion_avg_ref_ids = set()
    if torsion_enabled:
        for group in (state.get("torsion_avg_groups", []) or []):
            if bool(group.get("enabled", True)):
                torsion_avg_ref_ids.update(int(r) for r in group.get("rotating_atoms", ()))
    pseudo_ref_ids = set(state.get("symavg_pseudo_ref_ids", set()) or set())

    # Δχ_ax(tensor) Entry에서
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

    metal = np.zeros(3, dtype=float)

    r_arr, theta_arr, phi_arr, Gax_arr, Grh_arr = effective_geometry_factors(
        ids,
        rotated_coords,
        metal,
        pseudo_members=state.get("symavg_members_by_pseudo_id", {}) or {},
        raw_coords_by_id=state.get("last_rotated_raw_by_id", {}) or {},        torsion_groups=(state.get("torsion_avg_groups", []) or [])
        if bool(getattr(state.get("torsion_avg_enabled_var"), "get", lambda: False)()) else [],
    )

    rows = []
    for i, (atom, _, _) in enumerate(polar_data):
        ref_id = ids[i] if i < len(ids) else (i + 1)

        atom_disp = label_overrides.get(ref_id, atom)
        if ref_id in torsion_avg_ref_ids and ref_id not in pseudo_ref_ids:
            atom_disp = f"{atom_disp} ⟨avg⟩"

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

        rows.append((
            ref_id,
            atom_disp,
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