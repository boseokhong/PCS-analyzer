"""
Diagnostic tools for axial-only PCS approximation
-------------------------------------------------

Axial PCS model (Δχ_rh = 0):
    δ_i = (Δχ_ax / 12π) · G_ax,i
    G_ax,i = (3 cos²θ_i − 1) / r_i³

After axial-only linear regression, residuals are defined as:
    ε_i = δ_exp,i − (k · G_ax,i + b)

If a finite rhombic component is present but omitted in the model,
the residuals follow:
    ε_i ≈ (Δχ_rh / 12π) · G_rh,i
    G_rh,i = (3/2) sin²θ_i cos(2φ_i) / r_i³

Diagnostic strategy:
  • ε vs φ   → cos(2φ) signature indicates rhombicity
  • ε vs G_rh → linear correlation confirms rhombic PCS origin

This module is strictly diagnostic: no full tensor fitting is performed.

In short:
  Axial fit → residuals → angular/geometric correlation → physical diagnosis.
"""

# logic/diagnostic.py

import numpy as np
import matplotlib.pyplot as plt
from logic.fitting import geom_factors_ax_rh
from logic.rotate_align import rotate_euler

def _get_obs_pairs(state, proton_ids=None):
    """
    Returns:
        obs_pairs: [(rid, delta_exp), ...]
        proton_ids_used: [rid, ...]
    """
    delta_values = state.get("delta_values", {}) or {}

    if proton_ids is None:
        # if user selected in fitting list, use that
        if "fit_proton_list" in state:
            sel = state["fit_proton_list"].curselection()
            if sel:
                proton_ids = [int(state["fit_proton_list"].get(i)) for i in sel]

    if proton_ids is None:
        # fallback: any keys in delta_values
        proton_ids = sorted(delta_values.keys())

    obs_pairs = [(rid, float(delta_values[rid])) for rid in proton_ids if rid in delta_values]
    return obs_pairs, proton_ids

def _rotated_coords_for_obs(state, obs_pairs):
    atom_data = state.get("atom_data", [])
    ids = state.get("atom_ids", [])
    if not atom_data or not ids:
        raise RuntimeError("No atom data loaded.")

    id2idx = {rid: i for i, rid in enumerate(ids)}
    metal = np.array([state["x0"], state["y0"], state["z0"]], float)
    abs_coords = np.array([[x, y, z] for (_, x, y, z) in atom_data], float)

    # -------------------------------------------------
    # (A) apply fit_override FIRST (exactly like filter_atoms)
    # -------------------------------------------------
    fo = state.get("fit_override")
    if fo:
        mode = (fo.get("mode") or "").lower()

        if mode == "theta_alpha_multi":
            donor_ids = fo.get("donor_ids") or []
            if donor_ids:
                from logic.fitting import _angles_to_rotation_multi  # local import OK
                donor_pts = [abs_coords[id2idx[rid]] for rid in donor_ids]

                abs_coords = _angles_to_rotation_multi(
                    points=abs_coords,
                    metal=metal,
                    donor_points=donor_pts,
                    theta_deg=float(fo.get("theta", 0.0)),
                    alpha_deg=float(fo.get("alpha", 0.0)),
                    axis_mode=fo.get("axis_mode", "bisector"),
                )

        elif mode == "euler_global":
            ax0 = float(fo.get("ax", 0.0))
            ay0 = float(fo.get("ay", 0.0))
            az0 = float(fo.get("az", 0.0))

            coords0 = abs_coords - metal
            rot0 = rotate_euler(coords0, ax0, ay0, az0)
            abs_coords = rot0 + metal

    # -------------------------------------------------
    # (B) apply UI sliders SECOND (x,y,z)
    # -------------------------------------------------
    ax = float(state["angle_x_var"].get()) if "angle_x_var" in state else 0.0
    ay = float(state["angle_y_var"].get()) if "angle_y_var" in state else 0.0
    az = float(state["angle_z_var"].get()) if "angle_z_var" in state else 0.0

    coords0 = abs_coords - metal
    rot0 = rotate_euler(coords0, ax, ay, az)
    rot_all = rot0 + metal

    obs_ids = [rid for (rid, _) in obs_pairs]
    pts_obs = np.array([rot_all[id2idx[rid]] for rid in obs_ids], float)
    return pts_obs, metal, obs_ids

def axial_fit_and_residuals(state, proton_ids=None, fit_intercept=True):
    """
    Diagnostic core:
      1) compute Gax, Grh, phi for observed nuclei (in current tensor frame)
      2) do axial-only linear fit: delta_exp ~ k*Gax (+ b)
      3) residuals = delta_exp - (k*Gax + b)
      4) provide also residual vs Grh (or vs cos2phi) diagnostics

    Returns dict with arrays + scalar summary.
    """
    obs_pairs, proton_ids_used = _get_obs_pairs(state, proton_ids=proton_ids)
    if len(obs_pairs) < 3:
        raise RuntimeError("Need ≥3 assigned δ_Exp points for diagnostics (select protons and set δ_Exp).")

    pts_obs, metal, obs_ids = _rotated_coords_for_obs(state, obs_pairs)
    delta_exp = np.array([v for (_, v) in obs_pairs], float)

    # geometry factors in current frame
    r, theta, phi, Gax, Grh = geom_factors_ax_rh(pts_obs, metal)

    # axial-only regression: delta = k*Gax (+ b)
    x = np.asarray(Gax, float)
    y = np.asarray(delta_exp, float)

    if fit_intercept:
        A = np.column_stack([x, np.ones_like(x)])
        k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        # forced through origin
        k = float(np.dot(x, y) / np.dot(x, x)) if float(np.dot(x, x)) != 0.0 else 0.0
        b = 0.0

    delta_pred = k * x + b
    resid = y - delta_pred

    rmsd = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else np.nan

    # convert slope -> dchi_ax (matches your PCS convention)
    # In your code: dpcs = (dchi_ax*Gax*1e4)/(12π)  with Gax already 1/r^3*(3cos^2θ-1) or 1.5... etc.
    # Here y ~ k*Gax => k = dchi_ax*1e4/(12π)  => dchi_ax = k*(12π)/1e4
    dchi_ax_est = float(k * (12.0 * np.pi) / 1e4)

    # wrap phi to [0, 2π) for plotting
    phi_wrapped = (phi + 2.0 * np.pi) % (2.0 * np.pi)

    return dict(
        obs_ids=obs_ids,
        delta_exp=y,
        delta_pred=delta_pred,
        resid=resid,
        rmsd=rmsd,
        k_slope=float(k),
        intercept=float(b),
        dchi_ax_est=dchi_ax_est,
        r=r,
        theta=theta,
        phi=phi_wrapped,
        Gax=Gax,
        Grh=Grh,
        n=len(obs_ids),
        used_proton_ids=proton_ids_used,
        fit_intercept=bool(fit_intercept),
    )

'''
def plot_axial_linearity(state, result, fig=None):
    if fig is None:
        fig = state.get("diag_fig_linearity")
    if fig is None:
        raise RuntimeError("diag_fig_linearity not found in state.")

    fig.clf()
    ax = fig.add_subplot(1, 1, 1)

    x = np.asarray(result["Gax"], float)
    y = np.asarray(result["delta_exp"], float)
    yhat = np.asarray(result["delta_pred"], float)

    ax.scatter(x, y, label="δ_exp")
    ax.plot(x, yhat, label="axial fit")

    ax.set_xlabel("G_ax")
    ax.set_ylabel("δ_exp (ppm)")
    ax.set_title("Axial linearity: δ_exp vs G_ax")
    ax.axhline(0, linewidth=1)
    ax.legend()

    fig.tight_layout()

    cv = state.get("diag_canvas_linearity")
    if cv is not None:
        cv.draw_idle()
'''

def plot_residual_vs_phi_and_grh(state, result, fig=None):
    if fig is None:
        fig = state.get("diag_fig_resphi")
    if fig is None:
        raise RuntimeError("diag_fig_resphi not found in state.")

    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    phi_deg = np.degrees(result["phi"])
    resid = result["resid"]
    Grh = result["Grh"]

    ax1.scatter(phi_deg, resid)
    ax1.set_xlabel("φ (deg)")
    ax1.set_ylabel("Residual (ppm)")
    ax1.set_title("Residual vs φ")
    ax1.set_xlim(0, 360)
    ax1.axhline(0, linewidth=1)

    ax2.scatter(Grh, resid)
    ax2.set_xlabel("G_rh")
    ax2.set_ylabel("Residual (ppm)")
    ax2.set_title("Residual vs G_rh")
    ax2.axhline(0, linewidth=1)

    # optional guide line
    if len(Grh) >= 2:
        X = np.column_stack([Grh, np.ones_like(Grh)])
        c1, c0 = np.linalg.lstsq(X, resid, rcond=None)[0]
        xs = np.linspace(np.min(Grh), np.max(Grh), 50)
        ax2.plot(xs, c1 * xs + c0)

    fig.tight_layout()

    cv = state.get("diag_canvas_resphi")
    if cv is not None:
        cv.draw_idle()

def update_diagnostic_panel(state):
    # UI checkbox meaning:
    #   checked  -> force b=0  -> do NOT fit intercept
    #   unchecked-> fit b      -> fit intercept
    force_b0 = False
    if "diag_fit_intercept_var" in state:
        force_b0 = bool(state["diag_fit_intercept_var"].get())

    fit_intercept = not force_b0

    res = axial_fit_and_residuals(state, proton_ids=None, fit_intercept=fit_intercept)
    state["last_diag_result"] = res

    plot_residual_vs_phi_and_grh(state, res, fig=state.get("diag_fig_resphi"))

    box = state.get("diag_result_box")
    if box is not None:
        box.delete("1.0", "end")
        box.insert("end", "[Diagnostic] Axial-only fit → residual analysis\n")
        box.insert("end", f"N = {res['n']}, RMSD(resid) = {res['rmsd']:.3f} ppm\n")
        box.insert(
            "end",
            ("Model: δ = k·Gax + b\n" if res["fit_intercept"]
             else "Model: δ = k·Gax  (b = 0 forced)\n")
        )
        box.insert("end", f"k = {res['k_slope']:.6g} ; b = {res['intercept']:.6g}\n")
        box.insert("end", f"Δχ_ax (from slope) ≈ {res['dchi_ax_est']:.3e} (E-32 m³)\n\n")
        box.insert("end", "Ref\tδ_exp\tδ_pred\tresid\tphi(deg)\tG_rh\n")
        for rid, de, dp, rr, ph, grh in zip(
                res["obs_ids"], res["delta_exp"], res["delta_pred"],
                res["resid"], np.degrees(res["phi"]), res["Grh"]
        ):
            box.insert("end", f"{rid}\t{de:.3f}\t{dp:.3f}\t{rr:+.3f}\t{ph:6.1f}\t{grh:+.3e}\n")

    return res
