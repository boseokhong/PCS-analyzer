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
  • ε vs φ    → cos(2φ) signature indicates rhombicity
  • ε vs G_rh → linear correlation confirms rhombic PCS origin

Extra (helpful for interpretation):
  • Fit ε vs G_rh → slope, intercept, Pearson r, R²
  • Convert slope → Δχ_rh estimate (same convention as axial slope conversion)
  • Show top |ε| outliers (often contact / misassignment / frame issues)

This module is strictly diagnostic: no full tensor fitting is performed.
"""

# logic/diagnostic.py

import numpy as np
import matplotlib.pyplot as plt

from logic.fitting import geom_factors_ax_rh
from logic.rotate_align import rotate_euler


# -----------------------
# small math helpers
# -----------------------

def _pearsonr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2:
        return np.nan
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    den = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    if den == 0.0:
        return np.nan
    return float(np.sum(x0 * y0) / den)

def _linreg(x, y, fit_intercept=True):
    """
    Simple linear regression y ≈ m*x (+ b).
    Returns: m, b, r2, yhat
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.full_like(y, np.nan)

    if fit_intercept:
        A = np.column_stack([x, np.ones_like(x)])
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        yhat = m * x + b
    else:
        xx = float(np.dot(x, x))
        m = float(np.dot(x, y) / xx) if xx != 0.0 else np.nan
        b = 0.0
        yhat = m * x

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot != 0.0 else np.nan

    return float(m), float(b), float(r2), yhat

# -----------------------
# core data extraction
# -----------------------

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
    """
    Build rotated coords for the Ref IDs in obs_pairs, consistent with:
      (A) fit_override (Mode A or Mode B)
      (B) UI sliders (x,y,z)
    """
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

# -----------------------
# diagnostic computations
# -----------------------

def axial_fit_and_residuals(state, proton_ids=None, fit_intercept=True):
    """
    Diagnostic core:
      1) compute Gax, Grh, phi for observed nuclei (in current tensor frame)
      2) do axial-only linear fit: delta_exp ~ k*Gax (+ b)
      3) residuals = delta_exp - (k*Gax + b)
      4) compute correlation/regression of residuals vs Grh
      5) estimate Δχ_rh from slope(resid vs Grh)
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
        xx = float(np.dot(x, x))
        k = float(np.dot(x, y) / xx) if xx != 0.0 else 0.0
        b = 0.0

    delta_pred = k * x + b
    resid = y - delta_pred
    rmsd = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else np.nan

    # convert slope -> dchi_ax (matches your PCS convention)
    # If δ = (Δχ_ax * 1e4 / 12π) * Gax, then slope k = Δχ_ax*1e4/(12π)
    dchi_ax_est = float(k * (12.0 * np.pi) / 1e4)

    # wrap phi to [0, 2π) for plotting
    phi_wrapped = (phi + 2.0 * np.pi) % (2.0 * np.pi)

    # --- residual vs Grh regression (interpretation aid) ---
    grh = np.asarray(Grh, float)
    rr = np.asarray(resid, float)

    m_rg, c_rg, r2_rg, rr_hat = _linreg(grh, rr, fit_intercept=True)
    r_rg = _pearsonr(grh, rr)

    # If resid ≈ (Δχ_rh*1e4/(12π)) * Grh  (+ const), then
    # Δχ_rh ≈ slope * (12π)/1e4
    dchi_rh_est = float(m_rg * (12.0 * np.pi) / 1e4) if np.isfinite(m_rg) else np.nan

    # outliers (top |resid|)
    order = np.argsort(np.abs(rr))[::-1]
    topN = int(min(5, len(order)))
    outliers = []
    for j in order[:topN]:
        outliers.append((
            int(obs_ids[j]),
            float(rr[j]),
            float(grh[j]),
            float(np.degrees(phi_wrapped[j])),
            float(y[j]),
            float(delta_pred[j]),
        ))

    return dict(
        # ids & primary data
        obs_ids=obs_ids,
        delta_exp=y,
        delta_pred=delta_pred,
        resid=resid,
        rmsd=rmsd,

        # axial fit params
        k_slope=float(k),
        intercept=float(b),
        dchi_ax_est=dchi_ax_est,

        # geometry
        r=r,
        theta=theta,
        phi=phi_wrapped,
        Gax=Gax,
        Grh=Grh,

        # bookkeeping
        n=len(obs_ids),
        used_proton_ids=proton_ids_used,
        fit_intercept=bool(fit_intercept),

        # interpretation aids (resid vs Grh)
        resid_grh_slope=float(m_rg),
        resid_grh_intercept=float(c_rg),
        resid_grh_r=float(r_rg),
        resid_grh_r2=float(r2_rg),
        dchi_rh_est=float(dchi_rh_est),
        outliers=outliers,
    )

# -----------------------
# plotting
# -----------------------

def plot_residual_vs_phi_and_grh(state, result, fig=None):
    """
    Draw 2 plots on one figure:
      (left)  residual vs phi (deg)
      (right) residual vs G_rh, with optional regression line and stats
    """
    if fig is None:
        fig = state.get("diag_fig_resphi")
    if fig is None:
        raise RuntimeError("diag_fig_resphi not found in state.")

    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    phi_deg = np.degrees(result["phi"])
    resid = np.asarray(result["resid"], float)
    Grh = np.asarray(result["Grh"], float)

    # (1) resid vs phi
    ax1.scatter(phi_deg, resid)
    ax1.set_xlabel("φ (deg)")
    ax1.set_ylabel("Residual (ppm)")
    ax1.set_title("Residual vs φ")
    ax1.set_xlim(0, 360)
    ax1.axhline(0, linewidth=1)

    # (2) resid vs Grh
    ax2.scatter(Grh, resid)
    ax2.set_xlabel("G_rh")
    ax2.set_ylabel("Residual (ppm)")
    ax2.set_title("Residual vs G_rh")
    ax2.axhline(0, linewidth=1)

    # regression guide line
    m = result.get("resid_grh_slope", np.nan)
    c = result.get("resid_grh_intercept", np.nan)
    if len(Grh) >= 2 and np.isfinite(m) and np.isfinite(c):
        xs = np.linspace(np.min(Grh), np.max(Grh), 100)
        ax2.plot(xs, m * xs + c)

        # show tiny stats box (no fixed colors)
        r = result.get("resid_grh_r", np.nan)
        r2 = result.get("resid_grh_r2", np.nan)
        txt = f"r={r:+.3f}\nR²={r2:.3f}\nΔχ_rh≈{result.get('dchi_rh_est', np.nan):.2e}"
        ax2.text(
            0.02, 0.98, txt,
            transform=ax2.transAxes,
            va="top", ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    fig.tight_layout()

    cv = state.get("diag_canvas_resphi")
    if cv is not None:
        cv.draw_idle()

# -----------------------
# UI entry point
# -----------------------

def update_diagnostic_panel(state):
    """
    One-shot entry point for UI.
    Checkbox meaning (components.py):
      - state["diag_force_b0_var"] checked  -> force b=0 -> do NOT fit intercept
      - unchecked -> fit intercept
    """
    # robust: support older variable name as fallback
    if "diag_force_b0_var" in state:
        force_b0 = bool(state["diag_force_b0_var"].get())
    elif "diag_fit_intercept_var" in state:
        # older naming (checked means force b0 in your recent convention)
        force_b0 = bool(state["diag_fit_intercept_var"].get())
    else:
        force_b0 = False

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
        box.insert("end", f"Δχ_ax (from slope) ≈ {res['dchi_ax_est']:.3e} (E-32 m³)\n")

        # interpretation aids
        box.insert("end", "\n[Rhombic diagnostic: resid vs G_rh]\n")
        box.insert("end", f"Pearson r = {res['resid_grh_r']:+.3f} ; R² = {res['resid_grh_r2']:.3f}\n")
        box.insert(
            "end",
            f"resid ≈ m·G_rh + c   with  m={res['resid_grh_slope']:+.6g}, c={res['resid_grh_intercept']:+.6g}\n"
        )
        box.insert("end", f"Δχ_rh (from slope) ≈ {res['dchi_rh_est']:.3e} (E-32 m³)\n")

        r_abs = abs(res["resid_grh_r"]) if np.isfinite(res["resid_grh_r"]) else 0.0
        if r_abs > 0.7:
            box.insert("end", "→ Strong correlation: residuals likely dominated by rhombic PCS (finite Δχ_rh).\n")
        elif r_abs > 0.4:
            box.insert("end", "→ Moderate correlation: rhombicity possible; check outliers/contact/axis definition.\n")
        else:
            box.insert("end", "→ Weak correlation: residuals likely not mainly rhombic PCS (noise/contact/frame mismatch).\n")

        # table
        box.insert("end", "\nRef\tδ_exp\tδ_pred\tresid\tphi(deg)\tG_rh\n")
        for rid, de, dp, rr, ph, grh in zip(
            res["obs_ids"],
            res["delta_exp"],
            res["delta_pred"],
            res["resid"],
            np.degrees(res["phi"]),
            res["Grh"],
        ):
            box.insert("end", f"{rid}\t{de:.3f}\t{dp:.3f}\t{rr:+.3f}\t{ph:6.1f}\t{grh:+.3e}\n")

        # outliers
        box.insert("end", "\n[Top |resid| outliers]\n")
        box.insert("end", "Ref\tresid(ppm)\tG_rh\tphi(deg)\tδ_exp\tδ_pred\n")
        for rid, rr, grh, ph, de, dp in res.get("outliers", []):
            box.insert("end", f"{rid}\t{rr:+.3f}\t{grh:+.3e}\t{ph:6.1f}\t{de:+.3f}\t{dp:+.3f}\n")

    return res
