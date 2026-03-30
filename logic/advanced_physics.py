# logic/advanced_physics.py

"""
Advanced physics extensions for PCS Analyzer:

1. Multi-conformer ensemble fitting
   - weighted PCS averaging across multiple conformers

2. Joint PCS + RDC fitting
   - RDC prediction physically coupled to the magnetic susceptibility tensor

3. PRE fitting (Solomon-Bloembergen)
   - Gamma1 / Gamma2 calculation
   - correlation time (tau_c) fitting

4. Isosurface data generation
   - 2D PCS slices
   - 3D spherical shell PCS maps

5. Multi-lanthanide fitting
   - simultaneous fitting across multiple lanthanide datasets

Conventions (consistent with logic/fitting.py):
- Δχ is handled in units of 10^-32 m^3
- r is in Å
- δPCS = (Δχax * Gax + Δχrh * Grh) * 1e4 / (12π)   [ppm]
- Gax  = (3cos²θ − 1) / r³
- Grh  = 1.5 * sin²θ * cos(2φ) / r³
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize


# ============================================================================
# Physical constants
# ============================================================================

MU0 = 4 * np.pi * 1e-7
HBAR = 1.054571817e-34
KB = 1.380649e-23
NA = 6.02214076e23
MU_B = 9.2740100783e-24

GAMMA_H = 2.67522187e8
GAMMA_N = -2.71262e7
GAMMA_C = 6.72828e7

GYRO = {
    "H": GAMMA_H,
    "1H": GAMMA_H,
    "N": GAMMA_N,
    "15N": GAMMA_N,
    "C": GAMMA_C,
    "13C": GAMMA_C,
}

# Lanthanide database
# dchi_ax0 values are initial guesses in units of 10^-32 m^3
LANTHANIDE_DB = {
    "Pr": dict(J=4, g_J=4 / 5, mu_eff=3.62, dchi_ax0=-4.0),
    "Nd": dict(J=9 / 2, g_J=8 / 11, mu_eff=3.68, dchi_ax0=-3.0),
    "Sm": dict(J=5 / 2, g_J=2 / 7, mu_eff=1.55, dchi_ax0=-1.0),
    "Eu": dict(J=0, g_J=0, mu_eff=3.40, dchi_ax0=1.0),
    "Tb": dict(J=6, g_J=3 / 2, mu_eff=9.72, dchi_ax0=-25.0),
    "Dy": dict(J=15 / 2, g_J=4 / 3, mu_eff=10.65, dchi_ax0=-54.0),
    "Ho": dict(J=8, g_J=5 / 4, mu_eff=10.60, dchi_ax0=22.0),
    "Er": dict(J=15 / 2, g_J=6 / 5, mu_eff=9.58, dchi_ax0=7.0),
    "Tm": dict(J=6, g_J=7 / 6, mu_eff=7.57, dchi_ax0=-16.0),
    "Yb": dict(J=7 / 2, g_J=8 / 7, mu_eff=4.54, dchi_ax0=3.0),
}


# ============================================================================
# Geometry helpers
# ============================================================================

def geom_factors(coords: np.ndarray, metal: np.ndarray):
    """
    Compute geometry factors in the tensor frame.

    Parameters
    ----------
    coords : (N, 3) ndarray
        Atomic coordinates in Å.
    metal : (3,) ndarray
        Metal position in Å.

    Returns
    -------
    r : ndarray
        Distance from metal in Å.
    theta : ndarray
        Polar angle in radians.
    phi : ndarray
        Azimuthal angle in radians.
    Gax : ndarray
        Axial geometry factor in Å^-3.
    Grh : ndarray
        Rhombic geometry factor in Å^-3.
    """
    vecs = np.asarray(coords, float) - np.asarray(metal, float)
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]

    r = np.linalg.norm(vecs, axis=1)
    r_safe = np.where(r == 0.0, np.inf, r)

    theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    phi = np.arctan2(y, x)

    Gax = (3.0 * np.cos(theta) ** 2 - 1.0) / (r_safe ** 3)
    Grh = 1.5 * np.sin(theta) ** 2 * np.cos(2.0 * phi) / (r_safe ** 3)

    return r, theta, phi, Gax, Grh


def pcs_from_G(Gax, Grh, dchi_ax, dchi_rh=0.0) -> np.ndarray:
    """
    Compute PCS values from axial/rhombic geometry factors.

    Returns PCS in ppm.
    """
    return (dchi_ax * Gax + dchi_rh * Grh) * 1e4 / (12.0 * np.pi)


def pcs_full(coords, metal, dchi_ax, dchi_rh=0.0):
    """
    Compute PCS directly for all coordinates.

    Returns PCS in ppm.
    """
    _, _, _, Gax, Grh = geom_factors(coords, metal)
    return pcs_from_G(Gax, Grh, dchi_ax, dchi_rh)


def bond_geom_in_frame(coords, labels, atom1, atom2):
    """
    Return bond-vector geometry for atom1 -> atom2 in the rotated tensor frame.

    Parameters
    ----------
    coords : (N, 3) ndarray
        Coordinates in the current tensor frame.
    labels : list[str]
        Atom labels corresponding to coords.
    atom1, atom2 : str
        Labels of the two atoms defining the bond vector.

    Returns
    -------
    tuple[float, float, float] | None
        (theta, phi, r_angstrom) if both atoms are found and the distance is valid,
        otherwise None.
    """
    label_to_index = {label: i for i, label in enumerate(labels)}
    if atom1 not in label_to_index or atom2 not in label_to_index:
        return None

    vec = coords[label_to_index[atom2]] - coords[label_to_index[atom1]]
    r = np.linalg.norm(vec)
    if r < 1e-9:
        return None

    theta = np.arccos(np.clip(vec[2] / r, -1.0, 1.0))
    phi = np.arctan2(vec[1], vec[0])
    return theta, phi, float(r)


# ============================================================================
# State extraction helpers
# ============================================================================

def get_current_coords_and_metal(state: dict):
    """
    Extract the current rotated coordinates and display labels from the main app state.

    Returns
    -------
    coords : (N, 3) ndarray
        Rotated coordinates in Å, already centered in the current tensor frame.
    labels : list[str]
        Display labels corresponding to coords.
    metal : (3,) ndarray
        Metal position in the rotated frame. This is always [0, 0, 0].
    polar_data : list
        Raw polar data returned by state['filter_atoms'].
    """
    filter_fn = state.get("filter_atoms")
    if filter_fn is None:
        raise RuntimeError("state['filter_atoms'] was not found.")

    polar_data, rotated_sel = filter_fn(state)
    coords = np.array([[dx, dy, dz] for dx, dy, dz in rotated_sel], dtype=float)
    labels = [atom for atom, _, _ in polar_data]
    metal = np.zeros(3, dtype=float)

    return coords, labels, metal, polar_data


def get_tensor_values(state: dict) -> tuple[float, float]:
    """
    Read Δχ_ax and Δχ_rh from the shared application state.

    Returns
    -------
    tuple[float, float]
        (dchi_ax, dchi_rh) in units of 10^-32 m^3.
    """
    try:
        dchi_ax = float(state["tensor_entry"].get() or 1.0)
    except Exception:
        dchi_ax = 1.0

    try:
        dchi_rh = float(state.get("rh_dchi_rh", 0.0) or 0.0)
    except Exception:
        dchi_rh = 0.0

    return dchi_ax, dchi_rh


def get_exp_pcs(state: dict) -> dict[int, float]:
    """
    Return experimental PCS values from state['delta_exp_values'].
    """
    return dict(state.get("delta_exp_values") or {})


def get_selected_ids(state: dict) -> list[int]:
    """
    Return the currently selected Ref IDs used by the main app.
    """
    return list(state.get("current_selected_ids") or [])


# ============================================================================
# Module 1: Multi-conformer ensemble
# ============================================================================

@dataclass
class Conformer:
    """
    One conformer used for ensemble PCS fitting.

    Attributes
    ----------
    name : str
        Display name of the conformer.
    coords : ndarray
        (N, 3) coordinates centered on the metal, in Å.
    labels : list[str]
        Atom labels matching coords.
    weight : float
        Current conformer weight.
    """
    name: str
    coords: np.ndarray
    labels: list[str]
    weight: float = 1.0


def ensemble_pcs(
    conformers: list[Conformer],
    dchi_ax: float,
    dchi_rh: float = 0.0,
) -> np.ndarray:
    """
    Compute ensemble-averaged PCS values.

    The conformers must contain the same atoms in the same order.
    """
    if not conformers:
        raise ValueError("No conformers were provided.")

    weight_sum = sum(conf.weight for conf in conformers)
    if weight_sum <= 0:
        raise ValueError("Sum of conformer weights must be positive.")

    metal = np.zeros(3, dtype=float)
    pcs_avg = np.zeros(len(conformers[0].coords), dtype=float)

    for conf in conformers:
        _, _, _, Gax, Grh = geom_factors(conf.coords, metal)
        pcs_avg += (conf.weight / weight_sum) * pcs_from_G(Gax, Grh, dchi_ax, dchi_rh)

    return pcs_avg


def fit_multiconf(
    conformers: list[Conformer],
    obs: dict[int, float],
    ids: list[int],
    dchi_ax0: float,
    dchi_rh0: float = 0.0,
    fit_weights: bool = True,
) -> dict:
    """
    Fit Δχ_ax, optionally Δχ_rh, and optionally conformer weights.

    Parameters
    ----------
    conformers : list[Conformer]
        Conformer list.
    obs : dict[int, float]
        Experimental PCS values as {ref_id: pcs_exp}.
    ids : list[int]
        Ref IDs corresponding to the coordinate order in the conformers.
    dchi_ax0, dchi_rh0 : float
        Initial tensor values.
    fit_weights : bool
        Whether to co-fit the conformer weights.

    Returns
    -------
    dict
        Fitted tensor values, weights, RMSD, and per-point diagnostics.
    """
    if len(conformers) < 2:
        raise ValueError("At least 2 conformers are required.")

    id_to_index = {rid: i for i, rid in enumerate(ids)}
    obs_pairs = [(rid, id_to_index[rid], val) for rid, val in obs.items() if rid in id_to_index]

    if len(obs_pairs) < 3:
        raise ValueError("At least 3 experimental PCS values are required.")

    obs_ids = [p[0] for p in obs_pairs]
    obs_idx = np.array([p[1] for p in obs_pairs], dtype=int)
    obs_val = np.array([p[2] for p in obs_pairs], dtype=float)

    n_conf = len(conformers)

    def residuals(x):
        dchi_ax = x[0]
        dchi_rh = x[1]

        if fit_weights and len(x) > 2:
            raw = np.abs(x[2:]) + 1e-8
            weights = raw / raw.sum()
            for conf, w in zip(conformers, weights):
                conf.weight = float(w)

        pcs_pred = ensemble_pcs(conformers, dchi_ax, dchi_rh)
        return pcs_pred[obs_idx] - obs_val

    x0 = np.concatenate(
        [
            [dchi_ax0, dchi_rh0],
            np.ones(n_conf, dtype=float) / n_conf if fit_weights else np.array([], dtype=float),
        ]
    )

    bounds_lo = [-np.inf, -np.inf] + ([-np.inf] * n_conf if fit_weights else [])
    bounds_hi = [np.inf, np.inf] + ([np.inf] * n_conf if fit_weights else [])

    res = least_squares(residuals, x0, bounds=(bounds_lo, bounds_hi))

    dchi_ax = float(res.x[0])
    dchi_rh = float(res.x[1])

    if fit_weights and len(res.x) > 2:
        raw = np.abs(res.x[2:]) + 1e-8
        weights = raw / raw.sum()
        for conf, w in zip(conformers, weights):
            conf.weight = float(w)

    pcs_pred = ensemble_pcs(conformers, dchi_ax, dchi_rh)
    resid = pcs_pred[obs_idx] - obs_val
    rmsd = float(np.sqrt(np.mean(resid ** 2)))

    per_point = [
        (obs_ids[i], float(obs_val[i]), float(pcs_pred[obs_idx[i]]), float(resid[i]))
        for i in range(len(obs_pairs))
    ]

    return {
        "dchi_ax": dchi_ax,
        "dchi_rh": dchi_rh,
        "weights": [conf.weight for conf in conformers],
        "rmsd": rmsd,
        "n": len(obs_pairs),
        "per_point": per_point,
    }


# ============================================================================
# Module 2: Joint PCS + RDC fitting
# ============================================================================

def rdc_from_dchi(
    theta_b: float,
    phi_b: float,
    r_bond_ang: float,
    dchi_ax: float,
    dchi_rh: float,
    B0: float = 14.1,
    T: float = 298.0,
    gamma1: float = GAMMA_H,
    gamma2: float = GAMMA_N,
) -> float:
    """
    Compute RDC [Hz] from the susceptibility tensor.

    Parameters
    ----------
    theta_b, phi_b : float
        Bond-vector angles in the tensor frame.
    r_bond_ang : float
        Bond length in Å.
    dchi_ax, dchi_rh : float
        Tensor components in units of 10^-32 m^3.
    B0 : float
        Magnetic field in Tesla.
    T : float
        Temperature in Kelvin.
    gamma1, gamma2 : float
        Gyromagnetic ratios of the two nuclei.

    Returns
    -------
    float
        Predicted RDC in Hz.
    """
    dchi_ax_si = dchi_ax * 1e-32
    dchi_rh_si = dchi_rh * 1e-32

    prefactor = B0 ** 2 / (15.0 * MU0 * NA * KB * T)
    Aa = dchi_ax_si * prefactor
    Ar = dchi_rh_si * prefactor

    r_m = r_bond_ang * 1e-10
    Dmax = -(MU0 / (4.0 * np.pi)) * (HBAR * gamma1 * gamma2) / (2.0 * np.pi * r_m ** 3)

    cos2 = np.cos(theta_b) ** 2
    sin2 = 1.0 - cos2

    return Dmax * (Aa * (3.0 * cos2 - 1.0) + 1.5 * Ar * sin2 * np.cos(2.0 * phi_b))


def fit_joint_pcs_rdc(
    coords: np.ndarray,
    labels: list[str],
    obs_pcs: dict[int, float],
    ids: list[int],
    rdc_rows: list[dict],
    dchi_ax0: float,
    dchi_rh0: float = 0.0,
    B0: float = 14.1,
    T: float = 298.0,
    w_rdc: float = 1.0,
) -> dict:
    """
    Simultaneous PCS + RDC fitting with fixed geometry.

    Parameters
    ----------
    coords : ndarray
        Coordinates in the tensor frame.
    labels : list[str]
        Atom labels matching coords.
    obs_pcs : dict[int, float]
        Experimental PCS values.
    ids : list[int]
        Ref IDs corresponding to coords.
    rdc_rows : list[dict]
        RDC input rows with keys such as atom1, atom2, rdc_exp, rdc_err, nuc1, nuc2.
    dchi_ax0, dchi_rh0 : float
        Initial tensor values.
    B0 : float
        Magnetic field in Tesla.
    T : float
        Temperature in Kelvin.
    w_rdc : float
        Relative weighting factor for RDC residuals.

    Returns
    -------
    dict
        Fitted tensor values and PCS/RDC diagnostics.
    """
    metal = np.zeros(3, dtype=float)
    id_to_index = {rid: i for i, rid in enumerate(ids)}

    obs_pairs = [(rid, id_to_index[rid], obs_pcs[rid]) for rid in obs_pcs if rid in id_to_index]
    obs_ids = [p[0] for p in obs_pairs]
    obs_idx = np.array([p[1] for p in obs_pairs], dtype=int)
    obs_val = np.array([p[2] for p in obs_pairs], dtype=float)

    _, _, _, Gax_all, Grh_all = geom_factors(coords, metal)

    rdc_geom = []
    for row in rdc_rows:
        geo = bond_geom_in_frame(coords, labels, row["atom1"], row["atom2"])
        if geo is None:
            continue

        gamma1 = GYRO.get(str(row.get("nuc1", "H")).upper(), GAMMA_H)
        gamma2 = GYRO.get(str(row.get("nuc2", "N")).upper(), GAMMA_N)

        rdc_geom.append(
            (
                *geo,
                float(row["rdc_exp"]),
                float(row.get("rdc_err", 0.5)),
                gamma1,
                gamma2,
            )
        )

    def residuals(x):
        dchi_ax = x[0]
        dchi_rh = x[1]

        pcs_pred = pcs_from_G(Gax_all[obs_idx], Grh_all[obs_idx], dchi_ax, dchi_rh)
        res_pcs = pcs_pred - obs_val

        res_rdc = []
        for theta_b, phi_b, r_bond, rdc_exp, rdc_err, gamma1, gamma2 in rdc_geom:
            rdc_pred = rdc_from_dchi(
                theta_b,
                phi_b,
                r_bond,
                dchi_ax,
                dchi_rh,
                B0=B0,
                T=T,
                gamma1=gamma1,
                gamma2=gamma2,
            )
            res_rdc.append((rdc_pred - rdc_exp) / max(rdc_err, 0.1) * w_rdc)

        return np.concatenate([res_pcs, np.array(res_rdc, dtype=float)])

    res = least_squares(residuals, [dchi_ax0, dchi_rh0])

    dchi_ax = float(res.x[0])
    dchi_rh = float(res.x[1])

    pcs_pred_all = pcs_from_G(Gax_all, Grh_all, dchi_ax, dchi_rh)
    pcs_resid = pcs_pred_all[obs_idx] - obs_val
    rmsd_pcs = float(np.sqrt(np.mean(pcs_resid ** 2))) if len(pcs_resid) else np.nan

    rdc_results = []
    for theta_b, phi_b, r_bond, rdc_exp, _, gamma1, gamma2 in rdc_geom:
        rdc_pred = rdc_from_dchi(
            theta_b,
            phi_b,
            r_bond,
            dchi_ax,
            dchi_rh,
            B0=B0,
            T=T,
            gamma1=gamma1,
            gamma2=gamma2,
        )
        rdc_results.append(
            {
                "rdc_pred": float(rdc_pred),
                "rdc_exp": float(rdc_exp),
                "residual": float(rdc_pred - rdc_exp),
            }
        )

    per_point_pcs = [
        (obs_ids[i], float(obs_val[i]), float(pcs_pred_all[obs_idx[i]]), float(pcs_resid[i]))
        for i in range(len(obs_idx))
    ]

    return {
        "dchi_ax": dchi_ax,
        "dchi_rh": dchi_rh,
        "rmsd_pcs": rmsd_pcs,
        "rdc_results": rdc_results,
        "per_point_pcs": per_point_pcs,
        "B0": B0,
        "T": T,
    }


# ============================================================================
# Module 3: PRE (Solomon-Bloembergen)
# ============================================================================

def pre_gamma2(
    r_ang: np.ndarray,
    mu_eff_muB: float,
    tau_c: float,
    omega_H: float,
) -> np.ndarray:
    """
    Compute Gamma2 [s^-1] for PRE.

    Gamma2 = (1/15) * (mu0/4pi)^2 * gamma_H^2 * mu_eff^2
             * tau_c * [4 + 3 / (1 + omega_H^2 * tau_c^2)] / r^6
    """
    mu_si = mu_eff_muB * MU_B
    r6 = np.where(r_ang > 0, (r_ang * 1e-10) ** 6, 1e-60)
    spectral_density = 4.0 * tau_c + 3.0 * tau_c / (1.0 + (omega_H * tau_c) ** 2)
    prefactor = (1.0 / 15.0) * (MU0 / (4.0 * np.pi)) ** 2 * GAMMA_H ** 2 * mu_si ** 2
    return prefactor * spectral_density / r6


def pre_gamma1(
    r_ang: np.ndarray,
    mu_eff_muB: float,
    tau_c: float,
    omega_H: float,
) -> np.ndarray:
    """
    Compute Gamma1 [s^-1] for PRE.

    Gamma1 = (2/5) * (mu0/4pi)^2 * gamma_H^2 * mu_eff^2
             * tau_c * 3 / (1 + omega_H^2 * tau_c^2) / r^6
    """
    mu_si = mu_eff_muB * MU_B
    r6 = np.where(r_ang > 0, (r_ang * 1e-10) ** 6, 1e-60)
    spectral_density = 3.0 * tau_c / (1.0 + (omega_H * tau_c) ** 2)
    prefactor = (2.0 / 5.0) * (MU0 / (4.0 * np.pi)) ** 2 * GAMMA_H ** 2 * mu_si ** 2
    return prefactor * spectral_density / r6


def fit_pre(
    r_ang: np.ndarray,
    obs_gamma2: dict[int, float],
    ids: list[int],
    mu_eff: float,
    B0: float = 14.1,
    mode: str = "r2",
) -> dict:
    """
    Fit tau_c for a fixed geometry using PRE Gamma2 data.

    Parameters
    ----------
    r_ang : ndarray
        Distances in Å.
    obs_gamma2 : dict[int, float]
        Observed PRE values as {ref_id: Gamma2_exp}.
    ids : list[int]
        Ref IDs corresponding to r_ang.
    mu_eff : float
        Effective magnetic moment in mu_B.
    B0 : float
        Magnetic field in Tesla.
    mode : str
        Currently kept for future extension. Present implementation uses Gamma2 fitting.

    Returns
    -------
    dict
        Fitted tau_c and PRE diagnostics.
    """
    omega_H = GAMMA_H * B0
    id_to_index = {rid: i for i, rid in enumerate(ids)}
    pairs = [(rid, id_to_index[rid], val) for rid, val in obs_gamma2.items() if rid in id_to_index]

    if not pairs:
        raise ValueError("No matching Ref IDs were found for PRE fitting.")

    obs_ids = [p[0] for p in pairs]
    obs_idx = np.array([p[1] for p in pairs], dtype=int)
    obs_val = np.array([p[2] for p in pairs], dtype=float)

    def cost(log_tau):
        tau_c = 10 ** log_tau[0]
        pred = pre_gamma2(r_ang[obs_idx], mu_eff, tau_c, omega_H)
        return np.sum(((pred - obs_val) / np.maximum(obs_val * 0.05, 0.1)) ** 2) / len(obs_val)

    res = minimize(cost, [np.log10(5e-9)], method="L-BFGS-B", bounds=[(-12, -6)])
    tau_c = 10 ** res.x[0]

    g2_all = pre_gamma2(r_ang, mu_eff, tau_c, omega_H)
    g1_all = pre_gamma1(r_ang, mu_eff, tau_c, omega_H)

    resid = g2_all[obs_idx] - obs_val
    rmsd = float(np.sqrt(np.mean(resid ** 2)))

    per_point = [
        (obs_ids[i], float(obs_val[i]), float(g2_all[obs_idx[i]]), float(r_ang[obs_idx[i]]))
        for i in range(len(pairs))
    ]

    return {
        "tau_c": float(tau_c),
        "rmsd": rmsd,
        "chi2": float(res.fun),
        "success": bool(res.success),
        "omega_H": float(omega_H),
        "mu_eff": float(mu_eff),
        "per_point": per_point,
        "g2_all": g2_all,
        "g1_all": g1_all,
    }


# ============================================================================
# Module 4: Isosurface data generation
# ============================================================================

def compute_pcs_slice(
    dchi_ax: float,
    dchi_rh: float,
    metal: np.ndarray,
    fixed_axis: str,
    fixed_val: float,
    g1: np.ndarray,
    g2: np.ndarray,
    grid_n: int,
) -> np.ndarray:
    """
    Compute PCS on a 2D slice grid with one fixed coordinate.

    Parameters
    ----------
    fixed_axis : {"x", "y", "z"}
        Which coordinate is fixed.
    fixed_val : float
        Fixed coordinate value in Å.
    g1, g2 : ndarray
        Grid arrays.
    grid_n : int
        Number of grid points along one dimension.

    Returns
    -------
    ndarray
        PCS slice in ppm.
    """
    n = grid_n

    if fixed_axis == "z":
        pts = np.column_stack([g1.ravel(), g2.ravel(), np.full(n * n, fixed_val)])
    elif fixed_axis == "y":
        pts = np.column_stack([g1.ravel(), np.full(n * n, fixed_val), g2.ravel()])
    else:
        pts = np.column_stack([np.full(n * n, fixed_val), g1.ravel(), g2.ravel()])

    _, _, _, Gax, Grh = geom_factors(pts, metal)
    pcs = pcs_from_G(Gax, Grh, dchi_ax, dchi_rh)

    # Mask out the immediate near-metal region
    r = np.linalg.norm(pts - metal, axis=1)
    pcs[r < 1.5] = np.nan

    return np.clip(pcs, -20.0, 20.0).reshape(n, n)


def compute_sphere_pcs(
    dchi_ax: float,
    dchi_rh: float,
    metal: np.ndarray,
    r_shell: float,
    n_u: int = 60,
    n_v: int = 30,
):
    """
    Compute PCS values on a spherical shell of radius r_shell around the metal.
    """
    cx, cy, cz = metal
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)

    Xs = cx + r_shell * np.outer(np.cos(u), np.sin(v))
    Ys = cy + r_shell * np.outer(np.sin(u), np.sin(v))
    Zs = cz + r_shell * np.outer(np.ones(n_u), np.cos(v))

    pts = np.column_stack([Xs.ravel(), Ys.ravel(), Zs.ravel()])
    _, _, _, Gax, Grh = geom_factors(pts, metal)
    pcs = pcs_from_G(Gax, Grh, dchi_ax, dchi_rh)

    return Xs, Ys, Zs, pcs.reshape(n_u, n_v)


def build_isosurface_data(
    dchi_ax: float,
    dchi_rh: float,
    metal: np.ndarray,
    span: float = 15.0,
    grid_n: int = 50,
    r_shells: list | None = None,
) -> dict:
    """
    Build all data required for isosurface-style PCS visualization.

    Returns
    -------
    dict
        Contains 2D slice grids, PCS values, shell surfaces, and plotting limits.
    """
    if r_shells is None:
        r_shells = [5.0, 8.0, 12.0]

    cx, cy, cz = metal

    xs = np.linspace(cx - span, cx + span, grid_n)
    ys = np.linspace(cy - span, cy + span, grid_n)
    zs = np.linspace(cz - span, cz + span, grid_n)

    G_xy = np.meshgrid(xs, ys)
    G_xz = np.meshgrid(xs, zs)
    G_yz = np.meshgrid(ys, zs)

    pcs_xy = compute_pcs_slice(dchi_ax, dchi_rh, metal, "z", cz, G_xy[0], G_xy[1], grid_n)
    pcs_xz = compute_pcs_slice(dchi_ax, dchi_rh, metal, "y", cy, G_xz[0], G_xz[1], grid_n)
    pcs_yz = compute_pcs_slice(dchi_ax, dchi_rh, metal, "x", cx, G_yz[0], G_yz[1], grid_n)

    vlim = max(
        np.nanpercentile(np.abs(pcs_xy), 97),
        np.nanpercentile(np.abs(pcs_xz), 97),
        np.nanpercentile(np.abs(pcs_yz), 97),
        0.1,
    )

    spheres = []
    for r_shell in r_shells:
        Xs, Ys, Zs, pcs_s = compute_sphere_pcs(dchi_ax, dchi_rh, metal, r_shell)
        spheres.append((Xs, Ys, Zs, pcs_s))

    return {
        "xs": xs,
        "ys": ys,
        "zs": zs,
        "G_xy": G_xy,
        "G_xz": G_xz,
        "G_yz": G_yz,
        "pcs_xy": pcs_xy,
        "pcs_xz": pcs_xz,
        "pcs_yz": pcs_yz,
        "vlim": vlim,
        "spheres": spheres,
        "metal": metal,
        "r_shells": r_shells,
    }


# ============================================================================
# Module 5: Multi-lanthanide fitting
# ============================================================================

def fit_multilanthanid(
    coords: np.ndarray,
    ids: list[int],
    datasets: list[dict],
    lanthanides: list[str],
    shared_geometry: bool = True,
) -> dict:
    """
    Simultaneous fitting for multiple lanthanide datasets.

    Parameters
    ----------
    coords : ndarray
        Coordinates in the tensor frame.
    ids : list[int]
        Ref IDs corresponding to coords.
    datasets : list[dict]
        Each dataset should contain {"obs": {ref_id: pcs_exp}, ...}.
    lanthanides : list[str]
        Lanthanide labels for the datasets.
    shared_geometry : bool
        Reserved for future extension.
        In the current implementation, geometry is already fixed by coords,
        so only Δχ_ax / Δχ_rh are fitted per lanthanide.

    Returns
    -------
    dict
        Fitted tensor values and diagnostics per lanthanide.
    """
    metal = np.zeros(3, dtype=float)
    _, _, _, Gax_all, Grh_all = geom_factors(coords, metal)
    id_to_index = {rid: i for i, rid in enumerate(ids)}

    obs_idx_list = []
    obs_val_list = []
    obs_id_list = []

    for ds in datasets:
        obs_pairs = [(rid, id_to_index[rid], ds["obs"][rid]) for rid in ds["obs"] if rid in id_to_index]
        obs_id_list.append([p[0] for p in obs_pairs])
        obs_idx_list.append(np.array([p[1] for p in obs_pairs], dtype=int))
        obs_val_list.append(np.array([p[2] for p in obs_pairs], dtype=float))

    x0 = []
    for ln in lanthanides:
        db = LANTHANIDE_DB.get(ln, LANTHANIDE_DB["Tb"])
        x0 += [db["dchi_ax0"], 0.0]
    x0 = np.array(x0, dtype=float)

    def residuals(x):
        res_all = []
        for m, (obs_idx, obs_val) in enumerate(zip(obs_idx_list, obs_val_list)):
            dchi_ax = x[2 * m]
            dchi_rh = x[2 * m + 1]
            pred = pcs_from_G(Gax_all[obs_idx], Grh_all[obs_idx], dchi_ax, dchi_rh)
            res_all.append(pred - obs_val)
        return np.concatenate(res_all)

    result = least_squares(residuals, x0)
    x_opt = result.x

    results_per_ln = []
    for m, ln in enumerate(lanthanides):
        dchi_ax = float(x_opt[2 * m])
        dchi_rh = float(x_opt[2 * m + 1])

        obs_idx = obs_idx_list[m]
        obs_val = obs_val_list[m]
        obs_ids = obs_id_list[m]

        pred = pcs_from_G(Gax_all[obs_idx], Grh_all[obs_idx], dchi_ax, dchi_rh)
        resid = pred - obs_val
        rmsd = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else np.nan

        per_point = [
            (obs_ids[i], float(obs_val[i]), float(pred[i]), float(resid[i]))
            for i in range(len(obs_idx))
        ]

        results_per_ln.append(
            {
                "lanthanide": ln,
                "dchi_ax": dchi_ax,
                "dchi_rh": dchi_rh,
                "rmsd": rmsd,
                "n": len(obs_idx),
                "per_point": per_point,
            }
        )

    return {
        "results": results_per_ln,
        "lanthanides": lanthanides,
    }


# ============================================================================
# Diagnostic helpers
# ============================================================================

def fit_quality(pred: np.ndarray, obs_idx: np.ndarray, obs_val: np.ndarray) -> dict:
    """
    Compute simple fit-quality metrics.

    Returns
    -------
    dict
        Contains R^2, RMSD, and Q factor.
    """
    p = pred[obs_idx]
    d = p - obs_val

    rmsd = float(np.sqrt(np.mean(d ** 2)))
    ss_res = float(np.sum(d ** 2))
    ss_tot = float(np.sum((obs_val - obs_val.mean()) ** 2))

    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    q_factor = float(np.sqrt(ss_res / np.sum(obs_val ** 2))) if np.any(obs_val != 0) else float("nan")

    return {
        "r2": r2,
        "rmsd": rmsd,
        "q_factor": q_factor,
    }