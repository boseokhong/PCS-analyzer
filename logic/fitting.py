# logic/fitting.py

import numpy as np
from scipy.optimize import least_squares, differential_evolution
from logic.rotate_align import rotate_euler

def compute_quality_metrics(per_point):
    """
    per_point: [(ref_id, delta_exp, delta_pred, residual), ...]
    Returns: r2, rmsd, q_factor, chi2_n, n
    """
    if not per_point:
        return dict(
            r2=float("nan"),
            rmsd=float("nan"),
            q_factor=float("nan"),
            chi2_n=float("nan"),
            n=0,
        )

    exp_v = np.array([p[1] for p in per_point], float)
    pred_v = np.array([p[2] for p in per_point], float)
    resid = pred_v - exp_v

    rmsd = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((exp_v - exp_v.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else float("nan")

    ss_exp = float(np.sum(exp_v ** 2))
    q_factor = float(np.sqrt(ss_res / ss_exp)) if ss_exp > 1e-15 else float("nan")

    chi2_n = float(ss_res / len(resid))

    return dict(
        r2=r2,
        rmsd=rmsd,
        q_factor=q_factor,
        chi2_n=chi2_n,
        n=len(per_point),
    )

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def rodrigues(points, axis, angle_deg, origin=(0,0,0)):
    """points: (N,3), axis: (3,), origin: (3,) — origin을 기준으로 axis 방향으로 angle만큼 회전"""
    pts = np.asarray(points, float) - np.asarray(origin, float)
    a = _unit(np.asarray(axis, float))
    th = np.deg2rad(angle_deg)
    K = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    rot = pts @ R.T
    return rot + np.asarray(origin, float)

# ---------- single donor axis(Mode A: single donor) ----------
def _angles_to_rotation(points, metal, donor, theta_deg, alpha_deg):
    """
    1) v = Metal→Donor 단위벡터
    2) v와 z 사이 각도 β를 θ로 맞추기 위해, u = v × z 축으로 (θ-β) 만큼 회전
    3) 그 후, 새 v' 축으로 α 회전 (리간드 axial 회전)
    모든 회전은 metal을 기준(origin)으로 수행.
    """
    z = np.array([0,0,1.0])
    metal = np.asarray(metal, float)
    donor = np.asarray(donor, float)
    v = _unit(donor - metal)

    beta = np.degrees(np.arccos(np.clip(np.dot(v, z), -1.0, 1.0)))
    delta = float(theta_deg) - beta

    u = np.cross(v, z)
    if np.linalg.norm(u) < 1e-9:
        u = np.array([1.0, 0.0, 0.0])

    # 1단계: u축으로 delta 회전
    pts1 = rodrigues(points, u, delta, origin=metal)
    v1 = _unit(rodrigues([metal + v], u, delta, origin=metal)[0] - metal)

    # 2단계: v'축으로 alpha 회전
    pts2 = rodrigues(pts1, v1, alpha_deg, origin=metal)
    return pts2

# ---------- multi donor axis(Mode A: multi-donor) ----------
def _axis_from_donors(metal, donor_points, mode='bisector'):
    """
    mode:
      - 'first'         : 첫 donor 방향
      - 'bisector'/'average' : donor 방향 단위벡터들의 합 방향(평균 방향)
      - 'centroid'      : 금속→donor 좌표의 무게중심(centroid) 방향
      - 'normal'        : donor들이 만드는 평면의 법선(2개면 cross, 3개 이상 SVD의 최소분산 성분)
      - 'pca'           : donor 방향들의 최대분산 주성분(PC1) 방향
    """
    pts = [np.asarray(p, float) for p in donor_points]
    if not pts:
        raise ValueError("No donor points.")

    vs = np.array([_unit(p - metal) for p in pts])
    m = (mode or 'bisector').lower()

    if m in ('first', 'v1'):
        a = vs[0]

    elif m in ('bisector', 'average', 'avg', 'mean_dir'):
        a = _unit(np.sum(vs, axis=0))

    elif m in ('centroid',):
        centroid = np.mean(np.stack(pts, axis=0), axis=0)
        a = _unit(centroid - metal)

    elif m in ('normal', 'plane_normal', 'perp'):
        if len(vs) == 1:
            a = vs[0]
        elif len(vs) == 2:
            n = np.cross(vs[0], vs[1])
            a = vs[0] if np.linalg.norm(n) < 1e-12 else _unit(n)
        else:
            X = vs - vs.mean(axis=0)
            try:
                # SVD: 최소분산 성분(마지막)이 평면의 법선
                _, _, Vt = np.linalg.svd(X, full_matrices=False)
                n = Vt[-1]
            except np.linalg.LinAlgError:
                n = np.cross(vs[0], vs[1])
            a = vs[0] if np.linalg.norm(n) < 1e-12 else _unit(n)

    elif m in ('pca', 'principal', 'pc1', 'principal_axis'):
        if len(vs) == 1:
            a = vs[0]
        else:
            X = vs - vs.mean(axis=0)
            try:
                # SVD: 최대분산 성분(첫 번째)
                _, _, Vt = np.linalg.svd(X, full_matrices=False)
                a = _unit(Vt[0])
            except np.linalg.LinAlgError:
                a = vs[0]
    else:
        # 알 수 없는 모드는 평균 방향으로 폴백
        a = _unit(np.sum(vs, axis=0))

    return a

def _angles_to_rotation_multi(points, metal, donor_points, theta_deg, alpha_deg, axis_mode='bisector'):
    """
    다중 donor들의 축을 만들어 동일한 로직(θ 맞추기 → α 비틀기) 적용
    """
    z = np.array([0,0,1.0])
    metal = np.asarray(metal, float)
    donor_points = [np.asarray(p, float) for p in donor_points]
    v = _axis_from_donors(metal, donor_points, mode=axis_mode)

    beta = np.degrees(np.arccos(np.clip(np.dot(v, z), -1.0, 1.0)))
    delta = float(theta_deg) - beta

    u = np.cross(v, z)
    if np.linalg.norm(u) < 1e-9:
        u = np.array([1.0, 0.0, 0.0])

    pts1 = rodrigues(points, u, delta, origin=metal)
    v1 = _unit(rodrigues([metal + v], u, delta, origin=metal)[0] - metal)
    pts2 = rodrigues(pts1, v1, alpha_deg, origin=metal)
    return pts2

# ---------- PCS calc (ax only) ----------
def geom_factor_and_pcs(coords, metal, delta_chi_ax):
    """
    coords: (N,3), metal: (3,)
    Returns: r (Å), theta (radians), Gi, and δ_PCS (ppm).
    """
    metal = np.asarray(metal, float)
    vecs = coords - metal
    r = np.linalg.norm(vecs, axis=1)
    z = np.array([0,0,1.0])
    cos_th = np.clip((vecs @ z) / np.where(r==0, 1.0, r), -1.0, 1.0)
    theta = np.arccos(cos_th)

    Gi = (3*np.cos(theta)**2 - 1) / np.where(r==0, np.inf, r**3)
    K = 1e4 * float(delta_chi_ax) / (12*np.pi)
    dpcs = K * Gi
    return r, theta, Gi, dpcs

# ---------- PCS calc (ax + rh) ----------
def geom_factors_ax_rh(coords, metal):
    """
    coords: (N,3) rotated into tensor frame (z is principal axis)
    returns: r, theta, phi, Gax, Grh
    """
    metal = np.asarray(metal, float)
    vecs = np.asarray(coords, float) - metal
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]

    r = np.linalg.norm(vecs, axis=1)
    r_safe = np.where(r == 0.0, np.inf, r)

    cos_th = np.clip(z / r_safe, -1.0, 1.0)
    theta = np.arccos(cos_th)
    phi = np.arctan2(y, x)

    Gax = (3.0 * np.cos(theta)**2 - 1.0) / (r_safe**3)
    Grh = (1.5 * (np.sin(theta)**2) * np.cos(2.0 * phi)) / (r_safe**3)
    return r, theta, phi, Gax, Grh

def pcs_ax_rh_from_G(Gax, Grh, dchi_ax, dchi_rh):
    """axial + rhombic PCS (ppm)"""
    return ((float(dchi_ax) * Gax + float(dchi_rh) * Grh) * 1e4) / (12.0 * np.pi)

# ---------- Differential evolution (DE function helper) ----------
def _make_de_bounds_finite(lb, ub, x0=None, default_span=50.0):
    """
    Convert possibly infinite bounds into finite bounds for differential_evolution.
    least_squares can handle inf bounds, but DE cannot.
    """
    if x0 is None:
        x0 = [0.0] * len(lb)

    lb2, ub2 = [], []
    for lo, hi, x in zip(lb, ub, x0):
        lo_finite = np.isfinite(lo)
        hi_finite = np.isfinite(hi)

        if lo_finite and hi_finite:
            lb2.append(float(lo))
            ub2.append(float(hi))
        elif (not lo_finite) and (not hi_finite):
            lb2.append(float(x) - default_span)
            ub2.append(float(x) + default_span)
        elif not lo_finite:
            lb2.append(float(hi) - 2.0 * default_span)
            ub2.append(float(hi))
        else:  # upper is infinite
            lb2.append(float(lo))
            ub2.append(float(lo) + 2.0 * default_span)

    return lb2, ub2

def _run_two_stage_fit(residual_func, x0, lb, ub,
                       use_global_search=False,
                       progress_cb=None):
    """
    Optional global search (DE) followed by local least-squares refinement.
    DE requires finite bounds, so infinite bounds are converted temporarily.
    """
    if use_global_search:
        lb_de, ub_de = _make_de_bounds_finite(lb, ub, x0=x0, default_span=50.0)
        bounds_de = list(zip(lb_de, ub_de))

        if progress_cb:
            progress_cb("Running global search (DE)...")

        def objective(x):
            r = residual_func(x)
            return float(np.sum(np.asarray(r, float) ** 2))

        de_res = differential_evolution(
            objective,
            bounds=bounds_de,
            polish=False,
            seed=42,
        )
        x_start = np.array(de_res.x, float)
    else:
        x_start = np.array(x0, float)

    if progress_cb:
        progress_cb("Running local refinement...")

    res = least_squares(residual_func, x_start, bounds=(lb, ub))

    if progress_cb:
        progress_cb("Building fit summary...")

    return res

# ---------- Mode A: θ, α (multi-donor) ----------
def fit_theta_alpha_multi(state, donor_ids, proton_ids,
                          axis_mode='bisector',
                          fit_visible_as_group=True,
                          fit_delta_chi=False,
                          fit_delta_chi_rh=False,
                          use_global_search=False,
                          progress_cb=None):
    """
    donor_ids: Ref ID 리스트(1..N)
    proton_ids: δ_Exp가 들어있는 Ref ID 목록 (보통 H들)
    axis_mode: 'first' | 'bisector' | 'average' | 'centroid' | 'normal' | 'pca'
    """
    atom_data = state.get("atom_data_eff") or state.get("atom_data") or []
    ids = state.get("atom_ids_eff") or state.get("atom_ids") or []
    if not atom_data or not ids:
        raise RuntimeError("No atom data")
    if progress_cb:
        progress_cb("Preparing donor-axis fit...")

    id2idx = {rid: i for i, rid in enumerate(ids)}
    metal = np.array([state['x0'], state['y0'], state['z0']])
    abs_coords = np.array([a[1:4] for a in atom_data])
    donor_pts = [abs_coords[id2idx[rid]] for rid in donor_ids if rid in id2idx]
    if not donor_pts:
        raise RuntimeError("No valid donor ids for current atom set.")

    # rigid group
    if fit_visible_as_group:
        state['filter_atoms'](state)
        sel_ids = state.get('current_selected_ids', []) or ids
        group_indices = [id2idx[rid] for rid in sel_ids if rid in id2idx]
    else:
        group_indices = list(range(len(atom_data)))
    # 현재는 group_indices를 직접 쓰지는 않지만, 원하면 points=abs_coords[group_indices]로 바꿔 확장 가능
    # group_indices is prepared for future rigid-group extensions,
    # but the current implementation still fits against all selected observation points.

    delta_exp_values = state.get('delta_exp_values', {})
    obs_pairs = [(rid, delta_exp_values[rid]) for rid in proton_ids if rid in delta_exp_values]
    if len(obs_pairs) < 3 and not fit_delta_chi:
        pass  # 경고 없이 진행

    theta0, alpha0 = 0.0, 0.0
    tensor_entry = state.get('tensor_entry')
    dchi_ax0 = float((tensor_entry.get() if tensor_entry else None) or state.get('tensor', 1.0) or 1.0)
    dchi_rh0 = float(state.get('rh_dchi_rh', 0.0) or 0.0)

    if fit_delta_chi and fit_delta_chi_rh:
        x0 = np.array([theta0, alpha0, dchi_ax0, dchi_rh0])
    elif fit_delta_chi:
        x0 = np.array([theta0, alpha0, dchi_ax0])
    elif fit_delta_chi_rh:
        x0 = np.array([theta0, alpha0, dchi_rh0])  # Δχ_ax는 고정, Δχ_rh만 fit
    else:
        x0 = np.array([theta0, alpha0])

    def coords_for_ids(points, ids_subset):
        idxs = [id2idx[rid] for rid in ids_subset if rid in id2idx]
        return np.array([points[i] for i in idxs])

    def residuals(x):
        # --------------------------------------------
        # 0) UI/상태에서 "고정값"으로 쓸 Δχ 파라미터를 먼저 읽어둔다
        #    - fit 옵션에 따라 일부는 x에서 오고, 일부는 UI entry/state에서 고정값으로 온다.
        # --------------------------------------------
        tensor_entry = state.get('tensor_entry')
        dchi_ax_fixed = float((tensor_entry.get() if tensor_entry else None) or state.get('tensor', 1.0) or 1.0)
        dchi_rh_fixed = float(state.get('rh_dchi_rh', 0.0) or 0.0)  # Δχ_rh 기본값(고정용, Rh tab 입력)

        # --------------------------------------------
        # 1) least_squares가 최적화하는 파라미터 벡터 x 해석
        #    - 체크박스 조합에 따라 x의 길이가 달라진다.
        #    - (theta, alpha)는 항상 fit 대상
        #    - dchi_ax / dchi_rh는 체크된 것만 fit, 나머지는 고정값 사용
        # --------------------------------------------
        if fit_delta_chi and fit_delta_chi_rh:
            # 둘 다 fit: x = [theta, alpha, dchi_ax, dchi_rh]
            theta, alpha, dchi_ax, dchi_rh = x
        elif fit_delta_chi:
            # Δχ_ax만 fit: x = [theta, alpha, dchi_ax], Δχ_rh는 고정
            theta, alpha, dchi_ax = x
            dchi_rh = dchi_rh_fixed
        elif fit_delta_chi_rh:
            # Δχ_rh만 fit: x = [theta, alpha, dchi_rh], Δχ_ax는 고정
            theta, alpha, dchi_rh = x
            dchi_ax = dchi_ax_fixed
        else:
            # 둘 다 고정: x = [theta, alpha]
            theta, alpha = x
            dchi_ax = dchi_ax_fixed
            dchi_rh = dchi_rh_fixed

        # --------------------------------------------
        # 2) 좌표 회전(Mode A: multi-donor axis)
        #    - donor들로부터 축(v)을 만들고,
        #    - 그 축이 z축과 이루는 각을 theta로 맞추도록 회전,
        #    - 이후 새 축(v')을 기준으로 alpha 회전 (리간드 axial twist)
        #    => 이 단계까지는 "z축(주축)" 방향을 정렬하는 작업에 해당
        # --------------------------------------------
        rot_all = _angles_to_rotation_multi(
            points=abs_coords,
            metal=metal,
            donor_points=donor_pts,
            theta_deg=theta, alpha_deg=alpha,
            axis_mode=axis_mode
        )

        # --------------------------------------------
        # 3) φ(azimuth) 기준 고정: z축 주위 회전(az) 추가 적용
        #    왜 필요한가?
        #    - axial PCS(Δχ_ax)는 θ만 보므로, z축이 맞으면 충분하다. (φ에 무관)
        #    - rhombic PCS(Δχ_rh)는 cos(2φ) 항을 포함한다. 즉 φ가 바뀌면 예측값이 변한다.
        #
        #    그런데 φ는 "z축을 기준으로 xy평면에서의 각도"라서,
        #    같은 z축 정렬 상태에서도 xy축이 어디를 향하느냐(= z축 주위 회전)에 따라
        #    φ가 통째로 이동(φ -> φ + const)한다.
        #
        #    => Δχ_rh을 의미 있게 fit하려면,
        #       'φ=0의 기준(=x축 방향)'을 어떤 방식으로든 "고정"해야 한다.
        #
        #    여기서는:
        #    - Rhombicity 탭에서 사용자가 조절하는 angle_z_var(az)를
        #      fitting에서도 동일하게 적용해서,
        #      "사용자가 정의한 tensor frame의 xy기준"과
        #      "fitting이 사용하는 φ 기준"을 일치시키는 전략을 택한다.
        # --------------------------------------------
        try:
            az = float(state.get('angle_z_var', 0.0).get())  # Rhombicity 탭의 z-rotation 슬라이더 값
        except Exception:
            az = 0.0

        if abs(az) > 1e-12:
            # rotate_euler는 원점 기준 회전이므로, metal을 원점으로 이동했다가 회전 후 복귀
            coords0 = rot_all - metal

            # z축(=tensor 주축) 주위로 az 만큼 회전
            # -> z축은 유지되지만, xy 기준이 회전하면서 φ 기준이 함께 이동한다.
            rot0 = rotate_euler(coords0, 0.0, 0.0, az)

            # 다시 metal 기준 절대좌표로 복귀
            rot_all = rot0 + metal

        valid_obs_pairs = [(rid, v) for rid, v in obs_pairs if rid in id2idx]
        if not valid_obs_pairs:
            raise RuntimeError("No valid proton ids with δ_exp for current atom set.")

        pts_obs = coords_for_ids(rot_all, [rid for rid, _ in valid_obs_pairs])
        delta_exp = np.array([v for _, v in valid_obs_pairs], float)

        # --------------------------------------------
        # 4) PCS 모델 선택
        #    - Δχ_rh를 fit하거나(체크박스 ON),
        #    - 혹은 고정 Δχ_rh가 0이 아닌 경우엔(ax+rh 모델 활성),
        #      반드시 φ가 들어가는 Grh를 계산해야 한다.
        # --------------------------------------------
        if fit_delta_chi_rh or (abs(dchi_rh) > 0.0):
            # r,theta,phi 및 Gax/Grh 계산 (Grh에 cos(2φ) 포함)
            _, _, _, Gax, Grh = geom_factors_ax_rh(pts_obs, metal)

            # δ_pred = (Δχ_ax*Gax + Δχ_rh*Grh) * 1e4 / (12π)
            dpcs = pcs_ax_rh_from_G(Gax, Grh, dchi_ax, dchi_rh)
        else:
            # axial-only: φ가 필요 없으므로 기존 함수 사용
            _, _, _, dpcs = geom_factor_and_pcs(pts_obs, metal, dchi_ax)

        # --------------------------------------------
        # 6) residual = (예측 PCS) - (실험 δ_Exp)
        #    least_squares는 residual 벡터의 제곱합을 최소화한다.
        # --------------------------------------------
        return dpcs - delta_exp

    if fit_delta_chi and fit_delta_chi_rh:
        # x = [theta, alpha, dchi_ax, dchi_rh]
        lb = [-180, -180, -np.inf, -np.inf]
        ub = [180, 180, np.inf, np.inf]
    elif fit_delta_chi:
        # x = [theta, alpha, dchi_ax]
        lb = [-180, -180, -np.inf]
        ub = [180, 180, np.inf]
    elif fit_delta_chi_rh:
        # x = [theta, alpha, dchi_rh]  (Δχ_ax는 고정)
        lb = [-180, -180, -np.inf]
        ub = [180, 180, np.inf]
    else:
        # x = [theta, alpha]
        lb = [-180, -180]
        ub = [180, 180]
    res = _run_two_stage_fit(
        residuals, x0, lb, ub,
        use_global_search=use_global_search,
        progress_cb=progress_cb,
    )

    if fit_delta_chi and fit_delta_chi_rh:
        theta, alpha, dchi_ax, dchi_rh = res.x
    elif fit_delta_chi:
        theta, alpha, dchi_ax = res.x
        dchi_rh = float(state.get('rh_dchi_rh', 0.0) or 0.0)
    elif fit_delta_chi_rh:
        theta, alpha, dchi_rh = res.x
        tensor_entry = state.get('tensor_entry')
        dchi_ax = float((tensor_entry.get() if tensor_entry else None) or state.get('tensor', 1.0) or 1.0)
    else:
        theta, alpha = res.x
        tensor_entry = state.get('tensor_entry')
        dchi_ax = float((tensor_entry.get() if tensor_entry else None) or state.get('tensor', 1.0) or 1.0)
        dchi_rh = float(state.get('rh_dchi_rh', 0.0) or 0.0)

    # 결과 요약
    rot_all = _angles_to_rotation_multi(abs_coords, metal, donor_pts, theta, alpha, axis_mode)

    # φ 기준 고정: UI의 z-rotation(angle_z_var)을 동일 적용
    try:
        az = float(state.get('angle_z_var', 0.0).get())
    except Exception:
        az = 0.0

    if abs(az) > 1e-12:
        coords0 = rot_all - metal
        rot0 = rotate_euler(coords0, 0.0, 0.0, az)
        rot_all = rot0 + metal

    valid_obs_pairs = [(rid, v) for rid, v in obs_pairs if rid in id2idx]
    if not valid_obs_pairs:
        raise RuntimeError("No valid proton ids with δ_exp for current atom set.")

    pts_obs = coords_for_ids(rot_all, [rid for rid, _ in valid_obs_pairs])

    # axial-only vs ax+rh 모델을 residuals()와 동일 조건으로 선택
    if fit_delta_chi_rh or (abs(dchi_rh) > 0.0):
        _, _, _, Gax, Grh = geom_factors_ax_rh(pts_obs, metal)
        dpcs = pcs_ax_rh_from_G(Gax, Grh, dchi_ax, dchi_rh)
    else:
        _, _, _, dpcs = geom_factor_and_pcs(pts_obs, metal, dchi_ax)

    delta_exp = np.array([v for _, v in valid_obs_pairs], float)
    resid = dpcs - delta_exp
    per_point = [(rid, float(exp), float(pred), float(r))
                 for (rid, exp), pred, r in zip(valid_obs_pairs, dpcs, resid)]

    if progress_cb:
        progress_cb("Computing fit quality metrics...")

    metrics = compute_quality_metrics(per_point)

    return dict(
        mode='theta_alpha_multi',
        donor_ids=list(donor_ids),
        axis_mode=axis_mode,
        theta=float(theta),
        alpha=float(alpha),
        delta_chi_ax=float(dchi_ax),
        delta_chi_rh=float(dchi_rh),
        per_point=per_point,
        **metrics,
    )

# ---------- Mode B: 3D Euler rotation (ax, ay, az) ----------
def fit_euler_global(state, proton_ids,
                     fit_visible_as_group=True,
                     fit_delta_chi=False,
                     fit_delta_chi_rh=False,
                     use_global_search=False,
                     progress_cb=None):
    atom_data = state.get("atom_data_eff") or state.get("atom_data") or []
    ids = state.get("atom_ids_eff") or state.get("atom_ids") or []
    if not atom_data or not ids:
        raise RuntimeError("No atom data")
    if progress_cb:
        progress_cb("Preparing Euler fit...")

    id2idx = {rid: i for i, rid in enumerate(ids)}
    metal = np.array([state['x0'], state['y0'], state['z0']])
    abs_coords = np.array([a[1:4] for a in atom_data])

    if fit_visible_as_group:
        state['filter_atoms'](state)
        sel_ids = state.get('current_selected_ids', []) or ids
        group_indices = [id2idx[rid] for rid in sel_ids if rid in id2idx]
    else:
        group_indices = list(range(len(atom_data)))

    delta_exp_values = state.get('delta_exp_values', {})
    obs_pairs = [(rid, delta_exp_values[rid]) for rid in proton_ids if rid in delta_exp_values]
    if len(obs_pairs) < 3 and not fit_delta_chi:
        pass

    # 초기값
    ax0, ay0, az0 = 0.0, 0.0, 0.0
    tensor_entry = state.get('tensor_entry')
    dchi_ax0 = float((tensor_entry.get() if tensor_entry else None) or state.get('tensor', 1.0) or 1.0)
    dchi_rh0 = float(state.get('rh_dchi_rh', 0.0) or 0.0)

    if fit_delta_chi and fit_delta_chi_rh:
        x0 = np.array([ax0, ay0, az0, dchi_ax0, dchi_rh0])
    elif fit_delta_chi:
        x0 = np.array([ax0, ay0, az0, dchi_ax0])
    elif fit_delta_chi_rh:
        # Δχ_ax 고정, Δχ_rh만 fit
        x0 = np.array([ax0, ay0, az0, dchi_rh0])
    else:
        x0 = np.array([ax0, ay0, az0])

    def coords_for_ids(points, ids_subset):
        idxs = [id2idx[rid] for rid in ids_subset if rid in id2idx]
        return np.array([points[i] for i in idxs])

    def residuals(x):
        tensor_entry = state.get('tensor_entry')
        dchi_ax_fixed = float((tensor_entry.get() if tensor_entry else None) or state.get('tensor', 1.0) or 1.0)
        dchi_rh_fixed = float(state.get('rh_dchi_rh', 0.0) or 0.0)

        if fit_delta_chi and fit_delta_chi_rh:
            ax, ay, az, dchi_ax, dchi_rh = x
        elif fit_delta_chi:
            ax, ay, az, dchi_ax = x
            dchi_rh = dchi_rh_fixed
        elif fit_delta_chi_rh:
            ax, ay, az, dchi_rh = x
            dchi_ax = dchi_ax_fixed
        else:
            ax, ay, az = x
            dchi_ax = dchi_ax_fixed
            dchi_rh = dchi_rh_fixed

        # 중심 기준 이동 → 오일러 회전 → 복귀
        coords0 = abs_coords - metal
        rot0 = rotate_euler(coords0, ax, ay, az)
        rot_all = rot0 + metal

        valid_obs_pairs = [(rid, v) for rid, v in obs_pairs if rid in id2idx]
        if not valid_obs_pairs:
            raise RuntimeError("No valid proton ids with δ_exp for current atom set.")

        pts_obs = coords_for_ids(rot_all, [rid for rid, _ in valid_obs_pairs])
        delta_exp = np.array([v for _, v in valid_obs_pairs], float)

        # axial-only vs ax+rh
        if fit_delta_chi_rh or (abs(dchi_rh) > 0.0):
            _, _, _, Gax, Grh = geom_factors_ax_rh(pts_obs, metal)
            dpcs = pcs_ax_rh_from_G(Gax, Grh, dchi_ax, dchi_rh)
        else:
            _, _, _, dpcs = geom_factor_and_pcs(pts_obs, metal, dchi_ax)

        return dpcs - delta_exp

    if fit_delta_chi and fit_delta_chi_rh:
        lb = [-180, -180, -180, -np.inf, -np.inf]
        ub = [180, 180, 180, np.inf, np.inf]
    elif fit_delta_chi:
        lb = [-180, -180, -180, -np.inf]
        ub = [180, 180, 180, np.inf]
    elif fit_delta_chi_rh:
        lb = [-180, -180, -180, -np.inf]  # ax,ay,az,dchi_rh
        ub = [180, 180, 180, np.inf]
    else:
        lb = [-180, -180, -180]
        ub = [180, 180, 180]
    res = _run_two_stage_fit(
        residuals, x0, lb, ub,
        use_global_search=use_global_search,
        progress_cb=progress_cb,
    )

    if fit_delta_chi and fit_delta_chi_rh:
        ax, ay, az, dchi_ax, dchi_rh = res.x
    elif fit_delta_chi:
        ax, ay, az, dchi_ax = res.x
        dchi_rh = float(state.get('rh_dchi_rh', 0.0) or 0.0)
    elif fit_delta_chi_rh:
        ax, ay, az, dchi_rh = res.x
        tensor_entry = state.get('tensor_entry')
        dchi_ax = float((tensor_entry.get() if tensor_entry else None) or state.get('tensor', 1.0) or 1.0)
    else:
        ax, ay, az = res.x
        tensor_entry = state.get('tensor_entry')
        dchi_ax = float((tensor_entry.get() if tensor_entry else None) or state.get('tensor', 1.0) or 1.0)
        dchi_rh = float(state.get('rh_dchi_rh', 0.0) or 0.0)

    # 결과 요약
    coords0 = abs_coords - metal
    rot0 = rotate_euler(coords0, ax, ay, az)
    rot_all = rot0 + metal

    valid_obs_pairs = [(rid, v) for rid, v in obs_pairs if rid in id2idx]
    if not valid_obs_pairs:
        raise RuntimeError("No valid proton ids with δ_exp for current atom set.")

    pts_obs = coords_for_ids(rot_all, [rid for rid, _ in valid_obs_pairs])

    if fit_delta_chi_rh or (abs(dchi_rh) > 0.0):
        _, _, _, Gax, Grh = geom_factors_ax_rh(pts_obs, metal)
        dpcs = pcs_ax_rh_from_G(Gax, Grh, dchi_ax, dchi_rh)
    else:
        _, _, _, dpcs = geom_factor_and_pcs(pts_obs, metal, dchi_ax)

    delta_exp = np.array([v for _, v in valid_obs_pairs], float)
    resid = dpcs - delta_exp
    per_point = [(rid, float(exp), float(pred), float(r))
                 for (rid, exp), pred, r in zip(valid_obs_pairs, dpcs, resid)]

    if progress_cb:
        progress_cb("Computing fit quality metrics...")

    metrics = compute_quality_metrics(per_point)

    return dict(
        mode='euler_global',
        ax=float(ax),
        ay=float(ay),
        az=float(az),
        delta_chi_ax=float(dchi_ax),
        delta_chi_rh=float(dchi_rh),
        per_point=per_point,
        **metrics,
    )

# ---------- Mode C: Fit full tensor ----------
def fit_full_tensor(state, proton_ids,
                    fit_delta_chi=True,
                    fit_delta_chi_rh=False,
                    use_global_search=False,
                    progress_cb=None):
    atom_data = state.get("atom_data_eff") or state.get("atom_data") or []
    ids = state.get("atom_ids_eff") or state.get("atom_ids") or []
    if not atom_data or not ids:
        raise RuntimeError("No atom data")

    if progress_cb:
        progress_cb("Preparing 8-parameter fit...")

    id2idx = {rid: i for i, rid in enumerate(ids)}
    abs_coords = np.array([a[1:4] for a in atom_data], float)

    metal0 = np.array([state['x0'], state['y0'], state['z0']], float)

    delta_exp_values = state.get('delta_exp_values', {})
    obs_pairs = [(rid, delta_exp_values[rid]) for rid in proton_ids if rid in delta_exp_values]
    valid_obs_pairs = [(rid, v) for rid, v in obs_pairs if rid in id2idx]
    if len(valid_obs_pairs) < 3:
        raise RuntimeError("Need at least 3 valid proton points with δ_exp.")

    tensor_entry = state.get('tensor_entry')
    dchi_ax0 = float((tensor_entry.get() if tensor_entry else None) or state.get('tensor', 1.0) or 1.0)
    dchi_rh0 = float(state.get('rh_dchi_rh', 0.0) or 0.0)

    ax0, ay0, az0 = 0.0, 0.0, 0.0
    mx0, my0, mz0 = metal0
    pos_window = 1.0        # metal center +- 1 A range

    if fit_delta_chi and fit_delta_chi_rh:
        x0 = np.array([ax0, ay0, az0, mx0, my0, mz0, dchi_ax0, dchi_rh0], float)
        lb = [-180, -180, -180, mx0 - pos_window, my0 - pos_window, mz0 - pos_window, -np.inf, -np.inf]
        ub = [ 180,  180,  180, mx0 + pos_window, my0 + pos_window, mz0 + pos_window,  np.inf,  np.inf]
    elif fit_delta_chi:
        x0 = np.array([ax0, ay0, az0, mx0, my0, mz0, dchi_ax0], float)
        lb = [-180, -180, -180, mx0 - pos_window, my0 - pos_window, mz0 - pos_window, -np.inf]
        ub = [ 180,  180,  180, mx0 + pos_window, my0 + pos_window, mz0 + pos_window,  np.inf]
    elif fit_delta_chi_rh:
        x0 = np.array([ax0, ay0, az0, mx0, my0, mz0, dchi_rh0], float)
        lb = [-180, -180, -180, mx0 - pos_window, my0 - pos_window, mz0 - pos_window, -np.inf]
        ub = [ 180,  180,  180, mx0 + pos_window, my0 + pos_window, mz0 + pos_window,  np.inf]
    else:
        x0 = np.array([ax0, ay0, az0, mx0, my0, mz0], float)
        lb = [-180, -180, -180, mx0 - pos_window, my0 - pos_window, mz0 - pos_window]
        ub = [ 180,  180,  180, mx0 + pos_window, my0 + pos_window, mz0 + pos_window,]

    def coords_for_ids(points, ids_subset):
        idxs = [id2idx[rid] for rid in ids_subset if rid in id2idx]
        return np.array([points[i] for i in idxs], float)

    def unpack_params(x):
        dchi_ax_fixed = float((tensor_entry.get() if tensor_entry else None) or state.get('tensor', 1.0) or 1.0)
        dchi_rh_fixed = float(state.get('rh_dchi_rh', 0.0) or 0.0)

        if fit_delta_chi and fit_delta_chi_rh:
            ax, ay, az, mx, my, mz, dchi_ax, dchi_rh = x
        elif fit_delta_chi:
            ax, ay, az, mx, my, mz, dchi_ax = x
            dchi_rh = dchi_rh_fixed
        elif fit_delta_chi_rh:
            ax, ay, az, mx, my, mz, dchi_rh = x
            dchi_ax = dchi_ax_fixed
        else:
            ax, ay, az, mx, my, mz = x
            dchi_ax = dchi_ax_fixed
            dchi_rh = dchi_rh_fixed

        metal = np.array([mx, my, mz], float)
        return ax, ay, az, metal, dchi_ax, dchi_rh

    def residuals(x):
        ax, ay, az, metal, dchi_ax, dchi_rh = unpack_params(x)

        coords0 = abs_coords - metal
        rot0 = rotate_euler(coords0, ax, ay, az)
        rot_all = rot0 + metal

        pts_obs = coords_for_ids(rot_all, [rid for rid, _ in valid_obs_pairs])
        delta_exp = np.array([v for _, v in valid_obs_pairs], float)

        if fit_delta_chi_rh or abs(dchi_rh) > 0.0:
            _, _, _, Gax, Grh = geom_factors_ax_rh(pts_obs, metal)
            dpcs = pcs_ax_rh_from_G(Gax, Grh, dchi_ax, dchi_rh)
        else:
            _, _, _, dpcs = geom_factor_and_pcs(pts_obs, metal, dchi_ax)

        return dpcs - delta_exp

    res = _run_two_stage_fit(
        residuals, x0, lb, ub,
        use_global_search=use_global_search,
        progress_cb=progress_cb,
    )

    ax, ay, az, metal, dchi_ax, dchi_rh = unpack_params(res.x)

    coords0 = abs_coords - metal
    rot0 = rotate_euler(coords0, ax, ay, az)
    rot_all = rot0 + metal

    pts_obs = coords_for_ids(rot_all, [rid for rid, _ in valid_obs_pairs])
    delta_exp = np.array([v for _, v in valid_obs_pairs], float)

    if fit_delta_chi_rh or abs(dchi_rh) > 0.0:
        _, _, _, Gax, Grh = geom_factors_ax_rh(pts_obs, metal)
        dpcs = pcs_ax_rh_from_G(Gax, Grh, dchi_ax, dchi_rh)
    else:
        _, _, _, dpcs = geom_factor_and_pcs(pts_obs, metal, dchi_ax)

    resid = dpcs - delta_exp
    per_point = [
        (rid, float(exp), float(pred), float(r))
        for (rid, exp), pred, r in zip(valid_obs_pairs, dpcs, resid)
    ]

    if progress_cb:
        progress_cb("Computing fit quality metrics...")

    metrics = compute_quality_metrics(per_point)

    return dict(
        mode='full_tensor',
        euler_deg=(float(ax), float(ay), float(az)),
        metal_pos=(float(metal[0]), float(metal[1]), float(metal[2])),
        delta_chi_ax=float(dchi_ax),
        delta_chi_rh=float(dchi_rh),
        per_point=per_point,
        **metrics,
    )


# ---------- UI wiring helpers ----------
def populate_fitting_controls(state):
    # Prefer effective data for 2D/table/fitting consistency
    atom_data = state.get("atom_data_eff") or state.get("atom_data") or []
    ids = state.get("atom_ids_eff") or state.get("atom_ids") or []

    if not atom_data or not ids:
        return

    # Donor list: exclude H
    items = [f"{rid}:{atom}" for rid, (atom, *_ ) in zip(ids, atom_data) if atom != 'H']
    lst = state.get('fit_donor_list')
    if lst:
        lst.delete(0, "end")
        for s in items:
            lst.insert("end", s)

    # Proton list: visible atoms among current selection, H only
    state['filter_atoms'](state)
    sel_ids = state.get('current_selected_ids', []) or ids

    id2idx = {rid: i for i, rid in enumerate(ids)}
    proton_ids = []
    for rid in sel_ids:
        i = id2idx.get(rid)
        if i is None:
            continue
        if atom_data[i][0] == 'H':
            proton_ids.append(rid)

    pl = state.get('fit_proton_list')
    if pl:
        pl.delete(0, "end")
        for rid in proton_ids:
            pl.insert("end", str(rid))

def apply_fit_to_views(state):
    res = state.get('last_fit_result')
    if not res:
        return

    mode = res.get('mode')

    if mode == 'theta_alpha_multi':
        state['fit_override'] = dict(
            mode='theta_alpha_multi',
            donor_ids=res['donor_ids'],
            theta=res['theta'],
            alpha=res['alpha'],
            axis_mode=res.get('axis_mode', 'bisector')
        )
        state['messagebox'].showinfo("Fitting", "Mode A applied. Click 'Update' to redraw.")

    elif mode == 'euler_global':
        state['fit_override'] = dict(
            mode='euler_global',
            ax=res['ax'],
            ay=res['ay'],
            az=res['az']
        )
        state['messagebox'].showinfo("Fitting", "Mode B applied. Click 'Update' to redraw.")

    elif mode == 'full_tensor':
        state['fit_override'] = dict(
            mode='full_tensor',
            euler_deg=res['euler_deg'],
            metal_pos=res['metal_pos'],
            dchi_ax=res['delta_chi_ax'],
            dchi_rh=res.get('delta_chi_rh', 0.0)
        )

        try:
            state['tensor_entry'].delete(0, "end")
            state['tensor_entry'].insert(0, f"{float(res['delta_chi_ax']):g}")
        except Exception:
            pass

        if 'rh_dchi_rh_var' in state:
            try:
                state['rh_dchi_rh_var'].set(f"{float(res.get('delta_chi_rh', 0.0)):g}")
                state['rh_dchi_rh'] = float(res.get('delta_chi_rh', 0.0))
            except Exception:
                pass

        state['messagebox'].showinfo("Fitting", "Mode C applied. Click 'Update' to redraw.")