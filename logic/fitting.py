# logic/fitting.py
# PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE3

import numpy as np
from typing import Mapping
from scipy.optimize import least_squares, differential_evolution
from logic.pcs_forward_api import AngleUnit, EulerConvention
from logic.pcs_fitting_adapter import (
    predict_pcs_from_factors,
    predict_structured_pcs_from_factors,
)
from logic.pcs_orientation import matrix_to_euler, orientation_matrix
from logic.pcs_fit_result_schema import annotate_standard_fit_result
from logic.pcs_fit_fingerprint import (
    fingerprint_fit_input_snapshot,
    fit_input_reconstruction_config,
)
from logic.pcs_fit_snapshot import (
    FitGeometrySnapshot,
    FitObservation,
    FitOptimizerSnapshot,
    FitTensorSnapshot,
    StandardFitInputSnapshot,
    snapshot_torsion_group,
)
from logic.pcs_tensor_state import ApplicationPCSTensor, PCS_TENSOR_STATE_KEY
from logic.pcs_fit_weighting import FitWeightingMode, parse_fit_weighting_mode
from logic.pcs_structured_observations import (
    OBSERVATIONS_KEY,
    canonical_provenance_json,
    finite_pcs,
    structure_target_from_payload,
)
from logic.rotate_align import rotate_euler
from logic.symmetry_geometry import effective_geometry_factors
from logic.visibility import visible_ref_id_set


_UNSET = object()


def _raw_structure_arrays(state):
    """Return raw Ref IDs and absolute raw coordinates."""
    raw = state.get("atom_data_raw") or state.get("atom_data") or []
    raw_ids = state.get("atom_ids_raw") or list(range(1, len(raw) + 1))
    raw_coords = np.array([a[1:4] for a in raw], dtype=float) if raw else np.empty((0, 3), dtype=float)
    return list(raw_ids), raw_coords


def _effective_obs_factors(
    geometry, obs_ids, obs_coords, metal, raw_ids, raw_transformed
):
    """Geometry factors for observations, with member averaging for pseudo atoms."""
    raw_map = {
        rid: np.asarray(coord, dtype=float)
        for rid, coord in zip(raw_ids, raw_transformed)
    } if raw_transformed is not None else {}
    return effective_geometry_factors(
        obs_ids,
        obs_coords,
        metal,
        pseudo_members=geometry.pseudo_member_mapping(),
        raw_coords_by_id=raw_map,
        torsion_groups=geometry.active_torsion_mappings(),
    )


def _pcs_from_effective_factors(Gax, Grh, dchi_ax, dchi_rh=0.0):
    return predict_pcs_from_factors(Gax, Grh, dchi_ax, dchi_rh)


def _legacy_fit_tensor_guesses(state):
    """Return the established fitting guesses without consulting canonical state."""
    tensor_entry = state.get('tensor_entry')
    dchi_ax = float(
        (tensor_entry.get() if tensor_entry else None)
        or state.get('tensor', 1.0)
        or 1.0
    )
    dchi_rh = float(state.get('rh_dchi_rh', 0.0) or 0.0)
    return dchi_ax, dchi_rh


def standard_fit_initial_values(
    state,
    *,
    mode,
    use_current_tensor=False,
    legacy_values=None,
    canonical_tensor=_UNSET,
):
    """Return ``(xyz_degrees, dchi_ax, dchi_rh)`` for a standard fit.

    With the opt-in disabled this exactly preserves the established zero-angle
    and legacy tensor guesses. The opt-in is intentionally supported only by
    the matrix-parameterized Euler and Full tensor fit modes. Canonical Euler
    coordinates are never copied: their physical rotation matrix is
    deterministically decomposed into the fitter's active extrinsic XYZ
    convention.

    Mode B retains its established rhombic guess. Mode C uses the canonical
    rhombic value only when rhombicity is enabled; otherwise it also retains
    the legacy/default guess. This helper reads but never replaces or updates
    canonical tensor state.
    """
    if legacy_values is None:
        legacy_ax, legacy_rh = _legacy_fit_tensor_guesses(state)
    else:
        legacy_ax, legacy_rh = (float(value) for value in legacy_values)
    legacy = ((0.0, 0.0, 0.0), legacy_ax, legacy_rh)
    if not use_current_tensor:
        return legacy
    if mode not in ('euler_global', 'full_tensor'):
        return legacy

    canonical = (
        state.get(PCS_TENSOR_STATE_KEY)
        if canonical_tensor is _UNSET
        else canonical_tensor
    )
    if not isinstance(canonical, ApplicationPCSTensor):
        raise RuntimeError("Canonical PCS tensor state is not available.")

    matrix = orientation_matrix(
        canonical.euler_angles,
        canonical.frame_convention,
        canonical.angle_unit,
    )
    xyz_degrees = matrix_to_euler(
        matrix,
        EulerConvention.EXTRINSIC_XYZ,
        AngleUnit.DEGREE,
    )
    dchi_rh = legacy_rh
    if mode == 'full_tensor' and canonical.rhombic_enabled:
        dchi_rh = canonical.delta_chi_rh
    return xyz_degrees, canonical.delta_chi_ax, dchi_rh

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
    dpcs = predict_pcs_from_factors(Gi, np.zeros_like(Gi), delta_chi_ax)
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
    return predict_pcs_from_factors(Gax, Grh, dchi_ax, dchi_rh)

# ---------- Differential evolution (DE function helper) ----------
def _make_de_bounds_finite(lb, ub, x0=None, default_span=50.0):
    """
    Convert possibly infinite bounds into finite bounds for differential_evolution.
    least_squares can handle inf bounds, but DE cannot.

    A dimension unbounded on both sides uses ``x0 +/- default_span``. Thus an
    opt-in current-tensor guess moves the center of that temporary DE interval
    without changing its width or this fallback rule.
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
                       progress_cb=None,
                       effective_de_bounds=None,
                       de_seed=42,
                       de_polish=False):
    """
    Optional global search (DE) followed by local least-squares refinement.
    DE requires finite bounds, so infinite bounds are converted temporarily.
    """
    if use_global_search:
        if effective_de_bounds is None:
            lb_de, ub_de = _make_de_bounds_finite(
                lb, ub, x0=x0, default_span=50.0
            )
            bounds_de = list(zip(lb_de, ub_de))
        else:
            bounds_de = [tuple(bound) for bound in effective_de_bounds]

        if progress_cb:
            progress_cb("Running global search (DE)...")

        def objective(x):
            r = residual_func(x)
            return float(np.sum(np.asarray(r, float) ** 2))

        de_res = differential_evolution(
            objective,
            bounds=bounds_de,
            polish=de_polish,
            seed=de_seed,
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


def _read_state_value(value, default):
    if value is None:
        return default
    try:
        value = value.get()
    except AttributeError:
        pass
    return default if value is None else value


def _fit_observations(
    state, *, proton_ids, observation_ids, available_ref_ids, visible_ref_ids=None
):
    """Normalize structured or legacy inputs into one observable-level model."""
    records = state.get(OBSERVATIONS_KEY, ()) or ()
    if records:
        if not isinstance(records, (list, tuple)):
            raise ValueError("structured PCS observations must be a sequence")
        by_id = {}
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"structured PCS observation {index} is malformed")
            observable_id = str(record.get("id", "")).strip()
            if not observable_id:
                raise ValueError(f"structured PCS observation {index} has no stable id")
            if observable_id in by_id:
                raise ValueError(f"duplicate structured PCS observation id {observable_id!r}")
            by_id[observable_id] = record
        if observation_ids is None:
            selected = list(by_id.items())
        else:
            selected = []
            seen_selected = set()
            for value in observation_ids:
                key = str(value)
                if key in seen_selected:
                    raise ValueError(f"duplicate selected PCS observation id {key!r}")
                if key not in by_id:
                    raise ValueError(f"structured PCS observation {key!r} is unavailable")
                selected.append((key, by_id[key]))
                seen_selected.add(key)
        observations = []
        available = set(available_ref_ids)
        visible = None if visible_ref_ids is None else set(visible_ref_ids)
        explicit_selection = observation_ids is not None
        for observable_id, record in selected:
            pcs = finite_pcs(record.get("pcs_ppm"))
            assignment = str(record.get("assignment", "") or "")
            uncertainty = record.get("uncertainty_ppm")
            target = structure_target_from_payload(
                record.get("target"),
                assignment=assignment,
                target_id=observable_id,
                experimental_pcs_ppm=pcs,
                uncertainty_ppm=None if uncertainty is None else float(uncertainty),
            )
            missing = sorted(set(target.member_ref_ids) - available)
            if missing:
                raise ValueError(
                    f"structured PCS observation {observable_id!r} is missing structure Ref "
                    + ", ".join(map(str, missing))
                )
            if visible is not None and not set(target.member_ref_ids).issubset(visible):
                if explicit_selection:
                    raise ValueError(
                        f"structured PCS observation {observable_id!r} is hidden by the current visibility filter"
                    )
                continue
            primary = record.get("target", {}).get("primary_ref_id")
            if primary is None:
                primary_ref = target.member_ref_ids[0]
            else:
                try:
                    primary_ref = int(primary)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"structured PCS observation {observable_id!r} has an invalid primary Ref"
                    ) from exc
                if primary_ref not in target.member_ref_ids:
                    raise ValueError(
                        f"structured PCS observation {observable_id!r} primary Ref is not a target member"
                    )
            observations.append(FitObservation(
                ref_id=primary_ref,
                pcs=pcs,
                observable_id=observable_id,
                assignment=assignment,
                target=target,
                source="structured_v1",
                source_dataset_ids=tuple(record.get("source_dataset_ids", ()) or ()),
                source_observable_ids=tuple(record.get("source_observable_ids", ()) or ()),
                provenance_json=canonical_provenance_json(record.get("provenance", {})),
                uncertainty_ppm=target.uncertainty_ppm,
            ))
        if not observations:
            raise RuntimeError("No structured PCS observations selected for fitting.")
        return tuple(observations)

    delta_exp_values = state.get("delta_exp_values", {}) or {}
    visible = None if visible_ref_ids is None else set(int(value) for value in visible_ref_ids)
    observations = tuple(
        FitObservation(int(ref_id), float(delta_exp_values[ref_id]))
        for ref_id in proton_ids
        if ref_id in delta_exp_values
        and int(ref_id) in available_ref_ids
        and (visible is None or int(ref_id) in visible)
    )
    if not observations:
        raise RuntimeError("No valid proton ids with δ_exp for current atom set.")
    return observations


def _predict_snapshot_observations(
    snapshot, effective_rotated, raw_rotated, metal, dchi_ax, dchi_rh
):
    """Predict one value per snapshot observation at current fit geometry."""
    effective_map = {
        ref_id: np.asarray(coord, dtype=float)
        for ref_id, coord in zip(snapshot.geometry.effective_ref_ids, effective_rotated)
    }
    raw_map = {
        ref_id: np.asarray(coord, dtype=float)
        for ref_id, coord in zip(snapshot.geometry.raw_ref_ids, raw_rotated)
    }
    member_ids = tuple(dict.fromkeys(
        ref_id
        for observation in snapshot.observations
        for ref_id in observation.target.member_ref_ids
    ))
    positions = []
    for ref_id in member_ids:
        coordinate = effective_map.get(ref_id)
        if coordinate is None:
            coordinate = raw_map.get(ref_id)
        if coordinate is None:
            raise ValueError(f"fit target is missing structure Ref {ref_id}")
        positions.append(coordinate)
    _, _, _, g_ax, g_rh = _effective_obs_factors(
        snapshot.geometry,
        member_ids,
        np.asarray(positions, dtype=float),
        np.asarray(metal, dtype=float),
        snapshot.geometry.raw_ref_ids,
        raw_rotated,
    )
    return predict_structured_pcs_from_factors(
        member_ids,
        g_ax,
        g_rh,
        dchi_ax,
        dchi_rh,
        tuple(observation.target for observation in snapshot.observations),
    )


def _fit_result_rows(snapshot, predicted):
    observed = np.asarray([item.pcs for item in snapshot.observations], dtype=float)
    residual = np.asarray(predicted, dtype=float) - observed
    per_point = [
        (item.ref_id, float(exp), float(pred), float(resid))
        for item, exp, pred, resid in zip(snapshot.observations, observed, predicted, residual)
    ]
    per_observable = []
    for item, exp, pred, resid in zip(snapshot.observations, observed, predicted, residual):
        target = item.target
        per_observable.append({
            "observable_id": item.observable_id,
            "assignment": item.assignment,
            "primary_ref_id": item.ref_id,
            "source": item.source,
            "source_dataset_ids": list(item.source_dataset_ids),
            "source_observable_ids": list(item.source_observable_ids),
            "target_id": target.target_id,
            "scope": target.metadata.get("scope"),
            "site_member_ref_ids": [list(group) for group in target.site_member_ref_ids],
            "site_ids": list(target.metadata.get("site_ids", ())),
            "averaging_hint": target.metadata.get("averaging_hint"),
            "matching_status": target.metadata.get("matching_status"),
            "observed": float(exp),
            "predicted": float(pred),
            "residual": float(resid),
            "uncertainty_ppm": item.uncertainty_ppm,
            "weighted_residual": (
                float(resid) / float(item.uncertainty_ppm)
                if snapshot.weighting_mode
                == FitWeightingMode.EXPERIMENTAL_UNCERTAINTY.value
                else None
            ),
        })
    return observed, residual, per_point, per_observable


def _optimizer_residuals(snapshot, raw_residual):
    raw = np.asarray(raw_residual, dtype=float)
    if snapshot.weighting_mode == FitWeightingMode.UNWEIGHTED.value:
        return raw
    sigma = np.asarray(
        [item.uncertainty_ppm for item in snapshot.observations], dtype=float
    )
    return raw / sigma


def _weighted_quality_metrics(snapshot, raw_residual):
    if snapshot.weighting_mode == FitWeightingMode.UNWEIGHTED.value:
        return {}
    weighted = _optimizer_residuals(snapshot, raw_residual)
    chi_square = float(np.sum(weighted ** 2))
    return {
        "chi_square": chi_square,
        "chi_square_per_datum": float(chi_square / len(weighted)),
    }


def build_standard_fit_input_snapshot(
    state,
    *,
    mode,
    proton_ids=(),
    observation_ids=None,
    donor_ids=(),
    axis_mode=None,
    fit_delta_chi=False,
    fit_delta_chi_rh=False,
    use_global_search=False,
    use_current_tensor=False,
    weighting_mode=FitWeightingMode.UNWEIGHTED.value,
):
    """Freeze every scientific input consumed by one standard fit run."""
    atom_data = state.get("atom_data_eff") or state.get("atom_data") or []
    ids = (
        state.get("atom_ids_eff")
        or state.get("atom_ids")
        or state.get("atom_ids_raw")
        or []
    )
    if not atom_data or not ids:
        raise RuntimeError("No atom data")
    effective_ids = tuple(int(ref_id) for ref_id in ids)
    effective_coordinates = tuple(
        (float(atom[1]), float(atom[2]), float(atom[3]))
        for atom in atom_data
    )
    id2coord = dict(zip(effective_ids, effective_coordinates))

    raw_ids_list, raw_coordinates_array = _raw_structure_arrays(state)
    raw_ids = tuple(int(ref_id) for ref_id in raw_ids_list)
    raw_coordinates = tuple(
        tuple(float(value) for value in row)
        for row in raw_coordinates_array
    )
    effective_visible_ids = visible_ref_id_set(
        state, atom_data=atom_data, ref_ids=ids
    )
    raw_atom_data = state.get("atom_data_raw") or state.get("atom_data") or []
    raw_visible_ids = visible_ref_id_set(
        state, atom_data=raw_atom_data, ref_ids=raw_ids_list
    )
    fit_visible_ids = set(effective_visible_ids) | set(raw_visible_ids)
    observations = _fit_observations(
        state,
        proton_ids=proton_ids,
        observation_ids=observation_ids,
        available_ref_ids=set(effective_ids) | set(raw_ids),
        visible_ref_ids=fit_visible_ids,
    )
    normalized_weighting = parse_fit_weighting_mode(weighting_mode)
    if normalized_weighting is FitWeightingMode.EXPERIMENTAL_UNCERTAINTY:
        invalid = [
            item.observable_id
            for item in observations
            if item.uncertainty_ppm is None
            or not np.isfinite(item.uncertainty_ppm)
            or item.uncertainty_ppm <= 0.0
        ]
        if invalid:
            preview = ", ".join(invalid[:5])
            suffix = "" if len(invalid) <= 5 else f" (+{len(invalid) - 5} more)"
            raise ValueError(
                "Experimental-uncertainty weighting requires finite positive "
                f"uncertainty_ppm for every observation; invalid {len(invalid)}: "
                f"{preview}{suffix}"
            )
    if mode == "full_tensor" and len(observations) < 3:
        raise RuntimeError("Need at least 3 PCS observations.")
    metal = (
        float(state["x0"]),
        float(state["y0"]),
        float(state["z0"]),
    )
    pseudo_members = tuple(
        sorted(
            (
                int(ref_id),
                tuple(int(member) for member in members),
            )
            for ref_id, members in (
                state.get("symavg_members_by_pseudo_id", {}) or {}
            ).items()
        )
    )
    torsion_enabled = bool(
        _read_state_value(state.get("torsion_avg_enabled_var"), False)
    )
    torsion_groups = tuple(
        snapshot_torsion_group(group)
        for group in (state.get("torsion_avg_groups", []) or [])
    )
    geometry = FitGeometrySnapshot(
        effective_ref_ids=effective_ids,
        effective_coordinates=effective_coordinates,
        raw_ref_ids=raw_ids,
        raw_coordinates=raw_coordinates,
        metal_position=metal,
        pseudo_members=pseudo_members,
        torsion_enabled=torsion_enabled,
        torsion_groups=torsion_groups,
    )

    legacy_ax, legacy_rh = _legacy_fit_tensor_guesses(state)
    canonical_tensor = state.get(PCS_TENSOR_STATE_KEY)
    donor_ids_tuple = tuple(int(ref_id) for ref_id in donor_ids)
    donor_coordinates = tuple(
        id2coord[ref_id] for ref_id in donor_ids_tuple if ref_id in id2coord
    )
    if mode == "theta_alpha_multi":
        if not donor_coordinates:
            raise RuntimeError("No valid donor ids for current atom set.")
        xyz0 = (0.0, 0.0, 0.0)
        dchi_ax0, dchi_rh0 = legacy_ax, legacy_rh
        angle_z = float(_read_state_value(state.get("angle_z_var"), 0.0))
    elif mode in {"euler_global", "full_tensor"}:
        xyz0, dchi_ax0, dchi_rh0 = standard_fit_initial_values(
            state,
            mode=mode,
            use_current_tensor=use_current_tensor,
            legacy_values=(legacy_ax, legacy_rh),
            canonical_tensor=canonical_tensor,
        )
        angle_z = 0.0
    else:
        raise ValueError(f"Unknown fitting mode: {mode}")

    ax0, ay0, az0 = (float(value) for value in xyz0)
    if mode == "theta_alpha_multi":
        base_x0 = [0.0, 0.0]
        base_lb = [-180.0, -180.0]
        base_ub = [180.0, 180.0]
        free_parameters = ["theta", "alpha"]
    elif mode == "euler_global":
        base_x0 = [ax0, ay0, az0]
        base_lb = [-180.0, -180.0, -180.0]
        base_ub = [180.0, 180.0, 180.0]
        free_parameters = ["ax", "ay", "az"]
    else:
        mx0, my0, mz0 = metal
        pos_window = 1.0
        base_x0 = [ax0, ay0, az0, mx0, my0, mz0]
        base_lb = [
            -180.0, -180.0, -180.0,
            mx0 - pos_window, my0 - pos_window, mz0 - pos_window,
        ]
        base_ub = [
            180.0, 180.0, 180.0,
            mx0 + pos_window, my0 + pos_window, mz0 + pos_window,
        ]
        free_parameters = [
            "ax", "ay", "az", "metal_x", "metal_y", "metal_z",
        ]

    if fit_delta_chi:
        base_x0.append(float(dchi_ax0))
        base_lb.append(-np.inf)
        base_ub.append(np.inf)
        free_parameters.append("delta_chi_ax")
    if fit_delta_chi_rh:
        base_x0.append(float(dchi_rh0))
        base_lb.append(-np.inf)
        base_ub.append(np.inf)
        free_parameters.append("delta_chi_rh")

    de_bounds = None
    if use_global_search:
        de_lb, de_ub = _make_de_bounds_finite(
            base_lb, base_ub, x0=base_x0, default_span=50.0
        )
        de_bounds = tuple(zip(de_lb, de_ub))
    optimizer = FitOptimizerSnapshot(
        identity=(
            "differential_evolution_then_least_squares"
            if use_global_search else "least_squares"
        ),
        use_global_search=bool(use_global_search),
        local_optimizer="least_squares",
        de_seed=42 if use_global_search else None,
        de_polish=False if use_global_search else None,
        de_default_span=50.0 if use_global_search else None,
        effective_de_bounds=de_bounds,
    )
    return StandardFitInputSnapshot(
        mode=mode,
        observations=observations,
        geometry=geometry,
        donor_ids=donor_ids_tuple,
        donor_coordinates=donor_coordinates,
        axis_mode=str(axis_mode or "bisector") if mode == "theta_alpha_multi" else None,
        mode_a_angle_z=angle_z,
        use_current_tensor=bool(use_current_tensor),
        tensor=FitTensorSnapshot(
            fit_delta_chi_ax=bool(fit_delta_chi),
            fit_delta_chi_rh=bool(fit_delta_chi_rh),
            fixed_delta_chi_ax=float(legacy_ax),
            fixed_delta_chi_rh=float(legacy_rh),
        ),
        free_parameters=tuple(free_parameters),
        x0=tuple(float(value) for value in base_x0),
        lower_bounds=tuple(float(value) for value in base_lb),
        upper_bounds=tuple(float(value) for value in base_ub),
        optimizer=optimizer,
        weighting_mode=normalized_weighting.value,
    )

# ---------- Mode A: θ, α (multi-donor) ----------
def fit_theta_alpha_multi(state, donor_ids, proton_ids=(),
                          axis_mode='bisector',
                          fit_visible_as_group=True,
                          fit_delta_chi=False,
                          fit_delta_chi_rh=False,
                          use_global_search=False,
                          progress_cb=None,
                          observation_ids=None,
                          weighting_mode=FitWeightingMode.UNWEIGHTED.value):
    """
    donor_ids: Ref ID 리스트(1..N)
    proton_ids: δ_Exp가 들어있는 Ref ID 목록 (보통 H들)
    axis_mode: 'first' | 'bisector' | 'average' | 'centroid' | 'normal' | 'pca'
    """
    if progress_cb:
        progress_cb("Preparing donor-axis fit...")
    snapshot = build_standard_fit_input_snapshot(
        state,
        mode="theta_alpha_multi",
        proton_ids=proton_ids,
        observation_ids=observation_ids,
        donor_ids=donor_ids,
        axis_mode=axis_mode,
        fit_delta_chi=fit_delta_chi,
        fit_delta_chi_rh=fit_delta_chi_rh,
        use_global_search=use_global_search,
        weighting_mode=weighting_mode,
    )
    if fit_visible_as_group and callable(state.get("filter_atoms")):
        state["filter_atoms"](state)
    ids = snapshot.geometry.effective_ref_ids
    metal = np.asarray(snapshot.geometry.metal_position, dtype=float)
    abs_coords = snapshot.geometry.effective_array()
    raw_ids = snapshot.geometry.raw_ref_ids
    raw_abs_coords = snapshot.geometry.raw_array()
    donor_pts = np.asarray(snapshot.donor_coordinates, dtype=float)
    x0 = snapshot.x0_array()
    lb = snapshot.lower_bounds
    ub = snapshot.upper_bounds

    def unpack_params(x):
        dchi_ax = snapshot.tensor.fixed_delta_chi_ax
        dchi_rh = snapshot.tensor.fixed_delta_chi_rh
        if snapshot.tensor.fit_delta_chi_ax and snapshot.tensor.fit_delta_chi_rh:
            theta, alpha, dchi_ax, dchi_rh = x
        elif snapshot.tensor.fit_delta_chi_ax:
            theta, alpha, dchi_ax = x
        elif snapshot.tensor.fit_delta_chi_rh:
            theta, alpha, dchi_rh = x
        else:
            theta, alpha = x
        return theta, alpha, dchi_ax, dchi_rh

    def rotated_geometry(theta, alpha):
        rot_all = _angles_to_rotation_multi(
            points=abs_coords,
            metal=metal,
            donor_points=donor_pts,
            theta_deg=theta, alpha_deg=alpha,
            axis_mode=snapshot.axis_mode,
        )
        raw_rot_all = _angles_to_rotation_multi(
            points=raw_abs_coords,
            metal=metal,
            donor_points=donor_pts,
            theta_deg=theta, alpha_deg=alpha,
            axis_mode=snapshot.axis_mode,
        ) if len(raw_abs_coords) else raw_abs_coords
        az = snapshot.mode_a_angle_z
        if abs(az) > 1e-12:
            coords0 = rot_all - metal
            rot0 = rotate_euler(coords0, 0.0, 0.0, az)
            rot_all = rot0 + metal
            if len(raw_rot_all):
                raw0 = raw_rot_all - metal
                raw_rot_all = rotate_euler(raw0, 0.0, 0.0, az) + metal
        return rot_all, raw_rot_all

    delta_exp = np.array([item.pcs for item in snapshot.observations], dtype=float)

    def residuals(x):
        theta, alpha, dchi_ax, dchi_rh = unpack_params(x)
        rot_all, raw_rot_all = rotated_geometry(theta, alpha)
        dpcs = _predict_snapshot_observations(
            snapshot, rot_all, raw_rot_all, metal,
            dchi_ax,
            dchi_rh if (snapshot.tensor.fit_delta_chi_rh or abs(dchi_rh) > 0.0) else 0.0,
        )
        return _optimizer_residuals(snapshot, dpcs - delta_exp)

    res = _run_two_stage_fit(
        residuals, x0, lb, ub,
        use_global_search=snapshot.optimizer.use_global_search,
        progress_cb=progress_cb,
        effective_de_bounds=snapshot.optimizer.effective_de_bounds,
        de_seed=snapshot.optimizer.de_seed,
        de_polish=snapshot.optimizer.de_polish,
    )
    theta, alpha, dchi_ax, dchi_rh = unpack_params(res.x)
    rot_all, raw_rot_all = rotated_geometry(theta, alpha)
    dpcs = _predict_snapshot_observations(
        snapshot, rot_all, raw_rot_all, metal,
        dchi_ax,
        dchi_rh if (snapshot.tensor.fit_delta_chi_rh or abs(dchi_rh) > 0.0) else 0.0,
    )
    _, resid, per_point, per_observable = _fit_result_rows(snapshot, dpcs)

    if progress_cb:
        progress_cb("Computing fit quality metrics...")

    metrics = compute_quality_metrics(per_point)
    metrics.update(_weighted_quality_metrics(snapshot, resid))

    return annotate_standard_fit_result(
        dict(
            mode='theta_alpha_multi',
            donor_ids=list(snapshot.donor_ids),
            axis_mode=snapshot.axis_mode,
            theta=float(theta),
            alpha=float(alpha),
            delta_chi_ax=float(dchi_ax),
            delta_chi_rh=float(dchi_rh),
            per_point=per_point,
            per_observable=per_observable,
            **metrics,
        ),
        mode='theta_alpha_multi',
        use_global_search=snapshot.optimizer.use_global_search,
        free_parameters=snapshot.free_parameters,
        optimizer=snapshot.optimizer.identity,
        weighting_mode=snapshot.weighting_mode,
        input_fingerprint=fingerprint_fit_input_snapshot(snapshot),
        input_configuration=fit_input_reconstruction_config(snapshot),
    )

# ---------- Mode B: 3D Euler rotation (ax, ay, az) ----------
def fit_euler_global(state, proton_ids=(),
                     fit_visible_as_group=True,
                     fit_delta_chi=False,
                     fit_delta_chi_rh=False,
                     use_global_search=False,
                     progress_cb=None,
                     use_current_tensor=False,
                     observation_ids=None,
                     weighting_mode=FitWeightingMode.UNWEIGHTED.value):
    if progress_cb:
        progress_cb("Preparing Euler fit...")
    snapshot = build_standard_fit_input_snapshot(
        state,
        mode="euler_global",
        proton_ids=proton_ids,
        observation_ids=observation_ids,
        fit_delta_chi=fit_delta_chi,
        fit_delta_chi_rh=fit_delta_chi_rh,
        use_global_search=use_global_search,
        use_current_tensor=use_current_tensor,
        weighting_mode=weighting_mode,
    )
    if fit_visible_as_group and callable(state.get("filter_atoms")):
        state["filter_atoms"](state)
    ids = snapshot.geometry.effective_ref_ids
    metal = np.asarray(snapshot.geometry.metal_position, dtype=float)
    abs_coords = snapshot.geometry.effective_array()
    raw_ids = snapshot.geometry.raw_ref_ids
    raw_abs_coords = snapshot.geometry.raw_array()
    delta_exp = np.asarray([item.pcs for item in snapshot.observations], dtype=float)
    x0 = snapshot.x0_array()

    def unpack_params(x):
        dchi_ax = snapshot.tensor.fixed_delta_chi_ax
        dchi_rh = snapshot.tensor.fixed_delta_chi_rh
        if snapshot.tensor.fit_delta_chi_ax and snapshot.tensor.fit_delta_chi_rh:
            ax, ay, az, dchi_ax, dchi_rh = x
        elif snapshot.tensor.fit_delta_chi_ax:
            ax, ay, az, dchi_ax = x
        elif snapshot.tensor.fit_delta_chi_rh:
            ax, ay, az, dchi_rh = x
        else:
            ax, ay, az = x
        return ax, ay, az, dchi_ax, dchi_rh

    def rotated_geometry(ax, ay, az):
        coords0 = abs_coords - metal
        rot0 = rotate_euler(coords0, ax, ay, az)
        rot_all = rot0 + metal
        if len(raw_abs_coords):
            raw0 = raw_abs_coords - metal
            raw_rot_all = rotate_euler(raw0, ax, ay, az) + metal
        else:
            raw_rot_all = raw_abs_coords
        return rot_all, raw_rot_all

    def residuals(x):
        ax, ay, az, dchi_ax, dchi_rh = unpack_params(x)
        rot_all, raw_rot_all = rotated_geometry(ax, ay, az)
        dpcs = _predict_snapshot_observations(
            snapshot, rot_all, raw_rot_all, metal,
            dchi_ax,
            dchi_rh if (snapshot.tensor.fit_delta_chi_rh or abs(dchi_rh) > 0.0) else 0.0,
        )
        return _optimizer_residuals(snapshot, dpcs - delta_exp)

    res = _run_two_stage_fit(
        residuals, x0, snapshot.lower_bounds, snapshot.upper_bounds,
        use_global_search=snapshot.optimizer.use_global_search,
        progress_cb=progress_cb,
        effective_de_bounds=snapshot.optimizer.effective_de_bounds,
        de_seed=snapshot.optimizer.de_seed,
        de_polish=snapshot.optimizer.de_polish,
    )
    ax, ay, az, dchi_ax, dchi_rh = unpack_params(res.x)
    rot_all, raw_rot_all = rotated_geometry(ax, ay, az)
    dpcs = _predict_snapshot_observations(
        snapshot, rot_all, raw_rot_all, metal,
        dchi_ax,
        dchi_rh if (snapshot.tensor.fit_delta_chi_rh or abs(dchi_rh) > 0.0) else 0.0,
    )
    _, resid, per_point, per_observable = _fit_result_rows(snapshot, dpcs)

    if progress_cb:
        progress_cb("Computing fit quality metrics...")

    metrics = compute_quality_metrics(per_point)
    metrics.update(_weighted_quality_metrics(snapshot, resid))

    return annotate_standard_fit_result(
        dict(
            mode='euler_global',
            ax=float(ax),
            ay=float(ay),
            az=float(az),
            delta_chi_ax=float(dchi_ax),
            delta_chi_rh=float(dchi_rh),
            per_point=per_point,
            per_observable=per_observable,
            **metrics,
        ),
        mode='euler_global',
        use_global_search=snapshot.optimizer.use_global_search,
        free_parameters=snapshot.free_parameters,
        optimizer=snapshot.optimizer.identity,
        weighting_mode=snapshot.weighting_mode,
        input_fingerprint=fingerprint_fit_input_snapshot(snapshot),
        input_configuration=fit_input_reconstruction_config(snapshot),
    )

# ---------- Mode C: Fit full tensor ----------
def fit_full_tensor(state, proton_ids=(),
                    fit_delta_chi=True,
                    fit_delta_chi_rh=False,
                    use_global_search=False,
                    progress_cb=None,
                    use_current_tensor=False,
                    observation_ids=None,
                    weighting_mode=FitWeightingMode.UNWEIGHTED.value):
    if progress_cb:
        progress_cb("Preparing 8-parameter fit...")
    snapshot = build_standard_fit_input_snapshot(
        state,
        mode="full_tensor",
        proton_ids=proton_ids,
        observation_ids=observation_ids,
        fit_delta_chi=fit_delta_chi,
        fit_delta_chi_rh=fit_delta_chi_rh,
        use_global_search=use_global_search,
        use_current_tensor=use_current_tensor,
        weighting_mode=weighting_mode,
    )
    ids = snapshot.geometry.effective_ref_ids
    abs_coords = snapshot.geometry.effective_array()
    raw_ids = snapshot.geometry.raw_ref_ids
    raw_abs_coords = snapshot.geometry.raw_array()
    delta_exp = np.asarray([item.pcs for item in snapshot.observations], dtype=float)
    x0 = snapshot.x0_array()

    def unpack_params(x):
        dchi_ax = snapshot.tensor.fixed_delta_chi_ax
        dchi_rh = snapshot.tensor.fixed_delta_chi_rh
        if snapshot.tensor.fit_delta_chi_ax and snapshot.tensor.fit_delta_chi_rh:
            ax, ay, az, mx, my, mz, dchi_ax, dchi_rh = x
        elif snapshot.tensor.fit_delta_chi_ax:
            ax, ay, az, mx, my, mz, dchi_ax = x
        elif snapshot.tensor.fit_delta_chi_rh:
            ax, ay, az, mx, my, mz, dchi_rh = x
        else:
            ax, ay, az, mx, my, mz = x
        metal = np.array([mx, my, mz], float)
        return ax, ay, az, metal, dchi_ax, dchi_rh

    def rotated_geometry(ax, ay, az, metal):
        coords0 = abs_coords - metal
        rot0 = rotate_euler(coords0, ax, ay, az)
        rot_all = rot0 + metal
        if len(raw_abs_coords):
            raw0 = raw_abs_coords - metal
            raw_rot_all = rotate_euler(raw0, ax, ay, az) + metal
        else:
            raw_rot_all = raw_abs_coords
        return rot_all, raw_rot_all

    def residuals(x):
        ax, ay, az, metal, dchi_ax, dchi_rh = unpack_params(x)
        rot_all, raw_rot_all = rotated_geometry(ax, ay, az, metal)
        dpcs = _predict_snapshot_observations(
            snapshot, rot_all, raw_rot_all, metal,
            dchi_ax,
            dchi_rh if (snapshot.tensor.fit_delta_chi_rh or abs(dchi_rh) > 0.0) else 0.0,
        )
        return _optimizer_residuals(snapshot, dpcs - delta_exp)

    res = _run_two_stage_fit(
        residuals, x0, snapshot.lower_bounds, snapshot.upper_bounds,
        use_global_search=snapshot.optimizer.use_global_search,
        progress_cb=progress_cb,
        effective_de_bounds=snapshot.optimizer.effective_de_bounds,
        de_seed=snapshot.optimizer.de_seed,
        de_polish=snapshot.optimizer.de_polish,
    )

    ax, ay, az, metal, dchi_ax, dchi_rh = unpack_params(res.x)
    rot_all, raw_rot_all = rotated_geometry(ax, ay, az, metal)
    dpcs = _predict_snapshot_observations(
        snapshot, rot_all, raw_rot_all, metal,
        dchi_ax,
        dchi_rh if (snapshot.tensor.fit_delta_chi_rh or abs(dchi_rh) > 0.0) else 0.0,
    )
    _, resid, per_point, per_observable = _fit_result_rows(snapshot, dpcs)

    if progress_cb:
        progress_cb("Computing fit quality metrics...")

    metrics = compute_quality_metrics(per_point)
    metrics.update(_weighted_quality_metrics(snapshot, resid))

    return annotate_standard_fit_result(
        dict(
            mode='full_tensor',
            euler_deg=(float(ax), float(ay), float(az)),
            metal_pos=(float(metal[0]), float(metal[1]), float(metal[2])),
            delta_chi_ax=float(dchi_ax),
            delta_chi_rh=float(dchi_rh),
            per_point=per_point,
            per_observable=per_observable,
            **metrics,
        ),
        mode='full_tensor',
        use_global_search=snapshot.optimizer.use_global_search,
        free_parameters=snapshot.free_parameters,
        optimizer=snapshot.optimizer.identity,
        weighting_mode=snapshot.weighting_mode,
        input_fingerprint=fingerprint_fit_input_snapshot(snapshot),
        input_configuration=fit_input_reconstruction_config(snapshot),
    )


# ---------- UI wiring helpers ----------
def populate_fitting_controls(state):
    # Prefer effective data for 2D/table/fitting consistency
    atom_data = state.get("atom_data_eff") or state.get("atom_data") or []
    ids = state.get("atom_ids_eff") or state.get("atom_ids") or []

    if not atom_data or not ids:
        refresh = state.get("refresh_fitting_commands")
        if callable(refresh):
            refresh()
        return

    visible_ids = visible_ref_id_set(state, atom_data=atom_data, ref_ids=ids)

    # Donor list: visible non-H atoms only. Hidden Element/Ligand atoms must not
    # remain selectable as fitting inputs.
    items = [
        f"{rid}:{atom}"
        for rid, (atom, *_) in zip(ids, atom_data)
        if atom != 'H' and int(rid) in visible_ids
    ]
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
        records = state.get(OBSERVATIONS_KEY, ()) or ()
        if records:
            selectors = []
            raw_data = state.get("atom_data_raw") or state.get("atom_data") or []
            raw_ids = state.get("atom_ids_raw") or list(range(1, len(raw_data) + 1))
            visible_fit_refs = set(visible_ids) | visible_ref_id_set(
                state, atom_data=raw_data, ref_ids=raw_ids
            )
            for index, record in enumerate(records):
                observable_id = str(record.get("id", "") or f"invalid:{index}")
                assignment = str(record.get("assignment", "") or "Unassigned")
                try:
                    target = structure_target_from_payload(
                        record.get("target"),
                        assignment=assignment,
                        target_id=observable_id,
                    )
                    if not set(target.member_ref_ids).issubset(visible_fit_refs):
                        continue
                except Exception:
                    # Preserve existing validation behavior for malformed records;
                    # the fitting boundary will still report the actual error.
                    pass
                pl.insert("end", f"{assignment} · {observable_id}")
                selectors.append(observable_id)
            state["fit_observation_selectors"] = selectors
        else:
            state["fit_observation_selectors"] = None
            for rid in proton_ids:
                pl.insert("end", str(rid))

    refresh = state.get("refresh_fitting_commands")
    if callable(refresh):
        refresh()

def apply_fit_to_views(state, *, notify=True):
    res = state.get('last_fit_result')
    if not res:
        return

    from logic.pcs_fit_result_status import (
        FIT_RESULT_STALE,
        STALE_FIT_RESULT_MESSAGE,
        evaluate_fit_result_status,
    )
    result_status = evaluate_fit_result_status(res, state)
    if result_status["status"] == FIT_RESULT_STALE:
        messagebox = state.get('messagebox')
        if messagebox is not None and hasattr(messagebox, 'showerror'):
            messagebox.showerror("Fitting", STALE_FIT_RESULT_MESSAGE)
        return False

    mode = res.get('mode')

    if mode == 'theta_alpha_multi':
        current_ids = (
            state.get('atom_ids_eff')
            or state.get('atom_ids')
            or state.get('atom_ids_raw')
            or list(range(1, len(state.get('atom_data') or []) + 1))
        )
        available = {int(ref_id) for ref_id in current_ids}
        missing = [
            int(ref_id) for ref_id in res.get('donor_ids', ())
            if int(ref_id) not in available
        ]
        if missing:
            message = (
                "The restored Mode A result cannot be applied to the plot "
                f"because donor Ref IDs are missing: {missing}."
            )
            messagebox = state.get('messagebox')
            if messagebox is not None and hasattr(messagebox, 'showerror'):
                messagebox.showerror("Fitting", message)
            return False

    if mode == 'theta_alpha_multi':
        state['fit_override'] = dict(
            mode='theta_alpha_multi',
            donor_ids=res['donor_ids'],
            theta=res['theta'],
            alpha=res['alpha'],
            axis_mode=res.get('axis_mode', 'bisector')
        )
        if notify:
            state['messagebox'].showinfo("Fitting", "Mode A applied. Click 'Update' to redraw.")

    elif mode == 'euler_global':
        state['fit_override'] = dict(
            mode='euler_global',
            ax=res['ax'],
            ay=res['ay'],
            az=res['az']
        )
        if notify:
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
            state['tensor_entry'].insert(0, str(float(res['delta_chi_ax'])))
        except Exception:
            pass

        if 'rh_dchi_rh_var' in state:
            try:
                state['rh_dchi_rh_var'].set(str(float(res.get('delta_chi_rh', 0.0))))
                state['rh_dchi_rh'] = float(res.get('delta_chi_rh', 0.0))
            except Exception:
                pass

        if notify:
            state['messagebox'].showinfo("Fitting", "Mode C applied. Click 'Update' to redraw.")
