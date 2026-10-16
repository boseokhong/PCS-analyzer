"""Detached fixed-frame joint PCS/RDC inputs, CSV validation and fitting."""
from dataclasses import dataclass, asdict, replace
from hashlib import sha256
import csv
import json

import numpy as np

from logic.advanced_physics import GYRO, rdc_from_dchi
from logic.pcs_fit_snapshot import FitGeometrySnapshot, FitObservation, TorsionGroupSnapshot
from logic.pcs_forward_api import PCSFrame, PCSStructureTarget
from logic.pcs_fitting_adapter import predict_structured_pcs_from_factors
from logic.symmetry_geometry import effective_geometry_factors

MODEL = "joint_fixed_frame_molecular_v2"


def default_options():
    return {"B0": 14.1, "T": 298., "w_rdc": 1., "use_pcs_errors": False,
            "observation_ids": None}


def normalize_options(value=None):
    options = default_options()
    options.update(value or {})
    for key in ("B0", "T", "w_rdc"):
        options[key] = float(options[key])
    if not all(np.isfinite(options[k]) for k in ("B0", "T", "w_rdc")):
        raise ValueError("Field, temperature and RDC multiplier must be finite")
    if options["B0"] <= 0 or options["T"] <= 0 or options["w_rdc"] <= 0:
        raise ValueError("Joint fitting requires positive field, temperature and RDC multiplier")
    options["use_pcs_errors"] = bool(options["use_pcs_errors"])
    if options["observation_ids"] is not None:
        options["observation_ids"] = list(dict.fromkeys(map(str, options["observation_ids"])))
    return options


def parse_rdc_csv(stream):
    """Prefer explicit raw Ref pairs; retain unambiguous legacy atom labels."""
    reader = csv.DictReader(stream)
    columns = set(reader.fieldnames or ())
    by_ref = {"ref1", "ref2"} <= columns
    if not by_ref and not {"atom1", "atom2"} <= columns:
        raise ValueError("RDC CSV requires ref1/ref2 or atom1/atom2")
    if not ({"rdc_exp", "d"} & columns):
        raise ValueError("RDC CSV requires rdc_exp (Hz)")
    rows = []
    for number, record in enumerate(reader, 2):
        try:
            row = {"rdc_exp": float(record.get("rdc_exp", record.get("d"))),
                   "rdc_err": float(record.get("rdc_err") or .5),
                   "nuc1": (record.get("nuc1") or "H").upper(),
                   "nuc2": (record.get("nuc2") or "N").upper(), "source_row": number}
            if not np.isfinite(row["rdc_exp"]) or not np.isfinite(row["rdc_err"]) or row["rdc_err"] <= 0:
                raise ValueError("finite RDC and positive uncertainty required")
            if row["nuc1"] not in GYRO or row["nuc2"] not in GYRO:
                raise ValueError("unsupported nucleus")
            if by_ref:
                row.update(ref1=int(record["ref1"]), ref2=int(record["ref2"]))
            else:
                row.update(atom1=record["atom1"].strip(), atom2=record["atom2"].strip())
                if not row["atom1"] or not row["atom2"]:
                    raise ValueError("both atom labels are required")
            rows.append(row)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"RDC CSV line {number}: {exc}") from exc
    if not rows:
        raise ValueError("RDC CSV contains no observations")
    return rows


@dataclass(frozen=True)
class RDCObservation:
    ref1: int
    ref2: int
    value: float
    uncertainty: float
    nuc1: str
    nuc2: str
    source_row: int


@dataclass(frozen=True)
class JointSnapshot:
    geometry: FitGeometrySnapshot
    orientation: PCSFrame
    observations: tuple[FitObservation, ...]
    bonds: tuple[RDCObservation, ...]
    B0: float = 14.1
    temperature: float = 298.
    rdc_multiplier: float = 1.
    use_pcs_errors: bool = False
    skipped_rows: tuple[str, ...] = ()


def snapshot_payload(snapshot):
    observations = []
    for obs in snapshot.observations:
        target = obs.target
        observations.append({"observable_id": obs.observable_id, "ref_id": obs.ref_id,
            "pcs": obs.pcs, "uncertainty_ppm": obs.uncertainty_ppm, "assignment": obs.assignment,
            "site_member_ref_ids": target.site_member_ref_ids,
            "member_reduction": target.member_reduction.value,
            "site_reduction": target.site_reduction.value})
    payload = {"model": MODEL, "geometry": asdict(snapshot.geometry),
            "orientation": asdict(snapshot.orientation), "observations": observations,
            "bonds": [asdict(b) for b in snapshot.bonds], "B0": snapshot.B0,
            "temperature": snapshot.temperature, "rdc_multiplier": snapshot.rdc_multiplier,
            "use_pcs_errors": snapshot.use_pcs_errors, "skipped_rows": snapshot.skipped_rows}
    return json.loads(json.dumps(payload, allow_nan=False))


def fingerprint(snapshot):
    return sha256(json.dumps(snapshot_payload(snapshot), sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()).hexdigest()


def build_design(snapshot):
    """Use established structured PCS reductions; RDC bonds are raw Ref pairs."""
    normalize_options({"B0": snapshot.B0, "T": snapshot.temperature, "w_rdc": snapshot.rdc_multiplier})
    if not snapshot.observations or not snapshot.bonds:
        raise ValueError("Select PCS observations and load at least one matched RDC bond")
    geometry = snapshot.geometry
    for refs, coords in ((geometry.raw_ref_ids, geometry.raw_coordinates),
                        (geometry.effective_ref_ids, geometry.effective_coordinates)):
        if np.asarray(coords).shape != (len(refs), 3) or len(set(refs)) != len(refs):
            raise ValueError("Structure coordinates need unique Ref IDs and matching N-by-3 arrays")
    if geometry.torsion_enabled:
        raise ValueError("This fixed-bond RDC model does not support torsional averaging")
    if len(set(geometry.raw_ref_ids)) != len(geometry.raw_ref_ids):
        raise ValueError("Raw structure Ref IDs must be unique")
    if len(set(o.observable_id for o in snapshot.observations)) != len(snapshot.observations):
        raise ValueError("PCS observation IDs must be unique")
    rotation = np.asarray(snapshot.orientation.rotation_matrix).T
    raw = (geometry.raw_array() - geometry.metal_position) @ rotation
    effective = (geometry.effective_array() - geometry.metal_position) @ rotation
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(effective)):
        raise ValueError("Structure coordinates must be finite")
    raw_map = dict(zip(geometry.raw_ref_ids, raw))
    effective_map = dict(zip(geometry.effective_ref_ids, effective))
    targets = tuple(o.target for o in snapshot.observations)
    refs = tuple(dict.fromkeys(r for t in targets for r in t.member_ref_ids))
    pseudo = geometry.pseudo_member_mapping()
    for rid in refs:
        members = pseudo.get(rid, (rid,))
        if any(m not in raw_map and m not in effective_map for m in members):
            raise ValueError(f"PCS target Ref {rid} has missing structure members")
        if any(np.linalg.norm(raw_map[m] if m in raw_map else effective_map[m]) <= 0 for m in members):
            raise ValueError("PCS target member coincides with the metal centre")
    try:
        coords = np.array([effective_map[r] if r in effective_map else raw_map[r] for r in refs])
    except KeyError as exc:
        raise ValueError(f"PCS target Ref {exc.args[0]} is unavailable") from exc
    _, _, _, ga, gr = effective_geometry_factors(refs, coords, pseudo_members=pseudo, raw_coords_by_id=raw_map)
    pcs = np.column_stack([predict_structured_pcs_from_factors(refs, ga, gr, ax, rh, targets)
                           for ax, rh in ((1., 0.), (0., 1.))])
    rdc = []
    seen = set()
    for b in snapshot.bonds:
        if b.nuc1 not in GYRO or b.nuc2 not in GYRO:
            raise ValueError("Unsupported RDC nucleus")
        key = tuple(sorted((b.ref1, b.ref2)))
        if key in seen:
            raise ValueError("Duplicate RDC bond; use separate datasets for repeated conditions")
        seen.add(key)
        if b.ref1 not in raw_map or b.ref2 not in raw_map:
            raise ValueError("RDC bonds require two raw structure Ref IDs")
        vector = raw_map[b.ref2] - raw_map[b.ref1]
        length = np.linalg.norm(vector)
        if length <= 0 or not np.isfinite(b.value) or not np.isfinite(b.uncertainty) or b.uncertainty <= 0:
            raise ValueError("RDC needs a nonzero bond and finite value/positive uncertainty")
        theta, phi = np.arccos(np.clip(vector[2]/length, -1, 1)), np.arctan2(vector[1], vector[0])
        rdc.append([rdc_from_dchi(theta, phi, length, ax, rh, snapshot.B0, snapshot.temperature,
                                  GYRO[b.nuc1], GYRO[b.nuc2]) for ax, rh in ((1., 0.), (0., 1.))])
    measured = np.array([o.pcs for o in snapshot.observations] + [b.value for b in snapshot.bonds])
    errors = np.ones(len(snapshot.observations))
    if snapshot.use_pcs_errors:
        errors = np.array([o.uncertainty_ppm if o.uncertainty_ppm is not None else np.nan for o in snapshot.observations])
        if np.any(~np.isfinite(errors)) or np.any(errors <= 0):
            raise ValueError("Every selected PCS observation requires a positive uncertainty")
    scales = np.r_[1/errors, [snapshot.rdc_multiplier/max(b.uncertainty, .1) for b in snapshot.bonds]]
    basis = np.vstack([pcs, rdc])
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(basis)):
        raise ValueError("PCS/RDC data and predictions must be finite")
    return basis, measured, scales


def execute_fit(snapshot):
    basis, measured, scales = build_design(snapshot)
    design, target = basis*scales[:, None], measured*scales
    values, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    if rank < 2:
        raise ValueError("Data cannot independently determine Δχ_ax and Δχ_rh (rank below 2)")
    predicted = basis @ values
    residual = predicted - measured
    n = len(snapshot.observations)
    return {"model": MODEL, "fingerprint": fingerprint(snapshot), "input_snapshot": snapshot_payload(snapshot),
        "success": True, "dchi_ax": float(values[0]), "dchi_rh": float(values[1]),
        "rank": int(rank), "condition_number": float(singular[0]/singular[-1]),
        "singular_values": singular.tolist(), "weighted_sum_squares": float(np.sum((residual*scales)**2)),
        "rmsd_pcs": float(np.sqrt(np.mean(residual[:n]**2))),
        "rmsd_rdc": float(np.sqrt(np.mean(residual[n:]**2))),
        "per_point_pcs": [{"observable_id": o.observable_id, "ref_id": o.ref_id, "assignment": o.assignment,
                            "experimental": float(measured[i]), "predicted": float(predicted[i]),
                            "residual": float(residual[i]), "uncertainty": o.uncertainty_ppm}
                           for i, o in enumerate(snapshot.observations)],
        "rdc_results": [{"ref1": b.ref1, "ref2": b.ref2, "source_row": b.source_row,
                         "experimental": float(measured[n+i]), "predicted": float(predicted[n+i]),
                         "residual": float(residual[n+i]), "uncertainty": b.uncertainty}
                        for i, b in enumerate(snapshot.bonds)], "skipped_rows": list(snapshot.skipped_rows)}


def snapshot_from_payload(value):
    if not isinstance(value, dict) or value.get('model') != MODEL:
        raise ValueError('Unsupported joint PCS/RDC snapshot model')
    data = value['geometry']
    geometry = FitGeometrySnapshot(
        tuple(map(int, data['effective_ref_ids'])), tuple(tuple(map(float, p)) for p in data['effective_coordinates']),
        tuple(map(int, data['raw_ref_ids'])), tuple(tuple(map(float, p)) for p in data['raw_coordinates']),
        tuple(map(float, data['metal_position'])),
        tuple((int(r), tuple(map(int, members))) for r, members in data['pseudo_members']),
        bool(data['torsion_enabled']), tuple(TorsionGroupSnapshot(**g) for g in data['torsion_groups']))
    observations = []
    for item in value['observations']:
        target = PCSStructureTarget(target_id=item['observable_id'],
            site_member_ref_ids=tuple(tuple(map(int, site)) for site in item['site_member_ref_ids']),
            member_reduction=item['member_reduction'], site_reduction=item['site_reduction'])
        observations.append(FitObservation(item['ref_id'], item['pcs'], observable_id=item['observable_id'],
            assignment=item['assignment'], uncertainty_ppm=item['uncertainty_ppm'], target=target))
    return JointSnapshot(geometry, PCSFrame(**value['orientation']), tuple(observations),
        tuple(RDCObservation(**b) for b in value['bonds']), float(value['B0']), float(value['temperature']),
        float(value['rdc_multiplier']), bool(value['use_pcs_errors']), tuple(value['skipped_rows']))


def validate_result(value):
    """Validate recorded predictions against their saved model without refitting."""
    if value is None:
        return None
    value = json.loads(json.dumps(value, allow_nan=False))
    if value.get('model') != MODEL or value.get('success') is not True:
        raise ValueError('Unsupported or unsuccessful joint PCS/RDC result')
    snapshot = snapshot_from_payload(value['input_snapshot'])
    if value['fingerprint'] != fingerprint(snapshot):
        raise ValueError('Joint result fingerprint does not match its recorded inputs')
    basis, measured, scales = build_design(snapshot)
    design = basis*scales[:, None]
    singular = np.linalg.svd(design, compute_uv=False)
    if np.linalg.matrix_rank(design) != 2 or value['rank'] != 2:
        raise ValueError('Saved joint result is rank deficient')
    params = np.array([value['dchi_ax'], value['dchi_rh']], dtype=float)
    predicted = basis @ params
    residual = predicted-measured
    # Confirm stationarity of the recorded fixed-frame least-squares solution.
    gradient = design.T @ (residual*scales)
    gradient_scale = max(1., np.linalg.norm(design)*np.linalg.norm(measured*scales))
    if np.linalg.norm(gradient) > 1e-9*gradient_scale:
        raise ValueError('Saved joint parameters are inconsistent with the recorded fit objective')
    n = len(snapshot.observations)
    expected_metrics = {'condition_number': singular[0]/singular[-1],
        'weighted_sum_squares': np.sum((residual*scales)**2),
        'rmsd_pcs': np.sqrt(np.mean(residual[:n]**2)), 'rmsd_rdc': np.sqrt(np.mean(residual[n:]**2))}
    for key, expected in expected_metrics.items():
        if not np.isclose(value[key], expected, rtol=1e-8, atol=1e-10):
            raise ValueError(f'Inconsistent saved joint metric: {key}')
    if not np.allclose(value['singular_values'], singular, rtol=1e-8, atol=1e-10):
        raise ValueError('Inconsistent saved joint singular values')
    for points, observations, offset in ((value['per_point_pcs'], snapshot.observations, 0),
                                         (value['rdc_results'], snapshot.bonds, n)):
        if len(points) != len(observations):
            raise ValueError('Saved joint observation count mismatch')
        for i, (point, obs) in enumerate(zip(points, observations)):
            if offset == 0:
                if point['observable_id'] != obs.observable_id or point['ref_id'] != obs.ref_id or point['assignment'] != obs.assignment:
                    raise ValueError('Saved PCS observation identity mismatch')
                uncertainty = obs.uncertainty_ppm
            else:
                if (point['ref1'], point['ref2'], point['source_row']) != (obs.ref1, obs.ref2, obs.source_row):
                    raise ValueError('Saved RDC observation identity mismatch')
                uncertainty = obs.uncertainty
            if point['uncertainty'] != uncertainty:
                raise ValueError('Saved observation uncertainty mismatch')
            for key, expected in (('experimental', measured[offset+i]), ('predicted', predicted[offset+i]), ('residual', residual[offset+i])):
                if not np.isclose(point[key], expected, rtol=1e-9, atol=1e-10):
                    raise ValueError('Saved joint predictions are inconsistent')
    if value['skipped_rows'] != list(snapshot.skipped_rows):
        raise ValueError('Saved joint matching diagnostics are inconsistent')
    return value


def project_payload(rows, options, result):
    value = {'format': 'joint_pcs_rdc_project', 'schema_version': 1, 'rdc_rows': rows,
             'options': normalize_options(options), 'last_result': validate_result(result)}
    return json.loads(json.dumps(value, allow_nan=False))


def parse_project_payload(value):
    if value is None:
        return [], default_options(), None
    if not isinstance(value, dict) or value.get('format') != 'joint_pcs_rdc_project' or value.get('schema_version') != 1:
        raise ValueError('Unsupported joint PCS/RDC project schema')
    rows = json.loads(json.dumps(value['rdc_rows'], allow_nan=False))
    if not isinstance(rows, list):
        raise ValueError('Saved RDC rows must be a list')
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Saved RDC row must be an object')
        if not {'ref1','ref2'} <= row.keys() and not {'atom1','atom2'} <= row.keys():
            raise ValueError('Saved RDC row requires a Ref pair or atom labels')
        if {'ref1','ref2'} <= row.keys():
            if any(type(row[k]) is not int for k in ('ref1','ref2')):
                raise ValueError('Saved RDC Ref IDs must be integers')
        elif any(not isinstance(row[k], str) or not row[k].strip() for k in ('atom1','atom2')):
            raise ValueError('Saved RDC labels must be nonempty strings')
        if not np.isfinite(float(row['rdc_exp'])) or not np.isfinite(float(row.get('rdc_err',.5))) or float(row.get('rdc_err',.5)) <= 0:
            raise ValueError('Saved RDC value/error is invalid')
        if any(str(row.get(k,default)).upper() not in GYRO for k,default in (('nuc1','H'),('nuc2','N'))):
            raise ValueError('Saved RDC nucleus is unsupported')
    return rows, normalize_options(value['options']), validate_result(value['last_result'])


def result_status(result, current):
    if not result or result.get('model') != MODEL:
        return 'Unknown'
    if current is None:
        return 'Stale'
    recorded = snapshot_from_payload(result['input_snapshot'])
    # Legacy .pcsp coordinate/shift tables use eight decimal places. Accept
    # only their absolute rounding envelope; never relative large-value tolerance.
    g, old = current.geometry, recorded.geometry
    changes = {}
    for key in ('raw_coordinates','effective_coordinates','metal_position'):
        a,b=np.asarray(getattr(g,key)),np.asarray(getattr(old,key))
        if a.shape == b.shape and np.allclose(a,b,rtol=0,atol=5.1e-9):
            changes[key]=getattr(old,key)
    current = replace(current,geometry=replace(g,**changes))
    if np.allclose(current.orientation.rotation_matrix, recorded.orientation.rotation_matrix, rtol=0, atol=1e-12):
        current=replace(current,orientation=recorded.orientation)
    if len(current.observations)==len(recorded.observations):
        observations=[]
        for obs,old_obs in zip(current.observations,recorded.observations):
            if obs.observable_id==old_obs.observable_id and abs(obs.pcs-old_obs.pcs)<=5.1e-9:
                obs=replace(obs,pcs=old_obs.pcs)
            observations.append(obs)
        current=replace(current,observations=tuple(observations))
    return 'Current' if fingerprint(current)==result['fingerprint'] else 'Stale'
