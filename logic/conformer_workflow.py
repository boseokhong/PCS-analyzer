"""Plain-data MC project/result boundary; no widgets or application state."""
from dataclasses import asdict, replace
import json

import numpy as np

from logic.conformer_fitting import (
    ConformerSnapshot, MultiConformerSnapshot, JointFitOptions, MC_MODEL_VERSION,
    MC_SOLVER_VERSION, MC_JOINT_SOLVER_VERSION, build_basis,
    fit_tensor, fit_populations, fit_joint, objective_fingerprint, _digest,
)
from logic.pcs_fit_snapshot import FitGeometrySnapshot, FitObservation, FitTensorSnapshot, TorsionGroupSnapshot
from logic.pcs_forward_api import PCSFrame, PCSStructureTarget

PROJECT_FORMAT = "pcs_multiconformer_v1"
RESULT_FORMAT = "pcs_multiconformer_result_v1"


def default_options():
    return {"target": "both", "fit_ax": True, "fit_rh": False,
            "dchi_ax": -2., "dchi_rh": 0., "weighting": "unweighted",
            "alignment": "confirmed_common_frame", "core_ids": [],
            "observation_ids": None, "initialization": "user", "starts": 8, "seed": 0}


def normalize_options(value):
    if value is not None and not isinstance(value, dict):
        raise ValueError("MC options must be an object")
    options = default_options()
    options.update(value or {})
    if options["target"] not in ("both", "tensor", "populations"):
        raise ValueError("unsupported MC fit target")
    if options["weighting"] not in ("unweighted", "experimental_uncertainty"):
        raise ValueError("unsupported MC weighting")
    if options["alignment"] not in ("confirmed_common_frame", "metal_centered_core"):
        raise ValueError("unsupported MC alignment")
    for name in ("fit_ax", "fit_rh"):
        if not isinstance(options[name], bool):
            raise ValueError(f"{name} must be Boolean")
    for name in ("dchi_ax", "dchi_rh"):
        options[name] = float(options[name])
        if not np.isfinite(options[name]):
            raise ValueError(f"{name} must be finite")
    core = tuple(int(x) for x in options["core_ids"])
    if len(core) != len(set(core)) or any(x < 0 for x in core):
        raise ValueError("invalid MC alignment core")
    options["core_ids"] = list(core)
    ids = options["observation_ids"]
    if ids is not None:
        if not isinstance(ids, (list, tuple)) or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("selected observation IDs must be unique strings")
        options["observation_ids"] = list(ids)
    joint = JointFitOptions(seed=options["seed"], starts=options["starts"], initialization=options["initialization"])
    options.update(seed=joint.seed, starts=joint.starts)
    return options


def conformer_payload(conformer):
    if isinstance(conformer, ConformerSnapshot):
        return {"format": "mapped_v1", **asdict(conformer)}
    if isinstance(conformer, dict) and conformer.get("format") == "unmapped_xyz_v1":
        # Validate and detach pending source rows without inventing an atom mapping.
        rows = conformer.get("atoms", [])
        if not rows or any(len(row) != 4 or not str(row[0]).strip() for row in rows):
            raise ValueError("invalid unmapped structure rows")
        if not np.isfinite(np.asarray([row[1:] for row in rows], dtype=float)).all():
            raise ValueError("unmapped coordinates must be finite")
        if (not str(conformer.get("conformer_id", "")).strip() or not str(conformer.get("name", "")).strip()
                or not isinstance(conformer.get("included", True), bool)):
            raise ValueError("unmapped structure requires a stable ID/name and Boolean inclusion")
        return json.loads(json.dumps(conformer, allow_nan=False))
    raise ValueError("Legacy conformer has no explicit mapping/frame; reload and configure it")


def conformer_from_payload(value):
    if not isinstance(value, dict):
        raise ValueError("conformer payload must be an object")
    if value.get("format") == "unmapped_xyz_v1":
        return conformer_payload(value)
    if value.get("format") != "mapped_v1":
        raise ValueError("unsupported conformer format")
    data = dict(value)
    data.pop("format")
    geometry = dict(data.pop("geometry"))
    geometry["torsion_groups"] = tuple(TorsionGroupSnapshot(**g) for g in geometry["torsion_groups"])
    data["geometry"] = FitGeometrySnapshot(**geometry)
    return ConformerSnapshot(**data)


def snapshot_payload(snapshot):
    observations = []
    for obs in snapshot.observations:
        target = obs.target
        observations.append({
            "ref_id": obs.ref_id, "pcs": obs.pcs, "observable_id": obs.observable_id,
            "assignment": obs.assignment, "source": obs.source,
            "source_dataset_ids": list(obs.source_dataset_ids),
            "source_observable_ids": list(obs.source_observable_ids),
            "provenance_json": obs.provenance_json, "uncertainty_ppm": obs.uncertainty_ppm,
            "target": {"target_id": target.target_id, "site_member_ref_ids": target.site_member_ref_ids,
                       "member_reduction": target.member_reduction, "site_reduction": target.site_reduction},
        })
    return {"model": MC_MODEL_VERSION, "conformers": [conformer_payload(c) for c in snapshot.conformers],
            "observations": observations, "orientation": asdict(snapshot.orientation),
            "tensor": asdict(snapshot.tensor), "alignment_mode": snapshot.alignment_mode,
            "core_ref_ids": snapshot.core_ref_ids, "weighting_mode": snapshot.weighting_mode,
            "run_id": snapshot.run_id, "document_id": snapshot.document_id,
            "document_revision": snapshot.document_revision}


def snapshot_from_payload(value):
    if not isinstance(value, dict):
        raise ValueError("MC snapshot must be an object")
    data = dict(value)
    if data.pop("model", None) != MC_MODEL_VERSION:
        raise ValueError("unsupported MC physical model")
    data["conformers"] = tuple(conformer_from_payload(c) for c in data["conformers"])
    observations = []
    for item in data["observations"]:
        item = dict(item)
        item["target"] = PCSStructureTarget(**item["target"])
        observations.append(FitObservation(**item))
    data["observations"] = tuple(observations)
    data["orientation"] = PCSFrame(**data["orientation"])
    data["tensor"] = FitTensorSnapshot(**data["tensor"])
    return MultiConformerSnapshot(**data)


def input_fingerprint(snapshot, options):
    options = normalize_options(options)
    # Selection, frame, tensor and weighting dependencies already belong to the snapshot.
    return _digest({"objective": objective_fingerprint(snapshot),
                    "target": options["target"], "initialization": options["initialization"],
                    "starts": options["starts"] if options["target"] == "both" else None,
                    "seed": options["seed"] if options["target"] == "both" else None})


def execute_fit(snapshot, options, *, input_rows=(), basis=None):
    """Worker entry point accepts detached data and returns a plain payload."""
    options = normalize_options(options)
    validate_snapshot_options(snapshot, options)
    basis = build_basis(snapshot) if basis is None else basis
    target = options["target"]
    if target == "both":
        result = fit_joint(snapshot, basis=basis, options=JointFitOptions(
            starts=options["starts"], seed=options["seed"], initialization=options["initialization"]))
        runs = [asdict(run) for run in result.runs]
        selected = result.selected_index
        alternatives = list(result.alternative_indices)
        ambiguous = result.population_ambiguous
        solver_version = result.solver_version
    else:
        if target == "tensor":
            from logic.conformer_fitting import population_start
            result = fit_tensor(snapshot, basis=basis, populations=population_start(snapshot, options["initialization"]))
        else:
            result = fit_populations(snapshot, basis=basis, initialization=options["initialization"])
        runs = [{name: getattr(result, name) for name in (
            "initial_populations", "populations", "dchi_ax", "dchi_rh", "predictions", "residuals",
            "objective", "rmsd", "iterations", "success", "message", "stationarity", "rank",
            "singular_values", "identifiable", "free_parameters") }]
        selected, alternatives = (0 if result.success else None), []
        ambiguous = (not result.identifiable if target == "populations" else False) if result.success else None
        solver_version = result.solver_version
    best = None if selected is None else dict(runs[selected])
    if best is not None:
        best["weights"] = best["populations"]
        best["n"] = len(snapshot.observations)
        best["per_point"] = [(o.ref_id, o.pcs, p, r) for o, p, r in
                                 zip(snapshot.observations, best["predictions"], best["residuals"])]
        predictions = best["dchi_ax"] * basis.axial + best["dchi_rh"] * basis.rhombic
        best["conformer_pcs"] = predictions.tolist()
        best["contributions"] = (predictions * np.asarray(best["populations"])).tolist()
    payload = {"format": RESULT_FORMAT, "schema_version": 1, "solver_version": solver_version,
               "input_snapshot": snapshot_payload(snapshot), "options": options,
               "input_fingerprint": input_fingerprint(snapshot, options),
               "input_rows": [conformer_payload(c) for c in input_rows],
               "conformer_ids": list(basis.conformer_ids), "observation_ids": list(basis.observation_ids),
               "alignment": [{"conformer_id": c.conformer_id, "metal_origin": list(c.geometry.metal_position),
                              "row_rotation": rotation, "rmsd_angstrom": rmsd}
                             for c, rotation, rmsd in zip((c for c in snapshot.conformers if c.included),
                                                         basis.alignment_row_rotations, basis.alignment_rmsd_angstrom)],
               "success": selected is not None, "selected_index": selected, "best": best,
               "runs": runs, "alternative_indices": alternatives, "population_ambiguous": ambiguous}
    return json.loads(json.dumps(payload, allow_nan=False))


def validate_result(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("MC result must be an object")
    value = json.loads(json.dumps(value, allow_nan=False))
    if value.get("format") != RESULT_FORMAT or value.get("schema_version") != 1:
        raise ValueError("unsupported MC result schema")
    snapshot = snapshot_from_payload(value["input_snapshot"])
    options = normalize_options(value["options"])
    validate_snapshot_options(snapshot, options)
    expected_solver = MC_JOINT_SOLVER_VERSION if options["target"] == "both" else MC_SOLVER_VERSION
    if value["solver_version"] != expected_solver:
        raise ValueError("unsupported MC solver version")
    if value["input_fingerprint"] != input_fingerprint(snapshot, options):
        raise ValueError("MC result input fingerprint mismatch")
    included = tuple(c for c in snapshot.conformers if c.included)
    if value["conformer_ids"] != [c.conformer_id for c in included] or value["observation_ids"] != [o.observable_id for o in snapshot.observations]:
        raise ValueError("MC result identity/order mismatch")
    if [a["conformer_id"] for a in value["alignment"]] != value["conformer_ids"]:
        raise ValueError("MC alignment identity mismatch")
    for item, conformer in zip(value["alignment"], included):
        rotation = np.asarray(item["row_rotation"], dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-10)
                or tuple(item["metal_origin"]) != conformer.geometry.metal_position
                or not np.isfinite(item["rmsd_angstrom"]) or item["rmsd_angstrom"] < 0):
            raise ValueError("invalid saved MC alignment")
    best = value["best"]
    runs, selected = value["runs"], value["selected_index"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("MC result must retain attempted runs")
    if not isinstance(value["success"], bool) or any(not isinstance(run.get("success"), bool) for run in runs):
        raise ValueError("MC success flags must be Boolean")
    if bool(value["success"]) != (best is not None):
        raise ValueError("MC result success/best mismatch")
    if best is not None:
        if not isinstance(selected, int) or isinstance(selected, bool) or not 0 <= selected < len(runs) or not runs[selected]["success"]:
            raise ValueError("invalid selected MC run")
        if any(best.get(k) != v for k, v in runs[selected].items()):
            raise ValueError("MC best result differs from its selected run")
        weights = np.asarray(best["populations"], dtype=float)
        if weights.shape != (len(included),) or np.any(weights < 0) or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-10):
            raise ValueError("invalid saved MC populations")
        for i, c in enumerate(included):
            if c.population_fixed and weights[i] != c.initial_population:
                raise ValueError("saved MC result changed a fixed population")
        tensor = snapshot.tensor
        if ((options["target"] == "populations" or not tensor.fit_delta_chi_ax) and best["dchi_ax"] != tensor.fixed_delta_chi_ax
                or (options["target"] == "populations" or not tensor.fit_delta_chi_rh) and best["dchi_rh"] != tensor.fixed_delta_chi_rh):
            raise ValueError("saved MC result changed a fixed tensor component")
        for name in ("predictions", "residuals"):
            if len(best[name]) != len(snapshot.observations):
                raise ValueError("saved MC observation shape mismatch")
        if not np.isfinite([*weights, *best["predictions"], *best["residuals"], best["dchi_ax"], best["dchi_rh"], best["objective"], best["rmsd"]]).all():
            raise ValueError("nonfinite saved MC result")
        if best["objective"] < 0 or best["rmsd"] < 0:
            raise ValueError("saved MC error metrics must be nonnegative")
        for name in ("conformer_pcs", "contributions"):
            array = np.asarray(best[name], dtype=float)
            if array.shape != (len(snapshot.observations), len(included)) or not np.isfinite(array).all():
                raise ValueError("invalid saved conformer contribution shape")
        pred, residual = np.asarray(best["predictions"]), np.asarray(best["residuals"])
        observed = np.asarray([o.pcs for o in snapshot.observations])
        sigma = np.asarray([o.uncertainty_ppm for o in snapshot.observations]) if snapshot.weighting_mode != "unweighted" else np.ones(len(observed))
        contributions = np.asarray(best["contributions"])
        if (not np.allclose(residual, pred-observed, rtol=1e-9, atol=1e-10)
                or not np.allclose(contributions, np.asarray(best["conformer_pcs"])*weights, rtol=1e-9, atol=1e-10)
                or not np.allclose(pred, contributions.sum(axis=1), rtol=1e-9, atol=1e-10)
                or not np.isclose(best["objective"], np.sum((residual/sigma)**2), rtol=1e-9, atol=1e-10)
                or not np.isclose(best["rmsd"], np.sqrt(np.mean(residual**2)), rtol=1e-9, atol=1e-10)):
            raise ValueError("inconsistent saved MC predictions/residuals/objective")
    elif selected is not None:
        raise ValueError("failed MC result cannot select a best run")
    if any(not isinstance(i, int) or not 0 <= i < len(runs) or not runs[i]["success"] for i in value["alternative_indices"]):
        raise ValueError("invalid MC alternative run")
    if value.get("reliability") is not None:
        from logic.conformer_reliability import validate_reliability
        validate_reliability(value["reliability"], value)
    return value


def validate_snapshot_options(snapshot, options):
    tensor = snapshot.tensor
    if (tensor.fit_delta_chi_ax != options["fit_ax"] or tensor.fit_delta_chi_rh != options["fit_rh"]
            or tensor.fixed_delta_chi_ax != options["dchi_ax"] or tensor.fixed_delta_chi_rh != options["dchi_rh"]
            or snapshot.alignment_mode != options["alignment"] or tuple(snapshot.core_ref_ids) != tuple(options["core_ids"])
            or snapshot.weighting_mode != options["weighting"]):
        raise ValueError("MC snapshot and recorded fit options disagree")


def project_payload(conformers, options, result):
    payload = {"format": PROJECT_FORMAT, "schema_version": 1,
               "conformers": [conformer_payload(c) for c in conformers],
               "options": normalize_options(options), "last_result": validate_result(result)}
    return json.loads(json.dumps(payload, allow_nan=False))


def parse_project_payload(value):
    if not value:
        return [], default_options(), None
    if not isinstance(value, dict) or value.get("format") != PROJECT_FORMAT or value.get("schema_version") != 1:
        raise ValueError("unsupported MC project schema")
    conformers = [conformer_from_payload(c) for c in value["conformers"]]
    ids = [c.conformer_id if isinstance(c, ConformerSnapshot) else c["conformer_id"] for c in conformers]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate saved MC conformer ID")
    return conformers, normalize_options(value["options"]), validate_result(value["last_result"])


def result_status(result, current_snapshot, current_options):
    if not result or result.get("format") != RESULT_FORMAT:
        return "Unknown"
    if current_snapshot is None:
        return "Stale"
    recorded_frame = PCSFrame(**result["input_snapshot"]["orientation"])
    if np.allclose(recorded_frame.rotation_matrix, current_snapshot.orientation.rotation_matrix, rtol=1e-12, atol=1e-12):
        # Save/reload may project a canonical ZYZ/radian frame to equivalent XYZ
        # degrees. Preserve recorded provenance without staling identical physics.
        current_snapshot = replace(current_snapshot, orientation=recorded_frame)
    return "Current" if result["input_fingerprint"] == input_fingerprint(current_snapshot, current_options) else "Stale"


def fitted_scalar_changes(result):
    """Apply only actually fitted scalars; never local fixed overrides or populations."""
    if not result or not result.get("success") or result.get("format") != RESULT_FORMAT:
        return {}
    if result["options"]["target"] == "populations":
        return {}
    tensor = result["input_snapshot"]["tensor"]
    changes = {}
    if tensor["fit_delta_chi_ax"]:
        changes["delta_chi_ax"] = result["best"]["dchi_ax"]
    if tensor["fit_delta_chi_rh"]:
        changes["delta_chi_rh"] = result["best"]["dchi_rh"]
    return changes
