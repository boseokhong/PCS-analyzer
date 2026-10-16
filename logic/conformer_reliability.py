"""MC5: refit-based uncertainty and out-of-sample diagnostics, without UI state.

Intervals are conditional percentile estimates, not simultaneous/calibrated coverage
guarantees. Geometry, orientation, assignments and supplied fixed values are held
constant. Solver multistart dispersion is never treated as sampling uncertainty.
"""
from dataclasses import dataclass, asdict, replace
import json
import math

import numpy as np

from logic.conformer_fitting import (
    ConformerBasis, basis_fingerprint, build_basis, population_start,
    joint_rank_diagnostics, _digest,
)

RELIABILITY_FORMAT = "pcs_multiconformer_reliability_v1"
RELIABILITY_VERSION = "refit-percentile-grouped-cv-v1"


@dataclass(frozen=True)
class ReliabilityOptions:
    method: str = "parametric"
    replicates: int = 100
    folds: int = 5
    seed: int = 2026
    confidence: float = .95
    similarity_relative: float = .01

    def __post_init__(self):
        if self.method not in ("parametric", "cases"):
            raise ValueError("bootstrap method must be parametric or cases")
        for name, low, high in (("replicates", 20, 2000), ("folds", 2, 20), ("seed", 0, 2**32-1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in {low}..{high}")
            object.__setattr__(self, name, int(value))
        for name in ("confidence", "similarity_relative"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and between zero and one")
            object.__setattr__(self, name, value)


def observation_groups(snapshot):
    """Connected overlapping physical members/source observables stay together.

    Expand pseudo IDs to physical raw members. A shared dataset ID alone is not
    proof of covariance. Further correlations remain an explicit user assumption.
    """
    pseudo = dict(next(c for c in snapshot.conformers if c.included).geometry.pseudo_members)
    keys = []
    for obs in snapshot.observations:
        physical = {r for ref in obs.target.member_ref_ids for r in pseudo.get(ref, (ref,))}
        keys.append({("ref", r) for r in physical} |
                    {("source", s) for s in obs.source_observable_ids if s})
    groups = []
    remaining = set(range(len(keys)))
    while remaining:
        group = {min(remaining)}
        combined = set(keys[min(remaining)])
        remaining -= group
        changed = True
        while changed:
            added = {i for i in remaining if keys[i] & combined}
            changed = bool(added)
            for i in added:
                combined.update(keys[i])
            group.update(added)
            remaining -= added
        groups.append(sorted(group))
    return groups


def _subset(snapshot, basis, indices, *, measured=None, unique=False):
    observations = []
    for position, index in enumerate(indices):
        obs = snapshot.observations[int(index)]
        if unique:
            identifier = f"bootstrap:{position}:{obs.observable_id}"
            obs = replace(obs, observable_id=identifier, target=replace(obs.target, target_id=identifier))
        if measured is not None:
            obs = replace(obs, pcs=float(measured[position]))
        observations.append(obs)
    child = replace(snapshot, observations=tuple(observations))
    def matrix(values):
        sliced = np.asarray(values)[indices]
        return np.frombuffer(sliced.tobytes(), dtype=float).reshape(sliced.shape)
    sliced_basis = ConformerBasis(basis_fingerprint(child), basis.conformer_ids,
                                 tuple(o.observable_id for o in child.observations),
                                 matrix(basis.axial), matrix(basis.rhombic),
                                 basis.alignment_row_rotations, basis.alignment_rmsd_angstrom)
    return child, sliced_basis


def _refit(snapshot, options, basis, seed):
    # Lazy import keeps the optional reliability payload boundary acyclic.
    from logic.conformer_workflow import execute_fit
    try:
        fit = execute_fit(snapshot, dict(options, seed=int(seed)), basis=basis)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return {"success": False, "solver_seed": int(seed), "attempted_starts": 0,
                "messages": [f"{type(exc).__name__}: {exc}"], "population_ambiguous": None, "estimate": None}
    best = fit["best"]
    return {"success": fit["success"], "solver_seed": int(seed),
            "attempted_starts": len(fit["runs"]),
            "messages": [r["message"] for r in fit["runs"]],
            "population_ambiguous": fit["population_ambiguous"],
            "estimate": None if best is None else {k: best[k] for k in
                ("populations", "dchi_ax", "dchi_rh", "objective", "rmsd")}}


def _interval(values, confidence):
    tail = (1-confidence)/2
    return [float(x) for x in np.quantile(values, (tail, 1-tail))]


def similarity_diagnostics(snapshot, basis, best, relative_threshold):
    """Deterministic complete-link suggestions at the recorded fitted tensor.

    Similar PCS columns do not imply similar structures or prove model identity.
    Never merge/remove rows, refit, or change the forward model here.
    """
    pcs = best["dchi_ax"]*basis.axial + best["dchi_rh"]*basis.rhombic
    pairs, similar = [], set()
    included = tuple(c for c in snapshot.conformers if c.included)
    ref_order = included[0].geometry.raw_ref_ids
    aligned = []
    for conformer, rotation in zip(included, basis.alignment_row_rotations):
        mapping = dict(zip(conformer.geometry.raw_ref_ids, conformer.geometry.raw_coordinates))
        coordinates = np.asarray([mapping[ref] for ref in ref_order])-conformer.geometry.metal_position
        common_rotation = np.asarray(rotation)
        if conformer.coordinate_frame == "structure":
            common_rotation = common_rotation @ np.asarray(snapshot.orientation.rotation_matrix).T
        aligned.append(coordinates @ common_rotation)
    for left in range(pcs.shape[1]):
        for right in range(left+1, pcs.shape[1]):
            difference = pcs[:, left]-pcs[:, right]
            rms = float(np.sqrt(np.mean(difference**2)))
            scale = max(float(np.sqrt(np.mean(pcs[:, left]**2))),
                        float(np.sqrt(np.mean(pcs[:, right]**2))), 1e-12)
            relative = rms/scale
            close = relative <= relative_threshold
            if close:
                similar.add((left, right))
            pairs.append({"conformer_ids": [basis.conformer_ids[left], basis.conformer_ids[right]],
                          "rms_difference_ppm": rms, "relative_difference": relative, "similar": close,
                          "coordinate_rmsd_angstrom": float(np.sqrt(np.mean(np.sum((aligned[left]-aligned[right])**2, axis=1))))})
    groups = []
    for index in range(pcs.shape[1]):
        match = next((g for g in groups if all((min(i, index), max(i, index)) in similar for i in g)), None)
        if match is None:
            groups.append([index])
        else:
            match.append(index)
    return pairs, [g for g in groups if len(g) > 1]


def evaluate_reliability(result, options=ReliabilityOptions(), *, progress=None):
    """Assess the recorded successful fit. Refit every selected free field.

    progress is a plain callable for caller-owned messaging, never a widget.
    CV starts use recorded user/uniform initial values, not full-data fitted values.
    """
    from logic.conformer_workflow import validate_result, snapshot_from_payload
    result = validate_result(result)
    if not result or not result["success"]:
        raise ValueError("reliability requires a successful recorded MC fit")
    if not isinstance(options, ReliabilityOptions):
        options = ReliabilityOptions(**options)
    snapshot = snapshot_from_payload(result["input_snapshot"])
    fit_options, best = result["options"], result["best"]
    basis = build_basis(snapshot)
    observed = np.asarray([o.pcs for o in snapshot.observations])
    groups = observation_groups(snapshot)
    if len(groups) < options.folds:
        raise ValueError("not enough independent observation groups for the requested folds")
    sigma = [o.uncertainty_ppm for o in snapshot.observations]
    if options.method == "parametric":
        if any(s is None or s <= 0 for s in sigma):
            raise ValueError("parametric bootstrap requires positive known uncertainty for every PCS observation")
        if any(len(g) > 1 for g in groups):
            raise ValueError("overlapping observations require covariance; use grouped cases bootstrap")
    seeds = np.random.SeedSequence(options.seed).spawn(options.replicates+options.folds+1)
    shuffled = np.random.default_rng(seeds[-1]).permutation(len(groups))
    partitions = [sorted(i for group_index in part for i in groups[int(group_index)])
                  for part in np.array_split(shuffled, options.folds)]
    if any(len(observed)-len(test) < 3 for test in partitions):
        raise ValueError("CV partition leaves fewer than three training observations; add data or adjust folds")
    records = []
    for repetition in range(options.replicates):
        rng = np.random.default_rng(seeds[repetition])
        if options.method == "parametric":
            indices = list(range(len(observed)))
            measured = np.asarray(best["predictions"])+rng.normal(0., sigma)
        else:
            chosen = rng.integers(0, len(groups), size=len(groups))
            indices = [i for group_index in chosen for i in groups[int(group_index)]]
            measured = observed[indices]
        seed = int(rng.integers(0, 2**32))
        child, child_basis = _subset(snapshot, basis, indices, measured=measured, unique=True)
        record = {"replicate": repetition+1, "indices": indices, "measured_ppm": measured.tolist()}
        if len(indices) < 3:
            record.update(success=False, solver_seed=seed, attempted_starts=0,
                          messages=["fewer than three sampled observations"], population_ambiguous=None, estimate=None)
        else:
            record.update(_refit(child, fit_options, child_basis, seed))
        records.append(record)
        if progress:
            progress("bootstrap", repetition+1, options.replicates)

    fold_records = []
    for fold_index, test in enumerate(partitions):
        train = sorted(set(range(len(observed)))-set(test))
        if len(train) < 3:
            raise ValueError("CV partition leaves fewer than three training observations; reduce folds or add data")
        child, child_basis = _subset(snapshot, basis, train)
        seed = int(np.random.default_rng(seeds[options.replicates+fold_index]).integers(0, 2**32))
        record = {"fold": fold_index+1, "train_indices": train, "test_indices": test}
        record.update(_refit(child, fit_options, child_basis, seed))
        if record["success"]:
            estimate = record["estimate"]
            predicted = basis.predict(estimate["populations"], estimate["dchi_ax"], estimate["dchi_rh"])[test]
            residual = predicted-observed[test]
            record.update(predictions_ppm=predicted.tolist(), residuals_ppm=residual.tolist(),
                          test_rmsd_ppm=float(np.sqrt(np.mean(residual**2))))
        fold_records.append(record)
        if progress:
            progress("cross_validation", fold_index+1, options.folds)

    accepted = [r["estimate"] for r in records if r["success"]]
    minimum = max(20, math.ceil(.8*options.replicates))
    enough = len(accepted) >= minimum
    included = tuple(c for c in snapshot.conformers if c.included)
    diagnostics_snapshot = snapshot
    if fit_options["target"] == "populations":
        diagnostics_snapshot = replace(snapshot, tensor=replace(snapshot.tensor, fit_delta_chi_ax=False, fit_delta_chi_rh=False))
    elif fit_options["target"] == "tensor":
        start = population_start(snapshot, fit_options["initialization"])
        diagnostics_snapshot = replace(snapshot, conformers=tuple(
            replace(c, population_fixed=True, initial_population=float(start[list(basis.conformer_ids).index(c.conformer_id)]))
            if c.included else c for c in snapshot.conformers))
    diagnostic_basis = replace(basis, fingerprint=basis_fingerprint(diagnostics_snapshot))
    rank = asdict(joint_rank_diagnostics(diagnostics_snapshot, diagnostic_basis, best["populations"], best["dchi_ax"], best["dchi_rh"]))
    pairs, suggested = similarity_diagnostics(snapshot, basis, best, options.similarity_relative)
    identifiable = bool(rank["locally_full_rank"] and not result["population_ambiguous"])
    intervals = {"tensor": {}, "populations": []}
    for key, fit_flag in (("dchi_ax", snapshot.tensor.fit_delta_chi_ax), ("dchi_rh", snapshot.tensor.fit_delta_chi_rh)):
        fixed = fit_options["target"] == "populations" or not fit_flag
        intervals["tensor"][key] = {"estimate": best[key], "fixed": fixed,
                                    "interval": _interval([e[key] for e in accepted], options.confidence) if enough and not fixed and identifiable else None}
    for index, c in enumerate(included):
        fixed = c.population_fixed or fit_options["target"] == "tensor"
        group_ambiguous = any(index in g for g in suggested)
        values = [e["populations"][index] for e in accepted]
        intervals["populations"].append({"conformer_id": c.conformer_id, "estimate": best["populations"][index], "fixed": fixed,
            "interval": _interval(values, options.confidence) if enough and not fixed and identifiable and not group_ambiguous else None,
            "boundary_fraction": float(np.mean(np.asarray(values) <= 1e-8)) if values else None})
    aggregate = []
    for group in suggested:
        values = [sum(e["populations"][i] for i in group) for e in accepted]
        aggregate.append({"conformer_ids": [basis.conformer_ids[i] for i in group],
                          "population": float(sum(best["populations"][i] for i in group)),
                          "bootstrap_range": _interval(values, options.confidence) if enough else None,
                          "interpretation": "descriptive aggregate range; not proof of group identifiability"})
    successful_folds = [r for r in fold_records if r["success"]]
    residuals = [r for fold in successful_folds for r in fold["residuals_ppm"]]
    warnings = ["Conditional on fixed geometry, assignments, orientation and fixed parameters; no model-error or orientation uncertainty.",
                "Percentile intervals are approximate marginal estimates; boundary/nonlinear/multimodal coverage is not guaranteed."]
    if options.replicates < 200:
        warnings.append("Few bootstrap replicates: percentile endpoints can be unstable; increase the count for scientific interpretation.")
    if not enough:
        warnings.append("Too few successful bootstrap refits; intervals withheld.")
    if not identifiable:
        warnings.append("Rank deficiency or competing population solutions; individual parameter intervals withheld.")
    if suggested:
        warnings.append("Similar PCS columns detected at this tensor; inspect aggregate populations, not unique individual weights.")
    if rank["boundary_conformer_ids"] or any(p["boundary_fraction"] and p["boundary_fraction"] > .05 for p in intervals["populations"] if not p["fixed"]):
        warnings.append("Population boundary encountered; ordinary interior uncertainty interpretation is unreliable.")
    if len(successful_folds) < options.folds:
        warnings.append("Some CV refits failed; pooled successful-fold error is partial, not full cross-validation error.")
    payload = {"format": RELIABILITY_FORMAT, "schema_version": 1, "algorithm_version": RELIABILITY_VERSION,
               "source_input_fingerprint": result["input_fingerprint"],
               "source_snapshot_fingerprint": _digest(result["input_snapshot"]),
               "source_best": {k: best[k] for k in ("populations", "dchi_ax", "dchi_rh")},
               "options": asdict(options), "assumption": "independent Gaussian known-sigma errors" if options.method == "parametric" else "independent exchangeable observation groups; cases sampled with replacement",
               "observation_groups": groups, "bootstrap": {"requested": options.replicates, "successful": len(accepted),
                   "minimum_successful": minimum, "enough_successful_refits": enough,
                   "intervals_available": any(v["interval"] is not None for v in intervals["tensor"].values()) or
                                          any(v["interval"] is not None for v in intervals["populations"]),
                   "runs": records, "intervals": intervals},
               "cross_validation": {"folds": fold_records, "successful_folds": len(successful_folds), "predicted_observations": len(residuals),
                   "complete": len(successful_folds) == options.folds,
                   "rmsd_ppm": float(np.sqrt(np.mean(np.asarray(residuals)**2))) if residuals else None},
               "rank": rank, "similarity_pairs": pairs, "suggested_groups": aggregate, "warnings": warnings}
    return json.loads(json.dumps(payload, allow_nan=False))


def validate_reliability(value, result):
    """Validate optional saved diagnostics without running scientific refits."""
    if not isinstance(value, dict) or value.get("format") != RELIABILITY_FORMAT or value.get("schema_version") != 1 or value.get("algorithm_version") != RELIABILITY_VERSION:
        raise ValueError("unsupported MC reliability schema")
    options = ReliabilityOptions(**value["options"])
    best = result.get("best")
    if not best or value["source_input_fingerprint"] != result["input_fingerprint"] or value["source_snapshot_fingerprint"] != _digest(result["input_snapshot"]) or value["source_best"] != {k: best[k] for k in ("populations", "dchi_ax", "dchi_rh")}:
        raise ValueError("MC reliability does not match its recorded fit")
    bootstrap = value["bootstrap"]
    records = bootstrap["runs"]
    if len(records) != options.replicates or bootstrap["requested"] != options.replicates or bootstrap["successful"] != sum(r["success"] is True for r in records):
        raise ValueError("inconsistent MC reliability bootstrap counts")
    n, k = len(result["observation_ids"]), len(result["conformer_ids"])
    from logic.conformer_workflow import snapshot_from_payload
    snapshot = snapshot_from_payload(result["input_snapshot"])
    included = tuple(c for c in snapshot.conformers if c.included)
    target = result["options"]["target"]
    if value["observation_groups"] != observation_groups(snapshot):
        raise ValueError("inconsistent MC reliability observation groups")
    for record in records + value["cross_validation"]["folds"]:
        if not isinstance(record["success"], bool):
            raise ValueError("MC reliability success must be Boolean")
        indices = record.get("indices", record.get("train_indices"))
        if not indices or any(type(i) is not int or not 0 <= i < n for i in indices):
            raise ValueError("invalid MC reliability observation indices")
        estimate = record["estimate"]
        if record["success"] != (estimate is not None):
            raise ValueError("inconsistent MC reliability estimate")
        if estimate:
            weights = np.asarray(estimate["populations"])
            if weights.shape != (k,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1., atol=1e-10, rtol=0) or not np.isfinite([*weights, estimate["dchi_ax"], estimate["dchi_rh"], estimate["objective"], estimate["rmsd"]]).all():
                raise ValueError("invalid MC reliability parameters")
            if estimate["objective"] < 0 or estimate["rmsd"] < 0:
                raise ValueError("invalid MC reliability error metric")
            for i, conformer in enumerate(included):
                if (conformer.population_fixed and weights[i] != conformer.initial_population or
                    target == "tensor" and weights[i] != best["populations"][i]):
                    raise ValueError("MC reliability changed a fixed population")
            for key, free in (("dchi_ax", snapshot.tensor.fit_delta_chi_ax), ("dchi_rh", snapshot.tensor.fit_delta_chi_rh)):
                if (target == "populations" or not free) and estimate[key] != best[key]:
                    raise ValueError("MC reliability changed a fixed tensor")
        if "indices" in record and (len(record["measured_ppm"]) != len(indices) or not np.isfinite(record["measured_ppm"]).all()):
            raise ValueError("invalid MC bootstrap measurements")
    folds = value["cross_validation"]["folds"]
    if len(folds) != options.folds or sorted(i for f in folds for i in f["test_indices"]) != list(range(n)):
        raise ValueError("invalid MC reliability CV partition")
    for fold in folds:
        if any(type(i) is not int or not 0 <= i < n for i in fold["test_indices"]):
            raise ValueError("invalid MC reliability test indices")
        if set(fold["train_indices"]) != set(range(n))-set(fold["test_indices"]):
            raise ValueError("MC reliability CV train/test leakage")
        if any(set(g) & set(fold["test_indices"]) and not set(g) <= set(fold["test_indices"]) for g in value["observation_groups"]):
            raise ValueError("MC reliability split correlated observation members")
        if fold["success"]:
            residual = np.asarray(fold["residuals_ppm"], dtype=float)
            predicted = np.asarray(fold["predictions_ppm"], dtype=float)
            measured = np.asarray([snapshot.observations[i].pcs for i in fold["test_indices"]])
            if residual.shape != measured.shape or predicted.shape != measured.shape or not np.isfinite(predicted).all() or not np.allclose(residual, predicted-measured) or not np.isclose(fold["test_rmsd_ppm"], np.sqrt(np.mean(residual**2))):
                raise ValueError("inconsistent MC reliability CV predictions")
    accepted = [r["estimate"] for r in records if r["success"]]
    minimum = max(20, math.ceil(.8*options.replicates))
    enough = len(accepted) >= minimum
    if bootstrap["minimum_successful"] != minimum or bootstrap["enough_successful_refits"] != enough:
        raise ValueError("inconsistent MC reliability interval availability")
    items = list(bootstrap["intervals"]["tensor"].values()) + bootstrap["intervals"]["populations"]
    if (len(bootstrap["intervals"]["populations"]) != k or
        [p["conformer_id"] for p in bootstrap["intervals"]["populations"]] != result["conformer_ids"] or
        bootstrap["intervals_available"] != any(p["interval"] is not None for p in items)):
        raise ValueError("inconsistent MC reliability parameter intervals")
    def check_interval(item, estimates):
        interval = item["interval"]
        if interval is not None and (not enough or item["fixed"] or np.asarray(interval).shape != (2,) or
                                     not np.allclose(interval, _interval(estimates, options.confidence), rtol=1e-10, atol=1e-12)):
            raise ValueError("inconsistent MC reliability percentile interval")
    for key, item in bootstrap["intervals"]["tensor"].items():
        if key not in ("dchi_ax", "dchi_rh") or item["estimate"] != best[key]:
            raise ValueError("invalid MC reliability tensor interval")
        expected_fixed = target == "populations" or not getattr(snapshot.tensor, "fit_delta_chi_" + key[5:])
        if item["fixed"] != expected_fixed:
            raise ValueError("invalid MC reliability fixed tensor flag")
        check_interval(item, [e[key] for e in accepted])
    for i, item in enumerate(bootstrap["intervals"]["populations"]):
        if item["estimate"] != best["populations"][i] or item["fixed"] != (included[i].population_fixed or target == "tensor"):
            raise ValueError("invalid MC reliability population interval")
        check_interval(item, [e["populations"][i] for e in accepted])
    cv = value["cross_validation"]
    good = [f for f in folds if f["success"]]
    residuals = [r for f in good for r in f["residuals_ppm"]]
    expected_rmsd = float(np.sqrt(np.mean(np.asarray(residuals)**2))) if residuals else None
    if cv["successful_folds"] != len(good) or cv["complete"] != (len(good) == options.folds) or cv["predicted_observations"] != len(residuals) or (cv["rmsd_ppm"] is None) != (expected_rmsd is None) or expected_rmsd is not None and not np.isclose(cv["rmsd_ppm"], expected_rmsd):
        raise ValueError("inconsistent MC reliability pooled CV metric")
    for group in value["suggested_groups"]:
        ids = group["conformer_ids"]
        if len(ids) < 2 or len(set(ids)) != len(ids) or not set(ids) <= set(result["conformer_ids"]):
            raise ValueError("invalid MC reliability suggested group")
        indices = [result["conformer_ids"].index(c) for c in ids]
        if not np.isclose(group["population"], sum(best["populations"][i] for i in indices)):
            raise ValueError("invalid MC reliability aggregate population")
        if group["bootstrap_range"] is not None and (not enough or not np.allclose(group["bootstrap_range"], _interval([sum(e["populations"][i] for i in indices) for e in accepted], options.confidence))):
            raise ValueError("invalid MC reliability aggregate range")
    return value
