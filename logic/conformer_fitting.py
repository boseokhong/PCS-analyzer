"""Detached conformer inputs, engine-derived PCS bases and single-block fits.

No application state, UI or persistence integration lives here.
Ref IDs explicitly identify coordinate rows; file order never establishes identity.
Coordinates are in angstrom, susceptibility in 1e-32 m3, and predictions in ppm.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import fsum

import numpy as np
from scipy.optimize import minimize

from logic.pcs_fit_snapshot import FitGeometrySnapshot, FitObservation, FitTensorSnapshot
from logic.pcs_forward_api import PCSFrame, PCSStructureTarget
from logic.pcs_forward_model import PCS_FORWARD_ENGINE_ID, PCS_FORWARD_ENGINE_VERSION
from logic.pcs_fitting_adapter import predict_structured_pcs_from_factors
from logic.symmetry_geometry import effective_geometry_factors


MC_MODEL_VERSION = "shared-fixed-orientation-v1"
MC_BASIS_VERSION = "1"
MC_SOLVER_VERSION = "blocks-v1"
MC_JOINT_SOLVER_VERSION = "alternating-multistart-v1"


def _coordinates(values, count, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (count, 3) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {count} finite XYZ rows")
    return tuple(tuple(float(x) for x in row) for row in array)


def _refs(values, name):
    refs = tuple(int(x) for x in values)
    if any(x != original or x < 0 for x, original in zip(refs, values)):
        raise ValueError(f"{name} must contain nonnegative integer Ref IDs")
    if len(set(refs)) != len(refs):
        raise ValueError(f"{name} contains duplicate Ref IDs")
    return refs


def _freeze_geometry(geometry):
    raw = _refs(geometry.raw_ref_ids, "raw mapping")
    effective = _refs(geometry.effective_ref_ids, "effective mapping")
    if not raw or not effective:
        raise ValueError("raw and effective geometry must be nonempty")
    pseudo = tuple((int(rid), _refs(members, "pseudo members"))
                   for rid, members in geometry.pseudo_members)
    if len(dict(pseudo)) != len(pseudo):
        raise ValueError("duplicate pseudo Ref IDs")
    for rid, members in pseudo:
        if rid not in effective or not members or not set(members) <= set(raw):
            raise ValueError("pseudo geometry requires all raw members and an effective row")
    if not set(effective) <= set(raw) | set(dict(pseudo)):
        raise ValueError("effective Ref requires raw identity or explicit pseudo members")
    groups = tuple(replace(g, axis_atoms=_refs(g.axis_atoms, "torsion axis"),
                           rotating_atoms=_refs(g.rotating_atoms, "torsion members"))
                   for g in geometry.torsion_groups)
    xyz = _coordinates(geometry.raw_coordinates, len(raw), "raw geometry")
    raw_map = dict(zip(raw, xyz))
    for group in groups if geometry.torsion_enabled else ():
        if not group.enabled:
            continue
        if (len(group.axis_atoms) != 2 or not group.rotating_atoms
                or not set(group.axis_atoms + group.rotating_atoms) <= set(raw)):
            raise ValueError("torsion axis/members must exist in raw geometry")
        if (group.n_samples < 1 or int(group.n_samples) != group.n_samples
                or group.range_mode not in ("full", "restricted")
                or not np.isfinite([group.range_start_deg, group.range_end_deg]).all()):
            raise ValueError("invalid torsion sampling policy")
        if np.linalg.norm(np.subtract(raw_map[group.axis_atoms[0]],
                                      raw_map[group.axis_atoms[1]])) < 1e-12:
            raise ValueError("torsion axis has zero length")
    return FitGeometrySnapshot(
        effective, _coordinates(geometry.effective_coordinates, len(effective), "effective geometry"),
        raw, xyz, _coordinates([geometry.metal_position], 1, "metal")[0],
        pseudo, bool(geometry.torsion_enabled), groups,
    )


@dataclass(frozen=True)
class ConformerSnapshot:
    conformer_id: str
    name: str
    geometry: FitGeometrySnapshot
    # Element at each raw mapping row; explicit mapping confirmation is separate.
    elements: tuple[str, ...]
    mapping_provenance: str
    frame_provenance: str
    coordinate_frame: str = "structure"  # structure or already tensor-oriented
    source_hash: str = ""
    included: bool = True
    initial_population: float = 1.0
    population_fixed: bool = False

    def __post_init__(self):
        if not self.conformer_id.strip() or not self.mapping_provenance.strip():
            raise ValueError("stable conformer ID and confirmed mapping provenance are required")
        if not self.frame_provenance.strip() or self.coordinate_frame not in ("structure", "tensor"):
            raise ValueError("explicit frame kind and provenance are required")
        geometry = _freeze_geometry(self.geometry)
        elements = tuple(str(x).strip() for x in self.elements)
        if len(elements) != len(geometry.raw_ref_ids) or any(not x for x in elements):
            raise ValueError("elements must identify every raw coordinate row")
        if not np.isfinite(self.initial_population) or self.initial_population < 0:
            raise ValueError("initial population must be finite and nonnegative")
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "elements", elements)


@dataclass(frozen=True)
class MultiConformerSnapshot:
    conformers: tuple[ConformerSnapshot, ...]
    observations: tuple[FitObservation, ...]
    orientation: PCSFrame = PCSFrame()
    tensor: FitTensorSnapshot = FitTensorSnapshot(True, True, 0.0, 0.0)
    alignment_mode: str = "confirmed_common_frame"
    core_ref_ids: tuple[int, ...] = ()
    weighting_mode: str = "unweighted"
    run_id: str = ""
    document_id: str = ""
    document_revision: int = 0

    def __post_init__(self):
        conformers = tuple(self.conformers)
        if not conformers or len({c.conformer_id for c in conformers}) != len(conformers):
            raise ValueError("conformers must have unique stable IDs")
        included = tuple(c for c in conformers if c.included)
        if not included:
            raise ValueError("at least one included conformer is required")
        if self.orientation.metal_position != (0.0, 0.0, 0.0):
            raise ValueError("common orientation must have zero origin; metals belong to conformers")
        if not np.isfinite([self.tensor.fixed_delta_chi_ax, self.tensor.fixed_delta_chi_rh]).all():
            raise ValueError("tensor scalar inputs must be finite")
        if self.alignment_mode not in ("confirmed_common_frame", "metal_centered_core"):
            raise ValueError("unsupported alignment mode")
        core = _refs(self.core_ref_ids, "alignment core")
        if self.alignment_mode == "metal_centered_core":
            if len(core) < 3 or any(c.coordinate_frame != "structure" for c in included):
                raise ValueError("core alignment requires three anchors in structure frames")
            for c in included:
                _core_coordinates(c, core)
        reference_elements = dict(zip(included[0].geometry.raw_ref_ids, included[0].elements))
        reference_pseudo = dict(included[0].geometry.pseudo_members)
        for c in included[1:]:
            if dict(zip(c.geometry.raw_ref_ids, c.elements)) != reference_elements:
                raise ValueError(f"{c.conformer_id}: atom identities/elements differ")
            if dict(c.geometry.pseudo_members) != reference_pseudo:
                raise ValueError(f"{c.conformer_id}: pseudo member definitions differ")
        observations = []
        for obs in self.observations:
            if not np.isfinite(obs.pcs):
                raise ValueError("observed PCS must be finite")
            if obs.uncertainty_ppm is not None and (
                    not np.isfinite(obs.uncertainty_ppm) or obs.uncertainty_ppm < 0):
                raise ValueError("uncertainty must be finite and nonnegative")
            target = obs.target
            # Only the numerical target contract belongs to the detached snapshot.
            detached = PCSStructureTarget(target.target_id, target.site_member_ref_ids,
                                          target.member_reduction, target.site_reduction)
            observations.append(replace(obs, target=detached))
            for c in included:
                available = set(c.geometry.raw_ref_ids) | set(c.geometry.effective_ref_ids)
                if not set(detached.member_ref_ids) <= available:
                    raise ValueError(f"{c.conformer_id}: observation {obs.observable_id} has missing members")
        if (not observations or len({o.observable_id for o in observations}) != len(observations)
                or len({o.target.target_id for o in observations}) != len(observations)):
            raise ValueError("observations and targets must have unique stable IDs")
        if self.weighting_mode not in ("unweighted", "experimental_uncertainty"):
            raise ValueError("unsupported objective weighting")
        if self.weighting_mode == "experimental_uncertainty" and any(
                o.uncertainty_ppm is None or o.uncertainty_ppm <= 0 for o in observations):
            raise ValueError("weighted objective requires positive uncertainty for every observation")
        object.__setattr__(self, "conformers", conformers)
        object.__setattr__(self, "observations", tuple(observations))
        object.__setattr__(self, "core_ref_ids", core)


def _core_coordinates(conformer, refs):
    geometry = conformer.geometry
    mapping = dict(zip(geometry.raw_ref_ids, geometry.raw_coordinates))
    if not set(refs) <= set(mapping):
        raise ValueError("alignment core is missing raw atom identities")
    xyz = np.asarray([mapping[r] for r in refs]) - geometry.metal_position
    if np.linalg.matrix_rank(xyz - xyz.mean(axis=0)) < 2 or np.linalg.matrix_rank(xyz) < 2:
        raise ValueError("alignment core is collinear or degenerate")
    return xyz


def _alignment(conformer, reference, snapshot):
    if snapshot.alignment_mode == "confirmed_common_frame":
        return np.eye(3), 0.0
    source = _core_coordinates(conformer, snapshot.core_ref_ids)
    target = _core_coordinates(reference, snapshot.core_ref_ids)
    u, _, vt = np.linalg.svd(source.T @ target)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt  # row-vector rotation; no centroid translation
    rmsd = float(np.sqrt(np.mean(np.sum((source @ rotation - target) ** 2, axis=1))))
    return rotation, rmsd


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()).hexdigest()


def basis_fingerprint(snapshot):
    """Numerical dependencies only: scalar, population and measured PCS edits reuse bases."""
    return _digest({
        "model": MC_MODEL_VERSION, "basis": MC_BASIS_VERSION,
        "engine": (PCS_FORWARD_ENGINE_ID, PCS_FORWARD_ENGINE_VERSION),
        "orientation": asdict(snapshot.orientation), "alignment": snapshot.alignment_mode,
        "core": snapshot.core_ref_ids,
        "conformers": [(c.conformer_id, c.coordinate_frame, asdict(c.geometry))
                       for c in snapshot.conformers if c.included],
        "targets": [(o.observable_id, o.target.target_id, o.target.site_member_ref_ids,
                     o.target.member_reduction, o.target.site_reduction) for o in snapshot.observations],
    })


def objective_fingerprint(snapshot):
    """Scientific objective identity; unused uncertainties do not stale unweighted runs."""
    return _digest({"basis": basis_fingerprint(snapshot), "tensor": asdict(snapshot.tensor),
                    "populations": [(c.conformer_id, c.initial_population, c.population_fixed)
                                    for c in snapshot.conformers if c.included],
                    "weighting": snapshot.weighting_mode,
                    "observed": [(o.observable_id, o.pcs,
                                  o.uncertainty_ppm if snapshot.weighting_mode != "unweighted" else None)
                                 for o in snapshot.observations]})


@dataclass(frozen=True)
class ConformerBasis:
    fingerprint: str
    conformer_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    axial: np.ndarray
    rhombic: np.ndarray
    alignment_row_rotations: tuple
    alignment_rmsd_angstrom: tuple[float, ...]

    def predict(self, populations, dchi_ax, dchi_rh):
        weights = np.asarray(populations, dtype=float)
        if (weights.shape != (len(self.conformer_ids),) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-10)):
            raise ValueError("populations must be finite, nonnegative and sum to one")
        if not np.isfinite([dchi_ax, dchi_rh]).all():
            raise ValueError("tensor components must be finite")
        return dchi_ax * (self.axial @ weights) + dchi_rh * (self.rhombic @ weights)


def build_basis(snapshot):
    included = tuple(c for c in snapshot.conformers if c.included)
    targets = tuple(o.target for o in snapshot.observations)
    axial, rhombic, rotations, rmsds = [], [], [], []
    for conformer in included:
        geometry = conformer.geometry
        row_rotation, rmsd = _alignment(conformer, included[0], snapshot)
        rotations.append(tuple(tuple(float(x) for x in row) for row in row_rotation))
        rmsds.append(rmsd)
        rotation = row_rotation
        if conformer.coordinate_frame == "structure":
            rotation = rotation @ np.asarray(snapshot.orientation.rotation_matrix).T
        raw = (geometry.raw_array() - geometry.metal_position) @ rotation
        effective = (geometry.effective_array() - geometry.metal_position) @ rotation
        raw_map = dict(zip(geometry.raw_ref_ids, raw))
        effective_map = dict(zip(geometry.effective_ref_ids, effective))
        refs = tuple(dict.fromkeys(rid for t in targets for rid in t.member_ref_ids))
        coords = np.asarray([effective_map[r] if r in effective_map else raw_map[r] for r in refs])
        _, _, _, gax, grh = effective_geometry_factors(
            refs, coords, pseudo_members=geometry.pseudo_member_mapping(),
            raw_coords_by_id=raw_map, torsion_groups=geometry.active_torsion_mappings())
        axial.append(predict_structured_pcs_from_factors(refs, gax, grh, 1.0, 0.0, targets))
        rhombic.append(predict_structured_pcs_from_factors(refs, gax, grh, 0.0, 1.0, targets))
    def immutable_matrix(columns):
        values = np.asarray(columns, dtype=float).T.copy()
        if not np.isfinite(values).all():
            raise ValueError("nonfinite PCS basis")
        # Immutable bytes backing prevents callers from re-enabling writeability.
        return np.frombuffer(values.tobytes(), dtype=float).reshape(values.shape)
    return ConformerBasis(basis_fingerprint(snapshot), tuple(c.conformer_id for c in included),
                          tuple(o.observable_id for o in snapshot.observations),
                          immutable_matrix(axial), immutable_matrix(rhombic),
                          tuple(rotations), tuple(rmsds))


class ConformerBasisCache:
    """Caller-owned bounded LRU; serialize access when used by multiple threads.

    Numerical results are immutable; cache bookkeeping itself is not thread-safe.
    """
    def __init__(self, max_entries=8, max_bytes=32 * 1024 * 1024):
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("positive cache limits required")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._entries = OrderedDict()
        self._bytes = 0

    def get(self, snapshot):
        key = basis_fingerprint(snapshot)
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        basis = build_basis(snapshot)
        size = basis.axial.nbytes + basis.rhombic.nbytes
        if size <= self.max_bytes:
            while self._entries and (len(self._entries) >= self.max_entries
                                     or self._bytes + size > self.max_bytes):
                _, old = self._entries.popitem(last=False)
                self._bytes -= old.axial.nbytes + old.rhombic.nbytes
            self._entries[key] = basis
            self._bytes += size
        return basis

    def clear(self):
        self._entries.clear()
        self._bytes = 0


def population_start(snapshot, initialization="user"):
    """Preserve fixed values; distribute only the remaining mass to free rows."""
    if initialization not in ("user", "uniform"):
        raise ValueError("initialization must be user or uniform")
    included = tuple(c for c in snapshot.conformers if c.included)
    fixed_sum = fsum(c.initial_population for c in included if c.population_fixed)
    if fixed_sum > 1.0:
        raise ValueError("fixed populations sum to more than one")
    free = [i for i, c in enumerate(included) if not c.population_fixed]
    mass = 1.0 - fixed_sum
    if not free and not np.isclose(fixed_sum, 1.0, rtol=0, atol=1e-12):
        raise ValueError("all-fixed populations must sum to one")
    weights = np.asarray([c.initial_population for c in included], dtype=float)
    if free:
        supplied = weights[free]
        # Scaling before summation avoids overflow for large finite initial values.
        maximum = float(supplied.max())
        if initialization == "uniform" or maximum == 0.0:
            proportions = np.full(len(free), 1.0 / len(free))
        else:
            proportions = supplied / maximum
            proportions /= proportions.sum()
        weights[free] = mass * proportions
    return tuple(float(x) for x in weights)


def _block_inputs(snapshot, basis, populations):
    if len(snapshot.observations) < 3:
        raise ValueError("at least three matched PCS observations are required")
    basis = build_basis(snapshot) if basis is None else basis
    included = tuple(c for c in snapshot.conformers if c.included)
    if (basis.fingerprint != basis_fingerprint(snapshot)
            or basis.conformer_ids != tuple(c.conformer_id for c in included)
            or basis.observation_ids != tuple(o.observable_id for o in snapshot.observations)):
        raise ValueError("basis does not match the execution snapshot")
    # Validate fixed-population feasibility even when a caller supplies a start.
    default_start = population_start(snapshot)
    weights = np.asarray(default_start if populations is None else populations, dtype=float)
    basis.predict(weights, 0.0, 0.0)  # shared simplex/shape validation
    for i, c in enumerate(included):
        if c.population_fixed and weights[i] != c.initial_population:
            raise ValueError("supplied populations changed a fixed value")
    remaining_mass = 1.0 - fsum(c.initial_population for c in included if c.population_fixed)
    free_weights = weights[[i for i, c in enumerate(included) if not c.population_fixed]]
    if remaining_mass == 0 and np.any(free_weights != 0):
        raise ValueError("zero remaining mass requires zero free populations")
    observed = np.asarray([o.pcs for o in snapshot.observations], dtype=float)
    scales = (np.ones(len(observed)) if snapshot.weighting_mode == "unweighted" else
              np.asarray([o.uncertainty_ppm for o in snapshot.observations], dtype=float))
    return basis, included, weights.copy(), observed, scales


@dataclass(frozen=True)
class ConformerBlockResult:
    """Numerical candidate and acceptance diagnostics, not a persisted UI result schema."""
    snapshot: MultiConformerSnapshot
    kind: str
    model_version: str
    solver_version: str
    input_fingerprint: str
    conformer_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    initial_populations: tuple[float, ...]
    populations: tuple[float, ...]
    dchi_ax: float
    dchi_rh: float
    predictions: tuple[float, ...]
    residuals: tuple[float, ...]
    objective: float
    initial_objective: float
    rmsd: float
    success: bool
    message: str
    iterations: int
    free_parameters: tuple[str, ...]
    rank: int
    singular_values: tuple[float, ...]
    identifiable: bool
    stationarity: float


def _metrics(predicted, observed, scales):
    residual = predicted - observed
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        objective = float(np.sum((residual / scales) ** 2))
        rmsd = float(np.sqrt(np.mean(residual ** 2)))
    if not np.isfinite([objective, rmsd]).all():
        raise ValueError("objective is nonfinite; check numerical input scale")
    return residual, objective, rmsd


def _block_result(snapshot, basis, kind, start, weights, ax, rh, observed, scales,
                  initial_objective, success, message, iterations, free_parameters,
                  rank, singular_values, stationarity, options=None):
    prediction = basis.predict(weights, ax, rh)
    residual, objective, rmsd = _metrics(prediction, observed, scales)
    return ConformerBlockResult(
        snapshot, kind, MC_MODEL_VERSION, MC_SOLVER_VERSION,
        _digest({"objective": objective_fingerprint(snapshot), "solver": MC_SOLVER_VERSION,
                 "kind": kind, "start": tuple(start), "options": options or {}}),
        basis.conformer_ids, basis.observation_ids, tuple(map(float, start)),
        tuple(map(float, weights)), float(ax), float(rh), tuple(map(float, prediction)),
        tuple(map(float, residual)), objective, initial_objective, rmsd,
        bool(success), str(message), int(iterations), tuple(free_parameters), int(rank),
        tuple(map(float, singular_values)), bool(rank == len(free_parameters)), float(stationarity),
    )


def fit_tensor(snapshot, *, basis=None, populations=None):
    """Weighted linear tensor block at supplied feasible, fixed populations.

    Fixed tensor components are subtracted from the target and never enter the
    least-squares variables. Rank-deficient solutions are explicitly nonunique.
    """
    basis, _, weights, observed, scales = _block_inputs(snapshot, basis, populations)
    tensor = snapshot.tensor
    values = np.array([tensor.fixed_delta_chi_ax, tensor.fixed_delta_chi_rh], dtype=float)
    free = np.flatnonzero([tensor.fit_delta_chi_ax, tensor.fit_delta_chi_rh])
    columns = np.column_stack((basis.axial @ weights, basis.rhombic @ weights))
    _, initial_objective, _ = _metrics(columns @ values, observed, scales)
    fixed = np.setdiff1d(np.arange(2), free)
    matrix = columns[:, free] / scales[:, None]
    target = (observed - columns[:, fixed] @ values[fixed]) / scales
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite weighted tensor system")
    rank, singular, stationarity = 0, (), 0.0
    if len(free):
        normalizer = max(float(np.max(np.abs(matrix))), float(np.max(np.abs(target))), 1e-150)
        normalized_matrix, normalized_target = matrix / normalizer, target / normalizer
        solution, _, rank, singular = np.linalg.lstsq(normalized_matrix, normalized_target, rcond=None)
        singular = singular * normalizer
        values[free] = solution
        gradient = normalized_matrix.T @ (normalized_matrix @ solution - normalized_target)
        matrix_norm = float(np.linalg.norm(normalized_matrix))
        denominator = max(matrix_norm * (matrix_norm * float(np.linalg.norm(solution))
                          + float(np.linalg.norm(normalized_target))), 1e-300)
        stationarity = float(np.linalg.norm(gradient, ord=np.inf)) / denominator
    _, objective, _ = _metrics(columns @ values, observed, scales)
    improved = objective <= initial_objective + 1e-10 * max(1.0, initial_objective)
    success = improved and np.isfinite(stationarity) and stationarity <= 1e-10
    message = ("Tensor block solved" if len(free) else "All tensor components fixed; evaluated only")
    if rank < len(free):
        message += "; rank deficient: minimum-norm components are not unique"
    if not improved:
        message += "; objective increased"
    if not np.isfinite(stationarity) or stationarity > 1e-10:
        message += "; normal-equation stationarity check failed"
    names = tuple(("dchi_ax", "dchi_rh")[i] for i in free)
    return _block_result(snapshot, basis, "tensor", weights, weights, *values, observed, scales,
                         initial_objective, success, message, int(bool(len(free))),
                         names, rank, singular, stationarity)


def _simplex_projection(values):
    """Euclidean projection onto the unit simplex, used only for KKT checking."""
    values = values - np.max(values)  # projection is invariant to a common shift
    ordered = np.sort(values)[::-1]
    offsets = (np.cumsum(ordered) - 1.0) / np.arange(1, len(values) + 1)
    active = np.flatnonzero(ordered > offsets)
    return np.maximum(values - offsets[active[-1]], 0.0)


def fit_populations(snapshot, *, basis=None, populations=None, initialization="user",
                    max_iterations=500, tolerance=1e-12, stationarity_tolerance=1e-7):
    """Convex population block at the supplied tensor, preserving fixed mass.

    SLSQP uses an analytical gradient and equality constraint. Acceptance also
    requires feasibility, no objective increase and projected-gradient stationarity.
    No OptimizeResult.multipliers, covariance or dense D.T@D is required.
    """
    if initialization not in ("user", "uniform"):
        raise ValueError("initialization must be user or uniform")
    if (not np.isfinite(max_iterations) or int(max_iterations) != max_iterations or max_iterations < 1
            or not np.isfinite([tolerance, stationarity_tolerance]).all()
            or tolerance <= 0 or stationarity_tolerance <= 0):
        raise ValueError("positive finite solver tolerances and iteration limit required")
    supplied = population_start(snapshot, initialization) if populations is None else populations
    basis, included, start, observed, scales = _block_inputs(snapshot, basis, supplied)
    free = np.array([i for i, c in enumerate(included) if not c.population_fixed], dtype=int)
    fixed = np.array([i for i, c in enumerate(included) if c.population_fixed], dtype=int)
    mass = 1.0 - fsum(c.initial_population for c in included if c.population_fixed)
    ax, rh = snapshot.tensor.fixed_delta_chi_ax, snapshot.tensor.fixed_delta_chi_rh
    design = ax * basis.axial + rh * basis.rhombic
    _, initial_objective, _ = _metrics(design @ start, observed, scales)
    weights = start.copy()
    dimensions = max(0, len(free) - 1) if mass > 0 else 0
    rank, singular, stationarity, iterations = 0, (), 0.0, 0
    success, message = True, "Populations determined by fixed mass; evaluated only"
    if len(free) > 1 and mass > 0:
        # Fractional free weights sum to one even when remaining mass is small.
        matrix = mass * design[:, free] / scales[:, None]
        target = (observed - design[:, fixed] @ weights[fixed]) / scales
        # Remove the common column via the equality, improving scale robustness.
        target = target - matrix[:, -1]
        matrix = matrix - matrix[:, -1, None]
        if not np.isfinite(matrix).all() or not np.isfinite(target).all():
            raise ValueError("nonfinite weighted population system")
        singular = np.linalg.svd(matrix[:, :-1], compute_uv=False)
        rank = int(np.linalg.matrix_rank(matrix[:, :-1]))
        normalizer = max(float(np.max(np.abs(matrix))), float(np.max(np.abs(target))), 1e-150)
        matrix, target = matrix / normalizer, target / normalizer

        def cost(x):
            residual = matrix @ x - target
            return float(residual @ residual)

        def gradient(x):
            return 2.0 * (matrix.T @ (matrix @ x - target))

        result = minimize(cost, start[free] / mass, jac=gradient, method="SLSQP",
                          bounds=[(0.0, 1.0)] * len(free),
                          constraints={"type": "eq", "fun": lambda x: float(x.sum() - 1.0),
                                       "jac": lambda x: np.ones(len(x))},
                          options={"maxiter": int(max_iterations), "ftol": float(tolerance)})
        x = np.asarray(result.x, dtype=float)
        feasible = (x.shape == (len(free),) and np.isfinite(x).all()
                    and np.all(x >= 0) and np.all(x <= 1)
                    and abs(x.sum() - 1.0) <= 1e-10)
        if not feasible:
            # Keep a safe start for failed candidates; do not normalize/clip a bad solution.
            success, message = False, f"Population solver produced infeasible candidate: {result.message}"
            lipschitz = 2.0 * float(np.sum(matrix ** 2))
            safe_x = start[free] / mass
            stationarity = (float(np.max(np.abs(safe_x - _simplex_projection(
                safe_x - gradient(safe_x) / lipschitz)))) if lipschitz > 0 else 0.0)
        else:
            weights[free] = mass * x
            lipschitz = 2.0 * float(np.sum(matrix ** 2))
            stationarity = (float(np.max(np.abs(x - _simplex_projection(x - gradient(x) / lipschitz))))
                            if lipschitz > 0 else 0.0)
            _, objective, _ = _metrics(design @ weights, observed, scales)
            improved = objective <= initial_objective + 1e-10 * max(1.0, initial_objective)
            success = bool(result.success and improved and stationarity <= stationarity_tolerance)
            message = str(result.message)
            if not improved:
                message += "; objective increased"
            if stationarity > stationarity_tolerance:
                message += "; projected-gradient stationarity check failed"
        iterations = int(result.nit)
        if rank < dimensions:
            message += "; population contrasts are rank deficient (uniqueness not established)"
    names = tuple(f"population_contrast:{included[i].conformer_id}" for i in free[:-1]) if dimensions else ()
    return _block_result(snapshot, basis, "populations", start, weights, ax, rh, observed, scales,
                         initial_objective, success, message, iterations, names, rank, singular,
                         stationarity, {"initialization": initialization, "max_iterations": int(max_iterations),
                                        "tolerance": float(tolerance),
                                        "stationarity_tolerance": float(stationarity_tolerance)})


@dataclass(frozen=True)
class JointFitOptions:
    seed: int = 0
    starts: int = 8
    initialization: str = "user"
    max_iterations: int = 300
    objective_tolerance: float = 1e-10
    parameter_tolerance: float = 1e-7
    stationarity_tolerance: float = 1e-7
    population_max_iterations: int = 500
    population_tolerance: float = 1e-14
    rank_relative_tolerance: float = 1e-8
    alternative_objective_absolute: float = 1e-10
    alternative_objective_relative: float = 1e-4
    alternative_population_distance: float = .05

    def __post_init__(self):
        for name in ("seed", "starts", "max_iterations", "population_max_iterations"):
            value = getattr(self, name)
            if not np.isfinite(value) or int(value) != value or value < (0 if name == "seed" else 1):
                raise ValueError(f"invalid joint option: {name}")
            object.__setattr__(self, name, int(value))
        if self.starts > 20 or self.initialization not in ("user", "uniform"):
            raise ValueError("joint starts must be 1..20 and initialization user or uniform")
        for name in ("objective_tolerance", "parameter_tolerance", "stationarity_tolerance",
                     "population_tolerance", "rank_relative_tolerance",
                     "alternative_objective_absolute", "alternative_objective_relative",
                     "alternative_population_distance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"positive finite joint option required: {name}")
            object.__setattr__(self, name, value)
        if self.rank_relative_tolerance >= 1 or self.alternative_population_distance > 1:
            raise ValueError("invalid rank tolerance or population distance")


@dataclass(frozen=True)
class JointRankDiagnostics:
    parameter_names: tuple[str, ...]
    parameter_scales: tuple[float, ...]
    singular_values: tuple[float, ...]
    rank_threshold: float
    rank: int
    locally_full_rank: bool
    population_dimensions: int
    effective_population_rank: int
    populations_distinguishable: bool
    boundary_conformer_ids: tuple[str, ...]


@dataclass(frozen=True)
class ConformerJointRun:
    initial_populations: tuple[float, ...]
    populations: tuple[float, ...]
    dchi_ax: float
    dchi_rh: float
    predictions: tuple[float, ...]
    residuals: tuple[float, ...]
    objective: float
    rmsd: float
    objective_history: tuple[float, ...]
    rejected_updates: tuple[str, ...]
    iterations: int
    success: bool
    message: str
    tensor_stationarity: float
    population_stationarity: float
    parameter_change: float
    diagnostics: JointRankDiagnostics


@dataclass(frozen=True)
class ConformerJointResult:
    snapshot: MultiConformerSnapshot
    options: JointFitOptions
    model_version: str
    solver_version: str
    input_fingerprint: str
    conformer_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    runs: tuple[ConformerJointRun, ...]
    selected_index: int | None
    alternative_indices: tuple[int, ...]
    population_ambiguous: bool | None

    @property
    def best(self):
        return None if self.selected_index is None else self.runs[self.selected_index]

    @property
    def success(self):
        return self.selected_index is not None


def _joint_layout(snapshot):
    included = tuple(c for c in snapshot.conformers if c.included)
    free = np.array([i for i, c in enumerate(included) if not c.population_fixed], dtype=int)
    mass = 1.0 - fsum(c.initial_population for c in included if c.population_fixed)
    tensor_free = np.flatnonzero([snapshot.tensor.fit_delta_chi_ax, snapshot.tensor.fit_delta_chi_rh])
    return included, free, mass, tensor_free


def _joint_stationarity(snapshot, basis, weights, ax, rh, observed, scales):
    included, free, mass, tensor_free = _joint_layout(snapshot)
    values = np.array([ax, rh])
    columns = np.column_stack((basis.axial @ weights, basis.rhombic @ weights))
    tensor_step = 0.0
    if len(tensor_free):
        fixed = np.setdiff1d(np.arange(2), tensor_free)
        matrix = columns[:, tensor_free] / scales[:, None]
        target = (observed - columns[:, fixed] @ values[fixed]) / scales
        normalizer = max(float(np.max(np.abs(matrix))), float(np.max(np.abs(target))), 1e-150)
        matrix, target = matrix / normalizer, target / normalizer
        gradient = matrix.T @ (matrix @ values[tensor_free] - target)
        norm = float(np.linalg.norm(matrix))
        denominator = max(norm * (norm * float(np.linalg.norm(values[tensor_free]))
                                  + float(np.linalg.norm(target))), 1e-300)
        tensor_step = float(np.linalg.norm(gradient, ord=np.inf)) / denominator
    population_step = 0.0
    if len(free) > 1 and mass > 0:
        fixed = [i for i, c in enumerate(included) if c.population_fixed]
        design = ax * basis.axial + rh * basis.rhombic
        matrix = mass * design[:, free] / scales[:, None]
        target = (observed - design[:, fixed] @ weights[fixed]) / scales - matrix[:, -1]
        matrix = matrix - matrix[:, -1, None]
        normalizer = max(float(np.max(np.abs(matrix))), float(np.max(np.abs(target))), 1e-150)
        matrix, target = matrix / normalizer, target / normalizer
        x = weights[free] / mass
        lipschitz = 2.0 * float(np.sum(matrix ** 2))
        if lipschitz > 0:
            gradient = 2.0 * (matrix.T @ (matrix @ x - target))
            population_step = float(np.max(np.abs(x - _simplex_projection(x - gradient / lipschitz))))
    return tensor_step, population_step


def joint_rank_diagnostics(snapshot, basis, populations, dchi_ax, dchi_rh, *, relative_tolerance=1e-8):
    """Local weighted Jacobian rank; no covariance or global uniqueness claim.

    Tensor variables use a common susceptibility scale max(|ax|, |rh|, 1).
    Population variables are fractions of remaining free mass. Equality/fixed
    variables are eliminated. Boundary IDs are retained for tangent-cone review.
    """
    if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("rank relative tolerance must be between zero and one")
    basis, included, weights, _, scales = _block_inputs(snapshot, basis, populations)
    _, free, mass, tensor_free = _joint_layout(snapshot)
    values = np.array([dchi_ax, dchi_rh], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("tensor components must be finite")
    tensor_scale = max(float(np.max(np.abs(values))), 1.0)
    tensor_columns = np.column_stack((basis.axial @ weights, basis.rhombic @ weights))[:, tensor_free]
    tensor_jacobian = tensor_columns / scales[:, None] * tensor_scale
    dimensions = max(0, len(free) - 1) if mass > 0 else 0
    design = dchi_ax * basis.axial + dchi_rh * basis.rhombic
    population_jacobian = (mass * (design[:, free[:-1]] - design[:, free[-1], None]) / scales[:, None]
                           if dimensions else np.empty((len(scales), 0)))
    jacobian = np.column_stack((tensor_jacobian, population_jacobian))
    if not np.isfinite(jacobian).all():
        raise ValueError("joint sensitivity is nonfinite")
    singular = np.linalg.svd(jacobian, compute_uv=False) if jacobian.shape[1] else np.array([])
    threshold = float(singular[0] * relative_tolerance) if len(singular) else 0.0
    rank = int(np.sum(singular > threshold))
    projected = population_jacobian
    if dimensions and len(tensor_free):
        # Thin projection: never form an observation-by-observation projector.
        u, tensor_singular, _ = np.linalg.svd(tensor_jacobian, full_matrices=False)
        q = u[:, tensor_singular > threshold]
        projected = projected - q @ (q.T @ projected)
    pop_singular = np.linalg.svd(projected, compute_uv=False) if dimensions else np.array([])
    population_rank = int(np.sum(pop_singular > threshold))
    names = tuple(("dchi_ax", "dchi_rh")[i] for i in tensor_free)
    names += tuple(f"population_fraction_contrast:{included[i].conformer_id}" for i in free[:-1]) if dimensions else ()
    boundaries = tuple(included[i].conformer_id for i in free if weights[i] <= 1e-8)
    return JointRankDiagnostics(names, (tensor_scale,) * len(tensor_free) + (1.0,) * dimensions,
                                tuple(map(float, singular)), threshold, rank, rank == jacobian.shape[1],
                                dimensions, population_rank, population_rank == dimensions, boundaries)


def _joint_run(snapshot, basis, start, options):
    basis, included, weights, observed, scales = _block_inputs(snapshot, basis, start)
    ax, rh = snapshot.tensor.fixed_delta_chi_ax, snapshot.tensor.fixed_delta_chi_rh
    _, objective, _ = _metrics(basis.predict(weights, ax, rh), observed, scales)
    history, rejected = [objective], []
    objective_scale = max(objective, _metrics(np.zeros_like(observed), observed, scales)[1], 1e-300)
    _, free, mass, tensor_free = _joint_layout(snapshot)
    population_dimensions = max(0, len(free) - 1) if mass > 0 else 0
    stages = (("tensor",) if not population_dimensions else ("populations",) if not len(tensor_free)
              else ("tensor", "populations", "tensor reconciliation"))
    single_block = len(stages) == 1
    success, message, change, iteration = False, "Joint iteration limit reached", 0.0, 0
    for iteration in range(1, options.max_iterations + 1):
        previous_weights, previous_values, previous_objective = weights.copy(), np.array([ax, rh]), objective
        failure = None
        # The final tensor solve reconciles the population update before convergence testing.
        for stage in stages:
            current = replace(snapshot, tensor=replace(snapshot.tensor,
                              fixed_delta_chi_ax=float(ax), fixed_delta_chi_rh=float(rh)))
            if stage == "populations":
                block = fit_populations(current, basis=basis, populations=weights,
                                        max_iterations=options.population_max_iterations,
                                        tolerance=options.population_tolerance,
                                        stationarity_tolerance=options.stationarity_tolerance)
            else:
                block = fit_tensor(current, basis=basis, populations=weights)
            if not block.success:
                failure = f"{stage} failed: {block.message}"
                break
            try:
                proposed_weights = np.asarray(block.populations, dtype=float)
                proposed_prediction = basis.predict(proposed_weights, block.dchi_ax, block.dchi_rh)
                _, proposed_objective, _ = _metrics(proposed_prediction, observed, scales)
            except ValueError as exc:
                failure = f"{stage} invalid candidate: {exc}"
                break
            if (any(c.population_fixed and proposed_weights[i] != c.initial_population
                    for i, c in enumerate(included))
                    or (mass == 0 and np.any(proposed_weights[free] != 0))
                    or (not snapshot.tensor.fit_delta_chi_ax and block.dchi_ax != snapshot.tensor.fixed_delta_chi_ax)
                    or (not snapshot.tensor.fit_delta_chi_rh and block.dchi_rh != snapshot.tensor.fixed_delta_chi_rh)):
                failure = f"{stage} changed a fixed parameter"
                break
            if proposed_objective > objective:
                failure = f"{stage} objective increase rejected"
                rejected.append(failure)
                break
            weights = proposed_weights
            ax, rh, objective = block.dchi_ax, block.dchi_rh, proposed_objective
            history.append(objective)
        values = np.array([ax, rh])
        change = max(float(np.max(np.abs(weights - previous_weights))),
                     float(np.max(np.abs(values - previous_values) /
                                  np.maximum(1.0, np.maximum(np.abs(values), np.abs(previous_values))))))
        tensor_step, population_step = _joint_stationarity(snapshot, basis, weights, ax, rh, observed, scales)
        stable = ((previous_objective - objective) / objective_scale <= options.objective_tolerance
                  and change <= options.parameter_tolerance
                  and max(tensor_step, population_step) <= options.stationarity_tolerance)
        if failure:
            # An increasing proposal is never installed. A stationary retained state
            # can still finish, but the rejected update is recorded explicitly.
            success = bool(rejected and failure == rejected[-1] and stable)
            message = ("Converged at retained state; " if success else "") + failure
            break
        if single_block and max(tensor_step, population_step) <= options.stationarity_tolerance:
            success, message = True, "Single block converged; no multi-start required"
            break
        if stable:
            success, message = True, "Joint blocks converged"
            break
    prediction = basis.predict(weights, ax, rh)
    residual, objective, rmsd = _metrics(prediction, observed, scales)
    diagnostics = joint_rank_diagnostics(snapshot, basis, weights, ax, rh,
                                         relative_tolerance=options.rank_relative_tolerance)
    if not diagnostics.locally_full_rank:
        message += "; joint Jacobian is rank deficient (local uniqueness not established)"
    return ConformerJointRun(tuple(start), tuple(map(float, weights)), float(ax), float(rh),
                             tuple(map(float, prediction)), tuple(map(float, residual)), objective, rmsd,
                             tuple(history), tuple(rejected), iteration, success, message,
                             tensor_step, population_step, change, diagnostics)


def fit_joint(snapshot, *, basis=None, options=JointFitOptions()):
    """Seeded alternating fits; select only converged, feasible candidates.

    Multi-start reduces dependence on initialization, without guaranteeing a
    global optimum. Similar-quality population alternatives are not intervals.
    """
    supplied = population_start(snapshot, options.initialization)
    basis, _, _, _, _ = _block_inputs(snapshot, basis, supplied)
    included, free, mass, tensor_free = _joint_layout(snapshot)
    dimensions = max(0, len(free) - 1) if mass > 0 else 0
    starts = [supplied]
    if dimensions and len(tensor_free):
        uniform = population_start(snapshot, "uniform")
        if options.starts > 1 and not np.array_equal(uniform, supplied):
            starts.append(uniform)
        rng = np.random.default_rng(options.seed)
        while len(starts) < options.starts:
            weights = np.array(supplied)
            weights[free] = mass * rng.dirichlet(np.ones(len(free)))
            starts.append(tuple(map(float, weights)))
    runs = tuple(_joint_run(snapshot, basis, start, options) for start in starts)
    accepted = [i for i, run in enumerate(runs) if run.success]
    selected = min(accepted, key=lambda i: (runs[i].objective, i)) if accepted else None
    alternatives = ()
    ambiguous = None
    if selected is not None:
        best = runs[selected]
        allowance = options.alternative_objective_absolute + options.alternative_objective_relative * best.objective
        alternatives = tuple(i for i in accepted if i != selected
                             and runs[i].objective <= best.objective + allowance
                             and max(abs(a - b) for a, b in zip(runs[i].populations, best.populations))
                             >= options.alternative_population_distance)
        ambiguous = not best.diagnostics.populations_distinguishable or bool(alternatives)
    return ConformerJointResult(snapshot, options, MC_MODEL_VERSION, MC_JOINT_SOLVER_VERSION,
                                 _digest({"objective": objective_fingerprint(snapshot),
                                          "solver": MC_JOINT_SOLVER_VERSION, "options": asdict(options),
                                          "starts": starts}),
                                 tuple(c.conformer_id for c in included), basis.observation_ids,
                                 runs, selected, alternatives, ambiguous)
