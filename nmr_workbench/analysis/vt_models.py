"""Explicit observed-shift temperature models; no physical interpretation or UI."""
from dataclasses import asdict, dataclass, replace
import math
import numpy as np
from scipy.optimize import least_squares

from .vt_analysis import VTDiagnostic, VTCoordinates

MODELS = {"linear_T": "Empirical linear (T)", "curie": "Curie", "curie_weiss": "Curie-Weiss"}
ASSUMPTIONS = (
    "Input is observed chemical shift in ppm, not PCS/contact/shift difference.",
    "One unchanged resonance/species over the fitted temperature range; no exchange/population model.",
    "Offset is assumed temperature-independent; coefficient is phenomenological, not a susceptibility tensor.",
    "Independent shift errors; effective Kelvin coordinates have no modeled temperature uncertainty.",
)


@dataclass(frozen=True)
class VTModelConfig:
    model_id: str = "curie"
    quantity: str = "observed_chemical_shift"
    weighting: str = "unweighted"
    theta_bounds_k: tuple[float, float] | None = None
    max_evaluations: int = 3000

    def __post_init__(self):
        if self.model_id not in MODELS or self.quantity != "observed_chemical_shift":
            raise ValueError("Only declared observed-shift models are supported; PCS/contact/Δshift require separate inputs")
        if self.weighting not in {"unweighted", "measured_sigma"}:
            raise ValueError("Unknown VT weighting")
        if type(self.max_evaluations) is not int or self.max_evaluations < 1:
            raise ValueError("Invalid evaluation limit")
        if self.theta_bounds_k is not None:
            bounds = tuple(self.theta_bounds_k)
            if len(bounds) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bounds) or bounds[0] >= bounds[1]:
                raise ValueError("Theta bounds must be finite increasing Kelvin values")
            if self.model_id != "curie_weiss":
                raise ValueError("Theta bounds apply only to Curie-Weiss")
            object.__setattr__(self, "theta_bounds_k", bounds)

    def to_dict(self):
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, row):
        if not isinstance(row, dict) or type(row.get("schema_version")) is not int or row["schema_version"] != 1:
            raise ValueError("Unsupported VT model configuration schema")
        return cls(**{k: v for k, v in row.items() if k != "schema_version"})


def evaluate_vt_model(model_id, temperatures_k, parameters):
    t = np.asarray(temperatures_k, dtype=float)
    if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("Model temperatures must be finite positive Kelvin")
    expected = ("offset", "slope") if model_id == "linear_T" else ("offset", "C", "theta") if model_id == "curie_weiss" else ("offset", "C") if model_id == "curie" else ()
    if not expected or set(parameters) != set(expected) or any(not math.isfinite(float(v)) for v in parameters.values()):
        raise ValueError("Invalid model/parameters")
    if model_id == "linear_T":
        values = parameters["offset"] + parameters["slope"] * t
    else:
        denominator = t - parameters.get("theta", 0.)
        if np.any(denominator <= 0):
            raise ValueError("Curie-Weiss requires T > theta (high-temperature branch)")
        values = parameters["offset"] + parameters["C"] / denominator
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite model prediction")
    return tuple(map(float, values))


@dataclass(frozen=True)
class VTModelParameter:
    name: str
    value: float
    unit: str
    standard_error: float | None = None


@dataclass(frozen=True)
class VTModelResult:
    config: VTModelConfig
    coordinates: VTCoordinates
    status: str
    converged: bool
    parameters: tuple[VTModelParameter, ...]
    covariance: tuple[tuple[float, ...], ...] | None
    parameter_correlation: tuple[tuple[float, ...], ...] | None
    jacobian_condition: float | None
    fitted_values: tuple[float, ...]  # selected display-coordinate units
    residuals: tuple[float, ...]
    raw_fitted_values: tuple[float, ...]  # observed-shift ppm, actual objective
    raw_residuals: tuple[float, ...]
    rss: float | None
    rmse: float | None
    r_squared: float | None
    aic: float | None
    aicc: float | None
    chi_squared: float | None
    point_count: int
    included_identities: tuple
    excluded_identities: tuple
    temperature_range_k: tuple[float, float] | None
    diagnostics: tuple[VTDiagnostic, ...]
    assumptions: tuple[str, ...] = ASSUMPTIONS
    metric_quantity: str = "observed_chemical_shift"
    metric_unit: str = "ppm"

    @property
    def domain(self):
        return self.coordinates.domain

    @property
    def slope(self):
        return next((p.value for p in self.parameters if p.name == "slope"), None)

    @property
    def intercept(self):
        return next((p.value for p in self.parameters if p.name == "offset"), None)

    def to_dict(self):
        return {"schema_version": 1, **asdict(self), "config": self.config.to_dict()}


def _covariance(jacobian, variance, scales, diagnostic):
    norms = np.linalg.norm(jacobian, axis=0)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        diagnostic("singular_covariance", "A parameter has no identifiable sensitivity; uncertainties withheld")
        return None, None, None
    normalized = jacobian / norms
    u, s, vt = np.linalg.svd(normalized, full_matrices=False)
    condition = float(s[0] / s[-1]) if s[-1] > 0 else None
    if s[-1] <= s[0] * 1e-12 or condition is None:
        diagnostic("singular_covariance", "Rank-deficient parameter sensitivities; uncertainties withheld")
        return None, None, condition
    # Correlation depends on sensitivity, even if residual variance is unknown.
    inverse = (vt.T / s**2) @ vt
    inverse = inverse / norms[:, None] / norms[None, :]
    physical = inverse * np.asarray(scales)[:, None] * np.asarray(scales)[None, :]
    sd = np.sqrt(np.diag(physical))
    correlation = physical / sd[:, None] / sd[None, :]
    corr_tuple = tuple(tuple(map(float, row)) for row in correlation)
    if np.max(np.abs(correlation - np.eye(len(scales)))) > .995:
        diagnostic("strong_parameter_correlation", "Parameter correlation exceeds 0.995; individual parameters may not be separable")
    if condition > 1e6:
        diagnostic("ill_conditioned_covariance", "Column-normalized Jacobian condition exceeds 1e6; uncertainties withheld")
        return None, corr_tuple, condition
    if variance is None:
        return None, corr_tuple, condition
    covariance = physical * variance
    if not np.all(np.isfinite(covariance)):
        diagnostic("ill_conditioned_covariance", "Nonfinite covariance; uncertainties withheld")
        return None, corr_tuple, condition
    return tuple(tuple(map(float, row)) for row in covariance), corr_tuple, condition


def fit_vt_model(coordinates, config=None):
    """Fit raw observed δ(T); coordinate selection changes presentation only.

    Measured-sigma weighting requires complete positive absolute shift sigmas.
    Approximate covariance uses analytic sensitivities; it is never a physical proof.
    """
    config = config or VTModelConfig()
    if coordinates.domain.name not in {"delta_vs_T", "delta_vs_inverse_T", "deltaT_vs_inverse_T"}:
        raise ValueError("Physical models accept observed-shift coordinates only")
    points = tuple(p for p in coordinates.points if p.included)
    ids = tuple(p.observation.identity for p in points)
    excluded = tuple(p.observation.identity for p in coordinates.points if not p.included)
    diagnostics = list(coordinates.diagnostics)
    def diagnostic(code, message):
        diagnostics.append(VTDiagnostic(code, message, ids))
    n, pcount = len(points), 3 if config.model_id == "curie_weiss" else 2
    empty = VTModelResult(config, coordinates, "invalid", False, (), None, None, None,
        (), (), (), (), None, None, None, None, None, None, n, ids, excluded, None, ())
    def invalid():
        return replace(empty, diagnostics=tuple(diagnostics))
    if n < pcount:
        diagnostic("insufficient_points", f"{config.model_id} requires at least {pcount} points")
        diagnostic("underdetermined_fit", "Fewer observations than fitted parameters")
        return invalid()
    t = np.asarray([point.observation.temperature_k for point in points], dtype=float)
    y = np.asarray([point.observation.chemical_shift_ppm for point in points], dtype=float)
    if not np.all(np.isfinite(t)) or np.any(t <= 0) or not np.all(np.isfinite(y)):
        diagnostic("nonfinite_fit_data", "Finite shifts and positive Kelvin are required")
        return invalid()
    if len(np.unique(t)) < pcount:
        diagnostic("underdetermined_fit", "Too few distinct temperatures to identify all parameters; repeats retained")
        return invalid()
    sigma = np.ones(n)
    if config.weighting == "measured_sigma":
        raw_sigma = [point.observation.shift_uncertainty_ppm for point in points]
        if any(v is None or not math.isfinite(v) or v <= 0 for v in raw_sigma):
            diagnostic("incomplete_uncertainty", "Weighted fitting requires positive measured shift uncertainty for every included point")
            return invalid()
        sigma = np.asarray(raw_sigma)
    elif any(point.observation.shift_uncertainty_ppm is not None for point in points):
        diagnostic("uncertainty_not_weighted", "Measured uncertainties retained; explicit unweighted fit")
    tscale = float(np.mean(t))
    yscale = max(float(np.ptp(y)), 1.)
    bounds = config.theta_bounds_k
    if config.model_id == "curie_weiss":
        if bounds is None:
            bounds = (-10. * float(t.max()), .95 * float(t.min()))
            config = replace(config, theta_bounds_k=bounds)
            diagnostic("default_theta_bounds", "Numerical search bounds: -10*Tmax to 0.95*Tmin; review applicability")
        if bounds[1] >= t.min():
            diagnostic("invalid_parameter_region", "Theta upper bound must be below every fitted temperature")
            return invalid()
    try:
        if config.model_id in {"linear_T", "curie"}:
            x = t if config.model_id == "linear_T" else 1 / t
            center, scale = float(x.mean()), float(np.max(np.abs(x - x.mean())))
            design = np.column_stack((np.ones(n), (x - center) / scale))
            q = np.linalg.lstsq(design / sigma[:, None], y / sigma, rcond=None)[0]
            prediction = design @ q
            transform = np.array([[1., -center / scale], [0., 1 / scale]])
            parameters = transform @ q
            # Covariance uses original physical parameters through a stable basis.
            covariance_basis, correlation, condition = _covariance(design / sigma[:, None], 1., (1., 1.), diagnostic)
            jac = design / sigma[:, None]
            names = ("offset", "slope") if config.model_id == "linear_T" else ("offset", "C")
            units = ("ppm", "ppm K^-1") if config.model_id == "linear_T" else ("ppm", "ppm K")
            converged = True
        else:
            # Scaled [offset/yscale, C/(yscale*Tscale), theta/Tscale].
            x = 1 / t
            linear = np.linalg.lstsq(np.column_stack((np.ones(n), x)), y, rcond=None)[0]
            initial_theta = 0. if bounds[0] < 0 < bounds[1] else sum(bounds) / 2
            initial = [linear[0] / yscale, linear[1] / (yscale * tscale), initial_theta / tscale]
            def values(q):
                return yscale * (q[0] + q[1] / (t / tscale - q[2]))
            def jacobian(q):
                d = t / tscale - q[2]
                return yscale * np.column_stack((np.ones(n), 1 / d, q[1] / d**2)) / sigma[:, None]
            solved = least_squares(lambda q: (values(q) - y) / sigma, initial, jac=jacobian,
                bounds=([-np.inf, -np.inf, bounds[0] / tscale], [np.inf, np.inf, bounds[1] / tscale]),
                x_scale="jac", max_nfev=config.max_evaluations, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            converged = bool(solved.success)
            if not converged:
                diagnostic("nonconvergence", str(solved.message))
                return invalid()
            prediction, jac = values(solved.x), jacobian(solved.x)
            scales = (yscale, yscale * tscale, tscale)
            parameters = solved.x * scales
            names, units = ("offset", "C", "theta"), ("ppm", "ppm K", "K")
            if abs(parameters[2] - bounds[0]) < 1e-5 * (bounds[1] - bounds[0]) or abs(parameters[2] - bounds[1]) < 1e-5 * (bounds[1] - bounds[0]):
                diagnostic("parameter_at_bound", "Theta reached a search bound; covariance withheld, review bounds/model")
        residual = y - prediction
        rss = float(np.dot(residual, residual))
        total = float(np.dot(y - y.mean(), y - y.mean()))
        chi2 = float(np.dot(residual / sigma, residual / sigma))
        if not np.all(np.isfinite(parameters)) or not all(math.isfinite(v) for v in (rss, total, chi2)):
            diagnostic("numerical_failure", "Numerical overflow in fit parameters/metrics")
            return invalid()
        variance = 1. if config.weighting == "measured_sigma" else (chi2 / (n - pcount) if n > pcount else None)
        if n == pcount:
            diagnostic("no_residual_degrees_of_freedom", "No residual degrees of freedom to estimate noise or test model adequacy")
            if config.weighting == "unweighted":
                variance = None
        near_exact = rss <= np.finfo(float).eps**2 * max(float(np.dot(y, y)), 1.) * 100
        if config.weighting == "unweighted" and near_exact:
            diagnostic("residual_variance_unresolved", "Near-exact fit cannot estimate unknown measurement variance; no zero-error precision reported")
            variance = None
        if config.model_id == "curie_weiss":
            covariance, correlation, condition = _covariance(jac, variance, scales, diagnostic)
            if total == 0:
                diagnostic("singular_covariance", "Constant shift leaves Curie-Weiss theta unidentifiable; covariance withheld")
                covariance = None
            if any(d.code == "parameter_at_bound" for d in diagnostics):
                covariance = None
            if covariance is not None:
                theta_error = math.sqrt(max(covariance[2][2], 0.))
                if abs(parameters[2]) <= 1.96 * theta_error:
                    diagnostic("curie_not_distinguished", "Approximate theta interval includes zero; Curie cannot be excluded by this fit")
        else:
            # Stable numerical basis, then convert to the reported physical basis.
            base, _, condition = _covariance(jac, variance, (1., 1.), diagnostic)
            if base is not None:
                matrix = transform @ np.asarray(base) @ transform.T
                covariance = tuple(tuple(map(float, r)) for r in matrix)
            else:
                covariance = None
            sensitivity = transform @ np.asarray(covariance_basis) @ transform.T if covariance_basis is not None else None
            if sensitivity is not None:
                std = np.sqrt(np.diag(sensitivity))
                corr = sensitivity / std[:, None] / std[None, :]
                correlation = tuple(tuple(map(float, r)) for r in corr)
                if abs(corr[0, 1]) > .995:
                    diagnostic("strong_parameter_correlation", "Offset/coefficient correlation exceeds 0.995")
        if total == 0:
            diagnostic("constant_response", "R² is undefined for constant observed shift")
        k = pcount + (1 if config.weighting == "unweighted" else 0)
        if config.weighting == "measured_sigma":
            aic = chi2 + float(np.sum(math.log(2 * math.pi) + 2 * np.log(sigma))) + 2 * k
        elif near_exact:
            aic = None
            diagnostic("aic_unavailable", "Unknown-variance Gaussian likelihood is undefined at a near-zero residual variance")
        else:
            aic = n * (math.log(rss / n) + 1 + math.log(2 * math.pi)) + 2 * k
        aicc = aic + 2 * k * (k + 1) / (n - k - 1) if aic is not None and n > k + 1 else None
        if aicc is None:
            diagnostic("aicc_unavailable", "AICc requires a finite likelihood and n > k+1 (variance counts as a parameter when unknown)")
        factor = t if coordinates.domain.name == "deltaT_vs_inverse_T" else np.ones(n)
        errors = [math.sqrt(max(covariance[i][i], 0.)) if covariance is not None else None for i in range(pcount)]
        return VTModelResult(config, coordinates, "ok", converged,
            tuple(VTModelParameter(name, float(value), unit, error) for name, value, unit, error in zip(names, parameters, units, errors)),
            covariance, correlation, condition, tuple(map(float, prediction * factor)), tuple(map(float, residual * factor)),
            tuple(map(float, prediction)), tuple(map(float, residual)), rss, math.sqrt(rss / n),
            1 - rss / total if total > 0 else None, aic, aicc, chi2 if config.weighting == "measured_sigma" else None,
            n, ids, excluded, (float(t.min()), float(t.max())), tuple(dict.fromkeys(diagnostics)))
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        diagnostic("numerical_failure", str(exc))
        return invalid()


def compare_vt_models(coordinates, configs):
    """Same raw quantity, observations/mask and weighting; never rank physical truth."""
    configs = tuple(configs)
    if len({c.model_id for c in configs}) != len(configs) or len({c.weighting for c in configs}) > 1:
        raise ValueError("Comparison requires unique models with the same weighting")
    results = [fit_vt_model(coordinates, config) for config in configs]
    # Flag similar AICc, not a winner; do not use transformed-coordinate likelihoods.
    usable = [r for r in results if r.status == "ok" and r.aicc is not None]
    for index, result in enumerate(results):
        alternatives = [r for r in usable if r.config.model_id != result.config.model_id and result.aicc is not None and abs(r.aicc - result.aicc) < 2]
        if alternatives:
            results[index] = replace(result, diagnostics=result.diagnostics + (VTDiagnostic("models_not_distinguished",
                "AICc differs by less than 2 from " + ", ".join(r.config.model_id for r in alternatives) + "; fit quality does not establish physical correctness", result.included_identities),))
    return tuple(results)


def predict_vt_model(result, temperatures_k):
    """Explicit extrapolation request; unsafe T<=theta is rejected, never hidden."""
    if result.status != "ok":
        raise ValueError("No converged model for prediction")
    temperatures = tuple(temperatures_k)
    values = evaluate_vt_model(result.config.model_id, temperatures, {p.name: p.value for p in result.parameters})
    diagnostics = ()
    low, high = result.temperature_range_k
    if any(t < low or t > high for t in temperatures):
        diagnostics = (VTDiagnostic("extrapolation", "Prediction extends outside measured fit temperatures; physical validity is not established"),)
    return values, diagnostics


def model_curve(result, count=160):
    if result.status != "ok":
        return (), ()
    t = np.linspace(*result.temperature_range_k, count)
    values, _ = predict_vt_model(result, t)
    x = 1 / t if result.domain.x_quantity == "inverse_temperature" else t
    y = np.asarray(values) * t if result.domain.name == "deltaT_vs_inverse_T" else np.asarray(values)
    order = np.argsort(x)
    return tuple(map(float, x[order])), tuple(map(float, y[order]))


def analysis_evidence_diagnostics(series, evidence):
    """Historical fits remain unchanged; diagnose missing/changed live sources."""
    import json
    from .vt_analysis import extract_vt_track
    saved = json.loads(evidence.evidence_json)
    for row in saved["tracks"]:
        track_id = row["snapshot"]["track_id"]
        track = next((t for t in series.tracks if t.id == track_id), None)
        if track is None:
            yield "analysis_track_missing", f"evidence={evidence.id}, track={track_id}"
            continue
        current = extract_vt_track(series, track).to_dict()
        association = series.observable_association(track_id)
        if (json.dumps(current, sort_keys=True) != json.dumps(row["snapshot"], sort_keys=True)
            or json.dumps(association.to_dict() if association else None, sort_keys=True) != json.dumps(row.get("observable_association"), sort_keys=True)
            or json.dumps(observable_sources(series, track_id), sort_keys=True) != json.dumps(row.get("source_observables", []), sort_keys=True)):
            yield "analysis_source_changed", f"evidence={evidence.id}, track={track_id}; historical fit retained"


def observable_sources(series, track_id):
    """Separate dataset-local interpretation evidence, never used as fit values."""
    from .vt_analysis import extract_vt_track
    snapshot = extract_vt_track(series, series.get_track(track_id))
    identities = {(o.identity.dataset_id, o.identity.peak_id) for o in snapshot.observations}
    datasets = {m.dataset_id: m.dataset for m in series.members if m.dataset is not None}
    return [{key: ({"invalid_numeric": str(value)} if isinstance(value, float) and not math.isfinite(value) else value)
             for key, value in asdict(o).items()}
            for dataset in datasets.values() for o in dataset.observables
            if (dataset.id, o.source_peak_id) in identities]
