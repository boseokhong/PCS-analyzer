# Multi-Conformer PCS fitting and reliability demo

This deterministic synthetic example exercises the production fixed-orientation
Multi-Conformer fitter and its MC5 reliability evaluation. It contains 24 labelled
PCS probes around a metal at the origin, three structures, and independent Gaussian
measurement noise with known σ = 0.03 ppm. The probes are **not a chemically optimized
molecule**. These are generated data, not experimental evidence for conformational
exchange or for a particular lanthanide complex.

## Bundled scenarios

| Folder | Purpose | Ground truth |
|---|---|---|
| `informative/` | Different structure-specific PCS signatures | Populations 0.25 / 0.55 / 0.20; Δχ_ax = 12, Δχ_rh = 3 E-32 m³ |
| `ambiguous/` | C1 and C2 have identical coordinates/PCS signatures | Only their combined population 0.80 is distinguishable; C3 = 0.20 |

C3 is fixed at 0.20. C1/C2 start at 0.40/0.40 and share the remaining mass.
Both tensor scalars are fitted; orientation is fixed at zero Euler angles.
Weighted fitting uses the supplied 0.03 ppm σ. The generator records its seeds,
explicit Ref mapping, frames, inputs and truth in each `inputs.json`.

Each folder contains:

- `demo.pcsp`: ready-to-open project with mapped conformers and structured PCS
  observations, including uncertainties. No fit is pre-applied to the main tensor.
- `conformer_01.xyz`, `conformer_02.xyz`, `conformer_03.xyz`: source coordinates.
  Metal = Ref 1; H probes = Ref 2–25; all files have a confirmed common frame.
- `pcs.csv`: measured PCS and σ for inspection. The project already contains them;
  importing CSV is not needed and may not preserve uncertainties in older workflows.
- `inputs.json`: detached snapshot for the command-line runner, including truth.

## Run in the desktop application

1. Open `informative/demo.pcsp` with **Open Project**.
2. Select **Advanced Fitting → Multi-Conformer**. Inspect the three conformer rows;
   use row double-click/Enter to inspect mapping, metal, frame and fixed population.
3. The contextual Ribbon has **Fitting** first. Confirm **Fit Δχ_ax** and
   **Fit Δχ_rh** are checked. **Fit settings** already specifies `both`, uncertainty
   weighting and four starts with seed 11.
4. Click **Run Fit**. Review recorded conditions, recovered tensor/populations,
   PCS residuals and solver/rank warnings.
5. Click **Reliability** in **Results**. Start with `parametric`, 100 refits,
   5 grouped CV folds, seed 2026, confidence 0.95 and relative similarity threshold
   0.01. Click **Evaluate**. The job runs in the background; progress appears in
   the Analysis header. Larger refit counts can take several minutes.
6. Inspect successful/failed refit counts, marginal percentile intervals, grouped
   out-of-sample RMSD and warnings in the report. **Export result** retains every
   replicate/fold and the evaluation assumptions. Saving the project also retains
   diagnostics; reopening does not recompute them.
7. Repeat with `ambiguous/demo.pcsp`. A good PCS fit does not identify C1 and C2
   separately. Their individual intervals are withheld and their combined population
   is shown with a descriptive bootstrap range. No conformers are merged or removed.

Evaluating reliability does not Apply the fitted tensor or ensemble predictions.
**Apply** remains an explicit scalar-only operation. Input changes mark the old
result Stale; refit before launching a new reliability evaluation.
After editing uncertainties, refit even if an unweighted fit remains Current:
sampling uncertainty depends on σ although the unweighted objective does not.

## Command-line reproduction

From the repository root with its Python environment:

```bash
python examples/multi_conformer_demo/run_demo.py --scenario informative --replicates 100 --output informative_result.json
python examples/multi_conformer_demo/run_demo.py --scenario ambiguous --replicates 100 --output ambiguous_result.json
```

Use `--method cases` to resample observation groups rather than add parametric
noise. Use 20 refits for a smoke test; this is too few for stable percentile tails.
No files in the example are modified by the runner. Output paths are explicit.

To regenerate all bundled inputs (overwrites generated files in the selected folder):

```bash
python examples/multi_conformer_demo/generate_demo.py --output dev_docs/mc-review
python examples/multi_conformer_demo/generate_demo.py --overwrite
```

## What the evaluation means

- **Parametric bootstrap**: perturb the recorded fitted PCS by independent Gaussian
  errors with each observation's known positive σ, then refit every selected free
  parameter. Missing σ or overlapping physical observations reject this method;
  their covariance cannot be guessed.
- **Cases bootstrap**: sample independent, exchangeable observation groups with
  replacement. Overlapping member Ref IDs, pseudo-members or shared source observable
  IDs stay together. Independence between remaining groups is an assumption; a
  shared dataset ID alone does not specify a covariance model.
- **Grouped CV**: partition observation groups into folds. Every fit receives only
  training PCS and the recorded user/uniform initialization, never the full-data
  fitted tensor/populations. Each observation is tested once. Failed folds are
  retained; pooled successful-fold RMSD is labelled partial when necessary.
- **Similarity suggestions**: compare structure-specific PCS columns at the recorded
  fitted tensor using the stated relative RMS difference threshold. Coordinate RMSD
  is recorded separately. Complete-link suggestions prevent long similarity chains
  from being merged. This does not prove structural equivalence or identifiability.

Intervals are marginal percentile estimates, conditional on fixed geometry, mapping,
orientation, metal positions, model and fixed values. They do not account for errors
in those inputs, model selection, or unmodelled correlations. At least 20 and 80% of
requested refits must succeed before intervals are considered; ambiguous/rank-deficient
individual estimates are withheld. Fixed parameters have no estimated interval.
Population boundaries and multimodal solutions require caution. Multi-start spread is
optimizer sensitivity, **not** a confidence interval. Group ranges are descriptive,
not a guarantee that the group itself is uniquely identifiable.

The default 100 refits is a practical first review. Increase it (up to 2000 in the
desktop dialog) to examine endpoint stability before scientific interpretation.
Cross-validation here evaluates a preselected model; it is not nested model-selection
validation. Every fold needs at least three training observations. A particular 95%
interval need not contain the known synthetic truth; compare multiple noise
realizations rather than judging coverage from a single demo result.

## Recorded synthetic results

`validation_summary.json` records finite synthetic studies: `tensor` uses known
populations for a linear calibration check; `both` refits tensor and populations.
Empirical coverage in these studies does not establish coverage for arbitrary
experimental data. The example fit runner reproduces individual supplied cases.

Method references: [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
describes percentile intervals and their limitations; [scikit-learn grouped CV](https://scikit-learn.org/stable/modules/cross_validation.html)
and [data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)
describe training-only evaluation and group separation. No scikit-learn dependency
was added; PCS Analyzer uses its existing numerical owners.
