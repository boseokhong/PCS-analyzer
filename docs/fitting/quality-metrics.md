# Quality Metrics

The standard Mode A, B, and C reports calculate their metrics from the same
per-point values. Let

\[
e_i=\delta_{\mathrm{pred},i}-\delta_{\mathrm{Exp},i}
\]

be the fitting residual and let \(N\) be the number of matched, selected
experimental observations. A grouped target counts once, not once per member
atom. RMSD, R², Q and Raw SSE/N below use raw ppm residuals even in a weighted fit.

## Standard nonlinear-fitting metrics

### RMSD

\[
\mathrm{RMSD}
=
\sqrt{\frac{1}{N}\sum_i e_i^2}.
\]

RMSD is reported in ppm. It describes the typical residual magnitude but does
not account for the number of fitted parameters.

### R²

\[
R^2
=
1-
\frac{\sum_i e_i^2}
     {\sum_i(\delta_{\mathrm{Exp},i}-\overline{\delta}_{\mathrm{Exp}})^2}.
\]

R² is omitted when the experimental values have essentially zero variance.
It can be negative when the predictions are worse than using the experimental
mean as a constant prediction.

### Q-factor

\[
Q
=
\sqrt{
\frac{\sum_i e_i^2}
     {\sum_i\delta_{\mathrm{Exp},i}^2}
}.
\]

The Q-factor is omitted when the squared experimental values sum to
essentially zero.

### Raw SSE/N

The report's **Raw SSE/N** is

\[
\frac{1}{N}\sum_i e_i^2=\mathrm{RMSD}^2.
\]

It is an unweighted mean squared residual in ppm², not a chi-square statistic.

### Experimental-uncertainty weighting

With **Weighting: Experimental uncertainty**, the optimizer minimizes

\[
\chi^2=\sum_i\left(\frac{e_i}{\sigma_i}\right)^2,
\]

where each \(\sigma_i\) is the supplied uncertainty of one experimental observation
in ppm. All selected uncertainties must be finite and strictly positive.
The report adds **χ² = Σ(resid/σ)²** and **χ²/N (weighted)**, both dimensionless.
These are separate from the raw ppm metrics above.

The denominator N is the observation count, not N minus the number of fitted
parameters. **χ²/N (weighted) is chi-square per datum, not reduced chi-square.**
Its statistical interpretation depends on whether the uncertainties and error
model are appropriate; grouping does not supply extra independent data.

## Residual displays

The standard fitting report lists the five largest absolute residuals and a
complete per-point table. Its correlation plot compares δ_Exp with predicted
PCS, colors points by absolute residual, and labels each point with its Ref
ID. The residual bars use fixed display thresholds:

- green: |residual| ≤ 0.05 ppm;
- orange: 0.05 < |residual| ≤ 0.15 ppm;
- red: |residual| > 0.15 ppm.

These fixed plot thresholds are distinct from the editable cutoffs shared by
the main and Rhombicity tables. Those tables default to 0.10 and 0.30 ppm;
changing their cutoffs does not change the Fitting plot. See
[Residual Analysis](../diagnostics/residual-analysis.md#main-and-rhombicity-table-colouring)
for table controls and persistence.

## Metrics in other workflows

| Workflow | Metrics shown by the current UI |
| --- | --- |
| Main **G_ax vs PCS** linear regression | R², residual SD, RMSE, slope/intercept standard errors and 95% confidence intervals, and a converted Δχ_ax 95% confidence interval |
| **Rhombicity Check** | axial-only residual RMSD; Pearson r and R² for residual versus Grh |
| Advanced **Multi-Conformer** | PCS RMSD, recorded objective, fitted/fixed populations and tensor components, and attempted starts. Optional reliability evaluation adds conditional bootstrap intervals, held-out RMSD and ambiguity warnings; detailed sensitivity/rank records are retained in result JSON. |
| Advanced **Joint PCS+RDC** | Separate PCS RMSD (ppm) and RDC RMSD (Hz), rank, condition number and weighted residual sum of squares. There is no combined raw-unit RMSD. |
| Advanced **PRE** | Γ₂ RMSD (s⁻¹), weighted residual sum of squares, effective τ_c and whether the solution reaches a selected bound. |
| Advanced **Multi-Lanthanide** | Independent tensor RMSD per dataset; Reilley G_eff/F_eff with conditional standard errors, rank, condition, residual degrees of freedom and original-space RMSD (ppm). Other modes provide their own diagnostic regression metrics. |

Standard nonlinear Mode A/B/C fits do not report parameter covariance or
nonlinear confidence intervals. Joint PCS+RDC and PRE do not estimate confidence
intervals either. The 95% intervals from the main G_ax regression belong to that
linear model and must not be transferred to another workflow.

Multi-Conformer bootstrap intervals and held-out errors are conditional on its
recorded geometry, assignments, model and resampling assumptions. Ambiguous
individual populations can have their intervals withheld; a reported group
interval is not proof of unique individual populations. See the
[reliability tutorial](multi-conformer-tutorial.md#4-evaluate-reliability).

Reilley coefficient errors use supplied measurement uncertainties as known when
weighting is enabled, or estimated residual variance otherwise. Exactly two
points have no residual degrees of freedom and no reported coefficient errors
or R². Coefficient-table and temperature uncertainty are not propagated. See
[Reilley coefficients and assumptions](multi-lanthanide-tutorial.md#coefficients-and-assumptions).

Compare weighted objectives only when inputs, error scales and residual definitions
are compatible. Joint PCS+RDC includes its configured RDC multiplier/error floor;
PRE can use assumed errors instead of supplied uncertainties. These sums are not
automatically reduced chi-square. Workflow details are in the
[Joint PCS+RDC](joint-pcs-rdc-tutorial.md) and [PRE](pre-tutorial.md) tutorials.

## Comparing residual signs

Standard fitting and advanced fitting store residuals as predicted minus
experimental. The main table's residual colouring also uses calculated minus
δ_Exp. In contrast, **Rhombicity Check** and the Rhombicity Table display
experimental minus predicted. Always check the workflow before interpreting
the sign; absolute-value rankings and RMSD are unchanged.

See [Residual Analysis](../diagnostics/residual-analysis.md) for selection,
colouring, and export behavior.
