# Method References and Reporting

Use these references for the scientific background of an analysis. The linked
workflow pages define the model and options implemented in PCS Analyzer; a
reference to a general method is not evidence that every variant is supported.

## PCS and susceptibility conventions

The point-dipole PCS expression and its extension to distributed spin density
are discussed in [A partial differential equation for pseudocontact shift](https://doi.org/10.1039/C4CP03106G).
For broader context, see [Magnetic susceptibility and paramagnetism-based NMR](https://doi.org/10.1016/j.pnmrs.2019.06.003).

For calculations in this application, use the definitions in
[PCS Theory](../pcs/theory.md) and [Tensor Conventions](../pcs/tensor-conventions.md):
coordinates in Å, anisotropies in 10⁻³² m³ per molecule and output shifts in ppm.
Check axial/rhombic definitions and frame conventions before comparing published
tensors. The [PCS Workbench](../pcs-workbench/overview.md) describes the separate
point/distributed-field comparison and its grid assumptions.

## Lanthanide series and Reilley separation

- Bleaney's [Nuclear magnetic resonance shifts in solution due to lanthanide ions](https://doi.org/10.1016/0022-2364(72)90027-3)
  provides the historical lanthanide-shift framework.
- [Separation of contact and dipolar lanthanide induced nuclear magnetic resonance shifts: evaluation and application of some structure independent methods](https://doi.org/10.1021/ac50005a010)
  is an original-method reference for contact/dipolar separation.
- [Pinkerton et al. (1985)](https://doi.org/10.1016/0022-2364(85)90104-0),
  Table 1, is the source identified for the RHF spin-factor coefficients.
- The [DTMA fluoride-binding study](https://pubs.rsc.org/en/content/articlehtml/2015/dt/c5dt02398j)
  is an application example, not the original source of the separation method.

The [Multi-Lanthanide tutorial](../fitting/multi-lanthanide-tutorial.md#coefficients-and-assumptions)
states the application's relative C_D convention, temperature expression,
fixed Sm/Eu approximation and common-temperature requirements. Report those
choices when interpreting separated components. A linear trend alone does not
establish unchanged geometry or contact coupling across the series.

## Joint PCS and RDC

The general relationship between susceptibility anisotropy and paramagnetic
NMR observables, including RDC, is covered by the
[susceptibility review](https://doi.org/10.1016/j.pnmrs.2019.06.003).
The [Joint PCS+RDC tutorial](../fitting/joint-pcs-rdc-tutorial.md) specifies the
narrower implemented model: magnetic self-alignment, a fixed structure/frame,
two fitted tensor scalars and the stated residual weighting. RDC from a separate
alignment medium is outside that model.

## PRE

The [PRE tutorial](../fitting/pre-tutorial.md#model-and-limits) gives the actual
proton Γ₂ equation and omitted terms. Compare it with the
[Xplor-NIH PRE model description](https://nmr.cit.nih.gov/xplor-nih/xplorMan/node437.html)
and the [relaxation-model discussion, equations 12–14](https://pmc.ncbi.nlm.nih.gov/articles/PMC10500712/).
Report the magnetic field, moment, correlation-time bounds and errors, and
explain why the simplified model is appropriate for the sample. An ion preset
or a fitted effective time alone does not establish that appropriateness.

## Reliability evaluation

[SciPy's bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
explains percentile resampling intervals. The
[grouped cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
and [data-leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
provide statistical background for keeping related observations together and
withheld data out of training. These are method explanations, not additional
PCS Analyzer dependencies or a claim that its evaluator calls those APIs.

The [Multi-Conformer tutorial](../fitting/multi-conformer-tutorial.md) defines
the implemented parametric/cases bootstrap, grouping, failed-refit handling and
held-out score. Its intervals are conditional on the supplied geometry, mapping,
orientation and model; they do not propagate uncertainty in those inputs.

## Report a reproducible analysis

Alongside the software version and archived-release citation, retain:

- original structure and experimental data, acquisition/reference conditions,
  assignments and any excluded observations with reasons;
- tensor units/frame, averaging assumptions and fitted versus fixed parameters;
- initialization, bounds, objective weights and diagnostic results;
- a saved project and numerical result exports; NMR work also needs its session;
- for reliability evaluation, resampling method, group definition, seed, requested
  and successful refit counts, and failed/partial evaluation status.

Use the [file-format reproducibility checklist](file-formats.md#reproducing-an-analysis)
for the supported exports. A screenshot alone does not retain a complete fit.
