# Conformer Search Optimization

Conformer Search minimizes a bounded, composite objective. The tensor parameters, tensor orientation, metal position, and experimental PCS values remain fixed while structural variables change.

## Search variables

### Dihedral torsion search

Each selected free bond contributes one dihedral variable bounded from −180° to +180°. Applying a variable rotates the graph subtree on the side away from the metal when that side is identifiable; otherwise the smaller subtree rotates. Bond lengths, bond angles, and ring coordinates are not independently optimized.

### Rigid-body anchor rotation

The principal variable is a rotation angle φ from −180° to +180° around the selected metal-centred axis. With `angular_radial_free`, one additional bounded radial variable is added for each anchor. The default relative radial interval is 0.8–1.2 times the initial metal–anchor distance.

## Composite objective

The base data term is the unweighted sum of squared PCS residuals:

\[
S_{\mathrm{PCS}} = \sum_i\left(\delta_{i,\mathrm{calc}}-\delta_{i,\mathrm{Exp}}\right)^2.
\]

The optimizer then adds applicable heuristic terms:

- a non-bonded clash penalty in both search modes;
- a fixed-atom Cartesian penalty in dihedral Position mode;
- fixed-direction and short metal-contact penalties in dihedral Angular mode; and
- anchor-direction and short metal-contact penalties in the applicable rigid-body lock modes.

The final search score is therefore not a pure experimental χ², physical energy, or probability. It has no uncertainty weighting and should not be compared numerically with fit statistics from an unrelated workflow.

## Dihedral search stages

When **Use global search (Differential Evolution)** is enabled, the dihedral workflow runs:

1. SciPy differential evolution over every selected angle in the full −180° to +180° bounds; then
2. bounded SciPy `L-BFGS-B` local refinement of low-cost members from the final population.

**Optimization level** selects **Quick**, **Balanced**, or **Thorough** differential-evolution iteration and population settings. Editing **DE iterations** or **Population size** selects **Custom**. The current search uses a fixed random seed, so otherwise identical inputs and settings are intended to be repeatable.

If global search is disabled, local refinement starts only from the input dihedral vector. That path explores a local basin and currently supplies only one starting point, even if **Output mode: top_n** requests more candidates. Local refinement cannot be assumed to find the global minimum.

## Rigid-body search stages

Rigid-body mode does not use differential evolution. It scans φ on the number of **Grid steps** specified by the user, selects one or more low-cost grid starts, and refines each with bounded `L-BFGS-B`. The global-search checkbox and DE iteration/population controls apply to dihedral mode, not this rigid-body branch.

A finer grid samples the periodic rotation more densely but does not add other rigid-body degrees of freedom. The local stage can still converge several starts to the same minimum.

## Candidate ranking and deduplication

Local results are sorted by the complete objective, including penalties. The interface then reports PCS RMSD, R², Q factor, clashes, constraint drift, and per-atom residuals for interpretation. A candidate with the lowest PCS RMSD need not have the lowest composite objective if it incurs a geometry penalty.

With **Output mode: top_n**, **Top N** limits the retained list and **Min. angle difference [deg]** removes near-duplicate solutions. Dihedral candidates are compared component by component in their stored angle vectors; the current dihedral deduplication does not wrap +180° and −180° as neighbouring values. Rigid-body φ deduplication does use periodic angle differences.

These candidates are alternative local solutions from one search, not a statistically sampled conformer ensemble. Their ordering does not supply populations or confidence intervals.

## Comparison and export

The Results tab retains the candidate list and report. Its contextual Ribbon
provides **View**, **Select**, **Correlation**, **Save XYZ**, **Save Report** and
candidate **Apply / Revert / Discard** actions. This workflow supports:

- a text report for the active candidate;
- a calculated-versus-experimental PCS plot and residual bars;
- a polar structure comparison of the original and up to five candidates;
- selection of a candidate for main-window preview/application;
- a multi-model XYZ containing the checked candidates; and
- a TXT report for the active candidate.

The structure comparison can be saved as PNG, PDF, or SVG. The exported XYZ records candidate rank, RMSD, and R² in each model comment. Review the coordinates and diagnostics directly; an optimized score alone does not validate the chemical structure.

For metric definitions, see [Fit-quality Metrics](../fitting/quality-metrics.md). For the distinction between structural search and tensor fitting, return to the [Conformer Search overview](overview.md).
