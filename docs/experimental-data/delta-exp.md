# Experimental PCS Data

PCS Analyzer uses **δ_Exp** as the experimental PCS value supplied for a
specific structure Ref ID. These values are the targets used by the main
geometrical-factor regression, the Fitting tab, the rhombicity diagnostic,
and PCS-based advanced fitting workflows.

## Meaning of δ_Exp

In the current application, δ_Exp means **experimental PCS**. It is stored
separately from the observed and diamagnetic chemical-shift layers.

The related quantities are:

| Quantity | Meaning in PCS Analyzer |
| --- | --- |
| δ_Exp | User-supplied experimental PCS used as a regression or fitting target |
| δ_PCS or calculated PCS | PCS predicted from the current structure, metal centre, tensor, orientation, and active prediction model |
| δ_obs | Observed shift for the paramagnetic sample |
| δ_dia | Diamagnetic reference shift |
| δ_para | Paramagnetic shift contribution computed as δ_obs − δ_dia |
| δ_nonPCS | Diagnostic difference δ_para − calculated PCS |

Although δ_para may be used as an experimental PCS estimate when other
paramagnetic shift contributions are negligible, the two terms are not
automatically interchangeable. PCS Analyzer does not automatically copy
δ_para into δ_Exp. See [Observed and Diamagnetic Shifts](delta-obs-dia.md).

## Entering values in the main table

The main table contains a **δ_Exp** column. Double-click a cell in that
column, enter a numeric value in ppm, and press **Enter**. Submitting an empty
value clears the assignment. Other text is rejected as an invalid value.

For multiple assignments, use **Ribbon → Data → Import PCS** or **Paste**.
**Export Template** creates a file
containing the currently visible Ref IDs and atom labels. The complete import
workflow is described in [Import and Export](import-export.md).

Alternatively, [NMR Workbench → PCS Preparation](../nmr/non-pcs-analysis.md)
derives experimental PCS from paired observables and supplies reviewed
single-atom or structured targets through an explicit handoff.

## Ref IDs are the assignment key

Each δ_Exp value is associated with an integer Ref ID, not merely with an atom
or element label. A fit uses a value only when that Ref ID is also present in
the current effective atom set and is selected in the relevant workflow.

Reloading or replacing a structure can change the meaning or availability of
Ref IDs. After doing so, verify the assignments and re-import the data when
necessary. A value attached to a numeric Ref ID that is absent from the
current structure remains stored but cannot contribute to a fit.

## Experimental and calculated PCS

Calculated PCS values are refreshed from the current inputs and appear as
**δ_PCS** in the main table. δ_Exp is an independent experimental layer; an
update of the structure, tensor, centre, rotation, or averaging settings
changes δ_PCS but does not change δ_Exp.

The main-table δ_PCS follows **Prediction model**: **Axial** excludes the
rhombic term; **Full tensor (Axial + Rhombic)** includes it only when
**Enable Δχ_rh** is on. Linked NMR predictions follow this policy too.
Polar contours and **G_ax vs PCS** remain axial diagnostics. See
[Axial and Rhombic PCS](../pcs/axial-rhombic.md#active-prediction-model).

## Structured observations and grouped targets

Experimental PCS prepared for a grouped NMR site, matching ligand sites, or
a custom group is retained as **one observation** with one experimental value
and, when supplied, one uncertainty. It contributes one residual to standard
fitting, not one independent measurement for every member atom. Such grouped
observations need not appear as individual δ_Exp cells in the main table.

The prediction is calculated at each member's geometry, then reduced according
to the target's member/site hierarchy. The current grouped targets use
arithmetic means: first within each site, then across sites. For example,
sites containing two and three atoms give

\[
\delta_{\mathrm{pred}}
=\frac{1}{2}\left[
\frac{\delta_1+\delta_2}{2}
+\frac{\delta_3+\delta_4+\delta_5}{3}
\right].
\]

This gives equal weight to the two sites, not to all five atoms. A singleton
without reduction retains its own PCS. Predictions are not evaluated at a
mean coordinate: PCS depends nonlinearly on distance and direction. Existing
symmetry/torsional geometrical-factor averaging remains applicable to the
member predictions; see [Averaging](../structures/symmetry-averaging.md).

When structured observations are present, **PCS observations** lists their
assignments/identifiers; otherwise fitting uses the legacy proton Ref list.
Review the selected targets and their members before fitting. Group membership
does not justify treating members as additional independent observations.

## Residual sign

The main table uses the difference

\[
\delta_{\mathrm{PCS}}-\delta_{\mathrm{Exp}}
\]

for residual colouring. The nonlinear fitting reports use the same
predicted-minus-experimental sign. The axial-only rhombicity diagnostic uses
the opposite sign, experimental minus prediction, as explained in
[Residual Analysis](../diagnostics/residual-analysis.md).

For the PCS equation and susceptibility conventions, see
[PCS Theory](../pcs/theory.md) and
[Tensor Conventions](../pcs/tensor-conventions.md).
