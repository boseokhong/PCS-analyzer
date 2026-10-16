# Observed and Diamagnetic Shifts

PCS Analyzer keeps observed shifts, diamagnetic-reference shifts, derived
paramagnetic shifts, and experimental PCS assignments as distinct data
layers.

## Shift definitions

For a given Ref ID, the application uses

\[
\delta_{\mathrm{para}}
=
\delta_{\mathrm{obs}}-\delta_{\mathrm{dia}}.
\]

Here:

- **δ_obs** is the observed chemical shift in the paramagnetic sample;
- **δ_dia** is the corresponding diamagnetic reference shift; and
- **δ_para** is the shift difference computed by PCS Analyzer.

δ_para is calculated only when both δ_obs and δ_dia are present for the same
Ref ID. Clearing or changing either input recomputes the derived layer.

## Paramagnetic shift is not necessarily PCS

The paramagnetic shift may contain PCS and contributions not represented by
the current PCS model. PCS Analyzer therefore does not equate δ_para with
δ_Exp automatically.

The NMR analysis workflow calculates the diagnostic quantity

\[
\delta_{\mathrm{nonPCS}}
=
\delta_{\mathrm{para}}-\delta_{\mathrm{PCS,calc}},
\]

where δ_PCS,calc follows the active **Prediction model**. **Axial** excludes
the rhombic term; **Full tensor (Axial + Rhombic)** includes it only when
**Enable Δχ_rh** is on. This value is a residual-like
comparison, not a guaranteed physical separation of contact shift,
referencing error, and every other non-PCS contribution.

!!! caution

    A small δ_nonPCS is consistent with the compared PCS model accounting for
    most of δ_para. A large value can have several causes, including a genuine
    non-PCS contribution, an incorrect assignment, an unsuitable diamagnetic
    reference, or an inaccurate structure, centre, tensor, or orientation.

## Working with the shift layers

Open **Modules → NMR Workbench**, process and reference vendor spectra, and
assign corresponding signals in **NMR Data**, or import/add prepared absolute
observed shifts there without a spectrum. Set **Paramagnetic** / **Diamagnetic** dataset
roles and choose **Pair with**. **Inspector → Shift Analysis** compares
assignment-matched observables in Table, Overlay, Stacked, or Shift Diagnostics
views. Existing project shift layers also appear as read-only shift lists;
they are not measured intensity traces.

The project-layer equation above is keyed by structural Ref ID. Workbench
pairing instead matches reviewed observable assignment labels; those labels do
not accept structural targets automatically. See [NMR Data](../nmr/nmr-data.md)
and the [FID/CSV tutorial](../nmr/tutorial-fid-csv.md).

Use **PCS Preparation** to choose a derivation method, map structural targets,
and review a handoff to δ_Exp. δ_para is not transferred automatically. See
[Shift Analysis and PCS Preparation](../nmr/non-pcs-analysis.md).

See [Import and Export](import-export.md) for accepted files, column matching,
preview behavior, and Ref ID troubleshooting. To provide data to fitting,
assign [experimental PCS values](delta-exp.md) explicitly as δ_Exp.
