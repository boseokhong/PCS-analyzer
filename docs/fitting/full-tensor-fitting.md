# Full Tensor Fitting

**Mode C · Full tensor + metal** jointly refines tensor-frame orientation and metal
position against selected experimental PCS values. Depending on the tensor
checkboxes, the actual parameter vector contains six, seven, or eight
variables.

## Parameter combinations

The following parameters are always fitted:

- three Cartesian Euler angles;
- metal x, y, and z.

The tensor controls add parameters as follows:

| Fit Δχ_ax | Fit Δχ_rh | Number of variables | Tensor behavior |
| --- | --- | --- | --- |
| Off | Off | 6 | Both current tensor components remain fixed |
| On | Off | 7 | Δχ_ax fitted; current Δχ_rh fixed |
| Off | On | 7 | Current Δχ_ax fixed; Δχ_rh fitted |
| On | On | 8 | Both tensor components fitted |

Count the variable parameters for the chosen checkbox settings rather than
assuming that every full-tensor run fits eight parameters.

## Initialization and bounds

The Euler angles start at 0° unless **Use current tensor as initial guess**
is selected, which uses the current orientation. The metal coordinates start
at the currently assigned centre, and each coordinate is restricted to ±1 Å around that
starting value. A tensor component starts from its current input value when
it is fitted.

Euler angles are bounded from −180° to +180°. Tensor components are unbounded
in the final refinement. **Global search (DE)** provides a preliminary search;
see [Initialization](overview.md#initialization-bounds-and-optimization).

## Geometry evaluated during refinement

For each trial centre, the rigid structure is translated relative to that
centre, rotated with the active extrinsic X → Y → Z convention, and evaluated
in the tensor frame. The structure's internal coordinates do not deform.
Selected experimental observations provide one residual each, including
structured groups. Symmetry- and torsional-averaged geometry factors are used
when those features are active.

See [Coordinate Handling](../structures/coordinate-system.md) for the rotation
convention and [Tensor Conventions](../pcs/tensor-conventions.md) for Δχ
definitions and units.

## Results and application

The result report shows the refined metal coordinates, Euler α/β/γ values,
the fitted or fixed tensor values, quality metrics, and per-point residuals.
The displayed Greek angle names correspond to the same three X/Y/Z rotation
parameters reported as `ax`, `ay`, and `az` in Mode B.

**Apply to Tensor** copies Δχ_ax, Δχ_rh and orientation into the current
tensor, clears an existing geometry override, and refreshes the views. It
retains the assigned metal position, prediction model and rhombic enablement.
It does **not** permanently adopt the refined centre. Main predictions after
application can therefore differ from the fit report.

**Preview Geometry** temporarily uses the fitted orientation and refined
metal position in the views. It also fills and confirms the fitted Δχ input
values through an immediate redraw, retaining current tensor orientation,
prediction policy and rhombic enablement. It does not reassign the metal atom.
**Clear Preview** removes temporary geometry while keeping those current tensor
settings. Undoing Preview restores the previous geometry and tensor values;
Undoing Clear restores the preview. Neither action reruns the fit.
Saving a current project retains this active preview, including its refined
centre and orientation, separately from the assigned metal centre. Reopening
restores the preview without refitting or applying the result to the canonical
tensor. Older files that omitted those values disable the incomplete preview;
recreate it from the saved Fit History run if its status allows. See
[Projects and Persistence](../projects.md).

## Interpretation and limitations

Metal position, orientation, axial anisotropy, and rhombicity can compensate
for one another, especially with few observations or poor spatial coverage.
Use more independent PCS points than fitted parameters, inspect the residuals,
and compare repeated fits or constrained alternatives. The application does
not currently report parameter covariance or nonlinear confidence intervals.

See [Quality Metrics](quality-metrics.md) and
[Rhombicity Diagnostics](../diagnostics/rhombicity.md) for the available
checks.
