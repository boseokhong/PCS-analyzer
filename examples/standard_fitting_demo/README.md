# Standard fitting and rhombicity demo

These are noiseless synthetic geometrical probes, not an optimized molecule or
experimental evidence. Each project contains Nd at Ref 1, a donor N at Ref 2 and
32 H probes at Refs 3–34. Coordinates are Å, PCS is ppm, and tensor values use
E-32 m³. `truth.csv` records the explicit generating parameters. Independent
Cartesian equations generate PCS; the application fitter is used for validation.
The supplied 0.02 ppm uncertainty is an objective scale, not random noise or
experimental precision. No result or temporary geometry override is pre-applied.

| Project | Known input | Exercise |
|---|---|---|
| `mode_a/demo.pcsp` | Axial 12, rhombic 0; first donor Ref 2; θ=20°, α=25° | Donor-axis Mode A |
| `mode_b/demo.pcsp` | Axial 12, rhombic 3; active extrinsic XYZ 12°, −8°, 18°; centre at origin | Mode B orientation fitting |
| `mode_c/demo.pcsp` | Same scalars/orientation; true centre (0.20, −0.15, 0.10) Å; assigned start (0,0,0) | Mode C tensor and centre refinement |
| `axial/demo.pcsp` | Axial 12, rhombic 0; zero orientation | Axial diagnostic reference |
| `rhombic/demo.pcsp` | Axial 12, rhombic 3; zero orientation | Nonzero-rhombicity diagnostic reference |

## Fit in the application

1. Open the desired `.pcsp` with **Open Project**, then select **Fitting**.
2. Expand **Fit settings**, explicitly choose the indicated mode and select all
   32 PCS observations. Selection is deliberately
   not restored by project loading: verify the count before running.
3. For A, choose donor Ref **2**, **Axis vector mode: first**, fit Δχ_ax, and keep
   Δχ_rh fixed at zero. For B/C, fit both tensor components. The project selects
   its corresponding mode, uncertainty weighting and no global search.
4. Run **Run Fit** in the Ribbon. Expect RMSD below 0.00001 ppm after project
   coordinate rounding. All cases should recover axial 12; B/C recover rhombic 3.
   C should refine the centre to (0.20, −0.15, 0.10) Å.
5. Review the recorded conditions and predictions. Equivalent angular/tensor
   parameterizations can describe the same predictions; small residuals do not
   demonstrate global uniqueness or reliable uncertainty for real data.
6. Repeat a fit and inspect **Fit History** to compare its conditions/metrics.
   A new fit records history; opening these input-only projects does not.

Mode A **Preview Geometry** shows the donor-dependent orientation; **Apply to
Tensor** is unavailable. Mode B Apply retains the existing rhombic value even if
it was fitted. Mode C Apply preserves the assigned metal coordinate; Preview
Geometry uses the refined centre temporarily. Reopen the source project between
exercises to clear temporary overrides. Do not save review state over the supplied
input project unless intentionally making your own copy.

Mode C persistence exercise: after fitting, choose **Preview Geometry** and
**Update**, save a new `.pcsp` copy, close the document and reopen that copy.
The active preview centre/orientation and displayed PCS are restored without
refitting or changing the assigned metal position. The original input-only
project has no preview. Older files containing only a Mode C preview flag cannot
recover missing values; recreate the preview from a saved Fit History result.

## Rhombicity Check and Table

Open `axial/demo.pcsp`, select **Rhombicity Check**, and choose **Update Diagnostic** after
structure and δ_Exp are present. Compare with `rhombic/demo.pcsp`, using all 32
observations and zero orientation in each case. The axial dataset has near-zero
axial residuals; the rhombic dataset leaves a structured residual under an axial-
only fit. Inspect residual against φ and G_rh, the tensor reconstruction and the
**Rhombicity Table**. A full two-column G_ax/G_rh regression recovers 12 / 3 for
the rhombic case. Diagnostic residual sign is δ_Exp−δ_Pred; standard fit reports
use the opposite sign. Do not rotate the main structure between paired comparisons.

These fixtures intentionally remove contact shifts, structural uncertainty and
noise to explain the diagnostic. They do not define an experimental acceptance
threshold or validate statistical inference.

## Try session Undo/Redo

Open `mode_c/demo.pcsp`, run Mode C with the demonstrated settings, then use
**Apply to Tensor**. Click the Undo arrow beside Help: the previous main tensor
returns while the calculated fit report remains in Fit History. Click Redo to
restore the applied tensor without refitting. The dropdown supports returning
through multiple recorded Apply operations. Save/reopen keeps the current
scientific state but starts an empty session history. Mode A continues to use
its existing geometry-preview action.

## Try parameter history

In a scratch `mode_c/demo.pcsp` document, change Δχ_ax and confirm with Enter.
Undo/Redo restores the previous/applied tensor without refitting. Try Structure
XYZ or Tensor Euler ZYZ sliders: one complete pointer drag adds one record,
while individual keyboard steps are recorded separately. Neither changes the
existing XYZ/ZYZ conventions.

Right-click a raw atom row and choose **Set Ref … as metal center (clear preview)**.
This intentionally changes the scientific centre to that atom's raw position;
Undo restores the original centre and any removed preview. Use a scratch document
for this exercise rather than overwriting the known-tensor checkpoint.

## Try structure replacement history

Use a scratch copy of `mode_c/demo.pcsp`. Choose **Load XYZ**, select
`axial/structure.xyz` and enter `Nd` as the centre element. The atom coordinates
are used in absolute Å and new Refs start at 1. Previous shift layers, grouped
PCS targets and temporary geometry are cleared; the current tensor is retained.
Undo restores the original Mode C structure, assigned centre, experimental shifts
and any previous geometry preview. Redo restores the accepted new structure
without another centre prompt. Try canceling the centre dialog first: the original
structure and shifts should remain. Save to a new project path for this exercise.

## Try Reset history and Enter confirmation

For fitted geometry, run Mode C and choose **Preview** in the Fitting Results
ribbon group. The refined geometry and fitted Δχ strengths are shown immediately.
Choose **Clear Preview** to remove temporary geometry while retaining current
tensor settings. Undo restores the removed preview; Undo again restores the
state before Preview, including tensor values. Redo restores the saved states
without rerunning the fit. Try Mode A/B as well: their previews retain the
current tensor strengths. Fit reports and original raw atoms remain unchanged.

In a scratch Mode C demo document, enable a nonzero rhombic term using the
existing tensor controls, then select **Axial** and **Full tensor (Axial +
Rhombic)** under **Prediction model**. The PCS predictions should differ while
the stored tensor values stay the same. Undo/Redo restores the selection and
predictions without fitting again. With the rhombic term disabled or zero, the
two models can give identical predictions; the selection itself is still recorded.
Selecting the same model again adds no history. Finish pending tensor/orientation
input before changing models. Save a scratch copy and reopen it to verify the
selected model persists; its session Undo stack starts empty.

In a scratch project, collapse Plot & display, edit Δχ_ax and press Enter.
The tensor and main predictions update without opening that section; keypad
Enter is connected to the same command. A Mode C preview keeps its geometry
while accepting the newly confirmed working axial value; the original fit
report remains unchanged.

After recording a standard fit, change the PCS contour range and use Control
Panel Reset. The main working structure/data and standard Fit History clear.
Undo restores them, the tensor/policy, preview, visibility and previous range;
Redo resets them again without fitting. Save a restored copy to a new `.pcsp`
path: reopening retains its scientific state and reports with an empty edit stack.

## Combined history review

Use a scratch copy of the Mode C project, keeping all supplied files unchanged:

1. Run a fit and Preview its geometry; switch the prediction model, then Clear Preview.
2. Confirm a new Δχ_ax value with Enter, then clear δ_Exp through the existing command.
3. Load a scratch XYZ structure and Reset the document.
4. Undo each action and check the tensor input, prediction-model label, table,
   geometry and Fit History. Redo through the same states. Use the history
   dropdown to restore several steps together.
5. Undo a few steps, make a new confirmed edit, and verify the old redo branch
   disappears. Open/close a history dropdown and confirm that input editing works.
6. Switch to another document and return: its history must stay independent.
   Save a new scratch project and reopen it: values/reports should persist,
   with an empty session Undo stack.

Redo uses embedded values rather than rerunning fitting or rereading a replacement
XYZ. Preview can change result validity through its existing dependency checks;
Undo/Redo retains the stored report rather than bypassing those checks.

## Reproduce

```shell
python examples/standard_fitting_demo/generate_demo.py --output dev_docs/standard-review
python examples/standard_fitting_demo/generate_demo.py --overwrite
```

See the [public fitting tutorial](../../docs/fitting/standard-fitting-tutorial.md),
[fitting behavior](../../docs/fitting/overview.md) and
[rhombicity definitions](../../docs/diagnostics/rhombicity.md).
