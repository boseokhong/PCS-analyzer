# Tutorial: Process and Analyse a VT Series

Use `examples/nmr_vt_demo/` for a five-temperature workflow:
**import → batch processing → stacked spectra → manual tracking → VT analysis
→ optional assignment/PCS review**. These are simulated teaching signals,
not measurements or a physical model of the supplied structure.

## Files and expected behavior

[Get the example input directory](https://github.com/boseokhong/PCS-analyzer/tree/main/examples/nmr_vt_demo).
The same inputs are included in the portable release; see
[finding the example files](../examples.md#find-and-open-the-files).
Short paths below are relative to `examples/nmr_vt_demo/`. Keep each experiment's
`fid` and `procpar` together.

| Input | Use |
| --- | --- |
| `FID/Paramagnetic/` | Five members, acquisition order **298, 278, 318, 288, 308 K**. |
| `FID/Reference/Diamagnetic_298K.fid/` | Separate reference for the optional 298 K comparison; not a sixth VT member. |
| `CSV/Expected_shifts.csv` | 24 ideal corrected observed shifts, including the reference values. |
| `structure.xyz` | Toy nine-atom Nd/O/H cluster for optional target review. |

Open a fresh **NMR Workbench** from PCS Analyzer's **Tools** Ribbon. Use the main app for
structural operations. All provided temperatures are nominal; no calibration
or member override is supplied.

Importing `Expected_shifts.csv` through NMR Data creates independent values.
It does **not** recreate FIDs, picked peaks, series or tracks. Keep that
prepared-data exercise separate from this FID practice session.

## 1. Create and import the series

1. Open **VT Series → Manage VT Series…**, choose **New**, name it `Synthetic VT`.
2. Use **Import → Folder group…** on `FID/Paramagnetic`, or select the five
   `.fid` directories through **FID experiments…**.
3. Check five included members with valid temperatures and no import failures.
4. Compare **Acquisition**, **Temperature ↑**, and **Temperature ↓**. Ascending
   temperature order is **278, 288, 298, 308, 318 K**.

Keep the reference outside the series. If practising **Override / K**, clear it
again before fitting. See [Series and Stacked Spectra](vt-series.md).

## 2. Preview and apply processing

Open **VT Series → Batch Processing…**, select the series and check all five
experiments. Use **Current spectrum**; untouched FIDs receive default FT.
Leave apodization/zero filling off. Enable and order these operations, storing
each operation's parameters with **Set common** before moving to the next:

| Order | Operation | Settings |
| --- | --- | --- |
| 1 | Phase | Manual, **P0 −25°**, **P1 0°**, strongest-peak pivot. |
| 2 | Baseline | **arPLS**, lambda **100000**. |
| 3 | Reference | **Reference standard → TMS**, proton signal, **0 ppm**; search region **−0.30 to +0.30 ppm**. |
| 4 | Peak Picking | Height fraction **0.01**, prominence fraction **0.005**, distance **3 points**. |

The search region independently finds the small TMS peak in each member.
Initial TMS offsets in acquisition order are **+0.18, −0.14, +0.22, +0.07,
−0.09 ppm**. Do not share one observed reference position across members.

Choose **Preview checked**, inspect the member results and **Overlay**, then
**Apply reviewed previews** explicitly. Expect **five successful members**,
each with **four sample peaks plus TMS near zero**. Use the
[batch preview image](batch-processing.md) as a layout guide. Preview itself
does not commit history or peaks.

Do not apply −25° again to an already corrected current spectrum. Start a fresh
import/session, or review an **Original FID** recipe with the correct FT order
to repeat from scratch. Phase pivot and TMS chemical-shift reference are
separate settings.

## 3. Inspect spectra in temperature order

Choose **VT stacked**, **Temperature ↑**, and adjust spacing/normalization.
Select a trace to activate its experiment. Temporarily exclude a member in the
series manager and inspect its trace disappearing; restore all five members
before continuing. These display changes do not change peak measurements.

## 4. Establish four tracks

Open **Tracks** / **Peak Tracking…**. Create H1–H4. For each track, visit all
five members, select the corresponding **Existing peak**, then **Link peak**.
Use the four sample resonances in ascending ppm order; leave TMS untracked
and unassigned. At 298 K they are approximately **2, 5, 8, 11 ppm**.
Expect **four tracks with five confirmed links each**.

Optional assisted practice: link an anchor, set **Candidate window / ppm = 0.5**,
then **Suggest**. Inspect reasons before **Accept candidate**. This window is
appropriate for these separated teaching signals, not a general experimental
default. Scoring uses acquisition order even when display uses temperature order.

Try marking a point missing/ambiguous and then restore its correct link. These
states should not enter the fit. Reprocessing/re-picking can stale links and
requires explicit review. Track labels do not create dataset observables or
infer molecular/structural assignments. See [Peak Tracking](peak-tracking.md).

## 5. Fit observed-shift temperature dependence

Open **VT Analysis…**, select a track, choose **δ vs 1/T**, and first inspect
**Linear (coordinate)**. Then choose **Curie**, **unweighted**. The generator uses:

\[
\delta(T)=a+C/T.
\]

| Track | a / ppm | C / ppm·K | δ at 298 K / ppm |
| --- | ---: | ---: | ---: |
| H1 | 1.2 | 238.4 | 2.0 |
| H2 | 3.4 | 476.8 | 5.0 |
| H3 | 5.6 | 715.2 | 8.0 |
| H4 | 7.8 | 953.6 | 11.0 |

Expect five fit points per track and R² close to one. Grid sampling, processing
and reference picking affect the measured shifts, so fitted coefficients need
not equal generator constants exactly. No measured uncertainty is supplied;
do not invent σ or use **measured_sigma** for this exercise.

Also inspect **δ vs T**, **δT vs 1/T**, residuals and **Compare models (raw δ / ppm)**.
Curie fits raw observed δ and transforms its curve for display. The nonzero
offset means δT vs 1/T is not generally linear. Five points alone do not
establish Curie-Weiss parameter precision or physical meaning.

Exclude one valid row using **Exclude from fit**: n should decrease to four
while member inclusion and the confirmed track link remain unchanged. Restore
it, select a plotted point and navigate back with **Show spectrum**. Export the
analysis CSV/plot/results JSON. Exported results are extracted/fitted values;
`Expected_shifts.csv` is the ideal generator table. See [VT Analysis](vt-analysis.md).

## 6. Review assignments and optional structure targets

From tracking/analysis **Workflow → Assignment review…**, select each track,
enter its H1–H4 teaching assignment, **1H**, and a synthetic-example review note.
Choose **Associate**, inspect the member proposals, select all intended members
and **Confirm selected**. Expect four dataset-local observations per experiment.
TMS remains outside the assignments. Resolve existing conflicts explicitly.

For structure review in the main app, load `structure.xyz`, choose Nd as centre,
keep averaging disabled, and check the Workbench's **PCS:** binding. Explicitly
review single-atom targets **H1 → Ref 6, H2 → Ref 7, H3 → Ref 8, H4 → Ref 9**
using the existing PCS Preparation/structural-target controls and Chemical Map.
Labels do not infer these targets. The toy XYZ is not optimized, is not the
physical source of the shifts and contains no TMS atom.

## 7. Optional: Compare with the 298 K reference

Import `FID/Reference/Diamagnetic_298K.fid` separately. Correct phase/baseline/
TMS reference and pick peaks with the same parameters, individually or in a
separate temporary batch series. Its initial TMS is **−0.12 ppm**. Set its role
diamagnetic and assign its four sample peaks H1–H4; set the five VT datasets
paramagnetic.

Open **Workflow → VT PCS preparation…** (tracking labels the action **PCS
preparation…**). Choose a reviewed observation, compatible reference and
structural target explicitly. Only **298 K** has a matching reference here;
other temperatures remain unresolved. Expected 298 K differences are
**0.8, 1.6, 2.4, 3.2 ppm**, approximately subject to spectral sampling.

Use **shift_difference_only** to inspect provenance. These differences are not
independently established PCS. The example supplies no scientific basis for
direct PCS, contact separation or tensor recovery. If practising transfer under
an explicitly declared teaching assumption, review destination conflicts and
keep it separate from real scientific work. See [VT PCS Preparation](vt-pcs-preparation.md).

## Save and resume

Choose **Export → Keep results in session** in VT Analysis when retaining fit
evidence, then save **Files → Save NMR session…** in the series manager. Reopen
in a fresh Workbench and inspect identities, five members, four tracks and
associations. `.nmrseries` definitions alone do not preserve the full session.
Previews/caches/transient selections are not saved. See [Sessions](sessions.md).

The supplied FIDs are ready to import. Peak/track identities are established
during this exercise; start with fresh imports rather than previously embedded
session spectra.
