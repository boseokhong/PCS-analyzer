# VT NMR Workbench tutorial

Synthetic teaching data for the complete VT workflow: import a series, batch-process
its FIDs, inspect stacked spectra, link peaks manually, and analyse temperature
dependence. These are simulated signals, not experimental measurements or a
physical model of the supplied structure.

## Files

```text
nmr_vt_demo/
  FID/Paramagnetic/01_Paramagnetic_298K.fid/
  FID/Paramagnetic/02_Paramagnetic_278K.fid/
  FID/Paramagnetic/03_Paramagnetic_318K.fid/
  FID/Paramagnetic/04_Paramagnetic_288K.fid/
  FID/Paramagnetic/05_Paramagnetic_308K.fid/
  FID/Reference/Diamagnetic_298K.fid/
  CSV/Expected_shifts.csv
  structure.xyz
  generate_demo.py
```

Each Varian experiment directory contains `fid`, `procpar`, and `text`.
Temperature is imported from vendor metadata; Kelvin is used internally.
The directory prefixes intentionally give acquisition order **298, 278, 318,
288, 308 K**, different from temperature order. All temperatures are nominal;
there is no calibration or member override in the generated data.

The CSV is a ground-truth check containing 24 corrected observed shifts and
descriptive provenance. Importing it as prepared NMR Data creates independent
observable records; it does **not** recreate FIDs, peaks, series, or PeakTracks.
Do not import it into the same practice session as a shortcut for tracking.

## 1. Import the five-member series

Open NMR Workbench from PCS Analyzer, or run `tools/nmr_workbench.py` from the
repository root for standalone UI practice. Use the main application for the
optional structure/PCS exercise.

1. Open **VT Series** and create a series named `Synthetic VT`.
2. Use **Import > Folder group…** on `FID/Paramagnetic`, or select the five
   experiment directories with **FID experiments…**.
3. Check that five members are present, included, and have valid temperatures.
4. Compare **Acquisition**, **Temperature ↑**, and **Temperature ↓** ordering.

Keep `FID/Reference` outside this series. It contains a separate diamagnetic
reference, not a sixth temperature measurement. A member temperature override is
available for practice, but clear it again before fitting; it does not rewrite
the original acquisition metadata.

## 2. Batch phase, baseline, reference, and peak picking

Open **Batch Processing**, select `Synthetic VT`, and check all five experiments.
Choose **Current spectrum**. Untouched FIDs receive the default FT automatically.
Leave apodization and zero filling off for this reproducible exercise.

Enable and order these operations. For each operation, enter its settings and
choose **Set common** before moving to the next operation.

| Order | Operation | Settings |
| --- | --- | --- |
| 1 | Phase | Manual; P0 = **−25°**; P1 = **0°**; strongest-peak pivot |
| 2 | Baseline | **arPLS**; lambda = **100000** |
| 3 | Reference | Reference standard **TMS**, proton signal, target **0 ppm**; search region **−0.30 to +0.30 ppm** |
| 4 | Peak picking | Height fraction **0.01**; prominence fraction **0.005**; minimum separation **3 points** |

The common reference search region resolves a separate observed TMS peak in each
experiment. Do not enter one common observed peak position: the intentional
reference offsets differ between members.

| Acquisition | Temperature / K | TMS before referencing / ppm |
| --- | --- | --- |
| 1 | 298 | +0.18 |
| 2 | 278 | −0.14 |
| 3 | 318 | +0.22 |
| 4 | 288 | +0.07 |
| 5 | 308 | −0.09 |

Select an experiment row to inspect its input spectrum. Choose **Preview checked**,
inspect the overlay and member results, then **Apply reviewed previews** explicitly.
Expect five successful members, each with **four sample peaks plus a small TMS
peak** near zero. The broad baseline artifact is intentional. The known manual
phase correction keeps this first exercise independent of automatic-phase choices.

Do not apply −25° a second time to an already corrected current spectrum. To repeat
from scratch, use a fresh import/session or explicitly restart from **Original FID**.
Preview does not commit processing history or replace peaks; Apply does.

## 3. Inspect the stacked spectra

Switch to **VT stacked**. Choose temperature order, adjust spacing, and optionally
normalize for display. Select a trace to activate its existing dataset.
Include/exclude a member in VT Series to see the corresponding trace disappear
and return. Restore all five members to included before continuing.

Display normalization and offsets do not change peak measurements. Each member
retains its own ppm grid, processing history, and peak IDs.

## 4. Create four tracks manually

Open **Peak Tracking** for this series. Create tracks labelled `H1` through `H4`.
For each track, visit all five members, choose the corresponding **existing peak**,
and choose **Link peak**. Use the four sample resonances in ascending ppm order;
leave TMS untracked and unassigned.

At 298 K, H1–H4 are approximately **2, 5, 8, 11 ppm**. Each moves smoothly with
temperature. Inspect linked-point highlighting in the stacked view. Expect four
tracks with five confirmed links each.

Optional assisted practice: after linking an anchor, set the candidate window to
**0.5 ppm**, choose **Suggest**, inspect its reasons, and explicitly **Accept
candidate** only when it is the intended peak. This wider window is appropriate
for these well-separated synthetic peaks, not a universal experimental setting.
The current suggestion service uses acquisition-order continuity; changing display
order to temperature order does not change its scoring. A score is not an
assignment probability. Manual linking remains available when there is no candidate.

Try marking a point missing or ambiguous, then restore the correct link manually.
Neither state should enter a fit. Reprocessing or replacing peaks can make links
stale; review/relink them explicitly rather than assuming nearest-ppm recovery.
A track label identifies correspondence only: it does not assign an atom or create
dataset observables.

## 5. Analyse temperature dependence

Open **VT Analysis** and select one or more tracks. Use **δ vs 1/T** with
**Linear (coordinate)** first, then choose **Curie** with unweighted fitting.
The generator uses observed chemical shift:

```text
δ(T) = offset + C/T
```

| Track | Offset / ppm | C / ppm K | δ at 298 K / ppm |
| --- | --- | --- | --- |
| H1 | 1.2 | 238.4 | 2.0 |
| H2 | 3.4 | 476.8 | 5.0 |
| H3 | 5.6 | 715.2 | 8.0 |
| H4 | 7.8 | 953.6 | 11.0 |

Expect five fit points per track and R² close to one. Peak-grid quantization and
processing mean fitted parameters need not equal generator constants exactly.
No measured uncertainty is supplied; do not invent sigma values or enable measured-
uncertainty weighting for this example.

Also inspect **δ vs T**, **δT vs 1/T**, residuals, and model comparison. With a
nonzero offset, δT vs 1/T is not generally a straight line. The physical Curie model
fits raw observed δ against temperature and transforms its curve for display.
Curie is mathematically equivalent to a linear δ vs 1/T fit here. Curie-Weiss adds
a parameter; five points and good fit quality alone do not establish physical
meaning or reliable extra parameter precision.

Exclude one valid point using the analysis fit mask: n should decrease to four,
while the series member stays included and its track link remains confirmed.
Restore the mask, select a plotted point, and navigate back to its member/peak.
Export the analysis CSV, plot, or results JSON through the analysis export menu.
The CSV contains actual extracted/fitted values; `Expected_shifts.csv` contains
the ideal generator values for comparison.

## 6. Optional assignment and structure review

From the tracking/analysis workflow, open **Assignment review**. For each track,
enter the corresponding teaching assignment `H1`–`H4`, nucleus `1H`, and a note
identifying this synthetic example. Choose **Associate**, inspect member proposals,
select the intended members, then **Confirm selected** explicitly.

Expect four dataset-local observables per experiment. The association is reviewed
interpretation; correspondence alone is not assignment evidence. Existing conflicting
assignments must be resolved rather than overwritten silently.

In the main application, load `structure.xyz`. It is a nine-atom toy cluster in
Angstrom, not an optimized molecule. Nd is the centre at Ref 1; manually review
these teaching target mappings in the existing structural-target UI:

| Assignment | Structure Ref |
| --- | --- |
| H1 | 6 |
| H2 | 7 |
| H3 | 8 |
| H4 | 9 |

Labels do not infer structural targets. Use the existing atom/chemical map to
inspect the structure; review target mapping explicitly before any PCS handoff.
TMS has no target in this structure.

## 7. Optional 298 K reference / PCS preparation

Import `FID/Reference/Diamagnetic_298K.fid` separately. Process it using the same
phase/baseline/TMS/peak-picking settings, individually or in its own temporary
batch series. Its initial TMS is −0.12 ppm. Mark its dataset role diamagnetic and
assign the four sample peaks H1–H4. Mark the five VT datasets paramagnetic.

In **VT PCS preparation**, inspect references for a reviewed VT observation.
Only the **298 K** member has a matching-temperature reference here. Other
temperatures must remain unresolved; do not reuse 298 K silently for them.
Select the reference and structural target explicitly.

The expected 298 K observed shift differences are **0.8, 1.6, 2.4, 3.2 ppm**.
These are **shift differences**, not independently established PCS. Use the
shift-difference-only provider to inspect provenance. This synthetic example does
not supply a scientific basis for direct-PCS interpretation, contact separation,
or tensor recovery. If exercising reviewed transfer under an explicitly declared
teaching assumption, keep it separate from real scientific work and review the
destination conflicts before applying.

## Save and regenerate

Save a **`.nmrsession`** to recover committed spectra, dataset/peak identities,
series, tracks, and reviewed associations. A lightweight **`.nmrseries`** definition
alone does not provide the complete recoverable session. Processing previews,
render caches, and transient selection are not saved. Keep analysis results as
session evidence through the analysis menu if desired.

The committed FIDs are ready to use; regeneration is optional. From the repository
root, with the application's NumPy and nmrglue dependencies available:

```sh
python examples/nmr_vt_demo/generate_demo.py --overwrite
```

Without `--overwrite`, the generator refuses to replace existing output files.
It does not modify application code or create pre-linked scientific identities.
