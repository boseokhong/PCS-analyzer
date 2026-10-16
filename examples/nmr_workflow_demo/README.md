# NMR Workbench: Starting from FID or CSV

This **synthetic teaching example** provides the same four resonances through
two input paths. These are not experimental measurements. Paramagnetic and
Diamagnetic describe workflow roles; H1–H4 are illustrative correspondence
labels, not established atom or structural assignments. The FIDs also contain
a small TMS reference signal, a smooth baseline artifact and deliberate ppm
offsets so you can practice baseline correction and referencing. A separate toy
structure supports explicit target mapping and Chemical Map navigation.

```text
nmr_workflow_demo/
├─ FID/
│  ├─ Paramagnetic.fid/    # fid + procpar + text
│  └─ Diamagnetic.fid/
├─ CSV/
│  ├─ Paramagnetic.csv
│  └─ Diamagnetic.csv
├─ generate_demo.py       # Optional data regeneration script
├─ structure.xyz          # Toy Nd-centered cluster for mapping practice
└─ README.md
```

Run `python tools/nmr_workbench.py` from the repository root for standalone use,
or open NMR Workbench from PCS Analyzer. **Start each scenario in a fresh
Workbench.** Loading both input paths together creates separate datasets for
the same example. Regenerated files do not update spectra embedded in previously
saved sessions; use a fresh import for this revised exercise.

## Reference data and expected results

Both experiments use nucleus **1H** and temperature **298.15 K (25 °C)**.

| Assignment | Reference / Dia (ppm) | Observed / Para (ppm) | Δshift = Para − Dia (ppm) |
|---|---:|---:|---:|
| H1 | 1.20 | 2.00 | +0.80 |
| H2 | 3.40 | 5.00 | +1.60 |
| H3 | 5.60 | 8.00 | +2.40 |
| H4 | 7.80 | 11.00 | +3.20 |

The CSV files contain these values directly. FID peak positions differ slightly
because of the frequency grid. Use **±0.005 ppm** as the comparison tolerance
for this example. No measurement uncertainties are supplied; this grid tolerance
is not a measurement uncertainty.

## 1. Process an FID and pick peaks

Workflow: **Open FID → Phase → Baseline → TMS Reference → Peak Picking → Assign
in NMR Data → Shift Analysis**.

1. Use **File → Open Dataset Folder…** to open `FID/Paramagnetic.fid/`, then
   `FID/Diamagnetic.fid/`. Alternatively, use **Open FID File…** to select the
   `fid` file inside each folder. Keep `fid` and `procpar` together.
2. Select Paramagnetic in Experiments. The existing default FT is applied
   automatically on initial loading; you do not need to run Fourier Transform
   again.
3. Under **Processing → Phase → Manual adjustment**, enter **P0 = −25°,
   P1 = 0°** and click **Apply** once. Pivot does not affect this example because
   P1 is zero. Apply the same correction once to Diamagnetic. These values
   exactly undo the synthetic +25° phase error. Applying −25° again adds another
   correction, so do not repeat it on an already corrected spectrum.
4. Open **Processing → Baseline** and click **Auto Baseline (arPLS)**. Inspect
   the preview with Show fitted baseline and Show original spectrum enabled,
   then click **Apply**. The broad elevated/curved background should become
   nearly flat between peaks. Repeat for the other experiment. Auto Baseline
   uses the existing arPLS policy with lambda = 1e5. Alternatively, choose
   Method = arPLS in Manual adjustment, set Lambda = 1e5, and Preview → Apply.
   Do not apply both previews consecutively as if they were separate steps.
5. Open **Processing → Reference**. Set Type to **Reference standard**; the
   existing TMS preset supplies Target = **0 ppm**. Zoom near zero and use
   **Pick…** beside Observed to select the small TMS peak. Its apparent center
   is about **+0.179688 ppm for Paramagnetic** and **−0.120117 ppm for
   Diamagnetic**. Check that you selected TMS, not the stronger H1 signal.
   Click **Preview**, check Offset (approximately −0.18 / +0.12 ppm), then
   **Apply**. Repeat independently for each experiment. TMS should now be at
   0 ppm; the whole spectrum moves by the same offset. Do not apply a shared
   observed position or a residual-solvent preset to both datasets.
6. Run **Peak Picking → Auto Pick** for each experiment. The defaults
   (Min height 1%, Prominence 0.5%, Min distance 3 points) should detect
   **five peaks per experiment: four sample peaks plus TMS**. TMS is roughly
   6% of the strongest corrected peak height in Para and 3% in Dia. If a higher
   custom threshold hides it, restore the defaults or select it manually.
   The original FIDs contain no assignments or peak IDs.
7. In Inspector **NMR Data**, choose Experiment = Active experiment and
   Show = All. Click the Shift column heading to sort by ascending ppm.
   **Leave the TMS row near 0 ppm unassigned**; it is a reference standard,
   not one of the four structural observations. Starting with the lowest
   remaining shift, double-click each Assignment cell or press Enter and enter
   **H1, H2, H3, H4**. Tab advances to the next row. Assign both
   experiments and check that **Use** is selected for the assigned rows.
   If you label the reference TMS for convenience, uncheck its Use in both
   datasets so it stays outside the four-row comparison/PCS exercise.
8. Click **Compare shifts →** or open **Shift Analysis**. Choose
   Observed / Para = Paramagnetic and Reference / Dia = Diamagnetic, then
   click **Compare**. Confirm these roles if prompted.
9. Compare Table values against the reference table. With default FT, manual
   phase, Auto Baseline, TMS referencing and the default picker, Dia shifts are
   approximately 1.200195, 3.400391, 5.600586 and 7.799805 ppm, giving
   Δshift ≈ 0.799805, 1.599609, 2.399414 and 3.200195 ppm.
   Select Overlay or Stacked under View and expand Display for visual comparison.

The sample resonances are Lorentzian and no random noise is added. The artifact
and reference errors are deliberate: correct phase first, baseline second and
the reference axis third, before picking/assigning the final sample peaks.
The sample signals initially appear at their true shifts plus +0.18 ppm in Para
and −0.12 ppm in Dia. Skipping both reference corrections therefore biases
Para − Dia by approximately **+0.30 ppm**. Phase pivot selection is separate
from TMS referencing. Auto Phase can be explored, but compare its estimates
against the known manual correction on a fresh input.

## 2. Start directly from prepared CSV data

Workflow: **Import CSV → Review NMR Data → Shift Analysis**.

1. In a fresh Workbench, select Inspector **NMR Data → Import → Observed shifts
   CSV / TSV…** and open `CSV/Paramagnetic.csv`.
2. Check that Assignment maps to `assignment` and Shift maps to `shift_ppm`.
   Click **Preview**, review the four rows, then click **Apply**. The file
   supplies experiment, role, nucleus and temperature_k. ID updates and
   detached-copy options are not needed.
3. Import `CSV/Diamagnetic.csv` using the same Import → Preview → Apply steps.
   Select each experiment or All experiments in NMR Data to inspect the values.
4. In **Shift Analysis**, choose Observed / Para = Paramagnetic and
   Reference / Dia = Diamagnetic, then click **Compare**.
5. Verify that all H1–H4 assignments match and Δshift is
   **0.8, 1.6, 2.4 and 3.2 ppm**. The CSV files already specify the roles, so
   no additional role changes are needed.

This path contains no raw/processed spectrum or dataset-local peaks. Independent
NMRObservable records supply the existing analysis inputs. No picking, phase
correction or FT is required; you compare prepared observations rather than
spectral line shapes. The CSV files do not predefine session, dataset, peak or
observable IDs. Import creates new dataset/observable IDs, separate from the
identities created by the FID path.

The CSV values are already baseline-independent, TMS-referenced observed shifts.
Only the four sample observations are included; TMS is not a structural target
and does not appear as a comparison row.

## 3. Connect observations to a structure

This exercise requires the **main PCS Analyzer**, because standalone Workbench
has no structure/PCS document. It works with either the FID or CSV input path.

1. Start a fresh PCS document in the main application. Use **File → Load XYZ…**
   to load `structure.xyz` from this folder. When asked for the center atom,
   enter **Nd**. Keep averaging disabled for this single-atom mapping exercise.
   The center is Ref 1 at the origin, O atoms are Refs 2–5, and H atoms are
   Refs 6–9. XYZ comments do not automatically create H1–H4 assignment labels.
2. Open **Tools → NMR Workbench** from that document. Follow either scenario
   above, or save your standalone work as `.nmrsession` and open it through
   **VT Series → Files → Open NMR session…** in an empty integrated Workbench.
   Keep the intended PCS document active during the mapping exercise.
3. After comparing the two experiments, open **PCS Preparation**. Select H1
   and expand Selected assignment if needed. Set Target scope to **Single
   atom**, then explicitly select the PCS target with **Ref 6 (H)**. Repeat
   for the other rows using this mapping:

   | Observation label | Structure Ref | Element | XYZ position (Å) |
   |---|---:|---|---|
   | H1 | 6 | H | (3.370000, 0.000000, 0.000000) |
   | H2 | 7 | H | (−1.685000, 2.918506, 0.000000) |
   | H3 | 8 | H | (−1.685000, −2.918506, 0.000000) |
   | H4 | 9 | H | (0.000000, 0.000000, 3.370000) |

   The target dropdown displays the Ref and element, not necessarily the H1–H4
   label. If targets are unavailable, check that all four H atoms are visible
   in the main document, that the document is active, and refresh PCS Preparation.
4. Return to **NMR Data**, select H1 and choose **More → Show in Chemical Map**
   (also available in the row context menu). The selected target should be Ref 6.
   Repeat for H2–H4 to practice selection synchronization. If the Chemical Map
   hides hydrogens or labels, enable them and use Atom Map view to inspect Ref IDs.
   Selecting a row navigates to the reviewed target; it does not transfer PCS.
5. Inspect **Review & Apply…** only if you want to practice the final transfer
   review. Direct from Δshift may be used as an explicit **UI demonstration
   assumption**, not a claim that these values are genuine PCS. Check all four
   targets and destination conflicts before any explicit Apply. You can cancel
   the review and complete the structure-linking exercise without transferring.

`structure.xyz` is a nine-atom toy cluster with Nd and four O–H arms. It is not
an optimized molecule, a validated coordination model or the source of the
synthetic shifts. H1–H4 → Ref 6–9 is a deliberately supplied teaching mapping,
not a scientific assignment inferred from tracking or geometry. TMS is a
separate reference species and is not part of this XYZ structure. Do not use
this example to evaluate tensor recovery or physical PCS fitting quality.

## Saving, exporting and continuing to PCS

- **NMR Data → Export → All observations CSV…** exchanges observed shifts with
  source identities and provenance. Unassigned FID peaks are not included.
  Peak-linked CSV records reference peaks/revisions in the original session;
  reopening the vendor FID does not recreate those identities.
- **VT Series → Files → Save NMR session… / Open NMR session…** preserves and
  restores datasets, peaks, observables and committed processing state in an
  `.nmrsession` file, including this single-temperature example. You do not need
  to create a VT series first.
- **PCS Preparation** starts from a **shift difference**, which is not
  automatically PCS. Direct from Δshift requires an explicit scientific
  justification for neglecting other contributions.
- Standalone mode has no bound PCS document or structure. Review real structures
  and targets and transfer to PCS through the main PCS Analyzer application.
  Use section 3 for the toy mapping exercise. Structural targets are not
  inferred from H1–H4. For a separate PCS fitting
  example, see `examples/documentation_demo/Nd_iPr2BA3_demo/`.

## Synthetic signal specification and reproduction

Each Varian/Agilent file container holds 16,384 complex time-domain points.
The spectral width is 6,400 Hz, reference frequency is 400 MHz and carrier is
6 ppm. Four causal exponential sample signals are summed, with FWHM of 12 Hz
for Para and 6 Hz for Dia, and relative amplitudes of 1.0 / 0.85 / 0.7 / 0.6.
A fifth signal represents TMS at true 0 ppm, with relative time-domain amplitude
0.015 and FWHM 3 Hz. A common shift offset is added to every signal in each
experiment: +0.18 ppm for Para and −0.12 ppm for Dia.
The signals follow the VNMR positive-exponent FT convention, with half weighting
of the first time point. The global phase error is +25°. No random noise is
added, so regeneration is deterministic. Before applying that global phase,
a smooth real spectral baseline is added numerically via the inverse VNMR
transform. Its shape is `0.08 * peak_max * (1 + 0.6*x + 0.8*x² + 0.3*cos(π*x))`
for normalized `x` from −1 to +1. This is an artificial baseline artifact,
not a physical instrument response model.

To regenerate only the example data files, use Python with the application's
NumPy and nmrglue dependencies installed. The explicit option overwrites the
existing FID, CSV and `structure.xyz` example files:

```bash
python examples/nmr_workflow_demo/generate_demo.py --overwrite
```

Both paths were checked through the existing APIs: vendor loader → FT/phase →
arPLS baseline → TMS reference → peak picker → observable synchronization →
Shift Analysis, and CSV parser/import → Shift Analysis. The revised FIDs yield
five peaks (including TMS); the four assigned sample observations match the
expected values within ±0.005 ppm. The XYZ parser and canonical single-atom
target/assignment-selection APIs were also checked for Refs 6–9.
This was a scientific/API check; interactive Tk operations and the full
regression suite were not run. Application code and existing measured examples
were not modified for this demo.
