# Tutorial: From FID or CSV to Shift Comparison

This synthetic teaching example provides four resonances through two input
paths. It exercises processing, assignments, shift comparison and optional
structure mapping. It is not measured data or a validated model of PCS.

## Get the example files

Use the repository folder `examples/nmr_workflow_demo/`:

- [Example directory and complete README](https://github.com/boseokhong/PCS-analyzer/tree/main/examples/nmr_workflow_demo)
- [Paramagnetic CSV](https://github.com/boseokhong/PCS-analyzer/blob/main/examples/nmr_workflow_demo/CSV/Paramagnetic.csv)
- [Diamagnetic CSV](https://github.com/boseokhong/PCS-analyzer/blob/main/examples/nmr_workflow_demo/CSV/Diamagnetic.csv)
- [Toy structure](https://github.com/boseokhong/PCS-analyzer/blob/main/examples/nmr_workflow_demo/structure.xyz)

For FID practice, obtain the repository source and retain the complete
`FID/Paramagnetic.fid/` and `FID/Diamagnetic.fid/` directories, including
`fid`, `procpar` and `text`. For CSV-only practice, download the raw CSV files
from their GitHub file pages. Do not save an HTML preview as a CSV.

Open Workbench from PCS Analyzer, or run `python tools/nmr_workbench.py` from
the repository root with the application's runtime dependencies installed.
**Start each input route in a fresh Workbench.** Using both routes together
creates separate datasets for the same teaching observations.

Both experiments are **1H, 298.15 K**. Expected referenced values are:

| Assignment | Dia / ppm | Para / ppm | Para − Dia / ppm |
| --- | ---: | ---: | ---: |
| H1 | 1.20 | 2.00 | 0.80 |
| H2 | 3.40 | 5.00 | 1.60 |
| H3 | 5.60 | 8.00 | 2.40 |
| H4 | 7.80 | 11.00 | 3.20 |

H1–H4 are correspondence labels. The role names do not establish molecular
assignments, contact-shift separation or physical PCS.

## Route A: Process the FIDs

1. Use **File → Open Dataset Folder…** to load both `.fid` directories.
   The initial default FT creates a starting spectrum; do not run FT again.
   Leave apodization and zero filling off for this exercise.
2. Select Paramagnetic. In **Processing → Phase → Manual adjustment**, enter
   **P0 = −25°**, **P1 = 0°**, then **Apply** once. Repeat for Diamagnetic.
   This undoes the synthetic +25° phase error; a second application adds
   another correction. Pivot does not affect this P1 = 0 example.
3. In **Baseline**, choose **Auto Baseline (arPLS)**, inspect the fitted
   baseline/original-spectrum preview, then **Apply**. Repeat for both datasets.
   The default lambda is **1e5**. Alternatively use manual arPLS with lambda
   1e5 and Preview → Apply, rather than applying both methods consecutively.
4. In **Reference**, choose **Reference standard**, TMS, target **0 ppm**.
   Zoom near zero and **Pick…** the small TMS signal: approximately
   **+0.179688 ppm** in Para and **−0.120117 ppm** in Dia. Preview the offset
   (about −0.18 / +0.12 ppm) and Apply separately. Check TMS now lies at zero.
5. Choose **Peak Picking → Auto Pick** for each dataset. Defaults of **1%**
   height, **0.5%** prominence and **3 points** separation should yield
   **five peaks: four sample peaks and TMS**. Restore defaults if a custom
   threshold misses the small TMS peak.
6. In **NMR Data**, choose **Active experiment**, **Show = All**, and sort
   **Shift / ppm** ascending. Leave TMS near zero unassigned. Assign the four
   remaining peaks H1–H4 using double-click/Enter and Tab. Repeat in both
   datasets and check **Use** for all four sample observations. If you label
   TMS for convenience, exclude its Use in both datasets.
7. Choose **Compare shifts →**, select **Observed / Para = Paramagnetic** and
   **Reference / Dia = Diamagnetic**, then **Compare**. Confirm roles if asked.

Expect four matched rows. Peak-grid sampling gives differences approximately
**0.799805, 1.599609, 2.399414, 3.200195 ppm**. Compare with the table above
within **±0.005 ppm**; this teaching tolerance is not measured uncertainty.
Try **Overlay** or **Stacked** and expand **Display** for plot options.

The smooth baseline, global phase error and reference offsets are deliberate.
Skipping both reference corrections biases Para − Dia by about **+0.30 ppm**.
Phase pivot selection is separate from chemical-shift referencing. No random
noise is added. Automatic phase methods can be explored on a fresh input, but
compare them with the known manual correction before accepting results.

## Route B: Import prepared CSV values

1. In a fresh Workbench, choose **NMR Data → Import → Observed shifts CSV / TSV…**
   and open `CSV/Paramagnetic.csv`.
2. Check **Assignment = assignment**, **Shift = shift_ppm**. Choose **Preview**,
   review four rows, then **Apply**. The file supplies experiment, role,
   nucleus and temperature; ID-update/detached-copy options are unnecessary.
3. Repeat with `CSV/Diamagnetic.csv`. Use **All experiments** to inspect both.
4. Open **Shift Analysis**, choose Paramagnetic and Diamagnetic explicitly,
   then **Compare**. Expect four matches and differences **0.8, 1.6, 2.4,
   3.2 ppm**. Roles are already supplied by the files.

These values are already referenced. This route has no raw trace or picked
peaks and needs no FT, phase or baseline correction. An overlay of prepared
shifts is a stick/display representation, not measured lineshape data. TMS is
not present in these CSVs. See [NMR Data](nmr-data.md) for the import preview
and identity rules.

## Optional: Review structure mappings

This exercise requires the main PCS Analyzer, rather than standalone Workbench.

1. In a fresh PCS document, use **File → Load XYZ…** to load `structure.xyz`.
   Choose **Nd** as centre and leave averaging disabled.
2. Open Workbench from that document and follow one route above. Alternatively,
   save standalone work as `.nmrsession` and open it in a fresh integrated
   Workbench. Check **PCS:**; switching main document tabs does not rebind it.
3. In **PCS Preparation**, choose each H1–H4 assignment and **Single atom**
   target scope. Explicitly select targets and use **Update selected**:

   | Assignment | Structure Ref | Element |
   | --- | ---: | --- |
   | H1 | 6 | H |
   | H2 | 7 | H |
   | H3 | 8 | H |
   | H4 | 9 | H |

4. In **NMR Data**, select the corresponding row and **More → Show in Chemical
   Map**. Inspect Ref IDs and hydrogens; navigation does not transfer PCS.
5. If practising **Review & Apply…**, explicitly declare Direct from Δshift as
   a teaching assumption, review targets/conflicts and inspect the destination.
   You can cancel the review and finish the mapping exercise without transfer.

The XYZ is a nine-atom toy Nd/O/H cluster, not an optimized molecule or the
physical source of these shifts. Ref 1 is Nd, Refs 2–5 are O and Refs 6–9
are H. XYZ comments do not establish H1–H4 assignments. TMS is a separate
reference species and has no target here. Do not use this example to assess
tensor recovery or PCS fitting accuracy.

## Save and check your work

Choose **VT Series → Manage VT Series… → Files → Save NMR session…**; no VT
series is required. Reopen the session in a fresh Workbench and check the
committed data and assignments. **NMR Data → Export → All observations CSV…**
exports observations/provenance, not the unassigned TMS peak or a full session.
See [Sessions and Data Exchange](sessions.md).

The committed files are ready to use. Optional regeneration with NumPy and
nmrglue installed is `python examples/nmr_workflow_demo/generate_demo.py --overwrite`.
This explicitly replaces the example FID/CSV/XYZ files. Existing saved sessions
retain their embedded data; use a fresh import after regeneration.

For measured-spectrum processing see [Raw Data and Processing](raw-processing.md).
For the scientific derivation and reviewed handoff see
[Shift Analysis and PCS Preparation](non-pcs-analysis.md).
Continue with the separate [VT tutorial](tutorial-vt.md) to practise batch
processing, tracking and observed-shift temperature fitting on five FIDs.
