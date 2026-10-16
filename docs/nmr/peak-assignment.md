# Peaks and Assignments

Process and reference the spectrum before interpreting peak positions.
In **Processing → Peak Picking**, use **Start Picking** to select peaks on
the spectrum and **Stop** to leave picking mode, or use **Auto Pick**.
Advanced controls specify **Min height (% max)**, **Prominence (% range)**,
and **Min distance (points)**. Automatic picking detects positive peaks of
the real processed spectrum; inspect negative, missed, overlapped, and
baseline-distorted signals yourself.

## Inspect and assign

Open **Inspector → NMR Data**, choose **Active experiment** and **Show = All**.
The unified table shows assigned observations and unassigned picked peaks.
Inspect **Shift / ppm**, **Source**, **Status**, **Intensity**, and **Picking**
(scroll horizontally for additional columns). Select a spectrum row to highlight
its peak. Prepared independent shifts have no measured peak marker.
Double-click **Assignment**, or press **Enter**, to edit. **Tab** advances to
the next assignment; **Shift+Tab** goes back. Use **More → Remove selected…**
or **Delete** to request removal of the selected source peak and its linked
observation; review the confirmation. **Processing → Peak Picking → Clear Auto Peaks** retains manual picks; re-running
automatic picking replaces previous automatic picks.

Use the same chemically meaningful assignment in the paramagnetic and
diamagnetic datasets. Peak number and intensity order are not reliable
cross-sample identifiers. Assignment text alone does not establish a
structural atom mapping or chemical equivalence.

A peak assignment creates or updates an observable when required metadata is
valid; review **Status** and the **Use** column. Observables carry chemical shift and
assignment for downstream pairing. An annotation can remain a draft if missing
metadata prevents observable creation; inspect warnings instead of assuming
every label has become usable analysis data.

Click **Use** to include/exclude an assigned observation from shift comparison.
Leave reference-standard peaks such as TMS unassigned or explicitly excluded
when they are not sample observations. Clearing a spectrum assignment removes
its observable linkage without deleting the peak. Reprocessing or replacing
peaks can invalidate tracking evidence; do not assume links are repaired by
nearest ppm. See [NMR Data](nmr-data.md) for independent values and CSV input.

## Link to the structure

Use **NMR Data → More** or the row context menu to **Show in Chemical Map**,
**Preview NMR site**, **Preview matching ligand sites**, and explicitly
**Use reviewed target in PCS Preparation**.
A selection preview does not accept a target automatically. Check Ref IDs and
all group members, especially methyl sites and equivalent ligand instances.

Once assignments are reviewed, set dataset roles and pair them in
[Shift Analysis and PCS Preparation](non-pcs-analysis.md). Duplicate labels
within a dataset produce ambiguous pairs, not independent replicate
measurements or an automatic average. Molecular assignments and interpretation
of overlapping signals still require experimental judgment.
