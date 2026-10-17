# Quick Start

Start a PCS analysis with your own molecular structure and experimental PCS
values. This guide takes you from loading the files to an initial diagnostic
and a saved project. For installation, see [Installation](installation.md).

## Before you start

Prepare:

- a molecular structure in XYZ or supported ORCA `.out`/`.log` format;
- the identity of the paramagnetic centre; and
- experimental PCS values in ppm, with assignments to atoms in that structure.

**δ_Exp means experimental PCS**, not an absolute observed chemical shift.
If you have paramagnetic and diamagnetic shifts instead, start with
[Observed and Diamagnetic Shifts](experimental-data/delta-obs-dia.md).
For raw FIDs, use the [NMR Workbench](nmr/spectrum-viewer.md) first.

Need practice data? The [examples guide](examples.md) lists starting files and
expected results. A prepared `.pcsp` already contains its structure and saved
settings: open it with **File → Open PCS Project...**, then follow its tutorial
instead of repeating the raw-data preparation below.

## 1. Open the application and load your structure

Start the portable executable, or run `python main.py` from the source root.
Choose **Home → Load XYZ** on the Ribbon or **File → Load XYZ...**. The chooser
accepts XYZ and supported ORCA `.out`/`.log` files despite the command name.
For ORCA, the last recognized Cartesian-coordinate block is loaded.

Loading replaces the structure in the active document. To keep an existing
analysis, save it and click **+** in the document tab strip before loading
another structure.

See [Loading Structures](structures/loading.md) if your file is not recognized.

## 2. Assign and check the centre

In **Input center atom**, enter the element label as written in the file,
for example `Nd`. The first exact match becomes the PCS centre.

Check its **Ref** ID in the main table. If the structure contains several atoms
with that label, right-click the intended raw atom row and choose
**Set Ref … as metal center (clear preview)**.

Ref IDs follow structure-file order, starting from 1. Keep that ordering when
preparing assignments; the same Ref number in another structure need not refer
to the same atom. Use **Control Panel → Element visibility → Elements** to show
the nuclei needed for the analysis. Filtering does not renumber them.

## 3. Import your experimental PCS values

Use the table template to avoid ambiguous atom-label matching:

1. On **Data → Experimental PCS**, choose **Export Template**.
2. Save CSV or XLSX and fill the **δ_Exp** column in ppm. Preserve **Ref** values.
3. Choose **Import PCS** and select the completed file.
4. Review the mapped Ref IDs, values and proposed actions in the preview.
5. Select the intended rows and choose **Apply**.

For a few values, double-click individual **δ_Exp** cells. **Paste** is also
available; see [Import and Export](experimental-data/import-export.md) for its
formats and mapping rules.

**Checkpoint:** the intended visible rows contain your PCS values on the correct
Ref IDs. Blank and zero are different: blank means unassigned; zero is data.

For exchange-averaged or grouped observations, review
[structured targets](experimental-data/delta-exp.md#structured-observations-and-grouped-targets)
before entering the same resonance on several atom rows. Group members are not
additional independent measurements.

## 4. Check the model and coordinate frame

For an initial axial diagnostic, select **Control Panel → Tensor → Prediction
model → Axial**. If you already have Δχ_ax, enter it in units of
**10⁻³² m³ per molecule** and press **Enter**. Check external values against
[Tensor Conventions](pcs/tensor-conventions.md).

The **Orientation** controls define the structure relative to the tensor frame.
An axial diagnostic is meaningful only for the orientation being examined; the
default orientation is not an estimate of your sample's tensor direction.
If orientation is unknown, use [orientation fitting](fitting/orientation-fitting.md)
with enough independent observations.

Check [symmetry and torsional averaging](structures/symmetry-averaging.md) if
your assignments represent exchanging groups. **PCS range**, **Interval** and
the Half/Quarter polar view adjust the display; they do not fit the tensor.

## 5. Inspect the initial diagnostic

Select **G_ax vs PCS**. With at least two suitable visible assignments, the
all-point regression updates automatically. It estimates an axial susceptibility
from the slope for the current geometry/frame; it does not apply that estimate
to the main tensor.

- Check the point count against the observations you intended to analyze.
- Inspect the line, intercept and residuals. Use **Force origin** only when
  justified by your model.
- Select a table row or plotted point to check its Ref ID and values.
- If inspecting a scientifically justified subset, use **Lasso** or **Box**,
  then **Fit**. **Clear Selection** returns to all assigned points.

A poor correlation can reflect orientation, assignments, non-PCS contributions,
structural differences or averaging. A high R² alone does not validate those
assumptions. See [Residual Analysis](diagnostics/residual-analysis.md).

## 6. Choose further analysis if needed

| Your question | Continue with |
| --- | --- |
| What tensor orientation best matches my PCS? | [Standard fitting](fitting/overview.md), especially Mode B |
| Do axial residuals suggest a missing rhombic contribution? | [Rhombicity Diagnostics](diagnostics/rhombicity.md) |
| Do I need to refine the metal position as well? | [Full Tensor Fitting](fitting/full-tensor-fitting.md) |
| How do I compare conformers or combine other observables? | [Advanced workflows](fitting/overview.md#current-advanced-fitting-workflows) |

For standard fitting, refresh the inputs, select the required PCS observations
and choose the mode before **Run Fit**. Mode A also needs donor atoms. Review
the report and residuals before using **Apply Tensor** or **Preview**; running
a fit alone does not apply its tensor. The
[Fitting Overview](fitting/overview.md#running-and-applying-a-fit) explains the
different application/preview behavior of each mode.

<a id="9-save-the-project"></a>

## 7. Save and reopen your project

Choose **Home → Save Project** or **File → Save PCS Project** (**Ctrl+S**).
Save to `.pcsp`; keep your original structure and experimental files alongside
it as source data. Use **Save PCS Project As...** when you want a separate copy.

Reopen with **File → Open PCS Project...** and check the centre, assignments,
tensor/frame, averaging and diagnostic point count. A project restores analysis
data and supported results, not every open window or selection. See
[Projects and Persistence](projects.md) for save/reopen scope. NMR work also
requires its own `.nmrsession`.

To retain a set of document tabs, save their projects first and then save a
workspace. A `.pcsw` contains references to those projects, not embedded data.

If you get stuck, see [Troubleshooting](troubleshooting.md). For the terms used
here, see the [Glossary](reference/glossary.md).
