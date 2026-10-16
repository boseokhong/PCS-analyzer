# Quick Start

This guide follows a basic first-run workflow: load a structure, assign the PCS centre, attach experimental PCS values to atom reference IDs, inspect the **G_ax vs PCS** regression, and save and reopen a PCS Analyzer project.

It uses the axial model for orientation and avoids repeating the equations in [PCS Theory](pcs/theory.md). Verify susceptibility definitions and units in [Tensor Conventions](pcs/tensor-conventions.md) before interpreting fitted values.

## 1. Launch PCS Analyzer

Start the portable executable or run the source checkout from its repository root:

```bash
python main.py
```

The main window contains the Ribbon, document tabs, molecular table, analysis
area, and sectioned Control Panel described in [Interface Overview](interface.md).
The selected document is the active analysis; its inputs and results appear
in the shared workspace.

On a fresh configuration, the **2D PCS Plot** opens at startup. This behavior is configurable under **Settings → Open settings... → General → Open 2D PCS plot on startup**. The 3D structure viewer can likewise open after structure loading when **Auto-open 3D structure after loading XYZ** is enabled.

## 2. Load a structure

To explore a prepared example, open
`examples/documentation_demo/Nd_iPr2BA3_demo/Nd_iPr2BA3_axial.pcsp` with
**File → Open PCS Project...**. It contains the Nd(*i*Pr₂BA)₃ structure,
experimental `δExp`, averaging settings, and an axial tensor. The
[shared workspace screenshot](interface.md#main-interface) shows its
all-points diagnostic; no historical fitting subset is restored.
For a new analysis from raw coordinates, continue below.

Choose **Ribbon → Home → Load XYZ** or **File → Load XYZ...** and select
one of the supported structure inputs:

- `.xyz`
- ORCA `.out`
- ORCA `.log`

Despite the menu label, all three formats are accepted. A structure or `.pcsp` project can also be opened by drag and drop when `tkinterdnd2` is available.

Loading a raw structure replaces the structure in the active document. To
keep another analysis open, save it and click **+** in the document tab strip
before loading the next structure.

See [Loading Structures](structures/loading.md) for supported coordinate layouts and ORCA parsing behavior.

## 3. Assign the PCS centre

After loading the structure, the **Input center atom** dialog asks:

```text
Enter the center atom (element) :
```

Enter the element label exactly as it appears in the loaded structure, for example `Nd` or `U`. PCS Analyzer uses the first exact match and records that atom's reference ID as the metal/PCS centre. Its coordinates define the origin for the displayed PCS analysis.

!!! warning "Structures with repeated element labels"

    If more than one atom has the entered element label, the first matching atom in file order is used. The loading dialog does not choose among multiple matches. To assign a different raw atom afterward, right-click its main-table row and choose **Set Ref … as metal center (clear preview)**. This retains atom numbering and supports document Undo/Redo; see [Changing the assigned centre](structures/loading.md#changing-the-assigned-centre).

## 4. Inspect and select atoms

After a centre is assigned, the main table displays **Ref**, **Atom**, **X**, **Y**, **Z**, **G_ax**, **δ_PCS**, and **δ_Exp**.

- Reference IDs are assigned in structure-file order, starting from 1.
- Use **Control Panel → Element visibility → Elements** to include the nucleus types needed for the analysis.
- Select a table row to highlight the same reference ID in the G_ax-versus-PCS plot and in supported open viewers.
- Click a point or atom in a linked plot/viewer to select the corresponding table row.

Element filtering changes the rows immediately shown and analyzed; it does not renumber their reference IDs.

## 5. Set a basic axial PCS model

In **Control Panel → Tensor**, select **Prediction model → Axial** and enter
the axial susceptibility anisotropy in **Δχ_ax (×10⁻³² m³)**. Press
**Enter** to confirm it, or use **Update** in **Plot & display**. For example, `-4.0` represents

\[
\Delta\chi_{\mathrm{ax}}=-4.0\times10^{-32}\ \mathrm{m^3}
\]

in the convention used by PCS Analyzer.

In **Plot & display**, **PCS range** and **Interval** are in ppm;
**Half / Quarter (0–90°)** changes the angular extent of the polar plot.
In **Orientation → Structure**, X- and Y-axis controls rotate the structure
relative to the tensor frame. See [Coordinate System](structures/coordinate-system.md) and [PCS Contour Plots](pcs/contour-plots.md) for the corresponding conventions.

!!! warning "Tensor convention"

    Do not copy susceptibility values from another program or publication without checking its units and axial/rhombic definitions against [Tensor Conventions](pcs/tensor-conventions.md).

## 6. Add experimental PCS data

In the main workflow, **δ_Exp** is the experimental PCS value in ppm assigned to a structure reference ID. It is the dependent value used by **G_ax vs PCS** and by the main fitting tools.

Choose one of these input methods:

### Enter individual values

Double-click the **δ_Exp** cell for an atom, enter a number, and confirm. Leave the entry blank to clear that value.

### Use the table template

For the most reliable reference mapping:

1. Open **Ribbon → Data → Experimental PCS** and click **Export Template**.
2. Save the template as CSV or XLSX.
3. Fill its **δ_Exp** column without changing the **Ref** values.
4. Click **Import PCS** on the same Ribbon page and select the completed file.
5. Review the import preview, especially **mapped_id**, **parsed_value**, and **action**.
6. Leave only the intended rows selected and click **Apply**.

The importer recognizes reference-based columns such as `Ref`/`ref_id` and experimental-value names such as `δ_Exp`/`delta_exp`. Atom-label matching is also available, but reference IDs are safer when several atoms share the same label.

### Paste values

Choose **Ribbon → Data → Experimental PCS → Paste** and paste data such as:

```text
Ref, δ_Exp
2, 1.24
5, -0.38
```

Click **Preview**, review the mapping, and then **Apply**. The paste window accepts comma-, tab-, semicolon-, or whitespace-separated data.

On the **Data** Ribbon, use **Undo** to restore the state before the most
recent bulk import, or **Clear** to remove the active document's experimental
PCS assignments. Data actions remain available while using other analysis tabs.

## 7. Run the basic G_ax-versus-PCS regression

Open the **G_ax vs PCS** analysis tab; its contextual Ribbon page appears. Once at least two visible atoms have assigned `δ_Exp` values, the plot and result panel update automatically; there is no separate run button for the all-points regression.

1. On the contextual Ribbon, leave **Force origin** cleared for a slope-plus-intercept regression, or enable it when a through-origin model is scientifically justified.
2. Inspect the fitted line and the result panel.
3. Optionally use **Lasso** or **Box** to select a subset, then press **Fit** to regress only that subset.
4. Press **Clear Selection** to discard the subset and return to all assigned points.

The result panel reports the point count, fit mode and scope, slope, standard errors and 95% confidence intervals where available, intercept, \(R^2\), residual standard deviation, RMSE, and the converted axial susceptibility estimate.

This regression is a first diagnostic, not a substitute for checking the coordinate frame, PCS assumptions, residuals, and tensor convention. The **Fitting** tab provides nonlinear orientation/tensor workflows, but those require additional selections and are outside this first-run sequence.

## 8. Inspect atoms and results

Select a row in the table or a point in **G_ax vs PCS**. The result panel adds the selected point's reference ID, atom label, `G_ax`, `δ_Exp`, calculated `δ_PCS`, and residual when available.

Useful checks before saving include:

- the intended metal is at the PCS origin,
- the expected element types remain enabled,
- imported values appear on the intended reference IDs,
- the regression contains the expected number of points,
- the sign and magnitude of the converted `Δχ_ax` are plausible for the convention in use.

Use **View → 2D PCS Plot** or **Ribbon → Data → 2D PCS** for a
geometrical view of the axial contours. **Quick tools** can also hold a
shortcut to this viewer; its pinned buttons are customizable.

## 9. Save the project

Choose **Ribbon → Home → Save Project** or **File → Save PCS Project**
to save the active document as `.pcsp`. An untitled document prompts for a
path. Use **File → Save PCS Project As...** for a new path or **Save All**
to save all modified documents. Confirm that the tab's unsaved-change
asterisk clears before closing it.

The project format preserves the embedded working structure and principal
PCS-related inputs needed to continue the session. It does not serialize every
open window or selection. Complete standard fit results/history and structured
experimental observations can be restored when saved in the project. See
[Projects and Persistence](projects.md) for the exact restored categories,
recalculated values, and current round-trip limitations. Keep the original
source structure and experimental input files as provenance alongside the
project.

## 10. Reopen the project

Choose **Ribbon → Home → Open Project** or **File → Open PCS Project...**
and select the saved `.pcsp`. Opening a new path adds a document tab; an
already open project is activated. Project files can also be opened by drag
and drop when that support is available.

In the active project tab, verify:

1. the structure and centre,
2. the element visibility and table reference IDs,
3. `δ_Exp` assignments,
4. susceptibility and rotation controls,
5. the G_ax-versus-PCS plot and reported point count.

This verification is especially important when opening a project produced by
another application version. The current writer uses `.pcsp` schema version
`1.1`; a different schema version is loaded on a best-effort basis with a
console warning rather than a formal migration.

To reopen a set of document tabs, save their `.pcsp` projects first, then use
**File → Save Workspace As...**. The `.pcsw` workspace contains project
references rather than embedded analysis data.

## Common first-run problems

### The structure does not load or contains no atoms

- Use `.xyz`, `.out`, or `.log` with **File → Load XYZ...**.
- For XYZ, confirm that coordinate rows contain an element label and three numeric coordinates. If an atom-count header is present, retain its following comment line, even when the comment is blank.
- For ORCA output, confirm that the file contains a `CARTESIAN COORDINATES (ANGSTROEM)` or `CARTESIAN COORDINATES (BOHR)` block. The last recognized block is used.
- Unsupported extensions are rejected. A malformed supported file may yield no parsed atoms; correct the source file before continuing.

### The centre atom is not found or is wrong

- The element match is exact and case-sensitive: use `Nd`, not `ND` or `nd`, when the structure label is `Nd`.
- If the element occurs more than once, the first occurrence is selected.
- Canceling the centre dialog leaves the working document unchanged. Start Load XYZ again to load the new structure.
- For an already loaded structure, right-click the intended raw atom row and use **Set Ref … as metal center (clear preview)** instead of reloading.

### Experimental values map to the wrong atoms

- Reference IDs are based on structure-file order and are not interchangeable between differently ordered structures.
- Export a fresh template with **Data → Export Template** from the currently loaded structure and preserve its **Ref** column.
- Review every unexpected `mapped_id` or `skip` action in the import preview before applying it.
- Atom-only matching consumes repeated element labels in current table order, so use reference IDs for unambiguous mapping.

### The regression is empty or has too few points

- Confirm that at least two numeric `δ_Exp` values appear in the visible table.
- Enable the relevant element types under **Control Panel → Element visibility → Elements**.
- Check the import preview for skipped or cleared rows.
- If a Lasso/Box subset is active, press **Clear Selection** to return to all points.

### A fitting run reports invalid or empty input

In **Fitting**, use contextual Ribbon **Refresh** after changing the working
atom set, then select the required **PCS observations** and donor atoms
(for **Mode A · Ligand-donor**). Press **Run Fit** on the contextual Ribbon.
Choose enough independent observations to constrain the selected model;
being able to start a fit does not establish that the data are sufficient.
See [Fitting Overview](fitting/overview.md) for input requirements,
initialization, bounds, and result application.

## Workflow summary

```text
Launch PCS Analyzer
        ↓
Load XYZ or ORCA structure
        ↓
Assign the centre element
        ↓
Inspect reference IDs and visible elements
        ↓
Set Δχ_ax and update
        ↓
Enter, import, or paste δ_Exp
        ↓
Inspect G_ax vs PCS regression
        ↓
Save .pcsp project
        ↓
Reopen and verify the project
```

[Interface Overview](interface.md){ .md-button }
[Loading Structures](structures/loading.md){ .md-button .md-button--primary }
[Projects and Persistence](projects.md){ .md-button }
