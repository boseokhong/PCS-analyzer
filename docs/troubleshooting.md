# Troubleshooting

Start with the section matching your current task. When asking for help, include
the application version, file type, exact message and the steps leading to it.
Use a small shareable input when possible; remove personal paths or confidential
sample information from reports.

## My structure does not load

Use **File → Load XYZ...** for `.xyz` or supported ORCA `.out`/`.log` files.
XYZ rows need an element and three numeric coordinates. For conventional XYZ,
keep the comment line after the atom-count line even when it is blank.
ORCA input needs a recognized Cartesian-coordinate block; the last one is used.
See [Loading Structures](structures/loading.md) for other accepted row layouts.

## The wrong atom is the centre

The loading dialog matches the element label exactly and uses its first occurrence.
Check capitalization and the Ref ID. To choose another raw atom, right-click
its table row and use **Set Ref … as metal center (clear preview)**.
Canceling the loading dialog leaves the working document unchanged.

## Experimental values appear on the wrong atoms

Export a fresh **Data → Export Template** from the loaded structure and preserve
its Ref column. Ref IDs follow file order; they are not portable between
differently ordered structures. Review mapped IDs and skipped rows before
**Apply**. Repeated element labels make atom-label-only matching ambiguous.
See [Import and Export](experimental-data/import-export.md).

## The G_ax plot is empty or has the wrong point count

Check that numeric **δ_Exp** values are present and the relevant elements are
visible. The all-point regression needs at least two suitable points.
Use **Clear Selection** if an old Lasso/Box subset is active.

Grouped PCS observations need not appear as separate δ_Exp cells for each atom.
Inspect the Fitting **PCS observations** list and the
[structured-target explanation](experimental-data/delta-exp.md#structured-observations-and-grouped-targets).

## Rhombicity Check has no diagnostic plots

Load the structure, provide δ_Exp and establish the orientation, then choose
**Update Diagnostic**. It needs at least three valid points and uses a selection
from Fitting when present. See [Rhombicity Diagnostics](diagnostics/rhombicity.md).

## A fit rejects my inputs or the result looks implausible

In standard **Fitting**, choose **Refresh**, check **Mode details** and select
the required **PCS observations**. Mode A also requires donor atoms. There must
be enough independent observations to constrain the chosen parameters; the fact
that a run starts does not establish sufficiency.

Check the centre, assignments, orientation, tensor units, averaging and possible
non-PCS contributions. Inspect residuals and compare starting assumptions rather
than accepting a result from R² alone. See [Fitting Overview](fitting/overview.md)
and [Quality Metrics](fitting/quality-metrics.md).

## Apply is unavailable or the result is Stale

**Stale** means the recorded result no longer matches the relevant working inputs.
Review the changes and rerun with the intended data. **Unknown** means validity
could not be established. Do not substitute a historical report for a new fit.
Standard fitting mode and advanced workflow also determine what can be applied;
see [Fit History](fitting/fit-history.md) and the relevant workflow tutorial.

## The main tensor did not change after Run Fit

Running a fit records a result. Applying it or previewing its geometry is a
separate action, and the fields transferred depend on the mode. See
[Running and applying a fit](fitting/overview.md#running-and-applying-a-fit).

## A viewer does not open when running from source

Read the error and check the dependency table in [Installation](installation.md).
The Qt-backed structure viewer needs PySide6, qtpy and pyvistaqt; PyVista viewers
also need VTK. Install dependencies in the Python environment actually launching
PCS Analyzer. Optional viewer support and main calculations have different
requirements. Preserve the complete folder layout of a portable installation.

## Reopening a file did not restore my whole session

`.pcsp` saves the supported PCS analysis state; `.pcsw` references saved projects;
`.nmrsession` saves NMR work. A workspace alone contains neither embedded projects
nor unsaved changes. See [Projects and Persistence](projects.md) and
[NMR Sessions](nmr/sessions.md) before choosing what to save.

For module installation or loading errors, use
[Module Troubleshooting](plugins-docs/troubleshooting.md). Other known scientific
and workflow limits are listed in [Current Limitations](reference/limitations.md).
