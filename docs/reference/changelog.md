# Changelog

The application version recorded by this repository is **PCS Analyzer 1.4.0**.
PCS Workbench has its own displayed module version, recorded as **0.1.0**.

This page summarizes the repository-maintained version history. It is not a
complete commit-by-commit history: early releases have brief summaries, and
not every version named here has a corresponding Git tag.

For published packages and release assets, use the
[PCS Analyzer releases page](https://github.com/boseokhong/PCS-analyzer/releases).

## Unreleased

These entries describe current source development, not a newly published binary.

### Interface and help

- Initialize the pending tensor/orientation input baseline after all startup
  controls and defaults exist, so the first Load XYZ is not blocked as an
  unfinished edit. Real unconfirmed input still blocks unsafe replacement.
- Stabilize contextual Ribbon transitions by reserving measured label space,
  avoiding identical label/style assignments, and retaining already-active groups
  without remapping them. Preserve context activation, command refresh and tab-wheel
  navigation; recalculate label space after theme/font/DPI refresh.
- Refine the existing menu/Ribbon presentation while retaining its original
  surfaces and contextual-tab colours: use matching mirrored blue 16-pixel
  Undo/Redo icons, quiet idle controls, hover/press emphasis and solid focus.
  Render disabled history icons as smooth grey images without the toolkit's
  additional stipple pattern, while retaining disabled interaction.
  Add faint inactive-tab hover underlines and wider command-group spacing with
  short, subtle dividers. Preserve command handlers, shortcuts and history scope.
- Add installed module launchers and ribbon-only Arrange modules to the Tools
  Modules group, with category icons and application-persisted ordering.
  Limit direct module button width with font-aware ellipsis and full-name tooltips.
- Simplify File by removing New/Close Document entries; retain document-tab actions,
  Ctrl+N/W and the directly exposed Workspace/export commands. Load XYZ stays first.
- Add shared, editable document-owned residual colouring cutoffs for the main
  and Rhombicity tables, restore them from projects, and compact the Rhombicity
  controls into a responsive row. Changing cutoffs does not rerun calculations.
  Color off now shows quiet, non-editable values; Color on reveals aligned
  cutoff inputs without resizing. Group the rhombic Enable with its input and
  separate tensor, rotation and residual controls with responsive thin dividers.
- Scope PCS Motion Explorer v1.0 source/local-installed scrolling to each module
  scroll region and remove its individual bindings on destruction. Preserve host
  wheel handlers after opening/closing the module.

- Add Help Documentation (F1) and a grouped Keyboard Shortcuts reference.
  Show menu accelerators and connect Ctrl+N/O/S/Shift+S/W plus Ctrl+Tab and
  Ctrl+Shift+Tab to existing document commands. Retain save paths/unsaved-close
  confirmation, local input Undo/Redo and auxiliary-window isolation.

### Analysis and fitting

- Refresh the prediction-model display during document activation/project
  loading so the Control Panel shows its user-facing label rather than an
  internal policy value.
- Record fitted geometry Preview and Clear Preview in document Undo/Redo.
  Redraw Preview immediately; capture Mode C's fitted strength inputs in the
  same action. Clear Preview retains current tensor settings and fit reports.
  Preserve full numeric precision when projecting Mode C Δχ inputs.
- Record Axial/Full tensor prediction-model selection in document Undo/Redo;
  restore the dropdown and refresh PCS/NMR predictions through existing model
  events. Keep stored tensor parameters, rhombic enablement and fit reports.
- Make Control Panel Reset one Undo/Redo action, including its cleared standard
  Fit History, coupled shifts, geometry and range settings. Startup defaults do
  not create history; untracked fit-record changes block unsafe restoration.
- Confirm Δχ_ax from Enter without opening Plot & display; connect keypad Enter
  to the same command. An active Mode C preview retains a confirmed axial edit
  instead of overwriting the entry on redraw, preserving its geometry and the
  immutable fit report. Preview entry projection keeps full numeric precision.

- Add one-action Undo/Redo for main-window XYZ/ORCA structure loading, restoring
  coordinates, Refs, assigned centre, coupled PCS data, temporary geometry and
  structure-specific settings. Failed or canceled loads preserve the document;
  Redo uses captured data without rereading the file. New structures clear old
  Ref-based shift mappings while retaining tensor parameters and recorded results.

- Fix history-popup positioning and Windows focus restoration so Control Panel
  entries remain editable after dropdown use. Refresh XYZ entries with sliders
  and preserve numeric precision through Undo/Redo and legacy redraws. Reject
  non-finite PCS ranges and non-positive intervals before committing parameters.

- Record confirmed tensor/XYZ/ZYZ orientation edits and coalesce each pointer
  drag into one document Undo/Redo entry. Add explicit raw-Ref metal-centre
  assignment through the table context menu, clearing the geometry preview
  reversibly without changing raw structure coordinates or recorded results.

- Add document-scoped session Undo/Redo for standard Mode B/C tensor Apply and
  Conformer Apply/Revert. Compact icon/dropdown controls beside Help support
  multi-step restoration and preserve recorded results. Redo does not refit;
  project files do not store the undo stack.
- Extend session Undo/Redo to δ_Exp table edits, shift-import Apply/Clear and
  NMR Workbench PCS transfers, preserving uncertainty/provenance and legacy
  import buffers. Import previews reject an active-document change before Apply.

- Persist the latest Conformer Search candidates, reports, embedded inputs and
  selection state. Restore for review/export without rerunning or applying a
  pending candidate; changed host inputs block Apply. XYZ exports use UTF-8.
- Persist active Mode C geometry previews, including XYZ orientation and refined
  centre, separately from the assigned metal centre. Incomplete legacy previews
  are disabled without fabricating missing values; recorded fit runs remain.
- Restored standard Mode details on project load and document switching.
- Converted main XYZ orientation to equivalent ZYZ on Conformer Search sync;
  rerun earlier searches that used nonzero orientations. Fixed angular-drift
  reporting for zero-length metal-centred vectors.
- Connected the PCS Field Viewer's Bond tolerance to displayed bond detection.
- Refined the G_ax regression and Rhombicity Check presentation, with adaptive
  plot labels, clearer reports and preparation guidance.
- Added collapsible standard Fit settings, recorded fit conditions, donor-list
  space and adjustable result/plot boundaries.
- Expanded Multi-Conformer into explicit structure mapping, fixed/fitted
  populations, tensor/population fitting and conditional bootstrap/CV reliability.
- Added fixed-frame Joint PCS+RDC fitting and proton Γ₂ PRE fitting with explicit
  inputs, weighting, diagnostics and recorded results.
- Expanded Multi-Lanthanide with independent tensors, Reilley separation,
  series/nucleus/relative-geometry diagnostics, a coefficient table and
  **Normalization** choices. New inputs default to 300 K; saved temperatures
  are retained and Sm/Eu remain fixed at their 300 K values.
- Added advanced-workflow project sections, reproducible demos and tutorials.
  Results remain separate from standard Fit History and are not automatically applied.

### Examples and reproducibility

- Added known-tensor standard fitting/rhombicity, one-bond Conformer Search,
  observed-shift-to-PCS fitting and CF₃/tBu/visualization teaching examples.
- Added an examples index, public worked tutorials and a read-only whole-example
  checker. Synthetic teaching inputs remain distinct from supplied measurements.
- Standardized generator output/overwrite options, fixed direct-script demo
  imports and clean-checkout capture profile setup, and expanded the portable
  resource list. These source changes do not imply a newly built binary.

### Interface consistency

- Unified widget presentation and named Light/Dark palette variants, removed the
  independent Accent selector and improved layouts with larger UI fonts.
- Refined document/Ribbon tab styling, right-side overflow controls, Ribbon-tab
  wheel navigation and panel boundaries; reduced repeated UI image rendering.

### 3D Structure Viewer

- Replaced the default molecular viewer with PyVista, with linked atom selection
  and a compact Display/Figure sidebar. The legacy Matplotlib viewer remains
  available from the viewer's View menu.
- Added atom/bond appearance, background and lighting options, metal-centre and
  corner axes, camera projections and standard views. Line bonds follow zoom.
- Added PCS range, palette and legend controls, including missing-value colors.
- Added Figure preview and size/DPI preparation, PNG output, explicitly
  image-based PDF/SVG, and experimental viewport exports. Visualization settings
  can be saved and loaded separately from projects.
- Fixed viewer shutdown timeouts, unreadable startup/sidebar controls and faint
  averaged markers. Empty viewers now show loading guidance; loading the first
  visible structure establishes the correct bond thickness.

### Documentation

- Aligned common metrics, residual units, current limitations and Ribbon workflow
  paths with the implemented analysis tools. Refreshed standard/advanced fitting
  captures and excluded contributor instructions from the public site/search.

- Updated the molecular viewer manual and export/persistence references around
  the default viewer. Simplified legacy guidance and consolidated development
  details into the repository's development records.

## 1.4.0

This is a **BIG** one. The interface, NMR workflows, and analysis tools have all received substantial updates.

The interface now uses a Ribbon toolbar to support future expansion. Existing
buttons and functions from the Analysis panel are now available in the Ribbon
menus.
The new layout may take a little getting used to. If you prefer the familiar interface, 
v1.3.4 remains available with the previous core features.

- Added multiple project tabs and a more flexible Control Panel.
- Added Chemical Map for inspecting and editing molecular diagrams
  and linking NMR assignments to atoms.
- Expanded `NMR Shift Viewer` into `NMR Workbench`, with NMR data import,
  spectrum processing, peak assignments, and PCS preparation.
- Added variable-temperature NMR workflows, including batch processing,
  stacked spectra, peak tracking, and temperature-dependent shift analysis.
- Improved PCS calculations and fitting, with uncertainty weighting
  and Fit History for reviewing and comparing results.
- Expanded Module Manager with online browsing, installation, updates,
  and clearer status information.
- Added workspace and NMR session saving.
- Expanded the documentation with updated screenshots, worked examples,
  and step-by-step tutorials.

## 1.3.4

- Corrected the PCS Workbench isotropic-susceptibility calculation.
- Reorganized PCS Workbench controls and improved interaction with its PyVista viewer.
- Added Workbench camera presets, projection modes, view save/restore, and improved PNG export.
- Reorganized the PyVista 3D structure controls and added bond appearance, hydrogen visibility, camera, view-preservation, and PNG-export controls.
- Corrected symmetry-averaged PCS calculations to average geometrical factors rather than evaluate them only at averaged coordinates.
- Added rotational ensemble averaging for detected planar ring systems, including configurable sampling, averaged-coordinate visualization, and optional reference overlays.

## 1.3.3

- Added the main menu bar, Settings window, font options, and revised main control panel.
- Added the Modules menu and the local Module Manager.
- Added `.pcsp` project save/load and drag-and-drop loading for `.pcsp`, `.xyz`, `.out`, and `.log` files.
- Added Position and Angular constraint modes, false/skipped-bond handling, and rigid mode to Conformer Search.
- Added lasso selection to **Gᵢ vs PCS** and improved the 2D polar contour display.

## 1.3.2

- Added PCS Workbench for ORCA tensor and spin-density input, point-dipole and FFT/PDE PCS calculations, atom-level comparison, slices, isosurfaces, and tensor-spheroid visualization.
- Added Workbench PNG, CSV, compressed NPZ, and temperature-sweep GIF exports.

## 1.3.1

- Added PyVista-based 3D viewing for molecular structures and PCS fields.
- Corrected the coordinate reference used by the Rhombicity analysis table.
- Added residual-based PCS-table text coloring.
- Added export of the current visible structure as XYZ.

## 1.3.0

- Moved the 2D PCS polar plot into a dedicated viewer.
- Improved **Gᵢ vs PCS** interaction and linked table/plot selection.
- Updated the 3D structure and projection viewers.
- Added Mode C full-tensor/metal-position fitting and the Advanced Fitting tab.
- Added the integrated Conformer Search workflow.

## Earlier maintained history

The repository records the following concise summaries. They are retained
without attempting to infer additional detail from later code.

| Version | Verified repository summary |
| --- | --- |
| 1.2.0 | Added the NMR spectrum viewer, automatic methyl/CF₃ symmetry averaging, ORCA `.out`/`.log` structure support, and other fixes/refactoring. |
| 1.1.1 | Bug fix release. |
| 1.1.0 | Added diagnostics and rhombicity utilities and updated fitting functions. |
| 1.0.2 | Corrected CSV export encoding. |
| 1.0.1 | Added experimental δ_Exp table import/export. |
| 1.0.0 | Refactored the application and added PCS fitting. |
| 0.7 | Bug fixes. |
| 0.6 | Added the Half/Quarter PCS plot view toggle. |
| 0.5 | Reorganized the GUI and added a Mollweide projection. |
| 0.4 | Added a 3D molecular scatter plot. |
| 0.3.1 | Added table highlighting from atom selection in the PCS plot. |
| 0.3 | Added XYZ import, coordinate rotation, the atom table, and geometrical-parameter analysis. |
| 0.2 | Added tensor, PCS-range, interval, and molar-susceptibility controls. |
| 0.1 | Initial release with XLSX export and Cartesian-coordinate output. |

Documentation-only changes are not listed as separate application releases.
