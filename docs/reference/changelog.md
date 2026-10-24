# Changelog

The application version recorded by this repository is **PCS Analyzer 1.4.0**.
The separate module versions are **PCS Workbench 0.1.1** and **NMR Workbench 0.1.0**.

For published packages and release assets, use the
[PCS Analyzer releases page](https://github.com/boseokhong/PCS-analyzer/releases).

## PCS Workbench 0.1.1 — unreleased

- Added optional padding convergence checks and clearer recorded result conditions.
- Added ORCA tensor transfer to the active Analyzer document, with review,
  Undo/Redo and saved source information.
- Expanded volume and surface export formats, including CUBE and NumPy, through
  one **Export 3D data…** dialog with optional advanced settings.
- Made before-padding normalization the default while retaining after-padding
  and compatibility with older presets.
- Corrected padded-grid coordinates and FFT handling. Recalculating older PDE
  results can change numerical values; the point-dipole model is unchanged.
- Improved input/result consistency, background work and dialog layouts.

See [PCS Workbench](../pcs-workbench/overview.md) for the updated workflow.
These notes describe the current source; published assets are listed separately.

## 1.4.0

### Added

- Ribbon toolbar, multiple document tabs and a reorganized Control Panel.
- Document Undo/Redo with a history dropdown for structure loading, PCS data,
  tensor and orientation edits, and supported fit/conformer result applications.
- Shared, adjustable residual-color thresholds for the main and Rhombicity
  tables, saved with each project.
- Multi-Conformer fitting with structure mapping, fixed or fitted populations,
  and bootstrap/cross-validation reliability diagnostics.
- Joint PCS+RDC fitting and proton Γ₂ PRE fitting.
- Multi-Lanthanide analysis with independent tensor fits, Reilley separation,
  coefficient tables and diagnostic plots.
- NMR Workbench spectrum processing, peak assignments and PCS preparation,
  plus variable-temperature batch processing, peak tracking and shift analysis.
- Chemical Map for editing molecular diagrams and linking NMR assignments.
- Workspace and NMR session saving, and project restoration of Mode C geometry
  previews and Conformer Search candidates.
- Module Manager online browsing, installation and updates; module launchers
  and customizable module ordering in the Tools Ribbon.
- Help-menu documentation access (F1) and a keyboard-shortcut reference.

### Improved

- More consistent light/dark themes, additional color palettes and support for
  larger UI fonts. Theme palettes replace the separate Accent setting.
- Faster document switching and smoother Ribbon transitions, with improved tab
  navigation, spacing and overflow controls.
- Clearer G_ax and Rhombicity diagnostics, collapsible Fit settings, resizable
  result/plot areas and recorded fit conditions.
- Uncertainty-weighted fitting and Fit History for reviewing results.
- The default 3D Structure Viewer now uses PyVista, with linked atom selection,
  appearance and lighting controls, camera presets and PCS legends. Added figure
  previews, size/DPI controls and image export; PDF/SVG exports are image-based.
  The legacy Matplotlib viewer remains available.
- Updated tutorials and screenshots, with additional fitting, Conformer Search,
  NMR and visualization examples.

### Fixed

- Incorrect tensor orientation when importing main-window settings into
  Conformer Search. Earlier searches using nonzero orientations should be rerun.
- Incorrect angular-drift summaries for fixed atoms at the metal centre
  in Conformer Search.
- Bond tolerance having no effect in the PCS Field Viewer.

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
