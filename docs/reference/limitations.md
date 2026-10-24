# Current Limitations

This page describes limits relevant to the current source workflows. A successful
fit, saved report or visualization does not establish a unique scientific model.
See each workflow's manual for inputs, equations and interpretation.

## Fitting and confidence

- Standard Modes A/B accept nonempty matched selections without enforcing three
  independent points. Select enough independent observations for the parameters;
  the ability to run does not establish identifiability. Mode C rejects fewer
  than three matched observations but can still be underdetermined.
- Standard nonlinear fits, Joint PCS+RDC and PRE do not estimate parameter
  confidence intervals. Multi-Conformer reliability intervals and Reilley
  standard errors are conditional on their documented assumptions.
- Mode A provides geometry preview only. Mode B **Apply Tensor** copies axial
  anisotropy/orientation and retains the main rhombic value/enablement. Mode C
  tensor application retains the assigned metal centre; its refined centre is
  available in temporary geometry preview.
- Multi-Lanthanide separates PCS and total paramagnetic-shift inputs. Reilley
  assumes a compatible Ln series; its Sm/Eu coefficients remain fixed at 300 K
  even when another temperature is selected. Derived PCS values share fitted
  coefficient uncertainty and are not independent measured observations.

Details: [Fitting Overview](../fitting/overview.md), [Quality Metrics](../fitting/quality-metrics.md),
[Multi-Conformer reliability](../fitting/multi-conformer-tutorial.md), and
[Multi-Lanthanide/Reilley](../fitting/multi-lanthanide-tutorial.md).

## Saving and reproducibility

Document Undo/Redo covers standard Mode B/C tensor Apply, Conformer Apply/Revert,
main-table δ_Exp editing, shift-import Apply/Clear, the existing Undo import action,
NMR Workbench PCS transfers, confirmed tensor/XYZ/ZYZ orientation edits,
rhombic enablement, explicit raw-Ref metal-centre assignment and main-window
XYZ/ORCA structure replacement with its coupled Ref-based inputs.
Control Panel Reset also records its cleared working data and standard Fit History.
Axial/Full tensor prediction-model selection is recorded without changing the
stored tensor parameters or rhombic enablement.
Fitting Preview Geometry and Clear Preview are recorded, including Mode C's
fitted Δχ input values imported by its immediate redraw.
It is separate from recorded fit/search results and is limited
to the latest 100 session edits; project files do not persist its stack. Input
imports outside these workflows and project/workspace opening
are not recorded as standalone actions. Reset can restore the policy
and temporary geometry it cleared. NMR source sessions and transfer receipts are not undone.

Projects preserve complete standard fit runs and supported advanced results,
not an exact restoration of all windows and interaction state. Current files
retain active Mode C previews. Older files missing their centre/orientation
cannot restore that preview; use the retained fit run to recreate it.
Current projects retain the latest Conformer Search candidate set and its inputs;
older scalar summaries cannot reconstruct it. Search results remain viewable and
exportable when inputs change, but Apply requires matching recorded host and
search scientific inputs. Only completed search sets are persisted; replacement
input files without a completed search are not separate saved sessions.
Row selections, zoom/hover and import undo are transient. Residual-colour cutoffs
and Color state are restored for the main and Rhombicity tables; changes to these
display settings are outside scientific Undo/Redo. Export-DPI values recorded in
a project are not reapplied by the loader.

A workspace contains paths to saved projects rather than unsaved scientific
changes. NMR sessions must be saved separately. Older supported sections have
defaults and compatibility handling, but cross-version loading is not a guarantee
of identical behavior. Keep source data and exported run records.

See [Projects and Persistence](../projects.md), [File Formats](file-formats.md),
and [NMR Sessions](../nmr/sessions.md).

## Structure search and distributed PCS

Earlier source versions passed main-window XYZ angle numbers directly to the
Conformer Search's ZYZ calculator. Current **Sync Main** preserves the rotation
matrix through conversion; rerun nonzero-orientation searches made with older
versions. See [Conformer Search](../conformer-search/overview.md).

PCS Workbench treats `.3d` origin/spacing as Å and uses absolute density values
for calculation; original signed samples are retained separately for export.
It chooses a metal automatically, falling back to the first atom when needed.
Check these assumptions against the source calculation. Grid extent, resolution
and FFT padding can require substantial memory. See
[Workbench interpretation limits](../pcs-workbench/overview.md#interpretation-limits).

The [padding convergence check](../pcs-workbench/calculation.md#padding-convergence-check)
measures numerical padding sensitivity, not experimental accuracy. [Tensor transfer](../pcs-workbench/tensor-transfer.md)
requires the same ordered structure and assigned centre; automatic atom matching
and different conformations are not supported. Exported surfaces are individual
levels, not complete viewer scenes, and external-reader support varies.

## Viewers and data import

Viewer availability depends on installed/bundled PyVista, VTK and Qt dependencies.
Matplotlib export formats depend on the active backend. Legacy `.xls` reading
requires a compatible pandas Excel engine; use XLSX or CSV where appropriate.
See [Installation](../installation.md) and [Import and Export](../experimental-data/import-export.md).

## Modules and plugins

Plugins execute Python in the host process and are not sandboxed. Catalog
SHA-256 checks establish integrity against metadata, not safety or scientific
correctness. New package installation validates paths, manifests and compatibility
bounds; replacement uses backups and attempts rollback. Filesystem failures or
locked files can still require restart and leave recovery paths.

Restart after update/replacement to execute new code. Disabling/removing a plugin
does not undo its side effects or close its windows. Legacy manifest loading is
more permissive and can report compatibility warnings. PluginApp remains an
unversioned helper rather than a stable isolated extension API.

See [Module Manager](../plugins-docs/overview.md),
[Distribution](../plugins-docs/distribution.md), and [Troubleshooting](../plugins-docs/troubleshooting.md).

## Validation boundaries

Source checks and worked examples do not certify every platform, packaged build,
external module or experimental dataset. Platform/runtime dependencies, large
volumetric calculations and Windows file locks need environment-specific checks.
Current PCS Workbench UI captures use illustrative data. Earlier field, slice,
comparison and spheroid images remain scientific illustrations of their recorded
workflow; they are not newly recalculated 0.1.1 validation results.

Historical development findings are kept in the repository's development records;
this page is a current user guide, not a complete bug tracker or release guarantee.
