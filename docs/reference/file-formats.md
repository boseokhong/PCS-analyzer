# File Formats

PCS Analyzer uses different formats for structures, experimental values,
projects, numerical exports, figures, and viewer presets. Support is
workflow-specific: an extension accepted by one tool is not necessarily
accepted by another.

## Format summary

| Format | Read | Write | Principal use |
| --- | --- | --- | --- |
| `.xyz` | Yes | Yes | Main structures, conformers, and optimized structures |
| ORCA `.out`, `.log` | Yes | No | Main geometry input; PCS Workbench geometry and susceptibility-tensor input |
| `.csv` | Yes | Yes | Experimental values, advanced fitting inputs/templates, tables, and numerical results |
| `.xlsx` | Yes | Yes | Experimental shifts, templates, PCS plot data, and table exports |
| legacy `.xls` | Offered for experimental-shift input | No | Older spreadsheet input when a compatible pandas engine is installed |
| `.pcsp` | Yes | Yes | PCS Analyzer project persistence, including structured observations and standard fit history |
| `.pcsw` | Yes | Yes | Ordered references to saved `.pcsp` projects; no embedded project data |
| `.nmrsession` | NMR Workbench | NMR Workbench | Recoverable committed NMR spectra, observations and series/tracking state |
| `.nmrseries` | NMR Workbench | NMR Workbench | Series/track definitions referencing loaded datasets; no embedded spectra |
| `.png` | No | Yes | Raster figure and viewer export |
| `.pdf`, `.svg` | No | Selected views only | Vector/document figure export |
| `.json` | Yes | Yes | Fit-run import/export; molecular visualization settings and other workflow-specific viewer presets; plugin manifests |
| `.3d` | PCS Workbench | No | ORCA simple-grid spin-density input |
| `.npz` | No | PCS Workbench | Compressed distributed-PCS field data |
| `.gif` | No | PCS Workbench | Temperature-dependent tensor-spheroid animation |
| `.txt` | No | Selected workflows | Regression companion summary and conformer-search report |
| plugin `.py`, folder, `.zip` | Module Manager | No | Local external-plugin installation |

## Structure inputs

### Main structure loader

**File → Load XYZ...** accepts `.xyz`, `.out`, and `.log`. The extension
selects the parser.

- XYZ may use the standard atom-count/comment header or supported headerless
  coordinate rows.
- ORCA input uses the last recognized Cartesian-coordinate block.
- ORCA coordinates in Bohr are converted to Å.

See [Structure Files and Parsing](../structures/loading.md) for accepted XYZ
row layouts and ORCA block detection.

The Multi-Conformer fitting loader accepts the same three extensions through
the common structure loader. The standalone Conformer Search file chooser is
stricter: it accepts conventional XYZ with an atom-count line, comment line,
and the declared number of coordinate rows.

### PCS Workbench inputs

PCS Workbench accepts ORCA `.out`/`.log` and ORCA simple-grid spin-density
`.3d` files. In this context, the ORCA loader also looks for susceptibility
tensors associated with `TEMPERATURE/K` and `Tensor in molecular frame`
records. This tensor parsing is additional to the main structure loader's
geometry-only use.

The `.3d` reader expects a title, grid dimensions, origin, grid spacing, and
exactly the declared number of scalar values. Its parser rejects non-positive
dimensions or spacing and a mismatched data count.

## PCS Analyzer projects

`.pcsp` embeds the structure, principal analysis inputs and complete standard
Fit History. It is intended for reopening analyses, and is not interchangeable
with XYZ or a spreadsheet.

See [Projects and Persistence](../projects.md) for save/load behavior,
compatibility handling, recalculated state, fit-history restoration and the
Mode C preview persistence.

The existing `[visibility]` fields `residual_thr_ok`, `residual_thr_warn`,
and `residual_color_enabled` retain the shared main/Rhombicity Table display
classification. Cutoffs must be finite with 0 ≤ green < orange; invalid or
missing pairs use 0.10/0.30 ppm. Missing Color defaults to off.
Ribbon module order is an application preference and is not embedded here.

Current files also embed the latest Conformer Search candidate set in a versioned
JSON record within `[conformer_search]`. It includes the original search geometry
(Å), PCS observations (ppm), fixed/bond selections, recorded conditions, reports
and candidate coordinates. Current manual graph edits are separate from the
immutable run graph. Undefined R²/Q metrics use JSON null and remain
undefined on restoration. Search inputs do not depend on temporary file paths.
Older scalar-only summaries contain insufficient data to reconstruct candidates.

### Workspace references

`.pcsw` is a workspace containing paths to saved `.pcsp` projects, their
order and the active project reference. It does not store unsaved project
changes or embed the projects. Save the project documents separately and keep
referenced files available when moving or sharing a workspace. See
[Workspaces](../projects.md#workspaces-are-project-references).

## Experimental-data spreadsheets and pasted text

The main δ_Exp and PCS-project δ_obs/δ_dia importers accept CSV, XLSX,
and—when a compatible engine is available—legacy XLS. Clipboard dialogs
accept comma-, tab-, semicolon-, or whitespace-separated text.

The separate **NMR Data → Observed shifts CSV / TSV…** importer accepts UTF-8
CSV/TSV/text and mapped absolute observed shifts, with its own identity rules.
Its paste input supports comma, tab and semicolon delimiters. XLSX and the
structural Ref-matching rules below belong to the PCS-project importers, not
this general NMR input route. See [NMR Data](../nmr/nmr-data.md).

The preferred structure is:

```text
Ref,δ_Exp
12,0.084
18,-0.137
```

`Ref` is the recommended identifier and must contain integer Ref IDs matching
the current structure. An atom/element column can be used instead, but exact
label matching and repeated-label ordering make it less reliable. A third
`Atom` column is optional when Ref IDs are present.

The value heading depends on the data layer. Common forms include `δ_Exp` or
`delta_exp`, `δ_obs` or `delta_obs`, and `δ_dia` or `delta_dia`. The importer
also supports documented aliases and a preview with selective set, clear, and
skip actions.

See [Import and Export](../experimental-data/import-export.md) for the full
recognized-heading table, fallback behavior, preview semantics, encodings,
and mismatch troubleshooting. Those rules are the authoritative reference
for the main experimental-data importers.

## Specialized CSV inputs

These files belong to specific workflows and do not use the general
experimental-shift preview.

| Workflow | Required identifiers and values | Optional/defaulted fields |
| --- | --- | --- |
| Conformer Search PCS | First column: one-based `Ref`/ID; second value column, or third column when an element column is present | Header row may be present; comma, tab, semicolon, or whitespace dialect is detected |
| Joint PCS+RDC | `ref1`, `ref2`, `rdc_exp`, or unambiguous `atom1`, `atom2`, `rdc_exp` | Explicit raw Ref pairs recommended. `rdc_err` defaults to 0.5 Hz; `nuc1`/`nuc2` default to H/N. `d` is accepted for `rdc_exp`; `h`/`n` are not label-column aliases. |
| PRE | Integer `ref_id`, `ref` or `id`; `gamma2`/`g2`, or paired `r2_para,r2_dia` | Proton Γ₂ increments in s⁻¹. `gamma2_err`/`sigma` optional; paired rate errors combine by RSS. Single `r2_para`/`r2` are legacy increment aliases with a warning. Optional `nucleus` supports H/1H only. |
| Multi-Lanthanide | Wide `Nucleus,Ln...` matrix for total paramagnetic shifts, or single-dataset `ref_id`/`ref`/`id` with `pcs_exp` (Tensor fitting) or `delta_para` (series tools) | `Ln_err`/`error` uncertainties, `G_common`/`Ln_G_i` geometry, nucleus/Ref mapping. Legacy `delta_exp`/`d_exp`/`ppm` aliases use the selected input type. Missing cells remain missing; invalid numbers and duplicate labels are rejected. |

The Advanced Fitting tab can export UTF-8-with-BOM CSV templates for RDC and PRE.
The optional versioned `[joint_pcs_rdc]` project section retains RDC inputs,
selected PCS observation IDs, conditions and the last recorded joint result.
The main structure/PCS observations remain in their normal sections. Loading
checks recorded predictions against recorded inputs without refitting or
applying the tensor. Corrupt joint sections reject project loading; older files
without the section use empty joint inputs and defaults.

PRE rejects malformed/duplicate observations; it does not silently skip them.
Its optional versioned `[pre_fitting]` section retains observations, fit conditions
and a verified recorded result without rerunning. See the [PRE tutorial](../fitting/pre-tutorial.md).
Multi-Lanthanide adds an optional versioned `[multi_lanthanide]` JSON section for typed
datasets, conditions and verified results per analysis. See its
[tutorial](../fitting/multi-lanthanide-tutorial.md) for coefficient and transfer limits.
See [Fitting Overview](../fitting/overview.md#current-advanced-fitting-workflows)
for current workflow limitations.

## Main-window exports

### PCS plot data

**File → Save plot data...** behaves according to the chosen extension:

| Choice | Output |
| --- | --- |
| XLSX | One workbook with **PCS Data** and **Atom Coordinates** sheets |
| CSV | Two UTF-8-with-BOM files: `<base>_pcs.csv` and `<base>_atoms.csv` |
| PNG | The currently open 2D PCS polar plot at the configured export DPI |

The PCS-data output contains θ samples plus PCS level, radius, and Cartesian
plot coordinates for each contour. The atom-coordinate output contains the
visible atom labels, radius, θ, and geometrical factor. The exported `Theta`
values in these data files are radians. PNG export requires the 2D PCS Plot
window to be open. An absent or unrecognized extension falls back to XLSX.

### Main and rhombicity tables

**File → Save table...** exports the rows currently displayed in the main
table.

- XLSX creates a **Table** sheet and, when populated, a **Rhombicity** sheet.
- CSV creates `<base>_table.csv` and, when populated,
  `<base>_rhombicity.csv`.

The main table columns are `Ref`, `Atom`, `X`, `Y`, `Z`, `G_i`, `δ_PCS`, and
`δ_Exp`. The rhombicity export includes its displayed angular, `Gi_ax`,
`Gi_rh`, axial/rhombic PCS, and residual columns.

The main-table `δ_PCS` column follows the active Axial/Full tensor prediction
model; the rhombic term contributes only when enabled. Axial-plus-rhombic
predictions, when enabled, remain separate columns in the rhombicity export.

### Experimental template

**Export δ_Exp Template** writes visible main-table rows as `Ref`, `Atom`, and
`δ_Exp` in CSV or XLSX. It preserves existing δ_Exp cells so the file can be
edited and re-imported. See
[Experimental-data Import and Export](../experimental-data/import-export.md).

### Visible XYZ

**File → Save current XYZ...** writes a standard XYZ atom-count line, a PCS
Analyzer comment line, and coordinate rows. It exports only currently visible
elements from the effective structure. Coordinates include an applied fit
orientation/centre override when supported, are centred on the active metal,
and include the current X/Y/Z rotations. This is therefore a transformed
analysis view, not necessarily the untouched input coordinates.

## Plot and viewer exports

Export capabilities differ by view.

| View | Explicit file export |
| --- | --- |
| **G_ax vs PCS** | PNG, SVG, or PDF figure, plus `<base>_summary.txt` and `<base>_points.csv` companion files |
| Standard **Fitting** | PNG or PDF correlation/residual figure |
| **Projection** | PNG, PDF, or SVG figure |
| **3D Structure → Figure output** | PNG, or PDF/SVG containing an embedded image |
| **3D Structure → Viewport capture** | PNG; experimental PDF/SVG that may contain raster content |
| Retained **Matplotlib structure viewer** | PNG/PDF/SVG from its own camera |
| Main **3D PCS Plot** | PNG only; either live-view resolution or an off-screen DPI/width render |
| **Conformer Search** | PNG/PDF/SVG structure plot, multi-model XYZ for selected candidates, and TXT report |
| NMR Workbench | Spectrum figures through the Matplotlib toolbar; NMR Data observed-shift CSV; assignment CSV and analysis tables/JSON/figures through their respective windows. Save `.nmrsession` for recovery. |

Figure PDF/SVG from the molecular viewer do not contain editable vector atoms,
bonds or labels. See the [3D Structure Viewer](../visualization/3d-viewer.md#figure-and-export)
for output choices.

The Gᵢ companion CSV contains `Ref`, `Atom`, `G_i`, `δ_Exp`, `δ_PCS`, and
`Residual`. The standard Fitting export saves the plot only; the textual fit
report and numeric result are not written by that button. Use Fit History
JSON/CSV export to retain numerical results.

Viewer-specific details are documented under
[Visualization](../visualization/polar-plot.md) and in the
[NMR Workbench](../nmr/spectrum-viewer.md). See
[Raw Data and Processing](../nmr/raw-processing.md) for vendor dataset inputs.

## PCS Workbench outputs

PCS Workbench has its own export paths:

- PNG render of the distributed PCS field;
- compressed NPZ containing the PCS field, source term, origin, spacing,
  temperature, tensor arrays, and calculation options;
- UTF-8-with-BOM CSV comparing per-atom PDE and point-tensor PCS;
- tensor-spheroid PNG;
- temperature-sweep tensor-spheroid GIF; and
- oblique PCS slice plot as PNG, PDF, or SVG.

These are PCS Workbench outputs, not exports from every PCS Analyzer viewer.

## Fit-run exports

In **Fit History → More ▾**, **Export selected as JSON…** saves complete standard
runs for **Import Fit Run(s)…**. **Export selected as CSV…** writes
`<base>_summary.csv`; for one selected run it also writes
`<base>_observables.csv`. Multiple selected runs produce only the summary CSV.
CSV and figure files cannot be imported as complete fit runs. Importing a JSON
run does not restore the source project or apply its tensor.
See [Fit History](../fitting/fit-history.md).

## Reproducing an analysis

Keep the following together:

1. The **.pcsp project** containing the structure, centre, prediction policy,
   tensor/orientation, experimental targets/uncertainties and averaging settings.
2. Original structures and experimental source data, with assignments and
   processing/referencing notes; retain raw FIDs for NMR-derived shifts.
3. Complete **fit-run JSON** and, when useful, observation/residual CSVs.
   Label runs by subset, fitted/fixed parameters and model assumptions; record
   excluded observations and reasons. A plot or screenshot alone is insufficient.
4. Application version and analysis notes, including initialization, weighting,
   averaging ranges/sample counts, and any temporary geometry preview.
5. Optional **.pcsw** plus all referenced projects when reproducing a multi-document
   session. Keep their folder layout; the workspace does not embed data.

After reopening, check inputs and fit status, then verify live predictions.
Stored fit predictions describe the original run. In particular, a temporary
Mode C preview is retained separately from the assigned metal centre in current
files. Older files without its complete preview values require recreating it
from the recorded run. Retain that run and its reported coordinates.
A visible XYZ export may contain transformed/averaged geometry and filtered
atoms, so preserve the original structure separately.

## JSON viewer presets

The **3D Structure** viewer saves appearance, background, PCS colors, axes and
Figure options through **File → Visualization settings → Save… / Load…**.
These files use format `pcs-analyzer-visualization`, version `1`. They exclude
coordinates, calculated PCS/tensors, camera, selection and shared main-application
controls; keep the `.pcsp` project separately. See
[Save and reuse visualization settings](../visualization/3d-viewer.md#save-and-reuse-visualization-settings).

The main **3D PCS Plot** can save and load a JSON preset containing its level
styles, grid/surface/atom display controls, colors, camera preset/projection,
and PNG export options. PCS Workbench also saves and loads JSON parameter
presets for its own control panel.

Preset files store controls, not structures, experimental data, or calculated
fields. These workflows use separate presets; keep them with their corresponding
project or source data.

## Plugin files

At a high level, the Module Manager accepts a single Python file (`.py`), a
selected plugin folder, or a ZIP package (`.zip`). Folder and ZIP packages use
`manifest.json` and a Python entry file; a `.py` installation generates that
installed layout. JSON viewer presets and plugin manifests serve unrelated
purposes even though both use `.json`.

See [Distribution and Installation](../plugins-docs/distribution.md) for the
accepted layouts and current security limitations, and
[Plugin Manifest](../plugins-docs/manifest.md) for fields that are operational,
descriptive, or not enforced.
