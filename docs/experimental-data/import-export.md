# Import and Export

The experimental-data controls accept CSV and Excel tables and provide a
preview before modifying the current assignments. δ_Exp is managed on
**Ribbon → Data → Experimental PCS**. For observed/reference shifts, the
use assigned vendor spectra or prepared observed values in
[NMR Workbench](../nmr/spectrum-viewer.md). **NMR Data → Import** accepts
observed-shift CSV/TSV or pasted tables without a spectrum. Existing project
δ_obs/δ_dia layers remain readable as read-only projections.

## Supported input paths

| Data | File import | Clipboard-style input | Where |
| --- | --- | --- | --- |
| δ_Exp | CSV, XLSX, XLS | Yes | Data Ribbon → Experimental PCS |
| NMR observed shifts | Vendor 1D data; observed-shift CSV/TSV/text | Paste or manual values; peak assignment editing | NMR Workbench → NMR Data; see [NMR Data](../nmr/nmr-data.md) |
| PCS-project δ_obs / δ_dia | CSV, XLSX, XLS | Yes | NMR Data → Import → PCS project observed/diamagnetic shifts; read-only Workbench projections |

The table-import rules below describe the current δ_Exp importer and the
legacy δ_obs/δ_dia table importers. Legacy import dialogs are not the current
Workbench's acquisition/processing workflow. Recognized legacy headings are
retained here for interpreting existing tables and projects. The general
**Observed shifts CSV / TSV…** importer uses separate validation and identity
rules described in [NMR Data](../nmr/nmr-data.md). In particular, do not apply
the blank-cell clearing or structural Ref-matching rules below to that importer.

CSV reading first tries UTF-8 with a byte-order mark and then Windows cp1252.
Excel reading uses the engines available to pandas. The standard dependency
set includes support for XLSX. Although the file chooser also offers legacy
XLS files, reading an XLS file can fail if no compatible legacy Excel engine
is installed; the application reports the read error.

## Recognized columns

Column matching ignores capitalization and surrounding whitespace. Use a
Ref ID column whenever possible.

| Role | Recognized headings |
| --- | --- |
| Ref ID | `Ref`, `ID`, `ref_id`, `rid` |
| Atom label | `Atom`, `Element` |
| δ_Exp | `δ_exp`, `delta_exp`, `exp`, `delta values` |
| δ_obs | `δ_obs`, `delta_obs`, `obs`, `ppm`, `shift` |
| δ_dia | `δ_dia`, `delta_dia`, `dia`, `diamagnetic`, `delta_diamagnetic` |

If neither a Ref nor an Atom heading is recognized but the table has at least
two columns, the importer treats the first two columns as Ref ID and value.
For predictable results, include explicit headings rather than relying on
this fallback.

## How rows are matched

### Ref ID matching

With a recognized Ref column, the importer converts each value to an integer
and assigns the data to that Ref ID. A numeric ID is accepted by the preview
even if it is absent from the current structure; such an assignment remains
stored but is ignored by structure-dependent fitting and diagnostics until a
matching atom exists.

### Atom-label matching

If no Ref column is present but an Atom column is available, labels are
matched exactly, including case, against the currently visible table. When a
label occurs more than once, imported values are consumed in visible table
order. This can be ambiguous for repeated labels such as multiple `H` atoms,
so Ref IDs are strongly preferred.

If both Ref and Atom columns are present, Ref ID matching takes precedence.

## Preview and selective application

After reading a file or pasted block, PCS Analyzer opens an import preview.
Each row shows its source identifier and value, mapped Ref ID, parsed value,
and one of three actions:

- **set** assigns a numeric value;
- **clear** removes the existing value for a matched Ref ID; and
- **skip** makes no change.

Blank cells, `NaN`, and non-numeric cells parse as **clear**, not as an
automatic skip. Review these rows carefully before applying them. Unmapped
rows cannot be selected.

Click the first preview column to toggle a row, or use **All on**, **All off**,
and **Invert**. **Apply** changes only the selected rows. **Undo last** restores
the data layer to its state before the most recent applied preview in the
current session.

## Clipboard input

The **Paste** dialogs provide an editable text area rather than reading the
system clipboard without review. Paste a block with a header, then choose
**Preview**. Comma, tab, semicolon, and whitespace separators are detected
from the first line.

Recommended two-column examples are:

```text
Ref,δ_Exp
12,0.084
18,-0.137
```

```text
Ref	δ_obs
12	7.913
18	8.204
```

## Exporting experimental-data tables

**Data → Export Template** writes the currently visible main-table rows as
`Ref`, `Atom`, and `δ_Exp`. Existing δ_Exp values are retained in the output.
CSV is written as UTF-8 with a byte-order mark; XLSX is also available.

**File → Save table...** exports the current main table. An XLSX export also
includes a **Rhombicity** worksheet when that table contains rows. A CSV
export creates a `_table.csv` file and, when available, a separate
`_rhombicity.csv` file.

For the wider file-format inventory, see
[File Formats](../reference/file-formats.md). The detailed matching and
preview behavior on this page remains the authoritative experimental-data
reference.

## Common mismatch cases

- The imported Ref IDs belong to another structure or to an earlier atom
  order.
- A Ref value contains non-integer text and is skipped.
- An atom label differs in case or spelling from the visible table.
- Repeated atom labels are assigned in an unintended order.
- Filtering hides rows needed for atom-label matching or template export.
- A blank or invalid numeric cell is selected in the preview and therefore
  clears an existing value.
- δ_obs and δ_dia use different Ref IDs, so δ_para is not calculated.

When the fit cannot find expected points, compare the imported Ref IDs with
the current main table, then use **⟳ Refresh** in the Fitting tab.
