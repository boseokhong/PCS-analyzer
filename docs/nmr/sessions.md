# NMR Sessions and Data Exchange

Save NMR work separately from a PCS project. A `.pcsp` retains PCS analysis
inputs and handed-off observations; it does not replace a recoverable NMR session.

## Save and reopen committed work

Open **VT Series → Manage VT Series…** (also available from the **VT Series**
toolbar button), then choose **Files → Save NMR session…**. You can save a
single-temperature experiment or prepared shift data without creating a series.

A `.nmrsession` preserves session-owned datasets, conditions, identities,
embedded raw data when available, committed spectra, processing history,
peaks and observables. It also retains series/tracks, reviewed associations,
saved preparations/handoff evidence, retained analysis evidence and processing
presets where present. Pending processing previews, render caches and transient
UI selections are not saved. Apply a reviewed preview before saving if you
intend to preserve its processing result.

Use **Files → Open NMR session…** in a fresh Workbench to resume. Check recovery
diagnostics before continuing. Missing, moved or changed vendor sources are
reported; embedded committed data remains available. **Files → Rebind loaded
source…** lets you locate a vendor folder and check its source fingerprint.
Rebinding a path does not silently reconfirm stale tracking links.

Keep original vendor directories for independent reproducibility. Reimporting
the same FID is not equivalent to reopening its session: the new import has
new dataset and peak IDs.

Read-only PCS-project projections are omitted from NMR session ownership and
reported when applicable. Save the corresponding `.pcsp` separately, or use a
reviewed detached copy for independent NMR data. Opening a session in the
integrated Workbench does not restore or choose its PCS-document binding;
check **PCS:** and rebind explicitly before structural operations or handoff.

## Choose the right file

| File | Purpose | Limit |
| --- | --- | --- |
| **`.nmrsession`** | Resume committed NMR work with stable identities. | Does not preserve a live UI layout or pending previews. |
| **`.nmrseries`** | Export/import series membership and track definitions against loaded datasets. | Does not embed spectra or replace a session; links can need review. |
| **Observed-shift CSV** | Exchange prepared/spectrum-linked observables with conditions and provenance via NMR Data. | Does not recreate FIDs, peaks or tracks. Linked values need their matching session or explicit detachment. |
| **Assignment-review CSV** | Review assignment/uncertainty edits for existing peak-linked observables. | Requires matching identities; reopening FIDs does not recreate them. |
| **Analysis CSV** | Report extracted coordinates, inclusion, predictions and residuals. | Not a recoverable session or an observed-shift import template. |
| **Analysis results JSON** | Retain/import historical fit configuration, source snapshots, results and diagnostics. | Import does not recalculate or change live fits. Use **Keep in session** to retain evidence before saving. |
| **Plot PNG/PDF/SVG** | Share a figure. | Does not preserve scientific input identities or processing state. |

Series definition commands are **Files → Export series definitions…** and
**Import series definitions…** in the series manager. Assignment CSV and saved
results JSON use **Assignments / Analysis** (File menu: **Assignments / Saved
Analysis…**). Analysis tables/figures are exported from the VT analysis window.

For a practical save/reopen exercise, see the [FID/CSV tutorial](tutorial-fid-csv.md).
For recovery of a multi-temperature series and its tracks, see the [VT tutorial](tutorial-vt.md).
For PCS-project/workspace storage, see [Projects and Persistence](../projects.md).
