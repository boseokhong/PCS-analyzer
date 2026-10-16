# Mathematical Conventions

This reference collects conventions that affect how project and exported data
should be interpreted. It links to the canonical scientific pages rather than
repeating their full derivations.

## Shift notation

| Mathematical notation | UI label | Meaning |
| --- | --- | --- |
| \(\delta_{\mathrm{obs}}\) | `δ_obs` | Observed chemical shift in the paramagnetic sample |
| \(\delta_{\mathrm{dia}}\) | `δ_dia` | Corresponding diamagnetic-reference shift |
| \(\delta_{\mathrm{para}}=\delta_{\mathrm{obs}}-\delta_{\mathrm{dia}}\) | `δ_para` | Total paramagnetic shift contribution derived from those two layers |
| \(\delta_{\mathrm{Exp}}\) | `δ_Exp` | User-assigned experimental PCS target used by the main regression and fitting workflows |
| \(\delta_{\mathrm{PCS,calc}}\) | `δ_PCS` | Calculated PCS under the active Axial or Full tensor prediction policy; Full tensor includes rhombicity only when enabled |
| \(\delta_{\mathrm{nonPCS}}=\delta_{\mathrm{para}}-\delta_{\mathrm{PCS,calc}}\) | `δ_nonPCS` | Residual-like NMR diagnostic; not a uniquely assigned physical contribution |

UI and exported column labels retain their implemented capitalization and
underscores. In equations, the same quantities are typeset as mathematical
subscripts. Paramagnetic shift and experimental PCS are not automatically
interchangeable; see [Experimental PCS Data](../experimental-data/delta-exp.md)
and [Observed and Diamagnetic Shifts](../experimental-data/delta-obs-dia.md).

## Atom identifiers

Main-workflow Ref IDs are integers assigned from structure order, starting at
1. Experimental values, project data, table exports, conformer-search CSV,
and many advanced inputs use these IDs as their join key.

The exact table and preferred file-column label is **Ref**. This documentation
uses **Ref ID** as the short prose form and **reference ID** when the expanded
term reads more clearly; both refer to the same identifier.

Filtering atoms does not renumber them. A Ref ID from a reordered or different
structure must not be assumed to identify the same nucleus. See
[Experimental-data Import and Export](../experimental-data/import-export.md#how-rows-are-matched).

## Coordinate frame and rotations

The main PCS workflow retains absolute input coordinates but evaluates PCS in
a metal-centred tensor frame. Rotations are active, extrinsic rotations about
fixed Cartesian axes in X → Y → Z order in the **Structure** controls.
**Tensor Euler (ZYZ)** expresses the same molecular-to-tensor rotation using
active intrinsic ZYZ angles. The same numeric triplet generally represents
different rotations in XYZ and ZYZ. UI and project angles are degrees;
projects retain the equivalent XYZ representation.

See [Coordinate Handling](../structures/coordinate-system.md) for matrices,
row-vector application, and the coordinate flow.

## Tensor and angular conventions

PCS Analyzer defines Δχ_ax, Δχ_rh, tensor principal components, θ, and φ as
documented in [Tensor Conventions](../pcs/tensor-conventions.md). The axial
term is invariant to z-axis rotation; a rhombic term depends on the in-plane
x/y reference through φ.

The implemented definitions are

\[
\Delta\chi_{\mathrm{ax}}
=\chi_{zz}-\frac{\chi_{xx}+\chi_{yy}}{2},
\qquad
\Delta\chi_{\mathrm{rh}}=\chi_{xx}-\chi_{yy}.
\]

The **Half/Quarter plot toggle** names display domains, not fractions of a
stored tensor or sampling interval. Half view displays
\(0^\circ\leq\theta\leq180^\circ\); Quarter view displays
\(0^\circ\leq\theta\leq90^\circ\) and folds atom positions from the other
hemisphere. See [PCS Contour Plots](../pcs/contour-plots.md#half-and-quarter-plane-angular-views).

Some exported angle columns use radians while UI/table columns use degrees.
The exact file-level distinction is listed in [Units](units.md).

## Residual signs

Residual sign is workflow-specific:

- standard and advanced nonlinear fits use predicted minus experimental;
- the main table colours calculated PCS minus δ_Exp;
- Rhombicity Check and the Rhombicity Table display experimental minus
  predicted; and
- NMR δ_nonPCS is δ_para minus calculated PCS.

See [Residual Analysis](../diagnostics/residual-analysis.md) before comparing
signed residual columns from different exports. RMSD and absolute-residual
rankings are unaffected by reversing the sign.

## Numeric text and missing values

Project files use a period as the decimal separator and UTF-8 text. Their
fixed-column structure/delta rows are parsed strictly. Spreadsheet import has
its own header matching, preview, and clear/skip rules; do not apply `.pcsp`
editing assumptions to CSV or Excel data.

See [Projects and Persistence](../projects.md) and
[File Formats](file-formats.md) for format-specific behavior.
