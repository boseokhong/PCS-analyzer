# 2D Polar PCS Plot

The **2D Polar PCS Plot** places the current structure on an axial PCS contour
map. Open it with **View → 2D PCS Plot** or **Open viewers → 2D PCS Plot**.
The same rendering also appears in the main workspace.

This view remains an axial diagnostic even when **Prediction model** is
**Full tensor (Axial + Rhombic)**: its contours use the current
Δχ_ax and the axial \(r,\theta\) expression. A single polar contour cannot
represent the full \(\phi\)-dependent rhombic field. See
[PCS Contour Plots](../pcs/contour-plots.md) for the scientific interpretation
and [Projection Viewer](projection-viewer.md) or
[3D PCS Plot](3d-pcs-isosurface.md) for tensor-aware angular views.

## Contours and angular domain

**PCS plot range (ppm)** and **PCS plot interval (ppm)** choose the contour
values. Positive contours are drawn on a red scale, negative contours on a
blue scale, and the legend is titled **PCS legend (ppm)**. Dashed lines mark
the axial magic angles. The radial scale is displayed in Å and the current
plot is limited to 10 Å.

The **Half/Quarter plot toggle** selects the actual angular domain:

- **Half-plane:** \(0^\circ\leq\theta\leq180^\circ\).
- **Quarter-plane:** \(0^\circ\leq\theta\leq90^\circ\).

In Quarter view, atom positions with \(\theta>90^\circ\) are folded to
\(180^\circ-\theta\). The axial contours are mirrored consistently. This is a
compact display of axial symmetry; it does not change coordinates, geometrical
factors, calculated PCS values, regressions, or fits.

## Atom markers and bonds

Markers use element CPK colors. **Atom radii** switches between fixed marker
sizes and sizes scaled from covalent radii; **Atom scale** multiplies the
display size. These controls affect marker appearance only.

**Bonds** draws guide lines between currently displayed atoms. The test uses
their three-dimensional distance and

\[
d_{ij}\leq(r_i^\mathrm{cov}+r_j^\mathrm{cov})(1+\text{tol.}).
\]

**width** changes line thickness and **tol.** changes this visualization-only
cutoff. These lines do not encode bond order or experimentally determined
connectivity, and the tolerance is unrelated to the fixed connectivity rules
used by functional-group or torsional detection. See
[Symmetry and Torsional Ensemble Averaging](../structures/symmetry-averaging.md#connectivity-detection).

## Averaging and reference overlays

The plot distinguishes active averaging representations:

- ordinary/reference positions are filled element-colored circles;
- torsional ensemble-average positions are hollow element-colored circles;
- CH₃ symmetry pseudo atoms are warm-orange `×` markers; and
- CF₃ symmetry pseudo atoms are teal `×` markers.

When **Keep original atoms** or **Keep reference structure** produces a
reference overlay, the plot shows it separately. The **Averaging** legend is
dynamic and lists only representations that are currently visible; it is
separate from the PCS contour legend.

An averaged marker is a representative coordinate. PCS Analyzer calculates
averaged PCS geometry from member/sample geometrical factors rather than from
that marker position. The precise distinction is documented in
[Symmetry and Torsional Ensemble Averaging](../structures/symmetry-averaging.md).

## Selection synchronization

Clicking a marker selects, focuses, and scrolls to its Ref row in the main
table. Selecting a table row redraws both the embedded and popup polar plots;
the selected atom is enlarged and, for circular markers, surrounded by a
yellow ring.

Only visible effective atoms have clickable markers. A folded Quarter-view
position still selects the atom's original Ref ID.

## Calculation controls versus display controls

| Control or input | Effect |
| --- | --- |
| Δχ_ax, structure, metal centre, rotations, and averaging model | Changes the scientific coordinates and/or PCS calculation used by the view |
| PCS plot range and interval | Changes which axial contour levels are drawn |
| Half/Quarter toggle | Changes only the displayed angular domain |
| Atom radii and Atom scale | Changes marker size only |
| Bonds, width, and tol. | Changes visual connectivity guides only |

## Export

The popup includes the standard Matplotlib navigation toolbar. In addition,
**File → Save plot data...** supports:

- PNG for the currently open 2D PCS plot;
- XLSX with **PCS Data** and **Atom Coordinates** sheets; or
- CSV as separate `_pcs.csv` and `_atoms.csv` files.

PNG export through this command requires the popup to be open. Exported
`Theta` values in the numerical plot-data files are radians. See
[File Formats](../reference/file-formats.md#pcs-plot-data) for exact outputs.
