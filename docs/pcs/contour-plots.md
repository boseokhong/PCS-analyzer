# PCS Contour Plots

PCS contour plots provide a geometrical representation of the pseudocontact-shift field around the paramagnetic centre.

PCS Analyzer uses these plots primarily to visualize the axial PCS model that forms the default description for approximately rotational coordination geometries.

---

## Axial PCS contours

For the axial model,

$$
\delta_{\mathrm{PCS}}
=
\frac{10^4}{12\pi}
\Delta\chi_{\mathrm{ax}}
\frac{3\cos^2\theta-1}{r^3}.
$$

For a selected PCS value, the corresponding radial coordinate can be written as

$$
r
=
\left[
\frac{
10^4\Delta\chi_{\mathrm{ax}}
\left(3\cos^2\theta-1\right)
}{
12\pi\delta_{\mathrm{PCS}}
}
\right]^{1/3}.
$$

The resulting contour therefore shows the spatial positions that produce the same axial PCS value for a given \(\Delta\chi_{\mathrm{ax}}\).

---

## Polar representation

The 2D polar PCS plot displays:

- \(\theta\) as the angular coordinate,
- \(r\) as the radial coordinate,
- PCS values as contour lines,
- selected nuclei at their corresponding \((\theta,r)\) positions.

Because the default axial model is rotationally symmetric around the \(z\)-axis, the azimuthal coordinate \(\phi\) is not required for this representation.

This makes the polar plot particularly useful for rapidly comparing molecular geometry with an axial PCS field.

---

## Sign of the PCS field

The angular factor

$$
3\cos^2\theta-1
$$

changes sign at approximately

$$
54.7^\circ.
$$

Consequently, the axial PCS field is divided into angular regions of opposite sign.

The sign of the observed PCS additionally depends on the sign of \(\Delta\chi_{\mathrm{ax}}\).

---

## Rhombic systems

When

$$
\Delta\chi_{\mathrm{rh}}\neq0,
$$

the PCS becomes dependent on the azimuthal angle \(\phi\):

$$
\delta_{\mathrm{PCS}}
=
\frac{10^4}{12\pi}
\left[
\Delta\chi_{\mathrm{ax}}
\frac{3\cos^2\theta-1}{r^3}
+
\Delta\chi_{\mathrm{rh}}
\frac{3}{2}
\frac{\sin^2\theta\cos(2\phi)}{r^3}
\right].
$$

A single \(r\)-versus-\(\theta\) polar contour can no longer represent the complete three-dimensional PCS field because the field is no longer rotationally invariant around \(z\).

For this reason, the standard 2D polar PCS plot should primarily be interpreted as an axial representation.

Rhombic contributions are better inspected using tensor-aware tables, fitting/diagnostic tools, projection views, or three-dimensional PCS visualization.

!!! note

    The 2D polar PCS plot is intentionally optimized for the axial model used as the default PCS Analyzer workflow.

    When rhombicity is included, use the additional rhombic and 3D visualization tools to inspect the full angular dependence.

---

## Plot range and interval

The displayed contours are controlled by the selected PCS range and contour interval.

A narrower interval provides more detailed visual resolution but may produce a crowded plot.

For qualitative interpretation, a contour interval that clearly separates positive and negative PCS regions is usually preferable.

---

## Half- and quarter-plane angular views

The **Half/Quarter plot toggle** switches between two polar angular domains:

- **Half-plane view:** \(0^\circ \le \theta \le 180^\circ\)
- **Quarter-plane view:** \(0^\circ \le \theta \le 90^\circ\)

In the quarter-plane view, positions with \(\theta > 90^\circ\) are folded to \(180^\circ-\theta\). This compact representation is useful for the axially symmetric contour because the axial angular factor is symmetric about \(90^\circ\). Use the half-plane view when the two sides of the tensor \(z\)-axis need to remain visually distinct.

---

## Interpreting atom positions

Overlaying nuclei on the PCS contour plot allows direct comparison between molecular geometry and the expected axial PCS field.

Nuclei close to the same contour are expected to have similar axial PCS contributions, provided that the same tensor parameters apply.

Deviations between experimental values and the axial contour expectation can motivate further analysis, including:

- tensor reorientation,
- structural refinement,
- rhombicity analysis,
- conformational averaging,
- evaluation of non-PCS contributions.

See [Axial and Rhombic PCS](axial-rhombic.md) for the full model.
For the implemented markers, selection, averaging overlays, and exports, see
[2D Polar PCS Plot](../visualization/polar-plot.md).
