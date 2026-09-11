# PCS Theory

PCS Analyzer is primarily designed for coordination complexes with approximately rotational molecular geometry, where the magnetic susceptibility anisotropy can often be described using an effective axial model.

In this default case, the pseudocontact shift is written as

$$
\delta_{\mathrm{PCS}}\;(\mathrm{ppm})
=
\frac{10^4}{12\pi}
\Delta\chi_{\mathrm{ax}}G_{\mathrm{ax}},
$$

with the axial geometrical factor

$$
G_{\mathrm{ax}}
=
\frac{3\cos^2\theta-1}{r^3}.
$$

Here,

- \(r\) is the distance from the paramagnetic centre to the observed nucleus,
- \(\theta\) is the polar angle relative to the principal \(z\)-axis,
- \(\Delta\chi_{\mathrm{ax}}\) is the axial magnetic susceptibility anisotropy.

This form provides the default geometrical description used throughout the basic PCS Analyzer workflow.

---

## Rhombic extension

For systems in which in-plane anisotropy cannot be neglected, PCS Analyzer can include a rhombic contribution.

The full axial–rhombic expression is

$$
\delta_{\mathrm{PCS}}\;(\mathrm{ppm})
=
\frac{10^4}{12\pi}
\left(
\Delta\chi_{\mathrm{ax}}G_{\mathrm{ax}}
+
\Delta\chi_{\mathrm{rh}}G_{\mathrm{rh}}
\right),
$$

where

$$
G_{\mathrm{rh}}
=
\frac{3}{2}
\frac{\sin^2\theta\cos(2\phi)}{r^3}.
$$

The azimuthal angle \(\phi\) is measured in the tensor \(xy\)-plane.

When

$$
\Delta\chi_{\mathrm{rh}}=0,
$$

the full expression reduces to the axial model.

!!! note

    The axial model is the default representation in PCS Analyzer because the software is primarily intended for approximately rotational coordination geometries.

    Rhombicity can be introduced when the system shows significant deviation from effective axial symmetry or when a full susceptibility-tensor treatment is required.

---

## Geometrical interpretation

The PCS depends strongly on both distance and orientation.

The \(r^{-3}\) dependence causes nuclei close to the paramagnetic centre to experience substantially larger PCS values.

For the axial term, the angular dependence is controlled by

$$
3\cos^2\theta-1.
$$

The axial PCS changes sign at the magic angle,

$$
\theta_{\mathrm{m}}
=
\arccos\left(\frac{1}{\sqrt{3}}\right)
\approx 54.7^\circ.
$$

The rhombic contribution additionally depends on \(\phi\) through \(\cos(2\phi)\), introducing an in-plane angular dependence that is absent from the axial approximation.

---

## Coordinate frame

PCS calculations are performed in the magnetic susceptibility tensor frame.

In the default axial treatment:

- the \(z\)-axis defines the principal axial direction,
- only \(r\) and \(\theta\) are required,
- rotation about the \(z\)-axis does not change the calculated axial PCS.

When rhombicity is included:

- the \(x\) and \(y\) directions become physically distinct,
- the azimuthal angle \(\phi\) must also be considered,
- rotation about the \(z\)-axis can change the calculated PCS.

See [Tensor Conventions](tensor-conventions.md) for the susceptibility definitions used by PCS Analyzer.

---

## Units used by PCS Analyzer

PCS Analyzer accepts

$$
\Delta\chi_{\mathrm{ax}},\Delta\chi_{\mathrm{rh}}
$$

in units of

$$
10^{-32}\,\mathrm{m^3}
$$

per molecule, while structural coordinates are given in Å.

The factor \(10^4\) in the program-level PCS equations accounts for these input units and returns the calculated shift in ppm.

For detailed tensor definitions and reconstruction rules, see [Tensor Conventions](tensor-conventions.md).
