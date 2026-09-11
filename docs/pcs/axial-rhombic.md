# Axial and Rhombic PCS

PCS Analyzer uses an axial PCS description as the default model and provides an optional rhombic extension when deviations from effective rotational symmetry need to be considered.

This distinction is particularly useful for coordination complexes that are approximately rotational but not perfectly axially symmetric.

---

## Axial model

For the default axial treatment,

$$
\delta_{\mathrm{PCS}}
=
\frac{10^4}{12\pi}
\Delta\chi_{\mathrm{ax}}G_{\mathrm{ax}},
$$

where

$$
G_{\mathrm{ax}}
=
\frac{3\cos^2\theta-1}{r^3}.
$$

Only the distance \(r\) and polar angle \(\theta\) are required.

Because the axial tensor is rotationally symmetric around \(z\), the azimuthal angle \(\phi\) does not affect the calculated PCS.

This makes the axial model particularly convenient for approximately rotational molecular geometries and for initial tensor-orientation analysis.

---

## Rhombic contribution

When in-plane anisotropy is significant, the full PCS expression is used:

$$
\delta_{\mathrm{PCS}}
=
\frac{10^4}{12\pi}
\left(
\Delta\chi_{\mathrm{ax}}G_{\mathrm{ax}}
+
\Delta\chi_{\mathrm{rh}}G_{\mathrm{rh}}
\right),
$$

with

$$
G_{\mathrm{rh}}
=
\frac{3}{2}
\frac{\sin^2\theta\cos(2\phi)}{r^3}.
$$

The rhombic contribution therefore depends on both \(\theta\) and \(\phi\).

A non-zero \(\Delta\chi_{\mathrm{rh}}\) breaks the rotational equivalence of the \(x\) and \(y\) directions.

---

## When to consider rhombicity

The axial approximation is generally useful when:

- the coordination geometry is approximately rotational,
- the fitted or calculated tensor is close to axial,
- the purpose is an initial structure–PCS comparison,
- the available experimental data do not justify a more highly parameterized model.

Rhombicity may be worth considering when:

- significant systematic residuals remain after an axial fit,
- equivalent angular regions show different PCS behavior,
- computational magnetic-susceptibility calculations predict \(\chi_{xx}\neq\chi_{yy}\),
- the molecular geometry shows substantial distortion from rotational symmetry,
- a full susceptibility-tensor analysis is required.

!!! caution

    Improvement after introducing \(\Delta\chi_{\mathrm{rh}}\) should not automatically be interpreted as evidence for physical rhombicity.

    Structural mismatch, incorrect assignments, non-PCS contributions, conformational averaging, or over-parameterization can produce similar improvements.

---

## Axial and full-model comparison

PCS Analyzer can evaluate both the axial-only prediction and the axial–rhombic prediction for the same molecular geometry.

For a nucleus \(i\),

$$
\delta_i^{\mathrm{ax}}
=
\frac{10^4}{12\pi}
\Delta\chi_{\mathrm{ax}}G_{\mathrm{ax},i},
$$

whereas

$$
\delta_i^{\mathrm{ax+rh}}
=
\frac{10^4}{12\pi}
\left(
\Delta\chi_{\mathrm{ax}}G_{\mathrm{ax},i}
+
\Delta\chi_{\mathrm{rh}}G_{\mathrm{rh},i}
\right).
$$

The difference

$$
\delta_i^{\mathrm{rh}}
=
\delta_i^{\mathrm{ax+rh}}
-
\delta_i^{\mathrm{ax}}
$$

represents the contribution introduced by the rhombic term.

---

## Geometrical factors

PCS Analyzer evaluates the geometrical factors from coordinates already expressed in the tensor frame:

$$
G_{\mathrm{ax}}
=
\frac{3\cos^2\theta-1}{r^3},
$$

$$
G_{\mathrm{rh}}
=
\frac{3}{2}
\frac{\sin^2\theta\cos(2\phi)}{r^3}.
$$

The sign and magnitude of the rhombic contribution therefore depend on both the rhombic susceptibility parameter and the position of the nucleus within the \(xy\)-plane.

---

## Relation to diagnostics and fitting

The rhombic term can be incorporated in fitting when a full axial–rhombic model is required.

Residual-based diagnostics can also be used to examine whether deviations from an axial model correlate with the rhombic geometrical factor.

These approaches should be interpreted as model diagnostics rather than as proof that a rhombic tensor is uniquely required.

See:

- [PCS Theory](theory.md)
- [Tensor Conventions](tensor-conventions.md)
- [Residual Analysis](../diagnostics/residual-analysis.md)
- [Rhombicity Diagnostics](../diagnostics/rhombicity.md)
