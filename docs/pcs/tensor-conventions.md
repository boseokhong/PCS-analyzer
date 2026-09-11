# Tensor Conventions

PCS Analyzer uses an axial–rhombic representation of the magnetic susceptibility tensor.

The software is primarily designed for approximately rotational coordination geometries, so the axial anisotropy is treated as the default PCS parameter. A rhombic component can be included when in-plane anisotropy is significant.

---

## Isotropic susceptibility

The isotropic magnetic susceptibility is defined as

$$
\chi_{\mathrm{iso}}
=
\frac{\chi_{xx}+\chi_{yy}+\chi_{zz}}{3}.
$$

When an experimental molar susceptibility is supplied, PCS Analyzer uses

$$
\chi_{\mathrm{iso}}=\chi_{\mathrm{mol}}.
$$

---

## Axial anisotropy

PCS Analyzer defines the axial susceptibility anisotropy as

$$
\Delta\chi_{\mathrm{ax}}
=
\chi_{zz}
-
\frac{\chi_{xx}+\chi_{yy}}{2}.
$$

For an axially symmetric tensor,

$$
\chi_{xx}=\chi_{yy},
$$

and \(\Delta\chi_{\mathrm{ax}}\) fully describes the anisotropic part relevant to the default PCS model.

---

## Rhombic anisotropy

The rhombic susceptibility anisotropy is defined as

$$
\Delta\chi_{\mathrm{rh}}
=
\chi_{xx}-\chi_{yy}.
$$

A non-zero \(\Delta\chi_{\mathrm{rh}}\) introduces an in-plane anisotropy and makes the \(x\) and \(y\) tensor directions distinct.

---

## Reconstruction of the principal tensor components

Using the definitions above,

$$
\chi_{xx}
=
\chi_{\mathrm{iso}}
-
\frac{\Delta\chi_{\mathrm{ax}}}{3}
+
\frac{\Delta\chi_{\mathrm{rh}}}{2},
$$

$$
\chi_{yy}
=
\chi_{\mathrm{iso}}
-
\frac{\Delta\chi_{\mathrm{ax}}}{3}
-
\frac{\Delta\chi_{\mathrm{rh}}}{2},
$$

and

$$
\chi_{zz}
=
\chi_{\mathrm{iso}}
+
\frac{2\Delta\chi_{\mathrm{ax}}}{3}.
$$

For the axial case,

$$
\Delta\chi_{\mathrm{rh}}=0,
$$

so that

$$
\chi_{xx}=\chi_{yy}
=
\chi_{\mathrm{iso}}
-
\frac{\Delta\chi_{\mathrm{ax}}}{3},
$$

and

$$
\chi_{zz}
=
\chi_{\mathrm{iso}}
+
\frac{2\Delta\chi_{\mathrm{ax}}}{3}.
$$

---

## Tensor reconstruction modes

PCS Analyzer supports two reconstruction modes.

### Experimental isotropic susceptibility

If \(\chi_{\mathrm{mol}}\) is available, the program uses

$$
\chi_{\mathrm{iso}}=\chi_{\mathrm{mol}}
$$

and reconstructs the full tensor while preserving the isotropic susceptibility.

This mode is appropriate when an experimental molar susceptibility is available and the absolute principal susceptibility values are required.

### Traceless anisotropic reconstruction

If no isotropic susceptibility is supplied, PCS Analyzer uses

$$
\chi_{\mathrm{iso}}=0.
$$

The reconstructed tensor then contains only the anisotropic part:

$$
\chi_{xx}
=
-\frac{\Delta\chi_{\mathrm{ax}}}{3}
+
\frac{\Delta\chi_{\mathrm{rh}}}{2},
$$

$$
\chi_{yy}
=
-\frac{\Delta\chi_{\mathrm{ax}}}{3}
-
\frac{\Delta\chi_{\mathrm{rh}}}{2},
$$

$$
\chi_{zz}
=
\frac{2\Delta\chi_{\mathrm{ax}}}{3}.
$$

This satisfies

$$
\chi_{xx}+\chi_{yy}+\chi_{zz}=0.
$$

!!! note

    The traceless reconstruction represents the anisotropic tensor used for PCS analysis and visualization. It does not contain an absolute isotropic susceptibility offset.

---

## Coordinate convention

PCS Analyzer evaluates PCS values in the tensor coordinate frame.

- \(z\): principal axial direction
- \(x,y\): in-plane principal directions
- \(\theta\): polar angle relative to \(z\)
- \(\phi\): azimuthal angle in the \(xy\)-plane

For the axial model, rotation around \(z\) is irrelevant because \(\chi_{xx}=\chi_{yy}\).

For the rhombic model, the orientation of the \(x\) and \(y\) axes matters because the PCS contains a \(\cos(2\phi)\) term.

---

## Units

PCS Analyzer handles

$$
\Delta\chi_{\mathrm{ax}}
\quad\text{and}\quad
\Delta\chi_{\mathrm{rh}}
$$

in units of

$$
10^{-32}\,\mathrm{m^3}
$$

per molecule.

For tensor reconstruction, molecular susceptibility anisotropies are converted to SI molar susceptibility using Avogadro's constant:

$$
\Delta\chi_{\mathrm{ax,mol}}
=
\Delta\chi_{\mathrm{ax}}
\times
10^{-32}
\times
N_{\mathrm A},
$$

and

$$
\Delta\chi_{\mathrm{rh,mol}}
=
\Delta\chi_{\mathrm{rh}}
\times
10^{-32}
\times
N_{\mathrm A},
$$

where

$$
N_{\mathrm A}
=
6.0221408\times10^{23}\,\mathrm{mol^{-1}}.
$$

Thus, an input value of

$$
\Delta\chi = 1.0
$$

corresponds to

$$
6.0221408\times10^{-9}\,\mathrm{m^3\,mol^{-1}}.
$$

When an experimental molar susceptibility is supplied, PCS Analyzer uses it as

$$
\chi_{\mathrm{iso}}=\chi_{\mathrm{mol}}.
$$

!!! warning

    Magnetic susceptibility conventions are not universal. Always verify the definitions of \(\Delta\chi_{\mathrm{ax}}\), \(\Delta\chi_{\mathrm{rh}}\), tensor-axis ordering, and units before comparing parameters obtained from different programs or publications.

!!! warning "CGS susceptibility data"

    Molar magnetic susceptibilities are often reported in magnetochemistry
    in cgs units of \(\mathrm{cm^3\,mol^{-1}}\).

    These values must be converted to SI before use in PCS Analyzer:

    $$
    \chi_{\mathrm{mol}}^{\mathrm{SI}}
    =
    4\pi\times10^{-6}
    \chi_M^{\mathrm{cgs}}.
    $$

    Do not convert cgs susceptibility to SI using only the geometric
    \(\mathrm{cm^3}\rightarrow\mathrm{m^3}\) factor.

See [PCS Theory](theory.md) for the corresponding PCS equations.
