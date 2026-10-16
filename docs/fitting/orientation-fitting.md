# Orientation Fitting

PCS Analyzer provides two fixed-centre orientation models in the **🔍
Fitting** tab. Mode A defines an orientation from selected ligand donors;
Mode B applies a general rigid-body Cartesian Euler rotation. Both fit the
selected δ_Exp points while leaving the assigned metal position fixed.

The canonical rotation and tensor-frame definitions are given in
[Coordinate Handling](../structures/coordinate-system.md) and
[Tensor Conventions](../pcs/tensor-conventions.md).

## Mode A: ligand-donor fit

Select **Mode A · Ligand-donor**, one or more donor Ref IDs, an **Axis
vector mode**, and the PCS observations to fit.

The implementation first constructs a unit donor axis from the assigned metal
to the selected donor coordinates. It then fits:

- **θ**, the target angle between that donor axis and the positive tensor
  z-axis; and
- **α**, a twist of the rigid structure around the donor axis after the θ
  alignment.

Both angles start at 0° and are bounded from −180° to +180°. The mathematical
polar angle between two vectors is 0–180°, but the optimizer permits a signed
−180° to +180° θ parameter because it is used as a rotation target. Do not
interpret the fitted θ as an unrestricted conventional spherical polar angle
without considering this parameterization.

### Donor-axis construction

| Axis vector mode | Construction used by the implementation |
| --- | --- |
| `first` | Metal-to-first-donor unit vector |
| `bisector` | Normalized sum of the metal-to-donor unit vectors |
| `average` | Same normalized unit-vector sum as `bisector` |
| `centroid` | Vector from the metal to the arithmetic centroid of donor coordinates |
| `normal` | Normal to the donor plane (two donors define a cross-product normal) |
| `pca` | Principal direction of the donor unit vectors |

With one donor, `normal` and `pca` reduce to the first donor direction. A
degenerate axis construction also falls back to the first donor direction.
Plane-normal and PCA directions have an inherent sign ambiguity, so equivalent
orientations or different angle signs can occur.

When Δχ_rh is nonzero or fitted, Mode A also applies the current
**Rotate around Z-axis (°)** value from the Rhombicity Table as the in-plane
azimuth reference. Changing that control changes the rhombic frame used by
the fit while leaving the axial z direction unchanged.

## Mode B: Euler fit

Select **Mode B · Euler** and the PCS observations to fit. No donor selection
is required. The mode fits a rigid-body rotation about the fixed metal centre
using three angles shown in the result as `ax`, `ay`, and `az`.

These are active, extrinsic rotations about fixed Cartesian axes in **X → Y →
Z** order. Angles start at 0° unless
**Use current tensor as initial guess** is selected, and are bounded from
−180° to +180°. The Tensor Euler ZYZ controls express the same orientation
with different angle values; do not substitute their triplet directly for XYZ.

The fitted `az` establishes the in-plane reference for a rhombic calculation.
For a purely axial tensor, rotation about the tensor z-axis does not change
the axial PCS term and may therefore be weakly or non-identifiable.

## Atom and tensor behavior

The structure is treated as a rigid coordinate set; the selected PCS
observations define the residuals, not a set of independently moved atoms. The application
uses its current effective geometry, including the averaging-aware geometry
factors when symmetry or torsional averaging is enabled.

Use **Refresh** after changing visibility or assignments. Visibility filters
the available observations; it does not make a separately deformable rigid group.

For either mode, Δχ_ax and Δχ_rh may be fitted or held at their current
values. Mode A supports **Preview Geometry** only. Mode B **Apply to Tensor**
copies Δχ_ax and orientation, retaining the existing Δχ_rh and rhombic
enablement; its preview changes only the geometry override. See [Fitting Overview](overview.md) for initialization and application and [Quality Metrics](quality-metrics.md) for the report fields.
