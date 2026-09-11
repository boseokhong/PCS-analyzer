# Coordinate Handling

PCS calculations depend on the position of each observed nucleus relative to the magnetic-susceptibility tensor frame.

PCS Analyzer therefore keeps the input structure and the PCS working coordinate frame conceptually separate. The original structure remains stored in its input coordinate system, while centre translation and rotations are applied to working coordinates as required.

## Coordinate representations

The structure workflow uses several related coordinate representations.

### Original coordinates

The coordinates parsed from the input structure are stored as the original structure. These coordinates are expressed in Å and retain the coordinate frame of the source file.

The original representation is retained so that the initially loaded geometry can be restored after operations such as conformer manipulation.

### Raw coordinates

The raw structure is the current unaveraged molecular geometry used by the application.

Immediately after structure loading, the raw and original coordinates are identical. Operations that intentionally modify the current geometry may update the raw coordinates while leaving the initially stored original geometry available for restoration.

### Effective coordinates

When symmetry averaging is enabled, PCS Analyzer generates a separate effective coordinate representation for the averaging-aware analysis path.

When symmetry averaging is disabled,

\[
\mathbf r_i^{\mathrm{eff}}
=
\mathbf r_i^{\mathrm{raw}}.
\]

The raw structure is retained independently so that the original atomic geometry remains available to viewers and other tools that require explicit atoms.

See [Symmetry Averaging](symmetry-averaging.md).

## PCS origin

Let the selected paramagnetic-centre coordinate in the input frame be

\[
\mathbf r_\mathrm{M}
=
(x_\mathrm{M},y_\mathrm{M},z_\mathrm{M}).
\]

For PCS geometrical analysis, working coordinates are translated to the centre:

\[
\mathbf r_i^{\,0}
=
\mathbf r_i-\mathbf r_\mathrm{M}.
\]

Thus,

\[
\mathbf r_\mathrm{M}^{\,0}
=
(0,0,0).
\]

This translation is performed on the working coordinates. The raw atom list remains stored in the absolute coordinate frame of the imported structure.

This distinction is important for modules that explicitly require the original absolute geometry.

## Coordinate-frame rotation

After translation to the PCS origin, coordinates can be rotated relative to the tensor frame.

PCS Analyzer uses active, extrinsic rotations about fixed Cartesian axes. The default Euler order is

```text
XYZ
```

meaning that the rotations are applied in the order:

```text
X → Y → Z
```

For column-vector notation, the corresponding combined rotation matrix is

\[
\mathbf R
=
\mathbf R_z
\mathbf R_y
\mathbf R_x.
\]

PCS Analyzer stores coordinate arrays as row vectors and therefore applies the transformation as

\[
\mathbf r_i'
=
\mathbf r_i^{\,0}\mathbf R^{\mathrm T}.
\]

### Rotation matrices

For an angle \(\alpha\) about the \(x\)-axis,

\[
\mathbf R_x(\alpha)
=
\begin{pmatrix}
1 & 0 & 0\\
0 & \cos\alpha & -\sin\alpha\\
0 & \sin\alpha & \cos\alpha
\end{pmatrix}.
\]

For an angle \(\beta\) about the \(y\)-axis,

\[
\mathbf R_y(\beta)
=
\begin{pmatrix}
\cos\beta & 0 & \sin\beta\\
0 & 1 & 0\\
-\sin\beta & 0 & \cos\beta
\end{pmatrix}.
\]

For an angle \(\gamma\) about the \(z\)-axis,

\[
\mathbf R_z(\gamma)
=
\begin{pmatrix}
\cos\gamma & -\sin\gamma & 0\\
\sin\gamma & \cos\gamma & 0\\
0 & 0 & 1
\end{pmatrix}.
\]

Angles entered through the application are interpreted in degrees and converted internally to radians for the trigonometric operations.

## Rotation about an arbitrary centre

The general coordinate-rotation helper can rotate a structure about any specified point \(\mathbf c\).

The transformation is

\[
\mathbf r_i'
=
\mathbf c
+
\mathbf R
\left(
\mathbf r_i-\mathbf c
\right)
\]

in column-vector notation.

For the main PCS geometrical calculation, the structure has already been translated relative to the paramagnetic centre, so the rotation centre is

\[
\mathbf c=(0,0,0).
\]

Other viewers may instead rotate coordinates around an explicitly supplied molecular centre.

## Geometrical coordinates

After translation and rotation, a nucleus is represented relative to the PCS tensor frame by

\[
\mathbf r_i'
=
(x_i,y_i,z_i).
\]

The radial distance is

\[
r_i
=
\sqrt{x_i^2+y_i^2+z_i^2}.
\]

The polar angle is

\[
\theta_i
=
\cos^{-1}
\left(
\frac{z_i}{r_i}
\right).
\]

For rhombic PCS calculations, the azimuthal angle is defined by the orientation of the coordinate in the \(xy\)-plane and enters through the rhombic geometrical factor.

The axial term is invariant to rotation around the principal \(z\)-axis, whereas a rhombic term distinguishes the \(x\) and \(y\) directions.

## X-, Y-, and Z-axis roles

For the default axial PCS model, the \(z\)-axis defines the principal axial direction.

Changing the X- and Y-axis rotations changes the orientation of the molecular geometry relative to this axis and therefore changes \(r,\theta\)-dependent PCS geometry.

When rhombicity is included, the orientation of the \(x\)- and \(y\)-axes also becomes physically relevant. Rotation around the \(z\)-axis changes the azimuthal relationship between the structure and the rhombic tensor component without changing the axial \(G_{\mathrm{ax}}\) term.

## Vector alignment helper

PCS Analyzer also contains a vector-alignment operation based on Rodrigues' rotation formula.

Given two vectors,

\[
\mathbf v_1
\quad\text{and}\quad
\mathbf v_2,
\]

the operation constructs a rotation that aligns \(\mathbf v_1\) with \(\mathbf v_2\).

The alignment helper first translates the coordinate set so that its first atom is at the origin and then applies the rotation.

Parallel vectors require no rotation. For antiparallel vectors, the current helper returns the inverted translated coordinate set.

This operation is separate from the standard X/Y/Z coordinate controls and is used by command-based alignment functionality.

## Coordinate flow

The standard PCS coordinate path can be summarized as

```text
Parsed absolute coordinates
        ↓
Raw/effective structure selection
        ↓
Subtract paramagnetic-centre coordinate
        ↓
Centre-relative coordinates
        ↓
Apply active XYZ rotation
        ↓
Tensor-frame coordinates
        ↓
r, θ and azimuthal geometry
        ↓
PCS geometrical factors
```

!!! note "Absolute versus centre-relative coordinates"

    PCS Analyzer deliberately retains absolute raw coordinates while generating centre-relative coordinates for PCS calculations.

    This prevents coordinate-frame ambiguity in modules that operate directly on the molecular geometry while allowing the PCS analysis to use the selected paramagnetic centre as its origin.
