# Symmetry and Torsional Ensemble Averaging

PCS Analyzer provides two complementary averaging approaches for PCS analysis:

1. **Local symmetry averaging** for locally threefold groups such as methyl and trifluoromethyl groups.
2. **Torsional ensemble averaging** for conformational motion about rotatable bonds, including automatically detected planar ring systems and manually defined torsional fragments.

Both approaches distinguish between a **representative averaged coordinate** used for display and an **averaged geometrical factor** used for PCS calculation.

!!! important "Averaged coordinates are not used to approximate averaged PCS"

    PCS geometrical factors depend non-linearly on position through their angular terms and the \(r^{-3}\) distance dependence.

    Therefore, in general,

    \[
    G(\langle \mathbf r \rangle)
    \neq
    \langle G(\mathbf r)\rangle.
    \]

    PCS Analyzer consequently averages the geometrical factors calculated from the underlying atomic or sampled coordinates rather than evaluating \(G\) at the averaged coordinate.

## Averaging Settings

Averaging is configured from the dedicated **Averaging Settings** window.

The window contains controls for:

- local CH\(_3\) averaging,
- local CF\(_3\) averaging,
- preservation of original local-symmetry atoms,
- torsional ensemble averaging,
- automatic planar-ring detection,
- manual torsion definition,
- angular range and sampling density,
- preservation of the torsional reference structure.

The main interface shows only a compact summary of the currently active averaging modes.

---

## Local symmetry averaging

### Supported groups

The built-in local-symmetry workflow detects:

- methyl groups, CH\(_3\),
- trifluoromethyl groups, CF\(_3\).

Both are treated through the same generic AX\(_3\) group logic.

A candidate group consists of a centre atom A bonded to exactly three equivalent atoms X of the same element, together with the required external substituent connectivity for the supported group type.

### Connectivity detection

XYZ and ORCA coordinate files do not normally contain the bond graph required by this feature.

PCS Analyzer therefore infers connectivity from interatomic distances and tabulated covalent radii (Cordero et al., 2008).

Two atoms \(i\) and \(j\) are treated as bonded when

\[
d_{ij}
\le
(1+\mathrm{tol.})
\left(
r_i^\mathrm{cov}
+
r_j^\mathrm{cov}
\right),
\]

where `tol.` is the **Bond tolerance** value defined in the PCS Workbench controls.

The default value is

\[
\mathrm{tol.}=0.03,
\]

corresponding to a bond-distance scale factor of

\[
1+\mathrm{tol.}=1.03.
\]

The tolerance can be adjusted when the default distance criterion is not appropriate for a particular structure.

If an element is not present in the internal covalent-radius table, a fallback radius of

\[
0.77\ \text{Å}
\]

is used for bond-graph construction.

!!! warning "Displayed connectivity is not a bond path"

    The lines shown between atoms in the **2D Polar PCS Plot** are intended only as a visual aid for structural connectivity.

    They do **not** represent experimentally determined bond paths, bond orders, or electronic bonding information.

    The connections are generated from distance-based criteria and are used only to make the spatial relationship between atoms easier to follow in the 2D representation.

### Methyl-group detection

A methyl group is identified when a carbon atom has:

1. exactly three bonded hydrogen atoms, and
2. exactly one bonded non-hydrogen atom.

Schematically,

```text
    H
    |
H — C — R
    |
    H
```

The three hydrogen atoms are the members of the detected symmetry group.

### CF3-group detection

A trifluoromethyl group is identified analogously when a carbon atom has:

1. exactly three bonded fluorine atoms, and
2. exactly one bonded non-fluorine atom.

Schematically,

```text
    F
    |
F — C — R
    |
    F
```

The three fluorine atoms are the members of the detected symmetry group.

### Pseudo-atom coordinate

For a detected AX\(_3\) group with member coordinates

\[
\mathbf r_1,\quad
\mathbf r_2,\quad
\mathbf r_3,
\]

PCS Analyzer defines the pseudo-atom position as the Cartesian centroid,

\[
\mathbf r_\mathrm{avg}
=
\frac{
\mathbf r_1+\mathbf r_2+\mathbf r_3
}{3}.
\]

For methyl groups, the pseudo atom represents the three H atoms.

For CF\(_3\) groups, the pseudo atom represents the three F atoms.

The pseudo atom is assigned a new reference ID, while the mapping between the pseudo atom and its original member atoms is retained for subsequent geometrical-factor averaging.

Labels are generated from the group type and the original centre atom, for example:

```text
MeH@C12
CF3F@C7
```

### Local-symmetry display modes

#### Averaging disabled

When the corresponding local-symmetry averaging option is disabled, no pseudo atom is generated for that group type and the original structure is retained.

#### Keep original atoms disabled

When local symmetry averaging is enabled and **Keep original atoms** is disabled, the three original H or F member atoms are removed from the effective representation and replaced by one pseudo atom.

Conceptually,

```text
H1 + H2 + H3        F1 + F2 + F3
     ↓                   ↓
   H_avg               F_avg
```

#### Keep original atoms enabled

When **Keep original atoms** is enabled, the original member atoms remain visible and the corresponding pseudo atom is displayed in addition to them.

This option affects the representation only; the pseudo atom still represents the averaged local-symmetry site for PCS analysis.

### Reference-ID handling

PCS Analyzer preserves the existing atom reference IDs for original atoms that remain in the effective structure.

Pseudo atoms receive newly generated IDs greater than the current maximum reference ID.

When original local-symmetry member atoms are hidden from the effective representation, the IDs of unaffected atoms are retained.

This allows experimental-data references to remain associated with the original atom numbering wherever those atoms are preserved.

### Local-symmetry geometrical-factor averaging

The pseudo-atom centroid is a representative coordinate only. PCS geometrical factors are calculated from the original member positions.

For the axial PCS model,

\[
G_{\mathrm{ax,avg}}
=
\frac{1}{N}
\sum_i G_{\mathrm{ax},i}.
\]

For a threefold group,

\[
G_{\mathrm{ax,avg}}
=
\frac{
G_{\mathrm{ax},1}
+
G_{\mathrm{ax},2}
+
G_{\mathrm{ax},3}
}{3}.
\]

When rhombicity is included, the axial and rhombic terms are averaged independently:

\[
G_{\mathrm{ax,avg}}
=
\frac{1}{N}
\sum_i G_{\mathrm{ax},i},
\]

\[
G_{\mathrm{rh,avg}}
=
\frac{1}{N}
\sum_i G_{\mathrm{rh},i}.
\]

The averaged PCS is then

\[
\delta_{\mathrm{PCS,avg}}
=
\frac{10^4}{12\pi}
\left(
\Delta\chi_{\mathrm{ax}}G_{\mathrm{ax,avg}}
+
\Delta\chi_{\mathrm{rh}}G_{\mathrm{rh,avg}}
\right).
\]

For symmetry-equivalent nuclei with equal statistical weight, this is equivalent to averaging their individual PCS values.

---

## Torsional ensemble averaging

### Concept

Torsional ensemble averaging represents conformational motion about a rotatable bond by explicitly sampling a set of molecular geometries.

For a torsional coordinate \(\phi\), the ensemble average of a geometrical factor is

\[
\langle G\rangle
=
\int G(\phi)P(\phi)\,d\phi,
\]

where \(P(\phi)\) is the torsional probability distribution.

In the discrete implementation used by PCS Analyzer,

\[
\langle G\rangle
\approx
\sum_k w_k G_k,
\qquad
\sum_k w_k=1.
\]

For uniform sampling,

\[
w_k=\frac{1}{N}.
\]

The current workflow therefore treats the sampled torsional states as equally weighted unless another weighting model is introduced externally.

### Automatic planar-ring detection

Automatic torsional-group detection starts from the inferred molecular graph.

The detection workflow is designed to avoid confusing ligand torsions with metal coordination geometry:

```text
Full molecular graph
        ↓
Remove metal atoms / metal-coordination edges
        ↓
Ligand-only covalent graph
        ↓
Cycle detection
        ↓
Planarity test
        ↓
Planar-system classification
        ↓
External attachment-bond detection
        ↓
Rotating-fragment definition
```

Candidate cycles are tested for planarity using their Cartesian geometry.

Metal-containing coordination rings or chelate planes are not intended to be treated as automatically rotatable ligand rings.

#### Fused planar systems

Coplanar rings that share edges are treated as a single rigid planar system rather than as independent overlapping rotors.

This is important for fused systems such as naphthalene-, quinoline-, or indole-like fragments, where rotation of an individual constituent ring would not represent a physically meaningful rigid-body torsion.

### Ring classification

Detected planar systems are assigned descriptive interface labels from their ring size and elemental composition.

Examples include:

- `Phenyl-like`
- `Pyridyl-like`
- `Diazine-like`
- `Triazine-like`
- `Pyrrolyl-like`
- `Furyl-like`
- `Thienyl-like`
- `Diazolyl-like`
- `Oxazolyl-like`
- `Thiazolyl-like`
- `Planar N-member ring`

These labels are **descriptive only**. They do not assign aromaticity, protonation state, formal bond order, or a unique positional isomer.

### Torsional axis and rotating fragment

For an automatically detected planar system, the external bond connecting the planar system to the remainder of the molecular structure defines the candidate torsional axis.

Conceptually, cutting this bond separates the graph into two components. The connected component on the ring side defines the rotating fragment.

The rotating fragment can therefore include substituents that are rigidly attached to the planar ring; it is not restricted to the cycle atoms themselves.

Metal-containing torsional axes are rejected from the automatic ligand-ring workflow.

### Manual torsion definition

When automatic detection is not appropriate, a torsional group can be defined manually.

The manual workflow specifies:

- the two atoms defining the rotation axis, and
- the fragment that moves about that axis.

Manual definition is useful for non-standard rotors, unusual connectivity, or cases where automatic cycle detection does not capture the intended motion.

### Angular range

A torsional ensemble can represent either a full rotation or a restricted angular interval.

#### Full rotation

For full rotational averaging, the interval is sampled over 0–360° without duplicating the 360° endpoint.

#### Restricted rotation

A user-defined start and end angle can be used when the motion is known or assumed to occupy only part of the full torsional coordinate.

### Sampling density

The available sampling presets are:

| Preset | Samples | Angular spacing for 360° |
| --- | ---: | ---: |
| Fast | 12 | 30° |
| Normal | 24 | 15° |
| Fine | 72 | 5° |
| Custom | user-defined | user-defined |

**Normal** is the default compromise between angular resolution and computational cost.

### Ensemble-averaged coordinates

Each atom in a torsionally averaged fragment retains its original reference ID.

For atom \(i\), the representative averaged Cartesian coordinate is

\[
\langle \mathbf r_i\rangle
=
\sum_k w_k\mathbf r_{i,k}.
\]

This coordinate is used for table display and structural visualization.

It should not be interpreted as a necessarily populated physical conformer. For a wide rotational distribution, the Cartesian average may lie at a position that is not occupied by any individual member of the ensemble.

No new pseudo atom is created for torsional averaging.

### Torsional geometrical-factor averaging

For each atom \(i\), geometrical factors are calculated for the sampled coordinates and then averaged:

\[
\bar G_{\mathrm{ax},i}
=
\sum_k w_kG_{\mathrm{ax},i,k},
\]

and, when rhombicity is included,

\[
\bar G_{\mathrm{rh},i}
=
\sum_k w_kG_{\mathrm{rh},i,k}.
\]

The PCS for that atom is calculated as

\[
\delta_{\mathrm{PCS},i}
=
\frac{10^4}{12\pi}
\left(
\Delta\chi_{\mathrm{ax}}\bar G_{\mathrm{ax},i}
+
\Delta\chi_{\mathrm{rh}}\bar G_{\mathrm{rh},i}
\right).
\]

The PCS is therefore **not** calculated from

\[
G(\langle\mathbf r_i\rangle).
\]

#### Coordinate transformations during fitting

During fitting or tensor-frame transformations, the sampled conformer coordinates are transformed into each trial frame before their geometrical factors are evaluated and averaged.

This preserves the relationship between the torsional ensemble and the current susceptibility-tensor frame throughout the fitting procedure.

### Multiple torsional groups

Multiple torsional groups can be enabled in the same structure.

The current workflow treats separate local torsions independently rather than constructing the full Cartesian product of all torsional samples.

For example, two independently sampled 24-point torsions are not automatically expanded into a 24 × 24 = 576-member global conformational ensemble.

Overlapping rotating fragments require caution because independent averaging is not appropriate when the same atoms participate in coupled torsional motions. Fused planar systems are therefore merged where appropriate, while more general coupled torsions remain outside the scope of the current local-averaging model.

---

## Relationship between the two averaging approaches

Local symmetry averaging and torsional ensemble averaging use the same central PCS principle but represent different physical situations.

| Feature | Local symmetry averaging | Torsional ensemble averaging |
| --- | --- | --- |
| Typical use | CH\(_3\), CF\(_3\) | rotating planar ring / defined torsion |
| Representative coordinate | pseudo-atom centroid | per-atom ensemble-averaged coordinate |
| New reference ID | yes, for pseudo atom | no |
| Original atom identity | collapsed or retained for display | retained |
| PCS geometry | average member \(G\) values | average sampled \(G\) values |

---

## Visualization and table representation

### 2D Polar PCS Plot

PCS Analyzer distinguishes reference and averaged representations visually.

- **Reference atoms** are shown as filled circles using element CPK colors.
- **Torsional ensemble-averaged coordinates** are shown as hollow circles with element-colored outlines.
- **CH\(_3\) pseudo atoms** are shown as `×` markers using a dedicated warm-orange representation color.
- **CF\(_3\) pseudo atoms** are shown as `×` markers using a dedicated teal representation color.

The averaging legend is generated dynamically and contains only representations that are currently visible.

The `Reference` legend entry is shown only when a reference representation is actually displayed, for example when **Keep original atoms** or **Keep reference structure** is enabled.

If no averaging representation is active, no averaging legend is displayed.

The PCS contour legend remains separate from the averaging-representation legend.

### 3D structure view

The 3D structure view follows the same reference-versus-averaged representation concept.

When torsional averaging is active, the ensemble-averaged geometry can be displayed as the active structural representation. If **Keep reference structure** is enabled, the reference geometry is retained as an overlay for comparison.

The reference overlay is visual only and does not create duplicate PCS observations or fitting rows.

### Table representation

Torsional ensemble averaging does not create additional table rows.

Each atom retains its original reference ID, while the displayed Cartesian coordinates and geometrical factors correspond to the ensemble-averaged values.

For clarity, torsionally averaged atoms can be marked in the Atom field using a display-only suffix, for example:

```text
C ⟨avg⟩
H ⟨avg⟩
N ⟨avg⟩
```

The suffix does not modify the internal element identity.

Local-symmetry pseudo atoms retain labels such as:

```text
MeH@C12
CF3F@C7
```

---

## Detection and model limitations

### Local symmetry

Local symmetry-group recognition depends on the distance-based bond graph.

Consequently, unusual geometries, distorted bond distances, missing atoms, non-standard element labels, or structures near the bond-distance threshold can affect group detection.

The automatic local-symmetry workflow is specifically configured for CH\(_3\) and CF\(_3\) groups with the expected substituent connectivity. It should therefore be interpreted as a geometry-based analysis convenience rather than a general chemical-symmetry engine.

### Torsional ensemble averaging

Automatic torsional detection is also dependent on the inferred molecular graph and Cartesian planarity.

Potential limitations include:

- incomplete or unusual ligand connectivity,
- strongly distorted rings,
- ambiguity in the external attachment bond,
- coupled torsional degrees of freedom,
- sterically inaccessible sampled orientations,
- non-uniform conformational populations.

Uniform torsional sampling is a geometrical averaging model. It does not by itself provide an energetic or Boltzmann-weighted conformational distribution.

For systems where conformer energies or experimentally determined populations are important, the sampled-state weights should be interpreted accordingly or a more explicit conformational-ensemble treatment should be used.

---

## Processing summary

```text
Raw Cartesian structure
        ↓
Build distance-based molecular graph
        ↓
Local symmetry detection (CH3, CF3 groups)
        ↓
Generate symmetry pseudo atoms
        ↓
Planar-system / torsional-group detection
        ↓
Define torsional axis and rotating fragment
        ↓
Generate torsional conformer samples
        ↓
Apply tensor-frame translation / rotation
        ↓
Calculate Gax and, when required, Grh
for original member or sampled coordinates
        ↓
Average geometrical factors
        ↓
Generate representative averaged coordinates
        ↓
PCS analysis / fitting / visualization
```

---

## Reference

B. Cordero, V. Gómez, A. E. Platero-Prats, M. Revés, J. Echeverría, E. Cremades, F. Barragán, S. Alvarez, **Covalent radii revisited**, *Dalton Transactions* (2008), 2832–2838. DOI: 10.1039/B801115J.
