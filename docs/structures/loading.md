# Structure Files & Parsing

PCS Analyzer converts supported structure files into a common internal atom representation,

```text
(element, x, y, z)
```

where the Cartesian coordinates are stored in Å.

This page describes the file syntax accepted by the structure loader and the internal processing performed before PCS analysis.

## Supported structure formats

The structure loader accepts:

- XYZ files (`.xyz`)
- ORCA output files (`.out`)
- ORCA log files (`.log`)

The file extension determines which parser is used.

## XYZ files

### Standard XYZ syntax

A conventional XYZ file contains an atom count, one comment line, and one coordinate line for each atom:

```text
5
example structure
U   0.000000   0.000000   0.000000
N   2.350000   0.000000   0.000000
N  -1.175000   2.035160   0.000000
N  -1.175000  -2.035160   0.000000
H   0.000000   0.000000   3.000000
```

If the first non-empty line is an integer, PCS Analyzer interprets it as the atom-count header and skips exactly two lines: the atom-count line and the following comment line. The comment line may be blank.

Headerless XYZ-like coordinate files are also accepted.

### Accepted coordinate-line forms

The XYZ parser accepts several common whitespace-separated formats.

#### Element and coordinates

```text
U  0.000000  0.000000  0.000000
```

interpreted as

```text
Element  X  Y  Z
```

#### Atom index, element, and coordinates

```text
1  U  0.000000  0.000000  0.000000
```

interpreted as

```text
Index  Element  X  Y  Z
```

#### Element, atomic number, and coordinates

```text
U  92  0.000000  0.000000  0.000000
```

interpreted as

```text
Element  AtomicNumber  X  Y  Z
```

#### Atom index, element, atomic number, and coordinates

```text
1  U  92  0.000000  0.000000  0.000000
```

interpreted as

```text
Index  Element  AtomicNumber  X  Y  Z
```

Only the element symbol and the three Cartesian coordinates are retained internally. Atom indices and atomic-number columns, when present, are not stored as part of the coordinate tuple.

Lines that do not match one of the supported coordinate patterns are skipped.

## ORCA output files

For `.out` and `.log` files, PCS Analyzer searches for ORCA Cartesian-coordinate sections with headers of the form

```text
CARTESIAN COORDINATES (ANGSTROEM)
```

or

```text
CARTESIAN COORDINATES (BOHR)
```

Coordinate rows are read as

```text
Element  X  Y  Z
```

until the coordinate block ends.

### Multiple coordinate blocks

ORCA calculations may contain several Cartesian-coordinate blocks, for example during a geometry optimization.

The standard PCS Analyzer structure loader uses the **last detected coordinate block**, which normally corresponds to the latest geometry written to the output file.

Internally, the parser also supports extraction of the first block or all coordinate blocks, although the normal loading workflow uses the last block.

### Unit conversion

Coordinates reported by ORCA in Å are used directly.

Coordinates reported in Bohr are converted according to

\[
1\ \mathrm{Bohr}
=
0.529177210903\ \text{Å}.
\]

All structures therefore enter the main PCS Analyzer workflow in Å.

## Internal processing

After parsing, supported structure formats are represented by the same internal atom list and pass through a common processing sequence.

```text
Input structure file
        ↓
File-extension dispatch
        ↓
XYZ or ORCA coordinate parsing
        ↓
Internal atom list: (element, x, y, z)
        ↓
Store original/raw coordinates
        ↓
Identify the selected paramagnetic centre
        ↓
Generate optional symmetry-averaged coordinates
        ↓
Translate working coordinates relative to the centre
        ↓
Apply coordinate-frame rotation
        ↓
Calculate geometrical quantities
        ↓
PCS analysis, fitting, and visualization
```

### Raw and original coordinates

Immediately after loading, PCS Analyzer stores the parsed coordinates as the original and raw structure representations.

The atom reference IDs initially correspond to the one-based order of atoms in the parsed structure:

```text
1, 2, 3, ...
```

The raw coordinates remain in the absolute coordinate frame of the input file.

### Paramagnetic-centre identification

The user-specified centre element is searched in the parsed atom list from the beginning of the structure.

The **first exact matching element label** is selected. Its original Cartesian coordinates are stored as the paramagnetic-centre position.

If no matching atom is found, structure initialization is stopped and an error is reported.

The raw structure itself is not permanently translated at this stage. Instead, the centre coordinate is subtracted when centre-relative working coordinates are required.

### Effective structure representation

If coordinate symmetry averaging is disabled, the effective structure is identical to the raw structure.

If symmetry averaging is enabled, PCS Analyzer creates a separate effective atom list while preserving the raw coordinates. The averaging procedure is described in [Symmetry Averaging](symmetry-averaging.md).

### Centre-relative coordinates

For geometrical PCS analysis, the paramagnetic-centre coordinate

\[
\mathbf r_\mathrm{M}
=
(x_\mathrm{M},y_\mathrm{M},z_\mathrm{M})
\]

is subtracted from each working coordinate,

\[
\mathbf r_i^{\,0}
=
\mathbf r_i-\mathbf r_\mathrm{M}.
\]

The selected centre therefore becomes the origin of the PCS coordinate system without modifying the stored raw structure.

### Coordinate-frame rotation

Centre-relative coordinates are then rotated according to the current coordinate-frame orientation.

The rotation implementation and coordinate conventions are described in [Coordinate Handling](coordinate-system.md).

### Geometrical quantities

The transformed coordinates are used to derive the geometrical quantities required by PCS analysis.

For a coordinate

\[
(x,y,z),
\]

the radial distance is

\[
r=\sqrt{x^2+y^2+z^2},
\]

and the polar angle relative to the \(z\)-axis is

\[
\theta=\cos^{-1}\left(\frac{z}{r}\right).
\]

When a rhombic contribution is evaluated, the azimuthal orientation in the \(xy\)-plane is additionally relevant.

These centre-relative geometrical quantities are subsequently used by the PCS calculation, fitting, diagnostics, and visualization routines.

## Parsing limitations

The structure loader is intentionally lightweight and geometry-oriented.

It does not reconstruct chemical connectivity from the input file during basic parsing, and it does not use bond-order or electronic-structure information from ORCA output as part of the structure representation.

Bond connectivity required by features such as symmetry averaging is inferred separately from Cartesian distances and covalent radii.

!!! note "Project files"

    PCS Analyzer project files (`.pcsp`) are application project containers rather than structure-file formats. They are handled separately from the XYZ/ORCA structure parser described on this page.
