# Quick Start

This guide introduces the basic PCS Analyzer workflow using a molecular structure and an axial magnetic susceptibility anisotropy.

The purpose is to become familiar with the main interface before moving on to tensor fitting, diagnostics, or advanced visualization.

---

## 1. Start PCS Analyzer

Launch PCS Analyzer.

The main interface contains controls for:

- magnetic susceptibility parameters
- PCS contour settings
- molecular structure loading
- coordinate handling
- experimental shift data
- fitting and diagnostics
- visualization tools

The 2D PCS visualization window is opened automatically when PCS Analyzer starts.

---

## 2. Load a molecular structure

**[File] -> Load XYZ...**

The structure loader currently accepts:

- `.xyz`
- ORCA `.out` `.log` files

Structure files can also be opened by **drag and drop** directly into the PCS Analyzer window.

After selecting the structure file, PCS Analyzer asks for the **centre atom**.

For example:

```text
U
```

or

```text
Nd
```

Enter the element symbol corresponding to the paramagnetic centre.

PCS Analyzer uses the coordinates of the **first matching atom** in the structure as the PCS origin.

!!! note

    The selected centre defines the origin used for the PCS geometrical analysis.

    Verify that the correct atom is selected when the structure contains more than one atom of the same element.

!!! note "Project files"
    PCS Analyzer supports its own project file format, `.pcsp`, for saving and reopening analysis sessions as a project.

    `.pcsp` project files can be loaded from the File menu or opened by drag and drop.

!!! note "Recent files"
    Recently opened structure files are recorded and can be reopened from **[File] -> Recent files**.

---

## 3. Inspect the molecular structure

The loaded atoms are shown in the coordinate table and visualization windows.

The molecular coordinates are evaluated relative to the selected paramagnetic centre.

Elements can be included or excluded from the displayed dataset using the **Select elements to display** controls.

This is useful when the PCS analysis is intended only for a selected nucleus type, for example hydrogen atoms.

---

## 4. Enter the axial susceptibility anisotropy

Enter the axial magnetic susceptibility anisotropy in:

**Δχ_ax values (E-32 m³)**

For example:

```text
-4.0
```

corresponds to

$$
\Delta\chi_{\mathrm{ax}}
=
-4.0\times10^{-32}\ \mathrm{m^3}
$$

in the convention used by the PCS calculation.

Click:

**Update**

to recalculate the PCS distribution.

!!! warning "Tensor convention"

    Always verify the susceptibility convention and units before importing tensor values from another program or publication.

    See [Tensor Conventions](pcs/tensor-conventions.md) for details.

---

## 5. Adjust the PCS plot

The PCS contour display can be controlled using:

- **PCS plot range (ppm)**
- **PCS plot interval (ppm)**
- **Half/Quarter plot toggle**

After changing the parameters, click **Update**.

The resulting PCS contour shows the angular dependence of the PCS field for the selected susceptibility anisotropy.

---

## 6. Inspect calculated PCS values

For each selected atom, PCS Analyzer calculates quantities including:

- Cartesian coordinates
- distance from the paramagnetic centre
- angular position relative to the tensor frame
- geometrical factors \(G_{\mathrm{ax}}\) and, where applicable, \(G_{\mathrm{rh}}\)
- calculated PCS value

The main atom table links the molecular geometry to the calculated PCS values.

Selecting atoms in the table can also be used to inspect their positions in the available visualization tools.

---

## 7. Rotate the coordinate frame

The molecular coordinates can be rotated using the X- and Y-axis rotation controls.

This allows the molecular structure to be examined relative to a chosen tensor orientation.

Rotation changes the angular relationship between each nucleus and the PCS tensor and therefore changes the calculated PCS values.

Use the angle sliders or enter angles directly.

For analyses including a rhombic susceptibility component, rotation about the **z-axis** can additionally be adjusted in **[Rhombicity Table]**. This changes the orientation of the \(x\)- and \(y\)-axes around the principal \(z\)-axis and therefore affects the \(G_{\mathrm{rh}}\) contribution while leaving the axial \(G_{\mathrm{ax}}\) term unchanged.

---

## 8. Add experimental shift data

Experimental values can be associated with nuclei in the main table.

PCS Analyzer provides tools to:

- enter individual experimental values
- import experimental data from a file
- paste experimental data from the clipboard
- export a template for data entry

Once experimental data are present, calculated and experimental shifts can be compared directly.

These data are also used by the fitting and diagnostic modules.

---

## 9. Open additional viewers

The **Open viewers** section provides access to several complementary visualization tools.

### 2D PCS Plot

Displays the angular PCS distribution.

### 3D structure

Displays the molecular structure in three dimensions.

### Projection

Provides alternative angular projections of the PCS geometry.

### NMR Spectrum

Visualizes calculated and experimental shift data in an NMR-oriented representation.

---

## 10. Continue to fitting

Once both structure and experimental shift data have been loaded, more advanced analysis can be performed using the fitting and diagnostic tools.

Typical next steps include:

- fitting tensor orientation
- fitting axial and rhombic susceptibility parameters
- evaluating calculated versus experimental PCS values
- residual analysis
- testing possible rhombic contributions
- conformational analysis

See [Fitting Overview](fitting/overview.md) for the available fitting modes.

---

## Basic workflow summary

```text
Load structure
      ↓
Select paramagnetic centre
      ↓
Enter Δχax
      ↓
Update PCS calculation
      ↓
Inspect / rotate geometry
      ↓
Add experimental shifts
      ↓
Compare, fit, diagnose, visualize
```

---

[Interface Overview](interface.md){ .md-button }
[Fitting Overview](fitting/overview.md){ .md-button .md-button--primary }
