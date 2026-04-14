# Pseudocontact Chemical Shift (PCS) Analyzer
![version](https://img.shields.io/badge/version-1.3.2-blue) ![license](https://img.shields.io/badge/license-BSD%203--Clause-green) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18752129.svg)](https://doi.org/10.5281/zenodo.18752129)

<img width="1409" height="919" alt="PCS Analyzer main interface" src="https://github.com/user-attachments/assets/e4a0b469-4a98-4552-803d-6cb8a3fbfdab" />
<img width="2420" height="882" alt="PCS Analyzer additional views" src="https://github.com/user-attachments/assets/7981899a-5cc4-46a6-9462-152e7a0f38bb" />

**PCS Analyzer** is a Python-based desktop application for the analysis, visualization, and fitting of pseudocontact chemical shifts (PCS) in paramagnetic molecular systems. The program is designed for interactive structure-based PCS interpretation and is particularly suited to small and medium-sized coordination complexes for which tensor-frame analysis, geometrical inspection, and rapid model comparison are essential.

The software integrates molecular structure import, 2D and 3D PCS visualization, tensor utilities, regression-based diagnostics, rhombicity analysis, fitting workflows, NMR-oriented data inspection, and conformer-based refinement in a unified graphical interface.

> [!NOTE]
> Required Python packages: `numpy`, `scipy`, `matplotlib`, `pandas`, and `openpyxl`  
> Optional / additional packages: `ttkbootstrap`, `pyvista`, `vtk`, `pyfftw`, and `imageio`
> - `pyvista` for 3D PCS field / molecular visualization, and `ttkbootstrap` for enhanced GUI styling
> - `pyfftw` is optional. The code falls back to `numpy.fft` if it is not installed.
> - `imageio` is required for GIF export [PyVista's `Plotter.open_gif()`].
---
## Overview

PCS Analyzer provides an interactive workflow that connects molecular coordinates, PCS geometry factors, experimental shift data, and tensor-based fitting procedures. Rather than treating plotting, table inspection, fitting, and diagnostics as separate tasks, the program links them directly in the GUI to support iterative analysis.

Core use cases include:
- inspection of tensor-frame molecular geometry
- comparison of calculated and experimental PCS values
- axial and rhombic PCS diagnostics
- donor-axis and Euler-based fitting strategies
- advanced fitting and conformer-assisted refinement

## Recent Changes

Release **v1.3.2**
- added **PCS Workbench**, a standalone workspace for FFT-based PDE PCS analysis from ORCA magnetic susceptibility tensors and spin-density grids `.3d`
  - direct import of ORCA `.out` / `.log` files together with ORCA `.3d` spin-density files for PDE workflows
  - temperature-resolved tensor selection, ORCA tensor conversion, and rank-2 traceless tensor handling
  - configurable PDE options including zero-padding, density normalization, and contour auto-scaling
  - PyVista-based PDE field visualization with multi-level PCS isosurfaces, spin-density surfaces, and slice display
  - quantitative comparison between distributed PDE PCS and point-dipole PCS
  - export of computed PDE results to `.png`, `.csv`, compressed **NumPy (`.npz`)**, and temperature-dependent `.gif`
  - **oblique PCS slice plotting** for user-defined planes through the metal centre
  - a **traceless tensor spheroid viewer**
  - PDE field calculation follows the distributed PCS formalism of Charnock and Kuprov: *Phys. Chem. Chem. Phys.*, **2014**, DOI: `10.1039/C4CP03106G`

Release **v1.3.1**
- added a PyVista-based 3D viewer for PCS fields and molecular structures.
- fixed coordinate reference used in the Rhombicity analysis table
- added residual-based text color highlighting for PCS tables
- added export function to save the current visible structure as an `.xyz` file

Release **v1.3.0** introduces several workflow and interface upgrades:
- separated the **2D PCS polar plot** from the main window into a dedicated viewer
- improved the **G_i vs δ plot** with better point interaction and an expanded result/report box
- added **bidirectional table ↔ plot selection/highlighting**
- updated the **3D molecular viewer** and **projection viewer**
- added **Mode C** 8-parameter fitting to the main fitting workflow
- added a dedicated **Advanced Fitting** tab
- added an integrated **Conformer Search and Fitting** workflow
---
## Main capabilities

### 1. Structure import and coordinate handling
- import molecular structures from `.xyz`
- extract coordinates from ORCA `.out` and `.log` files
- inspect and rotate coordinates in the current tensor frame
- visualize molecular structures and PCS fields in interactive 3D viewers
- inspect angular distributions in a projection viewer (`φ / cos(θ)` and Mollweide modes)

### 2. PCS visualization
- display axial PCS contour maps in a dedicated **2D polar plot** window
- overlay atomic positions directly onto PCS plots
- support symmetry-averaged pseudo-points for methyl and CF3 groups
- visualize PCS fields in a PyVista-based 3D viewer
- support interactive 3D inspection of isosurfaces / slices for PCS field analysis
- export PCS-related data and figures

### 3. Table-driven analysis and linked interaction
- interactive main table containing `Ref`, atom labels, rotated coordinates, `G_i`, `δ_PCS`, and `δ_Exp`
- import, paste, clear, undo, and export workflows for experimental `δ_Exp`
- linked highlighting between the table and graphical views
- synchronized inspection across PCS plots, cartesian plots, and structure viewers

### 4. Rhombicity check and analysis
- linear analysis of `G_i` versus experimental shift data
- residual-based rhombicity inspection
- rhombicity table with `G_ax`, `G_rh`, `δ_PCS(ax)`, `δ_PCS(ax+rh)`, and residual comparison
- optional use of `Δχ_rh` and tensor-component reconstruction utilities

### 5. Fitting workflows
- **Mode A**: donor-axis-based fitting
- **Mode B**: global Euler-angle rigid-body fitting
- **Mode C**: 8-parameter fitting including tensor, metal position, and orientation terms
- optional global search for selected fitting modes
- fit summaries and graphical correlation analysis

### 6. Extended analysis modules
- **Advanced Fitting** tab for extended PCS-related workflows
- NMR spectrum viewer with layered `PCS`, `OBS`, `DIA`, and `PARA` displays
- NMR analysis window for non-PCS-oriented inspection
- integrated **Conformer Search and Fitting** with preview, apply, revert, and discard workflows

### 7. PCS Workbench (distributed PCS / PDE)
- standalone FFT-based distributed PCS workflow using ORCA susceptibility tensors and spin-density grids
- comparison between distributed PDE PCS and point-dipole PCS
- interactive 3D visualization of PDE fields with signed PCS isosurfaces and spin-density overlays
- oblique PCS slice plotting for user-defined planes through the metal centre
- tensor spheroid visualization with temperature-dependent `GIF` export
- export of PDE fields, atom-wise comparison tables

The PDE implementation follows the distributed PCS / Kuprov-equation framework (*PCCP*, 2014, DOI: `10.1039/C4CP03106G`).

---
## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/boseokhong/PCS-analyzer.git
cd PCS-analyzer
pip install numpy scipy matplotlib pandas openpyxl
```

Optional packages for 3D visualization, FFT acceleration, and GIF export:

```bash
pip install ttkbootstrap pyvista vtk pyfftw imageio
```

Run the program with:

```bash
python main.py
```
---
## Suggested workflow

A typical analysis sequence is:

1. load a molecular structure (`.xyz` or supported ORCA output)
2. define the metal center and tensor input values
3. inspect rotated coordinates and the main PCS table
4. open the **2D PCS polar plot** for contour-based visualization
5. import or paste experimental `δ_Exp` values
6. inspect the **G_i vs δ** analysis and regression summary
7. evaluate residual patterns using **Rhombicity Check** and **Rhombicity Table**
8. apply fitting in the main **Fitting** tab
9. extend the analysis using **Advanced Fitting** or **Conformer Search** where appropriate

## Fitting modes

<details>
<summary><b>Click to expand fitting mode descriptions</b></summary>

### Mode A — donor-axis fit
This mode defines the fitting axis from one or more donor atoms and is useful when tensor orientation should follow a chemically intuitive ligand-based reference direction.

Available donor-axis definitions include:
- `bisector`
- `normal`
- `pca`
- `centroid`
- `average`
- `first`

### Mode B — Euler fit
This mode performs rigid-body fitting in the global frame using Euler rotations. It is useful when no donor-based axis definition should be imposed a priori.

### Mode C — 8-parameter fit
This mode provides a more complete fitting model and may include:
- `Δχ_ax`
- optional `Δχ_rh`
- metal position `(x, y, z)`
- Euler angles `(α, β, γ)`

It is appropriate when simultaneous refinement of tensor magnitude, anisotropy, orientation, and metal position is required.

</details>

## Export

PCS Analyzer supports export of:
- PCS tables / current visible structure (`.xyz`)
- cartesian and PCS plots
- fitting summaries
- CSV / Excel tables
- graphical outputs (`PNG`, `PDF`, `SVG`, depending on the module)
---
## Version history

<details>
<summary><b>Click to expand version history</b></summary>

**v0.1**
- initial release
- `.xlsx` export with cartesian coordinate output

**v0.2**
- tensor values, PCS range, and interval input
- molar susceptibility tensor calculation

**v0.3**
- XYZ file import and plotting
- coordinate rotation
- atom coordinate table
- geometrical parameter analysis plot

**v0.3.1**
- clicking an atom point on the PCS plot highlights the corresponding table entry

**v0.4**
- added 3D molecular scatter plot

**v0.5**
- reorganized GUI
- added Mollweide projection plot

**v0.6**
- added half / quarter PCS plot view toggle

**v0.7**
- bug fixes

**v1.0.0**
- code refactoring
- PCS fitting

**v1.0.1**
- added `δ_exp` table export / import

**v1.0.2**
- fixed CSV export encoding

**v1.1.0**
- diagnostics
- rhombicity utilities
- fitting function updates

**v1.2.0**
- added NMR spectrum viewer
- added automatic methyl / CF3 symmetry averaging
- improved XYZ loader with ORCA `.out` / `.log` support
- bug fixes and refactoring

**v1.3.0**
- separated 2D PCS polar plot from the main window
- improved `G_i` vs `δ` plot interaction and result reporting
- added linked table ↔ plot highlighting / selection
- updated 3D molecular and projection plots
- added **Mode C** 8-parameter fitting
- added **Advanced Fitting** tab
- added integrated **Conformer Search and Fitting** workflow

**v1.3.1**
- added a PyVista-based 3D viewer for PCS fields and molecular structures.
- fixed coordinate reference used in the Rhombicity analysis table
- added residual-based text color highlighting for PCS tables
- added export function to save the current visible structure as an `.xyz` file

**v1.3.2**
- added **PCS Workbench** for FFT-based distributed PCS / PDE analysis
  - direct import of ORCA `.out` / `.log` files together with ORCA `.3d` spin-density files for PDE workflows
  - temperature-resolved tensor selection, ORCA tensor conversion, and rank-2 traceless tensor handling
  - configurable PDE options including zero-padding, density normalization, and contour auto-scaling
  - PyVista-based PDE field visualization with multi-level PCS isosurfaces, spin-density surfaces, and slice display
  - quantitative comparison between distributed PDE PCS and point-dipole PCS
  - export of computed PDE results to `.png`, `.csv`, compressed **NumPy (`.npz`)**, and temperature-dependent `.gif`
  - **oblique PCS slice plotting** for user-defined planes through the metal centre
  - a **traceless tensor spheroid viewer**
  - PDE field calculation follows the distributed PCS formalism of Charnock and Kuprov: *Phys. Chem. Chem. Phys.*, **2014**, DOI: `10.1039/C4CP03106G`

</details>

---
## Citation

If you use PCS Analyzer in academic work, please cite the Zenodo record associated with this project.

- DOI (all versions): **10.5281/zenodo.18752129**

A dedicated paper describing **PCS Analyzer**, including its methodology, implementation, and representative applications in paramagnetic NMR analysis, is currently in preparation.

## Author

**Boseok Hong**  
Department of Chemistry of the f-elements, Institute of Resource Ecology, HZDR

Contact: [bshong66@gmail.com](mailto:bshong66@gmail.com)

GitHub: [boseokhong/PCS-analyzer](https://github.com/boseokhong/PCS-analyzer)

## Acknowledgements

This project includes code or implementation ideas derived in part from the work of **Sebastian Dechert** for aspects of the 3D molecular scatter-plot functionality, distributed under the BSD 3-Clause License. The original project is available at:

- https://github.com/radi0sus/xyz2tab
