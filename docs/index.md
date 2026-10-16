# PCS Analyzer

!!! info "Documentation status"
    The core structure of this manual is now in place. Further revisions and
    step-by-step tutorials are in progress. Some topics remain incomplete;
    review the limitations documented for each workflow.

**PCS Analyzer** is a Python-based application for the analysis, fitting, and visualization of **pseudocontact shifts (PCS)** in paramagnetic molecular systems.

It provides an integrated workflow that connects molecular structure, magnetic susceptibility anisotropy, and experimental NMR shift data.

<div class="grid cards" markdown>

-   :material-molecule:{ .lg .middle } **Structure-based analysis**

    Load molecular geometries and evaluate PCS in a defined molecular or tensor coordinate frame.

-   :material-chart-scatter-plot:{ .lg .middle } **PCS fitting and diagnostics**

    Compare experimental and calculated shifts, optimize tensor parameters, and inspect fitting residuals.

-   :material-axis-arrow:{ .lg .middle } **Scientific visualization**

    Explore PCS distributions using polar, Cartesian, projection, and three-dimensional representations.

</div>


## Main interface

<figure>
  <a href="images/main-workspace.png" class="glightbox" data-gallery="main-workspace">
    <img src="images/main-workspace.png" alt="PCS Analyzer workspace with Ribbon, document tabs, data table, analysis area, and Control Panel">
  </a>
  <figcaption>PCS Analyzer main workspace.</figcaption>
</figure>

The main workspace combines a **Ribbon**, **document tabs**, a molecular
and PCS data table, analysis tabs, and a sectioned **Control Panel**. Use
**Home** for project actions, **Data** for experimental PCS and viewers, and
**Tools** for workbenches and modules. A contextual Ribbon page provides
commands for the selected analysis.

Each document holds a separate PCS analysis; the table and Control Panel
show the active document. See [Interface Overview](interface.md) for the
layout and [Projects and Persistence](projects.md) for saving projects and
workspace references.


## Typical workflow

A PCS analysis generally follows four steps:

1. **Load a molecular structure**  
   Import a molecular geometry from an XYZ file or supported ORCA output.

2. **Define the magnetic susceptibility tensor**  
   Specify the axial susceptibility anisotropy and, when required, the rhombic component and tensor orientation.

3. **Add experimental NMR data**  
   Assign experimental PCS values to the corresponding nuclei. Observed and diamagnetic chemical-shift layers are managed separately.

4. **Analyze, fit, and visualize**  
   Compare calculated and experimental shifts, perform fitting or diagnostic analysis, and inspect the resulting PCS distribution.

[Quick Start](quick-start.md){ .md-button .md-button--primary }
[Installation](installation.md){ .md-button }


## Visualization

<div class="grid cards" markdown>

-   <a href="images/2d-polar-plot.png"
       class="glightbox"
       data-gallery="visualization">
      <img src="images/2d-polar-plot.png"
           alt="2D polar PCS plot">
    </a>

    **2D Polar PCS Plot**

    Angular representation of the PCS field for inspection of axial contributions.

    [Learn more](visualization/polar-plot.md)

-   <a href="images/3d-viewer.png"
       class="glightbox"
       data-gallery="visualization">
      <img src="images/3d-viewer.png"
           alt="PCS Analyzer Molecular 3D viewer (Qt/PyVista)">
    </a>

    **3D Viewer**

    Inspect molecular geometry and calculated PCS colors, adjust appearance and axes, and prepare figures at a chosen size and DPI.

    [Learn more](visualization/3d-viewer.md)

-   <a href="images/3d-pcs-plot.png"
       class="glightbox"
       data-gallery="visualization">
      <img src="images/3d-pcs-plot.png"
           alt="3D PCS isosurface plot">
    </a>

    **3D PCS Isosurface Plot**

    Three-dimensional visualization of PCS isosurfaces around the molecular structure.

    [Learn more](visualization/3d-pcs-isosurface.md)

-   <a href="images/pcs-workbench-pcs-isosurface.png" class="glightbox" data-gallery="pcs-workbench"><img class="workbench-main" src="images/pcs-workbench-pcs-isosurface.png" alt="PCS Workbench PCS isosurface"></a>

    <span class="workbench-row"><a href="images/pcs-workbench-spindens-isosurface.png" class="glightbox" data-gallery="pcs-workbench"><img src="images/pcs-workbench-spindens-isosurface.png" alt="PCS Workbench spin-density isosurface"></a><a href="images/pcs-workbench-pde-pd.png" class="glightbox" data-gallery="pcs-workbench"><img src="images/pcs-workbench-pde-pd.png" alt="PCS Workbench PDE and PD analysis"></a></span>

    **PCS Workbench**

    Separate ORCA and spin-density workflow for comparing point-dipole and distributed PCS fields, with molecular and field visualization.

    [Learn more](pcs-workbench/overview.md)

-   <a href="images/nmr-workbench.png"
       class="glightbox"
       data-gallery="nmr-workbench">
      <img src="images/nmr-workbench.png"
           alt="Current NMR Workbench with a processed spectrum and the NMR Data table">
    </a>

    **NMR Workbench**

    Process raw 1D NMR data, reference spectra, pick and assign peaks, and prepare chemical-shift observables for PCS analysis.

    `NMR Workbench` incorporates the `NMR Shift Viewer` functionality available through v1.3.4.

    [Learn more](nmr/spectrum-viewer.md)

</div>

## Analysis capabilities

PCS Analyzer includes tools for:

- molecular structure import and coordinate handling
- axial and rhombic PCS calculations
- magnetic susceptibility tensor analysis
- comparison of calculated and experimental shifts
- tensor and orientation fitting
- residual and rhombicity diagnostics
- 2D and 3D PCS visualization
- NMR-oriented analysis tools
- conformational analysis and PCS-guided conformer search

For worked fitting examples, start with [Standard fitting and rhombicity](fitting/standard-fitting-tutorial.md),
then see [Multi-Conformer and reliability](fitting/multi-conformer-tutorial.md),
[Joint PCS+RDC](fitting/joint-pcs-rdc-tutorial.md), [PRE](fitting/pre-tutorial.md),
and [Multi-Lanthanide/Reilley](fitting/multi-lanthanide-tutorial.md).
Review [Current Limitations](reference/limitations.md) when choosing a workflow.

[Conformer Search](conformer-search/tutorial.md), [NMR to PCS fitting](nmr/tutorial-pcs-fitting.md)
and [averaging/visualization practice](visualization/averaging-demo.md) have separate
worked examples. The [repository examples index](https://github.com/boseokhong/PCS-analyzer/tree/main/examples)
identifies starting files, synthetic versus measured data and related tutorials.


## PCS model

PCS Analyzer is primarily designed for approximately rotational molecular geometries, where the magnetic susceptibility tensor can often be treated as effectively axial.

In this default case, the PCS contribution is written as

$$
\delta_{\mathrm{PCS}}\;(\mathrm{ppm})
=
\frac{10^4}{12\pi}
\Delta\chi_{\mathrm{ax}}
G_{\mathrm{ax}},
$$

with

$$
G_{\mathrm{ax}}
=
\frac{3\cos^2\theta-1}{r^3}.
$$

Here, \(r\) is the distance from the paramagnetic centre to the observed nucleus and \(\theta\) is the polar angle relative to the principal \(z\)-axis.

For systems where rhombicity cannot be neglected, PCS Analyzer can include the rhombic contribution and use the full axial–rhombic expression,

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
\frac{\sin^2\theta\cos(2\phi)}{r^3},
$$

and \(\phi\) is the azimuthal angle in the \(xy\)-plane.

The default **Prediction model** is **Axial**. **Full tensor (Axial + Rhombic)**
adds the rhombic term only when **Enable Δχ_rh** is on. Polar contours and
**G_ax vs PCS** remain axial diagnostics under either selection. See
[Active prediction model](pcs/axial-rhombic.md#active-prediction-model).

!!! note "Tensor conventions"

    PCS Analyzer defines

    \[
    \Delta\chi_{\mathrm{ax}}
    =
    \chi_{zz}
    -
    \frac{\chi_{xx}+\chi_{yy}}{2},
    \]

    and

    \[
    \Delta\chi_{\mathrm{rh}}
    =
    \chi_{xx}-\chi_{yy}.
    \]

    In PCS calculations, \(\Delta\chi_{\mathrm{ax}}\) and \(\Delta\chi_{\mathrm{rh}}\) are entered in units of \(10^{-32}\,\mathrm{m^3}\) per molecule, with molecular coordinates in Å.

    The \(z\)-axis defines the principal axial direction, while the \(x\) and \(y\) axes define the orientation of the rhombic contribution.

See [Tensor Conventions](pcs/tensor-conventions.md) and [Mathematical Conventions](reference/mathematical-conventions.md) for detailed definitions and unit conversions.


## Documentation

New users may want to begin with:

- [Installation](installation.md)
- [Quick Start](quick-start.md)
- [Interface Overview](interface.md)

For the theoretical and numerical definitions used by the program:

- [PCS Theory](pcs/theory.md)
- [Tensor Conventions](pcs/tensor-conventions.md)
- [Mathematical Conventions](reference/mathematical-conventions.md)

## Extending PCS Analyzer

PCS Analyzer supports external plugins for adding custom analysis tools, viewers, fitting utilities, and research-specific workflows.

[Plugin Documentation](plugins-docs/overview.md){ .md-button }

## Citation

PCS Analyzer is research software developed for the analysis, fitting, and visualization of pseudocontact shifts in paramagnetic molecular systems.

Archived releases of PCS Analyzer are available through Zenodo:

[**PCS Analyzer on Zenodo**](https://doi.org/10.5281/zenodo.18752129){ .md-button }

Please cite the archived Zenodo release when using PCS Analyzer in academic work.

!!! info "Manuscript"
    A dedicated software description manuscript presenting the implementation, methodology, and scientific applications of PCS Analyzer is currently in preparation.
