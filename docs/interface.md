# Interface Overview

PCS Analyzer is organized as a single analysis workspace combining molecular data, PCS calculations, fitting, diagnostics, and visualization tools.

## Main interface

<figure>
  <a href="../images/main-interface.png" class="glightbox" data-gallery="interface-overview">
    <img src="../images/main-interface.png" alt="PCS Analyzer main interface">
  </a>
  <figcaption>
    PCS Analyzer main interface.
  </figcaption>
</figure>

The main window can be divided into four functional areas.

## 1. Menu bar

The menu bar provides access to application-level functions such as file loading, visualization options, modules, settings, and help.

Structure files and PCS Analyzer project files can be opened from the File menu, while additional built-in tools and external plugins are accessed through the Modules menu.

---

## 2. Molecular data table

The upper part of the main window contains the molecular data table.

This area provides a numerical overview of the currently loaded structure and its PCS-related quantities, including atomic coordinates, geometrical factors, calculated PCS values, and experimental shift data.

Experimental values can also be imported, edited, or cleared from this area.

---

## 3. Analysis workspace

The large central area is the main analysis workspace.

Different analysis tools are organized into tabs, allowing the same molecular and experimental dataset to be examined using several complementary approaches.

These include geometrical-factor analysis, rhombicity analysis, PCS fitting, advanced fitting, and conformer-search tools.

Results, plots, fitting statistics, and diagnostics are displayed within this workspace depending on the selected tab.

---

## 4. Control panel

The right-hand panel contains the principal controls for the current PCS model and molecular coordinate frame.

This area is used to define susceptibility parameters, PCS plotting conditions, molecular display options, and coordinate-frame rotations.

It also provides direct access to the main visualization tools, including the 2D PCS plot, projection view, 3D structure viewer, 3D PCS plot, NMR shift viewer, and PCS Workbench.

The available controls may change slightly depending on the analysis mode or software version.

---

## Workflow

The interface is designed so that structure definition, PCS calculation, experimental comparison, fitting, and visualization can be performed within the same working environment.

A typical workflow proceeds from the molecular data table and control panel to the central analysis workspace, with additional viewers opened when needed.