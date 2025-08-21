# Pseudocontact chemical shift (PCS) analyzer
![version](https://img.shields.io/badge/version-1.0.1-blue) ![license](https://img.shields.io/badge/license-BSD%203--Clause-green)

<img width="961" alt="PCS-analyzer v0 6 capture" src="https://github.com/user-attachments/assets/56e82957-0522-4cc8-86f4-34582192512e" />

**A tool for PCS analysis and 2D polar contour visualization, particularly suited for small paramagnetic complexes with rotational symmetry.**
 > [!NOTE]
 > The code requires `numpy`, `matplotlib`, `pandas`, and `openpyxl` packages to run.  

---

## Version and Changelog 
Latest update: **v.1.0.1** – Added table δ<sub>exp</sub> data export/import

<details>
<summary><b>Click to expand version history</b></summary>

**v.0.1 updates**
- Initial release
- `.xlsx` export: x, y cartesian coordinates are calculated, and each column is separated by PCS values.

**v.0.2 updates**
- Tensor values, PCS range, and intervals can now be inputted for calculations.
- Molar susceptibility tensor calculation  
  ∆χ<sub>ax</sub> = χ<sub>zz</sub> - ((χ<sub>xx</sub> + χ<sub>yy</sub>))/2  
  ∆χ<sub>rh</sub> = χ<sub>xx</sub> - χ<sub>yy</sub>

**v.0.3 updates**
- XYZ file import and plot, coordinate rotation, atom coordinate table added, geometrical parameter analysis plot

**v.0.3.1 updates**
- Clicking on an atom point on the PCS plot to highlight the corresponding table entry, enhancing data visualization and interaction.

**v.0.4 updates**
- Added 3D molecule scatter plot

**v.0.5 updates**
- Reorganized GUI
- Added Mollweide projection plot

**v.0.6 updates**
- PCS polar plot: Half/Quarter view toggle function added  
<img width="961" alt="PCS-analyzer v0 6 capture_2" src="https://github.com/user-attachments/assets/a5a9b0a9-de3c-4cec-821d-45a30d6ef685" />

**v.0.7 updates**
- Bug fix

**v.1.0.0 updates**
- Code refactoring
- PCS fitting
<img width="1402" height="885" alt="image" src="https://github.com/user-attachments/assets/2b492d4b-30c4-4f1c-b139-36fee20cd024" />

**v.1.0.1 updates**
- Added table δ<sub>exp</sub> data export/import

</details>

---

## Fitting Vector Options

The program provides several options for defining the **fitting vector** (the reference axis for PCS fitting). Each option reflects a different way of combining donor atom directions or positions.

<details>

### 1. Bisector
- **Axis:**  
  - For two donors: `normalize(v₁ + v₂)` — the internal angle bisector  
  - For three or more donors: practically equivalent to *average*  
- **When to use:** Especially useful for bidentate ligands.  
- **Pros:** Physically intuitive for bidentate geometries.  
- **Caution:** If `v₁ ≈ −v₂`, the axis becomes ill-defined (like *average*).  

### 2. Normal
- **Axis:** Normal vector to the least-squares plane through the donor set (e.g., the **PC3** direction from PCA/SVD).  
- **When to use:** For planar donor arrangements (e.g., tridentates), where the perpendicular axis is meaningful.  
- **Pros:** Captures the out-of-plane direction of planar ligands.  
- **Caution:** For only 2 donors or collinear donors, the plane is ill-defined → fallback to *average* or *bisector* is recommended.  

### 3. PCA
- **Axis:** Principal component of **maximum variance** in the donor distribution (usually PC1).  
- **When to use:** For elongated, chain-like, or non-planar donor arrays.  
- **Pros:** Captures the dominant elongation axis of donor geometry.  
- **Caution:** For planar sets, the *normal* (PC3) may be more intuitive.  

### 4. Centroid
- **Axis:** `normalize(mean(Dᵢ) − M)` — vector from the metal center *M* to the centroid of the donor coordinates  
- **When to use:** When donor positions (including distances) should influence the axis.  
- **Pros:** Naturally incorporates distance effects; geometric center-based.  
- **Caution:** A single distant donor can dominate the direction.  

### 5. Average
- **Axis:** `normalize(Σ vᵢ)` — normalized vector sum of all donor directions  
- **When to use:** To equally account for multiple donor directions.  
- **Pros:** Straightforward; reflects all donors equally.  
- **Caution:** Nearly opposite donors may cancel out, making the axis unstable.  

### 6. First
- **Axis:** `v₁` (the first donor direction itself)  
- **When to use:** When only one donor should define the axis (e.g., monodentate, or a specific donor chosen as reference).  
- **Pros:** Simple and intuitive.  
- **Caution:** If multiple donors exist, the other directions are not reflected in the axis definition.  

### Practical Guide
- **Bidentate:** `bisector` (recommended), or `average`  
- **Planar tridentate (or similar):** `normal` is most intuitive  
- **Chain-like / elongated donor arrays:** `pca`  
- **Single donor focus:** `first`  
- **Equal weighting of all donors:** `average`  
- **Distance-sensitive cases:** `centroid`  

</details>

---

## About
This project was created by **Boseok Hong** (**Department of Chemistry of the f-elements, Institute of Resource Ecology, HZDR**).  
For inquiries, please contact [bshong66@gmail.com](mailto:bshong66@gmail.com) or [b.hong@hzdr.de](mailto:b.hong@hzdr.de).

2024.12. Boseok Hong  

---

## Acknowledgements
 This project includes code or concepts from the work of Sebastian Dechert for implementing a 3D molecular scatter plot, licensed under the BSD 3-Clause License. The original code is available [here](https://github.com/radi0sus/xyz2tab).
