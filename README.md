PCS plot analyzer

The following code requires the numpy, matplotlib, pandas, and openpyxl module packages to run.
It is designed for PCS analysis and visualization of 2D polar contour plots (using only the axiality of the magnetic susceptibility tensor).
The file extension ".xlsx" should be included when exporting to an Excel file. e.g.[filename.xlsx]


v.0.1 updates
Excel export: x, y cartesian coordinates are calculated, and each column is separated by PCS values.

v.0.2 updates
-Tensor values, PCS range, and intervals can now be inputted for calculations.
-molar susceptibility tensor calculation

v.0.3 updates
-XYZ file import and plot, coordinate rotation, atom coordinate table added, geometrical parameter analysis plot

v.0.3.1 updates
-clicking on an atom point on the PCS plot to highlight the corresponding table entry, enhancing data visualization and interaction.

v.0.4 updates
-Added 3d molecule scatter plot
-The code from the following project was referred to for implementing the 3D scatter plot.
 https://github.com/radi0sus/xyz2tab

v.0.5 updates
-Reorganized GUI
-Added Mollweide projection plot


2024.11. Boseok Hong [Department of Chemistry of the f-elements, Institute of Resource Ecology, HZDR]
<bshong66@gmail.com> / <b.hong@hzdr.de>
