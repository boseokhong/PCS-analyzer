 # PCS analyzer

![figure_1a](https://github.com/user-attachments/assets/c9656b87-5b0e-4f2c-b010-bd238f86ab23)

The following code requires the `numpy`, `matplotlib`, `pandas`, and `openpyxl` module packages to run.
It is designed for PCS analysis and visualization of 2D polar contour plots (using only the axiality of the magnetic susceptibility tensor).
The file extension `.xlsx` should be included when exporting to an Excel file. e.g.[filename.xlsx]


 ## Version and Changelog
Current Version: `0.5`

v.0.1 updates
- Initial release
- `.xlsx` export: x, y cartesian coordinates are calculated, and each column is separated by PCS values.

v.0.2 updates
- Tensor values, PCS range, and intervals can now be inputted for calculations.
- molar susceptibility tensor calculation

v.0.3 updates
- XYZ file import and plot, coordinate rotation, atom coordinate table added, geometrical parameter analysis plot

v.0.3.1 updates
- Clicking on an atom point on the PCS plot to highlight the corresponding table entry, enhancing data visualization and interaction.

v.0.4 updates
- Added 3d molecule scatter plot
 
v.0.5 updates
- Reorganized GUI
- Added Mollweide projection plot


## About
This project was created by **Boseok Hong**, affiliated with **Department of Chemistry of the f-elements, Institute of Resource Ecology, HZDR**.  
For inquiries, please contact [bshong66@gmail.com](bshong66@gmail.com) or [b.hong@hzdr.de](b.hong@hzdr.de).

2024.11. Boseok Hong


 ## Acknowledgements
 This project includes code or concepts from the work of Sebastian Dechert for implementing a 3D molecular scatter plot, licensed under the BSD 3-Clause License. The original code is available [here](https://github.com/radi0sus/xyz2tab).
