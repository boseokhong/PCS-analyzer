'''
PCS analyzer

The following code requires the numpy, matplotlib, pandas, and openpyxl module packages to run.
A tool for PCS analysis and 2D polar contour visualization, particularly suited for small paramagnetic complexes with rotational symmetry.


v.0.1 updates
-Excel export: x, y cartesian coordinates are calculated, and each column are separated by PCS values.

v.0.2 updates
-tensor values, PCS range, and intervals can now be input for calculations.
-molar susceptibility tensor calculation

v.0.3 updates
-XYZ file import and plot, coordinate rotation, atom coordinate table added, geometrical parameter analysis plot

v.0.3.1 updates
-clicking on a atom point on the PCS plot to highlight the corresponding table entry.

v.0.4 updates
-Added 3d molecule scatter plot
-The code from the following project was referred to for implementing the 3D scatter plot.
 This file includes portions of code derived from work by Sebastian Dechert,
 licensed under the BSD 3-Clause License. See LICENSE or the project's README.md for full license details. The original code can be found at:
 <https://github.com/radi0sus/xyz2tab>

v.0.5 updates
-Reorganized GUI
-Added Mollweide projection plot

v.0.6 updates
-PCS plot: Half/Quarter view toggle function

v.0.7 updates
-bug fix

v.1.0.0 updates
-Code refactoring
-PCS fitting

v.1.0.1 updates
-Added table δ_exp data export/import

2025.08. Boseok Hong [Department of Chemistry of the f-elements, Institute of Resource Ecology, HZDR]
<bshong66@gmail.com> / <b.hong@hzdr.de>
<https://github.com/boseokhong/PCS-analyzer>
'''


from ui.components import build_app, wire

def main():
    state = build_app()
    wire(state)
    state['root'].mainloop()

if __name__ == "__main__":
    main()


'''
When we try to pick out anything by itself, we find it hitched to everything else in the universe.

—John Muir, My First Summer in the Sierra
'''