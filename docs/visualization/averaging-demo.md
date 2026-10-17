# Practice: Averaging and PCS visualization

The existing `examples/documentation_demo/Nd_iPr2BA3_demo/Nd_iPr2BA3_axial.pcsp`
is suitable for molecular display and averaging practice. Keep its original
experimental assignment caveats: disabling averaging does not turn a group
observation into independent measurements of all member atoms.
Use **File → Open PCS Project...** for that project; see
[finding the examples](../examples.md#find-and-open-the-files) for its location.

## Compare views using the Nd project

Open the project with its original tensor/frame. Compare methyl symmetry averaging
on/off and phenyl rotational averaging on/off. Inspect Ref membership and pseudo-
or group targets. Keep the tensor fixed when comparing the [2D polar plot](polar-plot.md),
[Projection Viewer](projection-viewer.md) and [3D molecular viewer](3d-viewer.md).
Choose PCS coloring and show hydrogens in the molecular viewer to inspect the
scientific color scale; color changes are not numerical fits.

In [3D PCS Controls](3d-pcs-isosurface.md), compare positive/negative isosurfaces
and slice planes at the same levels. Change grid spacing/padding deliberately;
the sampled analytic field is a display calculation, not extra experimental
precision. Mask the near-metal singular region. Save review settings and exported
figures separately from the canonical input project.

## Additional CF₃/tBu fixture

`examples/averaging_visualization_demo/groups.xyz` supplies synthetic disconnected
fragments with three methyl groups, one CF₃ group and one tBu centre. These are
topology/detection examples, not optimized chemistry or experimental PCS data.
Ref 2 is the tBu centre, Refs 4–6 are its methyl carbons, and Ref 17 is the
CF₃ carbon. Nd at Ref 1 is a display reference, not a validated coordination site.
Click **+** in the document tab strip, then use **File → Load XYZ...** to load
`groups.xyz` in that new document
to inspect grouping controls; do not assign it the Nd example's experimental data.

Raw Ref IDs and generated pseudo-atom indices are distinct. Group detection
does not by itself establish rapid exchange, equivalence or a physical tensor.
