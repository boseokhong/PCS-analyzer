# Averaging and visualization practice

Use the supplied molecular Nd example for display/averaging comparisons:
`../documentation_demo/Nd_iPr2BA3_demo/Nd_iPr2BA3_axial.pcsp`.
The additional `groups.xyz` is a synthetic, disconnected group-detection fixture,
not a chemically optimized complex or experimental PCS dataset. It supplies
three methyl groups, one CF₃ group and one tBu centre absent from the original
Nd example's CF₃/tBu coverage. Do not interpret its connectivity as chemistry.

## Nd molecular exercise

1. Open the canonical Nd project and retain its axial tensor and metal/frame.
2. Compare methyl symmetry averaging on/off, then phenyl rotational averaging
   on/off. Inspect original atoms, pseudo/group targets and their Ref membership.
   The project has experimental group assignments; do not treat them as independent
   measurements of every underlying H when averaging is disabled.
3. Keep the same tensor and viewing direction while comparing the 2D polar plot
   and Projection Viewer. Display changes do not constitute a new fit.
4. Open the molecular 3D viewer, select PCS coloring, show hydrogens and inspect
   the PCS legend. Compare the original phenyl geometry with the averaging view.
5. Open **3D PCS Controls**, add positive/negative levels and optional slice planes.
   Compare grid spacing and padding using the same levels. This is an analytic
   sampled field; display resolution is not extra experimental precision. Mask
   the singular near-metal region and avoid unnecessarily fine grids.
6. Save a separate project/preset or exported figure to preserve review settings.
   Keep the canonical source example unchanged.

## CF₃/tBu detection exercise

Load `groups.xyz` in a fresh document and inspect the corresponding symmetry
averaging/NMR-site controls. Ref 2 is the tBu centre, Refs 4–6 its three methyl
carbons; Ref 17 is the CF₃ carbon. Nd at Ref 1 is a display reference, not a
validated coordination site. Expected group membership is a topology check,
not evidence of physical exchange or a fitted tensor.

```shell
python examples/averaging_visualization_demo/generate_demo.py --output example-output/groups-review
python examples/averaging_visualization_demo/generate_demo.py --overwrite
```

The fixture contains methyl=3, CF₃=1, tBu=1. The production main UI may present these
through its own grouping policy; do not equate generated pseudo-atom indices
with raw Ref IDs. See [public guidance](../../docs/visualization/averaging-demo.md).
