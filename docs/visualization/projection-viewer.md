# Projection Viewer

Open **View → Projection** or **Open viewers → Projection** to display the
current PCS angular field at a fixed radius. The viewer uses
the current Δχ_ax, Δχ_rh, tensor-frame rotation, effective atom coordinates,
and main-window element filter.

## Projection modes

The **Projection mode:** control provides two representations of the same
angular field:

- **φ / cos(θ)** is a rectangular map with \(\phi\) from \(-\pi\) to \(\pi\)
  and \(\cos\theta\) from −1 to 1.
- **Mollweide** maps longitude \(\phi\) and latitude
  \(\pi/2-\theta\) onto a Mollweide globe projection.

Both modes show a filled red/blue PCS map, white contour lines, and a
**PCS (ppm)** color bar. The UI calls the first mode **φ / cos(θ)**; it is a
rectangular angular projection, not a plot of molecular Cartesian coordinates.

## Fixed radius

**Fixed r [Å]:** is the radius used to calculate every background grid point.
The background PCS therefore scales with \(r^{-3}\). A blank, nonnumeric, or
non-positive value falls back to 10 Å.

The fixed radius is a scientific input to this projection map, not merely a
zoom control. Atom markers, however, are placed from their angular direction
only; their actual distance from the metal does not move them radially within
the map. The projection is therefore useful for orientation, not for reading
an atom's complete distance-dependent PCS directly.

See [Axial and Rhombic PCS](../pcs/axial-rhombic.md) and
[Tensor Conventions](../pcs/tensor-conventions.md) for the underlying terms.

The projection follows **Prediction model**: **Axial** excludes Δχ_rh, while
**Full tensor (Axial + Rhombic)** includes it only when **Enable Δχ_rh** is on.
A stored nonzero Δχ_rh alone does not activate a rhombic field. See
[Prediction policy](../pcs/axial-rhombic.md#active-prediction-model).

## Atoms, filtering, and selection

**Show atoms** overlays the currently effective, rotated atom set. Marker
colors follow element CPK colors. The main **Select elements to display**
checklist applies first; **Show H** provides an additional hydrogen-only
filter inside the viewer.

The current Projection Viewer has no atom-label control. Click a marker to
select and reveal its Ref row in the main table. Selection is one-way in this
viewer: selecting a main-table row does not currently add a projection-marker
highlight.

Symmetry pseudo atoms and torsionally averaged positions can appear because
the viewer uses the effective main-workflow coordinates. Their marker
positions remain a geometric display; averaging calculations follow the
rules in [Symmetry and Torsional Ensemble Averaging](../structures/symmetry-averaging.md).

## Refresh and export

Changing the projection mode or committing **Fixed r [Å]** redraws the map.
Use **⟳ Refresh** after changing relevant main-window state if the open view
has not otherwise redrawn.

**💾 Export Figure** saves the current figure as PNG, PDF, or SVG at 600 dpi.
The Matplotlib toolbar also provides its normal pan, zoom, and save controls.
The export contains the rendered map and optional atom markers; it does not
write a numerical projection grid.
