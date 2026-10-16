# 3D PCS Plot

**View → 3D PCS Plot** opens the **PCS Field Viewer — Controls** window. Its
**Open / Refresh Viewer** action builds an interactive PyVista scene containing
PCS isosurfaces, optional slices, and a molecular overlay.

This viewer requires PyVista and VTK. If PyVista cannot be imported, PCS
Analyzer reports the import error and suggests installing `pyvista` and `vtk`.
Availability therefore depends on the source environment or packaged build.

## Field and grid generation

The viewer samples the current analytic PCS expression on
a uniform Cartesian grid around the visible structure. Coordinates are in the
current metal-centred, rotated tensor frame; Δχ_ax and Δχ_rh come from the
current tensor inputs. **Prediction model: Axial** excludes the rhombic term;
**Full tensor (Axial + Rhombic)** includes it only when **Enable Δχ_rh** is on.
See [Prediction policy](../pcs/axial-rhombic.md#active-prediction-model).

The result is a visualization grid, not a new fit or an independently refined
field. PyVista extracts surfaces between sampled voxels, so apparent surface
smoothness and position depend on the chosen grid spacing. Do not interpret
the display resolution as additional experimental or numerical precision.

### Grid Parameters

| Control | Effect on the sampled field |
| --- | --- |
| **Grid spacing (Å)** | Distance between grid samples; smaller values increase spatial sampling and cost |
| **Padding (Å): Auto/Manual** | Extent added around the molecular coordinate bounds; Auto estimates an extent from the tensor and lowest requested level |
| **Metal mask r (Å)** | Sets samples nearer than this metal-centred radius to missing values, suppressing the singular near-origin region |
| **Clip ±ppm** | Optionally clips scalar values before slices and surfaces are generated |

These controls affect the numerical display grid. They do not modify the
structure, tensor, experimental data, fitting objective, or main calculated
atom PCS values.

## Isosurfaces and slices

Each **Isosurface Levels** row specifies a positive ppm magnitude. The viewer
extracts both \(+\text{level}\) and \(-\text{level}\) surfaces with separate
**Pos** and **Neg** colors. **Style** supports **surface**, **mesh**, or
**both**, and **Opacity** applies to that level. Rows can be added or removed.

The **Isosurfaces** toggle controls all level surfaces. **Slice planes** adds
three orthogonal planes through the metal-centred origin, colored by PCS with
a scalar bar. **Slice opacity** affects only their appearance. **Grid** and
**Outline** display PyVista aids for the sampled volume.

Clipping is not merely a color-range option: because it changes stored grid
values, a clip magnitude below a requested isosurface level can remove or
alter that surface.

## Molecular overlay and appearance

**Atoms**, **H atoms**, **Bonds**, and **Labels** control the molecular
overlay. Atoms use element CPK colors, while labels use `Ref:Atom` text.
**Background**, **Bond color**, **Bond radius (Å)**, **Ambient light**, surface
colors, opacity, and mesh style are visual settings.

Bond lines are distance-based guides. **Bond tolerance** scales the sum of the
two covalent radii by `1 + tolerance`: increasing it admits longer pairs.
The default 0.05 uses a scale of 1.05. This changes displayed connectivity,
not molecular coordinates, chemical bonding assignments or PCS calculations.

The molecular overlay follows the active 3D coordinate pipeline and main
element filter, but the field viewer does not provide atom-picking or
main-table selection synchronization. It also does not render the integrated
3D viewer's reference-versus-average overlay legend.

## Camera and view preservation

Camera **Preset** choices are **Iso**, **Top**, **Front**, and **Side**.
**Projection** selects **perspective** or **orthographic**. **Apply** sends the
chosen combination to the live viewer.

**Save view** stores the complete current camera state in memory; **Restore**
reapplies it while the application is running. **Open / Refresh Viewer** also
preserves the current live camera when rebuilding an existing scene.

**Save Preset** and **Load Preset** use a workflow-specific JSON file for
grid, level, display, named-camera, and export controls. The JSON preset does
not store the structure, calculated field, or arbitrary camera position saved
by **Save view**. See [File Formats](../reference/file-formats.md#json-viewer-presets).

## PNG export

**Save PNG** supports only PNG and requires the live viewer to be open.

- **Export view: current** captures the live viewer at its current window
  resolution.
- **Export view: preset** rebuilds an off-screen square scene at
  **Width (inch) × DPI** pixels and copies the live camera state.

**Transparent background** applies to either path. The “preset” export mode
means high-resolution off-screen rendering; it does not replace the current
camera with the named camera preset automatically.

## Memory and performance

Grid memory and work scale with
\(N_xN_yN_z\). Halving **Grid spacing (Å)** over the same volume creates
approximately eight times as many grid points, before accounting for temporary
coordinate arrays, contour meshes, and rendering objects. Large padding,
small spacing, many surface levels, and simultaneous slices can therefore use
substantial memory and make refresh or high-resolution export slow.

Start with coarser spacing and the lowest number of scientifically relevant
levels. Refine only after checking the region of interest, and remember that a
denser display grid does not improve the underlying structure, tensor, or
experimental data.
