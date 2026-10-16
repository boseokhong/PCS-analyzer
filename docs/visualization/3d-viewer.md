# 3D Structure Viewer

Use **View → 3D Structure** or the **3D Structure** button to inspect molecular
geometry, calculated PCS colors and atom selections in the PyVista viewer.
It follows the active PCS document and the main application's structure controls.

## Open and navigate

You can open the viewer before loading a structure. Load a structure in PCS
Analyzer to display it; the first visible structure is fitted automatically.
If the view is empty after loading, check the main element and ligand filters.
Opening the viewer again brings the existing window forward.

Drag in the structure view to rotate the camera, use the mouse wheel to zoom,
and use **Camera → Fit to molecule** to bring the visible structure into view.
Hover over toolbar icons for their descriptions. The **Sidebar** toggle opens
or hides the permanent **Display** and **Figure** tabs. You can also open either
page from the viewer's **View** menu. The sidebar has a fixed width; its contents
scroll vertically.

Click an atom to select its **Ref** in the main table. Selecting a table row
highlights that atom when it is visible. The viewer highlights the primary
selected Ref, rather than the entire regression subset. Selection and ordinary
updates retain the camera.

## Display options

Use the toolbar's **Color mode:** selector for **Element** (CPK colors) or
**PCS** (calculated shifts in ppm). The adjacent **Atom labels** icon toggles
labels. Hydrogen and other element visibility follow the main application filters.

In the **Display** tab, expand a group to reach its controls. Reset icons restore
that group's options. Collapsing a group retains its values.

| Control | Use |
| --- | --- |
| Preset | Basic restores default appearance; Figure uses smoother, higher-detail surfaces and cylinder bonds; Custom retains individual choices. |
| Background | Choose a color from the swatch or enter a HEX code. |
| Atom size | Adjust atom-marker size; shared with the main application. |
| Bond thickness | Adjust Line bonds. Thickness follows zoom relative to the first fitted view. |
| Atoms | Set finish, detail and smooth shading. |
| Bonds | Set distance tolerance, Line/Cylinder display, finish and color. Cylinder mode uses Radius instead of line thickness. |
| Lighting | Adjust illumination, reflections and anti-aliasing. Auto uses the selected finish. |

Color controls provide a swatch and HEX field. Click the swatch to choose a
color, or enter a code such as `#808080`. Invalid entries and cancelled color
choices keep the previous value.

Bond tolerance determines which visible atom pairs receive distance-based
guides; these guides do not indicate bond order. Bond color, thickness, radius
and finish change their appearance without changing coordinates or PCS values.
The Figure appearance preset does not set export dimensions or DPI.

### PCS colors and legend

In **Color mode: PCS**, expand **Display → PCS colors**.

| Control | Use |
| --- | --- |
| Range | Auto follows visible PCS values. Manual enables Min/Max in ppm; values beyond the limits use endpoint colors. |
| Centre at zero | Use symmetric Auto limits around zero; enabled by default. |
| Palette / Reverse | Choose a gradient and its direction. The default is RdBu_r. |
| Missing PCS | Choose the color for atoms without calculated PCS; gray by default. |
| Legend | Show or hide the PCS color bar, tick labels and PCS (ppm) title. |
| Tick step | Auto chooses readable intervals; enter a positive interval in ppm for fixed spacing. Very dense requests fall back to Auto. |
| Legend font | Auto follows the plot fonts; set a point size to override it. |

These options change the color display, not calculated PCS. If no visible atom
has a calculated PCS value, the default display uses element colors and no PCS
color bar. Averaged and reference markers retain their distinct representation.

## Camera and axes

| Camera action | Use |
| --- | --- |
| Fit to molecule | Fit visible working/reference geometry while keeping view direction and projection. |
| Reset camera | Restore the initial camera and fit the current structure. |
| Projection | Choose Perspective or Orthographic. |
| Standard views | Look from +X, −X, +Y, −Y, +Z, −Z or the Isometric direction. |

Camera movement changes your viewpoint, not molecular coordinates or tensor
orientation. A signed standard view looks from that side of the working frame
towards the molecule. Camera actions are unavailable without visible geometry.

In **Display → Axes**, choose **Metal centre**, **Screen corner**, **Both** or
**Hidden**. Metal-centre controls set arrow length, thickness, tip shape and
labels. Corner controls set position, size and labels; the marker follows camera
rotation. **X / Y / Z** provides individual visibility and color controls for
both placements. Axis label sizes use renderer pixels; Figure output label sizes
use points.

## Coordinates and averaging

The view uses the real working structure, including the active metal centre,
applied fit override, main X/Y/Z rotations and visibility filters. Coordinates
are displayed in the metal-centred working frame. See
[Fitting Overview](../fitting/overview.md#running-and-applying-a-fit) for applied-fit
orientation and centre behavior.

When torsional averaging is enabled, the active averaged geometry can replace
raw positions. Averaged atoms are hollow markers. **Keep reference structure**
adds filled reference markers for moving atoms and a separate **Averaging**
legend. The reference overlay is visual only and adds no observations or fitting rows.

Local CH₃/CF₃ pseudo atoms used by the table and fitting are not substituted into
this real-structure view. See
[Symmetry and Torsional Ensemble Averaging](../structures/symmetry-averaging.md#visualization-and-table-representation).

## Figure and export

Choose **Figure output** for an image prepared at a chosen size and DPI, or
**Viewport capture** for the current view. Both are available in **File** and
the toolbar's **Export** menu.

### Prepare a figure

Open the **Figure** tab and set **Width**, **Height**, **Units** (mm, in or px),
**DPI** and **Background**. Defaults are **180 × 135 mm**, **300 DPI**, **White**.
**Viewport** uses the current display background; **Transparent** removes it.
With pixel units, changing DPI changes physical size rather than pixel dimensions.

Expand **Style** to set margins, label size and line width. Explicit label/line
sizes are in points. Auto scales Line bonds to output magnification and PCS
legend typography to output DPI; other annotations retain their existing sizing
behavior. Check the preview at the intended output size.

Expand **Include** to exclude axes, atom labels, PCS/Averaging legends,
selection highlighting or reference structure from the output. These controls
only include layers already visible in the view. All are enabled by default.
Each section has its own reset; the footer reset restores all Figure options.

Use **Preview** to inspect a snapshot, then **Save** to choose PNG, PDF (image)
or SVG (image). Regenerate the preview after changing the camera or settings.
Output does not resize the live view. A different aspect ratio or large margin
can crop the structure, so check framing before saving.

### Choose an output format

| Output | Contents |
| --- | --- |
| Figure PNG | Prepared image with the selected size, DPI, margins and layers. |
| Figure PDF/SVG (image) | The same prepared image embedded in a document at its physical size. |
| Viewport PNG | Current viewport at its current pixel size, including visible annotations. |
| Viewport PDF/SVG (experimental) | Graphics export that may contain rasterized geometry or annotations. Transparent layers are unsupported. |

!!! note "PDF/SVG and vector editing"
    Figure PDF/SVG contain an embedded image; atoms, bonds and labels are not
    editable vector objects. Experimental viewport PDF/SVG are also not
    guaranteed to be purely vector. Inspect the saved file before publication.

PDF Figure output requires **White** or **Viewport** background; SVG supports
**Transparent**. Figure output supports DPI from 72 to 1200, at most 16 million
pixels and 8192 pixels per side. The panel shows output dimensions and explains
invalid requests. If the document changes during an export dialog, reopen export.

## Save and reuse visualization settings

Choose **File → Visualization settings → Save… / Load…** to reuse appearance,
background, PCS colors, axes and Figure options. The same menu is available
through **Export**. These JSON files contain display choices, not molecular
structures, calculated results or a complete session.

| State | How it is retained |
| --- | --- |
| Color mode and atom-label visibility | Saved in the PCS `.pcsp` project. |
| Atom size and bond tolerance | Shared main-application display controls; excluded from visualization JSON. |
| Appearance, PCS colors, axes and Figure options | Retained during updates to the same document; reset on document switch. Save visualization JSON to reuse them. |
| Camera, selection, window layout and expanded groups | Temporary viewer state; not saved in visualization JSON. |

Viewer-local display options are not stored in `.pcsp` or application settings.
Older visualization files without PCS color options load with their defaults.
See [Projects and Persistence](../projects.md) and
[File Formats](../reference/file-formats.md) for other saved data.

## Availability and legacy viewer

The PyVista molecular viewer replaces the legacy Matplotlib viewer used as the
main structure viewer through **v1.3.4**. The legacy viewer remains available
under **View → Open Matplotlib viewer…**; its export uses its own camera.

The default viewer requires PySide6, PyVista, VTK and pyvistaqt. If it cannot start,
or the structure exceeds the current limit of **1000 visible working/reference
markers**, PCS Analyzer offers the Matplotlib viewer. See
[Installation](../installation.md) for source dependencies.

Mixed-monitor DPI behavior, packaged execution and PDF/SVG output quality still
need broader verification. Check labels, colors, legends and framing in saved
figures. For the separate volumetric field viewer, see
[3D PCS Plot](3d-pcs-isosurface.md).
