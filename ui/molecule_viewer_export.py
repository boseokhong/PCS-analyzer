"""Child viewport export policies; native vector parity remains unverified."""

import os
import re
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from logic.molecule_viewer_protocol import matches_scene

MAX_VIEWPORT_PIXELS = 16_000_000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_VECTOR_CELLS = 250_000
MAX_VECTOR_BYTES = 64 * 1024 * 1024


def save_viewport_vector(plotter, path):
    """Experimental GL2PS output; OpenGL2 may rasterize 3D geometry itself.

    Native annotation/depth/font parity is an acceptance gate, not established by
    these basic container checks. Run only on the child render thread.
    """
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in (".pdf", ".svg"):
        raise ValueError("Viewport vector export requires a .pdf or .svg filename")
    actors = plotter.renderer.GetActors()
    actors.InitTraversal()
    cells = 0
    for _ in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        if not actor.GetVisibility():
            continue
        if actor.GetProperty().GetOpacity() != 1:
            raise ValueError("Vector export cannot preserve transparent layers. Use viewport PNG or View → Open Matplotlib viewer… for its own-camera export.")
        mapper = actor.GetMapper()
        if mapper is not None:
            mapper.Update()
            data = mapper.GetInput()
            if data is not None:
                cells += data.GetNumberOfCells()
    if cells > MAX_VECTOR_CELLS:
        raise ValueError("This scene exceeds the experimental vector export geometry budget. Use viewport PNG or View → Open Matplotlib viewer… for its own-camera export.")
    # GL2PS may create additional files; isolate all outputs in an owned directory.
    with tempfile.TemporaryDirectory(prefix=".pcs-vector-", dir=target.parent) as folder:
        temporary = Path(folder) / ("viewport" + suffix)
        plotter.save_graphic(str(temporary), title="PCS Analyzer molecular viewport",
                             raster=False, painter=False)
        if not temporary.is_file() or not 0 < temporary.stat().st_size <= MAX_VECTOR_BYTES:
            raise OSError("The viewer produced no vector file or exceeded the 64 MiB output limit")
        payload = temporary.read_bytes()
        if suffix == ".svg":
            try:
                root = ET.fromstring(payload)
            except ET.ParseError as exc:
                raise OSError("The viewer produced invalid SVG") from exc
            if root.tag != "{http://www.w3.org/2000/svg}svg":
                raise OSError("The viewer produced no SVG document")
            tags = {element.tag.rsplit("}", 1)[-1] for element in root.iter()}
            if "foreignObject" in tags:
                raise OSError("Embedded foreign content is not accepted for viewport export")
            for element in root.iter():
                if element.tag.rsplit("}", 1)[-1] == "image":
                    href = element.get("href", element.get("{http://www.w3.org/1999/xlink}href", ""))
                    if not href.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
                        raise OSError("SVG images must be embedded PNG/JPEG data, not external files")
            if not tags.intersection({"path", "polygon", "polyline", "line", "circle", "ellipse", "rect", "image"}):
                raise OSError("The viewer produced no drawing content")
            # OpenGL2 may rasterize all 3D props even with raster=False.
            # Preserve self-contained output and disclose raster content.
            mixed = "image" in tags
        else:
            # Conservative checks for GL2PS output; this is not a general PDF parser.
            if not payload.startswith(b"%PDF-") or not payload.rstrip().endswith(b"%%EOF"):
                raise OSError("The viewer produced invalid PDF")
            mixed = bool(re.search(rb"/Subtype\s*/Image\b", payload))
        os.replace(temporary, target)
    return mixed


def save_viewport_png(plotter, path):
    """Capture current framebuffer at scale one, then replace the target atomically.

    Call on the child Qt/render thread. Do not reset the camera, resize the window
    or apply a DPI multiplier. A file failure must leave the old destination intact.
    """
    target = Path(path)
    if target.suffix.lower() != ".png":
        raise ValueError("Viewport export requires a .png filename")
    width, height = plotter.window_size
    if (int(width) != width or int(height) != height or width <= 0 or height <= 0
            or width * height > MAX_VIEWPORT_PIXELS):
        raise ValueError("Viewport export supports at most 16 million pixels")
    # Same-directory replacement avoids copying across volumes; close the temp
    # handle before PyVista writes to it, including on Windows.
    with tempfile.NamedTemporaryFile(prefix=".pcs-viewport-", suffix=".png",
                                     dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        plotter.screenshot(str(temporary), transparent_background=False,
                           return_img=False, scale=1)
        with temporary.open("rb") as stream:
            if stream.read(8) != PNG_SIGNATURE or temporary.stat().st_size <= 8:
                raise OSError("The viewer did not produce a PNG image")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return int(width), int(height)


def export_viewport_dialog(window, plotter, current_scene, qt_widgets, format="png"):
    """Own the native save dialog on the child thread; scene updates may continue."""
    identity = dict(current_scene())
    identity.setdefault("scene_generation", 0)
    dialog = None
    try:
        if format not in ("png", "pdf", "svg"):
            raise ValueError("Unsupported viewport export format")
        label = format.upper()
        dialog = qt_widgets.QFileDialog(window, "Save viewport " + label)
        dialog.setAcceptMode(qt_widgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter(f"{label} {'image' if format == 'png' else 'graphic'} (*.{format})")
        dialog.setDefaultSuffix(format)
        dialog.selectFile("molecule." + format)
        if not dialog.exec() or not window.isVisible():
            return
        path = dialog.selectedFiles()[0]
        if not matches_scene(identity, current_scene()):
            raise ValueError("The document or scene changed. Reopen viewport export for the current view.")
        if format == "png":
            width, height = save_viewport_png(plotter, path)
            message = f"Saved viewport PNG ({width} × {height} pixels):\n{path}"
        else:
            if Path(path).suffix.lower() != "." + format:
                raise ValueError(f"Choose a .{format} filename for this export")
            mixed = save_viewport_vector(plotter, path)
            message = (f"Saved experimental GL2PS viewport {label}:\n{path}\n"
                       "Check labels, legend, fonts and overlapping geometry before use.")
            if mixed:
                message += "\nThe file includes raster elements and is not purely vector. Structure geometry may also be rasterized."
        qt_widgets.QMessageBox.information(window, "Molecular 3D",
            message)
    except Exception as exc:
        # Recoverable file errors do not close the viewer or change host state.
        if window.isVisible():
            qt_widgets.QMessageBox.warning(window, "Molecular 3D", f"Save failed:\n{str(exc)[:768]}")
    finally:
        if dialog is not None:
            try:
                dialog.deleteLater()
            except RuntimeError:
                # Qt may already have destroyed the dialog with its parent.
                pass
