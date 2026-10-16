"""Figure PNG preparation; independent output render window, no live resizing."""

import math
import base64
import os
from pathlib import Path
import struct
import tempfile
import zlib

from logic.molecule_viewer_protocol import matches_scene
from ui.molecule_viewer_export import MAX_VIEWPORT_PIXELS, PNG_SIGNATURE

LAYERS = (("world_axes", "Metal-centre axes"), ("corner_axes", "Corner axes"),
          ("labels", "Atom labels"), ("pcs_legend", "PCS legend"),
          ("averaging_legend", "Averaging legend"), ("selection", "Selection highlight"),
          ("reference", "Reference structure"))


def figure_layout(size, dpi, options):
    """Output-only margin and point-to-pixel conversion, never Qt/Tk scaling."""
    values = [options.get(key, 0.) for key in ("margin_mm", "font_pt", "line_pt")]
    if any(isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Figure margin, font and line size must be finite and nonnegative")
    margin, font, line = values
    if margin > 100 or font > 72 or line > 12:
        raise ValueError("Figure margin/font/line size exceeds the supported range")
    border = int(margin * dpi / 25.4 + .5)
    if 2 * border >= min(size):
        raise ValueError("Figure margin leaves no room for the structure")
    return border, max(1, int(font * dpi / 72 + .5)) if font else 0, line * dpi / 72 if line else 0.


def inset_viewport(viewport, size, border):
    x, y = border / size[0], border / size[1]
    return tuple(offset + component * (1 - 2 * offset)
                 for component, offset in zip(viewport, (x, y, x, y)))


def figure_layer_policy(renderer, options):
    excluded, scales, text = set(), {}, {}
    groups = {"world_axes": renderer.axes.actors, "labels": renderer.annotations,
              "pcs_legend": [renderer.scalar_bar, getattr(renderer, "scalar_bar_outline", None), *getattr(renderer, "scalar_bar_text", [])], "averaging_legend": [renderer.averaging_legend],
              "selection": [renderer.highlight, renderer.average_highlight],
              "reference": [renderer.reference_actor]}
    for key, actors in groups.items():
        if not options.get("include_" + key, True):
            excluded.update(actor.GetAddressAsString("") for actor in actors if actor is not None)
    if not options.get("include_selection", True):
        for ref in renderer.averaged:
            scales[renderer.atoms[ref].GetAddressAsString("")] = renderer.radius_by_ref[ref] * 1.18 ** .5
    if not options.get("include_reference", True) and renderer.averaging_legend is not None:
        from logic.molecule_viewer_snapshot import averaging_legend_entries
        entries = [entry for entry in averaging_legend_entries(renderer.scene) if entry != "Reference"]
        address = renderer.averaging_legend.GetAddressAsString("")
        if not entries:
            excluded.add(address)
        else:
            text[address] = "Averaging\n" + "\n".join("○ " + entry for entry in entries)
    policy = dict(excluded=excluded, normal_scales=scales, text=text,
        neutral_text={actor.GetAddressAsString("") for actor in [*renderer.annotations, renderer.scalar_bar, renderer.averaging_legend] if actor is not None})
    presentation = renderer.scene.get("presentation")
    from logic.molecule_viewer_pcs import PcsColors
    colors = getattr(renderer, "pcs_colors", None)
    legend_font = colors.options()["legend_font"] if isinstance(colors, PcsColors) else 0
    if renderer.scalar_bar is not None and isinstance(presentation, dict):
        policy["pcs_font_points"] = {"Title": legend_font or presentation["label_size"],
                                    "Label": legend_font or presentation["tick_size"],
                                    "Annotation": legend_font or presentation["tick_size"]}
        outline = getattr(renderer, "scalar_bar_outline", None)
        if outline is not None:
            policy["pcs_outline"] = outline.GetAddressAsString("")
        actors = getattr(renderer, "scalar_bar_text", [])
        policy["pcs_text"] = {actor.GetAddressAsString(""): legend_font or presentation["label_size" if index == len(actors) - 1 else "tick_size"] for index, actor in enumerate(actors)}
        policy["neutral_text"].update(policy["pcs_text"])
    from ui.molecule_viewer_appearance import BondLineSizing
    sizing = getattr(renderer, "bond_sizing", None)
    if isinstance(sizing, BondLineSizing) and renderer.bonds is not None and renderer.appearance.bond_mode == "Line":
        diameter = sizing.diameter(renderer.appearance.line_width)
        if diameter is not None:
            policy["bond_line"] = (renderer.bonds.GetAddressAsString(""), diameter)
    return policy


def _text_copy(source, font, color=None):
    prop = source.NewInstance()
    prop.ShallowCopy(source)
    if font:
        prop.SetFontSize(font)
    if color is not None:
        prop.SetColor(*color)
    return prop


def style_figure_prop(source, clone, font, line, color=None, scalar_fonts=None):
    """All editable output properties/pipelines are detached from live actors."""
    if source.IsA("vtkActor"):
        prop = source.GetProperty().NewInstance()
        prop.DeepCopy(source.GetProperty())
        if line:
            prop.SetLineWidth(line)
        clone.SetProperty(prop)
    if not font and color is None and not scalar_fonts:
        return
    if source.IsA("vtkTextActor"):
        clone.SetTextProperty(_text_copy(source.GetTextProperty(), font, color))
        clone.SetTextScaleModeToNone()
    elif source.IsA("vtkScalarBarActor"):
        for field in ("Title", "Label", "Annotation"):
            original = getattr(source, f"Get{field}TextProperty")()
            size = font or (scalar_fonts or {}).get(field, 0)
            getattr(clone, f"Set{field}TextProperty")(_text_copy(original, size, color))
        clone.SetUnconstrainedFontSize(True)
    elif source.IsA("vtkAxesActor"):
        for axis in "XYZ":
            original = getattr(source, f"Get{axis}AxisCaptionActor2D")()
            caption = getattr(clone, f"Get{axis}AxisCaptionActor2D")()
            caption.SetCaptionTextProperty(_text_copy(original.GetCaptionTextProperty(), font))
            caption.GetTextActor().SetTextScaleModeToNone()
    elif source.IsA("vtkActor2D") and source.GetMapper() is not None:
        mapper = source.GetMapper()
        if not mapper.IsA("vtkLabelPlacementMapper"):
            return
        from vtkmodules.vtkRenderingLabel import vtkPointSetToLabelHierarchy
        original = mapper.GetInputAlgorithm()
        if original is None or not original.IsA("vtkPointSetToLabelHierarchy"):
            raise ValueError("This label pipeline does not support output font overrides. Use Auto font size.")
        original.Update()
        hierarchy = vtkPointSetToLabelHierarchy()
        hierarchy.SetInputData(original.GetInput())
        hierarchy.SetLabelArrayName(original.GetLabelArrayName())
        hierarchy.SetTextProperty(_text_copy(original.GetTextProperty(), font, color))
        for field in ("TargetLabelCount", "MaximumDepth"):
            getattr(hierarchy, "Set" + field)(getattr(original, "Get" + field)())
        copied_mapper = mapper.NewInstance()
        copied_mapper.SetInputConnection(hierarchy.GetOutputPort())
        for field in ("Shape", "Style", "Margin", "BackgroundOpacity", "PlaceAllLabels", "UseDepthBuffer", "MaximumLabelFraction"):
            if hasattr(mapper, "Get" + field) and hasattr(copied_mapper, "Set" + field):
                getattr(copied_mapper, "Set" + field)(getattr(mapper, "Get" + field)())
        copied_mapper.SetBackgroundColor(*mapper.GetBackgroundColor())
        clone.SetMapper(copied_mapper)


def figure_pixels(width, height, unit, dpi):
    if unit not in ("mm", "in", "px"):
        raise ValueError("Unknown figure unit")
    if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in (width, height, dpi)):
        raise ValueError("Figure size and DPI must be positive finite numbers")
    if not 72 <= dpi <= 1200:
        raise ValueError("Figure DPI must be between 72 and 1200")
    factor = {"mm": dpi / 25.4, "in": dpi, "px": 1}[unit]
    size = tuple(max(1, int(value * factor + .5)) for value in (width, height))
    if size[0] * size[1] > MAX_VIEWPORT_PIXELS or max(size) > 8192:
        raise ValueError("Figure PNG supports at most 16 million pixels and 8192 pixels per side")
    return size


def png_with_dpi(payload, size, dpi, transparent):
    """Validate generated PNG chunks and add physical resolution metadata."""
    if not payload.startswith(PNG_SIGNATURE):
        raise OSError("The renderer produced invalid PNG")
    offset, chunks, seen = 8, [], []
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise OSError("Truncated PNG")
        length = struct.unpack(">I", payload[offset:offset + 4])[0]
        end = offset + length + 12
        if end > len(payload):
            raise OSError("Truncated PNG chunk")
        tag = payload[offset + 4:offset + 8]
        data = payload[offset + 8:end - 4]
        if zlib.crc32(tag + data) & 0xffffffff != struct.unpack(">I", payload[end - 4:end])[0]:
            raise OSError("Invalid PNG checksum")
        seen.append(tag)
        if tag == b"IHDR":
            if length != 13 or struct.unpack(">II", data[:8]) != size:
                raise OSError("Renderer output size differs from the requested figure size")
            if transparent and data[9] != 6:
                raise OSError("The renderer did not provide an RGBA PNG")
        if tag != b"pHYs":
            chunks.append(payload[offset:end])
        offset = end
    if (not seen or seen[0] != b"IHDR" or seen[-1] != b"IEND" or b"IDAT" not in seen
            or seen.count(b"IHDR") != 1 or seen.count(b"IEND") != 1):
        raise OSError("Incomplete PNG")
    ppm = int(dpi / .0254 + .5)
    data = struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", len(data)) + b"pHYs" + data + struct.pack(">I", zlib.crc32(b"pHYs" + data) & 0xffffffff)
    return PNG_SIGNATURE + chunks[0] + chunk + b"".join(chunks[1:])


def figure_background(options):
    if options.get("background") == "Viewport":
        from ui.molecule_viewer_color import rgb_code
        code = rgb_code(options["viewport_background"])
        return tuple(int(code[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return (1., 1., 1.)


def capture_figure(plotter, path, size, transparent, options=None, policy=None):
    from vtkmodules.vtkRenderingCore import vtkRenderWindow, vtkRenderer, vtkWindowToImageFilter
    from vtkmodules.vtkIOImage import vtkPNGWriter

    options, policy = options or {}, policy or {}
    border, font, line = figure_layout(size, options.get("dpi", 300), options)
    excluded = policy.get("excluded", set())
    output = vtkRenderWindow()
    try:
        output.SetOffScreenRendering(1)
        output.SetSize(*size)
        output.SetAlphaBitPlanes(1)
        output.SetDPI(72)  # Font overrides were already converted to output pixels.
        output.SetMultiSamples(plotter.render_window.GetMultiSamples())
        output.SetNumberOfLayers(plotter.render_window.GetNumberOfLayers() + bool(border))
        if border:
            background = vtkRenderer()
            background.SetLayer(0)
            background.SetBackground(*figure_background(options))
            background.SetBackgroundAlpha(0 if transparent else 1)
            output.AddRenderer(background)
        legend_copies = []
        sources = plotter.render_window.GetRenderers()
        for index in range(sources.GetNumberOfItems()):
            source = sources.GetItemAsObject(index)
            if not source.GetDraw():
                continue
            renderer = vtkRenderer()
            output.AddRenderer(renderer)
            renderer.SetLayer(source.GetLayer() + bool(border))
            renderer.SetViewport(*inset_viewport(source.GetViewport(), size, border))
            renderer.SetPreserveColorBuffer(True if border else source.GetPreserveColorBuffer())
            renderer.SetPreserveDepthBuffer(source.GetPreserveDepthBuffer())
            renderer.SetBackground(*figure_background(options))
            renderer.SetBackgroundAlpha(0 if transparent else 1)
            renderer.SetUseFXAA(source.GetUseFXAA())
            renderer.SetUseDepthPeeling(source.GetUseDepthPeeling())
            if hasattr(source, "GetUseOIT") and hasattr(renderer, "SetUseOIT"):
                renderer.SetUseOIT(source.GetUseOIT())
            camera = source.GetActiveCamera().NewInstance()
            camera.DeepCopy(source.GetActiveCamera())
            renderer.SetActiveCamera(camera)
            lights = source.GetLights()
            for number in range(lights.GetNumberOfItems()):
                light = lights.GetItemAsObject(number)
                clone = light.NewInstance()
                clone.DeepCopy(light)
                renderer.AddLight(clone)
            props = source.GetViewProps()
            bar_copy, outline_copy = None, None
            text_copies = []
            for number in range(props.GetNumberOfItems()):
                prop = props.GetItemAsObject(number)
                address = prop.GetAddressAsString("")
                if address in excluded or (not options.get("include_corner_axes", True) and prop.IsA("vtkAxesActor")):
                    continue
                clone = prop.NewInstance()
                if prop.IsA("vtkTextActor"):
                    # Copying its cached texture can clip newly enlarged text.
                    clone.SetInput(prop.GetInput())
                    clone.GetPositionCoordinate().SetCoordinateSystem(prop.GetPositionCoordinate().GetCoordinateSystem())
                    clone.SetPosition(*prop.GetPosition())
                    clone.SetVisibility(prop.GetVisibility())
                else:
                    clone.ShallowCopy(prop)
                if address == policy.get("pcs_outline"):
                    from ui.molecule_viewer_legend import make_pcs_outline
                    from ui.molecule_viewer_color import annotation_color
                    background = options.get("viewport_background", "#FFFFFF") if options.get("background") == "Viewport" else "#FFFFFF"
                    clone = make_pcs_outline(annotation_color(background))
                    clone.GetProperty().SetLineWidth(line or options.get("dpi", 300) / 96.)
                    outline_copy = clone
                text_color = None
                if address in policy.get("neutral_text", set()):
                    from ui.molecule_viewer_color import annotation_color
                    background = options.get("viewport_background", "#FFFFFF") if options.get("background") == "Viewport" else "#FFFFFF"
                    text_color = annotation_color(background, legend=bool(prop.IsA("vtkTextActor")))
                scalar_fonts = None
                if prop.IsA("vtkScalarBarActor") and policy.get("pcs_font_points"):
                    scalar_fonts = {key: max(1, int(pt * options.get("dpi", 300) / 72 + .5))
                                    for key, pt in policy["pcs_font_points"].items()}
                    clone.SetVerticalTitleSeparation(max(1, int(6 * options.get("dpi", 300) / 72 + .5)))
                text_font = font
                if address in policy.get("pcs_text", {}):
                    text_font = font or max(1, int(policy["pcs_text"][address] * options.get("dpi", 300) / 72 + .5))
                    clone.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                    text_copies.append(clone)
                style_figure_prop(prop, clone, text_font, line, text_color, scalar_fonts)
                if prop.IsA("vtkScalarBarActor"):
                    # VTK ShallowCopy omits these layout flags/custom ticks.
                    for field in ("BarRatio", "TitleRatio", "DrawTickLabels", "ForceVerticalTitle", "DrawFrame", "UseCustomLabels", "MaximumNumberOfColors"):
                        getattr(clone, "Set" + field)(getattr(prop, "Get" + field)())
                    clone.SetCustomLabels(prop.GetCustomLabels())
                    clone.SetTextPad(prop.GetTextPad())
                    bar_copy = clone
                bond_line = policy.get("bond_line")
                if not line and bond_line is not None and address == bond_line[0]:
                    from ui.molecule_viewer_appearance import projected_line_width
                    viewport = renderer.GetViewport()
                    height = size[1] * (viewport[3] - viewport[1])
                    width = projected_line_width(bond_line[1], camera, height, renderer.GetTiledAspectRatio())
                    if width is not None:
                        clone.GetProperty().SetLineWidth(width)
                if address in policy.get("text", {}):
                    clone.SetInput(policy["text"][address])
                if address in policy.get("normal_scales", {}):
                    clone.SetScale(policy["normal_scales"][address])
                if clone.IsA("vtkFollower"):
                    clone.SetCamera(camera)
                renderer.AddViewProp(clone)
            if bar_copy is not None and outline_copy is not None:
                legend_copies.append((bar_copy, outline_copy, renderer, text_copies))
        output.Render()
        if legend_copies:
            from ui.molecule_viewer_legend import update_pcs_outline
            for bar, outline, renderer, text in legend_copies:
                update_pcs_outline(bar, outline, renderer, text)
            output.Render()
        image = vtkWindowToImageFilter()
        image.SetInput(output)
        image.SetInputBufferTypeToRGBA()
        image.ReadFrontBufferOff()
        image.Update()
        writer = vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(image.GetOutputPort())
        writer.Write()
        if writer.GetErrorCode():
            raise OSError("The renderer could not write figure PNG")
    finally:
        output.Finalize()


def save_figure_png(plotter, path, options, policy=None):
    target = Path(path)
    if target.suffix.lower() != ".png":
        raise ValueError("Figure export requires a .png filename")
    size = figure_pixels(options["width"], options["height"], options["unit"], options["dpi"])
    figure_layout(size, options["dpi"], options)
    if options["background"] not in ("White", "Transparent", "Viewport"):
        raise ValueError("Unknown figure background")
    transparent = options["background"] == "Transparent"
    with tempfile.NamedTemporaryFile(prefix=".pcs-figure-", suffix=".png", dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        capture_figure(plotter, temporary, size, transparent, options, policy)
        if not temporary.is_file() or temporary.stat().st_size > 64 * 1024 * 1024:
            raise OSError("Figure PNG is missing or exceeds 64 MiB")
        temporary.write_bytes(png_with_dpi(temporary.read_bytes(), size, options["dpi"], transparent))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return size


def write_image_pdf(path, payload, size, dpi, gui, core):
    """One image on a custom physical page; QtGui only, no QtPrintSupport."""
    image = gui.QImage.fromData(payload, "PNG")
    if image.isNull() or (image.width(), image.height()) != size:
        raise OSError("Could not load the figure image for PDF")
    page = gui.QPageSize(core.QSizeF(*(value * 25.4 / dpi for value in size)),
                         gui.QPageSize.Unit.Millimeter, "Molecular figure",
                         gui.QPageSize.SizeMatchPolicy.ExactMatch)
    if not page.isValid():
        raise ValueError("Unsupported PDF page size")
    writer = gui.QPdfWriter(str(path))
    painter = gui.QPainter()
    try:
        writer.setResolution(int(dpi))
        writer.setTitle("PCS Analyzer molecular figure (embedded image)")
        if not writer.setPageSize(page) or not writer.setPageMargins(core.QMarginsF(0, 0, 0, 0)):
            raise OSError("Could not configure the PDF page")
        if not painter.begin(writer):
            raise OSError("Could not start PDF writing")
        painter.setRenderHint(gui.QPainter.RenderHint.LosslessImageRendering, True)
        painter.drawImage(core.QRectF(0, 0, writer.width(), writer.height()), image)
        if not painter.end():
            raise OSError("Could not finish PDF writing")
    finally:
        if painter.isActive():
            painter.end()
        del painter
        del writer  # Release the output file before replacement on Windows.


def save_figure_image_document(plotter, path, options, policy=None, *, gui=None, core=None):
    """Explicit image-based PDF/SVG, preserving the Figure PNG pipeline."""
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in (".pdf", ".svg"):
        raise ValueError("Figure image export requires a .pdf or .svg filename")
    if suffix == ".pdf" and options["background"] == "Transparent":
        raise ValueError("Image-based PDF requires an opaque White or Viewport background. Use PNG or SVG for transparency.")
    with tempfile.TemporaryDirectory(prefix=".pcs-figure-document-", dir=target.parent) as folder:
        image_path = Path(folder) / "figure.png"
        size = save_figure_png(plotter, image_path, options, policy)
        payload = image_path.read_bytes()
        temporary = Path(folder) / ("figure" + suffix)
        dpi = options["dpi"]
        if suffix == ".svg":
            width, height = (value * 25.4 / dpi for value in size)
            encoded = base64.b64encode(payload).decode("ascii")
            temporary.write_text(
                f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
                f'width="{width:.8f}mm" height="{height:.8f}mm" viewBox="0 0 {size[0]} {size[1]}">'
                '<title>PCS Analyzer molecular figure (embedded PNG)</title>'
                '<desc>Image-based figure; structure and labels are raster, not vector.</desc>'
                f'<image width="{size[0]}" height="{size[1]}" xlink:href="data:image/png;base64,{encoded}"/>'
                '</svg>', encoding="utf-8")
        else:
            write_image_pdf(temporary, payload, size, dpi, gui, core)
        if not temporary.is_file() or not 0 < temporary.stat().st_size <= 64 * 1024 * 1024:
            raise OSError("Figure document is missing or exceeds 64 MiB")
        if suffix == ".pdf":
            data = temporary.read_bytes()
            if not data.startswith(b"%PDF-") or not data.rstrip().endswith(b"%%EOF"):
                raise OSError("The writer produced invalid PDF")
        os.replace(temporary, target)
    return size


class FigureControls:
    def __init__(self, parent, widgets, core, current_scene, plotter, *, gui=None, renderer=None):
        self.widgets, self.window = widgets, parent
        self.current_scene, self.plotter = current_scene, plotter
        self.gui, self.core, self.renderer = gui, core, renderer
        self.document_id = None
        self.widget = widgets.QWidget(parent)
        self.widget.setAttribute(core.Qt.WidgetAttribute.WA_StyledBackground, True)
        form = widgets.QFormLayout(self.widget)
        from ui.molecule_viewer_style import inspector_form, InspectorSection
        inspector_form(form, widgets)
        size_section = InspectorSection(self.widget, widgets, "Size", gui=gui, core=core, expanded=True)
        style_section = InspectorSection(self.widget, widgets, "Style", gui=gui, core=core)
        form.addRow(size_section.widget)
        form.addRow(style_section.widget)
        self.width, self.height = [widgets.QDoubleSpinBox(self.widget) for _ in range(2)]
        for control in (self.width, self.height):
            control.setRange(.001, 1_000_000)
            control.setDecimals(3)
            control.setKeyboardTracking(False)
            control.setAlignment(core.Qt.AlignmentFlag.AlignRight)
        self.unit = widgets.QComboBox(self.widget)
        self.unit.addItems(("mm", "in", "px"))
        self.unit_value = "mm"
        self.dpi = widgets.QSpinBox(self.widget)
        self.dpi.setRange(72, 1200)
        self.dpi.setKeyboardTracking(False)
        self.dpi.setAlignment(core.Qt.AlignmentFlag.AlignRight)
        self.background = widgets.QComboBox(self.widget)
        self.background.addItems(("White", "Transparent", "Viewport"))
        for label, control in (("Width", self.width), ("Height", self.height), ("Units", self.unit),
                               ("DPI", self.dpi)):
            size_section.form.addRow(label, control)
        style_section.form.addRow("Background", self.background)
        self.output_controls = {}
        for key, label, maximum, suffix in (("margin_mm", "Margin", 100, " mm"),
                ("font_pt", "Label size", 72, " pt"), ("line_pt", "Line width", 12, " pt")):
            control = widgets.QDoubleSpinBox(self.widget)
            control.setRange(0, maximum)
            control.setDecimals(2)
            control.setSingleStep(.25 if key == "line_pt" else 1.)
            control.setKeyboardTracking(False)
            control.setAlignment(core.Qt.AlignmentFlag.AlignRight)
            control.setSuffix(suffix)
            if key != "margin_mm":
                control.setSpecialValueText("Auto")
            control.setToolTip({
                "margin_mm": "Equal empty margins in millimetres, included within the output size.",
                "font_pt": "Output label size in points. Auto retains live sizes, except PCS legend text uses the host plot sizes at the output DPI. Explicit values override all supported labels.",
                "line_pt": "Output stroke width in points. Auto scales Line bonds with camera and output resolution; explicit values override strokes, not cylinder radii."
            }[key])
            style_section.form.addRow(label, control)
            control.valueChanged.connect(self.update)
            self.output_controls[key] = control
        self.layers = {}
        layers = InspectorSection(self.widget, widgets, "Include", gui=gui, core=core)
        self.sections = (size_section, style_section, layers)
        self.group_resets = []
        for section, scope in ((size_section, "size"), (style_section, "style"), (layers, "include")):
            button = section.add_reset(widgets, lambda checked=False, scope=scope: self.reset_group(scope))
            self.group_resets.append((button, section.title))
        form.addRow(layers.widget)
        for key, label in LAYERS:
            control = widgets.QCheckBox(label, self.widget)
            control.setToolTip("Include this layer in Figure output when it is visible in the viewer.")
            layers.form.addRow(control)
            control.toggled.connect(self.update)
            self.layers[key] = control
        self.summary = widgets.QLabel(self.widget)
        self.summary.setWordWrap(True)
        self.summary.setToolTip("Exports only. Included layers must also be visible. Changing proportions or margins can crop the structure.")
        actions = self.footer = widgets.QWidget(self.widget)
        footer_layout = widgets.QVBoxLayout(actions)
        footer_layout.setContentsMargins(8, 6, 8, 6)
        footer_layout.setSpacing(4)
        footer_layout.addWidget(self.summary)
        row = widgets.QHBoxLayout()
        footer_layout.addLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.preview_button = widgets.QPushButton("Preview", actions)
        self.preview_button.clicked.connect(self.preview)
        row.addWidget(self.preview_button)
        self.save = widgets.QToolButton(actions)
        self.save.setText("Save")
        self.save.setPopupMode(widgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = widgets.QMenu(self.window)
        menu.setToolTipsVisible(True)
        self.png_action = menu.addAction("PNG…")
        self.png_action.triggered.connect(self.export)
        self.document_buttons = {}
        for format in ("pdf", "svg"):
            action = menu.addAction(f"{format.upper()} (image)…")
            action.setToolTip("Embedded rendered image, not editable vector objects. PDF requires an opaque background.")
            action.triggered.connect(lambda checked=False, format=format: self.export(format=format))
            self.document_buttons[format] = action
        self.save.setMenu(menu)
        row.addWidget(self.save)
        self.background.currentTextChanged.connect(self.update)
        reset = widgets.QPushButton("Reset", actions)
        self.reset_button = reset
        reset.clicked.connect(self.reset)
        row.addWidget(reset)
        row.addStretch(1)
        self.unit.currentTextChanged.connect(self.change_unit)
        for control in (self.width, self.height, self.dpi):
            control.valueChanged.connect(self.update)
        self.reset()

    def apply_icons(self, assets):
        from ui.molecule_viewer_style import compact_button
        for button, title in self.group_resets:
            compact_button(assets, button, "reset", "Reset", "Reset Figure " + title.lower() + " only.")
        compact_button(assets, self.preview_button, "preview", "Preview",
            "Preview a snapshot of Figure output. Regenerate after changing the view or settings.")
        compact_button(assets, self.save, "save", "Save",
            "Save Figure as PNG, PDF (image) or SVG (image). PDF requires an opaque background.")
        self.save.setToolButtonStyle(self.core.Qt.ToolButtonStyle.ToolButtonIconOnly
            if not self.save.icon().isNull() else self.core.Qt.ToolButtonStyle.ToolButtonTextOnly)
        compact_button(assets, self.reset_button, "reset", "Reset",
            "Reset Figure output settings only; the molecular view is retained.")

    def options(self):
        result = dict(width=self.width.value(), height=self.height.value(), unit=self.unit.currentText(),
                      dpi=self.dpi.value(), background=self.background.currentText())
        if result["background"] == "Viewport":
            result["viewport_background"] = self.renderer().appearance.background_color
        result.update({key: control.value() for key, control in self.output_controls.items()})
        result.update({"include_" + key: control.isChecked() for key, control in self.layers.items()})
        return result

    def reset(self):
        for control, setter, value in ((self.width, "setValue", 180), (self.height, "setValue", 135),
                (self.unit, "setCurrentText", "mm"), (self.dpi, "setValue", 300),
                (self.background, "setCurrentText", "White")):
            blocked = control.blockSignals(True)
            getattr(control, setter)(value)
            control.blockSignals(blocked)
        self.unit_value = "mm"
        for control in self.output_controls.values():
            blocked = control.blockSignals(True)
            control.setValue(0)
            control.blockSignals(blocked)
        for control in self.layers.values():
            blocked = control.blockSignals(True)
            control.setChecked(True)
            control.blockSignals(blocked)
        self.update()

    def reset_group(self, scope):
        if scope == "size":
            self.unit_value = "mm"
            for control, setter, value in ((self.width, "setValue", 180), (self.height, "setValue", 135),
                    (self.unit, "setCurrentText", "mm"), (self.dpi, "setValue", 300)):
                blocked = control.blockSignals(True)
                getattr(control, setter)(value)
                control.blockSignals(blocked)
        elif scope == "style":
            blocked = self.background.blockSignals(True)
            self.background.setCurrentText("White")
            self.background.blockSignals(blocked)
            for control in self.output_controls.values():
                blocked = control.blockSignals(True)
                control.setValue(0)
                control.blockSignals(blocked)
        elif scope == "include":
            for control in self.layers.values():
                blocked = control.blockSignals(True)
                control.setChecked(True)
                control.blockSignals(blocked)
        self.update()

    def document(self, document_id):
        if document_id != self.document_id:
            self.document_id = document_id
            self.reset()

    def change_unit(self, unit):
        factors = {"mm": 1., "in": 25.4, "px": 25.4 / self.dpi.value()}
        for control in (self.width, self.height):
            value = control.value() * factors[self.unit_value] / factors[unit]
            blocked = control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(blocked)
        self.unit_value = unit
        self.update()

    def update(self, *args):
        self.sections[0].summary(f"{self.width.value():g} × {self.height.value():g} {self.unit.currentText()}")
        self.sections[1].summary(self.background.currentText())
        self.sections[2].summary(f"{sum(control.isChecked() for control in self.layers.values())}/{len(self.layers)}")
        try:
            options = self.options()
            size = figure_pixels(options["width"], options["height"], options["unit"], options["dpi"])
            border, _, _ = figure_layout(size, options["dpi"], options)
            self.summary.setText(f"{size[0]} × {size[1]} px")
            self.summary.setToolTip(f"Output: {size[0]} × {size[1]} px; content inside margins: {size[0] - 2 * border} × {size[1] - 2 * border} px. Exports only; changing proportions or margins can crop the structure.")
            self.save.setEnabled(True)
            self.png_action.setEnabled(True)
            self.preview_button.setEnabled(True)
            for format, button in self.document_buttons.items():
                button.setEnabled(format != "pdf" or options["background"] != "Transparent")
        except ValueError as exc:
            self.summary.setText(str(exc))
            self.save.setEnabled(False)
            self.png_action.setEnabled(False)
            self.preview_button.setEnabled(False)
            for button in self.document_buttons.values():
                button.setEnabled(False)

    def policy(self, options):
        return figure_layer_policy(self.renderer(), options) if self.renderer is not None else {}

    def preview(self):
        dialog = None
        try:
            options = self.options()
            identity = dict(self.current_scene())
            with tempfile.TemporaryDirectory(prefix="pcs-figure-preview-") as folder:
                path = Path(folder) / "preview.png"
                size = save_figure_png(self.plotter(), path, options, self.policy(options))
                pixmap = self.gui.QPixmap()
                if not pixmap.loadFromData(path.read_bytes()):
                    raise OSError("Could not load the figure preview")
                dialog = self.widgets.QDialog(self.window)
                dialog.setWindowTitle("Figure preview — " + identity.get("document_title", "Molecular 3D"))
                dialog.setStyleSheet(self.widget.styleSheet())
                layout = self.widgets.QVBoxLayout(dialog)
                caption = self.widgets.QLabel(f"Snapshot preview: {size[0]} × {size[1]} pixels at {options['dpi']} DPI. Scaled for display; regenerate after changing the view or settings.", dialog)
                caption.setWordWrap(True)
                layout.addWidget(caption)
                picture = self.widgets.QLabel(dialog)
                picture.setPixmap(pixmap.scaled(900, 650, self.core.Qt.AspectRatioMode.KeepAspectRatio,
                                               self.core.Qt.TransformationMode.SmoothTransformation))
                scroll = self.widgets.QScrollArea(dialog)
                scroll.setWidget(picture)
                layout.addWidget(scroll)
                close = self.widgets.QPushButton("Close preview", dialog)
                close.clicked.connect(dialog.accept)
                layout.addWidget(close)
                dialog.resize(max(420, min(size[0], 940)), max(320, min(size[1] + 100, 800)))
                dialog.exec()
        except Exception as exc:
            if self.window.isVisible():
                self.widgets.QMessageBox.warning(self.window, "Molecular 3D", f"Figure preview failed:\n{str(exc)[:768]}")
        finally:
            if dialog is not None:
                try:
                    dialog.deleteLater()
                except RuntimeError:
                    pass  # Parent closure may already have destroyed its dialog.

    def export(self, checked=False, *, format="png"):
        identity = dict(self.current_scene())
        identity.setdefault("scene_generation", 0)
        options = self.options()
        dialog = self.widgets.QFileDialog(self.window, "Save figure " + format.upper())
        try:
            if format not in ("png", "pdf", "svg"):
                raise ValueError("Unsupported figure export format")
            dialog.setAcceptMode(self.widgets.QFileDialog.AcceptMode.AcceptSave)
            dialog.setNameFilter(f"{format.upper()} {'image' if format == 'png' else 'document (embedded image)'} (*.{format})")
            dialog.setDefaultSuffix(format)
            dialog.selectFile("molecule-figure." + format)
            if not dialog.exec() or not self.window.isVisible():
                return
            if not matches_scene(identity, self.current_scene()):
                raise ValueError("The document or scene changed. Reopen figure export.")
            path = dialog.selectedFiles()[0]
            if Path(path).suffix.lower() != "." + format:
                raise ValueError(f"Choose a .{format} filename for this export")
            if format == "png":
                size = save_figure_png(self.plotter(), path, options, self.policy(options))
            else:
                size = save_figure_image_document(self.plotter(), path, options, self.policy(options), gui=self.gui, core=self.core)
            description = "" if format == "png" else "\nThe structure and labels are an embedded image, not vector primitives."
            self.widgets.QMessageBox.information(self.window, "Molecular 3D",
                f"Saved figure {format.upper()} ({size[0]} × {size[1]} image pixels, {options['dpi']} DPI).\nCheck labels, axes and framing before use." + description)
        except Exception as exc:
            if self.window.isVisible():
                self.widgets.QMessageBox.warning(self.window, "Molecular 3D", f"Figure save failed:\n{str(exc)[:768]}")
        finally:
            try:
                dialog.deleteLater()
            except RuntimeError:
                pass  # Parent closure may already have destroyed its dialog.
