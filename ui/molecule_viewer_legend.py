"""PCS legend presentation; the host owns the lookup table and its limits."""


def style_pcs_legend(bar, presentation, foreground, pixels_per_point=1.):
    bar.SetOrientationToVertical()
    bar.SetPosition(.86, .15)
    bar.SetWidth(.035)
    bar.SetHeight(.70)
    bar.SetBarRatio(1.)
    bar.SetMaximumNumberOfColors(256)
    bar.SetTitleRatio(0.)
    bar.DrawTickLabelsOff()
    bar.SetLabelFormat("%.3g")
    bar.SetTextPositionToSucceedScalarBar()
    bar.SetVerticalTitleSeparation(8)
    bar.SetForceVerticalTitle(True)
    bar.DrawFrameOff()
    bar.SetUnconstrainedFontSize(True)
    bar.SetPickable(False)
    for field, size in (("Title", presentation["label_size"]),
                        ("Label", presentation["tick_size"]),
                        ("Annotation", presentation["tick_size"])):
        prop = getattr(bar, f"Get{field}TextProperty")()
        prop.SetColor(*foreground)
        prop.SetFontFamilyAsString("File")
        prop.SetFontFile(presentation["font_file"])
        prop.SetFontSize(max(1, int(size * pixels_per_point + .5)))
        prop.BoldOff()
        prop.ItalicOff()
        prop.ShadowOff()
    bar.SetTextPad(0)


def configure_pcs_ticks(bar, step=0.):
    from matplotlib.ticker import MaxNLocator
    from vtkmodules.vtkCommonCore import vtkDoubleArray
    low, high = bar.GetLookupTable().GetRange()
    ticks = vtkDoubleArray()
    if step:
        import math
        values = [index * step for index in range(math.ceil(low / step), math.floor(high / step) + 1)]
    else:
        values = MaxNLocator(nbins=8).tick_values(low, high)
    for tick in values:
        if low <= tick <= high:
            ticks.InsertNextValue(float(tick))
    bar.SetCustomLabels(ticks)
    bar.UseCustomLabelsOn()


def make_pcs_outline(foreground):
    from vtkmodules.vtkRenderingCore import vtkActor2D, vtkPolyDataMapper2D, vtkCoordinate
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    actor = vtkActor2D()
    mapper = vtkPolyDataMapper2D()
    coordinate = vtkCoordinate()
    coordinate.SetCoordinateSystemToNormalizedViewport()
    mapper.SetTransformCoordinate(coordinate)
    mapper.SetInputData(vtkPolyData())
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*foreground)
    actor.GetProperty().SetLineWidth(1.)
    actor.SetPickable(False)
    return actor


def make_pcs_text(bar, title, presentation, foreground, pixels_per_point):
    from vtkmodules.vtkRenderingCore import vtkTextActor
    ticks = bar.GetCustomLabels()
    actors = []
    for index in range(ticks.GetNumberOfValues() + 1):
        is_title = index == ticks.GetNumberOfValues()
        actor = vtkTextActor()
        actor.SetInput(title if is_title else format(ticks.GetValue(index), ".3g"))
        actor.SetPickable(False)
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.SetTextScaleModeToNone()
        prop = actor.GetTextProperty()
        prop.SetFontFamilyAsString("File")
        prop.SetFontFile(presentation["font_file"])
        prop.SetFontSize(max(1, int(presentation["label_size" if is_title else "tick_size"] * pixels_per_point + .5)))
        prop.SetColor(*foreground)
        prop.SetVerticalJustificationToCentered()
        if is_title:
            prop.SetOrientation(90)
            prop.SetJustificationToCentered()
        else:
            prop.SetJustificationToLeft()
        actors.append(actor)
    bar.SetTitle("")
    return actors


def update_pcs_outline(bar, outline, renderer, text=()):
    """Match VTK's actual color rectangle, excluding its labels/title."""
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    rect = [0, 0, 0, 0]
    bar.GetScalarBarRect(rect, renderer)
    x, y, width, height = rect
    viewport_width, viewport_height = renderer.GetSize()
    if min(width, height, viewport_width, viewport_height) <= 0:
        return
    points, lines = vtkPoints(), vtkCellArray()
    def segment(x1, y1, x2, y2):
        first = points.InsertNextPoint(x1 / viewport_width, y1 / viewport_height, 0)
        second = points.InsertNextPoint(x2 / viewport_width, y2 / viewport_height, 0)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(first)
        lines.InsertCellPoint(second)
    segment(x, y, x + width, y)
    segment(x + width, y, x + width, y + height)
    segment(x + width, y + height, x, y + height)
    segment(x, y + height, x, y)
    low, high = bar.GetLookupTable().GetRange()
    tick_length = max(3, bar.GetLabelTextProperty().GetFontSize() * .4)
    ticks = bar.GetCustomLabels()
    longest = 0
    if high > low:
        for index in range(ticks.GetNumberOfValues()):
            tick_y = y + height * (ticks.GetValue(index) - low) / (high - low)
            segment(x + width, tick_y, x + width + tick_length, tick_y)
            if text:
                actor = text[index]
                font = actor.GetTextProperty().GetFontSize()
                actor.SetPosition((x + width + tick_length + font * .4) / viewport_width,
                                  tick_y / viewport_height)
                from vtkmodules.vtkRenderingCore import vtkTextRenderer
                bounds = [0, 0, 0, 0]
                vtkTextRenderer.GetInstance().GetBoundingBox(actor.GetTextProperty(), actor.GetInput(), bounds, 72)
                longest = max(longest, bounds[1] - bounds[0])
    if text:
        font = text[-1].GetTextProperty().GetFontSize()
        text[-1].SetPosition((x + width + tick_length + longest + font * 1.4) / viewport_width,
                             (y + height / 2) / viewport_height)
    data = vtkPolyData()
    data.SetPoints(points)
    data.SetLines(lines)
    outline.GetMapper().SetInputData(data)
