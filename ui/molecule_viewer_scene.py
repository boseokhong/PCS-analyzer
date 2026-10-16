"""Child-only VTK presentation and picking; receives already prepared coordinates."""


class MolecularSceneRenderer:
    def __init__(self, plotter, window, on_pick):
        import pyvista as pv
        from vtkmodules.vtkRenderingCore import vtkCellPicker, vtkFollower, vtkPolyDataMapper
        from ui.molecule_viewer_camera import ViewportCamera
        from ui.molecule_viewer_axes import ViewportAxes
        from ui.molecule_viewer_appearance import ViewportAppearance, ViewportRendering, BondLineSizing

        self.pv = pv
        self.plotter = plotter
        self.window = window
        self.on_pick = on_pick
        self.scene = None
        self.camera_controls = ViewportCamera(plotter)
        self.axes = ViewportAxes(plotter)
        self.appearance = ViewportAppearance()
        from logic.molecule_viewer_pcs import PcsColors
        self.pcs_colors = PcsColors()
        self.bond_sizing = BondLineSizing()
        self.rendering = ViewportRendering(plotter)
        self.sphere_detail = "Standard"
        self.atoms = {}
        self.actor_refs = {}
        self.bonds = None
        self.annotations = []
        self.scalar_bar = None
        self.scalar_bar_outline = None
        self.scalar_bar_text = []
        self.reference_actor = None
        self.averaging_legend = None
        self.averaged = set()
        self.atom_kinds = {}
        self.circle_mapper = vtkPolyDataMapper()
        self.circle_mapper.SetInputData(pv.Polygon(n_sides=64, radius=1, fill=True))
        self.follower_type = vtkFollower
        self.press = None
        self.sphere = pv.Sphere(radius=1, theta_resolution=32, phi_resolution=32)
        self.highlight = plotter.add_mesh(self.sphere, color="yellow", style="wireframe",
                                          name="selection", pickable=False, reset_camera=False, render=False)
        self.highlight.SetVisibility(False)
        self.average_highlight = self._circle_actor("yellow", pickable=False)
        self.average_highlight.SetVisibility(False)
        # Geometric cell picking also reaches the centre of a hollow marker.
        self.picker = vtkCellPicker()
        self.picker.PickFromListOn()
        self.interactor = plotter.iren.interactor
        # PyVista's camera style captures release events. Register that observer
        # on the style, and retain each actual owner/tag for precise cleanup.
        style = self.interactor.GetInteractorStyle()
        self.observers = [(owner, owner.AddObserver(event, callback)) for owner, event, callback in (
            (self.interactor, "LeftButtonPressEvent", self._press),
            (self.interactor, "MouseMoveEvent", self._move),
            (style, "LeftButtonReleaseEvent", self._release),
            (plotter.renderer, "StartEvent", self._update_bond_width),
            (plotter.renderer, "EndEvent", self._update_legend_outline))]

    def _update_legend_outline(self, *args):
        if self.scalar_bar is not None and self.scalar_bar_outline is not None:
            from ui.molecule_viewer_legend import update_pcs_outline
            update_pcs_outline(self.scalar_bar, self.scalar_bar_outline, self.plotter.renderer, self.scalar_bar_text)

    def apply(self, scene):
        if hasattr(self, "axes"):
            self.axes.document(scene["document_id"])
        new_document = self.scene is None or scene["document_id"] != self.scene["document_id"]
        self.appearance.document(scene["document_id"])
        if getattr(self, "pcs_colors", None) is not None:
            self.pcs_colors.document(scene["document_id"])
        if getattr(self, "bond_sizing", None) is not None:
            self.bond_sizing.document(scene["document_id"])
        self.rendering.apply(self.appearance)
        self.plotter.set_background(self.appearance.background_color)
        self._sphere_detail()
        presentation = self.pcs_presentation(scene.get("presentation"))
        self.averaged = set(presentation.get("averaged_ref_ids", [])) if presentation else set()
        self.radius_by_ref = dict(zip(scene["ref_ids"], scene["radii"]))
        wanted = set(scene["ref_ids"])
        for ref in list(self.atoms):
            if new_document or ref not in wanted or self.atom_kinds[ref] != (ref in self.averaged):
                removed = self.atoms.pop(ref)
                self.appearance.forget(removed)
                self.plotter.remove_actor(removed, reset_camera=False, render=False)
                self.atom_kinds.pop(ref)
        self.picker.InitializePickList()
        self.actor_refs.clear()
        for ref, xyz, radius, color in zip(scene["ref_ids"], scene["coords"],
                                           scene["radii"], scene["colors"]):
            actor = self.atoms.get(ref)
            if actor is None:
                if ref in self.averaged:
                    actor = self._circle_actor(color, pickable=True)
                else:
                    actor = self.plotter.add_mesh(self.sphere, name=f"atom_{ref}",
                                                  smooth_shading=True, reset_camera=False, render=False)
                self.atoms[ref] = actor
                self.atom_kinds[ref] = ref in self.averaged
            actor.SetPosition(*xyz)
            actor.SetScale(radius * (1.18 ** .5 if ref in self.averaged else 1))
            actor.GetProperty().SetColor(*(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)))
            self.picker.AddPickList(actor)
            self.actor_refs[actor.GetAddressAsString("")] = ref
            if ref not in self.averaged:
                self.appearance.material(actor, self.appearance.atom_finish)
        if presentation is not None:
            for ref, color in zip(scene["ref_ids"], presentation["atom_rgba"]):
                if ref not in self.averaged:
                    self.atoms[ref].GetProperty().SetColor(*color[:3])
        self._bonds(scene)
        self.scene = scene
        self.press = None
        self.highlight.SetVisibility(False)
        self.average_highlight.SetVisibility(False)
        self._reference(scene.get("reference"))
        self._presentation(presentation)
        self.camera_controls.update(scene)
        if self.camera_controls.available and getattr(self, "bond_sizing", None) is not None:
            # An empty view has no fitted camera. Calibrate only after the first
            # visible geometry, retaining that baseline through later updates.
            self.bond_sizing.calibrate(self.plotter.camera, self.plotter.renderer)
        self._update_bond_width()
        self.plotter.render()

    def _update_bond_width(self, *args):
        if (self.scene is None or self.bonds is None or self.appearance.bond_mode != "Line"
                or getattr(self, "bond_sizing", None) is None):
            return
        from ui.molecule_viewer_appearance import projected_line_width
        renderer, camera = self.plotter.renderer, self.plotter.camera
        diameter = self.bond_sizing.diameter(self.appearance.line_width)
        if diameter is not None:
            width = projected_line_width(diameter, camera, renderer.GetSize()[1], renderer.GetTiledAspectRatio())
            if width is not None:
                self.bonds.GetProperty().SetLineWidth(width)

    def pcs_presentation(self, value):
        if getattr(self, "pcs_colors", None) is None:
            return value
        from ui.molecule_viewer_pcs import pcs_presentation
        return pcs_presentation(value, self.pcs_colors.options())

    def update_pcs(self):
        if self.scene is None:
            return
        value = self.pcs_presentation(self.scene.get("presentation"))
        if value is not None:
            for ref, color in zip(self.scene["ref_ids"], value["atom_rgba"]):
                if ref not in self.averaged:
                    self.atoms[ref].GetProperty().SetColor(*color[:3])
        self._presentation(value)
        self.plotter.render()

    def _sphere_detail(self):
        from ui.molecule_viewer_appearance import DETAILS
        if self.sphere_detail == self.appearance.detail:
            return
        resolution = DETAILS[self.appearance.detail]
        self.sphere = self.pv.Sphere(radius=1, theta_resolution=resolution, phi_resolution=resolution)
        self.sphere_detail = self.appearance.detail
        for ref, actor in self.atoms.items():
            if not self.atom_kinds[ref]:
                actor.GetMapper().SetInputData(self.sphere)
        self.highlight.GetMapper().SetInputData(self.sphere)

    def _bonds(self, scene):
        from ui.molecule_viewer_appearance import DETAILS
        if self.bonds is not None:
            self.appearance.forget(self.bonds)
            self.plotter.remove_actor(self.bonds, reset_camera=False, render=False)
            self.bonds = None
        if scene["bonds"]:
            cylinder = self.appearance.bond_mode == "Cylinder"
            bonds = scene["bonds"]
            if cylinder:
                # Coincident endpoints have no drawable cylinder. Preserve the
                # scientific connectivity snapshot and omit only that geometry.
                bonds = [(i, j) for i, j in bonds if scene["coords"][i] != scene["coords"][j]]
                if not bonds:
                    return
            mesh = self.pv.PolyData(scene["coords"])
            mesh.lines = [value for i, j in bonds for value in (2, i, j)]
            if cylinder:
                mesh = mesh.tube(radius=self.appearance.bond_radius,
                                 n_sides=DETAILS[self.appearance.detail] // 2, capping=True)
            self.bonds = self.plotter.add_mesh(mesh, color=self.appearance.bond_color, line_width=self.appearance.line_width,
                                              smooth_shading=cylinder and self.appearance.smooth,
                                              pickable=False, reset_camera=False, render=False)
            self.appearance.material(self.bonds, self.appearance.bond_finish, surface=cylinder)

    def update_appearance(self):
        if self.scene is None:
            return
        self.press = None
        self.rendering.apply(self.appearance)
        self.plotter.set_background(self.appearance.background_color)
        self._sphere_detail()
        for ref, actor in self.atoms.items():
            if ref not in self.averaged:
                self.appearance.material(actor, self.appearance.atom_finish)
        self._bonds(self.scene)
        if getattr(self, "_annotation_background", "#FFFFFF") != self.appearance.background_color:
            self._presentation(self.pcs_presentation(self.scene.get("presentation")))
        self.plotter.render()

    def navigate(self, action):
        from ui.molecule_viewer_camera import STANDARD_VIEWS
        if action not in ("fit", "reset", "projection:Perspective", "projection:Orthographic", *('view:' + name for name in STANDARD_VIEWS)):
            raise ValueError("Unknown viewport camera action")
        self.press = None
        if action.startswith("projection:"):
            return self.camera_controls.project(action.partition(":")[2])
        if action.startswith("view:"):
            return self.camera_controls.view(action.partition(":")[2])
        return getattr(self.camera_controls, action)()

    def _circle_actor(self, color, *, pickable):
        actor = self.follower_type()
        actor.SetMapper(self.circle_mapper)
        actor.SetCamera(self.plotter.camera)
        actor.SetPickable(pickable)
        prop = actor.GetProperty()
        prop.SetRepresentationToWireframe()
        prop.SetLineWidth(3.0)
        prop.SetOpacity(1.0)
        prop.LightingOff()
        if color == "yellow":
            prop.SetColor(1, 1, 0)
        else:
            prop.SetColor(*(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)))
        self.plotter.add_actor(actor, pickable=pickable, reset_camera=False, render=False)
        return actor

    def _reference(self, reference):
        if self.reference_actor is not None:
            self.plotter.remove_actor(self.reference_actor, reset_camera=False, render=False)
            self.reference_actor = None
        if reference is not None and reference["coords"]:
            import numpy as np
            points = self.pv.PolyData(reference["coords"])
            points["cpk"] = np.asarray([[int(color[i:i + 2], 16) for i in (1, 3, 5)]
                                         for color in reference["colors"]], dtype=np.uint8)
            # Legacy reference area is 24 square points, independent of atom scale.
            screen = self.window.screen()
            dpi = screen.logicalDotsPerInch() if screen else 96.
            size = 24 ** .5 * dpi / 72 * self.window.devicePixelRatioF()
            self.reference_actor = self.plotter.add_mesh(points, scalars="cpk", rgb=True,
                point_size=size, render_points_as_spheres=True, opacity=.72,
                name="reference", show_scalar_bar=False, pickable=False,
                reset_camera=False, render=False)

    def _presentation(self, value):
        from ui.molecule_viewer_color import annotation_color
        self._annotation_background = self.appearance.background_color
        foreground = annotation_color(self._annotation_background)
        from pathlib import Path
        from vtkmodules.vtkCommonCore import vtkLookupTable
        from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor

        for actor in self.annotations:
            self.plotter.remove_actor(actor, reset_camera=False, render=False)
        self.annotations.clear()
        if self.scalar_bar is not None:
            self.plotter.renderer.RemoveActor2D(self.scalar_bar)
            self.scalar_bar = None
        if self.scalar_bar_outline is not None:
            self.plotter.renderer.RemoveActor2D(self.scalar_bar_outline)
            self.scalar_bar_outline = None
        for actor in self.scalar_bar_text:
            self.plotter.renderer.RemoveActor2D(actor)
        self.scalar_bar_text.clear()
        if self.averaging_legend is not None:
            self.plotter.renderer.RemoveActor2D(self.averaging_legend)
            self.averaging_legend = None
        if value is None:
            self.axes.clear()
            return
        if not Path(value["font_file"]).is_file():
            raise ValueError("The host plot font is unavailable in the viewer process")
        common = dict(font_file=value["font_file"], show_points=False, shape=None,
                      always_visible=True, pickable=False, reset_camera=False, render=False)
        if value["show_labels"] and self.scene["coords"]:
            self.annotations.append(self.plotter.add_point_labels(
                self.scene["coords"], self.scene["labels"], name="atom_labels",
                text_color=foreground, bold=False, font_size=value["label_size"], **common))
        self.axes.apply(value)
        scale = value["color_scale"]
        if scale is not None and value.get("pcs_legend_visible", True):
            table = vtkLookupTable()
            table.SetNumberOfTableValues(256)
            table.SetRange(*scale["range"])
            table.Build()
            for i, color in enumerate(scale["rgba"]):
                table.SetTableValue(i, *color)
            bar = vtkScalarBarActor()
            bar.SetLookupTable(table)
            bar.SetTitle(scale["title"])
            from ui.molecule_viewer_legend import style_pcs_legend, configure_pcs_ticks
            legend_font = value.get("pcs_font_pt", 0)
            legend_value = dict(value, label_size=legend_font or value["label_size"],
                                tick_size=legend_font or value["tick_size"])
            style_pcs_legend(bar, legend_value, foreground,
                self.window.logicalDpiY() * self.window.devicePixelRatioF() / 72.)
            configure_pcs_ticks(bar, value.get("pcs_tick_step", 0))
            self.plotter.renderer.AddActor2D(bar)
            self.scalar_bar = bar
            from ui.molecule_viewer_legend import make_pcs_outline
            self.scalar_bar_outline = make_pcs_outline(foreground)
            self.plotter.renderer.AddActor2D(self.scalar_bar_outline)
            from ui.molecule_viewer_legend import make_pcs_text
            self.scalar_bar_text = make_pcs_text(bar, scale["title"], legend_value, foreground,
                self.window.logicalDpiY() * self.window.devicePixelRatioF() / 72.)
            for actor in self.scalar_bar_text:
                self.plotter.renderer.AddActor2D(actor)

        from logic.molecule_viewer_snapshot import averaging_legend_entries
        entries = averaging_legend_entries(self.scene)
        if entries:
            from vtkmodules.vtkRenderingCore import vtkTextActor
            legend = vtkTextActor()
            legend.SetInput("Averaging\n" + "\n".join(
                ("● " if entry == "Reference" else "○ ") + entry for entry in entries))
            legend.SetPickable(False)
            legend.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            legend.SetPosition(.98, .98)
            prop = legend.GetTextProperty()
            prop.SetFontFamilyAsString("File")
            prop.SetFontFile(value["font_file"])
            prop.SetFontSize(value["tick_size"])
            prop.SetColor(*annotation_color(self._annotation_background, legend=True))
            prop.SetJustificationToRight()
            prop.SetVerticalJustificationToTop()
            self.plotter.renderer.AddActor2D(legend)
            self.averaging_legend = legend

    def select(self, refs):
        for ref in self.averaged:
            selected = bool(refs) and ref == refs[0]
            self.atoms[ref].SetScale(self.radius_by_ref[ref] * ((1.35 if selected else 1.18) ** .5))
        actor = self.atoms.get(refs[0]) if refs else None
        averaged = bool(refs) and refs[0] in self.averaged
        self.highlight.SetVisibility(actor is not None and not averaged)
        self.average_highlight.SetVisibility(actor is not None and averaged)
        if actor is not None:
            highlight = self.average_highlight if averaged else self.highlight
            highlight.SetPosition(*actor.GetPosition())
            highlight.SetScale(*(value * (1.75 ** .5 if averaged else 1.18) for value in actor.GetScale()))
        self.plotter.render()

    def _press(self, obj, event):
        self.press = (self.interactor.GetEventPosition(), self.scene.get("scene_generation", 0),
                      False, tuple(self.plotter.camera_position))

    def _move(self, obj, event):
        if self.press is not None:
            start, generation, dragged, camera = self.press
            xy = self.interactor.GetEventPosition()
            distance = sum((a - b) ** 2 for a, b in zip(xy, start))
            self.press = (start, generation, dragged or distance > (3 * self.window.devicePixelRatioF()) ** 2,
                          camera)

    def _release(self, obj, event):
        self._move(obj, event)
        press, self.press = self.press, None
        if (press is None or press[2] or press[1] != self.scene.get("scene_generation", 0)
                or press[3] != tuple(self.plotter.camera_position)):
            return
        x, y = self.interactor.GetEventPosition()
        # VTK supplies framebuffer coordinates; do not multiply them by Qt DPR.
        if self.picker.Pick(x, y, 0, self.plotter.renderer):
            actor = self.picker.GetActor()
            ref = self.actor_refs.get(actor.GetAddressAsString("")) if actor is not None else None
            if ref is not None:
                self.on_pick(ref)

    def close(self):
        self.axes.clear()
        for owner, tag in self.observers:
            owner.RemoveObserver(tag)
        self.observers.clear()
        for actor in self.scalar_bar_text:
            self.plotter.renderer.RemoveActor2D(actor)
        self.scalar_bar_text.clear()
        if self.scalar_bar_outline is not None:
            self.plotter.renderer.RemoveActor2D(self.scalar_bar_outline)
            self.scalar_bar_outline = None
        if self.scalar_bar is not None:
            self.plotter.renderer.RemoveActor2D(self.scalar_bar)
            self.scalar_bar = None
        if self.averaging_legend is not None:
            self.plotter.renderer.RemoveActor2D(self.averaging_legend)
            self.averaging_legend = None
