"""Child viewport navigation; no coordinate, tensor or host-state mutation."""

import math

STANDARD_VIEWS = {
    "+X": ((1, 0, 0), (0, 0, 1)), "-X": ((-1, 0, 0), (0, 0, 1)),
    "+Y": ((0, 1, 0), (0, 0, 1)), "-Y": ((0, -1, 0), (0, 0, 1)),
    "+Z": ((0, 0, 1), (0, 1, 0)), "-Z": ((0, 0, -1), (0, 1, 0)),
    "Isometric": ((1, 1, 1), (0, 0, 1)),
}


def vertical_view_factor(angle, horizontal=False, aspect=1.):
    """Focal-plane half-height / distance; independent of coordinates and DPI."""
    if not math.isfinite(angle) or not 0 < angle < 180 or not math.isfinite(aspect) or aspect <= 0:
        raise ValueError("Invalid camera view angle or viewport aspect ratio")
    return math.tan(math.radians(angle) / 2) / (aspect if horizontal else 1)


def molecular_bounds(scene):
    """Bounds for validated working/reference geometry, excluding decorations.

    Reserve room for selection rings. Reference screen-space dots contribute
    positions only; actual screen-space padding still needs native QA.
    """
    averaged = set(scene.get("presentation", {}).get("averaged_ref_ids", []))
    limits = [[float("inf"), -float("inf")] for _ in range(3)]
    present = False
    for ref, xyz, radius in zip(scene["ref_ids"], scene["coords"], scene["radii"]):
        padding = radius * ((1.35 * 1.75) ** .5 if ref in averaged else 1.18)
        for i in range(3):
            limits[i][0] = min(limits[i][0], xyz[i] - padding)
            limits[i][1] = max(limits[i][1], xyz[i] + padding)
        present = True
    for xyz in scene.get("reference", {}).get("coords", []):
        for i in range(3):
            limits[i][0] = min(limits[i][0], xyz[i])
            limits[i][1] = max(limits[i][1], xyz[i])
        present = True
    return tuple(v for pair in limits for v in pair) if present else None


class ViewportCamera:
    """Document-local initial camera; ordinary scene updates preserve navigation."""

    def __init__(self, plotter):
        self.plotter = plotter
        self.document_id = None
        self.initial = None
        self.bounds = None

    @property
    def available(self):
        return self.bounds is not None and self.initial is not None

    def update(self, scene):
        if scene["document_id"] != self.document_id:
            self.document_id = scene["document_id"]
            self.initial = None
            self.plotter.camera.SetParallelProjection(False)
        self.bounds = molecular_bounds(scene)
        if self.bounds is not None and self.initial is None:
            self.plotter.reset_camera(bounds=self.bounds, render=False)
            self.initial = self.plotter.camera.copy()

    def fit(self):
        if not self.available:
            return False
        self.plotter.reset_camera(bounds=self.bounds, render=False)
        self.plotter.render()
        return True

    def reset(self):
        if not self.available:
            return False
        # Keep the same live camera object so camera-facing markers retain it.
        self.plotter.camera.DeepCopy(self.initial)
        return self.fit()

    @property
    def projection(self):
        return "Orthographic" if self.plotter.camera.GetParallelProjection() else "Perspective"

    def project(self, mode):
        if mode not in ("Perspective", "Orthographic"):
            raise ValueError("Unknown viewport projection")
        if not self.available:
            return False
        if mode == self.projection:
            return True
        camera = self.plotter.camera
        position, focal = camera.GetPosition(), camera.GetFocalPoint()
        distance = math.dist(position, focal)
        horizontal = bool(camera.GetUseHorizontalViewAngle())
        aspect = self.plotter.renderer.GetTiledAspectRatio() if horizontal else 1.
        factor = vertical_view_factor(camera.GetViewAngle(), horizontal, aspect)
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("Cannot change projection with an invalid camera distance")
        scale = distance * factor if mode == "Orthographic" else camera.GetParallelScale()
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Cannot change projection with an invalid camera scale")
        saved = camera.copy()
        try:
            if mode == "Orthographic":
                camera.SetParallelScale(scale)
            else:
                new_distance = scale / factor
                camera.SetPosition(*(focal[i] + (position[i] - focal[i]) * new_distance / distance for i in range(3)))
            camera.SetParallelProjection(mode == "Orthographic")
            self.plotter.renderer.ResetCameraClippingRange()
            self.plotter.render()
        except Exception:
            camera.DeepCopy(saved)
            raise
        return True

    def view(self, name):
        if name not in STANDARD_VIEWS:
            raise ValueError("Unknown standard viewport view")
        if not self.available:
            return False
        camera = self.plotter.camera
        centre = tuple((self.bounds[2*i] + self.bounds[2*i+1]) / 2 for i in range(3))
        direction, up = STANDARD_VIEWS[name]
        length = math.sqrt(sum(value * value for value in direction))
        distance = max(math.dist(camera.GetPosition(), camera.GetFocalPoint()), 1.)
        saved = camera.copy()
        try:
            camera.SetFocalPoint(*centre)
            camera.SetPosition(*(centre[i] + direction[i] * distance / length for i in range(3)))
            camera.SetViewUp(*up)
            camera.OrthogonalizeViewUp()
            self.plotter.reset_camera(bounds=self.bounds, render=False)
            self.plotter.render()
        except Exception:
            camera.DeepCopy(saved)
            raise
        return True


class CameraMenu:
    """Native child menu; only existing Fit/Reset actions occupy the toolbar."""

    def __init__(self, window, widgets, gui, renderer):
        self.window, self.widgets, self.renderer = window, widgets, renderer
        self.menu = window.menuBar().addMenu("Camera")
        self.toolbar_actions, self.actions, self.projections = [], [], {}
        for label, action in (("Fit to molecule", "fit"), ("Reset camera", "reset")):
            item = self.menu.addAction(label)
            item.triggered.connect(lambda checked=False, action=action: self.execute(action))
            item.setToolTip("Change the viewport camera only; molecular coordinates and tensor orientation are preserved.")
            self.toolbar_actions.append(item)
            self.actions.append(item)
        projection_menu = self.menu.addMenu("Projection")
        self.group = gui.QActionGroup(window)
        self.group.setExclusive(True)
        for mode in ("Perspective", "Orthographic"):
            item = projection_menu.addAction(mode)
            item.setCheckable(True)
            self.group.addAction(item)
            item.setToolTip("Preserve the apparent size at the focal plane; depth appearance changes with projection.")
            item.triggered.connect(lambda checked=False, mode=mode: self.execute("projection:" + mode))
            self.projections[mode] = item
            self.actions.append(item)
        views = self.menu.addMenu("Standard views")
        for name in STANDARD_VIEWS:
            item = views.addAction(name)
            item.setToolTip("Look from this working-coordinate direction towards the molecule and fit it; coordinates are unchanged.")
            item.triggered.connect(lambda checked=False, name=name: self.execute("view:" + name))
            self.actions.append(item)

    def sync(self):
        controls = self.renderer().camera_controls
        for item in self.actions:
            item.setEnabled(controls.available)
        for mode, item in self.projections.items():
            blocked = item.blockSignals(True)
            item.setChecked(mode == controls.projection)
            item.blockSignals(blocked)

    def execute(self, action):
        try:
            self.renderer().navigate(action)
        except Exception as exc:
            self.widgets.QMessageBox.warning(self.window, "Molecular 3D", f"Camera action failed:\n{str(exc)[:768]}")
        finally:
            self.sync()
