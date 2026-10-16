"""Visualization file dialogs and transactional application in the Qt child."""

import copy
import logging

from logic.molecule_viewer_settings import AXIS_KEYS, FORMAT, load_settings, save_settings, validate_settings
from ui.molecule_viewer_figure import figure_pixels, figure_layout
from logic.molecule_viewer_pcs import PcsColors, DEFAULTS as PCS_DEFAULTS


def snapshot(renderer, figure):
    options = figure.options()
    options.pop("viewport_background", None)
    result = dict(format=FORMAT, version=1, appearance=renderer.appearance.options(),
        axes={key: getattr(renderer.axes, key) for key in AXIS_KEYS}, figure=options)
    if isinstance(getattr(renderer, "pcs_colors", None), PcsColors):
        result["pcs_colors"] = renderer.pcs_colors.options()
    return copy.deepcopy(result)


def _figure_restore(figure, options):
    fields = {"width": (figure.width, "setValue"), "height": (figure.height, "setValue"),
        "dpi": (figure.dpi, "setValue"), "unit": (figure.unit, "setCurrentText"),
        "background": (figure.background, "setCurrentText")}
    fields.update({key: (control, "setValue") for key, control in figure.output_controls.items()})
    fields.update({"include_" + key: (control, "setChecked") for key, control in figure.layers.items()})
    for key, (control, setter) in fields.items():
        blocked = control.blockSignals(True)
        try:
            getattr(control, setter)(options[key])
        finally:
            control.blockSignals(blocked)
    figure.unit_value = options["unit"]
    figure.update()


def apply_settings(data, renderer, figure, axes_controls, appearance_controls, pcs_controls=None):
    validate_settings(data)
    options = data["figure"]
    size = figure_pixels(options["width"], options["height"], options["unit"], options["dpi"])
    figure_layout(size, options["dpi"], options)  # Reject invalid margins before mutation.
    previous = snapshot(renderer, figure)
    def restore(value):
        renderer.appearance.restore(copy.deepcopy(value["appearance"]))
        pcs = getattr(renderer, "pcs_colors", None)
        if isinstance(pcs, PcsColors):
            pcs.restore(value.get("pcs_colors", PCS_DEFAULTS))
        for key, item in value["axes"].items():
            setattr(renderer.axes, key, copy.deepcopy(item))
        _figure_restore(figure, value["figure"])
        axes_controls.sync()
        appearance_controls.sync()
        renderer.update_appearance()
        if isinstance(pcs, PcsColors):
            renderer.update_pcs()
        if pcs_controls is not None:
            pcs_controls.sync()
        renderer.axes.apply(renderer.scene.get("presentation"))
        renderer.plotter.render()
    try:
        restore(data)
    except Exception:
        try:
            restore(previous)
        except Exception:
            logging.getLogger(__name__).exception("Visualization rollback rendering failed")
        raise


def settings_dialog(window, widgets, renderer, figure, axes_controls, appearance_controls, *, save, pcs_controls=None):
    generation = renderer.appearance.document_generation
    dialog = widgets.QFileDialog(window, "Save visualization settings" if save else "Load visualization settings")
    try:
        if renderer.scene is None:
            raise ValueError("The molecular view is not ready yet.")
        dialog.setNameFilter("Visualization settings (*.json)")
        if save:
            dialog.setAcceptMode(widgets.QFileDialog.AcceptMode.AcceptSave)
            dialog.setDefaultSuffix("json")
            dialog.selectFile("molecule-visualization.json")
        else:
            dialog.setFileMode(widgets.QFileDialog.FileMode.ExistingFile)
        if not dialog.exec() or not window.isVisible():
            return
        if generation != renderer.appearance.document_generation:
            raise ValueError("The document changed. Reopen the settings dialog.")
        path = dialog.selectedFiles()[0]
        if save:
            data = snapshot(renderer, figure)
            options = data["figure"]
            size = figure_pixels(options["width"], options["height"], options["unit"], options["dpi"])
            figure_layout(size, options["dpi"], options)
            save_settings(path, data)
        else:
            apply_settings(load_settings(path), renderer, figure, axes_controls, appearance_controls, pcs_controls)
    except Exception as exc:
        if window.isVisible():
            widgets.QMessageBox.warning(window, "Visualization settings", str(exc)[:768])
    finally:
        try:
            dialog.deleteLater()
        except RuntimeError:
            logging.getLogger(__name__).debug("Settings dialog was already destroyed with its parent")
