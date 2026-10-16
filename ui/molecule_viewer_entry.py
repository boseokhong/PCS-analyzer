"""Child-process entry for the default Qt/VTK molecular viewer."""

import json
import os
import queue
import sys
import threading
from pathlib import Path

from logic.molecule_viewer_protocol import MAX_EVENT_BYTES, encode_event, event_for, read_scene
from logic.molecule_viewer_session import ViewerSession


def run(scene_path):
    scene = None
    plotter = None
    app = None
    renderer = None
    chrome = None

    def emit(kind, **values):
        sys.stdout.write(encode_event(event_for(scene, kind, **values)))
        sys.stdout.flush()

    try:
        scene = read_scene(scene_path)
        # This entry executes only in the child role, before host UI imports.
        os.environ["QT_API"] = "pyside6"
        from PySide6 import QtCore, QtWidgets, QtGui
        from pyvistaqt import QtInteractor
        from ui.molecule_viewer_scene import MolecularSceneRenderer
        from ui.molecule_viewer_export import export_viewport_dialog
        from ui.molecule_viewer_controls import DisplayControls
        from ui.molecule_viewer_chrome import ViewerChrome
        from ui.molecule_viewer_axes import AxesControls
        from ui.molecule_viewer_appearance import AppearanceControls
        from ui.molecule_viewer_figure import FigureControls

        session = ViewerSession(scene, Path(scene_path).parent)

        commands = queue.Queue(maxsize=16)

        def read_commands():
            try:
                from logic.molecule_viewer_stdio import iter_command_lines
                for line in iter_command_lines(sys.stdin.fileno(), MAX_EVENT_BYTES):
                    if len(line) > MAX_EVENT_BYTES:
                        break
                    command = json.loads(line)
                    if not isinstance(command, dict):
                        break
                    commands.put_nowait(command)
            except (OSError, ValueError, queue.Full):
                pass  # Invalid/closed transport is handled as a disconnect below.
            finally:
                try:
                    commands.put_nowait({"type": "disconnect"})
                except queue.Full:
                    # Main-thread drain always observes this flag even if full.
                    disconnected.set()

        disconnected = threading.Event()
        app = QtWidgets.QApplication([])
        class ViewerWindow(QtWidgets.QMainWindow):
            def closeEvent(self, event):
                super().closeEvent(event)
                if event.isAccepted():
                    # Hidden QtInteractor helpers must not keep the app alive.
                    app.quit()

        window = ViewerWindow()
        def update_title():
            window.setWindowTitle("Molecular 3D — "
                                  + scene.get("document_title", scene["document_id"]))

        update_title()
        frame = QtWidgets.QFrame(window)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        display_sequence = 0

        def display_changed(key, value):
            nonlocal display_sequence
            display_sequence += 1
            emit("display_change", key=key, value=value, display_sequence=display_sequence)

        # Keep controls outside the native VTK render-window subtree, including
        # before the toolbar reparents them. Native sibling reparenting can leave
        # a black, unpainted region on Windows.
        display_controls = DisplayControls(window, QtWidgets, QtGui, display_changed, core=QtCore)
        display_controls.sync(scene.get("display_controls"), scene["document_id"])
        window.setCentralWidget(frame)
        legacy_sequence = 0

        def request_matplotlib():
            nonlocal legacy_sequence
            legacy_sequence += 1
            emit("open_matplotlib", legacy_sequence=legacy_sequence)

        file_menu = window.menuBar().addMenu("File")
        figure_menu = file_menu.addMenu("Figure output")
        viewport_menu = file_menu.addMenu("Viewport capture")
        for menu in (file_menu, figure_menu, viewport_menu):
            menu.setToolTipsVisible(True)
        viewport_action = viewport_menu.addAction("PNG…")
        viewport_action.setToolTip("Save this viewport at its current pixel size and camera, including visible annotations.")

        def request_viewport_export():
            export_viewport_dialog(window, plotter, lambda: scene, QtWidgets)

        viewport_action.triggered.connect(request_viewport_export)
        for vector_format in ("pdf", "svg"):
            vector_action = viewport_menu.addAction(f"{vector_format.upper()} (experimental)…")
            vector_action.setToolTip("Experimental GL2PS output; the backend may rasterize structure and annotations. Transparent layers are unsupported; output parity is unverified.")
            def request_vector_export(checked=False, format=vector_format):
                export_viewport_dialog(window, plotter, lambda: scene, QtWidgets, format=format)
            vector_action.triggered.connect(request_vector_export)
        def send(event):
            sys.stdout.write(encode_event(event))
            sys.stdout.flush()

        def picked(ref):
            try:
                event = session.pick(ref)
                if event is not None:
                    renderer.select(session.selected_refs)
                    chrome.update(scene, session.selected_refs)
                    send(event)
            except Exception as exc:
                emit("error", message=f"Viewer picking failed: {str(exc)[:768]}")
                window.close()

        def axes_changed():
            try:
                renderer.axes.apply(scene.get("presentation"))
                plotter.render()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(window, "Molecular 3D", f"Axis update failed:\n{str(exc)[:768]}")

        def appearance_changed():
            try:
                renderer.update_appearance()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(window, "Molecular 3D", f"Appearance update failed:\n{str(exc)[:768]}")
                return False
            return True

        from ui.molecule_viewer_camera import CameraMenu
        camera_menu = CameraMenu(window, QtWidgets, QtGui, lambda: renderer)
        camera_actions = camera_menu.toolbar_actions
        update_camera_actions = camera_menu.sync

        figure_controls = FigureControls(window, QtWidgets, QtCore, lambda: scene, lambda: plotter,
                                         gui=QtGui, renderer=lambda: renderer)
        figure_controls.document(scene["document_id"])
        figure_export = figure_menu.addAction("PNG…")
        figure_export.triggered.connect(figure_controls.export)
        for figure_format in ("pdf", "svg"):
            action = figure_menu.addAction(f"{figure_format.upper()} (image)…")
            action.setToolTip("Embed the Figure PNG at its physical size/DPI, with output typography, margins and inclusion. Not vector; PDF requires an opaque background.")
            action.triggered.connect(lambda checked=False, format=figure_format: figure_controls.export(format=format))
        chrome = ViewerChrome(window, QtWidgets, QtCore, QtGui, display_controls,
                              file_menu, camera_actions, figure=figure_controls, viewport_layout=layout)
        view_menu = chrome.view_menu
        view_menu.addSeparator()
        legacy_action = view_menu.addAction("Open Matplotlib viewer…")
        legacy_action.setToolTip("Open the retained Tk/Matplotlib viewer for this document. Its Export plot uses its own camera.")
        legacy_action.triggered.connect(request_matplotlib)
        chrome.update(scene)
        # Finish widget ownership and chrome styling before VTK requests native
        # handles. No toolbar/dock reparenting occurs after that initialization.
        plotter = QtInteractor(frame)
        layout.addWidget(plotter.interactor, 1)
        plotter.set_background("white")
        renderer = MolecularSceneRenderer(plotter, window, picked)
        axes_controls = AxesControls(display_controls.panel, QtWidgets, renderer.axes, axes_changed, gui=QtGui, core=QtCore)
        appearance_controls = AppearanceControls(display_controls.panel, QtWidgets, QtCore,
                                                 renderer.appearance, appearance_changed, gui=QtGui)
        appearance_controls.arrange(display_controls, axes_controls)
        display_controls.panel.layout().addRow(appearance_controls.widget)
        appearance_controls.apply_icons(chrome.assets)
        axes_controls.apply_icons(chrome.assets)
        from ui.molecule_viewer_pcs import PcsColorControls
        def pcs_changed():
            renderer.update_pcs()
            return True
        pcs_controls = PcsColorControls(display_controls.panel, QtWidgets, QtCore, QtGui,
                                        renderer, pcs_changed)
        appearance_controls.form.insertRow(4, pcs_controls.widget)
        pcs_controls.apply_icons(chrome.assets)
        from ui.molecule_viewer_style import align_form_labels
        align_form_labels([appearance_controls.form] +
            [section.form for section in appearance_controls.sections + axes_controls.sections + figure_controls.sections] + [pcs_controls.form],
            QtWidgets, window.fontMetrics())
        from ui.molecule_viewer_style import compact_button
        for control, scope in ((axes_controls, "axes"), (appearance_controls, "appearance")):
            compact_button(chrome.assets, control.reset_button, "reset", "Reset",
                f"Reset {scope} settings only; atom size, bond tolerance and camera are retained.")
        settings_menu = file_menu.addMenu("Visualization settings")
        settings_menu.setToolTipsVisible(True)
        from ui.molecule_viewer_settings import settings_dialog
        for save, label in ((True, "Save…"), (False, "Load…")):
            action = settings_menu.addAction(label)
            action.setToolTip("Appearance, PCS colors, axes and Figure options only. Coordinates, camera, selection and host controls are excluded.")
            action.triggered.connect(lambda checked=False, save=save: settings_dialog(
                window, QtWidgets, renderer, figure_controls, axes_controls, appearance_controls, save=save, pcs_controls=pcs_controls))
        update_camera_actions()

        # All widget/render mutations occur on the Qt main thread. The pipe reader
        # queues plain commands; it never accesses actors, widgets or selections.
        def drain():
            nonlocal scene
            if disconnected.is_set():
                window.close()
                return
            for _ in range(8):
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    return
                if command.get("type") == "disconnect":
                    window.close()
                    return
                if command.get("session_id") != scene["session_id"]:
                    emit("error", message="Invalid lifecycle command session")
                    window.close()
                    return
                if command.get("type") == "shutdown":
                    window.close()
                    return
                try:
                    if command.get("type") == "scene":
                        replacement = session.replacement(command)
                        if replacement is not None:
                            renderer.apply(replacement)
                            scene = replacement
                            display_controls.sync(scene.get("display_controls"), scene["document_id"])
                            axes_controls.sync()
                            appearance_controls.sync()
                            pcs_controls.sync()
                            figure_controls.document(scene["document_id"])
                            update_title()
                            update_camera_actions()
                            send(session.commit(replacement))
                            chrome.update(scene, session.selected_refs)
                        continue
                    if command.get("type") == "focus":
                        if command.get("session_id") == session.scene["session_id"]:
                            if window.isMinimized():
                                window.showNormal()
                            window.raise_()
                            window.activateWindow()
                        continue
                    if command.get("type") == "selection":
                        acknowledgement = session.select(command)
                        if acknowledgement is not None:
                            renderer.select(session.selected_refs)
                            chrome.update(scene, session.selected_refs)
                            send(acknowledgement)
                        continue
                except Exception as exc:
                    emit("error", message=f"Viewer update failed: {str(exc)[:768]}")
                    window.close()
                    return
                emit("error", message="Unsupported prototype command")
                window.close()
                return

        timer = QtCore.QTimer(window)
        timer.timeout.connect(drain)
        timer.start(30)
        command_reader = threading.Thread(target=read_commands, daemon=True)
        command_reader.start()
        window.resize(800, 600)  # Initial logical size only; native resize remains enabled.
        window.show()

        def ready():
            try:
                if not window.isVisible():
                    return
                # The first scene/render uses the shown window's actual layout
                # and screen metrics, rather than pre-show temporary geometry.
                chrome.refresh_layout()
                renderer.apply(scene)
                pcs_controls.sync()
                # Reapply the notice after native VTK initialization/first show.
                chrome.update(scene, session.selected_refs)
                chrome.refresh_layout()
                update_camera_actions()
                screen = window.screen()
                emit("ready", pid=os.getpid(), device_pixel_ratio=window.devicePixelRatioF(),
                     logical_dpi=screen.logicalDotsPerInch() if screen else None)
            except Exception as exc:
                emit("error", message=f"Viewer readiness failed: {str(exc)[:768]}")
                window.close()

        QtCore.QTimer.singleShot(0, ready)
        code = app.exec()
        timer.stop()
        active_renderer, renderer = renderer, None
        active_renderer.close()
        plotter.close()
        plotter = None
        emit("closed")
        # The host responds with shutdown + EOF. Raw descriptor reads avoid a
        # buffered stdin lock even if the host is temporarily not polling.
        command_reader.join(timeout=1.0)
        return code
    except Exception as exc:
        if scene is not None:
            emit("error", message=f"{type(exc).__name__}: {str(exc)[:768]}")
        else:
            sys.stderr.write(f"Invalid molecular scene: {exc}\n")
        return 1
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception as exc:
                if scene is not None:
                    emit("error", message=f"Picking cleanup failed: {str(exc)[:768]}")
        if plotter is not None:
            try:
                plotter.close()
            except Exception as exc:
                if scene is not None:
                    emit("error", message=f"Viewer cleanup failed: {str(exc)[:768]}")
        if chrome is not None:
            try:
                chrome.close()
            except Exception as exc:
                if scene is not None:
                    emit("error", message=f"Viewer style cleanup failed: {str(exc)[:768]}")
