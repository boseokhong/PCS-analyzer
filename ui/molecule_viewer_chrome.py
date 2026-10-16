"""Child-owned viewer layout. No GUI imports or scientific state mutations."""


def empty_scene_message(scene):
    if scene["ref_ids"] or (scene.get("reference") or {}).get("coords"):
        return ""
    if scene.get("structure_loaded") is False:
        return "No structure loaded. Load a structure in PCS Analyzer to display it here."
    return "No visible atoms. Check the element filters in PCS Analyzer."


def sidebar_width(font_metrics):
    """Bound font-aware width in Qt logical pixels; do not multiply by DPI."""
    width = font_metrics.horizontalAdvance("M")
    if not isinstance(width, (int, float)):
        return 300
    return max(300, min(360, int(width * 22 + 24)))


def scene_status(scene, selected_refs):
    if scene.get("structure_loaded") is False and not scene["ref_ids"]:
        return empty_scene_message(scene)
    working = len(scene["ref_ids"])
    reference = len((scene.get("reference") or {}).get("coords", []))
    text = f"Working atoms: {working} · Reference: {reference}"
    primary = selected_refs[0] if selected_refs else None
    if primary in scene["ref_ids"]:
        index = scene["ref_ids"].index(primary)
        label = str(scene["labels"][index]).replace("\n", " ")[:64]
        text += f" | Selected Ref {primary}: {label}"
    else:
        text += " | No visible atom selected"
    if not working:
        text += " | No visible working atoms"
    return text


class ViewerChrome:
    def __init__(self, window, widgets, core, gui, controls, export_menu, camera_actions, figure=None,
                 viewport_layout=None):
        from ui.molecule_viewer_style import ChromeAssets
        self.window, self.gui, self.core, self.widgets = window, gui, core, widgets
        self.assets = ChromeAssets(gui, core)
        self.controls = controls
        self.figure = figure
        self.theme = None
        self.ui_font = None
        self.empty_notice = None
        if viewport_layout is not None:
            self.empty_notice = widgets.QLabel("", window)
            self.empty_notice.setWordWrap(True)
            self.empty_notice.setAlignment(core.Qt.AlignmentFlag.AlignCenter)
            self.empty_notice.setContentsMargins(12, 12, 12, 12)
            self.empty_notice.setSizePolicy(widgets.QSizePolicy.Policy.Expanding,
                                            widgets.QSizePolicy.Policy.Fixed)
            viewport_layout.addWidget(self.empty_notice)
            self.empty_notice.hide()
        self.toolbar = widgets.QToolBar("Molecular view", window)
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(core.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toolbar.setIconSize(core.QSize(20, 20))
        window.addToolBar(core.Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.toolbar.addWidget(controls.widget)
        self.toolbar.addSeparator()
        for index, action in enumerate(camera_actions):
            if index < 2:
                self.assets.set_icon(action, ("focus_center", "reset")[index])
                action.setToolTip(("Fit the visible molecule, preserving view direction and projection.",
                                   "Restore the document's initial camera and fit the molecule.")[index])
            self.toolbar.addAction(action)
            if not action.icon().isNull():
                button = self.toolbar.widgetForAction(action)
                button.setToolButtonStyle(core.Qt.ToolButtonStyle.ToolButtonIconOnly)
                button.setAccessibleName(action.text())
        self.toolbar.addSeparator()

        self.dock = widgets.QDockWidget("Options", window)
        self.dock.setAllowedAreas(core.Qt.DockWidgetArea.RightDockWidgetArea)
        self.dock.setFeatures(widgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.tabs = widgets.QTabWidget(self.dock)
        self.tabs.setDocumentMode(False)
        self.tabs.setMovable(False)
        self.tabs.tabBar().setExpanding(True)
        scroll = widgets.QScrollArea(self.tabs)
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls.panel)
        self.tabs.addTab(scroll, "Display")
        self.figure_dock = None  # Compatibility attribute: one shared dock now.
        if figure is not None:
            figure_scroll = widgets.QScrollArea(self.tabs)
            figure_scroll.setWidgetResizable(True)
            figure_scroll.setWidget(figure.widget)
            figure_page = widgets.QWidget(self.tabs)
            figure_layout = widgets.QVBoxLayout(figure_page)
            figure_layout.setContentsMargins(0, 0, 0, 0)
            figure_layout.setSpacing(0)
            figure_layout.addWidget(figure_scroll, 1)
            figure_layout.addWidget(figure.footer)
            self.tabs.addTab(figure_page, "Figure")
        self.dock.setWidget(self.tabs)
        window.addDockWidget(core.Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.dock.setFixedWidth(300)
        self.tabs.tabBar().setFixedWidth(298)
        self.dock.hide()
        view_menu = self.view_menu = window.menuBar().addMenu("View")
        # Qt owns checked state and close-button synchronization for this toggle.
        self.sidebar_action = self.dock.toggleViewAction()
        self.sidebar_action.setText("Sidebar")
        self.sidebar_action.setToolTip("Show or hide the sidebar. Display and Figure tabs remain available.")
        self.assets.set_icon(self.sidebar_action, "settings")
        self.toolbar.addAction(self.sidebar_action)
        view_menu.addAction(self.sidebar_action)
        view_menu.addSeparator()
        if not self.sidebar_action.icon().isNull():
            button = self.toolbar.widgetForAction(self.sidebar_action)
            button.setToolButtonStyle(core.Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setAccessibleName("Sidebar")
        self.options_action = gui.QAction("Display options", window)
        self.figure_action = gui.QAction("Figure", window) if figure is not None else None
        for index, action in enumerate([self.options_action] + ([self.figure_action] if figure is not None else [])):
            action.setCheckable(True)
            action.setToolTip("Open " + ("Display" if index == 0 else "Figure") + " settings in the sidebar.")
            self.assets.set_icon(action, "settings" if index == 0 else "graph")
            action.triggered.connect(lambda checked=False, index=index: self.open_tab(index))
            view_menu.addAction(action)
        self.tabs.currentChanged.connect(self.sync_tabs)
        self.dock.visibilityChanged.connect(self.sync_tabs)
        from ui.molecule_viewer_style import compact_button
        compact_button(self.assets, controls.labels, "chemical_map", "Labels",
                       "Show or hide atom labels. Shared with the main application.")
        controls.labels.setToolButtonStyle(core.Qt.ToolButtonStyle.ToolButtonIconOnly
            if not controls.labels.icon().isNull() else core.Qt.ToolButtonStyle.ToolButtonTextOnly)

        export = widgets.QToolButton(self.toolbar)
        export.setToolButtonStyle(core.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        export.setIconSize(core.QSize(20, 20))
        export.setText("Export")
        if self.assets.set_icon(export, "export"):
            export.setToolButtonStyle(core.Qt.ToolButtonStyle.ToolButtonIconOnly)
        export.setAccessibleName("Export")
        export.setToolTip("Export Figure output or the current viewport.")
        export.setPopupMode(widgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        export.setMenu(export_menu)
        self.toolbar.addWidget(export)
        if figure is not None:
            figure.apply_icons(self.assets)
        self.status = window.statusBar()
        self.status.setSizeGripEnabled(False)

    def update(self, scene, selected_refs=()):
        if self.empty_notice is not None:
            message = empty_scene_message(scene)
            self.empty_notice.setText(message)
            self.empty_notice.setVisible(bool(message))
        theme = scene.get("ui_theme")
        values = scene.get("display_controls")
        ui_font = tuple(values["ui_font"]) if values is not None else self.ui_font
        if theme is not None and (theme != self.theme or ui_font != self.ui_font):
            from ui.molecule_viewer_style import chrome_stylesheet, separator_stylesheet
            stylesheet = chrome_stylesheet(theme, self.assets.glyphs(theme), ui_font)
            # The divider is painted by QMainWindow, outside either styled dock.
            # Restrict its root rule to that subcontrol; never style VTK widgets.
            self.window.setStyleSheet(separator_stylesheet(theme))
            # Style the composite widget explicitly: generic QWidget does not
            # reliably paint an inherited stylesheet after native reparenting.
            for surface in (self.toolbar, self.dock, self.tabs, self.status,
                            self.controls.widget, self.controls.panel):
                surface.setStyleSheet(stylesheet)
            from ui.molecule_viewer_style import menubar_stylesheet
            self.window.menuBar().setStyleSheet(menubar_stylesheet(theme))
            if self.figure is not None:
                self.figure.widget.setStyleSheet(stylesheet)
                self.figure.footer.setStyleSheet(stylesheet)
            if self.empty_notice is not None:
                self.empty_notice.setStyleSheet(stylesheet)
            from ui.molecule_viewer_native import apply_native_palette, apply_titlebar
            app = self.widgets.QApplication.instance()
            apply_native_palette(app, self.gui, self.core, theme)
            if self.window.isVisible():
                apply_titlebar(self.window, theme)
            self.theme = dict(theme)
            self.ui_font = ui_font
        self.options_action.setEnabled(values is not None)
        if values is not None:
            self.window.setFont(self.gui.QFont(*values["ui_font"]))
            size = sidebar_width(self.window.fontMetrics())
            self.dock.setFixedWidth(size)
            self.tabs.tabBar().setFixedWidth(size - 2)
        if self.empty_notice is not None:
            self.empty_notice.setFont(self.window.font())
            height = self.window.fontMetrics().height()
            if isinstance(height, (int, float)):
                self.empty_notice.setMinimumHeight(int(height * 3 + 24))
            self.empty_notice.updateGeometry()
        text = scene_status(scene, selected_refs)
        compact = text.replace("Working atoms: ", "Atoms: ").replace(" · Reference: ", " · Ref atoms: ")
        compact = compact.replace(" | No visible atom selected", "")
        self.status.showMessage(compact)
        self.status.setToolTip(text)

    def open_tab(self, index):
        self.tabs.setCurrentIndex(index)
        self.dock.show()
        self.sync_tabs()

    def sync_tabs(self, *args):
        for index, action in enumerate([self.options_action] + ([self.figure_action] if self.figure_action is not None else [])):
            blocked = action.blockSignals(True)
            action.setChecked(self.dock.isVisible() and self.tabs.currentIndex() == index)
            action.blockSignals(blocked)

    def close(self):
        self.assets.close()

    def refresh_layout(self):
        """One deferred first-show layout/paint request; no DPI or resize policy."""
        if self.theme is not None:
            from ui.molecule_viewer_native import apply_titlebar
            apply_titlebar(self.window, self.theme)
        surfaces = [self.window, self.toolbar, self.controls.widget, self.status]
        if self.empty_notice is not None:
            surfaces.append(self.empty_notice)
        for surface in surfaces:
            surface.ensurePolished()
            layout = surface.layout()
            if layout is not None:
                layout.activate()
            surface.updateGeometry()
            surface.update()
