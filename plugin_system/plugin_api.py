# plugin_system/plugin_api.py

from __future__ import annotations


class PluginApp:
    """
    Minimal app wrapper exposed to external plugins.
    """

    def __init__(self, state: dict):
        self.state = state

    @property
    def root(self):
        return self.state.get("root")

    def add_menu_item(self, label: str, command, menu: str = "Modules"):
        menus = self.state.get("menus", {})
        target = menus.get(menu)

        if target is None:
            raise RuntimeError(f"Menu not found: {menu}")

        target.add_command(label=label, command=command)
        if menu == "Modules":
            from plugin_system.ribbon_modules import capture_launcher
            capture_launcher(self, label, command)

    def add_separator(self, menu: str = "Modules"):
        menus = self.state.get("menus", {})
        target = menus.get(menu)
        if target is not None:
            target.add_separator()

    def register_tool(
        self,
        tool_id: str,
        label: str,
        command,
        *,
        category: str = "Plugins",
        enabled=None,
        order: int = 0,
        tooltip: str = "",
    ):
        """Register a plugin launcher for menus and Control Panel pinning."""
        from plugin_system.tool_registry import ToolLauncher, ensure_tool_registry

        source = str(getattr(self, "current_plugin_id", "plugin"))
        tool = ToolLauncher(
            id=str(tool_id),
            label=str(label),
            category=str(category),
            launch=command,
            source=source,
            enabled=enabled,
            order=int(order),
            tooltip=str(tooltip),
        )
        ensure_tool_registry(self.state).register(tool)
        from plugin_system.ribbon_modules import capture_launcher
        capture_launcher(self, label, command, tool_id=tool_id,
                         category=None if category == "Plugins" else category,
                         enabled=enabled, tooltip=tooltip)
        refresh = self.state.get("refresh_control_panel_tools")
        if callable(refresh):
            refresh()
        return tool

    def get_current_structure(self):
        return self.state.get("atom_data_eff") or self.state.get("atom_data")

    def get_raw_structure(self):
        return self.state.get("atom_data_raw") or self.state.get("atom_data")

    def get_delta_exp_values(self):
        return dict(self.state.get("delta_exp_values") or {})

    def get_pcs_values_by_id(self):
        return dict(self.state.get("pcs_by_id") or {})

    def get_metal_position(self):
        return (
            float(self.state.get("x0", 0.0)),
            float(self.state.get("y0", 0.0)),
            float(self.state.get("z0", 0.0)),
        )

    def refresh_views(self):
        update_graph = self.state.get("update_graph")
        if callable(update_graph):
            try:
                update_graph()
            except Exception:
                pass

        plot_cartesian = self.state.get("plot_cartesian")
        if callable(plot_cartesian):
            try:
                plot_cartesian(self.state)
            except Exception:
                pass
