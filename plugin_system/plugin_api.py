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

    def add_separator(self, menu: str = "Modules"):
        menus = self.state.get("menus", {})
        target = menus.get(menu)
        if target is not None:
            target.add_separator()

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