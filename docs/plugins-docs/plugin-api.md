# PluginApp Reference

This page is for plugin authors; [Module Manager](overview.md) covers end-user
installation and diagnostics.

The current `PluginApp` object is a small convenience wrapper around the application's shared state. It is unversioned, exposes mutable internals, and should not be treated as a stable or mature public extension API.

PCS Analyzer calls exactly one plugin entry point:

```python
def register(app):
    ...
```

The manifest cannot select a different function.

## Implemented surface

| Member | Current behavior |
|---|---|
| `app.root` | Returns `state["root"]`, normally the Tk root. |
| `app.state` | The raw, mutable shared state dictionary. |
| `register_tool(tool_id, label, command, *, category="Plugins", enabled=None, order=0, tooltip="")` | Registers a deferred launcher for Quick tools pinning and, during successful plugin registration, the Tools Ribbon; does not itself add a menu item; returns a ToolLauncher. `enabled`, when supplied, is a callable predicate. Use a unique stable tool ID. |
| `add_menu_item(label, command, menu="Modules")` | Adds a command to a named host menu; raises if the menu is unavailable. Modules-menu commands registered by successfully loaded plugins also appear in the Tools Ribbon. |
| `add_separator(menu="Modules")` | Adds a separator when the named menu exists. |
| `get_current_structure()` | Returns `atom_data_eff`, falling back to `atom_data`. |
| `get_raw_structure()` | Returns `atom_data_raw`, falling back to `atom_data`. |
| `get_delta_exp_values()` | Returns a shallow dictionary copy of `delta_exp_values`, or an empty dictionary. |
| `get_pcs_values_by_id()` | Returns a shallow dictionary copy of `pcs_by_id`, or an empty dictionary. |
| `get_metal_position()` | Returns `(x0, y0, z0)` converted to floats, defaulting missing coordinates to zero. |
| `refresh_views()` | Attempts the registered `update_graph()` and `plot_cartesian(state)` callbacks. Exceptions from either callback are currently suppressed. |

Return values are current application data structures, not frozen public models. Accessors that return dictionaries make only shallow copies.

## Registering a tool launcher

```python
def register(app):
    app.register_tool(
        tool_id="my_tool.open",
        label="My Tool...",
        command=lambda: open_plugin(app),
        category="Plugins",
        tooltip="Open My Tool",
    )
```

Users can pin registered launchers
through **Quick tools → Customize Tools**. Add a menu command separately with
`add_menu_item()` when needed. Registration creates a launcher,
not a scientific-data service or a permission boundary. A repeated tool ID
replaces that registered launcher, so avoid IDs belonging to other plugins.

Ribbon icon categories are **NMR / PCS**, **Structure**, **Visualization**,
and **Other / Uncategorized**. An explicit launcher category takes precedence
over the manifest category; the default `Plugins` uses the manifest category,
then the generic fallback. Legacy `PCS / Dynamics` and `Structure Analysis`
map to NMR / PCS and Structure icons respectively. Existing free-text category
values remain valid. Ribbon order is controlled by the user, independently of
the `order` value used by the Quick tools registry.

Modules-menu-only registration is supported without modifying existing modules.
Separators and commands targeting other menus are not Ribbon launchers. Reusing
the same callback object through both registration methods produces one Ribbon
entry; separate callback objects are separate commands. Keep launcher IDs and
legacy menu labels stable to retain the user's saved arrangement.

## Registering a menu entry

```python
def open_plugin(app):
    ...


def register(app):
    app.add_menu_item(
        label="My Plugin...",
        command=lambda: open_plugin(app),
    )
```

The deferred callback prevents the tool from opening while PCS Analyzer imports and registers plugins.

## Reading data

Prefer the named helpers when they cover the required data:

```python
def open_plugin(app):
    structure = app.get_current_structure()
    delta_exp = app.get_delta_exp_values()
    metal_xyz = app.get_metal_position()
```

The manifest value `state_access` is descriptive only. It does not limit access, and `app.state` remains publicly reachable and writable. Raw state keys, widget references, callbacks, and value shapes can change without a Plugin API compatibility guarantee. A plugin that mutates them can invalidate caches, bypass refresh paths, or corrupt scientific state.

## GUI integration and errors

Use `tk.Toplevel(app.root)` for Tkinter child windows and do not start a second `mainloop()` inside PCS Analyzer. Report plugin workflow errors at the plugin UI boundary. Import and `register(app)` exceptions are caught by the host loader and recorded as load failures, but errors raised later by a plugin command remain the plugin's responsibility.

## Future direction

A future integrated Plugin API is intended to sit on stable application services, make units and conventions explicit, and expose narrower read-only capabilities before writable ones. That is design direction only. Current plugins must not assume API version negotiation, capability permissions
or schema stability. Manifest application-version checks are implemented,
but they are separate from API compatibility guarantees.

See [Creating Plugins](creating-plugins.md) for the current entry pattern and the [distribution design notes](../plugin_distribution_guide_v0_1.md) for future integrated API proposals.
