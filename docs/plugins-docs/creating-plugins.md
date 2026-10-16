# Creating Plugins

This is **plugin-author reference**. For installing and running existing tools,
see [Module Manager](overview.md).

The installer accepts a single Python file or a folder/ZIP package. In each case the installed entry module must define:

```python
def register(app):
    ...
```

`register(app)` is called while external plugins are loaded for the **Modules** menu. It should register commands and return promptly; launch the actual tool only from a deferred callback.

## Single-file plugins

A `.py` plugin must define both `PLUGIN_INFO` as a dictionary and `register(app)`:

```python
PLUGIN_INFO = {
    "id": "evans_calculator",
    "name": "Evans Method Calculator",
    "version": "0.1.0",
    "author": "Author Name",
    "description": "Calculate magnetic susceptibility from Evans method data.",
    "type": "window",
    "standalone": True,
    "dependencies": [],
}


def open_plugin(app):
    ...


def register(app):
    app.register_tool(
        tool_id="evans_calculator.open",
        label="Evans Method Calculator...",
        command=lambda: open_plugin(app),
        category="Plugins",
    )
```

During installation PCS Analyzer imports the selected file to inspect `PLUGIN_INFO`. Import-time code therefore runs before the user can use the plugin. The installer validates the ID, version and declared compatibility, copies the file to `plugins/installed/<plugin_id>/plugin.py`, and generates a manifest from selected `PLUGIN_INFO` fields. New installations are enabled; replacements retain the existing enabled state.

## Folder-based plugins

```text
bleaney_vt/
├─ manifest.json
├─ plugin.py
├─ plugin_ui/
├─ plugin_logic/
├─ resources/
└─ examples/
```

The selected folder may contain `manifest.json` directly, or it may contain one immediate child folder that does. The `entry` named by the manifest must exist. Package-style relative imports are supported:

```python
from .plugin_ui.main_window import MyWindow
from .plugin_logic.fitting import run_fit
```

The ZIP installer accepts the same layout after extraction. See [Distribution and installation](distribution.md) before sharing a ZIP, especially its current security limitations.

## Registering a deferred command

Use **register_tool()** for a launcher that can be pinned in Quick tools; add a
Modules command separately with **add_menu_item()** when desired. Give each launcher a unique stable ID to avoid replacing
another tool. The legacy **add_menu_item()** pattern below remains supported but
does not itself create a pinnable tool. See [PluginApp](plugin-api.md#registering-a-tool-launcher).

```python
def open_plugin(app):
    from .plugin_ui.main_window import BleaneyWindow
    return BleaneyWindow(app.root, app=app)


def register(app):
    app.add_menu_item(
        label="Bleaney / VT Shift Fitting...",
        command=lambda: open_plugin(app),
    )
```

Do not write `command=open_plugin(app)`: that calls the function during registration instead of when the user selects the menu item.

## Import-time safety

A plugin should not open windows, call `main()` or `mainloop()`, start calculations, launch subprocesses, load large data, or modify application state merely because it was imported. PCS Analyzer cannot enforce this rule or reverse import-time side effects.

Move optional or expensive imports into the launch function where practical, and keep standalone execution guarded:

```python
if __name__ == "__main__":
    run_standalone()
```

## GUI integration

For a Tkinter plugin launched inside PCS Analyzer, use `tk.Toplevel(app.root)` when practical. Do not create a second `tk.Tk()` root or start another event loop. A genuinely standalone path may create its own root only inside the guarded standalone entry point.

## Dependencies and resources

Manifest dependency fields are metadata only in the current host. The manager neither installs nor checks them, so plugin code must detect required and optional packages and report missing ones clearly.

Resolve resources relative to the plugin file rather than the process working directory:

```python
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = PLUGIN_DIR / "resources"
```

## Development checklist

1. Give the plugin a stable ID and define `register(app)`.
2. Keep import and registration lightweight.
3. Use deferred menu callbacks.
4. Prefer the documented [PluginApp helpers](plugin-api.md) to raw state access.
5. Handle unavailable dependencies in plugin code.
6. Install through Module Manager; restart after every update or replacement.
7. Verify that the plugin is both listed and successfully loaded.
8. Exercise enable, disable, removal, and reinstall behavior with disposable test data.
9. Test standalone mode separately if the plugin claims to support it.
