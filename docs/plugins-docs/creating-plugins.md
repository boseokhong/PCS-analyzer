# Creating Plugins

PCS Analyzer plugins can be distributed either as a single Python file or as a folder-based package.

## Single-file plugins

A small plugin may be distributed as:

```text
evans_calculator.py
```

It should contain at least:

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

def register(app):
    app.add_menu_item(
        label="Evans Method Calculator...",
        command=lambda: open_plugin(app),
    )

def open_plugin(app):
    ...
```

When installed through the Module Manager, PCS Analyzer converts it to the standard installed-folder layout.

## Folder-based plugins

Recommended structure:

```text
bleaney_vt/
├─ manifest.json
├─ plugin.py
├─ plugin_ui/
├─ plugin_logic/
├─ resources/
└─ examples/
```

A folder-based plugin must contain a valid `manifest.json`, the configured entry file, and a `register(app)` function.

Package-style relative imports are preferred:

```python
from .plugin_ui.main_window import MyWindow
from .plugin_logic.fitting import run_fit
```

## Entry file

The entry file is usually `plugin.py` and must define:

```python
def register(app):
    ...
```

Recommended pattern:

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

## Import-time safety

A plugin must not open windows, call `main()`, call `mainloop()`, start calculations, launch subprocesses, load large files unnecessarily, or modify PCS Analyzer state at import time.

### Correct

```python
def open_plugin(app):
    from my_tool.standalone import main
    return main()

def register(app):
    app.add_menu_item(
        label="My Tool...",
        command=lambda: open_plugin(app),
    )
```

Execution should occur only after the user launches the plugin.

## GUI integration

When running inside PCS Analyzer:

- use `tk.Toplevel(app.root)` where possible
- do not create a second `tk.Tk()` root
- do not call `mainloop()`

Standalone execution may use its own root, protected by:

```python
if __name__ == "__main__":
    run_standalone()
```

## Resource paths

Resolve resources relative to the plugin file:

```python
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = PLUGIN_DIR / "resources"
```

Avoid relying on the current working directory.

## Recommended development workflow

1. develop the tool independently where practical
2. add `PLUGIN_INFO`
3. add `open_plugin(app)`
4. add `register(app)`
5. ensure no tool code runs during import
6. install through the Module Manager
7. test loading from the **Modules** menu
8. test disable/enable
9. test reinstall/update
10. package as `.py`, folder, or `.zip`
