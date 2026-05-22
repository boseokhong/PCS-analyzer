# PCS Analyzer Plugin Distribution Guide

**Document status:** Draft specification  
**Target application:** PCS Analyzer  
**Recommended scope:** External plugin authors, internal module developers, and future maintainers

This document defines the recommended packaging, installation, loading, and distribution rules for PCS Analyzer plugins.

PCS Analyzer supports two user-facing plugin distribution styles:

1. **Single-file Python plugins** (`.py`)
2. **Folder-based plugins**, optionally distributed as `.zip` packages

Internally, all plugins are installed as folder-based packages under:

```text
plugins/installed/<plugin_id>/
```

Even when a user installs a single `.py` plugin, PCS Analyzer converts it into the standard installed folder layout during installation.

---

## 1. Core Design Principle

PCS Analyzer plugins should be as independent as possible.

The main application provides only:

- the `Modules` menu
- the plugin loading system
- the Module Manager
- a minimal `PluginApp` API
- optional read-only access to selected PCS Analyzer state

A plugin should not require modification of:

```text
ui/components.py
logic/
ui/
main.py
```

or any other core PCS Analyzer source file.

A plugin registers itself through:

```python
def register(app):
    ...
```

The main application loads the plugin and calls:

```python
plugin_module.register(plugin_app)
```

The plugin is responsible for adding its own menu entry through the provided `PluginApp` object.

---

## 2. Terminology

| Term | Meaning |
|---|---|
| **Plugin** | An external module installed under `plugins/installed/<plugin_id>/`. |
| **Built-in module** | A tool shipped as part of the PCS Analyzer core distribution, such as PCS Workbench. |
| **Module Manager** | GUI used to install, enable, disable, and remove external plugins. |
| **PluginApp** | Minimal API wrapper passed to plugins by PCS Analyzer. |
| **Registry** | The `plugins/plugins.json` file that stores installed plugin metadata and enabled state. |
| **Entry file** | The Python file loaded by the plugin loader, usually `plugin.py`. |

Both built-in modules and external plugins may be listed under the `Modules` menu. The Module Manager controls only external plugins.

---

## 3. Standard Installed Plugin Layout

Every installed plugin must have the following internal structure:

```text
plugins/installed/<plugin_id>/
├─ manifest.json
└─ plugin.py
```

For complex plugins, additional folders may be included:

```text
plugins/installed/<plugin_id>/
├─ manifest.json
├─ plugin.py
├─ plugin_ui/
│  ├─ __init__.py
│  └─ main_window.py
├─ plugin_logic/
│  ├─ __init__.py
│  ├─ models.py
│  └─ fitting.py
├─ resources/
│  └─ icon.png
└─ examples/
   └─ sample_data.csv
```

The plugin loader reads `manifest.json`, finds the entry file, imports it, and calls `register(app)`.

---

## 4. Single-File Plugin Distribution

Single-file plugins are recommended for small utilities, for example:

- simple calculators
- data converters
- quick plotting tools
- small export helpers
- prototype analysis tools

A single-file plugin may be distributed as:

```text
evans_calculator.py
```

It must contain at least:

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
    # Open a Toplevel window, launch a standalone tool, or call plugin UI code.
    ...
```

When installed through the Module Manager, PCS Analyzer converts it to:

```text
plugins/installed/evans_calculator/
├─ manifest.json
└─ plugin.py
```

The generated `manifest.json` is derived from `PLUGIN_INFO`.

Users should not need to manually create `manifest.json` for single-file plugins.

---

## 5. Folder-Based Plugin Distribution

Folder-based plugins are recommended for larger analysis modules, for example:

- Bleaney / VT shift fitting
- VT-NMR line-shape analysis
- CShM / coordination geometry analysis
- CONDON input/output helper
- crystal-field fitting utilities
- advanced paramagnetic NMR model fitting
- PCS ensemble or motion analysis tools

A folder-based plugin should be distributed as:

```text
bleaney_vt/
├─ manifest.json
├─ plugin.py
├─ plugin_ui/
├─ plugin_logic/
├─ resources/
└─ examples/
```

The folder must contain:

1. a valid `manifest.json`
2. the entry file specified by `manifest.json`, usually `plugin.py`
3. a `register(app)` function in the entry file

---

## 6. ZIP Plugin Distribution

For public distribution, complex plugins may be packaged as `.zip` files.

Recommended structure:

```text
bleaney_vt.zip
└─ bleaney_vt/
   ├─ manifest.json
   ├─ plugin.py
   ├─ plugin_ui/
   ├─ plugin_logic/
   ├─ resources/
   └─ examples/
```

During installation, PCS Analyzer should:

1. extract the zip to a temporary folder
2. locate `manifest.json`
3. validate the plugin ID and entry file
4. copy the plugin folder to `plugins/installed/<plugin_id>/`
5. register the plugin in `plugins/plugins.json`
6. refresh the `Modules` menu if supported

---

## 7. `manifest.json` Format

Every installed folder-based plugin must contain a `manifest.json`.

Minimum example:

```json
{
  "id": "test_plugin",
  "name": "Test Plugin",
  "version": "0.1.0",
  "author": "Author Name",
  "description": "Simple plugin loading test.",
  "entry": "plugin.py",
  "type": "window",
  "standalone": true
}
```

Recommended full example:

```json
{
  "id": "bleaney_vt",
  "name": "Bleaney / VT Shift Fitting",
  "version": "0.1.0",
  "author": "Author Name",
  "description": "Temperature-dependent paramagnetic shift fitting module.",
  "entry": "plugin.py",
  "type": "window",
  "standalone": true,
  "min_app_version": "1.4.0",
  "dependencies": [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib"
  ],
  "optional_dependencies": [
    "pyvista",
    "pyvistaqt",
    "imageio"
  ],
  "state_access": "read_only",
  "category": "Paramagnetic NMR"
}
```

### Required fields

| Field | Description |
|---|---|
| `id` | Stable and unique plugin identifier. Use lowercase letters, numbers, and underscores. |
| `name` | Display name shown in the Module Manager. |
| `version` | Plugin version. |
| `entry` | Entry Python file, usually `plugin.py`. |

### Recommended optional fields

| Field | Description |
|---|---|
| `author` | Plugin author. |
| `description` | Short plugin description. |
| `type` | Plugin type, for example `window`, `analysis`, `viewer`, `export`, or `tool`. |
| `standalone` | Whether the plugin can run independently. |
| `min_app_version` | Minimum PCS Analyzer version required. |
| `dependencies` | Required Python packages. |
| `optional_dependencies` | Optional Python packages used for enhanced functionality. |
| `state_access` | Intended state access level: `none`, `read_only`, or `read_write`. |
| `category` | Display or grouping category. |

### Deprecated aliases

New plugins should use:

```json
"dependencies": [],
"optional_dependencies": []
```

The following older aliases may be accepted by the installer for compatibility, but should not be used in new plugins:

```json
"requires": [],
"optional_requires": []
```

### Reserved fields

The following field is reserved for possible future loader support:

```json
"entry_function": "open_plugin"
```

Current PCS Analyzer plugin loading is based on `register(app)`. Therefore, plugins must not rely on `entry_function` unless the loader is explicitly updated to support it.

---

## 8. Plugin ID Rules

The plugin ID must be stable and unique.

Recommended format:

```text
lowercase_words_with_underscores
```

Good examples:

```text
bleaney_vt
evans_calculator
condon_helper
vt_lineshape
coordination_geometry_analyzer
pcs_motion_explorer
```

Avoid:

```text
Bleaney Module
my plugin!!!
test
plugin1
```

The plugin ID is used for:

- installation folder name
- registry key
- import namespace
- enable/disable state
- removal
- update/reinstall matching

Changing the plugin ID should be treated as creating a new plugin.

---

## 9. Entry File Rules

The entry file is usually:

```text
plugin.py
```

It must define:

```python
def register(app):
    ...
```

Recommended structure:

```python
PLUGIN_INFO = {
    "id": "bleaney_vt",
    "name": "Bleaney / VT Shift Fitting",
    "version": "0.1.0",
    "author": "Author Name",
    "description": "Temperature-dependent paramagnetic shift fitting module.",
}


def register(app):
    app.add_menu_item(
        label="Bleaney / VT Shift Fitting...",
        command=lambda: open_plugin(app),
    )


def open_plugin(app):
    from .plugin_ui.main_window import BleaneyWindow
    return BleaneyWindow(app.root, app=app)


def run_standalone():
    import tkinter as tk
    from .plugin_ui.main_window import BleaneyWindow

    root = tk.Tk()
    root.title("Bleaney / VT Shift Fitting")
    BleaneyWindow(root, app=None)
    root.mainloop()


if __name__ == "__main__":
    run_standalone()
```

For a single-file plugin, the relative imports above should be replaced by local functions or absolute imports appropriate for the file.

---

## 10. Import-Time Safety Rules

This is one of the most important rules.

A plugin must not perform any of the following at import time:

- open a GUI window
- call `main()`
- call `mainloop()`
- start calculations
- launch subprocesses
- load large files unnecessarily
- modify PCS Analyzer state
- show message boxes unless dependency validation explicitly requires it

Only definitions, constants, lightweight imports, and small helper functions should run at import time.

### Incorrect

```python
from my_tool.standalone import main
main()
```

This launches the plugin immediately when the plugin loader imports the file.

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

Execution must happen only after the user clicks the menu item.

---

## 11. Menu Callback Rules

Plugin menu callbacks must be deferred.

### Correct

```python
def register(app):
    app.add_menu_item(
        label="My Plugin...",
        command=lambda: open_plugin(app),
    )
```

### Incorrect

```python
def register(app):
    app.add_menu_item(
        label="My Plugin...",
        command=open_plugin(app),
    )
```

The incorrect form calls `open_plugin(app)` immediately during registration. This can cause the plugin window to open during PCS Analyzer startup.

---

## 12. Folder Plugin Import Rules

Folder-based plugins should be treated as package-like modules.

Recommended folder plugin imports:

```python
from .plugin_ui.main_window import MyWindow
from .plugin_logic.fitting import run_fit
```

This requires the PCS Analyzer plugin loader to load the entry module with package-style import support, for example:

```python
spec = importlib.util.spec_from_file_location(
    module_name,
    entry_path,
    submodule_search_locations=[str(plugin_dir)],
)
```

The loader should also register the module in `sys.modules` before executing it:

```python
sys.modules[module_name] = module
spec.loader.exec_module(module)
```

This is required for:

- relative imports such as `from .ui_dialog import MyDialog`
- `@dataclass`
- introspection-heavy decorators
- callbacks that refer to the plugin module after import

### Alternative local import style

If package-style loading is not available, plugin authors may use direct local imports:

```python
from ui_dialog import MyDialog
from plugin_logic.fitting import run_fit
```

However, this is less robust and may conflict with other plugins. Package-style relative imports are preferred for folder plugins.

---

## 13. Standalone Compatibility

A plugin may also run as a standalone tool.

The standalone entry point must be protected by:

```python
if __name__ == "__main__":
    run_standalone()
```

Never call standalone code directly at import time.

Recommended pattern:

```python
def run_standalone():
    import tkinter as tk
    root = tk.Tk()
    root.title("My Plugin")
    MyWindow(root)
    root.mainloop()


if __name__ == "__main__":
    run_standalone()
```

---

## 14. `PluginApp` API Usage

Plugins should interact with PCS Analyzer through the `PluginApp` wrapper whenever possible.

Common API methods may include:

```python
app.root
app.state
app.add_menu_item(label, command)
app.add_separator()
app.get_current_structure()
app.get_raw_structure()
app.get_delta_exp_values()
app.get_pcs_values_by_id()
app.get_metal_position()
app.refresh_views()
```

Recommended use:

```python
def open_plugin(app):
    structure = app.get_current_structure()
    delta_exp = app.get_delta_exp_values()
    ...
```

Direct modification of `app.state` should be avoided unless the plugin is explicitly designed as an integrated plugin.

---

## 15. State Access Levels

A plugin should declare its intended state access level in `manifest.json`:

```json
"state_access": "read_only"
```

Recommended values:

| Value | Meaning |
|---|---|
| `none` | Plugin does not use PCS Analyzer state. |
| `read_only` | Plugin reads structures, shifts, PCS values, or settings. |
| `read_write` | Plugin may modify PCS Analyzer state. Use only for tightly integrated plugins. |

Most plugins should use `none` or `read_only`.

Examples:

```json
{
  "type": "viewer",
  "state_access": "read_only"
}
```

```json
{
  "type": "integrated",
  "state_access": "read_write"
}
```

---

## 16. Dependency Handling

Plugins should list required dependencies in `manifest.json`:

```json
"dependencies": ["numpy", "scipy", "matplotlib"]
```

Optional dependencies should be listed separately:

```json
"optional_dependencies": ["pyvista", "pyvistaqt", "imageio"]
```

A plugin should handle missing optional dependencies gracefully.

Recommended runtime pattern:

```python
def _check_required_dependencies(show_gui=False):
    missing = []

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import scipy
    except ImportError:
        missing.append("scipy")

    if missing:
        msg = "Missing required packages:\n" + "\n".join(missing)
        if show_gui:
            from tkinter import messagebox
            messagebox.showerror("Plugin dependency error", msg)
        else:
            print(msg)
        return False

    return True
```

Dependency checks should not launch the main plugin GUI at import time.

---

## 17. GUI Integration Rules

For GUI plugins:

- Use `tk.Toplevel(app.root)` for child windows when possible.
- Do not call `tk.Tk()` when running inside PCS Analyzer.
- Do not call `mainloop()` when running inside PCS Analyzer.
- `tk.Tk()` and `mainloop()` are allowed only in standalone mode.

Recommended pattern:

```python
def open_plugin(app):
    win = tk.Toplevel(app.root)
    win.title("My Plugin")
    ...
    return win
```

Standalone pattern:

```python
def run_standalone():
    root = tk.Tk()
    root.title("My Plugin")
    MyWindow(root)
    root.mainloop()
```

---

## 18. Security Rules

Python plugins can execute arbitrary code on the user's computer.

The Module Manager should display a warning similar to:

```text
Only install modules from trusted sources.
Python plugins can execute code on your computer.
```

The installer should require confirmation before installing a plugin.

Plugin authors should avoid:

- unexpected network access
- modifying user files without confirmation
- hidden subprocess execution
- modifying PCS Analyzer core files
- storing credentials or private data
- silently changing application settings

---

## 19. Update and Reinstallation Policy

Updating a plugin may require restarting PCS Analyzer.

On Windows, an already imported plugin may prevent its folder from being deleted or replaced. The installer should avoid directly deleting active plugin folders when possible.

Recommended update strategy:

1. Rename the existing installed folder to a backup folder.
2. Copy the new plugin folder into `plugins/installed/<plugin_id>/`.
3. Update `plugins/plugins.json`.
4. Refresh the `Modules` menu.
5. Attempt to delete the backup folder.
6. If deletion fails, leave a `.delete_on_restart` marker and instruct the user to restart PCS Analyzer.

Example backup folder:

```text
plugins/installed/bleaney_vt.__old__20260516_153000/
```

If a plugin has already been imported, a clean restart is recommended after updating.

---

## 20. Module Manager Behavior

The Module Manager should support:

- listing installed plugins
- enabling plugins
- disabling plugins
- removing plugins
- installing `.py` plugins
- installing folder-based plugins
- installing `.zip` plugins
- refreshing the main `Modules` menu

The Module Manager reads from:

```text
plugins/plugins.json
```

The main `Modules` menu is updated only when the plugin loader successfully imports the plugin and calls `register(app)`.

Therefore:

```text
Visible in Module Manager ≠ successfully loaded into the main menu
```

If a plugin appears in Module Manager but not in the main menu, likely causes include:

- `enabled` is false
- `manifest.json` is missing or invalid
- entry file is missing
- `register(app)` is missing
- the plugin raises an exception during import
- the plugin raises an exception during `register(app)`
- the plugin uses unsupported relative imports
- required dependencies are missing

The loader should report failed plugin loads to the console or a diagnostics view.

---

## 21. Registry Format: `plugins/plugins.json`

The registry stores installed plugin records.

Example:

```json
{
  "vt_nmr_analysis": {
    "id": "vt_nmr_analysis",
    "name": "VT-NMR Analysis",
    "version": "3.2.0",
    "dir": "vt_nmr_analysis",
    "enabled": true
  },
  "xyz_compare_viewer": {
    "id": "xyz_compare_viewer",
    "name": "XYZ Compare Viewer",
    "version": "0.1.0",
    "dir": "xyz_compare_viewer",
    "enabled": true
  }
}
```

The registry should be treated as application-managed data. Plugin authors should not ask users to edit it manually except for debugging.

---

## 22. Loader Requirements

A robust PCS Analyzer plugin loader should:

1. read `plugins/plugins.json`
2. skip disabled plugins
3. resolve `plugins/installed/<plugin_id>/`
4. read `manifest.json`
5. validate `entry`
6. add the plugin folder to `sys.path` if needed
7. import the entry file using package-style loading
8. register the module in `sys.modules` before execution
9. verify that `register(app)` exists
10. call `register(app)`
11. store loading results and errors
12. avoid crashing the whole application when one plugin fails

Recommended import pattern:

```python
module_name = f"pcs_plugin_{plugin_id}"

spec = importlib.util.spec_from_file_location(
    module_name,
    entry_path,
    submodule_search_locations=[str(plugin_dir)],
)
module = importlib.util.module_from_spec(spec)

if spec.loader is None:
    raise RuntimeError("Could not create plugin loader.")

sys.modules[module_name] = module
spec.loader.exec_module(module)

if not hasattr(module, "register"):
    raise RuntimeError("register(app) function is missing.")

module.register(app)
```

The loader should not call plugin functions other than `register(app)` during startup.

---

## 23. Recommended Plugin Folder Naming

Use plugin-local folder names to avoid conflicts with PCS Analyzer core modules.

Recommended:

```text
plugin_ui/
plugin_logic/
resources/
examples/
```

Avoid using generic names that may conflict with the host application or other plugins:

```text
ui/
logic/
core/
utils/
```

If a plugin is loaded as a proper package using relative imports, this risk is reduced. However, plugin-local names are still clearer.

---

## 24. Resource and Data File Access

Plugins should resolve their own resource paths relative to the plugin file.

Recommended:

```python
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = PLUGIN_DIR / "resources"
EXAMPLE_DIR = PLUGIN_DIR / "examples"
```

Avoid relying on the current working directory.

Incorrect:

```python
open("resources/icon.png")
```

Correct:

```python
open(RESOURCE_DIR / "icon.png", "rb")
```

---

## 25. Versioning Policy

Plugin versions should follow semantic versioning where possible:

```text
MAJOR.MINOR.PATCH
```

Example:

```text
0.1.0
1.0.0
1.2.3
```

Recommended interpretation:

| Version part | Meaning |
|---|---|
| MAJOR | Breaking changes. |
| MINOR | New features without breaking compatibility. |
| PATCH | Bug fixes. |

---

## 26. Compatibility Policy

Plugins may declare the minimum required PCS Analyzer version:

```json
"min_app_version": "1.4.0"
```

Future plugin managers may refuse to load plugins requiring a newer PCS Analyzer version.

Plugins should avoid relying on undocumented internal state keys unless declared as `state_access: "read_write"` or marked as an internal/integrated plugin.

---

## 27. Logging and Error Handling

Plugins should catch user-facing errors and report them clearly.

Recommended:

```python
try:
    run_analysis()
except Exception as exc:
    from tkinter import messagebox
    messagebox.showerror("My Plugin", f"Analysis failed:\n\n{exc}")
```

Plugin loaders should catch import and registration errors and continue loading other plugins.

---

## 28. Distribution Checklist

Before distributing a plugin, verify:

```text
[ ] The plugin has a stable lowercase plugin ID.
[ ] The plugin defines PLUGIN_INFO or manifest.json.
[ ] Folder-based plugins include manifest.json.
[ ] The manifest has id, name, version, and entry.
[ ] The entry file defines register(app).
[ ] The plugin does not open windows at import time.
[ ] The plugin does not call main() at import time.
[ ] The plugin does not call mainloop() inside PCS Analyzer.
[ ] Menu callbacks use command=lambda: open_plugin(app).
[ ] Required dependencies are listed.
[ ] Missing optional dependencies are handled gracefully.
[ ] Resource paths are resolved relative to the plugin folder.
[ ] The plugin does not modify PCS Analyzer core source files.
[ ] The plugin can be removed without breaking the core application.
[ ] The plugin was tested after installation through Module Manager.
[ ] The plugin was tested after disabling and re-enabling.
[ ] The plugin was tested after reinstalling/updating.
```

---

## 29. Minimal Single-File Plugin Template

```python
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


PLUGIN_INFO = {
    "id": "minimal_plugin",
    "name": "Minimal Plugin",
    "version": "0.1.0",
    "author": "Author Name",
    "description": "A minimal PCS Analyzer plugin.",
    "type": "window",
    "standalone": True,
    "dependencies": [],
}


def open_plugin(app):
    win = tk.Toplevel(app.root)
    win.title("Minimal Plugin")
    win.geometry("360x160")

    frame = ttk.Frame(win, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Minimal plugin loaded successfully.",
    ).pack(anchor="w")

    return win


def register(app):
    app.add_menu_item(
        label="Minimal Plugin...",
        command=lambda: open_plugin(app),
    )


def run_standalone():
    root = tk.Tk()
    root.title("Minimal Plugin")

    class DummyApp:
        root = root
        state = {}

    open_plugin(DummyApp())
    root.mainloop()


if __name__ == "__main__":
    run_standalone()
```

---

## 30. Minimal Folder-Based Plugin Template

```text
my_analysis_plugin/
├─ manifest.json
├─ plugin.py
├─ plugin_ui/
│  ├─ __init__.py
│  └─ main_window.py
└─ plugin_logic/
   ├─ __init__.py
   └─ analysis.py
```

### `manifest.json`

```json
{
  "id": "my_analysis_plugin",
  "name": "My Analysis Plugin",
  "version": "0.1.0",
  "author": "Author Name",
  "description": "Example folder-based analysis plugin.",
  "entry": "plugin.py",
  "type": "analysis",
  "standalone": true,
  "dependencies": ["numpy", "matplotlib"],
  "state_access": "read_only"
}
```

### `plugin.py`

```python
from __future__ import annotations


PLUGIN_INFO = {
    "id": "my_analysis_plugin",
    "name": "My Analysis Plugin",
    "version": "0.1.0",
}


def open_plugin(app):
    from .plugin_ui.main_window import MyAnalysisWindow
    return MyAnalysisWindow(parent=app.root, app=app)


def register(app):
    app.add_menu_item(
        label="My Analysis Plugin...",
        command=lambda: open_plugin(app),
    )


def run_standalone():
    import tkinter as tk
    from .plugin_ui.main_window import MyAnalysisWindow

    root = tk.Tk()
    root.title("My Analysis Plugin")
    MyAnalysisWindow(parent=root, app=None)
    root.mainloop()


if __name__ == "__main__":
    run_standalone()
```

---

## 31. Common Failure Modes

### Plugin appears in Module Manager but not in the main menu

Likely causes:

```text
- enabled is false
- register(app) is missing
- import failed
- dependency missing
- relative import failed
- plugin opened itself and crashed during import
```

Check `state["plugin_load_results"]` or console diagnostics.

### Plugin opens automatically on startup

Likely causes:

```python
main()
```

was called at import time, or the menu callback was written as:

```python
command=open_plugin(app)
```

instead of:

```python
command=lambda: open_plugin(app)
```

### Relative import fails

Error:

```text
ImportError: attempted relative import with no known parent package
```

Fix:

- update the loader to use `submodule_search_locations=[str(plugin_dir)]`, or
- avoid relative imports and use local absolute imports.

Package-style relative imports are preferred.

### Windows refuses to update/delete plugin folder

Error:

```text
PermissionError: [WinError 5] Access is denied
```

Likely causes:

- plugin is currently imported
- plugin window is open
- Python process holds file handles
- IDE, antivirus, or OS indexing temporarily locks files

Recommended action:

```text
Close plugin windows, restart PCS Analyzer, and try again.
```

The installer should use backup-folder replacement instead of direct deletion.

---

## 32. Recommended Development Workflow

For new plugins:

1. Develop as standalone tool first.
2. Add `PLUGIN_INFO`.
3. Add `open_plugin(app)`.
4. Add `register(app)`.
5. Ensure no code runs at import time.
6. Install through Module Manager.
7. Test loading from the `Modules` menu.
8. Test disable/enable.
9. Test reinstall/update.
10. Package as `.py`, folder, or `.zip`.

---

## 33. Summary of Mandatory Rules

A valid PCS Analyzer plugin must obey the following rules:

```text
1. It must have a stable plugin ID.
2. It must define register(app).
3. It must not execute the tool at import time.
4. It must use deferred menu callbacks, usually command=lambda: open_plugin(app).
5. It must not modify PCS Analyzer core files.
6. It must declare dependencies when distributed publicly.
7. It must handle missing optional dependencies gracefully.
8. It must keep standalone execution behind if __name__ == "__main__".
```

These rules are intended to keep PCS Analyzer modular, stable, and safe as the plugin ecosystem grows.
