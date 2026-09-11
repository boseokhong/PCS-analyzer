# Plugins

PCS Analyzer supports external plugins that extend the application without modifying the core source code.

Plugins can add analysis tools, viewers, fitting utilities, data converters, export tools, and other research-specific functionality to the **Modules** menu.

PCS Analyzer supports two user-facing plugin distribution styles:

- **Single-file Python plugins** (`.py`)
- **Folder-based plugins**, optionally distributed as `.zip` packages

Internally, installed plugins use:

```text
plugins/installed/<plugin_id>/
```

A minimal installed plugin contains:

```text
plugins/installed/<plugin_id>/
├─ manifest.json
└─ plugin.py
```

More complex plugins may include `plugin_ui/`, `plugin_logic/`, `resources/`, and `examples/`.

## Core design principle

PCS Analyzer plugins should be as independent as possible. Plugins should not require modification of `main.py`, `ui/`, `logic/`, or other core source files.

A plugin registers itself through:

```python
def register(app):
    ...
```

The application loads the plugin and calls:

```python
plugin_module.register(plugin_app)
```

The plugin can then add its own menu entry through the provided `PluginApp` object.

## Terminology

| Term | Meaning |
|---|---|
| **Plugin** | External module installed under `plugins/installed/<plugin_id>/`. |
| **Built-in module** | Tool shipped as part of PCS Analyzer itself. |
| **Module Manager** | GUI used to install, enable, disable, and remove external plugins. |
| **PluginApp** | Minimal API wrapper passed to plugins by PCS Analyzer. |
| **Registry** | `plugins/plugins.json`, storing plugin metadata and enabled state. |
| **Entry file** | Python file loaded by the plugin loader, usually `plugin.py`. |

## Plugin types

### Single-file plugins

Suitable for calculators, converters, quick plotting tools, export helpers, and prototype analysis tools.

### Folder-based plugins

Recommended for larger modules such as VT fitting, line-shape analysis, coordination geometry tools, crystal-field utilities, and PCS ensemble analysis.

## Module Manager

The Module Manager can install, enable, disable, remove, and refresh external plugins.

!!! note

    A plugin visible in the Module Manager has not necessarily loaded successfully. It must also import correctly and define `register(app)`.

## Mandatory rules

A valid PCS Analyzer plugin should:

1. use a stable and unique plugin ID
2. define `register(app)`
3. avoid running tool code at import time
4. use deferred menu callbacks
5. avoid modifying PCS Analyzer core files
6. declare required dependencies
7. handle missing optional dependencies gracefully
8. keep standalone execution behind `if __name__ == "__main__"`

[Creating Plugins](creating-plugins.md){ .md-button .md-button--primary }
[Manifest Reference](manifest.md){ .md-button }
