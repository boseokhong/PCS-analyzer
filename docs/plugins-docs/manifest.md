# Plugin Manifest

Folder-based plugins use `manifest.json` to describe the plugin and its requirements.

## Minimal manifest

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

## Recommended manifest

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
  "dependencies": ["numpy", "scipy", "pandas", "matplotlib"],
  "optional_dependencies": ["pyvista", "pyvistaqt", "imageio"],
  "state_access": "read_only",
  "category": "Paramagnetic NMR"
}
```

## Required fields

| Field | Description |
|---|---|
| `id` | Stable and unique plugin identifier. |
| `name` | Display name shown in the Module Manager. |
| `version` | Plugin version. |
| `entry` | Entry Python file, usually `plugin.py`. |

## Recommended optional fields

| Field | Description |
|---|---|
| `author` | Plugin author. |
| `description` | Short plugin description. |
| `type` | Plugin type, e.g. `window`, `analysis`, `viewer`, `export`, or `tool`. |
| `standalone` | Whether the plugin can run independently. |
| `min_app_version` | Minimum PCS Analyzer version required. |
| `dependencies` | Required Python packages. |
| `optional_dependencies` | Optional Python packages. |
| `state_access` | `none`, `read_only`, or `read_write`. |
| `category` | Display/grouping category. |

## Plugin ID rules

Use:

```text
lowercase_words_with_underscores
```

Examples:

```text
bleaney_vt
evans_calculator
condon_helper
vt_lineshape
coordination_geometry_analyzer
pcs_motion_explorer
```

Changing the plugin ID should be treated as creating a new plugin.

## Dependency fields

```json
"dependencies": ["numpy", "scipy", "matplotlib"]
```

```json
"optional_dependencies": ["pyvista", "pyvistaqt", "imageio"]
```

Older aliases such as `requires` and `optional_requires` may be accepted for compatibility, but new plugins should use the current field names.

## State access

| Value | Meaning |
|---|---|
| `none` | Plugin does not use PCS Analyzer state. |
| `read_only` | Plugin reads structures, shifts, PCS values, or settings. |
| `read_write` | Plugin may modify application state. |

Most plugins should use `none` or `read_only`.

## Versioning

Plugin versions should follow semantic versioning where possible:

```text
MAJOR.MINOR.PATCH
```

## Compatibility

A plugin may declare:

```json
"min_app_version": "1.4.0"
```

## Reserved fields

`entry_function` is reserved for possible future loader support. Current plugin loading is based on `register(app)`.
