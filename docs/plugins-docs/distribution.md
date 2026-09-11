# Plugin Distribution

PCS Analyzer supports single-file, folder-based, and ZIP plugin distribution.

## Single-file distribution

Small plugins may be distributed directly as a `.py` file containing `PLUGIN_INFO` and `register(app)`.

## Folder-based distribution

```text
bleaney_vt/
├─ manifest.json
├─ plugin.py
├─ plugin_ui/
├─ plugin_logic/
├─ resources/
└─ examples/
```

## ZIP distribution

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

PCS Analyzer installs validated packages under:

```text
plugins/installed/<plugin_id>/
```

## Dependency handling

Required dependencies belong in:

```json
"dependencies": ["numpy", "scipy", "matplotlib"]
```

Optional dependencies belong in:

```json
"optional_dependencies": ["pyvista", "pyvistaqt", "imageio"]
```

Missing optional packages should be handled gracefully.

## Security

!!! warning

    Only install plugins from trusted sources. Python plugins can execute code on the user's computer.

Plugin authors should avoid unexpected network access, modifying user files without confirmation, hidden subprocess execution, modifying core files, storing credentials, or silently changing application settings.

## Updating and reinstalling

Updating an imported plugin may require restarting PCS Analyzer, especially on Windows.

## Distribution checklist

```text
[ ] Stable lowercase plugin ID
[ ] PLUGIN_INFO or manifest.json present
[ ] Folder plugins include manifest.json
[ ] Manifest includes id, name, version, entry
[ ] Entry file defines register(app)
[ ] No windows open at import time
[ ] No main() call at import time
[ ] No mainloop() inside PCS Analyzer
[ ] Menu callbacks are deferred
[ ] Required dependencies are declared
[ ] Optional dependencies fail gracefully
[ ] Resource paths are plugin-relative
[ ] Core source files are not modified
[ ] Plugin can be removed safely
[ ] Module Manager installation tested
[ ] Disable/re-enable tested
[ ] Reinstall/update tested
```
