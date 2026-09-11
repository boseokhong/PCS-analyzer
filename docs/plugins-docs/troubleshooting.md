# Plugin Troubleshooting

## Plugin appears in Module Manager but not in the Modules menu

Possible causes:

- plugin is disabled
- `manifest.json` is missing or invalid
- entry file is missing
- `register(app)` is missing
- import or registration raised an exception
- relative import failed
- required dependency is missing

Check console or plugin-loading diagnostics.

## Plugin opens automatically during startup

Avoid:

```python
main()
```

and:

```python
command=open_plugin(app)
```

Use:

```python
command=lambda: open_plugin(app)
```

## Relative import fails

Typical error:

```text
ImportError: attempted relative import with no known parent package
```

Folder plugins should use package-style loading and relative imports where supported.

## Required dependency is missing

Declare dependencies in `manifest.json`:

```json
"dependencies": ["numpy", "scipy"]
```

Optional dependencies should be handled gracefully.

## Windows refuses to update or delete a plugin folder

Typical error:

```text
PermissionError: [WinError 5] Access is denied
```

Possible causes include an imported plugin, open plugin window, held file handle, or temporary file locking.

Recommended action:

```text
Close plugin windows, restart PCS Analyzer, and try again.
```

## Plugin resource file cannot be found

Avoid:

```python
open("resources/icon.png")
```

Prefer:

```python
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = PLUGIN_DIR / "resources"

open(RESOURCE_DIR / "icon.png", "rb")
```
