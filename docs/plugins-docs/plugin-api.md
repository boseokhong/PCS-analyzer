# Plugin API

Plugins should interact with PCS Analyzer through the `PluginApp` wrapper whenever possible.

## Common API methods

Available methods may include:

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

The exact API available depends on the PCS Analyzer version.

## Registering a menu entry

```python
def register(app):
    app.add_menu_item(
        label="My Plugin...",
        command=lambda: open_plugin(app),
    )
```

Use a deferred callback so the plugin opens only when selected by the user.

## Reading application state

```python
def open_plugin(app):
    structure = app.get_current_structure()
    delta_exp = app.get_delta_exp_values()
    ...
```

Prefer dedicated accessor methods over undocumented state keys.

## State access levels

Plugins may declare `none`, `read_only`, or `read_write` access in `manifest.json`.

Most plugins should use `none` or `read_only`.

## GUI parent

```python
def open_plugin(app):
    win = tk.Toplevel(app.root)
    win.title("My Plugin")
    return win
```

Do not start a second Tk event loop inside PCS Analyzer.

## Error handling

```python
try:
    run_analysis()
except Exception as exc:
    from tkinter import messagebox
    messagebox.showerror("My Plugin", f"Analysis failed:\n\n{exc}")
```
