# PCS Analyzer Development and Project Format Guidelines

This document defines coding and data-format rules for future PCS Analyzer development.  
The goal is to keep the codebase reproducible, extensible, and easier to debug as new features are added.

---

## 1. Core Development Principles

### 1.1 Preserve reproducibility first

Any new feature that changes analysis results must have a clearly defined state representation.

A feature should be reproducible from:

```text
loaded structure
metal center
coordinate transformations
PCS/NMR input data
model parameters
user-selected options
```

If a result depends on a UI setting, that setting should either be saved in the `.pcsp` project file or be intentionally excluded with a clear reason.

### 1.2 Separate computation from UI

Use this separation whenever practical:

```text
logic/
    calculation, parsing, fitting, export, data conversion

ui/
    Tkinter windows, widgets, callbacks, visual controls
```

A function in `logic/` should not create Tkinter widgets unless it is already an established UI-helper exception.  
If a logic file must contain UI code, document that clearly at the top of the file.

Recommended pattern:

```python
# logic/example_model.py
def compute_result(input_data, parameters):
    ...

# ui/example_window.py
def open_example_window(state):
    result = compute_result(...)
    ...
```

### 1.3 Keep user-visible behavior stable

Before changing an existing function, check whether it affects:

```text
PCS calculation
coordinate transformation
fitting result
project load/save
export format
existing GUI workflow
```

For changes that affect scientific output, prefer adding a new option rather than silently changing behavior.

---

## 2. State Management Rules

PCS Analyzer currently uses a shared `state` dictionary. New features should follow consistent naming and lifetime rules.

### 2.1 Use explicit state keys

Use descriptive names:

```python
state["pcs_scene_kwargs"]
state["delta_exp_values"]
state["fit_override"]
state["atom_data_raw"]
```

Avoid vague names:

```python
state["data"]
state["result"]
state["tmp"]
state["flag"]
```

### 2.2 Distinguish persistent state from runtime handles

Persistent state may be saved to `.pcsp`.

Examples:

```text
atom_data_raw
atom_data_eff
atom_ids_raw
atom_ids_eff
metal_ref_id
x0, y0, z0
delta_exp_values
delta_obs_values
delta_dia_values
fit_override
last_fit_result
pcs_scene_kwargs
```

Runtime handles must not be saved.

Examples:

```text
root
menubar
Treeview widgets
Figure objects
FigureCanvasTkAgg objects
PyVista plotter objects
Toplevel windows
thread objects
callback IDs
```

### 2.3 Store callbacks explicitly only when needed

Callbacks stored in `state` should be used to connect independent modules.

Accepted examples:

```python
state["create_checklist"] = create_checklist
state["apply_symavg_to_state"] = apply_symavg_to_state
state["populate_fitting_controls"] = populate_fitting_controls
state["load_structure_file"] = lambda path: _load_xyz_from_path(state, path)
```

Avoid storing callbacks when direct imports are cleaner.

---

## 3. Coordinate System Rules

Coordinate handling is central to PCS Analyzer. All new features must state which coordinate frame they use.

### 3.1 Coordinate layers

Use the following meanings consistently:

```text
original_atoms
    The original absolute coordinates loaded from the input file.

working_atoms
    The current absolute coordinates used as the active structure.
    Example: after conformer search is applied.

effective_atoms
    The analysis coordinates after symmetry averaging or pseudo atom generation.
    These are still absolute coordinates unless explicitly stated otherwise.

table_atoms
    Snapshot of the currently visible main table.
    This may include filtering, metal-centering, rotation, pseudo labels, and computed values.
```

### 3.2 Preferred naming in code

Current code may still use historical names. Future code should prefer:

```text
state["atom_data_original"]  -> original_atoms
state["atom_data_raw"]       -> working_atoms
state["atom_data_eff"]       -> effective_atoms
state["current_selected_ids"] -> visible or table-selected atom IDs
```

Avoid using `current_atoms` in new project-file sections because it is ambiguous.

### 3.3 Absolute coordinates vs displayed coordinates

Project reconstruction should use absolute coordinates:

```text
original_atoms
working_atoms
effective_atoms
metal_xyz
```

Displayed or exported table coordinates may be metal-centered and rotated. These should be saved as snapshots only:

```text
table_atoms
visible XYZ export
```

### 3.4 Metal center convention

Store the metal center as an absolute coordinate in the same frame as `working_atoms`.

Recommended project representation:

```text
[structure]
metal_ref_id = 1
metal_element = U
metal_xyz = 0.00000000 0.00000000 0.00000000
authoritative_atoms = working_atoms
analysis_atoms = effective_atoms
visible_snapshot = table_atoms
```

### 3.5 Rotation convention

Use degrees in UI and project files.

Recommended project representation:

```text
[rotation]
x = 0.0
y = 0.0
z = 0.0
euler_order = XYZ
```

If a new module uses a different Euler convention, it must document the convention explicitly.

---

## 4. `.pcsp` Project File Rules

The `.pcsp` format is a human-readable project file format for PCS Analyzer.

### 4.1 Format philosophy

The file is primarily written by PCS Analyzer.  
Users may manually edit numeric values, coordinates, IDs, and simple options.

Therefore:

```text
writer output should be clean and standardized
parser should be strict
error messages should include line numbers
manual editing should be limited to obvious numeric/table fields
```

### 4.2 Allowed syntax

Only support the following syntax:

```text
# comment
key = value

[section]
key = value

[table_section]
row row row row
```

Do not support:

```text
key: value
quoted strings
multiline values
nested indentation
JSON fragments
multiple separator styles
```

### 4.3 Value conventions

Use these conventions consistently:

```text
bool        true / false
none        none
auto        auto
list        comma-separated, no spaces preferred
vector3     space-separated floats
string      raw text
number      int or float
```

Examples:

```text
selected_elements = U,N,C,H
metal_xyz = 0.00000000 0.00000000 0.00000000
rhombic_enabled = false
padding = auto
clip_abs_ppm = none
```

### 4.4 Required root keys

Every `.pcsp` file should begin with:

```text
schema_version = 1.0
app_name = PCS Analyzer
app_version = 1.3.3
```

When changing file structure, increment the schema version.

Recommended versioning:

```text
1.0     first stable project format
1.1     backward-compatible section additions
2.0     breaking format changes
```

### 4.5 Recommended core sections

A normal project file should contain:

```text
[provenance]
[structure]
[original_atoms]
[working_atoms]
[effective_atoms]
[table_atoms]
[symmetry_averaging]
[visibility]
[pcs_model]
[rotation]
[delta_exp]
[delta_obs]
[delta_dia]
[fitting]
[fit_override]
[last_fit_result]
[conformer_search]
[viewer.pcs_field]
[viewer.pcs_field.levels]
[viewer.plot3d]
[viewer.projection]
[export]
```

Not every section must be required in the first implementation, but the writer should keep a stable order.

### 4.6 Atom table formats

Use fixed-column-style rows for readability.

#### `[original_atoms]`

```text
# id element              x              y              z
1     U        0.00000000     0.00000000     0.00000000
2     N        2.34120000     0.12000000    -0.43000000
```

#### `[working_atoms]`

```text
# id element              x              y              z
1     U        0.00000000     0.00000000     0.00000000
2     N        2.31850000     0.13500000    -0.40100000
```

#### `[effective_atoms]`

```text
# id element label              x              y              z source members
1     U       U1       0.00000000     0.00000000     0.00000000 current -
101   H       MeH@C12  3.12000000     1.24000000    -0.82000000 pseudo  31,32,33
```

#### `[table_atoms]`

```text
# id label              x              y              z            G_i       delta_pcs       delta_exp
1     U       0.00000000     0.00000000     0.00000000       0.0000       0.0000          none
2     N       2.31850000     0.13500000    -0.40100000  -1.234e-02      -3.4567     12.3400
```

### 4.7 Experimental NMR/PCS tables

Use simple two-column tables.

```text
[delta_exp]
# ref ppm
2     12.34000000
3     -4.56000000

[delta_obs]
# ref ppm

[delta_dia]
# ref ppm
```

Use empty sections when no values are available.

### 4.8 Backward compatibility

Loaders should support older files where reasonable.

Example:

```text
If [working_atoms] is absent but [current_atoms] exists,
treat [current_atoms] as [working_atoms].
```

Deprecated sections should be read but not written by the current writer.

### 4.9 Parser error messages

Parser errors should include:

```text
file path
line number
section name
expected format
offending line
```

Example:

```text
Invalid atom row in [working_atoms], line 42.
Expected: id element x y z
Got: 2 N 1.0 2.0
```

---

## 5. Project Save and Load Rules

### 5.1 Save rule

Project save should collect state without changing the current analysis.

Saving must not:

```text
trigger fitting
change selected atoms
change rotation sliders
modify delta values
open or close viewer windows
```

### 5.2 Load rule

Project load may rebuild the UI state.

Recommended load order:

```text
1. parse file
2. validate schema and required sections
3. restore working/original structure
4. restore metal center
5. restore UI variables for PCS model and rotation
6. restore symmetry averaging settings
7. rebuild effective atoms
8. restore delta values
9. rebuild checklist
10. update graph and main table
11. restore fitting state
12. restore viewer settings
13. refresh dependent views if open
```

### 5.3 Cache values

Do not save cache values unless they are useful as a human-readable snapshot.

Examples:

```text
pcs_by_id
    normally recalculated from structure and tensor

table_atoms
    saved as snapshot for readability and inspection
```

---

## 6. Drag and Drop Rules

Drag and drop should remain a shortcut to existing file-loading functions.

Supported file types:

```text
.pcsp
.xyz
.out
.log
```

Expected behavior:

```text
.pcsp            open project
.xyz/.out/.log   load structure file
other            show unsupported-file warning
```

DnD must not implement separate loading logic. It should call:

```text
state["open_project_file"](path)
state["load_structure_file"](path)
```

If `tkinterdnd2` is unavailable, the app must still start normally.

---

## 7. UI Coding Rules

### 7.1 Font management

Do not hard-code fonts directly in new UI code.

Prefer:

```python
fonts = state.get("fonts", {})
font=fonts.get("section", ("Segoe UI", 10, "bold"))
font=fonts.get("report", ("Consolas", 9))
```

For Matplotlib:

```python
fontsize=fonts.get("plot_label", 9)
labelsize=fonts.get("plot_tick", 8)
```

For PyVista:

```python
font_size=fonts.get("viewer_label_size", 10)
```

### 7.2 Error handling

Do not silently swallow important errors.

Allowed:

```python
try:
    optional_refresh()
except Exception:
    pass
```

Preferred for user-triggered actions:

```python
try:
    run_action()
except Exception as exc:
    messagebox.showerror("Action failed", str(exc))
```

For complex actions, include traceback in development builds or print it to console.

### 7.3 Long-running work

Do not block the Tkinter main thread for expensive calculations.

Use worker threads for:

```text
fitting
conformer search
large grid/PDE calculation
network update checks
```

UI updates must be scheduled through:

```python
root.after(0, callback)
```

### 7.4 Viewer windows

Viewer windows should store handles in `state` and avoid duplicates.

Recommended pattern:

```python
win = state.get("some_window")
if win is not None and win.winfo_exists():
    win.lift()
    win.focus_force()
    return
```

On close:

```python
state["some_window"] = None
```

---

## 8. Fitting and Scientific Calculation Rules

### 8.1 Scientific output changes

Any change that may alter calculated PCS, fitting, tensor parameters, or residuals must be documented in the changelog.

Examples:

```text
formula changes
unit conversion changes
coordinate frame changes
fit parameter bounds
new weighting scheme
new averaging behavior
```

### 8.2 Preserve old behavior through options

When changing scientific behavior, prefer:

```text
old behavior remains default
new behavior behind an explicit option
```

unless the old behavior was clearly a bug.

### 8.3 Units

Always document units in variable names, UI labels, and project files.

Examples:

```text
dchi_ax        E-32 m^3
delta_exp      ppm
coordinates    angstrom
rotation        degrees
```

---

## 9. Plugin and Module Rules

### 9.1 Modules should not directly mutate unrelated state

A module should only modify state keys it owns or keys passed explicitly through the plugin API.

### 9.2 Plugin menu behavior

Built-in module entries should remain stable.  
External plugin entries should be appended below the built-in separator.

### 9.3 Plugin failure

A plugin load failure must not prevent PCS Analyzer from starting.

---

## 10. Packaging Rules

### 10.1 Optional dependencies

Optional features must fail gracefully.

Examples:

```text
tkinterdnd2 missing       drag and drop disabled
pyvista missing           3D PCS viewer unavailable
openpyxl missing          Excel export unavailable
```

### 10.2 PyInstaller spec

When adding a dependency that ships data files, update the spec file.

Examples:

```text
tkinterdnd2
    include package data for tkdnd Tcl/Tk resources

matplotlib
    include mpl-data if needed

pyvista/vtk
    confirm binaries are collected
```

### 10.3 Do not assume IDE behavior equals EXE behavior

Whenever adding UI or packaging-sensitive features, test both:

```text
python main.py
PyInstaller-built EXE
```

---

## 11. Recommended File Organization

Suggested locations:

```text
logic/project_io.py
    .pcsp parser/writer and state apply/collect helpers

ui/drag_drop.py
    drag-and-drop registration and dropped-file dispatch

ui/style.py
    theme and font system

logic/coordinate_frame.py
    future shared coordinate-frame utilities

logic/project_schema.py
    future schema constants and validation helpers
```

Avoid putting large new systems directly into `ui/components.py` unless they are tightly tied to main-window construction.

---

## 12. Pre-Commit Checklist

Before committing a new feature, check:

```text
[ ] App starts from IDE
[ ] Main window opens without warnings
[ ] Load XYZ still works
[ ] Load .pcsp still works
[ ] Save .pcsp still works
[ ] Drag and drop still works if tkinterdnd2 is installed
[ ] App still starts if tkinterdnd2 is not installed
[ ] PCS plot updates
[ ] G_i vs PCS plot updates
[ ] Main table updates
[ ] Relevant viewer opens
[ ] No unwanted state is saved to .pcsp
[ ] No runtime handles are saved to .pcsp
[ ] New settings have defaults
[ ] New project-file fields have backward-compatible loading behavior
[ ] PyInstaller spec is updated if new dependencies were added
```

---