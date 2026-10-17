# Installation

PCS Analyzer is distributed as a **portable Windows application** and can also be run from the Python source code. The prebuilt release is Windows-only; source operation on other platforms depends on the local Python, Tk, Qt, and VTK environment and is not verified by the portable-release workflow.

Use the version selector in the documentation header to match your application
release. **latest** refers to the latest published stable manual; **dev**, when
available, describes unreleased work. Archived versions retain their own pages
and images. The current source manual targets **1.4.0**.

## Windows portable version

The portable distribution includes its Python runtime and does not require a separate Python installation.

1. Download the latest Windows release from the [PCS Analyzer releases page](https://github.com/boseokhong/PCS-analyzer/releases).
2. Extract the complete archive to a writable folder.
3. Run the PCS Analyzer executable from the extracted folder.
4. Keep the distributed executable, `lib`, and other bundled folders together.

Do not run the executable from inside the compressed archive.

## Running from source

Python with Tkinter is required.

```bash
git clone https://github.com/boseokhong/PCS-analyzer.git
cd PCS-analyzer
python -m pip install -r requirements.txt
python main.py
```

The dependency roles are:

| Role | Packages | Requirement level |
|---|---|---|
| Core calculations and interface | `numpy`, `scipy`, `matplotlib`, `pandas`, and Python Tkinter | Required for the main application and scientific workflows. |
| Excel import/export | `openpyxl` | Required for `.xlsx` input and output; CSV workflows do not use it. |
| NMR vendor data and automatic phasing | `nmrglue` | Required for Bruker and Varian/Agilent data import, Bruker digital-filter correction, and NMR Workbench Auto Phase. |
| NMR baseline correction | `pybaselines` | Required for NMR Workbench Whittaker baseline methods, including AsLS and arPLS (Auto Baseline). |
| PyVista visualization | `pyvista`, `vtk` | Required by the PyVista-based 3D PCS and Workbench viewers. |
| Qt-backed structure viewer | `pyvistaqt`, `qtpy`, `PySide6` | Required when opening the interactive PyVista Qt structure viewer. |
| Chemical Map layout and cleanup | `rdkit` | Optional at runtime. Enables CoordGen and RDKit 2D layouts and chemical structure cleanup; Graph 2D remains available without it. |
| Drag and drop | `tkinterdnd2` | Enables drag-and-drop; the application falls back to ordinary Tk when it is unavailable. |
| GIF output | `imageio` | Used by PyVista GIF export. |
| Alternate theme engine | `ttkbootstrap` | Present in the full dependency set, but the current default UI uses standard ttk styling. |
| FFT acceleration | `pyfftw` | Optional at runtime. PCS Workbench falls back to `numpy.fft` when it is unavailable. |
| Module version and compatibility checks | `packaging` | Required for application/plugin version comparison and compatibility checks in the Module Manager. |

`requirements.txt` also installs the optional packages `pyfftw` and `rdkit`. Their runtime requirements and fallback behavior are listed above.

## Structure-file support

The main structure loader accepts:

| Format | Behavior |
|---|---|
| `.xyz` | Reads standard XYZ, headerless coordinate lists, and several common indexed/atomic-number variants. |
| `.out` | Reads the last Cartesian coordinate block from an ORCA output file. |
| `.log` | Uses the same ORCA coordinate-block parser as `.out`. |

ORCA coordinates in Bohr are converted to Ångström. See [Loading Structures](structures/loading.md) for the accepted layouts and parsing details.

## Updating PCS Analyzer

### Portable Windows release

Extract a new release into a separate folder instead of overwriting the current installation. Before removing the old folder, preserve:

```text
lib/app_settings.json
plugins/
```

Copy those items into the corresponding locations in the new release, start it, and verify settings and plugins before deleting the previous folder.

!!! info "Update checking is not automatic installation"

    **Help → Check for Updates...** compares the installed version with the latest GitHub release and can open the release page. PCS Analyzer does not download, replace, or restart the application automatically.

### Source checkout

Update the checkout using the Git workflow appropriate for that local repository, then refresh its environment if the requirements changed:

```bash
python -m pip install -r requirements.txt
```

## Next step

Once PCS Analyzer is running, continue with [Quick Start](quick-start.md).

[Quick Start](quick-start.md){ .md-button .md-button--primary }
