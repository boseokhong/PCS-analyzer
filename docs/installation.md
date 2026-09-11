# Installation

PCS Analyzer can be used either as a **portable Windows application** or directly from the Python source code.

As of **v1.3.4**, the prebuilt portable distribution is provided for Windows only. Support for macOS and Linux may be added in future releases if needed.

---

## Windows portable version

The portable distribution does not require a separate Python installation.

1. Download the latest Windows release of PCS Analyzer.
2. Extract the downloaded archive to a folder of your choice.
3. Run the PCS Analyzer executable.
4. Keep the distributed files and folders together unless otherwise specified.

The latest releases are available from the [PCS Analyzer GitHub repository](https://github.com/boseokhong/PCS-analyzer/releases).

---

## Running from source

PCS Analyzer can also be run directly from the Python source code, for example on systems where a prebuilt portable distribution is not available.

### Requirements

PCS Analyzer is written in Python and requires several scientific and graphical Python packages.

For **PCS Analyzer v1.3.4**:

!!! note "Python package requirements"

    **Required packages:** `numpy`, `scipy`, `matplotlib`, `pandas`, `openpyxl`

    **Optional / additional packages:** `ttkbootstrap`, `pyvista`, `pyvistaqt`, `vtk`, `qtpy`, `PySide6`, `pyfftw`, `tkinterdnd2`, `imageio`

    - `pyvista` is used for 3D PCS-field and molecular visualization.
    - `ttkbootstrap` provides enhanced GUI styling.
    - `pyvistaqt`, `qtpy`, and `PySide6` are required for the interactive Qt-based 3D structure viewer.
    - `pyfftw` is optional. If it is not installed, PCS Analyzer falls back to `numpy.fft`.
    - `tkinterdnd2` provides drag-and-drop support.
    - `imageio` is required for GIF export using PyVista's `Plotter.open_gif()`.

It is recommended to install the dependencies in a dedicated Python environment.

### Clone the repository

```bash
git clone https://github.com/boseokhong/PCS-analyzer.git
cd PCS-analyzer
```

Alternatively, download the source code archive from GitHub and extract it locally.

### Install dependencies

If a requirements file is provided with the release:

```bash
python -m pip install -r requirements.txt
```

### Start PCS Analyzer

From the project directory:

```bash
python main.py
```

The main PCS Analyzer interface should then open.

---

## Structure-file support

PCS Analyzer can directly load molecular structures from:

| Format | Description |
|---|---|
| `.xyz` | Standard or headerless XYZ coordinate files |
| `.out` | ORCA output files |
| `.log` | ORCA output/log files |

For ORCA output files, PCS Analyzer uses the **last Cartesian coordinate block** found in the file.

Coordinates given by ORCA in Bohr are automatically converted to Ångström.

---

## Updating PCS Analyzer

For the portable Windows distribution, it is recommended to extract each new PCS Analyzer release into a **separate folder** rather than overwriting the existing installation.

Before removing the previous version, back up any user-specific configuration and plugin data that should be retained.

### Recommended files to preserve

PCS Analyzer stores application settings in:

```text
lib/app_settings.json
```

To preserve your current application configuration, copy this file from the previous installation to the corresponding location in the new release.

It is also recommended to back up the entire:

```text
plugins/
```

folder.

This preserves installed external plugins together with their registry information and enabled/disabled state.

Existing settings and plugin data are expected to remain compatible between releases unless otherwise noted. If a release introduces incompatible changes, the corresponding release notes will provide the necessary migration or reset instructions.

!!! caution "Keep the previous installation temporarily"

    Do not delete the previous PCS Analyzer folder immediately.

    Start the new release first, verify that the application settings and plugins have been restored correctly, and only then remove the older installation if desired.

A typical portable update workflow is therefore:

```text
Download new release
        ↓
Extract to a new folder
        ↓
Copy lib/app_settings.json
        ↓
Copy the plugins/ folder
        ↓
Start PCS Analyzer
        ↓
Verify settings and plugins
        ↓
Remove old version if desired
```

!!! info "Automatic updates"

    Automatic updates from within PCS Analyzer are planned for a future release.
	
For source installations, update the local repository using Git:

```bash
git pull
```

Dependency changes between versions may require updating the Python environment:

```bash
python -m pip install -r requirements.txt
```

---

## Next step

Once PCS Analyzer is installed and running, continue with the [Quick Start](quick-start.md) guide.

[Quick Start](quick-start.md){ .md-button .md-button--primary }
