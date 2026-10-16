# py -3.12 -m pip install --upgrade pip
# py -3.12 -m pip install pyinstaller packaging numpy matplotlib pandas openpyxl ttkbootstrap scipy pyvista pyvistaqt vtk pyfftw imageio imageio-ffmpeg pillow tkinterdnd2 qtpy PySide6 nmrglue pybaselines rdkit
# py -3.12 build_version_info.py
# py -3.12 -m PyInstaller --clean --noconfirm --log-level=INFO PCS_Analyzer_Portable_v1.4.0.spec

# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
    collect_all,
)


APP_NAME = "PCS Analyzer"
BASE_DIR = os.path.abspath(".")
ICON_PATH = os.path.join(BASE_DIR, "img", "logoheader.ico")

# ============================================================
# Helper functions
# ============================================================

def collect_package(package_name, datas, binaries, hiddenimports):
    """
    Collect datas, binaries, and hidden imports for large or dynamic packages.
    If the package is not installed, the build continues with a warning.
    """
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package_name)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
        print(f"[OK] collected package: {package_name}")
    except Exception as e:
        print(f"[WARN] could not collect {package_name}: {e}")

    return datas, binaries, hiddenimports


def collect_project_package(package_name, hiddenimports):
    """
    Collect internal project packages.
    """
    try:
        hiddenimports += collect_submodules(
            package_name,
            filter=lambda name: name != "tools.docs_capture"
            and not name.startswith("tools.docs_capture.")
            and name != "tools.patches" and not name.startswith("tools.patches.")
            and name not in (
                "tools.check_examples", "tools.demo_func_group_collapse",
                "tools.demo_nmr_spectrum", "tools.generate_history_icons",
                "tools.generate_nmr_control_icons", "tools.molecule_viewer_probe",
                "tools.molecule_viewer_sidebar_review",
                "tools.plot_pcs_origin",
            ),
        )
        print(f"[OK] collected project package: {package_name}")
    except Exception as e:
        print(f"[WARN] could not collect project package {package_name}: {e}")

    return hiddenimports


def collect_resource_folder(folder, datas, excluded_dirs=(), excluded_files=()):
    """Copy resource files with their relative layout, excluding development files."""
    path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(path):
        return datas

    for root, dirs, files in os.walk(path):
        dirs[:] = [
            name for name in dirs
            if name not in excluded_dirs and name not in ("__pycache__", ".pytest_cache")
        ]
        destination = os.path.relpath(root, BASE_DIR)
        for name in files:
            if name not in excluded_files and not name.endswith((".pyc", ".pyo")):
                datas.append((os.path.join(root, name), destination))

    return datas


def collect_public_docs(datas):
    """Bundle only reviewed public files; never recursively copy local docs."""
    docs_root = os.path.realpath(os.path.join(BASE_DIR, "docs"))
    manifest = os.path.join(BASE_DIR, "tools", "public_docs_files.txt")
    with open(manifest, encoding="utf-8") as stream:
        entries = [line.strip() for line in stream if line.strip() and not line.startswith("#")]
    for relative in entries:
        parts = relative.split("/")
        if parts[0] != "docs" or any(part in ("", ".", "..") for part in parts) or "\\" in relative or ":" in relative:
            raise ValueError("Invalid public documentation manifest entry")
        source = os.path.realpath(os.path.join(BASE_DIR, *parts))
        if os.path.commonpath((docs_root, source)) != docs_root:
            raise ValueError("Public documentation escapes the documentation directory")
        if not os.path.isfile(source):
            raise FileNotFoundError("Public documentation manifest references a missing file: " + relative)
        datas.append((source, os.path.join(*parts[:-1])))
    return datas


# ============================================================
# Base containers
# ============================================================

datas = []
binaries = []
hiddenimports = []


# ============================================================
# Project metadata / docs
# ============================================================
for fname in [
    "README.md",
    "LICENSE",
    "CITATION.cff",
]:
    path = os.path.join(BASE_DIR, fname)
    if os.path.exists(path):
        datas.append((path, "."))

# Add new public documentation files to the reviewed manifest when publishing.
datas = collect_public_docs(datas)


# ============================================================
# Optional resource folders
# ============================================================

# Do NOT include "plugins" here.
# plugins/installed and plugins.json should be created at runtime
# next to the exe by plugin_system/plugin_registry.py.

for folder in [
    "img",
]:
    path = os.path.join(BASE_DIR, folder)
    if os.path.isdir(path):
        datas.append((path, folder))

# Keep runtime tools, but omit documentation capture utilities and caches.
datas = collect_resource_folder("tools", datas, excluded_dirs=("docs_capture", "patches"),
                                excluded_files=("check_examples.py", "demo_func_group_collapse.py",
                                                "demo_nmr_spectrum.py", "generate_history_icons.py",
                                                "generate_nmr_control_icons.py", "molecule_viewer_probe.py",
                                                "molecule_viewer_sidebar_review.py", "plot_pcs_origin.py"))

# NMR referencing presets and the supplied tutorial datasets.
for folder in [
    os.path.join("nmr_workbench", "resources"),
    os.path.join("examples", "documentation_demo"),
    os.path.join("examples", "nmr_workflow_demo"),
    os.path.join("examples", "nmr_vt_demo"),
    os.path.join("examples", "multi_conformer_demo"),
    os.path.join("examples", "joint_pcs_rdc_demo"),
    os.path.join("examples", "pre_demo"),
    os.path.join("examples", "multi_lanthanide_demo"),
    os.path.join("examples", "standard_fitting_demo"),
    os.path.join("examples", "conformer_search_demo"),
    os.path.join("examples", "nmr_pcs_fitting_demo"),
    os.path.join("examples", "averaging_visualization_demo"),
]:
    datas = collect_resource_folder(folder, datas, excluded_files=("log", "text", "check_coverage.py"))

datas.append((os.path.join(BASE_DIR, "examples", "README.md"), "examples"))


# ============================================================
# Internal project modules
# ============================================================

# main.py imports ui.components, and components.py imports many logic/ui modules.
# plugin_system is needed for Module Manager / external plugin loading.
for pkg in [
    "logic",
    "ui",
    "tools",
    "plugin_system",
    "nmr_workbench",
]:
    hiddenimports = collect_project_package(pkg, hiddenimports)

hiddenimports += [
    "app_version",
]


# ============================================================
# Standard library note
# ============================================================

# No need to add these to hiddenimports:
# importlib.util, json, sys, traceback, shutil, tempfile, zipfile,
# os, re, csv, math, threading, itertools, pathlib, typing, dataclasses, io


# ============================================================
# Explicit hidden imports
# ============================================================

hiddenimports += [
    # Tkinter
    "tkinter",
    "tkinter.ttk",
    "tkinter.filedialog",
    "tkinter.simpledialog",
    "tkinter.messagebox",
    "tkinter.scrolledtext",

    # Drag and drop support
    "tkinterdnd2",
    "tkinterdnd2.TkinterDnD",

    # Matplotlib / Tk backend
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.figure",
    "matplotlib.gridspec",
    "matplotlib.lines",
    "matplotlib.colors",
    "matplotlib.patches",
    "matplotlib.backends.backend_tkagg",
    "matplotlib.backends.backend_agg",
    "mpl_toolkits",
    "mpl_toolkits.mplot3d",

    # Scientific stack
    "numpy",
    "pandas",
    "openpyxl",
    "scipy",
    "scipy.optimize",
    "scipy.optimize._lsq",
    "scipy.optimize._differentialevolution",
    "scipy.spatial",
    "scipy.spatial.distance",
    "scipy.spatial.transform",

    # Chemical Map layout and cleanup
    "rdkit",
    "rdkit.Chem",
    "rdkit.Chem.rdDepictor",
    "rdkit.Chem.rdCoordGen",

    # Excel helpers
    "openpyxl.cell._writer",
    "openpyxl.styles",
    "openpyxl.worksheet._reader",

    # Image handling
    "PIL",
    "PIL.Image",
    "imageio",
    "imageio.v2",

    # PyVista / Qt / VTK
    # Keep these explicit so PyInstaller sees the Qt-based viewer, but do not
    # collect_all() PySide6/qtpy/pyvistaqt below.
    "pyvista",
    "pyvistaqt",
    "pyvistaqt.plotting",
    "qtpy",
    "qtpy.QtCore",
    "qtpy.QtGui",
    "qtpy.QtWidgets",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "vtk",
    "vtkmodules",
    "vtkmodules.all",
    "vtkmodules.util",
    "vtkmodules.util.numpy_support",
    "vtkmodules.vtkCommonCore",
    "vtkmodules.vtkCommonDataModel",
    "vtkmodules.vtkCommonExecutionModel",
    "vtkmodules.vtkFiltersCore",
    "vtkmodules.vtkFiltersGeneral",
    "vtkmodules.vtkFiltersGeometry",
    "vtkmodules.vtkFiltersSources",
    "vtkmodules.vtkIOCore",
    "vtkmodules.vtkIOGeometry",
    "vtkmodules.vtkIOImage",
    "vtkmodules.vtkImagingCore",
    "vtkmodules.vtkRenderingCore",
    "vtkmodules.vtkRenderingOpenGL2",
    "vtkmodules.vtkRenderingFreeType",
    "vtkmodules.vtkRenderingVolumeOpenGL2",
    "vtkmodules.vtkInteractionStyle",
]


# ============================================================
# Moderate packages
# ============================================================

for pkg in [
    "numpy",
    "matplotlib",
    "pandas",
    "openpyxl",
    "ttkbootstrap",
    "tkinterdnd2",
    "scipy",
    "nmrglue",
    "pybaselines",
]:
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception as e:
        print(f"[WARN] could not collect submodules for {pkg}: {e}")

    try:
        datas += collect_data_files(pkg)
    except Exception as e:
        print(f"[WARN] could not collect data files for {pkg}: {e}")

# RDKit's PyInstaller hooks handle imported extension modules; include package
# data explicitly for Chemical Map layout and cleanup.
try:
    datas += collect_data_files("rdkit")
except Exception as e:
    print(f"[WARN] could not collect data files for rdkit: {e}")


# ============================================================
# Heavy / dynamic packages
# ============================================================

# Keep Qt packages out of collect_all().
# PyInstaller's Qt/PySide6 hooks will collect the runtime actually required
# by pyvistaqt/qtpy, while avoiding the very large full PySide6 payload.
for pkg in [
    "pyvista",
    "vtkmodules",
    "vtk",
    "pyfftw",
    "imageio",
]:
    datas, binaries, hiddenimports = collect_package(
        pkg, datas, binaries, hiddenimports
    )

# Optional but useful for imageio video/GIF export.
datas, binaries, hiddenimports = collect_package(
    "imageio_ffmpeg", datas, binaries, hiddenimports
)

# tkinterdnd2 ships the Tcl/Tk tkdnd files needed at runtime.
datas, binaries, hiddenimports = collect_package(
    "tkinterdnd2", datas, binaries, hiddenimports
)


# ============================================================
# Extra dynamic libraries
# ============================================================

for pkg in [
    "vtkmodules",
    "vtk",
    "pyfftw",
    "rdkit",
]:
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception as e:
        print(f"[WARN] could not collect dynamic libs for {pkg}: {e}")


# ============================================================
# De-duplicate
# ============================================================

datas = list(dict.fromkeys(datas))
binaries = list(dict.fromkeys(binaries))
hiddenimports = list(dict.fromkeys(hiddenimports))


# ============================================================
# Excludes
# ============================================================

excludes = [
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "tests",
    "tkinter.test",
    "tools.docs_capture",
    "tools.check_examples",
    "tools.demo_func_group_collapse",
    "tools.demo_nmr_spectrum",
    "tools.generate_history_icons",
    "tools.generate_nmr_control_icons",
    "tools.molecule_viewer_probe",
    "tools.molecule_viewer_sidebar_review",
    "tools.plot_pcs_origin",
    "tools.patches",

    # Use PySide6 as the single Qt binding for pyvistaqt.
    "PyQt5",
    "PyQt6",
    "PySide2",

    # Not used by PCS Analyzer core; prevent optional dependency chains
    # from pulling large ML/CV stacks into the portable build.
    "torch",
    "torchvision",
    "transformers",
    "sklearn",
    "cv2",
]


# ============================================================
# Build
# ============================================================

a = Analysis(
    ["main.py"],
    pathex=[BASE_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=ICON_PATH if os.path.exists(ICON_PATH) else None,
    version="version_info.txt" if os.path.exists("version_info.txt") else None,
    contents_directory="lib",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
