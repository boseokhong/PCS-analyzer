"""Build-time filtering and notice collection; never imported by the application."""

from importlib import metadata
from importlib import util as import_util
from pathlib import Path
import os
import re
import shutil
import sys


EXCLUDED_QT_MODULES = (
    "qtpy.QtDataVisualization", "qtpy.QtDatavisualization",
    "PySide6.QtDataVisualization", "PySide6.QtVirtualKeyboard",
)
EXCLUDED_QT_FILES = frozenset({
    "qt6datavisualization.dll", "qtdatavisualization.pyd",
    "qt6virtualkeyboard.dll", "qtvirtualkeyboardplugin.dll",
})
PUBLIC_RELEASE_FILES = frozenset({"README.md", "LICENSE", "TRADEMARKS.md", "CITATION.cff"})
PUBLIC_RELEASE_DIRS = frozenset({"docs", "examples", "licenses"})
REQUIRED_RELEASE_NOTICES = ("GPL-3.0.txt", "LGPL-3.0.txt", "SOURCE_AND_REPLACEMENT.txt")


def collect_public_examples(project_root):
    """Collect only reviewed ready-to-open inputs, never local example tools."""
    project_root = Path(project_root).resolve()
    manifest = project_root / "tools" / "public_examples_files.txt"
    entries = []
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        relative = line.strip()
        if not relative or relative.startswith("#"):
            continue
        parts = relative.split("/")
        if (parts[0] != "examples" or len(parts) < 3
                or any(part in ("", ".", "..") for part in parts)
                or "\\" in relative or ":" in relative):
            raise ValueError("Invalid public example path: " + relative)
        path = Path(*parts)
        if path.suffix not in {".pcsp", ".xyz", ".csv"} and path.name not in {"fid", "procpar"}:
            raise ValueError("Not a desktop example input: " + relative)
        if relative.casefold() in seen:
            raise ValueError("Duplicate public example path: " + relative)
        seen.add(relative.casefold())
        source = (project_root / path).resolve()
        if not source.is_relative_to(project_root / "examples"):
            raise ValueError("Public example escapes examples directory")
        if not source.is_file():
            raise FileNotFoundError("Missing public example: " + relative)
        entries.append((str(source), str(path.parent)))
    if not entries:
        raise ValueError("Public example manifest is empty")
    return entries


def split_public_payload(entries):
    """Separate user-facing DATA from the runtime payload collected under lib/."""
    internal, public = [], []
    for entry in entries:
        relative = Path(entry[0])
        if not relative.parts or relative.anchor or ".." in relative.parts:
            raise ValueError("Unsafe release destination: " + entry[0])
        top = relative.parts[0]
        is_public = top in PUBLIC_RELEASE_DIRS or relative.as_posix() in PUBLIC_RELEASE_FILES
        if is_public and entry[2] != "DATA":
            raise ValueError("Public release payload must contain only data files")
        (public if is_public else internal).append(entry)
    return internal, public


def copy_public_payload(entries, directory):
    """Write the reviewed payload beside the exe after PyInstaller COLLECT.

    PyInstaller's contents_directory applies to every DATA entry. Keep it for
    libraries and copy this explicit subset separately, without traversal or
    overwriting runtime/user files.
    """
    directory = Path(directory).resolve()
    _, public = split_public_payload(entries)
    if len(public) != len(entries):
        raise ValueError("Non-public file passed to release-root collector")
    targets = set()
    planned = []
    for relative, source, _ in public:
        target = (directory / relative).resolve()
        if not target.is_relative_to(directory):
            raise ValueError("Release destination escapes output directory")
        if target in targets or target.exists():
            raise FileExistsError("Duplicate release destination: " + relative)
        if not Path(source).is_file():
            raise FileNotFoundError("Missing release payload: " + relative)
        targets.add(target)
        planned.append((source, target))
    for source, target in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in PUBLIC_RELEASE_FILES | PUBLIC_RELEASE_DIRS:
        if not (directory / name).exists():
            raise RuntimeError("Missing public release item: " + name)
        if (directory / "lib" / name).exists():
            raise RuntimeError("Public release item remains under lib/: " + name)


def require_portable_dependencies():
    """The full portable release includes NMR processing and RDKit layouts.

    RDKit remains optional during source execution; this check applies only to
    the complete release recipe, where its enhancement is deliberately bundled.
    """
    missing = [name for name in ("nmrglue", "pybaselines", "rdkit")
               if import_util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Missing portable-build dependencies: " + ", ".join(missing)
            + ". Install them in the build interpreter: " + sys.executable
            + " -m pip install " + " ".join(missing)
        )


def configure_windows_build_path():
    """Prevent unrelated desktop/build tools on PATH from supplying runtime DLLs."""
    if os.name != "nt":
        return
    windows = Path(os.environ["SystemRoot"])
    roots = [Path(sys.prefix), Path(sys.base_prefix)]
    paths = [str(root / suffix) for root in roots for suffix in ("", "DLLs", "Scripts")]
    paths += [str(windows / "System32"), str(windows)]
    os.environ["PATH"] = os.pathsep.join(dict.fromkeys(paths))


def check_native_sources(entries, project_root):
    """Fail if dependency analysis pulls native files from unrelated applications."""
    if os.name != "nt":
        return
    roots = [Path(project_root).resolve(), Path(sys.prefix).resolve(),
             Path(sys.base_prefix).resolve(), Path(os.environ["SystemRoot"]).resolve()]
    unexpected = []
    for entry in entries:
        source = Path(entry[1]).resolve()
        if not any(source.is_relative_to(root) for root in roots):
            unexpected.append(entry[0])
    if unexpected:
        raise RuntimeError("Unapproved native dependency sources: " + ", ".join(unexpected))


def filter_qt_binaries(entries):
    """Remove only the reviewed unused modules/plugin, keeping other Qt input plugins."""
    return [entry for entry in entries
            if not any(Path(str(value).replace("\\", "/")).name.casefold()
                       in EXCLUDED_QT_FILES for value in entry[:2])]


def collect_notices(entries, project_root, output_dir):
    """Collect notices from distributions owning actual Analysis input files.

    Retain original notice bytes and relative paths. Never publish build-machine
    paths or assume that Python package metadata covers every native dependency.
    """
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_RELEASE_NOTICES:
        if not (project_root / "licenses" / name).is_file():
            raise FileNotFoundError("Required release notice is missing: " + name)
    # Analysis and wheel RECORD paths are already absolute/installation-relative.
    # Normalize them lexically instead of opening every dependency file just to
    # compare names (large scientific wheels contain tens of thousands of files).
    def source_key(path):
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    sources = {source_key(entry[1]) for entry in entries if len(entry) > 1}
    rows = []
    datas = []
    warnings = []
    for dist in sorted(metadata.distributions(), key=lambda d: d.metadata.get("Name", "").lower()):
        files = dist.files or ()
        if not any(source_key(dist.locate_file(f)) in sources for f in files):
            continue
        name = dist.metadata.get("Name", "unknown")
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
        notice_files = []
        for relative in files:
            parts = Path(str(relative).replace("\\", "/")).parts
            if any(p in ("..", ".") for p in parts) or Path(relative).is_absolute():
                continue
            lower = [p.casefold() for p in parts]
            basename = lower[-1]
            if not ("licenses" in lower or any(basename.startswith(p) for p in
                    ("license", "licence", "copying", "copyright", "notice"))):
                continue
            source = Path(dist.locate_file(relative))
            if not source.is_file():
                continue
            destination = Path("licenses") / "packages" / safe_name / Path(*parts)
            datas.append((str(source), str(destination.parent)))
            notice_files.append(destination.as_posix())
        declared = dist.metadata.get("License-Expression") or dist.metadata.get("License") or "Not declared"
        if len(declared) > 240 or "\n" in declared:
            # Preserve long license metadata without filling the summary with it.
            target = output_dir / (safe_name + "-license-metadata.txt")
            target.write_text(declared, encoding="utf-8")
            datas.append((str(target), "licenses/packages/" + safe_name))
            declared = "See packages/" + safe_name + "/" + target.name
        rows.append((name, dist.version, declared, notice_files))
        if not notice_files:
            warnings.append(name + ": no license file recorded by the installed distribution")

    for source, destination in (
        (project_root / "LICENSE", "licenses"),
        (Path(sys.base_prefix) / "LICENSE.txt", "licenses/python"),
    ):
        if source.is_file():
            datas.append((str(source), destination))
        else:
            warnings.append("Missing required notice: " + source.name)
    # Never collect local dependency-policy or license-provenance records.
    for name in REQUIRED_RELEASE_NOTICES:
        source = project_root / "licenses" / name
        datas.append((str(source), "licenses"))

    lines = ["PCS Analyzer — Third-party component notices", "",
             "PCS Analyzer's own code is licensed under BSD 3-Clause (see LICENSE).",
             "Third-party components retain their respective licenses.", "",
             "This inventory is generated from the packages owning files selected by",
             "PyInstaller Analysis. Versions describe the build inputs, not every",
             "independently versioned library inside a wheel. Copied notices retain",
             "their original text. This inventory is not a license-clearance statement.", "",
             "Qt/PySide6, native dependencies and source/replacement information:",
             "see SOURCE_AND_REPLACEMENT.txt and the license texts in this directory.", ""]
    for name, version, declared, files in rows:
        lines += [name + " " + version, "Declared license: " + declared]
        lines += ["  " + path for path in files] or ["  No installed license file found; review required."]
        lines.append("")
    lines += ["Review items", "------------",
              "Native/vendor libraries, fonts, Tcl/Tk and compiler runtimes may have",
              "additional notices not represented by Python distribution metadata.",
              "FFTW and the FFmpeg executable have licensing separate from their",
              "Python wrappers. Retaining license texts alone does not resolve source",
              "availability, binary replacement or redistribution requirements.", ""]
    lines += warnings
    notice = output_dir / "THIRD_PARTY_NOTICES.txt"
    notice.write_text("\n".join(lines) + "\n", encoding="utf-8")
    datas.append((str(notice), "licenses"))
    for warning in warnings:
        print("[NOTICE REVIEW] " + warning)
    print(f"[OK] collected notices for {len(rows)} selected distributions")
    return datas


def check_qt_output(directory):
    forbidden = [p.relative_to(directory).as_posix() for p in Path(directory).rglob("*")
                 if p.is_file() and p.name.casefold() in EXCLUDED_QT_FILES]
    if forbidden:
        raise RuntimeError("Excluded Qt components remain in release: " + ", ".join(forbidden))
