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


def check_public_payload(entries, project_root):
    """Fail the build when reviewed public inputs are missing or local files leak."""
    project_root = Path(project_root).resolve()
    _, public = split_public_payload(entries)
    destinations = [Path(entry[0]).as_posix() for entry in public]
    if len({path.casefold() for path in destinations}) != len(destinations):
        raise ValueError("Duplicate public release destination")
    expected_examples = {
        (Path(destination) / Path(source).name).as_posix()
        for source, destination in collect_public_examples(project_root)
    }
    expected_docs = {
        line.strip() for line in
        (project_root / "tools" / "public_docs_files.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for folder, expected in (("examples", expected_examples), ("docs", expected_docs)):
        actual = {path for path in destinations if path.startswith(folder + "/")}
        missing, unexpected = expected - actual, actual - expected
        if missing or unexpected:
            raise ValueError(
                folder + " release manifest mismatch; missing: " + ", ".join(sorted(missing))
                + "; unexpected: " + ", ".join(sorted(unexpected))
            )
    allowed_notices = {*REQUIRED_RELEASE_NOTICES, "LICENSE", "THIRD_PARTY_NOTICES.txt"}
    for path in destinations:
        parts = Path(path).parts
        if parts[0] == "licenses":
            if len(parts) == 2 and parts[1] in allowed_notices:
                continue
            if len(parts) >= 3 and parts[1] == "packages":
                continue
            if path == "licenses/python/LICENSE.txt":
                continue
            raise ValueError("Unreviewed release notice: " + path)


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
            if "__pycache__" in lower or basename.endswith((".pyc", ".pyo")):
                continue
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


def check_release_privacy(directory, project_root=None):
    """Inspect the actual release; report locations/categories, never secret values.

    Public author/citation details and third-party attribution are intentional.
    This gate complements explicit public-file manifests and example review; it
    cannot identify every possible personal name inside arbitrary binary data.
    """
    import socket
    import shlex
    import zipfile
    from types import CodeType
    from PyInstaller.archive.readers import CArchiveReader

    directory = Path(directory).resolve()
    findings = set()
    public_email = b"bshong66@gmail.com"
    forbidden_parts = {"dev_docs", ".git", ".aws", ".ssh", ".agents", ".codex", "__pycache__"}
    forbidden_names = {"app_settings.json", "plugins.json", ".env", "id_rsa", "id_ed25519",
                       "credentials.json", "credentials", "AGENTS.md"}
    profile_tokens = set()
    for path in (Path.home(), Path(project_root).resolve() if project_root else directory):
        parts = path.parts
        for index, part in enumerate(parts[:-1]):
            if part.casefold() == "users" and parts[index + 1].casefold() not in {"public", "default"}:
                profile_tokens.add("/".join(parts[:index + 2]).replace("\\", "/").replace("//", "/").casefold().encode())
    hostname = socket.gethostname().encode().lower()
    profile_tokens |= {token.decode().encode(encoding) for token in list(profile_tokens)
                       for encoding in ("utf-16-le", "utf-16-be")}
    secrets = (
        ("access_token", re.compile(rb"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|xox[baprs]-[A-Za-z0-9-]{20,}|sk-proj-[A-Za-z0-9_-]{40,})(?![A-Za-z0-9_])")),
        ("aws_access_key", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
        ("private_key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----\s+[A-Za-z0-9+/=\r\n]{64,}")),
    )
    home_path = re.compile(rb"(?i)(?:[a-z]:[\\/]+Users[\\/]+|/Users/|/home/)([^\\/\s<>\"']+)")
    emails = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    placeholders = {b"public", b"default", b"user", b"username", b"example", b"runner", b"buildbot"}

    def scan(data, location, first_party=False):
        normalized = data.lower().replace(b"\\\\", b"/").replace(b"\\", b"/")
        if any(token in normalized for token in profile_tokens):
            findings.add((location, "local_user_profile_path"))
        if len(hostname) >= 6 and hostname in normalized:
            findings.add((location, "local_computer_name"))
        for category, pattern in secrets:
            if pattern.search(data):
                findings.add((location, category))
        if first_party:
            for match in home_path.finditer(data):
                if match.group(1).lower() not in placeholders:
                    findings.add((location, "personal_home_path"))
            for match in emails.finditer(data):
                value = match.group(0).lower()
                if value != public_email and not value.endswith((b"@example.com", b"@example.org", b"@example.invalid")):
                    findings.add((location, "unapproved_contact_email"))

    files = sorted(path for path in directory.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(directory).as_posix()
        parts = path.relative_to(directory).parts
        if (any(part.casefold() in forbidden_parts for part in parts)
                or path.name in forbidden_names or path.suffix.lower() in {".pfx", ".p12"}
                or parts[0].casefold() == "plugins"):
            findings.add((relative, "private_state_or_development_file"))
        first_party = (parts[0] in {"docs", "examples", "img"}
                       or relative in PUBLIC_RELEASE_FILES
                       or relative.startswith(("lib/tools/", "lib/img/", "lib/nmr_workbench/")))
        with path.open("rb") as stream:
            overlap = b""
            while chunk := stream.read(1024 * 1024):
                scan(overlap + chunk, relative, first_party)
                overlap = chunk[-4096:]
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if not member.is_dir():
                        with archive.open(member) as stream:
                            overlap = b""
                            while chunk := stream.read(1024 * 1024):
                                scan(overlap + chunk, relative + "!" + member.filename)
                                overlap = chunk[-4096:]
        if path.name == "procpar" and parts[0] == "examples":
            lines = path.read_text(encoding="utf-8").splitlines()
            identity_fields = {"operator_", "operator", "sampleowner", "studyowner", "emailaddr", "username", "userid", "hostname", "owner"}
            for index, line in enumerate(lines[:-1]):
                header = line.split()
                if len(header) > 3 and header[0].lower() in identity_fields and header[1].isdigit():
                    values = shlex.split(lines[index + 1])[1:]
                    if any(value.lower() not in {"", "demo", "synthetic", "example", "anonymous", "anonymized", "none"} for value in values):
                        findings.add((relative + ":" + header[0], "non_anonymized_acquisition_identity"))
        if first_party and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            from PIL import Image
            with Image.open(path) as picture:
                if picture.getexif().get(34853):
                    findings.add((relative, "image_gps_metadata"))
                scan(str(picture.info).encode("utf-8"), relative + ":image_metadata", True)

    module_count = 0
    executable = directory / "PCS Analyzer.exe"
    if executable.exists():
        archive = CArchiveReader(str(executable))
        pyz = archive.open_embedded_archive(next(key for key, entry in archive.toc.items() if entry[-1] == "z"))
        def scan_code(code, location, first_party):
            if isinstance(code, CodeType):
                scan(code.co_filename.encode("utf-8", errors="surrogatepass"), location, first_party)
                for value in code.co_consts:
                    scan_code(value, location, first_party)
            elif isinstance(code, str):
                scan(code.encode("utf-8", errors="surrogatepass"), location, first_party)
            elif isinstance(code, bytes):
                scan(code, location, first_party)
            elif isinstance(code, (tuple, list, set, frozenset)):
                for value in code:
                    scan_code(value, location, first_party)
            elif isinstance(code, dict):
                for key, value in code.items():
                    scan_code(key, location, first_party)
                    scan_code(value, location, first_party)
        for name in pyz.toc:
            first_party = name.split(".")[0] in {"logic", "ui", "plugin_system", "nmr_workbench", "tools", "app_version", "main"}
            scan_code(pyz.extract(name), "PYZ:" + name, first_party)
            module_count += 1
        import marshal
        for name, entry in archive.toc.items():
            if entry[-1] == "s":
                scan_code(marshal.loads(archive.extract(name)), "EXE:" + name, name == "main")
    report = {"passed": not findings, "files_scanned": len(files), "modules_scanned": module_count,
              "findings": [{"location": location, "category": category} for location, category in sorted(findings)],
              "public_author_citation_and_license_attribution_preserved": True}
    if findings:
        error = RuntimeError("Release privacy gate failed: " + "; ".join(location + " [" + category + "]" for location, category in sorted(findings)))
        error.report = report
        raise error
    return report
