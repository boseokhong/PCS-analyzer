"""One bounded discovery/normalization/vendor-import path for all entry points."""
from dataclasses import dataclass
import os
from pathlib import Path
from .loader import load_nmr_dataset
from ..core import NMRDataset


@dataclass(frozen=True)
class BatchLoadResult:
    loaded: tuple[tuple[str, NMRDataset], ...]
    failures: tuple[tuple[str, str], ...]
    skipped: tuple[str, ...] = ()


def source_key(path):
    return os.path.normcase(str(Path(path).expanduser().resolve())) if path is not None else None


def experiment_root(path):
    path = Path(path).expanduser()
    return path.parent if path.is_file() and path.name.lower() == "fid" else path


def discover_experiments(root, *, max_depth=3):
    """Lexical depth-first traversal, bounded to three directory edges by default.

    Stop at raw/processed roots, including malformed acquisition markers. Never
    follow directory symlinks; an explicitly selected linked root is permitted.
    """
    root = Path(root).expanduser()
    if type(max_depth) is not int or not 0 <= max_depth <= 8:
        raise ValueError("Discovery depth must be an integer from 0 to 8")
    if not root.is_dir():
        raise ValueError("Choose an existing experiment parent folder.")
    found = []
    def visit(directory, depth):
        if ((directory / "fid").is_file() or
            ((directory / "1r").is_file() and (directory / "procs").is_file()) or
            directory.name.lower().endswith(".fid")):
            found.append(directory.resolve())
            return
        if depth == max_depth:
            return
        try:
            children = sorted(directory.iterdir(), key=lambda p: (p.name.casefold(), p.name))
        except OSError:
            found.append(directory)  # The vendor loader reports this path explicitly.
            return
        for child in children:
            try:
                if child.is_dir() and not child.is_symlink():
                    visit(child, depth + 1)
            except OSError:
                found.append(child)  # An inaccessible child never hides later siblings.
    visit(root, 0)
    return tuple(found)


def load_experiments(paths, *, loader=load_nmr_dataset, existing_sources=()):
    """Worker-safe partial import; input order then lexical discovery, no UI/state."""
    if isinstance(paths, (str, Path)):
        paths = (paths,)
    loaded, failures, skipped = [], [], []
    seen = {source_key(experiment_root(p)) for p in existing_sources if p is not None}
    for original in paths:
        try:
            root = experiment_root(original)
            candidates = discover_experiments(root) if root.is_dir() else (root,)
            if not candidates:
                candidates = (root,)  # Unsupported/empty parent is an explicit failure.
            for candidate in candidates:
                key = source_key(candidate)
                if key in seen:
                    skipped.append(f"{candidate}: duplicate/already loaded source")
                    continue
                seen.add(key)
                try:
                    dataset = loader(candidate if candidate.exists() else str(candidate))
                    if not isinstance(dataset, NMRDataset):
                        raise TypeError("loader did not return an NMRDataset")
                    loaded.append((str(candidate), dataset))
                except Exception as exc:
                    failures.append((str(candidate), f"{type(exc).__name__}: {exc}"))
        except (OSError, ValueError, TypeError) as exc:
            failures.append((str(original), f"{type(exc).__name__}: {exc}"))
    return BatchLoadResult(tuple(loaded), tuple(failures), tuple(skipped))
