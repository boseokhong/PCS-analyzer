"""PCS Analyzer workspace persistence (.pcsw).

Workspaces reference canonical .pcsp files; they never embed scientific data.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


WORKSPACE_EXTENSION = ".pcsw"
WORKSPACE_VERSION = 1


class WorkspaceError(ValueError):
    """Raised when a workspace cannot be parsed or represented safely."""


class UnsavedWorkspaceDocumentError(WorkspaceError):
    """Raised when a dirty untitled document would otherwise be lost."""


@dataclass(frozen=True)
class WorkspaceLoadResult:
    path: Path
    loaded_paths: tuple[Path, ...]
    missing_paths: tuple[Path, ...]
    active_document_id: str


def normalize_workspace_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix.lower() == WORKSPACE_EXTENSION else path.with_suffix(WORKSPACE_EXTENSION)


def _path_key(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False)).casefold()


def _portable_reference(document_path: Path, workspace_path: Path) -> str:
    absolute = document_path.expanduser().resolve(strict=False)
    try:
        relative = Path(os.path.relpath(absolute, workspace_path.parent.resolve(strict=False)))
    except ValueError:  # different Windows drives
        return str(absolute)
    # Keep nearby projects relocatable without producing unwieldy references.
    if sum(part == ".." for part in relative.parts) <= 2:
        return relative.as_posix()
    return str(absolute)


def save_workspace(path: str | Path, manager: Any) -> Path:
    """Write ordered saved project references and the active reference."""
    target = normalize_workspace_path(path)
    references: list[str] = []
    active_reference: str | None = None
    for document in manager.documents.values():
        if document.file_path is None:
            if document.dirty:
                raise UnsavedWorkspaceDocumentError(
                    f"Save {document.title} as a PCS project before saving the workspace"
                )
            continue
        reference = _portable_reference(Path(document.file_path), target)
        references.append(reference)
        if document.document_id == manager.active_document_id:
            active_reference = reference
    payload = {
        "format": "pcs_analyzer_workspace",
        "version": WORKSPACE_VERSION,
        "documents": references,
        "active_document": active_reference,
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def _parse_workspace(path: Path) -> tuple[list[Path], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkspaceError(f"Could not read workspace: {exc}") from exc
    if not isinstance(payload, dict):
        raise WorkspaceError("Workspace root must be an object")
    if payload.get("format") != "pcs_analyzer_workspace":
        raise WorkspaceError("Not a PCS Analyzer workspace")
    if payload.get("version") != WORKSPACE_VERSION:
        raise WorkspaceError(f"Unsupported workspace version: {payload.get('version')}")
    raw_documents = payload.get("documents")
    if not isinstance(raw_documents, list) or not all(isinstance(item, str) for item in raw_documents):
        raise WorkspaceError("Workspace documents must be a list of paths")
    resolved: list[Path] = []
    seen: set[str] = set()
    for raw in raw_documents:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        candidate = candidate.resolve(strict=False)
        key = _path_key(candidate)
        if key not in seen:
            seen.add(key)
            resolved.append(candidate)
    active = payload.get("active_document")
    if active is not None and not isinstance(active, str):
        raise WorkspaceError("active_document must be a path or null")
    return resolved, active


def load_workspace(path: str | Path, state: dict[str, Any], manager: Any) -> WorkspaceLoadResult:
    """Load all available projects detached, then atomically replace the session."""
    source = normalize_workspace_path(path).expanduser().resolve(strict=False)
    resolved, raw_active = _parse_workspace(source)
    from logic.project_io import load_project_document

    loaded = []
    loaded_paths: list[Path] = []
    missing: list[Path] = []
    for project_path in resolved:
        if not project_path.is_file():
            missing.append(project_path)
            continue
        # Parsing every available project before manager mutation makes a
        # malformed project an all-or-nothing workspace failure.
        document = load_project_document(
            project_path, state, defaults=manager.default_snapshot()
        )
        loaded.append(document)
        loaded_paths.append(project_path)

    active_key = None
    if raw_active:
        active_path = Path(raw_active).expanduser()
        if not active_path.is_absolute():
            active_path = source.parent / active_path
        active_key = _path_key(active_path)
    active_id = next(
        (
            document.document_id
            for document in loaded
            if document.file_path is not None and _path_key(document.file_path) == active_key
        ),
        None,
    )
    active = manager.replace_documents(loaded, active_document_id=active_id)
    return WorkspaceLoadResult(
        path=source,
        loaded_paths=tuple(loaded_paths),
        missing_paths=tuple(missing),
        active_document_id=active.document_id,
    )


__all__ = [
    "WORKSPACE_EXTENSION", "WORKSPACE_VERSION", "WorkspaceError",
    "WorkspaceLoadResult", "UnsavedWorkspaceDocumentError",
    "load_workspace", "normalize_workspace_path", "save_workspace",
]
