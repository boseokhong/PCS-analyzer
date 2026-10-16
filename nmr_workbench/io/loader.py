"""Vendor detection and vendor-neutral NMR dataset loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from nmr_workbench.core import NMRDataset

from .bruker import Bruker1DReader
from .varian import Varian1DReader

NMRVendor = Literal["bruker", "varian"]


class VendorDetectionError(RuntimeError):
    """Raised when a directory is ambiguous, malformed, or unsupported."""


def resolve_nmr_dataset_path(path: str | Path) -> Path:
    """Return the canonical dataset root for a directory or raw ``fid`` file."""

    selected_path = Path(path).expanduser()
    if selected_path.is_dir():
        return selected_path.resolve()
    if selected_path.is_file():
        if selected_path.name.lower() != "fid":
            raise VendorDetectionError(
                f"Unsupported NMR acquisition file: {selected_path.name}; "
                "select a file named 'fid' or its dataset folder"
            )
        return selected_path.parent.resolve()
    raise VendorDetectionError(f"NMR dataset path not found: {selected_path}")


def detect_vendor(path: str | Path) -> NMRVendor:
    """Identify supported NMR data from structural marker files."""

    dataset_path = resolve_nmr_dataset_path(path)
    has_fid = (dataset_path / "fid").is_file()
    bruker_raw = has_fid and (dataset_path / "acqus").is_file()
    bruker_processed = (dataset_path / "1r").is_file() and (dataset_path / "procs").is_file()
    varian_raw = has_fid and (dataset_path / "procpar").is_file()
    candidates = {
        name
        for name, present in (
            ("bruker", bruker_raw or bruker_processed),
            ("varian", varian_raw),
        )
        if present
    }
    if len(candidates) > 1:
        raise VendorDetectionError(
            f"Ambiguous NMR dataset in {dataset_path}; both Bruker and Varian markers exist"
        )
    if candidates:
        return candidates.pop()  # type: ignore[return-value]
    if has_fid:
        raise VendorDetectionError(
            f"Malformed or unsupported raw NMR dataset in {dataset_path}; "
            "expected Bruker 'acqus' or Varian 'procpar'"
        )
    raise VendorDetectionError(f"Unsupported NMR dataset structure in {dataset_path}")


def load_nmr_dataset(path: str | Path, *, source: str = "auto") -> NMRDataset:
    """Load a supported 1D dataset from its folder or raw ``fid`` file."""

    dataset_path = resolve_nmr_dataset_path(path)
    vendor = detect_vendor(dataset_path)
    if vendor == "bruker":
        return Bruker1DReader().read(dataset_path, source=source)
    return Varian1DReader().read(dataset_path, source=source)
