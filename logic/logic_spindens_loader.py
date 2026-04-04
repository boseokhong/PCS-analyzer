# logic/logic_spindens_loader.py
"""
Loader for ORCA simple-grid spin density files (.3d format).

File format
-----------
Line 1 : title string
Line 2 : nx ny nz
Line 3 : ox oy oz
Line 4 : dx dy dz
Lines 5+: scalar values

This loader returns the raw grid together with origin/spacing metadata.
The grid ordering is arranged to match the expected [X, Y, Z] convention.
"""

from __future__ import annotations

import numpy as np


def load_spindens_3d(file_path: str) -> dict:
    """
    Load a simple-grid spin density file.

    Returns
    -------
    dict with keys:
        title
        shape
        origin
        spacing
        rho
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    if len(lines) < 5:
        raise ValueError(
            f"Spin density file {file_path!r} is too short "
            f"(expected at least 5 non-blank lines, got {len(lines)})."
        )

    title = lines[0]

    try:
        nx, ny, nz = map(int, lines[1].split())
    except Exception as exc:
        raise ValueError(f"Cannot parse grid dimensions from: {lines[1]!r}") from exc

    if any(n <= 0 for n in (nx, ny, nz)):
        raise ValueError(f"Grid dimensions must be positive, got ({nx}, {ny}, {nz}).")

    try:
        ox, oy, oz = map(float, lines[2].split())
    except Exception as exc:
        raise ValueError(f"Cannot parse origin from: {lines[2]!r}") from exc

    try:
        dx, dy, dz = map(float, lines[3].split())
    except Exception as exc:
        raise ValueError(f"Cannot parse spacing from: {lines[3]!r}") from exc

    if any(s <= 0 for s in (dx, dy, dz)):
        raise ValueError(f"Grid spacing must be positive, got ({dx}, {dy}, {dz}).")

    values: list[float] = []
    for ln in lines[4:]:
        for tok in ln.split():
            try:
                values.append(float(tok))
            except ValueError as exc:
                raise ValueError(f"Non-numeric token in density data: {tok!r}") from exc

    expected = nx * ny * nz
    if len(values) != expected:
        raise ValueError(
            f"Density value count mismatch: declared {nx}×{ny}×{nz} = {expected}, "
            f"but found {len(values)} values."
        )

    data = np.asarray(values, dtype=np.float64)

    # Exact translation of:
    # density = reshape(abs(A.data), npts(2), npts(3), npts(1));
    # density = permute(density, [3 2 1]);
    rho = np.abs(data).reshape((ny, nz, nx), order="F")
    rho = np.transpose(rho, (2, 1, 0))

    return {
        "title": title,
        "shape": tuple(int(v) for v in rho.shape),
        "origin": np.array([ox, oy, oz], dtype=np.float64),
        "spacing": (float(dx), float(dy), float(dz)),
        "rho": rho,
    }