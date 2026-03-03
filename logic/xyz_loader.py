# logic/xyz_loader.py

from __future__ import annotations

import os
import re
from typing import List, Tuple

Atom = Tuple[str, float, float, float]

BOHR_TO_ANGSTROM = 0.529177210903  # CODATA 2018; sufficiently precise for geometry

def parse_xyz(file_path: str) -> List[Atom]:
    """
    Robust XYZ parser.

    Supports:
      1) Standard XYZ:
         line1 = atom count (int)
         line2 = comment
         line3+ = "El  x  y  z"
      2) Headerless XYZ:
         all lines are "El  x  y  z"

    Returns:
      [(element, x, y, z), ...]
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        return []

    start = 0
    try:
        # If first non-empty line is integer -> assume standard XYZ header
        int(lines[0])
        start = 2
    except Exception:
        start = 0

    out: List[Atom] = []
    for ln in lines[start:]:
        parts = ln.split()
        if len(parts) < 4:
            continue
        el = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except Exception:
            continue
        out.append((el, x, y, z))
    return out

def _extract_orca_cartesian_blocks(text: str) -> List[Tuple[str, List[Atom]]]:
    """
    Extract ORCA cartesian coordinate blocks.

    Returns:
      list of (unit, atoms) where unit is "ANGSTROEM" or "BOHR".
      Order is the same as appearance in the file.
    """
    lines = text.splitlines()
    blocks: List[Tuple[str, List[Atom]]] = []

    # Match both:
    #   CARTESIAN COORDINATES (ANGSTROEM)
    #   CARTESIAN COORDINATES (BOHR)
    header_re = re.compile(r"^\s*CARTESIAN\s+COORDINATES\s*\(\s*(ANGSTROEM|BOHR)\s*\)\s*$", re.IGNORECASE)

    i = 0
    n = len(lines)
    while i < n:
        m = header_re.match(lines[i])
        if not m:
            i += 1
            continue

        unit = m.group(1).upper()

        # ORCA usually has dashed lines around/after header; skip a few non-data lines safely
        i += 1
        # Skip up to ~6 lines of separators/blank lines
        for _ in range(6):
            if i >= n:
                break
            s = lines[i].strip()
            # Stop skipping if this looks like data (>=4 tokens and token2/3/4 numeric-ish)
            toks = s.split()
            if len(toks) >= 4:
                break
            i += 1

        atoms: List[Atom] = []
        while i < n:
            s = lines[i].strip()
            if not s:
                break

            parts = s.split()
            if len(parts) < 4:
                break

            el = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except Exception:
                break

            atoms.append((el, x, y, z))
            i += 1

        if atoms:
            blocks.append((unit, atoms))

        i += 1

    return blocks

def parse_orca_out(file_path: str, *, which: str = "last") -> List[Atom]:
    """
    Parse ORCA .out/.log and return a coordinate block.

    Args:
      which:
        - "last": last coordinate block in file (usually final geometry)
        - "first": first block
        - "all": NOT returned here (use parse_orca_out_all if you need)

    Unit handling:
      - If block unit is BOHR -> converted to Angstrom.
      - If block unit is ANGSTROEM -> used as-is.

    Returns:
      [(element, x, y, z), ...] in Angstrom
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    blocks = _extract_orca_cartesian_blocks(text)
    if not blocks:
        return []

    if which == "first":
        unit, atoms = blocks[0]
    else:
        # default: last
        unit, atoms = blocks[-1]

    if unit == "BOHR":
        return [(el, x * BOHR_TO_ANGSTROM, y * BOHR_TO_ANGSTROM, z * BOHR_TO_ANGSTROM) for el, x, y, z in atoms]
    return atoms

def parse_orca_out_all(file_path: str) -> List[List[Atom]]:
    """
    Return ALL coordinate blocks from an ORCA output, each converted to Angstrom.

    Useful if later you want to animate optimization steps or debug geometry changes.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    blocks = _extract_orca_cartesian_blocks(text)
    out: List[List[Atom]] = []
    for unit, atoms in blocks:
        if unit == "BOHR":
            out.append([(el, x * BOHR_TO_ANGSTROM, y * BOHR_TO_ANGSTROM, z * BOHR_TO_ANGSTROM) for el, x, y, z in atoms])
        else:
            out.append(atoms)
    return out

def load_structure(file_path: str) -> List[Atom]:
    """
    Unified structure loader.

    Supports:
      - .xyz
      - .out / .log (ORCA)

    Returns:
      atom list in Angstrom
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".xyz":
        return parse_xyz(file_path)
    if ext in (".out", ".log"):
        return parse_orca_out(file_path, which="last")
    raise ValueError(f"Unsupported file type: {ext}")