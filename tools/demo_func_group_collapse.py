# tools/demo_func_group_collapse.py

from __future__ import annotations

import sys
from pathlib import Path

from logic.xyz_loader import load_structure
from logic.func_group_collapse import (
    build_bond_graph,
    collapse_methyl_groups,
    collapse_cf3_groups,
    find_tert_butyl_centers,
)


def _pick_file_via_dialog() -> str | None:
    """Open a simple file picker dialog (Tkinter). Returns selected path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Select structure file",
        filetypes=[
            ("Structure files", "*.xyz *.out *.log *.txt"),
            ("XYZ", "*.xyz"),
            ("ORCA output", "*.out *.log *.txt"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()
    return path or None


def main() -> int:
    if len(sys.argv) >= 2:
        path_str = sys.argv[1]
    else:
        path_str = _pick_file_via_dialog()
        if not path_str:
            print("No file selected. Exiting.")
            return 0

    path = Path(path_str)
    if not path.exists():
        print(f"File not found: {path}")
        return 2

    atom_data = load_structure(str(path))
    print(f"Loaded {len(atom_data)} atoms from {path.name}")

    # Build bond graph once for tBu center reporting
    neigh = build_bond_graph(atom_data, scale=1.10)

    # 1) Methyl collapse (mask mode recommended for integration)
    me_atoms, me_records, me_masked = collapse_methyl_groups(
        atom_data,
        mode="mask",
        pseudo_element="H",
        require_carbon_substituent_count=1,
    )

    print("\n[Methyl]")
    print(f"Detected methyl groups: {len(me_records)}")
    print(f"Masked original H atoms: {len(me_masked)}")
    print(f"Returned atom list length: {len(me_atoms)} (mask mode keeps originals + appends pseudos)")

    if me_records:
        print("Collapse records:")
        for i, r in enumerate(me_records, 1):
            print(
                f"[{i}] {r.label} | "
                f"C(original idx)={r.center_index_original} | "
                f"H(original idx)={r.member_indices_original} | "
                f"pseudo_index={r.pseudo_index}"
            )
    else:
        print("No methyl groups detected.")

    # 2) CF3 collapse
    cf_atoms, cf_records, cf_masked = collapse_cf3_groups(
        atom_data,
        mode="mask",
        pseudo_element="F",
        require_carbon_substituent_count=1,
    )

    print("\n[CF3]")
    print(f"Detected CF3 groups: {len(cf_records)}")
    print(f"Masked original F atoms: {len(cf_masked)}")
    print(f"Returned atom list length: {len(cf_atoms)} (mask mode keeps originals + appends pseudos)")

    if cf_records:
        print("Collapse records:")
        for i, r in enumerate(cf_records, 1):
            print(
                f"[{i}] {r.label} | "
                f"C(original idx)={r.center_index_original} | "
                f"F(original idx)={r.member_indices_original} | "
                f"pseudo_index={r.pseudo_index}"
            )
    else:
        print("No CF3 groups detected.")

    # 3) tert-butyl center reporting (based on methyl carbon indices)
    methyl_carbons = {r.center_index_original for r in me_records}
    tbu_centers = find_tert_butyl_centers(
        atom_data,
        neigh,
        methyl_center_indices=methyl_carbons,
        require_center_carbon_degree=4,  # set None if you want permissive detection
    )

    print("\n[tBu centers]")
    print(f"Detected tBu centers: {len(tbu_centers)}")
    for i, c0 in enumerate(tbu_centers, 1):
        print(f"[{i}] tBu center carbon original idx = {c0} (label: C{c0 + 1})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())