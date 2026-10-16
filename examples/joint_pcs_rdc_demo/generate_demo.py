"""Generate a fixed-frame synthetic example using independent Cartesian equations.

Run from the repository root: python examples/joint_pcs_rdc_demo/generate_demo.py
These labelled probes are not a chemically optimized molecular structure.
"""
from pathlib import Path
import csv
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def make_demo():
    rng = np.random.default_rng(20261013)
    origins = rng.normal(size=(12, 3))
    origins *= np.linspace(6, 15, 12)[:, None] / np.linalg.norm(origins, axis=1)[:, None]
    bonds = rng.normal(size=(12, 3))
    bonds *= 1.02 / np.linalg.norm(bonds, axis=1)[:, None]
    coords = np.stack([origins, origins + bonds], axis=1).reshape(-1, 3)
    ids = list(range(1, len(coords) + 1))
    labels = [f"{element}{i + 1}" for i in range(12) for element in ("N", "H")]
    ax, rh, field, temperature = -18.0, 5.0, 14.1, 298.0
    # Traceless susceptibility tensor in molecular SI units.
    tensor = np.diag([-ax / 3 + rh / 2, -ax / 3 - rh / 2, 2 * ax / 3]) * 1e-32
    metres = coords * 1e-10
    pcs = np.einsum("ni,ij,nj->n", metres, tensor, metres) / (
        4 * np.pi * np.linalg.norm(metres, axis=1) ** 5) * 1e6
    bond_m = bonds * 1e-10
    # Retain PCS Analyzer's existing negative dipolar prefactor.
    rdc = -3 * 1.054571817e-34 * 2.67522187e8 * -2.71262e7 * field ** 2 / (
        8 * np.pi ** 2 * 15 * 1.380649e-23 * temperature) * np.einsum(
            "ni,ij,nj->n", bond_m, tensor, bond_m) / np.linalg.norm(bond_m, axis=1) ** 5
    rows = [{"atom1": labels[2*i], "atom2": labels[2*i+1], "rdc_exp": float(value),
             "rdc_err": 0.5, "nuc1": "15N", "nuc2": "1H"} for i, value in enumerate(rdc)]
    return coords, ids, labels, dict(zip(ids, map(float, pcs))), rows, (ax, rh, field, temperature)


def main():
    from logic.advanced_physics import fit_joint_pcs_rdc
    coords, ids, labels, pcs, rows, truth = make_demo()
    from tools.demo_support import generation_args, prepare_output
    args = generation_args(Path(__file__).resolve().parent, __doc__)
    destination = prepare_output(args.output, ('structure.xyz', 'pcs.csv', 'rdc.csv'), overwrite=args.overwrite)
    xyz = [str(len(coords)), "Synthetic probes; metal at origin; fixed tensor frame"]
    xyz.extend(f"{label[0]} {x:.12g} {y:.12g} {z:.12g}" for label, (x, y, z) in zip(labels, coords))
    (destination / "structure.xyz").write_text("\n".join(xyz) + "\n", encoding="utf-8")
    for filename, records in (
        ("pcs.csv", [{"ref_id": rid, "pcs_exp": value, "pcs_err": 0.02} for rid, value in pcs.items()]),
        ("rdc.csv", rows),
    ):
        with (destination / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    result = fit_joint_pcs_rdc(coords, labels, pcs, ids, rows, 1.0,
                               pcs_errors={rid: 0.02 for rid in ids})
    np.testing.assert_allclose([result["dchi_ax"], result["dchi_rh"]], truth[:2], rtol=1e-10)
    print(f"Recovered axial={result['dchi_ax']:.6f}, rhombic={result['dchi_rh']:.6f}; rank={result['rank']}")
    print(f"PCS RMSD={result['rmsd_pcs']:.3g} ppm; RDC RMSD={result['rmsd_rdc']:.3g} Hz")


if __name__ == "__main__":
    main()
