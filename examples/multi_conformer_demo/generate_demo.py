"""Deterministic synthetic MC5 fixtures; no tests/UI dependencies.

The coordinates are labelled geometric probes, not a chemically optimized molecule.
Run from any working directory; outputs default to this example directory.
"""
from dataclasses import replace
from pathlib import Path
import csv
import json
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from logic.conformer_fitting import ConformerSnapshot, MultiConformerSnapshot, build_basis
from logic.conformer_workflow import default_options, snapshot_payload
from logic.pcs_fit_snapshot import FitGeometrySnapshot, FitObservation, FitTensorSnapshot
from logic.project_io import build_pcsp_text
from logic.pcs_structured_observations import OBSERVATIONS_KEY


def demo_inputs(scenario="informative", *, noise_seed=731, noise=True):
    if scenario not in ("informative", "ambiguous"):
        raise ValueError("unknown demo scenario")
    rng = np.random.default_rng(42)
    refs = tuple(range(1, 26))
    probes = rng.normal(size=(24, 3)) + (2., -1., 4.)
    rows = []
    for index in range(3):
        coords = probes + rng.normal(scale=.6*index, size=probes.shape)
        if scenario == "ambiguous" and index == 1:
            coords = np.asarray(rows[0].geometry.raw_coordinates)[1:]
        coords = np.vstack(((0., 0., 0.), coords))
        geometry = FitGeometrySnapshot(refs, coords, refs, coords, (0., 0., 0.), (), False, ())
        rows.append(ConformerSnapshot(f"C{index+1}", f"conformer_{index+1:02d}", geometry,
                    ("Yb",)+("H",)*24, "Synthetic explicit row-order Ref mapping", "Confirmed synthetic common frame",
                    initial_population=(.4, .4, .2)[index], population_fixed=index == 2))
    observations = tuple(FitObservation(ref, 0., observable_id=f"probe:{ref}", assignment=f"H{ref}",
                                      uncertainty_ppm=.03) for ref in refs[1:])
    snapshot = MultiConformerSnapshot(tuple(rows), observations, tensor=FitTensorSnapshot(True, True, -5., 1.))
    truth = {"populations": [.25, .55, .2], "dchi_ax": 12., "dchi_rh": 3., "sigma_ppm": .03,
             "noise_seed": noise_seed, "scenario": scenario, "geometry_seed": 42,
             "description": "Synthetic PCS probes; not experimental data or a chemically validated molecule"}
    exact = build_basis(snapshot).predict(truth["populations"], truth["dchi_ax"], truth["dchi_rh"])
    measured = exact + (np.random.default_rng(noise_seed).normal(0., truth["sigma_ppm"], len(exact)) if noise else 0.)
    snapshot = replace(snapshot, observations=tuple(replace(o, pcs=float(y)) for o, y in zip(observations, measured)))
    options = dict(default_options(), target="both", fit_ax=True, fit_rh=True, dchi_ax=-5., dchi_rh=1.,
                   weighting="experimental_uncertainty", starts=4, seed=11)
    snapshot = replace(snapshot, weighting_mode=options["weighting"])
    raw = [(element, *coord) for element, coord in zip(rows[0].elements, rows[0].geometry.raw_coordinates)]
    # Same explicit structured records accepted by the application collector.
    records = [{"id": o.observable_id, "assignment": o.assignment, "pcs_ppm": o.pcs,
                "uncertainty_ppm": o.uncertainty_ppm, "target": {"site_member_ref_ids": [[o.ref_id]],
                "scope": "Single atom", "matching_status": "unique", "averaging_hint": "none"}} for o in snapshot.observations]
    state = {"atom_data_raw": raw, "atom_data": raw, "atom_ids_raw": list(refs), "atom_ids": list(refs),
             "x0": 0., "y0": 0., "z0": 0., "metal_ref_id": 1, "tensor": -5., "rh_dchi_rh": 1.,
             "delta_exp_values": {o.ref_id: o.pcs for o in snapshot.observations}, OBSERVATIONS_KEY: records,
             "advfit_conformers": list(snapshot.conformers), "advfit_mc_options": options}
    return snapshot, options, truth, state


def generate(output, *, overwrite=False):
    from tools.demo_support import prepare_output
    names = [f'{s}/{n}' for s in ('informative', 'ambiguous') for n in
             ('inputs.json', 'demo.pcsp', 'conformer_01.xyz', 'conformer_02.xyz', 'conformer_03.xyz', 'pcs.csv')]
    output = prepare_output(output, names, overwrite=overwrite)
    for scenario in ("informative", "ambiguous"):
        folder = output/scenario
        folder.mkdir(parents=True, exist_ok=True)
        snapshot, options, truth, state = demo_inputs(scenario)
        (folder/"inputs.json").write_text(json.dumps({"snapshot": snapshot_payload(snapshot), "options": options,
                                            "truth": truth}, indent=2, allow_nan=False), encoding="utf-8")
        project_text = "\n".join(line.rstrip() for line in build_pcsp_text(state).splitlines())+"\n"
        (folder/"demo.pcsp").write_text(project_text, encoding="utf-8")
        for conformer in snapshot.conformers:
            lines = [str(len(conformer.elements)), "Synthetic probes; metal Ref 1 at origin; confirmed common frame"]
            lines += [f"{el} {x:.12g} {y:.12g} {z:.12g}" for el, (x, y, z) in
                      zip(conformer.elements, conformer.geometry.raw_coordinates)]
            (folder/f"{conformer.name}.xyz").write_text("\n".join(lines)+"\n", encoding="utf-8")
        with (folder/"pcs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("ref_id", "pcs_exp", "uncertainty_ppm"))
            writer.writerows((o.ref_id, o.pcs, o.uncertainty_ppm) for o in snapshot.observations)


if __name__ == "__main__":
    from tools.demo_support import generation_args
    args = generation_args(Path(__file__).resolve().parent, __doc__)
    generate(args.output, overwrite=args.overwrite)
