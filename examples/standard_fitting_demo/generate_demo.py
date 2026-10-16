"""Generate known-tensor synthetic probes for standard fits and rhombicity diagnostics."""
from pathlib import Path
import csv
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
SCENARIOS = ("mode_a", "mode_b", "mode_c", "axial", "rhombic")


def xyz_rotation(angles):
    """Independent active extrinsic XYZ matrix, in degrees."""
    x, y, z = np.deg2rad(angles)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def cartesian_pcs(coords, axial, rhombic):
    """Independent molecular PCS equation: coordinates Å, tensor E-32 m³, output ppm."""
    coords = np.asarray(coords, float)
    x, y, z = coords.T
    return (axial * (2*z*z-x*x-y*y) + 1.5*rhombic*(x*x-y*y)) * 1e4 / (
        12*np.pi*np.linalg.norm(coords, axis=1)**5)


def probe_coordinates():
    rng = np.random.default_rng(20261014)
    probes = rng.normal(size=(32, 3))
    probes *= np.linspace(6., 14., len(probes))[:, None] / np.linalg.norm(probes, axis=1)[:, None]
    return probes


def make_state(scenario):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown standard fitting scenario")
    probes = probe_coordinates()
    axial, rhombic = 12., (0. if scenario in ("mode_a", "axial") else 3.)
    angles = (12., -8., 18.) if scenario in ("mode_b", "mode_c") else (0., 0., 0.)
    metal = (.2, -.15, .1) if scenario == "mode_c" else (0., 0., 0.)
    if scenario == "mode_a":
        # First donor lies on z. The fitter's theta=20 rotation is about x.
        # Alpha spins about that rotated donor: R_x(theta) R_z(alpha).
        rotation = xyz_rotation((20., 0., 0.)) @ xyz_rotation((0., 0., 25.))
    else:
        rotation = xyz_rotation(angles)
    values = cartesian_pcs((probes-np.asarray(metal)) @ rotation.T, axial, rhombic)
    raw = [("Nd", 0., 0., 0.), ("N", 0., 0., 2.4)]
    raw += [("H", *map(float, p)) for p in probes]
    refs = list(range(1, len(raw)+1))
    records = [{"id": f"probe:{rid}", "assignment": f"H{rid-2}", "pcs_ppm": float(value),
                "uncertainty_ppm": .02, "source": "synthetic_standard_demo",
                "target": {"scope": "Single atom", "site_member_ref_ids": [[rid]],
                           "matching_status": "unique", "averaging_hint": "none"}}
               for rid, value in zip(refs[2:], values)]
    mode = "theta_alpha_multi" if scenario == "mode_a" else "full_tensor" if scenario == "mode_c" else "euler_global"
    state = dict(atom_data_raw=raw, atom_data=raw, atom_ids_raw=refs, atom_ids=refs,
                 x0=0., y0=0., z0=0., metal_ref_id=1, tensor=12., rh_dchi_rh=rhombic,
                 app_settings={"default_dchi_ax":12.},
                 pcs_prediction_model="full_tensor" if rhombic else "axial", rh_calc_enabled=bool(rhombic),
                 delta_exp_values={r: float(v) for r,v in zip(refs[2:],values)},
                 experimental_pcs_observations=records, fit_mode_var=mode, axis_mode_var="first",
                 fit_dchi_var=True, fit_dchi_rh_var=bool(rhombic), fit_global_search_var=False,
                 fit_weighting_var="experimental_uncertainty", fit_use_current_tensor_var=False)
    truth = dict(scenario=scenario, axial=axial, rhombic=rhombic, metal_x=metal[0], metal_y=metal[1], metal_z=metal[2],
                 xyz_x=angles[0], xyz_y=angles[1], xyz_z=angles[2], theta=20. if scenario == "mode_a" else "",
                 alpha=25. if scenario == "mode_a" else "", donor_ref=2, observations=32)
    return state, truth


def fit_state(state, scenario):
    from logic.fitting import fit_theta_alpha_multi, fit_euler_global, fit_full_tensor
    from tools.demo_support import detached_standard_state
    state = detached_standard_state(state)
    args = dict(fit_delta_chi=True, fit_delta_chi_rh=scenario not in ("mode_a", "axial"),
                use_global_search=False, weighting_mode="experimental_uncertainty")
    if scenario == "mode_a":
        return fit_theta_alpha_multi(state, donor_ids=(2,), axis_mode="first", fit_visible_as_group=False, **args)
    if scenario == "mode_c":
        return fit_full_tensor(state, **args)
    return fit_euler_global(state, fit_visible_as_group=False, **args)


def generate(destination, *, overwrite=False):
    from tools.demo_support import prepare_output, demo_project_text
    names = [f"{s}/{n}" for s in SCENARIOS for n in ("demo.pcsp", "structure.xyz", "pcs.csv")]
    destination = prepare_output(destination, [*names, "truth.csv"], overwrite=overwrite)
    truths = []
    for scenario in SCENARIOS:
        state, truth = make_state(scenario)
        folder = destination / scenario
        folder.mkdir(exist_ok=True)
        (folder / "demo.pcsp").write_text(demo_project_text(state), encoding="utf-8")
        xyz = [str(len(state["atom_data_raw"])), "Synthetic known-tensor probes, not a chemically optimized molecule"]
        xyz += [f"{e} {x:.12g} {y:.12g} {z:.12g}" for e,x,y,z in state["atom_data_raw"]]
        (folder / "structure.xyz").write_text("\n".join(xyz)+"\n", encoding="utf-8")
        with (folder / "pcs.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(("ref_id", "pcs_exp", "uncertainty_ppm"))
            writer.writerows((r["target"]["site_member_ref_ids"][0][0],r["pcs_ppm"],.02) for r in state["experimental_pcs_observations"])
        truths.append(truth)
    with (destination / "truth.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(truths[0])); writer.writeheader(); writer.writerows(truths)


if __name__ == "__main__":
    from tools.demo_support import generation_args
    args = generation_args(Path(__file__).resolve().parent, __doc__)
    generate(args.output, overwrite=args.overwrite)
