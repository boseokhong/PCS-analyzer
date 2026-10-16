"""One-bond synthetic search exercise; coordinates are teaching geometry, not an optimized complex."""
from pathlib import Path
import csv
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.standard_fitting_demo.generate_demo import cartesian_pcs

ELEMENTS = ["Nd", "N", "C", "C", "C", "H", "H", "H", "N"]
COORDS = np.array([[0,0,0], [0,0,2.55], [1.4,0,3.0], [2.8,0,3.4], [3.4,1.3,3.9],
                   [3.4,1.4,4.98], [4.42,1.55,3.55], [2.75,2.1,3.55], [0,2.55,0]], float)
BOND = (2, 3)  # zero-based indices; visible Ref 3–4
FIXED = {0, 8}  # visible Ref 1 and 9; do not pin the moving fragment
OBS = [5, 6, 7]
AXIAL, RHOMBIC = 12., 3.


def truth_coordinates():
    result = COORDS.copy()
    origin = result[BOND[0]]
    axis = result[BOND[1]] - origin
    axis /= np.linalg.norm(axis)
    p = result[3:8]-origin
    angle = np.deg2rad(45.)
    result[3:8] = origin + p*np.cos(angle) + np.cross(axis,p)*np.sin(angle) + (
        p @ axis)[:,None]*axis*(1-np.cos(angle))
    return result


def make_state():
    truth = truth_coordinates()
    values = cartesian_pcs(truth[OBS], AXIAL, RHOMBIC)
    raw = [(e,*map(float,p)) for e,p in zip(ELEMENTS, COORDS)]
    return dict(atom_data_raw=raw, atom_data=raw, atom_ids_raw=list(range(1,10)), atom_ids=list(range(1,10)),
                x0=0.,y0=0.,z0=0.,metal_ref_id=1,tensor=AXIAL,rh_dchi_rh=RHOMBIC,
                app_settings={"default_dchi_ax":AXIAL},
                pcs_prediction_model="full_tensor",rh_calc_enabled=True,
                delta_exp_values={i+1:float(y) for i,y in zip(OBS,values)})


def run_search(state=None):
    from ui.conformer_search import Molecule, ConformerOptimiser, classify_bonds_by_fixed
    coords = COORDS if state is None else np.array([a[1:4] for a in state["atom_data_raw"]])
    mol = Molecule(ELEMENTS, coords)
    assert BOND in mol.bonds, "Teaching search bond must exist in the detected graph"
    free, locked = classify_bonds_by_fixed(mol, [BOND], FIXED)
    assert free == [BOND] and not locked
    values = np.array([state["delta_exp_values"][i+1] for i in OBS]) if state else cartesian_pcs(truth_coordinates()[OBS], AXIAL, RHOMBIC)
    owner = ConformerOptimiser(mol, free, OBS, values, np.zeros(3), AXIAL, RHOMBIC, (0.,0.,0.),
                               metal_idx=0, fixed_atoms=FIXED, constraint_mode="position")
    result = owner.run(use_global=True, de_maxiter=80, de_popsize=8, de_workers=1,
                       n_candidates=3, selected_bonds=free, locked_bonds=locked)
    assert result["rmsd"] < 1e-4, result["rmsd"]
    assert result["fixed_displacement"] < 1e-8
    assert result["max_angular_dev_deg"] < 1e-5
    return result


def generate(destination, *, overwrite=False):
    from tools.demo_support import prepare_output, demo_project_text
    destination = prepare_output(destination, ("demo.pcsp", "start.xyz", "truth.xyz", "pcs.csv", "conditions.csv"), overwrite=overwrite)
    state = make_state()
    (destination/"demo.pcsp").write_text(demo_project_text(state),encoding="utf-8")
    for filename,coords in (("start.xyz",COORDS),("truth.xyz",truth_coordinates())):
        lines = ["9","Synthetic one-bond exercise; zero orientation; not chemically optimized"]
        lines += [f"{e} {x:.12g} {y:.12g} {z:.12g}" for e,(x,y,z) in zip(ELEMENTS,coords)]
        (destination/filename).write_text("\n".join(lines)+"\n",encoding="utf-8")
    with (destination/"pcs.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(("Ref","delta_exp"));writer.writerows(state["delta_exp_values"].items())
    with (destination/"conditions.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(("parameter","value"))
        writer.writerows((("dchi_ax",AXIAL),("dchi_rh",RHOMBIC),("fixed_refs","1,9"),
                          ("rotatable_refs","3,4"),("rotation_from_start_deg",45),("constraint_mode","position"),
                          ("orientation_degrees","0,0,0"),("synthetic",True)))


if __name__ == "__main__":
    from tools.demo_support import generation_args
    args = generation_args(Path(__file__).resolve().parent,__doc__)
    generate(args.output,overwrite=args.overwrite)
