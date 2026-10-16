"""Known-tensor observed shifts for explicit NMR pairing, PCS derivation and fitting."""
from pathlib import Path
import csv
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.standard_fitting_demo.generate_demo import make_state

SCENARIOS = ("direct", "contact_corrected")


def starting_state():
    state, _ = make_state("rhombic")
    for key in ("atom_data_raw", "atom_data", "atom_ids_raw", "atom_ids"):
        state[key] = state[key][:14]
    state["tensor"], state["rh_dchi_rh"] = 10., 2.
    state["app_settings"] = {"default_dchi_ax":10.}
    state["experimental_pcs_observations"] = []
    state["delta_exp_values"] = {}
    state["fit_weighting_var"] = "unweighted"
    return state


def derive_and_transfer(folder):
    """Use the real import/pair/provider/structured-handoff owners without opening UI."""
    from nmr_workbench.ui.view_model import WorkbenchState
    from nmr_workbench.io.observable_table import parse_observation_table, apply_observation_import
    from nmr_workbench.analysis.shift_analysis import build_shift_comparison
    from nmr_workbench.analysis.pcs_derivation import DirectDeltaShiftProvider, ManualFCSProvider, PCSDerivationContext
    from nmr_workbench.integration.structure_aware_pcs_targets import PCSStructureTargetChoice, SCOPE_SINGLE
    from nmr_workbench.integration.pcs_structured_handoff import build_handoff_rows, apply_structured_pcs_handoff
    folder = Path(folder)
    model = WorkbenchState()
    for filename in ("paramagnetic.csv", "diamagnetic.csv"):
        records = parse_observation_table((folder/filename).read_text(encoding="utf-8"))
        apply_observation_import(model, records, source=filename)
    para = next(d for d in model.datasets if d.experimental_role == "paramagnetic")
    dia = next(d for d in model.datasets if d.experimental_role == "diamagnetic")
    rows, summary = build_shift_comparison(para,dia)
    assert summary.matched == 12
    with (folder/"mapping_and_truth.csv").open(encoding="utf-8",newline="") as f:
        truth = {r["assignment"]:r for r in csv.DictReader(f)}
    corrections = {a:float(r["contact_ppm"]) for a,r in truth.items()}
    provider = ManualFCSProvider() if any(corrections.values()) else DirectDeltaShiftProvider()
    context = PCSDerivationContext(para.id,dia.id,temperature_K=300.,corrections_ppm=corrections)
    values = {r.assignment:provider.derive(r,context) for r in rows}
    targets = {a:PCSStructureTargetChoice(key=f'single:{r["ref_id"]}',scope=SCOPE_SINGLE,display=a,
                site_member_ref_ids=((int(r["ref_id"]),),),primary_ref_id=int(r["ref_id"])) for a,r in truth.items()}
    state = starting_state()
    applied = apply_structured_pcs_handoff(state,build_handoff_rows(values,targets))
    assert applied.applied == 12
    for a,v in values.items():
        assert abs(v.pcs_ppm-float(truth[a]["pcs_ppm"])) < 1e-9
    return state


def generate(destination, *, overwrite=False):
    from tools.demo_support import prepare_output, demo_project_text
    names = [f'{s}/{n}' for s in SCENARIOS for n in
             ('start.pcsp','prepared.pcsp','structure.xyz','paramagnetic.csv','diamagnetic.csv','mapping_and_truth.csv')]
    destination = prepare_output(destination,names,overwrite=overwrite)
    source,_ = make_state("rhombic")
    pcs = np.array([r["pcs_ppm"] for r in source["experimental_pcs_observations"][:12]])
    dia = np.linspace(1.2,8.5,12)
    for scenario in SCENARIOS:
        folder=destination/scenario;folder.mkdir(exist_ok=True)
        contact = np.zeros(12) if scenario == "direct" else np.linspace(-.3,.4,12)
        rows = [dict(assignment=f'H{i+1}',ref_id=i+3,diamagnetic_ppm=float(dia[i]),
                     paramagnetic_ppm=float(dia[i]+pcs[i]+contact[i]),contact_ppm=float(contact[i]),
                     pcs_ppm=float(pcs[i]),temperature_k=300.,synthetic=True) for i in range(12)]
        with (folder/'mapping_and_truth.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        for filename,role,key in (('paramagnetic.csv','paramagnetic','paramagnetic_ppm'),('diamagnetic.csv','diamagnetic','diamagnetic_ppm')):
            with (folder/filename).open('w',newline='',encoding='utf-8') as f:
                writer=csv.writer(f);writer.writerow(('assignment','shift_ppm','experiment','role','nucleus','temperature_k','quantity','included'))
                writer.writerows((r['assignment'],r[key],role,role,'1H',300,'observed_shift','true') for r in rows)
        states = {'start.pcsp':starting_state(),'prepared.pcsp':derive_and_transfer(folder)}
        for filename,state in states.items():
            (folder/filename).write_text(demo_project_text(state),encoding='utf-8')
        raw=states['start.pcsp']['atom_data_raw']
        xyz=[str(len(raw)),'Synthetic known PCS probes; Nd Ref 1, N Ref 2; H1-H12 map to Ref 3-14']
        xyz += [f'{e} {x:.12g} {y:.12g} {z:.12g}' for e,x,y,z in raw]
        (folder/'structure.xyz').write_text('\n'.join(xyz)+'\n',encoding='utf-8')


if __name__=='__main__':
    from tools.demo_support import generation_args
    args=generation_args(Path(__file__).resolve().parent,__doc__)
    generate(args.output,overwrite=args.overwrite)
