"""Reproduce synthetic proton Gamma2 data from an independent SI equation."""
from pathlib import Path
import csv
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def scenario_state(scenario='noisy'):
    distances = np.array([7.5, 9., 11., 13., 16., 19.])
    directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1.]])
    coords = directions / np.linalg.norm(directions, axis=1)[:, None] * distances[:, None]
    tau, moment, field = 3e-9, 3.68, 14.1
    omega = 2.67522187e8 * field
    # Independent simplified proton Gamma2 equation; not production pre_gamma2.
    rates = (1e-7)**2 * (2.67522187e8)**2 * (moment * 9.2740100783e-24)**2 / 15
    rates = rates * (4*tau + 3*tau/(1+(omega*tau)**2)) / (distances*1e-10)**6
    sigma = .05 * rates
    if scenario != 'noiseless':
        rates += sigma * np.array([.5, -.8, .3, .9, -.6, .2])
    raw = [('H', *map(float, xyz)) for xyz in coords]
    rows = [dict(ref_id=i+1, gamma2=float(y), uncertainty=float(e), nucleus='1H',
                 interpretation='gamma2_increment', source_row=i+2) for i,(y,e) in enumerate(zip(rates,sigma))]
    options = dict(B0=field, mu_eff=moment, preset='Nd', use_errors=True, tau_min_ns=.001, tau_max_ns=1000.)
    if scenario == 'boundary':
        options['tau_max_ns'] = 1.
    elif scenario not in ('noisy', 'noiseless'):
        raise ValueError('Unknown PRE scenario')
    return dict(atom_data_raw=raw, atom_data=raw, atom_ids_raw=list(range(1,7)), atom_ids=list(range(1,7)),
                x0=0., y0=0., z0=0., tensor=-2., rh_dchi_rh=0., delta_exp_values={},
                advfit_pre_obs=rows, advfit_pre_options=options)


def generate(destination, *, overwrite=False):
    from logic.project_io import build_pcsp_text, load_project_document
    from logic.pre_workflow import execute_fit, result_status
    from ui.pre_workspace import compile_snapshot
    from tools.demo_support import prepare_output
    names = [f'{s}/{n}' for s in ('noiseless', 'noisy', 'boundary') for n in ('demo.pcsp', 'structure.xyz', 'pre.csv')]
    destination = prepare_output(destination, [*names, 'invalid-pre.csv', 'validation.csv'], overwrite=overwrite)
    summary = []
    for scenario in ('noiseless', 'noisy', 'boundary'):
        folder = destination / scenario
        folder.mkdir(parents=True, exist_ok=True)
        state = scenario_state(scenario)
        result = execute_fit(compile_snapshot(state))
        state['advfit_pre_last_result'] = result
        text = '\n'.join(line.rstrip() for line in build_pcsp_text(state).splitlines())+'\n'
        project = folder/'demo.pcsp'
        project.write_text(text, encoding='utf-8')
        loaded = load_project_document(project, {}).data
        assert result_status(loaded['advfit_pre_last_result'], compile_snapshot(loaded)) == 'Current'
        xyz = ['6', 'Synthetic proton probes, not a molecular structure; metal at origin']
        xyz += [f'{el} {x:.12g} {y:.12g} {z:.12g}' for el,x,y,z in state['atom_data_raw']]
        (folder/'structure.xyz').write_text('\n'.join(xyz)+'\n', encoding='utf-8')
        with (folder/'pre.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['ref_id','gamma2','gamma2_err','nucleus'])
            writer.writerows((r['ref_id'],r['gamma2'],r['uncertainty'],'1H') for r in state['advfit_pre_obs'])
        summary.append(dict(scenario=scenario, truth_tau_ns=3., fitted_tau_ns=result['tau_ns'],
                            rmsd=result['rmsd'], weighted_sum_squares=result['weighted_sum_squares'], boundary=result['boundary']))
    (destination/'invalid-pre.csv').write_text('ref_id,gamma2,gamma2_err,nucleus\n1,4,.2,13C\n', encoding='utf-8')
    with (destination/'validation.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row)


if __name__ == '__main__':
    from tools.demo_support import generation_args
    args = generation_args(Path(__file__).resolve().parent, __doc__)
    generate(args.output, overwrite=args.overwrite)
