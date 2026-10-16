"""Reproducible ready-to-open projects and source CSVs for the joint workflow."""
from pathlib import Path
import csv
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from examples.joint_pcs_rdc_demo.generate_demo import make_demo
from logic.project_io import build_pcsp_text, load_project_document
from logic.joint_pcs_rdc_workflow import execute_fit, result_status
from ui.joint_pcs_rdc_workspace import compile_snapshot


def scenario_state(scenario):
    coords,ids,labels,pcs,rows,truth=make_demo()
    if scenario=='ambiguous':
        coords=np.array([(0.,0.,6+i//2*1.5+(i%2)*1.02) for i in range(24)])
        tensor=np.diag([-truth[0]/3+truth[1]/2,-truth[0]/3-truth[1]/2,2*truth[0]/3])*1e-32
        metres=coords*1e-10
        pcs=dict(zip(ids,np.einsum('ni,ij,nj->n',metres,tensor,metres)/(4*np.pi*np.linalg.norm(metres,axis=1)**5)*1e6))
        # The independent scalar equation for all z-oriented bonds.
        value=-(1.054571817e-34*2.67522187e8*-2.71262e7*14.1**2/(8*np.pi**2*15*1.380649e-23*298*1.02e-10**3))*2*truth[0]*1e-32
        for row in rows:
            row['rdc_exp']=float(value)
    elif scenario=='noisy':
        rng=np.random.default_rng(20261013)
        pcs={rid:float(value+rng.normal(0,.02)) for rid,value in pcs.items()}
        for row in rows:
            row['rdc_exp']+=float(rng.normal(0,.5))
    elif scenario!='noiseless':
        raise ValueError('Unknown scenario')
    for i,row in enumerate(rows):
        row.update(ref1=2*i+1,ref2=2*i+2,source_row=i+2)
        row.pop('atom1',None)
        row.pop('atom2',None)
    raw=[(label[0],*xyz) for label,xyz in zip(labels,coords)]
    records=[{'id':f'demo-ref:{rid}','assignment':f'Ref {rid}','pcs_ppm':float(value),'uncertainty_ppm':.02,
              'target':{'site_member_ref_ids':[[rid]],'scope':'Single atom','matching_status':'unique','averaging_hint':'none'}} for rid,value in pcs.items()]
    return {'atom_data_raw':raw,'atom_data':raw,'atom_ids_raw':ids,'atom_ids':ids,
            'x0':0.,'y0':0.,'z0':0.,'tensor':-2.,'rh_dchi_rh':0.,'delta_exp_values':pcs,
            'experimental_pcs_observations':records,'advfit_rdc_rows':rows,
            'advfit_rdc_options':dict(B0=14.1,T=298,w_rdc=1,use_pcs_errors=True)}


def generate(destination, *, overwrite=False):
    from tools.demo_support import prepare_output
    names = [f'{s}/{n}' for s in ('noiseless', 'noisy', 'ambiguous') for n in ('demo.pcsp', 'structure.xyz', 'rdc.csv', 'pcs.csv')]
    destination=prepare_output(destination, [*names, 'invalid-rdc.csv', 'validation.csv'], overwrite=overwrite)
    summary=[]
    for scenario in ('noiseless','noisy','ambiguous'):
        folder=destination/scenario
        folder.mkdir(parents=True,exist_ok=True)
        state=scenario_state(scenario)
        try:
            result=execute_fit(compile_snapshot(state))
            state['advfit_rdc_last_result']=result
        except ValueError as exc:
            if scenario!='ambiguous' or 'rank' not in str(exc):
                raise
            result=None
        project=folder/'demo.pcsp'
        text='\n'.join(line.rstrip() for line in build_pcsp_text(state).splitlines())+'\n'
        project.write_text(text,encoding='utf-8')
        loaded=load_project_document(project,{}).data
        status=result_status(loaded['advfit_rdc_last_result'],compile_snapshot(loaded)) if result else 'Not fitted (rank deficient)'
        if result and status!='Current':
            raise RuntimeError('Saved demo result must restore as Current')
        xyz=[str(len(state['atom_data_raw'])),'Synthetic labelled probes; metal at origin; fixed tensor frame']
        xyz += [f'{el} {x:.12g} {y:.12g} {z:.12g}' for el,x,y,z in state['atom_data_raw']]
        (folder/'structure.xyz').write_text('\n'.join(xyz)+'\n',encoding='utf-8')
        for filename,records in (
            ('rdc.csv',state['advfit_rdc_rows']),
            ('pcs.csv',[dict(ref_id=r,pcs_exp=float(p),pcs_err=.02) for r,p in state['delta_exp_values'].items()])):
            with (folder/filename).open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        summary.append(dict(scenario=scenario,expected_ax=-18,expected_rh=5,
            fitted_ax=result['dchi_ax'] if result else '',fitted_rh=result['dchi_rh'] if result else '',
            pcs_rmsd=result['rmsd_pcs'] if result else '',rdc_rmsd=result['rmsd_rdc'] if result else '',status=status))
    (destination/'invalid-rdc.csv').write_text('ref1,ref2,rdc_exp,rdc_err,nuc1,nuc2\n1,2,4,.5,unsupported,1H\n',encoding='utf-8')
    with (destination/'validation.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row)


if __name__=='__main__':
    from tools.demo_support import generation_args
    args = generation_args(Path(__file__).resolve().parent, __doc__)
    generate(args.output, overwrite=args.overwrite)
