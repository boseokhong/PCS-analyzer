"""Deterministic synthetic Ln series; not experimental model validation."""
from pathlib import Path
import sys
import csv
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))


def make_state(noisy=True):
    from logic.lanthanide_workflow import CD,coefficients
    rng=np.random.default_rng(20261013)
    coords=rng.normal(size=(8,3));coords*=np.linspace(7,16,8)[:,None]/np.linalg.norm(coords,axis=1)[:,None]
    raw=[('H',*map(float,p)) for p in coords];refs=list(range(1,9))
    g=(3*coords[:,2]**2/np.sum(coords**2,axis=1)-1)/np.linalg.norm(coords,axis=1)**3
    lns=['Nd','Eu','Tb','Dy','Er','Yb'];data=[];sz=coefficients(300)['sz']
    contact=np.linspace(-.03,.045,8)
    for ln in lns:
        rows=[];pcs_rows=[]
        for i in range(8):
            model=80*g[i]*CD[ln];shift=model+contact[i]*sz[ln]
            rows.append(dict(site=f'H{i+1}',ref_id=i+1,value=float(shift+(.015*rng.normal() if noisy else 0)),error=.03,geometry=float(g[i])))
            # Independent Cartesian susceptibility equation, including rhombicity.
            x,y,z=coords[i];ax,rh=CD[ln]*.30,CD[ln]*.055
            pcm=(ax*(3*z*z/(x*x+y*y+z*z)-1)+rh*1.5*(x*x-y*y)/(x*x+y*y+z*z))*1e4/(12*np.pi*np.linalg.norm(coords[i])**3)
            pcs_rows.append(dict(site=f'H{i+1}',ref_id=i+1,value=float(pcm),error=.02,geometry=float(g[i])))
        data += [dict(id=f'para:{ln}',name=f'{ln} total shift',ln=ln,kind='paramagnetic',rows=rows),
                 dict(id=f'pcs:{ln}',name=f'{ln} PCS',ln=ln,kind='pcs',rows=pcs_rows)]
    return dict(atom_data_raw=raw,atom_data=raw,atom_ids_raw=refs,atom_ids=refs,
                x0=0.,y0=0.,z0=0.,tensor=-2.,rh_dchi_rh=0.,delta_exp_values={},advfit_ml_datasets=data,
                advfit_ml_options=dict(mode='reilley',temperature=300.,method='direct',use_errors=True,
                                      reference_ln='Dy',nucleus_i='H1',nucleus_j='H2',nucleus_c=''))


def generate(folder, *, overwrite=False):
    from logic.lanthanide_workflow import execute_analysis,normalize_options
    from logic.project_io import build_pcsp_text,load_project_document
    from ui.lanthanide_workspace import compile_snapshot
    from tools.demo_support import prepare_output
    names = [f'{s}/{n}' for s in ('noiseless', 'noisy') for n in ('demo.pcsp', 'paramagnetic.csv')]
    folder=prepare_output(folder, [*names, 'invalid.csv'], overwrite=overwrite)
    for name,noisy in (('noiseless',False),('noisy',True)):
        state=make_state(noisy);outputs={}
        for mode in ('tensor','reilley','scaling','nuclei','geometry','parameters'):
            state['advfit_ml_options']['mode']=mode;outputs[mode]=execute_analysis(compile_snapshot(state))
        state['advfit_ml_options']['mode']='reilley';state['advfit_ml_results']=outputs
        dest=folder/name;dest.mkdir(parents=True,exist_ok=True)
        path=dest/'demo.pcsp';path.write_text('\n'.join(line.rstrip() for line in build_pcsp_text(state).splitlines())+'\n',encoding='utf-8')
        loaded=load_project_document(path,{}).data
        assert len(loaded['advfit_ml_results'])==6
        with (dest/'paramagnetic.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);lns=['Nd','Eu','Tb','Dy','Er','Yb'];writer.writerow(['Nucleus','ref_id','G_common',*lns,'error'])
            lookup={d['ln']:d for d in state['advfit_ml_datasets'] if d['kind']=='paramagnetic'}
            for i in range(8):writer.writerow([f'H{i+1}',i+1,lookup['Nd']['rows'][i]['geometry'],*[lookup[ln]['rows'][i]['value'] for ln in lns],.03])
        print(name, 'six recorded analyses saved/reopened')
    (folder/'invalid.csv').write_text('Nucleus,Nd,Eu,Yb\nH1,nan,1,2\n',encoding='utf-8')


def main():
    from tools.demo_support import generation_args
    args = generation_args(Path(__file__).resolve().parent, __doc__)
    generate(args.output, overwrite=args.overwrite)


if __name__=='__main__':main()
