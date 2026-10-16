"""Synthetic group-detection fixtures; use the existing Nd example for molecular displays."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def atom_rows():
    directions=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)/np.sqrt(3.)
    rows=[('Nd',-8.,0.,0.),('C',0.,0.,0.)]
    rows += [('C',*map(float,1.54*u)) for u in directions]
    def terminals(center,u,element,length):
        e1=np.cross(u,[0.,0.,1.]);e1/=np.linalg.norm(e1);e2=np.cross(u,e1)
        return [(element,*map(float,center+length*(u/3+np.sqrt(8/9)*(np.cos(a)*e1+np.sin(a)*e2))))
                for a in (0.,2*np.pi/3,4*np.pi/3)]
    for u in directions[1:]:
        rows += terminals(1.54*u,u,'H',1.09)
    rows += [('C',6.46,0.,0.),('C',8.,0.,0.)]
    rows += terminals(np.array([8.,0.,0.]),np.array([1.,0.,0.]),'F',1.34)
    return rows


def generate(destination, *, overwrite=False):
    from tools.demo_support import prepare_output
    destination=prepare_output(destination,('groups.xyz',),overwrite=overwrite)
    rows=atom_rows()
    lines=[str(len(rows)),'Synthetic disconnected tBu/CF3 fragments for group detection; not chemically optimized']
    lines += [f'{e} {x:.12g} {y:.12g} {z:.12g}' for e,x,y,z in rows]
    (destination/'groups.xyz').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    from tools.demo_support import generation_args
    args=generation_args(Path(__file__).resolve().parent,__doc__)
    generate(args.output,overwrite=args.overwrite)
