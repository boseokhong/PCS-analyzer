"""Fit a bundled project without launching Tk or applying its tensor."""
from pathlib import Path
import argparse
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from logic.project_io import load_project_document
from logic.joint_pcs_rdc_workflow import execute_fit, validate_result, parse_rdc_csv
from ui.joint_pcs_rdc_workspace import compile_snapshot


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scenario',choices=('noiseless','noisy','ambiguous'),default='noisy')
    args=parser.parse_args()
    folder=Path(__file__).resolve().parent
    document=load_project_document(folder/args.scenario/'demo.pcsp',{})
    try:
        result=execute_fit(compile_snapshot(document.data))
    except ValueError as exc:
        if args.scenario=='ambiguous' and 'rank' in str(exc):
            print('Expected rank-deficiency rejection:',exc)
            return
        raise
    validate_result(result)
    print(f"{args.scenario}: axial={result['dchi_ax']:.6f}, rhombic={result['dchi_rh']:.6f}")
    print(f"PCS RMSD={result['rmsd_pcs']:.6g} ppm; RDC RMSD={result['rmsd_rdc']:.6g} Hz; rank={result['rank']}")
    with (folder/'invalid-rdc.csv').open(encoding='utf-8') as stream:
        try:
            parse_rdc_csv(stream)
        except ValueError:
            print('Invalid-nucleus CSV correctly rejected')
        else:
            raise AssertionError('Invalid CSV must be rejected')


if __name__=='__main__':
    main()
