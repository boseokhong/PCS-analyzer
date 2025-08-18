# tools/plot_pcs_origin.py

import sys, os, argparse, traceback, struct, platform
import numpy as np
import pandas as pd

def main():
    try:
        ap = argparse.ArgumentParser(description='PCS CSV -> Origin Polar Plot')
        ap.add_argument('csv', help='path to *_pcs.csv')
        ap.add_argument('--half', action='store_true', help='0–90° mirror view')
        ap.add_argument('--rmax', type=float, default=10.0, help='radial max (Å)')
        ap.add_argument('--tick', type=float, default=2.0, help='radial major tick (Å)')
        ap.add_argument('--outdir', default=None, help='output dir (default: same as CSV)')
        args = ap.parse_args()

        csv_path = os.path.abspath(args.csv)
        if not os.path.exists(csv_path):
            print('[ERR] CSV not found:', csv_path, flush=True)
            sys.exit(1)

        print('[INFO] Python:', sys.version.replace('\n', ' '), flush=True)
        print('[INFO] Python-arch:', struct.calcsize("P")*8, 'bit', flush=True)
        print('[INFO] Platform:', platform.platform(), flush=True)

        # originpro 임포트 시도
        try:
            import originpro as op
        except Exception as e:
            print('[ERR] import originpro failed:', e, flush=True)
            traceback.print_exc()
            sys.exit(2)

        print('[INFO] Origin exe path:', getattr(op, 'path')('o'), flush=True)
        print('[INFO] Origin UFF path:', getattr(op, 'path')('u'), flush=True)

        # 출력 경로 구성
        outdir = os.path.abspath(args.outdir) if args.outdir else os.path.dirname(csv_path)
        os.makedirs(outdir, exist_ok=True)
        base = os.path.splitext(os.path.basename(csv_path))[0]
        opju_out = os.path.join(outdir, base + "_polar.opju")
        png_out  = os.path.join(outdir, base + "_polar.png")
        print('[INFO] Out OPJU:', opju_out, flush=True)
        print('[INFO] Out PNG :', png_out, flush=True)

        # Origin 열기
        op.open()
        op.set_show(True)  # UI 보이게(원하면 False)

        # ===== 그래프 생성 =====
        df = pd.read_csv(csv_path)
        if 'Theta_deg' not in df.columns:
            df['Theta_deg'] = np.degrees(df['Theta'])

        wb = op.new_book(type='w', lname='PCS_Data')
        wb.from_df(df)

        wks = wb
        # X/Y 지정
        theta_col = [i for i, c in enumerate(df.columns, start=1) if c == 'Theta_deg'][0]
        rcols = [i for i, c in enumerate(df.columns, start=1) if c.startswith('R ')]
        wks.cols[theta_col-1].set_designation('X')
        for c in rcols:
            wks.cols[c-1].set_designation('Y')

        g = op.new_graph(template='POLAR')
        ly = g[0]
        # plot_type=200이 버전에 따라 다를 수 있어 기본 add_plot 뒤 폴라 설정으로 우회
        for c in rcols:
            ly.add_plot(wks, xcol=theta_col, ycol=c)  # 기본 라인 추가

        # 레전드 라벨링 (PCS 열에서 값 읽기)
        pcs_cols = [i for i, cn in enumerate(df.columns, start=1) if cn.startswith('PCS (ppm) ')]
        pcs_map = {}
        for pc in pcs_cols:
            n = df.columns[pc-1].split()[-1]
            for rc in rcols:
                if df.columns[rc-1].split()[-1] == n:
                    pcs_map[rc] = f"{df.iloc[0, pc-1]:.1f} ppm"

        g.activate()
        # 그래프를 Polar로 전환 (템플릿이 이미 Polar면 생략 가능)
        # layer -polar angle/radius 설정
        if args.half:
            op.lt_exec('layer -polar angle:=0,90;')
        else:
            op.lt_exec('layer -polar angle:=0,180;')
        op.lt_exec(f'layer -polar radius:=0,{args.rmax},{args.tick};')
        op.lt_exec('legend -s 0;')  # custom legend
        for ii, _ in enumerate(rcols, start=1):
            lbl = pcs_map.get(rcols[ii-1], f"R{ii}")
            op.lt_exec(f'legend.text$ = "{{\\L{ii}}} {lbl}";')

        # ===== 저장 =====
        ok = op.save(opju_out)  # 프로젝트 파일 저장
        print('[INFO] op.save returned:', ok, flush=True)

        # PNG 내보내기
        g.activate()
        op.lt_exec(f'expGraph type:=png filename:="{png_out}" overwrite:=1;')

        print('SAVED_OPJU:', opju_out, flush=True)
        print('SAVED_PNG:',  png_out, flush=True)
        sys.exit(0)

    except SystemExit as e:
        raise
    except Exception as e:
        print('[ERR] Uncaught exception:', e, flush=True)
        traceback.print_exc()
        sys.exit(99)

if __name__ == '__main__':
    main()
