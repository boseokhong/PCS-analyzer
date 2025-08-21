# logic/table_utils.py

import numpy as np
from logic.chem_constants import AVOGADRO_CONSTANT
import pandas as pd
import os
import tkinter as tk
from tkinter import ttk, Toplevel

def export_delta_exp_template(state):
    """
    현재 테이블(보이는 행들)의 Ref, Atom을 기준으로
    δ_Exp 입력용 템플릿을 CSV/XLSX로 내보낸다.
    - 기존에 입력된 δ_Exp가 있으면 그 값을 채워서 저장(덮기 편의용)
    - 파일 확장자: .csv 또는 .xlsx 권장
    """
    tree = state.get('tree')
    if tree is None or not tree.get_children():
        state['messagebox'].showerror("Export δ_Exp template", "No data to export")
        return

    # 테이블에서 현재 보이는 행들을 모은다
    rows = []
    for item in tree.get_children():
        vals = tree.item(item, "values")
        # columns = ('Ref','Atom','X','Y','Z','G_i','δ_PCS','δ_Exp')
        ref = vals[0]
        atom = vals[1]
        dex  = vals[7] if len(vals) >= 8 else ""
        rows.append((ref, atom, dex))

    df = pd.DataFrame(rows, columns=["Ref", "Atom", "δ_Exp"])

    # 저장 경로
    fd = state['filedialog'].asksaveasfilename(
        title="Export δ_Exp template",
        defaultextension=".csv",
        filetypes=[("CSV file","*.csv"), ("Excel workbook","*.xlsx")]
    )
    if not fd:
        return

    base, ext = os.path.splitext(fd)
    ext = ext.lower()

    try:
        if ext == ".xlsx":
            df.to_excel(fd, index=False)
        else:
            # 기본 CSV
            df.to_csv(fd, index=False)
        state['messagebox'].showinfo("Export δ_Exp template", f"저장 완료:\n{fd}")
    except Exception as e:
        state['messagebox'].showerror("Export δ_Exp template", f"Save failed:\n{e}")

def _parse_delta_cell(cell):
    """빈칸/None/NaN/비수치 -> None (지우기), 숫자/콤마소수 -> float"""
    if cell is None:
        return None
    s = str(cell).strip()
    if s == "" or s.lower() == "nan":
        return None
    s = s.replace(",", ".")
    try:
        val = float(s)
        if np.isnan(val):
            return None
        return val
    except Exception:
        return None

def import_delta_exp_file(state, plot_cartesian_graph_fn):
    fd = state['filedialog'].askopenfilename(
        title="Load δ_Exp file",
        filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx;*.xls"), ("All files", "*.*")]
    )
    if not fd:
        return

    try:
        if fd.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(fd)
        else:
            df = pd.read_csv(fd)
    except Exception as e:
        state['messagebox'].showerror("Import δ_Exp", f"Cannot read the file:\n{e}")
        return

    try:
        rows = _build_preview_rows_from_df(state, df)
    except Exception as e:
        state['messagebox'].showerror("Import δ_Exp", str(e))
        return

    if not rows:
        state['messagebox'].showwarning("Import δ_Exp", "No data to apply")
        return

    _show_delta_preview_and_apply(state, rows, os.path.basename(fd), plot_cartesian_graph_fn)

def import_delta_exp_from_clipboard(state, plot_cartesian_graph_fn):
    top = Toplevel(state['root'])
    top.title("Paste δ_Exp from clipboard")
    top.geometry("720x700")

    ttk.Label(top, text="Paste your data and click “Preview”. Format (ex) Ref, δ_Exp").pack(anchor="w", padx=8, pady=6)
    txt = tk.Text(top, height=16)
    txt.pack(fill="both", expand=True, padx=8, pady=6)

    # 미리 헤더 한 줄 프리셋
    txt.insert("1.0", "Ref, δ_Exp\n")
    txt.mark_set("insert", "2.0")

    def _preview():
        raw = txt.get("1.0", "end").strip()
        if not raw:
            state['messagebox'].showwarning("Paste δ_Exp", "No data found")
            return

        # === 간단 구분자 추정 ===
        def _guess_sep(first_line: str):
            if "," in first_line:  return ","
            if "\t" in first_line: return "\t"
            if ";" in first_line:  return ";"
            return r"\s+"  # 공백 다중 구분

        lines = [ln for ln in raw.splitlines() if ln.strip()]
        first = lines[0] if lines else ""
        sep = _guess_sep(first)

        from io import StringIO
        sio = StringIO(raw)

        try:
            # 1차: 헤더 있다고 가정 (우리가 "Ref, δ_Exp" 미리 넣어줬기 때문)
            df = pd.read_csv(sio, sep=sep, engine="python",
                             header=0, dtype=str, na_filter=False)
        except Exception:
            # 2차: 헤더 없음으로 재시도 후 컬럼명 부여
            sio.seek(0)
            df = pd.read_csv(sio, sep=sep, engine="python",
                             header=None, dtype=str, na_filter=False)
            if df.shape[1] == 3:
                df.columns = ["Ref","Atom","δ_Exp"]
            elif df.shape[1] == 2:
                df.columns = ["Ref","δ_Exp"]

        try:
            rows = _build_preview_rows_from_df(state, df)  # 기존 함수 재사용
        except Exception as e:
            state['messagebox'].showerror("Paste δ_Exp", str(e))
            return

        if not rows:
            state['messagebox'].showwarning("Paste δ_Exp", "No data to apply")
            return

        _show_delta_preview_and_apply(state, rows, "clipboard", plot_cartesian_graph_fn)
        top.destroy()

    btns = ttk.Frame(top); btns.pack(fill="x", padx=8, pady=8)
    ttk.Button(btns, text="Preview", command=_preview).pack(side="right")
    ttk.Button(btns, text="Cancel", command=top.destroy).pack(side="right", padx=6)

def undo_last_delta_import(state, plot_cartesian_graph_fn=None):
    prev = state.get('_delta_values_undo')
    if prev is None:
        state['messagebox'].showwarning("Undo", "No undo buffer available.")
        return
    state['delta_values'] = prev
    state['update_graph']()
    if plot_cartesian_graph_fn:
        plot_cartesian_graph_fn(state)
    state['messagebox'].showinfo("Undo", "Restored to the previous δ_Exp state.")


def _show_delta_preview_and_apply(state, mapped_rows, source_label, plot_cartesian_graph_fn):
    """
    mapped_rows: 리스트[ dict( file_id, file_atom, file_delta_raw, mapped_id, parsed_delta, action ) ]
      - action: 'set' | 'clear' | 'skip'
    부분 적용(체크박스) + 되돌리기 버퍼 저장 후 적용
    """
    top = Toplevel(state['root'])
    top.title("δ_Exp Import Preview")
    top.geometry("860x520")

    ttk.Label(top, text=f"Source: {source_label}").pack(anchor="w", padx=8, pady=(8,2))

    # 적용 체크 플래그 (기본: set/clear만 True)
    apply_flags = [ (r['action'] in ('set','clear')) and (r.get('mapped_id') is not None) for r in mapped_rows ]

    info_var = tk.StringVar()
    def _refresh_info():
        tot = len(mapped_rows)
        sel = sum(apply_flags)
        set_n = sum(1 for f,r in zip(apply_flags,mapped_rows) if f and r['action']=='set')
        clr_n = sum(1 for f,r in zip(apply_flags,mapped_rows) if f and r['action']=='clear')
        skp_n = tot - sel
        info_var.set(f"Selected: {sel}  |  set: {set_n}, clear: {clr_n}  |  skipped: {skp_n}")

    ttk.Label(top, textvariable=info_var).pack(anchor="w", padx=8, pady=(0,6))

    cols = ("apply","file_id","file_atom","file_delta","mapped_id","parsed_delta","action")
    tv = ttk.Treeview(top, columns=cols, show="headings", height=16)
    for c, w in zip(cols, (60,100,100,160,100,120,80)):
        tv.heading(c, text=c)
        tv.column(c, width=w, anchor="center")
    tv.pack(fill="both", expand=True, padx=8, pady=6)

    def _flag_symbol(b): return "✓" if b else ""
    for i, r in enumerate(mapped_rows):
        tv.insert("", "end", values=(
            _flag_symbol(apply_flags[i]),
            r.get('file_id',''),
            r.get('file_atom',''),
            r.get('file_delta_raw',''),
            r.get('mapped_id',''),
            "" if r.get('parsed_delta') is None else f"{r['parsed_delta']:.6g}",
            r.get('action','')
        ))

    def _toggle_row(i):
        if mapped_rows[i].get('mapped_id') is None or mapped_rows[i].get('action') == 'skip':
            return
        apply_flags[i] = not apply_flags[i]
        item = tv.get_children()[i]
        vals = list(tv.item(item, "values"))
        vals[0] = _flag_symbol(apply_flags[i])
        tv.item(item, values=vals)
        _refresh_info()

    def _toggle_all(to=None):
        for i, r in enumerate(mapped_rows):
            if r.get('mapped_id') is None or r.get('action') == 'skip':
                continue
            apply_flags[i] = bool(to) if to is not None else not apply_flags[i]
            item = tv.get_children()[i]
            vals = list(tv.item(item,"values"))
            vals[0] = _flag_symbol(apply_flags[i])
            tv.item(item, values=vals)
        _refresh_info()

    # 클릭으로 첫 컬럼 토글
    def _on_click(event):
        x, y = event.x, event.y
        iid = tv.identify_row(y)
        col = tv.identify_column(x)
        if not iid or col != "#1":  # apply 칼럼
            return
        idx = tv.index(iid)
        _toggle_row(idx)

    tv.bind("<Button-1>", _on_click)

    btns = ttk.Frame(top); btns.pack(fill="x", padx=8, pady=8)
    ttk.Button(btns, text="All on", command=lambda: _toggle_all(True)).pack(side="left")
    ttk.Button(btns, text="All off", command=lambda: _toggle_all(False)).pack(side="left", padx=4)
    ttk.Button(btns, text="Invert", command=lambda: _toggle_all(None)).pack(side="left")

    def _apply():
        # 되돌리기 버퍼
        state['_delta_values_undo'] = state.get('delta_values', {}).copy()

        dv = state.setdefault('delta_values', {})
        applied_set = applied_clear = 0

        for flag, r in zip(apply_flags, mapped_rows):
            if not flag:
                continue
            rid = r.get('mapped_id')
            if rid is None:
                continue
            if r['action'] == 'set':
                dv[rid] = r['parsed_delta']
                applied_set += 1
            elif r['action'] == 'clear':
                dv.pop(rid, None)
                applied_clear += 1

        state['update_graph']()
        plot_cartesian_graph_fn(state)
        state['messagebox'].showinfo("δ_Exp import", f"Applied set: {applied_set}, clear: {applied_clear}")
        top.destroy()

    def _undo_last():
        prev = state.get('_delta_values_undo')
        if prev is None:
            state['messagebox'].showwarning("Undo", "No undo buffer available.")
            return
        state['delta_values'] = prev
        state['update_graph']()
        plot_cartesian_graph_fn(state)
        state['messagebox'].showinfo("Undo", "Restored to the previous δ_Exp state.")

    ttk.Button(btns, text="Apply", command=_apply).pack(side="right")
    ttk.Button(btns, text="Undo last", command=_undo_last).pack(side="right", padx=6)
    ttk.Button(btns, text="Close", command=top.destroy).pack(side="right", padx=6)

    _refresh_info()

def _build_preview_rows_from_df(state, df):
    """
    DF를 해석해 미리보기용 mapped_rows를 만든다.
    반환: 리스트[ dict(file_id,file_atom,file_delta_raw,mapped_id,parsed_delta,action) ]
    """
    def _norm(s): return str(s).strip().lower()
    cols = { _norm(c): c for c in df.columns }

    ref_col  = next((cols[k] for k in cols if k in ("ref","id","ref_id","rid")), None)
    atom_col = next((cols[k] for k in cols if k in ("atom","element")), None)
    delta_col = next((cols[k] for k in cols if k in ("δ_exp","delta_exp","delta","ppm","shift","d_exp","exp")), None)

    mapped_rows = []

    # 헤더가 전혀 없고 2열 이상인 경우
    if delta_col is None and df.shape[1] >= 2 and not str(df.columns.tolist()[0]).strip():
        delta_col = 1
        ref_col = 0

    if delta_col is None and df.shape[1] >= 2 and ref_col is None and atom_col is None:
        # 2열 가정: Ref, δ_Exp
        for _, row in df.iterrows():
            file_id = row.iloc[0]
            try:
                rid = int(file_id)
            except Exception:
                rid = None
            raw = row.iloc[1]
            val = _parse_delta_cell(raw)
            act = 'set' if (rid is not None and val is not None) else ('clear' if (rid is not None and val is None) else 'skip')
            mapped_rows.append(dict(
                file_id=file_id, file_atom="",
                file_delta_raw=str(raw) if raw is not None else "",
                mapped_id=rid, parsed_delta=val, action=act
            ))

    elif ref_col is not None and delta_col is not None:
        # Ref 기반
        for _, row in df.iterrows():
            file_id = row[ref_col]
            try:
                rid = int(str(file_id).strip())
            except Exception:
                rid = None
            raw = row[delta_col]
            val = _parse_delta_cell(raw)
            act = 'set' if (rid is not None and val is not None) else ('clear' if (rid is not None and val is None) else 'skip')
            mapped_rows.append(dict(
                file_id=file_id, file_atom="",
                file_delta_raw=str(raw) if raw is not None else "",
                mapped_id=rid, parsed_delta=val, action=act
            ))

    elif atom_col is not None and delta_col is not None:
        # Atom 기반: 테이블 순서대로 소비
        pool = {}
        for _, row in df.iterrows():
            key = row.get(atom_col, None)
            if key is None or str(key).strip()=="":
                continue
            key = str(key).strip()
            raw = row.get(delta_col, None)
            val = _parse_delta_cell(raw)
            pool.setdefault(key, []).append((raw, val))

        tree = state['tree']
        for item in tree.get_children():
            vals = tree.item(item, "values")
            rid = int(vals[0]); at = str(vals[1]).strip()
            if pool.get(at):
                raw, val = pool[at].pop(0)
                act = 'set' if val is not None else 'clear'
                mapped_rows.append(dict(
                    file_id="", file_atom=at,
                    file_delta_raw=str(raw) if raw is not None else "",
                    mapped_id=rid, parsed_delta=val, action=act
                ))
        # 남은 값(소비 못한 것)은 미적용 → 굳이 preview에 안 올려도 됨

    else:
        # 인식 실패
        raise ValueError("Cannot find δ_Exp columns. Use 'Ref, δ_Exp' or 'Atom, δ_Exp'.")

    return mapped_rows


def clear_delta_exp(state, plot_cartesian_graph_fn):
    """모든 δ_Exp를 초기화(비우기)하고 테이블/플롯을 갱신."""
    state.get('delta_values', {}).clear()
    # 테이블의 δ_Exp 컬럼을 비우려면 update_graph로 재빌드하는 게 가장 안전
    state['update_graph']()
    plot_cartesian_graph_fn(state)
    state['messagebox'].showinfo("δ_Exp", "δ_Exp data initialized.")

def update_molar_value(state, tensor):
    val = tensor * AVOGADRO_CONSTANT * 1e-32
    state['molar_value_label'].config(text=f"Δχ_mol_ax: {val:.2e} m³/mol")

def update_table(state, polar_data, rotated_coords, tensor, delta_values):
    tree = state['tree']
    # 모든 행 비우기
    for r in tree.get_children():
        tree.delete(r)

    # 현재 표시 중인 행의 Ref(ID) 목록
    ids = state.get('current_selected_ids', [])
    state['row_by_id'] = {}  # 클릭/하이라이트 용 역방향 매핑

    for i, ((atom, r, theta), (dx, dy, dz)) in enumerate(zip(polar_data, rotated_coords)):
        ref_id = ids[i] if i < len(ids) else (i + 1)  # 안전장치
        geom_param = (3 * (np.cos(theta))**2 - 1) / (r**3) if r != 0 else 0.0
        geom_value = geom_param
        delta_pcs = (tensor * (geom_value * 1e4)) / (12 * np.pi)
        # Ref ID 로 읽고/쓴다
        delta_exp_str = ""
        if ref_id in delta_values:
            v = delta_values[ref_id]
            delta_exp_str = f"{v:g}"

        item = tree.insert("", state['tk'].END, values=(
            ref_id, atom, f"{dx:.3f}", f"{dy:.3f}", f"{dz:.3f}",
            f"{geom_param:.5f}", f"{delta_pcs: .2f}", delta_exp_str
        ))
        state['row_by_id'][ref_id] = item

def on_delta_entry_change(state, event, delta_values, plot_cartesian_graph_fn):
    tree = state['tree']
    sel = tree.selection()
    if not sel: return
    item = sel[0]
    col = tree.identify_column(event.x)
    if col != '#8':  # δ_Exp 컬럼
        return

    cur = tree.set(item, col)
    prompt = "Enter experimental chemical shift value (ppm):\n(leave blank to clear)"
    new = state['simpledialog'].askstring("Input", prompt, initialvalue=cur)
    if new is None:
        return

    # Ref(ID)
    try:
        ref_id = int(tree.item(item, "values")[0])
    except Exception:
        return

    new_str = new.strip()
    if new_str == "":
        tree.set(item, col, "")
        delta_values.pop(ref_id, None)
        plot_cartesian_graph_fn(state)
        return

    try:
        val = float(new_str)
        tree.set(item, col, new_str)
        delta_values[ref_id] = val
        plot_cartesian_graph_fn(state)
    except ValueError:
        state['messagebox'].showerror(
            "Invalid input",
            "Please enter a valid number for δ_Exp or leave empty to clear."
        )

def calculate_tensor_components_ui(chi_mol_entry, molar_value_label, tensor_xx_label, tensor_yy_label, tensor_zz_label, messagebox):
    """
    chi_mol_entry: 실험 χ_mol (m^3/mol)
    molar_value_label: 'Δχ_mol_ax : ... m³/mol' 형식의 라벨
    tensor_*_label: 결과 출력 라벨들
    """
    try:
        chi_mol = float(chi_mol_entry.get())
        # 라벨에서 Δχ_mol_ax 수치만 파싱
        txt = molar_value_label['text']  # 예: 'Δχ_mol_ax : 1.23e-28 m³/mol'
        delta_chi_ax = float(txt.split(':')[1].strip().split()[0])

        chi_perp = chi_mol - delta_chi_ax/3
        chi_parallel = (2/3)*delta_chi_ax + chi_mol

        tensor_xx_label.config(text=f"χ_xx: {chi_perp:.2e} m³/mol")
        tensor_yy_label.config(text=f"χ_yy: {chi_perp:.2e} m³/mol")
        tensor_zz_label.config(text=f"χ_zz: {chi_parallel:.2e} m³/mol")
    except Exception:
        messagebox.showerror("Input Error", "Please enter valid numerical values for χ_mol.")