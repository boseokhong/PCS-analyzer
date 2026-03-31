# logic/table_utils.py
'''
# Susceptibility tensor definitions

# χ_iso (isotropic magnetic susceptibility):
χ_iso = (χ_xx + χ_yy + χ_zz) / 3
→ equal to the experimentally measured molar susceptibility χ_mol.

# Δχ_ax (axial anisotropy):
Δχ_ax = χ_zz − (χ_xx + χ_yy) / 2
→ describes the axial (uniaxial) deviation from isotropy.

# Δχ_rh (rhombic anisotropy):
Δχ_rh = χ_xx − χ_yy
→ quantifies the in-plane anisotropy (rhombicity).

# Inversion (used in this program):
χ_xx = χ_iso − Δχ_ax/3 + Δχ_rh/2
χ_yy = χ_iso − Δχ_ax/3 − Δχ_rh/2
χ_zz = χ_iso + 2Δχ_ax/3

# Special case (Δχ_rh = 0):
→ χ_xx = χ_yy = χ_perp  (axially symmetric tensor)
'''

'''
# Units:
Δχ_ax, Δχ_rh are handled internally in units of 10^−32 m^3 (per molecule),
while χ_iso, χ_xx, χ_yy, χ_zz are reported in m^3/mol.

# ============================================================
# Additional notes on tensor reconstruction
# ============================================================

# Two computational modes are supported in this program:

# ------------------------------------------------------------
# (1) Normal mode  → χ_iso = χ_mol (experimental input given)
# ------------------------------------------------------------
# If χ_mol is provided, χ_iso is taken as χ_mol and the full
# susceptibility tensor is reconstructed as:

# χ_xx = χ_mol − Δχ_ax/3 + Δχ_rh/2
# χ_yy = χ_mol − Δχ_ax/3 − Δχ_rh/2
# χ_zz = χ_mol + 2Δχ_ax/3

# Special case (Δχ_rh = 0):
# → axial symmetry (χ_xx = χ_yy)

# In this case:
# χ_xx = χ_yy = χ_mol − Δχ_ax/3
# χ_zz = χ_mol + 2Δχ_ax/3

# ------------------------------------------------------------
# (2) Traceless fallback mode  → χ_iso = 0 (no χ_mol input)
# ------------------------------------------------------------
# If χ_mol is not provided, the tensor is reconstructed assuming
# a traceless diagonal tensor:

# χ_xx + χ_yy + χ_zz = 0

# From the definition:
# Δχ_ax = χ_zz − (χ_xx + χ_yy)/2

# This gives:
# χ_zz = (2/3) Δχ_ax

# The in-plane sum is then fixed by traceless condition:
# χ_xx + χ_yy = −χ_zz = −(2/3) Δχ_ax

# If Δχ_rh = 0 (axial symmetry):
# χ_xx = χ_yy = −(1/3) Δχ_ax
# χ_zz =  (2/3) Δχ_ax

# If Δχ_rh ≠ 0 (rhombic case):
# χ_xx − χ_yy = Δχ_rh
#
# Solving:
# χ_xx = −Δχ_ax/3 + Δχ_rh/2
# χ_yy = −Δχ_ax/3 − Δχ_rh/2
# χ_zz =  2Δχ_ax/3


# ------------------------------------------------------------
# Physical meaning
# ------------------------------------------------------------
# Normal mode:
#   Uses experimental χ_mol and preserves absolute scale.
#
# Traceless mode:
#   Sets χ_iso = 0 and reconstructs only the anisotropic part.
#   Useful for testing, visualization, or when χ_mol is unknown.
#   Absolute isotropic offset is not represented in this mode.
# ============================================================

'''


import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import ttk, Toplevel

from logic.chem_constants import AVOGADRO_CONSTANT
from logic.nmr_delta_data_manager import push_layers_to_nmr_if_open

def _delta_kind_config(kind: str):
    kind = (kind or "").lower().strip()

    # δ_Exp (main window table)
    if kind in ("exp", "delta_exp", "deltaexp", "δ_exp", "δexp"):
        return (
            "delta_exp_values",
            "_delta_exp_undo",
            ("δ_exp", "δ_Exp", "delta_exp", "delta_Exp", "exp", "delta values"),
        )
    # δ_obs (NMR window)
    if kind in ("obs", "delta_obs", "observed"):
        return (
            "delta_obs_values",
            "_delta_obs_undo",
            ("δ_obs", "delta_obs", "obs", "ppm", "shift"),
        )
    # δ_dia (NMR window)
    if kind in ("dia", "delta_dia", "diamagnetic"):
        return (
            "delta_dia_values",
            "_delta_dia_undo",
            ("δ_dia", "delta_dia", "dia", "diamagnetic", "delta_diamagnetic"),
        )
    raise ValueError(f"Unknown kind: {kind}")

#========Main window table NMR===========
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
            df.to_csv(fd, index=False, encoding="utf-8-sig")
        state['messagebox'].showinfo("Export δ_Exp template", f"Save completed:\n{fd}")
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
            try:
                df = pd.read_csv(fd, encoding="utf-8-sig")
            except Exception:
                df = pd.read_csv(fd, encoding="cp1252")
    except Exception as e:
        state['messagebox'].showerror("Import δ_Exp", f"Cannot read the file:\n{e}")
        return

    try:
        rows = _build_preview_rows_from_df(state, df, kind="exp")
    except Exception as e:
        state['messagebox'].showerror("Import δ_Exp", str(e))
        return

    if not rows:
        state['messagebox'].showwarning("Import δ_Exp", "No data to apply")
        return

    _show_delta_preview_and_apply(
        state, rows, os.path.basename(fd), plot_cartesian_graph_fn,
        kind="exp"
    )

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
            rows = _build_preview_rows_from_df(state, df, kind="exp")  # 기존 함수 재사용
        except Exception as e:
            state['messagebox'].showerror("Paste δ_Exp", str(e))
            return

        if not rows:
            state['messagebox'].showwarning("Paste δ_Exp", "No data to apply")
            return

        _show_delta_preview_and_apply(
            state, rows, "clipboard", plot_cartesian_graph_fn,
            kind="exp"
        )
        top.destroy()

    btns = ttk.Frame(top); btns.pack(fill="x", padx=8, pady=8)
    ttk.Button(btns, text="Preview", command=_preview).pack(side="right")
    ttk.Button(btns, text="Cancel", command=top.destroy).pack(side="right", padx=6)

def undo_last_delta_import(state, plot_cartesian_graph_fn=None):
    prev = state.get('_delta_exp_undo')
    if prev is None:
        state['messagebox'].showwarning("Undo", "No undo buffer available.")
        return
    state['delta_exp_values'] = prev
    state['update_graph']()
    if plot_cartesian_graph_fn:
        plot_cartesian_graph_fn(state)
    state['messagebox'].showinfo("Undo", "Restored to the previous δ_Exp state.")

#=======================================

def _show_delta_preview_and_apply(state, mapped_rows, source_label, plot_cartesian_graph_fn, kind: str):
    """
    mapped_rows: 리스트[ dict( file_id, file_atom, file_delta_raw, mapped_id, parsed_delta, action ) ]
      - action: 'set' | 'clear' | 'skip'
    부분 적용(체크박스) + 되돌리기 버퍼 저장 후 적용
    kind: 'obs' or 'dia'
    """
    if not kind:
        raise ValueError("kind is required")
    target_key, undo_key, _ = _delta_kind_config(kind)
    state.setdefault(target_key, {})

    top = Toplevel(state['root'])
    top.title(f"{kind} Import Preview")
    top.geometry("860x520")

    ttk.Label(top, text=f"Source: {source_label}").pack(anchor="w", padx=8, pady=(8, 2))

    # 적용 체크 플래그 (기본: set/clear만 True)
    apply_flags = [ (r['action'] in ('set','clear')) and (r.get('mapped_id') is not None) for r in mapped_rows ]

    info_var = tk.StringVar()

    def _refresh_info():
        tot = len(mapped_rows)
        sel = sum(apply_flags)
        set_n = sum(1 for f, r in zip(apply_flags, mapped_rows) if f and r['action'] == 'set')
        clr_n = sum(1 for f, r in zip(apply_flags, mapped_rows) if f and r['action'] == 'clear')
        skp_n = tot - sel
        info_var.set(f"Selected: {sel}  |  set: {set_n}, clear: {clr_n}  |  skipped: {skp_n}")

    ttk.Label(top, textvariable=info_var).pack(anchor="w", padx=8, pady=(0, 6))

    cols = ("apply", "file_id", "file_atom", "file_value", "mapped_id", "parsed_value", "action")
    tv = ttk.Treeview(top, columns=cols, show="headings", height=16)
    for c, w in zip(cols, (60, 100, 100, 160, 100, 120, 80)):
        tv.heading(c, text=c)
        tv.column(c, width=w, anchor="center")
    tv.pack(fill="both", expand=True, padx=8, pady=6)

    def _flag_symbol(b):
        return "✓" if b else ""

    for i, r in enumerate(mapped_rows):
        tv.insert("", "end", values=(
            _flag_symbol(apply_flags[i]),
            r.get('file_id', ''),
            r.get('file_atom', ''),
            r.get('file_delta_raw', ''),
            r.get('mapped_id', ''),
            "" if r.get('parsed_delta') is None else f"{r['parsed_delta']:.6g}",
            r.get('action', '')
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
            vals = list(tv.item(item, "values"))
            vals[0] = _flag_symbol(apply_flags[i])
            tv.item(item, values=vals)
        _refresh_info()

    # 클릭으로 첫 컬럼 토글
    def _on_click(event):
        x, y = event.x, event.y
        iid = tv.identify_row(y)
        col = tv.identify_column(x)
        if not iid or col != "#1":
            return
        idx = tv.index(iid)
        _toggle_row(idx)

    tv.bind("<Button-1>", _on_click)

    btns = ttk.Frame(top); btns.pack(fill="x", padx=8, pady=8)
    ttk.Button(btns, text="All on", command=lambda: _toggle_all(True)).pack(side="left")
    ttk.Button(btns, text="All off", command=lambda: _toggle_all(False)).pack(side="left", padx=4)
    ttk.Button(btns, text="Invert", command=lambda: _toggle_all(None)).pack(side="left")

    def _apply():
        # undo buffer per kind
        state[undo_key] = state.get(target_key, {}).copy()

        dv = state.setdefault(target_key, {})
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

        # recompute δ_para and refresh NMR layers
        try:
            from logic.nmr_delta_data_manager import recompute_delta_para, push_layers_to_nmr_if_open
            recompute_delta_para(state)
            push_layers_to_nmr_if_open(state)
        except Exception:
            pass

        state['update_graph']()
        if callable(plot_cartesian_graph_fn):
            plot_cartesian_graph_fn(state)

        state['messagebox'].showinfo(f"{kind} import", f"Applied set: {applied_set}, clear: {applied_clear}")
        top.destroy()

    def _undo_last():
        prev = state.get(undo_key)
        if prev is None:
            state['messagebox'].showwarning("Undo", "No undo buffer available.")
            return
        state[target_key] = prev

        try:
            from logic.nmr_delta_data_manager import recompute_delta_para, push_layers_to_nmr_if_open
            recompute_delta_para(state)
            push_layers_to_nmr_if_open(state)
        except Exception:
            pass

        state['update_graph']()
        if callable(plot_cartesian_graph_fn):
            plot_cartesian_graph_fn(state)
        state['messagebox'].showinfo("Undo", f"Restored previous {kind} state.")

    ttk.Button(btns, text="Apply", command=_apply).pack(side="right")
    ttk.Button(btns, text="Undo last", command=_undo_last).pack(side="right", padx=6)
    ttk.Button(btns, text="Close", command=top.destroy).pack(side="right", padx=6)

    _refresh_info()

def _build_preview_rows_from_df(state, df, kind: str = "obs"):
    """
    DF를 해석해 미리보기용 mapped_rows를 만든다.
    kind: 'obs' or 'dia'
    반환: 리스트[ dict(file_id,file_atom,file_delta_raw,mapped_id,parsed_delta,action) ]
    """
    def _norm(s): return str(s).strip().lower()
    cols = {_norm(c): c for c in df.columns}

    ref_col = next((cols[k] for k in cols if k in ("ref", "id", "ref_id", "rid")), None)
    atom_col = next((cols[k] for k in cols if k in ("atom", "element")), None)
    #delta_col = next((cols[k] for k in cols if k in ("δ_exp","delta_exp","delta","ppm","shift","d_exp","exp")), None)

    # kind-dependent delta column detection
    _, _, delta_names = _delta_kind_config(kind)
    delta_col = next((cols[_norm_name] for _norm_name in (_norm(x) for x in delta_names) if _norm_name in cols), None)

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

    else:
        # 인식 실패
        raise ValueError("Cannot find δ_Exp columns. Use 'Ref, δ_Exp' or 'Atom, δ_Exp'.")

    return mapped_rows

#========NMR spectrum delta data manager=========
def import_delta_file(state, kind: str, plot_cartesian_graph_fn):
    """
    Generic importer for δ_obs / δ_dia.
    Reuses existing preview/apply workflow.
    """
    target_key, _, _ = _delta_kind_config(kind)

    # Ensure dict exists
    state.setdefault(target_key, {})

    fd = state['filedialog'].askopenfilename(
        title=f"Load {kind} file",
        filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx;*.xls"), ("All files", "*.*")]
    )
    if not fd:
        return

    try:
        if fd.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(fd)
        else:
            try:
                df = pd.read_csv(fd, encoding="utf-8-sig")
            except Exception:
                df = pd.read_csv(fd, encoding="cp1252")
    except Exception as e:
        state['messagebox'].showerror(f"Import {kind}", f"Cannot read the file:\n{e}")
        return

    try:
        rows = _build_preview_rows_from_df(state, df, kind=kind)
    except Exception as e:
        state['messagebox'].showerror(f"Import {kind}", str(e))
        return

    if not rows:
        state['messagebox'].showwarning(f"Import {kind}", "No data to apply")
        return

    _show_delta_preview_and_apply(
        state, rows, os.path.basename(fd), plot_cartesian_graph_fn,
        kind=kind
    )

def import_delta_from_clipboard(state, kind: str, plot_cartesian_graph_fn):
    """
    Paste δ_obs / δ_dia from clipboard-like text box, with preview/apply.
    """
    top = Toplevel(state['root'])
    top.title(f"Paste {kind} from clipboard")
    top.geometry("720x700")

    ttk.Label(top, text=f"Paste your data and click “Preview”. Format: Ref, value (ppm)").pack(anchor="w", padx=8, pady=6)
    txt = tk.Text(top, height=16)
    txt.pack(fill="both", expand=True, padx=8, pady=6)

    # Header preset
    if kind.lower().startswith("dia"):
        txt.insert("1.0", "Ref, δ_dia\n")
    else:
        txt.insert("1.0", "Ref, δ_obs\n")
    txt.mark_set("insert", "2.0")

    def _guess_sep(first_line: str):
        if "," in first_line:  return ","
        if "\t" in first_line: return "\t"
        if ";" in first_line:  return ";"
        return r"\s+"

    def _preview():
        raw = txt.get("1.0", "end").strip()
        if not raw:
            state['messagebox'].showwarning(f"Paste {kind}", "No data found")
            return

        lines = [ln for ln in raw.splitlines() if ln.strip()]
        first = lines[0] if lines else ""
        sep = _guess_sep(first)

        from io import StringIO
        sio = StringIO(raw)

        try:
            df = pd.read_csv(sio, sep=sep, engine="python",
                             header=0, dtype=str, na_filter=False)
        except Exception:
            sio.seek(0)
            df = pd.read_csv(sio, sep=sep, engine="python",
                             header=None, dtype=str, na_filter=False)
            if df.shape[1] == 3:
                df.columns = ["Ref", "Atom", "value"]
            elif df.shape[1] == 2:
                df.columns = ["Ref", "value"]

        # If user used "value", rename to a kind-expected column
        if "value" in [str(c).strip().lower() for c in df.columns]:
            # Create a canonical column that our detector can find
            if kind.lower().startswith("dia"):
                df = df.rename(columns={"value": "δ_dia"})
            else:
                df = df.rename(columns={"value": "δ_obs"})

        try:
            rows = _build_preview_rows_from_df(state, df, kind=kind)
        except Exception as e:
            state['messagebox'].showerror(f"Paste {kind}", str(e))
            return

        if not rows:
            state['messagebox'].showwarning(f"Paste {kind}", "No data to apply")
            return

        _show_delta_preview_and_apply(
            state, rows, "clipboard", plot_cartesian_graph_fn,
            kind=kind
        )
        top.destroy()

    btns = ttk.Frame(top); btns.pack(fill="x", padx=8, pady=8)
    ttk.Button(btns, text="Preview", command=_preview).pack(side="right")
    ttk.Button(btns, text="Cancel", command=top.destroy).pack(side="right", padx=6)

def clear_delta_kind(state, kind: str, plot_cartesian_graph_fn):
    """Clear δ_obs or δ_dia."""
    target_key, _, _ = _delta_kind_config(kind)
    state.setdefault(target_key, {}).clear()

    # Rebuild dependent values and refresh
    try:
        from logic.nmr_delta_data_manager import recompute_delta_para, push_layers_to_nmr_if_open
        recompute_delta_para(state)
        push_layers_to_nmr_if_open(state)
    except Exception:
        pass

    state['update_graph']()
    plot_cartesian_graph_fn(state)
    state['messagebox'].showinfo(f"Clear {kind}", f"{kind} data initialized.")

#================================================

def clear_delta_exp(state, plot_cartesian_graph_fn):
    """모든 δ_Exp를 초기화(비우기)하고 테이블/플롯을 갱신."""
    state.get('delta_exp_values', {}).clear()
    # 테이블의 δ_Exp 컬럼을 비우려면 update_graph로 재빌드하는 게 가장 안전
    state['update_graph']()
    plot_cartesian_graph_fn(state)
    state['messagebox'].showinfo("δ_Exp", "δ_Exp data initialized.")

def update_molar_value(state, tensor):
    val = tensor * AVOGADRO_CONSTANT * 1e-32
    state['molar_value_label'].config(text=f"Δχ_mol_ax: {val:.2e} m³/mol")

#================================================
# residual thresholds related helper
def get_residual_color_thresholds(state):
    ok_thr = float(state.get("residual_thr_ok", 0.10))
    warn_thr = float(state.get("residual_thr_warn", 0.30))
    return ok_thr, warn_thr


def classify_residual_tag(state, residual):
    """
    Return a Treeview tag tuple for residual colouring.
    - ()      : colouring disabled
    - ("none",): no residual available
    - ("ok",) / ("warn",) / ("bad",)
    """
    enabled_var = state.get("residual_color_enabled_var")
    enabled = bool(enabled_var.get()) if enabled_var is not None else True

    if not enabled:
        return ()

    if residual is None:
        return ("none",)

    ok_thr, warn_thr = get_residual_color_thresholds(state)
    a = abs(float(residual))

    if a <= ok_thr:
        return ("ok",)
    elif a <= warn_thr:
        return ("warn",)
    return ("bad",)

#================================================

def update_table(state, polar_data, rotated_coords, tensor, delta_exp_values):
    tree = state['tree']
    # 모든 행 비우기
    for r in tree.get_children():
        tree.delete(r)

    # 현재 표시 중인 행의 Ref(ID) 목록
    ids = state.get('current_selected_ids', [])
    state['row_by_id'] = {}  # 클릭/하이라이트 용 역방향 매핑

    pcs_by_id = state.setdefault('pcs_by_id', {})
    pcs_by_id.clear()

    atom_by_id = state.setdefault('atom_by_id', {})
    atom_by_id.clear()

    # Ref-ID -> display label override (pseudo atoms)
    label_overrides = state.get("ref_label_overrides", {}) or {}

    for i, ((atom, r, theta), (dx, dy, dz)) in enumerate(zip(polar_data, rotated_coords)):
        ref_id = ids[i] if i < len(ids) else (i + 1)  # for safety
        geom_param = (3 * (np.cos(theta))**2 - 1) / (r**3) if r != 0 else 0.0
        geom_value = geom_param
        delta_pcs = (tensor * (geom_value * 1e4)) / (12 * np.pi)

        atom_by_id[ref_id] = str(atom)
        pcs_by_id[ref_id] = float(delta_pcs)

        # Display label override only for table Atom column
        atom_disp = label_overrides.get(ref_id, atom)

        # Read/Write with Ref ID
        delta_exp = delta_exp_values.get(ref_id, None)
        delta_exp_str = f"{delta_exp:g}" if delta_exp is not None else ""

        residual = None
        if delta_exp is not None:
            try:
                residual = float(delta_pcs) - float(delta_exp)
            except Exception:
                residual = None

        tags = classify_residual_tag(state, residual)

        item = tree.insert(
            "",
            state['tk'].END,
            values=(
                ref_id, atom_disp, f"{dx:.6f}", f"{dy:.6f}", f"{dz:.6f}",
                f"{geom_param:.6f}", f"{delta_pcs: .2f}", delta_exp_str
            ),
            tags=tags
        )
        state['row_by_id'][ref_id] = item

    # push delta_pcs to NMR spectrum window if it exists
    push_layers_to_nmr_if_open(state)

def on_delta_entry_change(state, event, delta_exp_values, plot_cartesian_graph_fn):
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
        delta_exp_values.pop(ref_id, None)
        plot_cartesian_graph_fn(state)
        return

    try:
        val = float(new_str)
        tree.set(item, col, new_str)
        delta_exp_values[ref_id] = val
        plot_cartesian_graph_fn(state)
    except ValueError:
        state['messagebox'].showerror(
            "Invalid input",
            "Please enter a valid number for δ_Exp or leave empty to clear."
        )

def _parse_delta_chi_ax_from_label(molar_value_label) -> float:
    # "Δχ_mol_ax: 1.23e-28 m³/mol" or "Δχ_mol_ax : 1.23e-28 m³/mol"
    txt = str(molar_value_label["text"])
    if ":" not in txt:
        raise ValueError(f"Cannot parse Δχ_ax from label: {txt!r}")
    return float(txt.split(":", 1)[1].strip().split()[0])

def calculate_tensor_components_ui(
    chi_mol_entry,
    molar_value_label,
    tensor_xx_label,
    tensor_yy_label,
    tensor_zz_label,
    messagebox,
    *,
    assume_traceless_when_empty: bool = True,
    quiet: bool = False,
    placeholder_text: str | None = None,
):
    """
    Normal mode (χ_mol provided):
      χ_xx = χ_mol - Δχ_ax/3
      χ_yy = χ_mol - Δχ_ax/3
      χ_zz = χ_mol + 2Δχ_ax/3

    Fallback (χ_mol empty):
      assume traceless (χ_iso = 0) + axial symmetry (Δχ_rh = 0)
      χ_zz = 2Δχ_ax/3
      χ_xx = χ_yy = -Δχ_ax/3
    """
    try:
        delta_chi_ax = _parse_delta_chi_ax_from_label(molar_value_label)

        s_iso = (chi_mol_entry.get() or "").strip()
        if placeholder_text and s_iso == placeholder_text:
            s_iso = ""  # placeholder는 빈칸 처리

        if s_iso == "":
            if not assume_traceless_when_empty:
                if not quiet:
                    messagebox.showerror("Input Error", "Please enter χ_mol (or enable traceless mode).")
                return

            chi_xx = chi_yy = -(1.0 / 3.0) * delta_chi_ax
            chi_zz = (2.0 / 3.0) * delta_chi_ax
        else:
            chi_mol = float(s_iso)  # 5e-09 OK
            chi_xx = chi_yy = chi_mol - delta_chi_ax / 3.0
            chi_zz = chi_mol + (2.0 / 3.0) * delta_chi_ax

        tensor_xx_label.config(text=f"χ_xx: {chi_xx:.2e} m³/mol")
        tensor_yy_label.config(text=f"χ_yy: {chi_yy:.2e} m³/mol")
        tensor_zz_label.config(text=f"χ_zz: {chi_zz:.2e} m³/mol")

        try:
            tensor_xx_label.update_idletasks()
            tensor_yy_label.update_idletasks()
            tensor_zz_label.update_idletasks()
        except Exception:
            pass

    except Exception as e:
        if not quiet:
            messagebox.showerror("Input Error", f"Please enter valid numerical values.\n\n{e}")

def calculate_tensor_components_ui_ax_rh(
    chi_mol_entry,
    molar_value_label,
    rh_dchi_entry,
    tensor_xx_label,
    tensor_yy_label,
    tensor_zz_label,
    messagebox,
    *,
    assume_traceless_when_empty: bool = True,
    quiet: bool = False,
    placeholder_text: str | None = None,
):
    """
    χ_iso = χ_mol
    Δχ_ax = 라벨(Δχ_mol_ax)에서 읽음
    Δχ_rh = rh_dchi_entry에서 읽음 (E-32 m^3 scale)

    Normal mode (χ_mol provided):
      χ_xx = χ_mol − Δχ_ax/3 + Δχ_rh/2
      χ_yy = χ_mol − Δχ_ax/3 − Δχ_rh/2
      χ_zz = χ_mol + 2Δχ_ax/3

    Fallback mode (χ_mol empty and assume_traceless_when_empty=True):
      assume traceless diagonal tensor (χ_iso = 0)
      χ_zz = 2Δχ_ax/3
      χ_xx + χ_yy = -χ_zz
      χ_xx - χ_yy = Δχ_rh
    """
    # PCS calculations use Δχ_ax and Δχ_rh together with the corresponding
    # geometry factors G_ax and G_rh derived from atomic coordinates.

    try:
        delta_chi_ax = _parse_delta_chi_ax_from_label(molar_value_label)

        # Δχ_rh (entry is in "E-32 m³" scale)
        s_rh = (rh_dchi_entry.get() or "").strip()
        try:
            dchi_rh = float(s_rh) if s_rh else 0.0
        except Exception:
            if not quiet:
                messagebox.showerror("Input Error", "Invalid Δχ_rh value.")
            return
        delta_chi_rh_mol = dchi_rh * AVOGADRO_CONSTANT * 1e-32

        # χ_mol input
        s_iso = (chi_mol_entry.get() or "").strip()
        if placeholder_text and s_iso == placeholder_text:
            s_iso = ""  # placeholder는 빈칸 처리

        # ---- Traceless mode (blank) ----
        if s_iso == "":
            if not assume_traceless_when_empty:
                if not quiet:
                    messagebox.showerror("Input Error", "Please enter χ_mol (or enable traceless mode).")
                return

            chi_zz = (2.0 / 3.0) * delta_chi_ax
            S = -chi_zz
            R = delta_chi_rh_mol
            chi_xx = 0.5 * (S + R)
            chi_yy = 0.5 * (S - R)

        # ---- Normal mode (χ_mol provided) ----
        else:
            try:
                chi_mol = float(s_iso)
            except Exception:
                if quiet:
                    return
                messagebox.showerror("Input Error", "Please enter a valid χ_mol value.")
                return

            chi_xx = chi_mol - delta_chi_ax / 3.0 + delta_chi_rh_mol / 2.0
            chi_yy = chi_mol - delta_chi_ax / 3.0 - delta_chi_rh_mol / 2.0
            chi_zz = chi_mol + (2.0 / 3.0) * delta_chi_ax

        tensor_xx_label.config(text=f"χ_xx: {chi_xx:.2e} m³/mol")
        tensor_yy_label.config(text=f"χ_yy: {chi_yy:.2e} m³/mol")
        tensor_zz_label.config(text=f"χ_zz: {chi_zz:.2e} m³/mol")

        try:
            tensor_xx_label.update_idletasks()
            tensor_yy_label.update_idletasks()
            tensor_zz_label.update_idletasks()
        except Exception:
            pass

    except Exception as e:
        if not quiet:
            messagebox.showerror("Input Error", f"Please enter valid numerical values.\n\n{e}")