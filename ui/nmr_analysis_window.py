# ui/nmr_analysis_window.py

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def _safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def build_analysis_rows(state: dict):
    """
    Rows: ref, atom, pcs, obs, dia, para, nonpcs
    nonpcs = para - pcs (available only when both exist)
    """
    pcs_by_id = state.get("pcs_by_id", {}) or {}
    atom_by_id = state.get("atom_by_id", {}) or {}

    obs_by_id = state.get("delta_obs_values", {}) or {}
    dia_by_id = state.get("delta_dia_values", {}) or {}
    para_by_id = state.get("delta_para_values", {}) or {}

    # 표시 순서: 현재 테이블(트리) 순서가 있으면 그걸 우선
    ordered_ids = []
    tree = state.get("tree", None)
    if tree is not None:
        try:
            for item in tree.get_children():
                vals = tree.item(item, "values")
                if vals:
                    ordered_ids.append(int(vals[0]))
        except Exception:
            ordered_ids = []

    if not ordered_ids:
        # fallback: union keys
        keys = set(pcs_by_id.keys()) | set(obs_by_id.keys()) | set(dia_by_id.keys()) | set(para_by_id.keys())
        ordered_ids = sorted(keys)

    rows = []
    for rid in ordered_ids:
        pcs = _safe_float(pcs_by_id.get(rid))
        obs = _safe_float(obs_by_id.get(rid))
        dia = _safe_float(dia_by_id.get(rid))
        para = _safe_float(para_by_id.get(rid))
        atom = str(atom_by_id.get(rid, ""))

        nonpcs = None
        if (para is not None) and (pcs is not None):
            nonpcs = para - pcs

        rows.append(dict(
            ref=int(rid),
            atom=atom,
            pcs=pcs,
            obs=obs,
            dia=dia,
            para=para,
            nonpcs=nonpcs
        ))
    return rows


class NMRAnalysisWindow(tk.Toplevel):
    def __init__(self, parent: tk.Misc, state: dict):
        super().__init__(parent)
        self.state = state
        self.title("NMR Analysis")
        self.geometry("1150x650")

        # ---- controls ----
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="nonPCS threshold (ppm):").pack(side="left")
        self.var_thr = tk.DoubleVar(value=1.0)
        ent = ttk.Entry(top, textvariable=self.var_thr, width=8)
        ent.pack(side="left", padx=(6, 10))
        ent.bind("<Return>", lambda _e: self.refresh())

        self.var_abs = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="use |nonPCS|", variable=self.var_abs, command=self.refresh).pack(side="left")

        ttk.Button(top, text="Refresh", command=self.refresh).pack(side="right")

        # ---- notebook ----
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Tab 1: table
        tab1 = ttk.Frame(nb)
        nb.add(tab1, text="Table")

        cols = ("ref", "atom", "δ_PCS", "δ_obs", "δ_dia", "δ_para", "δ_nonPCS")
        self.tv = ttk.Treeview(tab1, columns=cols, show="headings", height=18)
        for c, w, a in [
            ("ref", 70, "center"),
            ("atom", 90, "center"),
            ("δ_PCS", 110, "e"),
            ("δ_obs", 110, "e"),
            ("δ_dia", 110, "e"),
            ("δ_para", 110, "e"),
            ("δ_nonPCS", 120, "e"),
        ]:
            self.tv.heading(c, text=c)
            self.tv.column(c, width=w, anchor=a)
        self.tv.pack(fill="both", expand=True)

        # highlight tags
        self.tv.tag_configure("bad", background="#ffd6d6")
        self.tv.tag_configure("missing", foreground="#888888")

        # Tab 2: histogram
        tab2 = ttk.Frame(nb)
        nb.add(tab2, text="Histogram")

        fig_h = Figure(figsize=(10, 4), constrained_layout=True)
        self.ax_h = fig_h.add_subplot(111)
        self.canvas_h = FigureCanvasTkAgg(fig_h, master=tab2)
        self.canvas_h.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas_h, tab2).update()

        # Tab 3: bars (per peak / per ref)
        tab3 = ttk.Frame(nb)
        nb.add(tab3, text="Bar chart")

        # controls row inside tab3
        bar_top = ttk.Frame(tab3)
        bar_top.pack(fill="x", padx=6, pady=(6, 0))

        ttk.Label(bar_top, text="Top N:").pack(side="left")
        self.var_topn = tk.IntVar(value=40)
        entn = ttk.Entry(bar_top, textvariable=self.var_topn, width=6)
        entn.pack(side="left", padx=(6, 10))
        entn.bind("<Return>", lambda _e: self.refresh())

        self.var_sort_abs = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            bar_top, text="sort by |nonPCS|",
            variable=self.var_sort_abs, command=self.refresh
        ).pack(side="left")

        fig_b = Figure(figsize=(10, 4), constrained_layout=True)
        self.ax_b = fig_b.add_subplot(111)
        self.canvas_b = FigureCanvasTkAgg(fig_b, master=tab3)
        self.canvas_b.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas_b, tab3).update()

        self.refresh()

    def refresh(self):
        rows = build_analysis_rows(self.state)

        # ---- table ----
        for it in self.tv.get_children():
            self.tv.delete(it)

        try:
            thr = float(self.var_thr.get())
        except Exception:
            thr = 1.0

        use_abs = bool(self.var_abs.get())

        nonpcs_vals = []

        for r in rows:
            ref = r["ref"]
            atom = r["atom"]
            pcs = r["pcs"]
            obs = r["obs"]
            dia = r["dia"]
            para = r["para"]
            nonpcs = r["nonpcs"]

            def fmt(x):
                return "" if x is None else f"{x:.4g}"

            tags = []
            if nonpcs is None:
                tags.append("missing")
            else:
                test = abs(nonpcs) if use_abs else nonpcs
                if abs(test) >= thr:
                    tags.append("bad")
                nonpcs_vals.append(nonpcs)

            self.tv.insert("", "end", values=(
                ref, atom, fmt(pcs), fmt(obs), fmt(dia), fmt(para), fmt(nonpcs)
            ), tags=tuple(tags))


        # ---- histogram ----
        self.ax_h.clear()
        self.ax_h.set_xlabel("δ_nonPCS (ppm)", fontsize=8)
        self.ax_h.set_ylabel("count", fontsize=8)
        self.ax_h.tick_params(axis="both", labelsize=8)

        if len(nonpcs_vals) >= 1:
            arr = np.asarray(nonpcs_vals, dtype=float)
            self.ax_h.hist(arr, bins=min(30, max(5, int(np.sqrt(len(arr))))))  # sensible bins
            # threshold lines
            self.ax_h.axvline(+thr, linestyle="--")
            self.ax_h.axvline(-thr, linestyle="--")
        else:
            self.ax_h.text(0.5, 0.5, "No δ_nonPCS data (need PCS + δ_para).",
                           transform=self.ax_h.transAxes, ha="center", va="center", fontsize=10)
        self.canvas_h.draw_idle()

        # ---- per-peak bar plot ----
        self.ax_b.clear()
        self.ax_b.set_xlabel("ppm", fontsize=8)
        self.ax_b.set_ylabel("Ref (Atom)", fontsize=8)
        self.ax_b.tick_params(axis="both", labelsize=8)

        # collect eligible points: need PCS + δ_para (=> nonpcs defined)
        pts = []
        for r in rows:
            if r["pcs"] is None or r["para"] is None:
                continue
            nonpcs = r["para"] - r["pcs"]
            pts.append(dict(
                ref=r["ref"],
                atom=r["atom"],
                pcs=float(r["pcs"]),
                para=float(r["para"]),
                nonpcs=float(nonpcs),
            ))

        if not pts:
            self.ax_b.text(
                0.5, 0.5, "No per-peak data (need PCS + δ_para).",
                transform=self.ax_b.transAxes, ha="center", va="center", fontsize=10
            )
            self.canvas_b.draw_idle()
            return

        # sorting
        sort_abs = bool(getattr(self, "var_sort_abs", tk.BooleanVar(value=True)).get())
        if sort_abs:
            pts.sort(key=lambda d: abs(d["nonpcs"]), reverse=True)
        else:
            pts.sort(key=lambda d: d["nonpcs"], reverse=True)

        # Top N
        try:
            topn = int(getattr(self, "var_topn", tk.IntVar(value=40)).get())
        except Exception:
            topn = 40
        if topn > 0:
            pts = pts[:topn]

        labels = [f"{p['ref']} ({p['atom']})" if p["atom"] else str(p["ref"]) for p in pts]
        pcs_v = np.array([p["pcs"] for p in pts], dtype=float)
        para_v = np.array([p["para"] for p in pts], dtype=float)
        non_v = np.array([p["nonpcs"] for p in pts], dtype=float)

        # barh with small vertical offsets (3 bars per ref)
        y = np.arange(len(pts), dtype=float)
        h = 0.25

        self.ax_b.barh(y - h, pcs_v, height=h, label="PCS")
        self.ax_b.barh(y, para_v, height=h, label="δ_para")
        self.ax_b.barh(y + h, non_v, height=h, label="δ_nonPCS")

        # threshold highlight: draw vertical lines (use thr, and abs-mode toggle already exists in top controls)
        try:
            thr = float(self.var_thr.get())
        except Exception:
            thr = 1.0
        self.ax_b.axvline(+thr, linestyle="--")
        self.ax_b.axvline(-thr, linestyle="--")

        # y labels
        self.ax_b.set_yticks(y)
        self.ax_b.set_yticklabels(labels)
        self.ax_b.invert_yaxis()  # biggest |nonPCS| at top

        self.ax_b.legend(fontsize=8)
        self.canvas_b.draw_idle()