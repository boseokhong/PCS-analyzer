# ui/nmr_spectrum_window.py
'''
logic/nmr_spectrum.py
'''

from __future__ import annotations
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from logic.nmr_spectrum import make_payload


class NMRSpectrumWindow(tk.Toplevel):
    def __init__(self, parent: tk.Misc, title: str = "Simulated NMR Spectrum"):
        super().__init__(parent)
        self.title(title)
        self.geometry("1150x450")

        # --------- state ---------
        self._shifts: np.ndarray | None = None
        self._intensities: np.ndarray | None = None
        self._labels: list[str] | None = None
        self._ref_ids: list[int] | None = None

        self._pick_callback = None  # callback(ref_ids: list[int]) -> None

        self._stick_x: np.ndarray | None = None
        self._stick_h: np.ndarray | None = None

        self._pinned: set[int] = set()            # pinned stick indices
        self._pinned_clusters: set[int] = set()   # pinned cluster indices
        self._highlight_ref: int | None = None

        self._clusters = None  # list[dict]: {"center":float,"members":[...],"refs":[...],"n":int}
        self._last_tol: float | None = None  # last grouping tolerance used for clustering

        # --------- controls ----------
        self.var_show_sticks = tk.BooleanVar(value=True)
        self.var_show_env = tk.BooleanVar(value=False)
        self.var_show_labels = tk.BooleanVar(value=False)  # default OFF
        self.var_normalize = tk.BooleanVar(value=False)
        self.var_lineshape = tk.StringVar(value="lorentzian")

        self.var_cluster = tk.BooleanVar(value=False)
        self.var_binwidth = tk.DoubleVar(value=0.10)

        self.var_auto_range = tk.BooleanVar(value=True)

        # --------- matplotlib ---------
        fig = Figure(figsize=(12, 2.5), constrained_layout=True)
        self.ax = fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()

        # Hover tooltip
        self._hover_annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=7,
            bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="0.3"),
        )
        self._hover_annot.set_visible(False)

        # Matplotlib events
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("figure_leave_event", self._on_mouse_leave)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_click)

        # --------- controls UI ---------
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", padx=6, pady=6)

        ttk.Checkbutton(ctrl, text="Stick", variable=self.var_show_sticks, command=self._refresh).pack(side="left")
        ttk.Checkbutton(ctrl, text="Labels", variable=self.var_show_labels, command=self._refresh).pack(side="left", padx=(6, 0))
        ttk.Checkbutton(ctrl, text="Normalize", variable=self.var_normalize, command=self._refresh).pack(side="left", padx=(6, 0))
        ttk.Checkbutton(ctrl, text="Auto range", variable=self.var_auto_range, command=self._refresh).pack(side="left", padx=10)

        ttk.Separator(ctrl, orient="vertical").pack(side="left", fill="y", padx=4)

        ttk.Checkbutton(ctrl, text="Broadened", variable=self.var_show_env, command=self._refresh).pack(side="left", padx=(6, 0))
        ttk.Label(ctrl, text="Lineshape:").pack(side="left", padx=(14, 2))
        ttk.Combobox(
            ctrl,
            textvariable=self.var_lineshape,
            values=["lorentzian", "gaussian"],
            width=12,
            state="readonly",
        ).pack(side="left")

        ttk.Label(ctrl, text="FWHM:").pack(side="left", padx=(14, 2))
        self.scale_fwhm = tk.Scale(
            ctrl,
            from_=0.01, to=5.0,
            resolution=0.01,
            orient="horizontal",
            length=100,
            command=lambda _e: self._refresh()
        )
        self.scale_fwhm.set(0.05)
        self.scale_fwhm.pack(side="left")

        ttk.Separator(ctrl, orient="vertical").pack(side="left", fill="y", padx=4)

        def _on_toggle_group():
            self._clusters = None
            self._pinned_clusters.clear()
            self._last_tol = None
            self._refresh()
        ttk.Checkbutton(ctrl, text="Group peaks", variable=self.var_cluster, command=_on_toggle_group).pack(side="left")
        ttk.Label(ctrl, text="Grouping tolerance (ppm):").pack(side="left", padx=(8, 2))
        bin_entry = ttk.Entry(ctrl, textvariable=self.var_binwidth, width=6)
        bin_entry.pack(side="left")
        def _on_tol_commit(_e=None):
            self._clusters = None  # invalidate cluster cache
            self._pinned_clusters.clear()
            self._last_tol = None
            self._refresh()
        bin_entry.bind("<Return>", _on_tol_commit)
        bin_entry.bind("<FocusOut>", _on_tol_commit)

        ttk.Button(ctrl, text="Clear", command=self._clear_pins).pack(side="right")

    # ---------------- public API ----------------

    def set_pick_callback(self, fn):
        """Set callback that receives picked ref_ids."""
        self._pick_callback = fn

    def set_data(
        self,
        shifts_ppm: np.ndarray,
        intensities: np.ndarray | None = None,
        labels: list[str] | None = None,
        ref_ids: list[int] | None = None,
    ):
        self._shifts = np.asarray(shifts_ppm, dtype=float)
        if intensities is None:
            self._intensities = np.ones_like(self._shifts)
        else:
            self._intensities = np.asarray(intensities, dtype=float)

        self._labels = labels
        self._ref_ids = ref_ids

        # reset caches
        self._clusters = None
        self._pinned.clear()
        self._pinned_clusters.clear()
        self._highlight_ref = None

        self._refresh()

    def highlight_ref(self, ref_id: int):
        """Highlight a given ref_id in the spectrum (table -> spectrum)."""
        self._highlight_ref = int(ref_id)
        self._refresh()

    # ---------------- internal helpers ----------------

    def _clear_pins(self):
        """Clear pinned labels/clusters and also clear highlight."""
        self._pinned.clear()
        self._pinned_clusters.clear()

        self._highlight_ref = None

        if hasattr(self, "_hover_annot") and self._hover_annot is not None:
            self._hover_annot.set_visible(False)

        self._refresh()

    def _refresh(self):
        if self._shifts is None or self._shifts.size == 0:
            return

        shifts = self._shifts.copy()
        intensities = (self._intensities.copy() if self._intensities is not None else np.ones_like(shifts))

        if self.var_normalize.get():
            m = float(np.max(intensities)) if intensities.size else 1.0
            if m > 0:
                intensities = intensities / m

        if self.var_auto_range.get():
            lo = float(np.min(shifts)) - 7.0
            hi = float(np.max(shifts)) + 7.0
            x_range = (lo, hi)
        else:
            # manual fallback range
            x_range = (-100.0, 100.0)

        # Invalidate clusters if tolerance changed (even without explicit UI event)
        try:
            tol = float(self.var_binwidth.get())
        except Exception:
            tol = 0.10  # fallback
        if self._last_tol is None or abs(tol - self._last_tol) > 1e-12:
            self._clusters = None
            self._pinned_clusters.clear()
            self._last_tol = tol

        payload = make_payload(
            shifts,
            intensities,
            show_envelope=self.var_show_env.get(),
            fwhm=float(self.scale_fwhm.get()),
            kind=str(self.var_lineshape.get()),
            x_range=x_range,
        )
        self._draw(payload)

    def _build_clusters(self, shifts: np.ndarray, tol: float):
        """Group peaks by distance tolerance in ppm"""
        tol = max(float(tol), 1e-6)

        order = np.argsort(shifts)[::-1]  # descending (NMR convention)
        clusters = []

        cur = [int(order[0])]
        for ii in order[1:]:
            ii = int(ii)
            if abs(float(shifts[ii]) - float(shifts[cur[-1]])) <= tol:
                cur.append(ii)
            else:
                clusters.append(cur)
                cur = [ii]
        clusters.append(cur)

        out = []
        for members in clusters:
            center = float(np.mean(shifts[members]))
            refs = []
            if self._ref_ids:
                for j in members:
                    if j < len(self._ref_ids):
                        refs.append(int(self._ref_ids[j]))
            out.append({"center": center, "members": members, "refs": refs, "n": len(members)})

        return out

    def _draw(self, payload: dict):
        sticks = payload["sticks"]
        env = payload["envelope"]
        meta = payload["meta"]

        self.ax.clear()

        # Re-create hover annotation because ax.clear() removes artists
        self._hover_annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=7,
            bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="0.3"),
        )
        self._hover_annot.set_visible(False)

        self.ax.set_xlabel("ppm", fontsize=8)
        self.ax.set_ylabel("a.u.", fontsize=8)
        self.ax.set_yticks([])

        self._stick_x = np.asarray(sticks["x"], dtype=float)
        self._stick_h = np.asarray(sticks["h"], dtype=float)

        # y-top reference for label placement (sticks + broadened spectrum)
        ytop = float(np.max(self._stick_h)) if self._stick_h.size else 1.0
        if env is not None and "y" in env and len(env["y"]) > 0:
            ytop = max(ytop, float(np.max(env["y"])))

        # cluster (group) mode
        if self.var_cluster.get():
            if self._clusters is None:
                self._clusters = self._build_clusters(self._stick_x, float(self.var_binwidth.get()))

            centers = np.array([c["center"] for c in self._clusters], dtype=float)
            counts = np.array([c["n"] for c in self._clusters], dtype=float)

            if counts.size:
                counts = counts / np.max(counts)

            # Positive and negative regions
            pos = centers >= 0.0
            neg = ~pos

            # + ppm → red
            if np.any(pos):
                self.ax.vlines(
                    centers[pos], 0.0, counts[pos], linewidth=2.0, color="red"
                )

            # - ppm → blue
            if np.any(neg):
                self.ax.vlines(
                    centers[neg], 0.0, counts[neg], linewidth=2.0, color="blue"
                )

            # Put n-label above each cluster stick
            if self.var_show_labels.get():
                for k, c in enumerate(self._clusters):
                    if c["n"] <= 1:
                        continue
                    y = float(counts[k]) + 0.03
                    self.ax.text(c["center"], y, f"n={c['n']}",
                                 ha="center", va="bottom", fontsize=7, clip_on=True)

            for k in sorted(self._pinned_clusters):
                if 0 <= k < len(self._clusters):
                    c = self._clusters[k]
                    self.ax.text(c["center"], 1.10, self._cluster_label(k),
                                 ha="center", va="top", fontsize=7, clip_on=True)

            # Fix y-limits so texts don't hit the border
            self.ax.set_ylim(0.0, 1.15)

        else:
            # stick mode
            if self.var_show_sticks.get():
                x = self._stick_x
                h = self._stick_h
                pos = x >= 0.0
                neg = ~pos
                # + : red, - : blue
                if np.any(pos):
                    self.ax.vlines(x[pos], 0.0, h[pos], linewidth=1.0, color="red")
                if np.any(neg):
                    self.ax.vlines(x[neg], 0.0, h[neg], linewidth=1.0, color="blue")

            # broadened mode
            if env is not None:
                self.ax.plot(env["x"], env["y"], color="#800000", linewidth=1.2)

            # Highlight ref as thick stick
            hi = self._index_of_ref(self._highlight_ref)
            if hi is not None:
                self.ax.vlines(float(self._stick_x[hi]), 0.0, float(self._stick_h[hi]), linewidth=4.0, color="orange")

            # Labels: avoid duplicates between pinned and highlighted
            if self.var_show_labels.get():
                idxs = set(self._pinned)
                if hi is not None:
                    idxs.add(hi)
                for idx in sorted(idxs):
                    self._draw_one_label(idx, force=(hi is not None and idx == hi), ytop=ytop)

            # Auto y-limit for non-cluster mode (leave headroom for labels)
            self.ax.set_ylim(0.0, ytop * 1.25)

        # NMR convention: left = higher ppm
        lo, hi2 = meta["x_range"]
        self.ax.set_xlim(hi2, lo)
        self.ax.tick_params(axis="both", labelsize=8)

        self.canvas.draw_idle()

    def _index_of_ref(self, ref_id):
        if ref_id is None or not self._ref_ids:
            return None
        try:
            return self._ref_ids.index(int(ref_id))
        except Exception:
            return None

    def _draw_one_label(self, idx: int, force: bool = False, ytop: float = 1.0):
        if self._stick_x is None or self._stick_h is None:
            return
        if idx < 0 or idx >= len(self._stick_x):
            return

        if self._labels and idx < len(self._labels):
            lab = self._labels[idx]
        elif self._ref_ids and idx < len(self._ref_ids):
            lab = str(self._ref_ids[idx])
        else:
            lab = str(idx)

        h = float(self._stick_h[idx])

        base_y = 0.55 * h
        floor_y = 0.12 * float(ytop)
        base_y = max(base_y, floor_y)

        # Stagger levels to reduce overlaps
        levels = 4
        level = (idx % levels)
        bump = 0.06 * float(ytop) * (level / (levels - 1))  # scale bump with ytop

        if force:
            bump += 0.06 * float(ytop)

        y = base_y + bump

        self.ax.text(
            float(self._stick_x[idx]),
            y,
            lab,
            rotation=0,
            ha="left",
            va="center",
            fontsize=7 if force else 6,
            fontweight="bold" if force else "normal",
            clip_on=True,
        )

    def _cluster_label(self, k: int) -> str:
        c = self._clusters[k]
        refs = c["refs"]
        header = f"{c['center']:.2f} ppm (n={c['n']})"
        if not refs:
            return header
        # Format refs with max 3 per line, cap total lines
        per_line = 3
        max_lines = 4  # prevent huge tooltip
        chunks = [refs[i:i + per_line] for i in range(0, len(refs), per_line)]
        lines = []
        for li, chunk in enumerate(chunks[:max_lines]):
            s = ", ".join(map(str, chunk))
            prefix = "Refs: " if li == 0 else "      "
            lines.append(prefix + s)
        if len(chunks) > max_lines:
            lines.append("      ...")
        return header + "\n" + "\n".join(lines)

    # ---------- hover / click ----------
    def _on_mouse_leave(self, _event):
        self._hover_annot.set_visible(False)
        self.canvas.draw_idle()

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        x = float(event.xdata)

        if self.var_cluster.get():
            if not self._clusters:
                return
            centers = np.array([c["center"] for c in self._clusters], dtype=float)
            k = int(np.argmin(np.abs(centers - x)))
            if abs(float(centers[k]) - x) > max(0.15, 1.5 * float(self.var_binwidth.get())):
                if self._hover_annot.get_visible():
                    self._hover_annot.set_visible(False)
                    self.canvas.draw_idle()
                return

            c = self._clusters[k]
            self._hover_annot.xy = (float(c["center"]), 0.5)
            self._hover_annot.set_text(self._cluster_label(k))
            self._hover_annot.set_visible(True)
            self.canvas.draw_idle()
            return

        if self._stick_x is None or self._stick_x.size == 0:
            return

        idx = int(np.argmin(np.abs(self._stick_x - x)))
        if abs(float(self._stick_x[idx]) - x) > 0.15:
            if self._hover_annot.get_visible():
                self._hover_annot.set_visible(False)
                self.canvas.draw_idle()
            return

        lab = ""
        if self._labels and idx < len(self._labels):
            lab = self._labels[idx]
        elif self._ref_ids and idx < len(self._ref_ids):
            lab = str(self._ref_ids[idx])
        else:
            lab = str(idx)

        y = float(self._stick_h[idx]) if self._stick_h is not None else 1.0
        self._hover_annot.xy = (float(self._stick_x[idx]), y)
        self._hover_annot.set_text(lab)
        self._hover_annot.set_visible(True)
        self.canvas.draw_idle()

    def _on_mouse_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        x = float(event.xdata)

        if self.var_cluster.get():
            if not self._clusters:
                return
            centers = np.array([c["center"] for c in self._clusters], dtype=float)
            k = int(np.argmin(np.abs(centers - x)))
            if abs(float(centers[k]) - x) > max(0.15, 1.5 * float(self.var_binwidth.get())):
                return

            if k in self._pinned_clusters:
                self._pinned_clusters.remove(k)
            else:
                self._pinned_clusters.add(k)

            if callable(self._pick_callback):
                refs = self._clusters[k]["refs"]
                if refs:
                    self._pick_callback(refs)

            self._refresh()
            return

        if self._stick_x is None or self._stick_x.size == 0:
            return

        idx = int(np.argmin(np.abs(self._stick_x - x)))
        if abs(float(self._stick_x[idx]) - x) > 0.15:
            return

        if idx in self._pinned:
            self._pinned.remove(idx)
        else:
            self._pinned.add(idx)

        if callable(self._pick_callback) and self._ref_ids and idx < len(self._ref_ids):
            self._pick_callback([int(self._ref_ids[idx])])

        self._refresh()