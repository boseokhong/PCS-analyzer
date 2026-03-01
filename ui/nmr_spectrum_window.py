# ui/nmr_spectrum_window.py

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
        self.geometry("1100x400")

        # --------- state ---------
        self._shifts = None
        self._intensities = None

        self._labels = None                                         #atom label
        self.var_show_labels = tk.BooleanVar(value=True)            #atom label

        self.var_show_sticks = tk.BooleanVar(value=True)
        self.var_show_env = tk.BooleanVar(value=False)
        self.var_normalize = tk.BooleanVar(value=False)
        self.var_lineshape = tk.StringVar(value="lorentzian")
        self.var_auto_range = tk.BooleanVar(value=True)

        # --------- matplotlib ---------
        fig = Figure(figsize=(12, 2.5), constrained_layout=True)
        self.ax = fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()

        # --------- controls ---------
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", padx=5, pady=5)

        ttk.Checkbutton(ctrl, text="Stick",
                        variable=self.var_show_sticks,
                        command=self._refresh).pack(side="left")

        ttk.Checkbutton(ctrl, text="Labels",
                        variable=self.var_show_labels,
                        command=self._refresh).pack(side="left")

        ttk.Checkbutton(ctrl, text="Peak",
                        variable=self.var_show_env,
                        command=self._refresh).pack(side="left")

        ttk.Checkbutton(ctrl, text="Normalize",
                        variable=self.var_normalize,
                        command=self._refresh).pack(side="left")

        ttk.Label(ctrl, text="Lineshape:").pack(side="left", padx=(15, 2))

        ttk.Combobox(ctrl,
                     textvariable=self.var_lineshape,
                     values=["lorentzian", "gaussian"],
                     width=12,
                     state="readonly").pack(side="left")

        ttk.Label(ctrl, text="FWHM:").pack(side="left", padx=(15, 2))

        self.scale_fwhm = tk.Scale(ctrl,
                                   from_=0.01, to=5.0,
                                   resolution=0.01,
                                   orient="horizontal",
                                   length=200,
                                   command=lambda e: self._refresh())
        self.scale_fwhm.set(0.05)
        self.scale_fwhm.pack(side="left")

        ttk.Checkbutton(ctrl, text="Auto range",
                        variable=self.var_auto_range,
                        command=self._refresh).pack(side="left", padx=10)

    # =========================================================

    def set_data(self, shifts_ppm: np.ndarray,
                 intensities: np.ndarray | None = None,
                 labels: list[str] | None = None):
        self._shifts = np.asarray(shifts_ppm, dtype=float)

        if intensities is None:
            self._intensities = np.ones_like(self._shifts)
        else:
            self._intensities = np.asarray(intensities, dtype=float)

        self._labels = labels
        self._refresh()

    # =========================================================

    def _refresh(self):

        if self._shifts is None:
            return

        shifts = self._shifts.copy()
        intensities = self._intensities.copy()

        if self.var_normalize.get():
            max_val = np.max(intensities)
            if max_val > 0:
                intensities /= max_val

        # ppm range
        if self.var_auto_range.get():
            lo = np.min(shifts) - 5
            hi = np.max(shifts) + 5
            x_range = (lo, hi)
        else:
            x_range = (-100, 100)  # 기본 수동 범위 (원하면 바꿀 수 있음)

        payload = make_payload(
            shifts,
            intensities,
            show_envelope=self.var_show_env.get(),
            fwhm=self.scale_fwhm.get(),
            kind=self.var_lineshape.get(),
            x_range=x_range,
        )

        self._draw(payload)

    # =========================================================

    def _draw(self, payload: dict):

        sticks = payload["sticks"]
        env = payload["envelope"]
        meta = payload["meta"]

        self.ax.clear()

        self.ax.set_xlabel("ppm")
        self.ax.set_ylabel("a.u.")
        self.ax.set_yticks([])

        # ---- sticks ----
        if self.var_show_sticks.get():
            self.ax.vlines(sticks["x"], 0, sticks["h"], linewidth=1.0)

        # ---- labels ----
        if self.var_show_labels.get() and getattr(self, "_labels", None):
            for i, (x0, h0, lab) in enumerate(zip(sticks["x"], sticks["h"], self._labels)):
                if not lab:
                    continue
                dy = 0.02 if (i % 2 == 0) else 0.08  # stagger
                self.ax.text(
                    float(x0),
                    float(h0) + dy,
                    lab,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    clip_on=True,
                )

        # ---- Peak ----
        if env is not None:
            self.ax.plot(env["x"], env["y"])

        # ---- NMR convention (left = higher ppm) ----
        lo, hi = meta["x_range"]
        self.ax.set_xlim(hi, lo)

        self.ax.set_ylim(bottom=0.0)

        self.canvas.draw_idle()