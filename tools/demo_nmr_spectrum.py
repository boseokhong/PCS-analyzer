# tools/demo_nmr_spectrum.py

import numpy as np
import tkinter as tk

from ui.nmr_spectrum_window import NMRSpectrumWindow

def main():
    root = tk.Tk()
    root.withdraw()

    win = NMRSpectrumWindow(root)

    shifts = np.array([12.1, 9.8, -7.25, -3.42, 1.15])
    intensities = np.array([1, 1, 1, 1, 1], dtype=float)

    win.set_data(shifts, intensities)

    # try envelope
    # win.set_data(shifts, intensities, show_envelope=True, fwhm=0.15, kind="lorentzian")

    root.mainloop()

if __name__ == "__main__":
    main()