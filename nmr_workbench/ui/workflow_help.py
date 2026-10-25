"""Shared user-facing workflow and file-purpose help; no scientific state."""
import tkinter as tk
import webbrowser
from tkinter import ttk, messagebox

from ui.dialog_layout import fit_dialog
from ui.documentation_help import DOCUMENTATION_URL
from ui.style import get_app_fonts


WORKFLOW = """## Prepared shifts — no spectrum required
1. Open Inspector → NMR Data and add, paste or import observed shifts.
2. In Shift Analysis, choose Observed / Para and Reference / Dia, then Compare.
3. In PCS Preparation, review the derivation and atom/group mapping.
4. Use Review & Apply to transfer the reviewed values to PCS Analyzer.

Use absolute observed shifts as input. δ_para is derived from the paired data;
it is not automatically a pure PCS. Read-only PCS-project data can be copied
into independent NMR data when needed.

## Single spectrum
1. Load the experiment and inspect its spectrum.
2. Preview phase, baseline and reference corrections; Apply after checking.
3. Pick peaks and review assignments in Inspector → NMR Data.
4. Save an NMR session to resume the committed work.

A phase pivot changes the phase ramp; Reference sets the chemical-shift scale.
Processing previews are not saved until applied.

## Variable-temperature series
1. Open VT Series → New and add experiments. Check temperatures in Kelvin.
2. Use Batch Processing to preview and apply corrections, then pick peaks.
3. Inspect the stacked spectra and review peak links in Tracks.
4. Review assignments, then inspect fits and residuals in VT Analysis.

Suggest proposes peak links; Accept confirms them. Tracking a resonance does
not assign it to an atom. Reprocessing can make existing links stale.
Batch checkboxes choose processing targets; series Include and analysis fit
exclusions remain separate controls.

## Save your work
Open VT Series → Manage VT Series… → Files → Save NMR session…
This also works for a single experiment or prepared shifts without a series.
Save the corresponding PCS project separately as .pcsp after a handoff."""

FILES = """## Resume work — NMR session (.nmrsession)
VT Series → Manage VT Series… → Files → Save NMR session… / Open NMR session…
Preserves committed spectra, processing history, peaks, assignments, series,
tracks and retained analysis evidence. Pending previews are not saved.
Reopen the session rather than reimporting FIDs to preserve data identities.

## Exchange shifts — observed-shift CSV
Inspector → NMR Data → Export / Import
Exchanges observed values and their conditions. Import previews changes first.
Linked spectrum values need their matching session or an explicitly detached copy.
This is a different format from the assignment-review CSV.

## Review assignments — assignment CSV
Export / Review import in the data-exchange window.
Edit assignment names and uncertainties; keep identity columns unchanged.
Reopen the original session before import, then review conflicts before Apply.
Blank assignment names do not delete existing names.

## Reuse series definitions (.nmrseries)
VT Series manager → Files → Export / Import series definitions…
Contains membership and track definitions for already loaded datasets.
It does not contain spectra and does not replace an NMR session.

## Share analysis — CSV, JSON and plots
CSV reports coordinates, fit inclusion, predictions and residuals.
JSON reopens historical results without recalculating or changing live fits.
Choose Keep in session before saving to retain imported analysis evidence.
PNG / PDF / SVG shares the figure. None of these replaces an NMR session."""

TERMS = """## Experiments and series
Dataset / experiment: one loaded spectrum and its own processing history, peaks and observables.
Series / member: an ordered collection of dataset references / one entry in that collection.
Include: use this member in the series workflow; excluded members remain loaded.
Acquisition order: original series order. Temperature order changes the view, not identity.
Temperature override: explicit effective temperature; original acquisition metadata is preserved.

## Spectrum processing
Apodization: multiply the FID by a window before FT. Exponential line broadening (LB)
is in Hz; larger LB reduces noise but broadens peaks. Gaussian width (GB) is in Hz.
Zero fill: pad the FID to more points for finer displayed sampling; adds no measured information.
FT: Fourier transform from time-domain FID to frequency-domain spectrum.
P0: constant phase rotation in degrees. P1: phase slope across the whole spectrum in degrees.
Pivot: origin of the phase ramp. Biggest peak uses the strongest magnitude per spectrum.
ACME: automatic phase estimator; convergence does not guarantee scientifically correct shape.
Baseline: correct background curvature/offset. Order controls polynomial flexibility;
λ controls smoothness in asymmetric least squares; p controls asymmetry.
Reference: align a known chemical-shift signal to its declared target ppm.
Peak height: signal amplitude. Prominence: contrast with surrounding background.
Height/prominence fraction: threshold as a fraction (0.01 = 1%) of the picker's signal scale.
Distance points: minimum picker separation in native spectral samples, not ppm.
Set common: store this operation's batch-wide settings; does not process spectra.
Set for selected: store this operation's settings for the selected experiment only.
Clear selected overrides: clear all operation overrides for that experiment, using common settings.
Processing preset: saved ordered operations/settings; Apply writes each dataset's own history.

## Peaks, assignments and fits
Peak: measured spectral feature. Track: reviewed correspondence between member peaks.
Observable: interpreted NMR quantity linked to an assignment; separate from the peak/track.
Structural target: atom/group mapping; not inferred merely from tracking a resonance.
Fit mask: analysis-only point inclusion, separate from series Include.
Residual: measured minus fitted value, in the selected coordinate's y units.
R²: variance explained by the fit; not proof of a physical model.
Uncertainty: measured/provided shift uncertainty; missing uncertainty is not invented."""


def _open_manual(window, page):
    url = DOCUMENTATION_URL + page
    try:
        if not webbrowser.open_new_tab(url):
            raise RuntimeError("No browser accepted the request.")
    except Exception as exc:
        messagebox.showerror("NMR documentation",
                             f"Could not open documentation.\n{url}\n\n{exc}",
                             parent=window)


def show_workflow_help(parent):
    window = tk.Toplevel(parent)
    window.title("NMR workflow & file guide")
    window.transient(parent)
    fonts = get_app_fonts(parent._root())
    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True, padx=10, pady=(10, 0))
    pages = (("Workflow", WORKFLOW, "nmr/spectrum-viewer/"),
             ("Terms & parameters", TERMS, "nmr/raw-processing/"),
             ("Save / export / import", FILES, "nmr/sessions/"))
    for label, content, _page in pages:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=label)
        text = tk.Text(frame, wrap="word", width=82, height=26,
                       padx=14, pady=12, font=fonts["ui"], spacing1=2, spacing3=3)
        text.tag_configure("heading", font=fonts["ui_bold"], spacing1=12, spacing3=6)
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)
        for line in content.splitlines():
            if line.startswith("## "):
                text.insert("end", line[3:] + "\n", "heading")
            else:
                text.insert("end", line + "\n")
        text.configure(state="disabled")
    from .actions import action, polish
    footer = ttk.Frame(window, padding=10)
    footer.pack(fill="x")
    action(footer, "Read full guide", "search",
           lambda: _open_manual(window, pages[notebook.index("current")][2]),
           description="Open the related online NMR documentation").pack(side="left")
    action(footer, "Close", "close", window.destroy).pack(side="right")
    polish(window)
    fit_dialog(window)
    return window
