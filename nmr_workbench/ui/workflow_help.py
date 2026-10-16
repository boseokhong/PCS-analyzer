"""Shared user-facing workflow and file-purpose help; no scientific state."""
import tkinter as tk
from tkinter import ttk
from ui.style import get_app_fonts


WORKFLOW = """Prepared-data route (no spectrum required)
Open Inspector → NMR Data. Add observations, paste a table, or Import observed
shifts CSV/TSV. Assignment and Shift columns can be mapped before Preview/Apply.
Legacy δ_obs + δ_dia columns represent absolute paramagnetic/diamagnetic shifts;
δ_para is calculated downstream, not imported as an absolute shift or called PCS.
Open Shift Analysis, choose Observed / Para and Reference / Dia experiments,
then Compare. Role changes require confirmation. Continue with PCS Preparation
→ Review & Apply when the derivation and structural mapping are justified.
Use controls single-pair shift comparison/PCS preparation; VT keeps its own
member inclusion and fit masks. NMR Data also shows unassigned picked peaks:
double-click Assignment or press Enter to assign; Tab moves to the next row.
Picked shifts/intensities stay owned by the source peak, not editable copies.
Use Experiment and Show to filter, and More for details, removal and Chemical Map.
Shift Analysis Display expands the existing plot-layer options; PCS selected-row
details can be collapsed without changing their values.
Rows and Chemical Map share semantic selection; More previews grouped sites
and explicitly links reviewed mappings without automatically transferring PCS.
The Import menu also exposes the original PCS-project observed/diamagnetic
Excel/CSV/paste review. These project projections remain read-only and outside
NMR session ownership. Copy values creates independent session-owned data;
explicitly select the actual nucleus when copying legacy unspecified nuclei.

1. Load experiments and create a VT series
Use VT Series → New, then Import FIDs / Import folder group or Add loaded.
Check each temperature in Kelvin (25 °C = 298.15 K), labels and Include.

2. Correct spectra and pick peaks
Batch Processing starts from the current committed spectrum; untouched FIDs get
the default FT. Usually: phase → baseline → reference → peak picking.
Check experiments in the table to set batch targets; row selection browses spectra.
Included members are checked initially. Checking an excluded member permits this
batch only, without changing series inclusion. The All header selects/clears targets.
Preset loads saved processing settings; Save/Delete manage presets in the series.
For selected-experiment batches, Save stores a Workbench common preset without
member IDs/overrides. Save session preserves these presets; loading one keeps the
current checked targets and clears individual overrides. Series presets retain
their saved targets and overrides.
Current spectrum keeps existing processing. Original FID restarts from raw and
enables the visible FT step; review operation order before applying.
Preview changes nothing. Inspect Before / After / Overlay, then Apply explicitly.
Phase pivot is the origin of the phase ramp; it is not a chemical-shift reference.
Reference searches each member independently. Missing/multiple candidates need review.

3. Compare and track resonances
VT stacked shows included members; normalization/spacing affect display only.
In Tracks, link an existing peak in each member. Suggest proposes candidates;
Accept establishes the link. Missing = no observation; ambiguous = uncertain identity;
stale = source changed and needs review. Reprocessing/referencing/re-picking may stale links.

4. Assign when justified
Assignment names an interpreted resonance. A track only asserts correspondence:
it does not assign atoms. Edit the peak Assignment column or use Assignment review.
Propagation suggestions require confirmation; existing conflicts are not overwritten.

5. Analyze and save
VT Analysis fits confirmed, valid, included points. δ is observed shift in ppm,
T is Kelvin, 1/T is K⁻¹ and δT is ppm·K. Analysis fit exclusions affect only the fit.
Inspect residuals (measured minus fitted), R² and diagnostics before interpreting.
Save the NMR session to resume work, and export tables/plots/results to share."""

FILES = """Observed-shift table (.csv) — NMR Data → Export / Import
All dataset-owned observations, including independent input values and excluded
rows. Includes observed quantity, experiment/observation/peak IDs, conditions,
source revisions and provenance. Import previews first; exact loaded-ID updates
must be enabled explicitly. Missing/stale spectrum links require the original
session or an explicitly detached copy retaining original source evidence.
This table is distinct from the existing assignment-review CSV below.

NMR session (.nmrsession) — Save session / Open session
Resume committed spectra, processing history, peaks, assignments, series, tracks
and saved analysis evidence with stable IDs. Previews are not saved.

Series definitions (.nmrseries) — Save / Load definitions
Membership and series/track definitions only. Load against existing datasets;
this is not a substitute for saving the NMR session.

Assignment table (.csv) — Export / Review import
Peak-linked observables with dataset/peak/observable IDs, shift, revision, nucleus,
assignment and existing uncertainty. Edit assignment/uncertainty, keep identity
columns. Reopen the original session first; reimporting vendor FIDs creates new IDs.
Import reviews conflicts and blocked rows before Apply. Blank names do not delete.
No nearest-ppm matching, track propagation or structural assignment is performed.

Analysis table (.csv) — Export table
Measured/transformed coordinates, fit inclusion, predictions, residuals and IDs.
A reporting table; it cannot reconstruct spectra or tracking identities.

Analysis results (.json) — Export / Import results
Historical source snapshots, model/configuration, fit results and diagnostics.
Import inspects the saved result without recalculating or changing live fits.
Retain it in the matching series, then Save session. Changed sources are reported.

Plot (.png / .pdf / .svg) — Export plot
The current analysis figure for sharing; not a recoverable scientific session."""

TERMS = """Dataset / experiment: one loaded spectrum and its own processing history, peaks and observables.
Series / member: an ordered collection of dataset references / one entry in that collection.
Include: use this member in the series workflow; excluded members remain loaded.
Acquisition order: original series order. Temperature order changes the view, not identity.
Temperature override: explicit effective temperature; original acquisition metadata is preserved.

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

Peak: measured spectral feature. Track: reviewed correspondence between member peaks.
Observable: interpreted NMR quantity linked to an assignment; separate from the peak/track.
Structural target: atom/group mapping; not inferred merely from tracking a resonance.
Fit mask: analysis-only point inclusion, separate from series Include.
Residual: measured minus fitted value, in the selected coordinate's y units.
R²: variance explained by the fit; not proof of a physical model.
Uncertainty: measured/provided shift uncertainty; missing uncertainty is not invented."""


def show_workflow_help(parent):
    window = tk.Toplevel(parent)
    window.title("NMR workflow & file guide")
    window.transient(parent)
    window.geometry("790x620")
    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)
    for label, content in (("Workflow", WORKFLOW), ("Terms & parameters", TERMS), ("Save / export / import", FILES)):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=label)
        text = tk.Text(frame, wrap="word", padx=12, pady=12, font=get_app_fonts(parent._root())["ui"])
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)
        text.insert("1.0", content)
        text.configure(state="disabled")
    from .actions import polish
    polish(window)
    return window
