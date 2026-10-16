# Nd(*i*Pr₂BA)₃ documentation example

This Nd(*i*Pr₂BA)₃ example connects a molecular structure with experimental
paramagnetic shifts and an axial PCS analysis. It is used in the PCS Analyzer
manual and for reproducible interface screenshots.

| File | Contents |
|---|---|
| `Nd_iPr2BA3.xyz` | Pre-aligned, 103-atom molecular structure; Nd is Ref 1. |
| `Nd_iPr2BA3_paramagnetic_shifts.csv` | Experimental paramagnetic shifts in ppm, assigned by structure Ref ID and used as `δExp` in this example. |
| `Nd_iPr2BA3_axial.pcsp` | Canonical project with embedded coordinates, experimental assignments, methyl symmetry averaging, and phenyl rotational averaging. |
| `Nd_iPr2BA3_observed_shifts.csv` | Observed Nd sample shifts, δ_obs in ppm, keyed by Ref ID. |
| `La_iPr2BA3_observed_shifts.csv` | Diamagnetic La reference shifts, δ_dia in ppm, keyed by the corresponding Ref IDs. |

Open the `.pcsp` with **File → Open PCS Project...**. It uses **Axial**
prediction and `Δχax = −3.90816 × 10⁻³² m³`; the rhombic term is disabled.
Three phenyl groups use uniform full rotations with 24 samples per group.
Original/reference averaging overlays are off, and all elements/ligands are visible.

The project preserves the tensor value, not a historical lasso subset or
rhombicity-check result. On reopening, the G_ax diagnostic uses all assigned
visible points, including iPr data. Its regression estimate therefore need
not equal the stored tensor. Possible Fermi-contact and conformational
contributions to iPr shifts should be considered before using all points for
a scientific fit; this screenshot example does not resolve those contributions.

See [Quick Start](../../../docs/quick-start.md) for the corresponding workspace workflow.

The observed-shift pair supports a separate Shift Analysis / PCS Preparation
example. Blank cells remain unassigned; numeric zeros remain numeric data.
The capture loads the supplied lists transiently through existing project
shift layers, without saving the `.pcsp`. Direct subtraction is an illustrative
derivation preview, not a claim that every difference is pure PCS or ready for
fitting. Inspect assignments, targets, and possible contact contributions first.
