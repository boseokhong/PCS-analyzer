# Glossary

These terms describe PCS Analyzer's controls and results. For numerical
definitions and units, see [Mathematical Conventions](mathematical-conventions.md)
and [Units](units.md).

| Term | Meaning |
| --- | --- |
| PCS | Pseudocontact shift: the contribution modeled from geometry and magnetic susceptibility anisotropy. It is not every paramagnetic shift contribution. |
| δ_obs / δ_dia | Observed paramagnetic-sample shift and its diamagnetic reference shift, in ppm. |
| δ_para | δ_obs − δ_dia. It can include contact and other contributions in addition to PCS. |
| δ_Exp | Experimental PCS supplied as a fitting/diagnostic target. Treating δ_para as δ_Exp requires a scientific assumption or correction. |
| δ_PCS | Calculated PCS under the active structure, tensor, orientation and averaging model. |
| δ_nonPCS | δ_para minus calculated PCS: a diagnostic difference, not a unique physical separation of contact shift. |
| Ref / Ref ID | One-based atom identifier assigned in structure-file order. An element label alone is not a unique assignment. |
| PCS observation / target | An experimental value and the structural atom or group to which it is assigned. A grouped target contributes one observation, not one per member. |
| NMR site | A structural grouping used for assignment/averaging. A grouping or suggested match does not establish experimental equivalence. |
| Assigned centre | The metal/PCS origin chosen for the working structure. |
| Refined centre | A centre estimated by fitting. Mode C Preview can use it temporarily without replacing the assigned centre. |
| G_ax / G_rh | Axial and rhombic geometrical factors in Å⁻³, calculated in the tensor frame. |
| Δχ_ax / Δχ_rh | Axial and rhombic susceptibility anisotropies, entered as numerical multiples of 10⁻³² m³ per molecule. |
| Structure XYZ / Tensor Euler ZYZ | Different rotation representations. Equivalent orientations can have different angle numbers; do not copy the three numbers between conventions. |
| Residual | Difference between prediction and experiment. Its sign depends on the workflow; see [Residual Analysis](../diagnostics/residual-analysis.md). |
| Current / Stale / Unknown | Result-validity states: matching working inputs, changed inputs, or insufficient evidence to establish a match. Current does not certify scientific correctness. |
| Preview / Apply | Preview temporarily displays supported fitted geometry; Apply transfers the fields supported by that workflow. Neither is performed just by running a fit. |
| Project / workspace / NMR session | `.pcsp` analysis data, `.pcsw` references to saved projects, and `.nmrsession` recoverable NMR work. Their scopes differ. |

For derivation and structural mapping, see
[Experimental PCS Data](../experimental-data/delta-exp.md). For result state
and application, see [Fitting Overview](../fitting/overview.md) and
[Projects and Persistence](../projects.md).
