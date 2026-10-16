# Conformer Search

Applying or reverting a conformer is recorded in the active document's
[Undo/Redo history](../projects.md#undo-and-redo). Undo restores the previous
working structure while keeping search candidates and their recorded reports.
**Revert** to the original structure remains a separate action; its own change can
also be undone. The session history is not saved in project files.

Conformer Search changes molecular coordinates to find structures whose calculated PCS values agree with the current experimental PCS data. It uses the current metal position, susceptibility tensor, and tensor orientation as fixed calculation inputs; it does **not** refit those quantities. The result is therefore a structural search under the supplied PCS model, not an alternative to [tensor fitting](../fitting/overview.md).

The tool is available in the main analysis notebook as **🔍 Conformer Search**. Its embedded workflow has four tabs: **① Fixed atoms**, **② Rotatable bonds**, **③ Run & Options**, and **④ Results**.

The contextual **Conformer Search** Ribbon follows the selected inner tab.
Shared actions are on that Ribbon; the analysis area retains input controls,
candidate tables and reports. The standalone tool can still show its original
local buttons. The instructions below describe the embedded workflow.

## Search and constraint choices

Two separate choices define a search:

| Choice | Options | Meaning |
| --- | --- | --- |
| **Search mode** | **Dihedral torsion search** | Varies selected rotatable-bond dihedral angles while preserving the bond lengths, bond angles, and ring geometry implied by the input structure. |
|  | **Rigid-body anchor rotation** | Rotates a rigid molecular fragment around a metal-centred axis defined by selected anchor atoms. This is intended for closed chelates or other rigid fragments without a useful free dihedral. |
| **Constraint model** for a dihedral search | `position` | Keeps fixed atoms near their original Cartesian positions. |
|  | `angular` | Keeps the metal-centred directions of fixed atoms while allowing radial changes, and penalizes short metal–atom contacts. |

Rigid-body search uses its own **Axis mode** and **Lock mode** controls rather than the dihedral **Constraint model**. See [Constraints](constraints.md) for exactly what each option preserves.

## Inputs inherited from PCS Analyzer

Before using **Conformer Search Ribbon → Sync Main**, the main window must contain:

- a loaded structure;
- experimental **δ_Exp** values matched to structure Ref IDs;
- a valid metal assignment and metal coordinates; and
- the tensor magnitude, rhombic component, and orientation intended for the comparison.

Syncing creates an internal conventional XYZ representation and a `Ref, δ_exp` PCS table for the search. Only Ref IDs that have δ_Exp values are included. The PCS terminology and matching rules are described in [Experimental PCS](../experimental-data/delta-exp.md) and [Import and Export](../experimental-data/import-export.md).

!!! note "Orientation convention in the embedded hand-off"

    **Sync Main** converts the main window's active extrinsic `XYZ` orientation to the search calculator's intrinsic `ZYZ` representation while preserving the same molecular-to-tensor rotation matrix. The displayed angle numbers can differ; the physical orientation is unchanged. Raw molecular coordinates and the metal position remain in the same absolute frame. Standalone search inputs and `--euler-zyz` still use ZYZ. Earlier source versions copied XYZ numbers directly, so rerun searches made with nonzero orientations in those versions. See [Coordinate Handling](../structures/coordinate-system.md) and [Tensor Conventions](../pcs/tensor-conventions.md).

## End-to-end workflow

1. Load the structure and experimental data in the main window, assign the metal centre, and set the tensor inputs.
2. Open **🔍 Conformer Search**, select **③ Run & Options**, and choose **Sync Main** on its contextual Ribbon. Review the imported structure, PCS point count, metal position, tensor, and orientation summary.
3. In **① Fixed atoms**, mark donor or rigid-core atoms that should constrain the search. The metal is fixed implicitly. Choose **Ribbon → Apply Locks** to rebuild and inspect the lock classification.
4. For **Dihedral torsion search**, review the automatically detected bonds in **② Rotatable bonds**, correct false or missing connectivity if necessary, and select the free bonds to vary. For **Rigid-body anchor rotation**, the selected fixed atoms other than the metal become the anchor list.
5. In **③ Run & Options**, choose the search mode, constraints, optimization level, and candidate output settings. Choose **Ribbon → Run Search**.
6. In **④ Results**, compare candidate metrics and the per-atom PCS report. **Ribbon → View** overlays the original and up to five candidates in a 0–180° or folded 0–90° polar structure view. **Correlation** shows calculated-versus-experimental PCS and residuals.
7. Use **Ribbon → Select** to send a chosen result to the main window as a preview. **Apply** commits the preview to the current raw structure; **Discard** removes the preview without changing the current structure; **Revert** restores the initially loaded structure.
8. Use **Ribbon → Save XYZ** for selected candidates and **Save Report** for the active text report. Save the structure comparison from **Save figure…** in its viewer.

Applying a candidate updates the current raw geometry and recalculates the effective representation when [symmetry or torsional averaging](../structures/symmetry-averaging.md) is active. Current projects retain the latest candidate set, reports, search inputs and selection state without needing the original temporary XYZ/CSV files. Reopening restores results without rerunning the search or applying a pending preview. Changed host inputs disable Apply while results remain available for review/export. Older summary-only projects cannot reconstruct candidates. See [Projects and Persistence](../projects.md).

## What the result means

Changing the current search's tensor/centre, constraints, fixed/bond selection or
manual connectivity blocks Apply for an older result. Execution settings such as
iteration count alone do not invalidate an existing candidate. Sync Main
explicitly replaces inputs and clears the old set; save/export it first.

The candidate report provides RMSD, R², Q factor, per-atom calculated PCS and residuals, clash counts, and constraint diagnostics. Candidate generation and ranking also include geometry penalties, so a candidate is not ranked by PCS RMSD alone. The reported metrics describe agreement under the fixed tensor and selected structural model; they do not establish a unique conformation or a thermodynamic population.

See [Optimization](optimization.md) for the objective and search stages, and [Fit-quality Metrics](../fitting/quality-metrics.md) for the definitions of the PCS metrics.
