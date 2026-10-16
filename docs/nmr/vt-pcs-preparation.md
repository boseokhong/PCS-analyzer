# VT PCS Pairing and Reviewed Transfer

Open **Peak Tracking → Workflow → PCS preparation…**, or **VT Analysis →
Workflow → VT PCS preparation…**. This workflow starts from reviewed
[track/observable associations](peak-tracking.md#convert-correspondence-into-reviewed-assignments).
It does not fit a tensor or silently reinterpret an observed-shift trend as PCS.

## Review temperature-specific references

1. Choose a reviewed **VT observation** in **Prepare / pair**.
2. Inspect **Reference dataset**, observable, **T / K**, **δ / ppm** and pairing
   diagnostics. Select a reference explicitly.
3. Choose a **Reviewed structural target** and inspect its membership in the
   bound PCS document.

Paramagnetic/diamagnetic role, nucleus, assignment, source validity and
temperature must be compatible. The current implementation requires **exact
effective Kelvin equality**: it does not interpolate or choose the nearest
reference temperature. Assignment matching trims whitespace but retains case
in this VT route, unlike the case-insensitive single-pair comparison.
Duplicate/repeated candidates need review, not averaging
or automatic selection. A separate reference dataset still needs a live peak
and synchronized assignment; general prepared CSV values alone do not supply
that spectral provenance for this VT reference route.

## Keep quantities and scientific basis explicit

The provider selector exposes these current choices:

| Choice | Meaning |
| --- | --- |
| **shift_difference_only** | Retain the reviewed para-minus-dia difference and provenance; it is not transfer-ready experimental PCS. |
| **direct_delta_shift** | Treat Δshift as PCS only under an explicitly justified basis. |
| **manual_fcs** | Subtract a supplied **Manual contact / ppm** correction from Δshift. |
| **manual_pcs** | Use supplied **Imported PCS / ppm** for the reviewed pair. |

For a PCS provider, supply the required finite correction/manual value and a
**Scientific basis / review note**. A note records an explicit decision; it
does not prove the approximation. These are not automatic contact separation
or temperature-dependent susceptibility fits.

Choose **Prepare reviewed pair** to save the reviewed evidence. **Preparation**
lets you inspect provenance or remove a saved preparation. Changes to source
peaks, assignments, conditions or targets can make saved evidence stale; inspect
readiness and prepare/review again rather than assuming automatic validity.

## Transfer only reviewed PCS observations

The main PCS Analyzer is required for a destination and structure. In
**Reviewed transfer**:

1. Check **Destination** and use **Choose document** when needed. A main tab
   switch alone does not redirect a bound Workbench.
2. Select intended preparations with Ctrl/Shift, or **Select temperature**.
   Nothing is initially selected or transferred automatically.
3. Choose **Preview transfer**. Review readiness, single/grouped targets,
   additions, replacements and destination conflicts.
4. Approve only the explicitly listed replacements if appropriate, then choose
   **Transfer reviewed** and confirm. Changed selection/sources/destination
   invalidate the preview and require another review.

Transfer produces a receipt and updates the destination's PCS observations;
it runs no fitting. Unrelated destination data remains intact. Later NMR source
changes do not rewrite transferred PCS values automatically. Save both the
[NMR session](sessions.md) and the [PCS project](../projects.md).

In the [synthetic VT tutorial](tutorial-vt.md), the only reference is at 298 K.
Other temperatures remain unresolved. The known differences at 298 K are a
workflow check, not an independently established PCS or contact-shift model.
