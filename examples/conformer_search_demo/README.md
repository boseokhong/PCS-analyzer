# One-bond Conformer Search demo

The nine-atom fragment is synthetic teaching geometry, not a chemically optimized
complex or force-field validation. The tensor and metal are fixed: Δχ_ax=12,
Δχ_rh=3 E-32 m³, metal at the origin, zero orientation. H Refs 6–8 have noiseless
PCS generated from `truth.xyz`, which rotates the terminal fragment by +45° from
`start.xyz` about **Ref 3–4**. **Sync Main** converts the standard project's XYZ
orientation to the physically equivalent search ZYZ representation.

## Desktop exercise

1. Open `demo.pcsp` with **Open Project**. The project contains the starting
   structure and target PCS, not the solution coordinates or search candidates.
2. Select **Conformer Search → Run & Options** and choose **Sync Main** in its
   contextual Ribbon. Verify the origin, tensor 12 / 3, zero orientation and
   three experimental observations.
3. In **Fixed atoms**, fix **Refs 1 and 9**. Apply the locks in the Ribbon.
   Use **Position** constraints for this controlled comparison, and dihedral
   search rather than rigid-body search. Keep Anchor mode off.
4. In **Rotatable bonds**, select only **Ref 3–4**. Automatically detected other
   bonds are not part of this one-parameter example; clear their selections.
5. In **Run & Options**, enable global search. Use DE max iterations **80**,
   population **8**, workers **1**, and three candidates. Run **Run Search**.
6. Compare the best candidate with the original. Expect approximately +45°
   change, PCS RMSD below 0.0001 ppm, no detected clashes and unchanged fixed atoms.
   Candidate count/order may vary because equivalent solutions are deduplicated.
7. Inspect **View**, **Correlation** and the report. Save the best structure with
   **Save XYZ** or a report with **Save Report**. Existing **Apply / Revert** are
   explicit actions; a successful search alone does not overwrite the main geometry.

To demonstrate locking, mark a terminal H (for example Ref 6) fixed and apply
locks. Ref 3–4 should become locked. Restore Refs 1/9 as the fixed set before
rerunning. Do not feed the truth coordinates into the starting model.

The search changes structure while holding the tensor fixed; it does not fit
populations and is different from Multi-Conformer fitting. A synthetic known
answer does not prove uniqueness, chemical stability or suitability for a real
complex. Nonzero main orientations are converted through their rotation matrix;
standalone search angles remain ZYZ. Rerun nonzero-orientation searches made with
older source versions that copied XYZ angle numbers directly.

In Position mode, use **Max fixed-atom drift** (Cartesian displacement) to verify
the lock. **Max angular drift** excludes zero-length metal-centred vectors;
this exercise should show negligible drift for its fixed donor. Cartesian and
radial diagnostics retain all fixed atoms. With no valid directions, angular
drift is zero and does not imply a defined metal-centred direction.

## Reproduce

Persistence exercise: after running, save to a new `.pcsp` path, close its document
and reopen that copy. Candidate coordinates/reports, displayed selection, pending
preview, fixed atoms and bond checkboxes are restored without the temporary
Sync Main files. Viewing and queueing a candidate are separate actions: choose
**Select** before **Apply** if changing the queued candidate. Reopening does not
apply a pending preview. Already applied geometry stays applied and Revert remains
available. Changed main or search scientific inputs disable Apply; export/review
remains available. Current manual graph edits stay separate from the recorded
run graph. Sync Main replaces the search inputs and clears the previous set and
queued preview, so save/export first. Only the latest search set is saved, not
all earlier searches; pending replacement input files without a completed search
are not separate saved search sessions.

```shell
python examples/conformer_search_demo/generate_demo.py --output dev_docs/search-review
python examples/conformer_search_demo/generate_demo.py --overwrite
```

See the [public tutorial](../../docs/conformer-search/tutorial.md) to run the
search and inspect residuals, rotation, clashes and fixed displacement.
