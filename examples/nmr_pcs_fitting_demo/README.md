# NMR → PCS → standard fitting

This is a synthetic **CSV-only** exercise with twelve H probes. No FID, real
assignment or chemical model is inferred. The independent generating tensor is
Δχ_ax=12, Δχ_rh=3 E-32 m³ with zero orientation at the origin, at a declared 300 K.
H1–H12 map explicitly to Refs 3–14; Nd is Ref 1 and N is Ref 2.

| Folder | Generating assumption | PCS Preparation provider |
|---|---|---|
| `direct/` | δ_obs = δ_dia + PCS; contact is exactly zero by construction | **Direct from Δshift** |
| `contact_corrected/` | δ_obs = δ_dia + PCS + supplied synthetic contact | **FCS-corrected** with the listed correction per assignment |

Each folder contains absolute observed `paramagnetic.csv` and `diamagnetic.csv`,
`mapping_and_truth.csv`, `structure.xyz`, an empty-PCS `start.pcsp` and a
`prepared.pcsp` checkpoint. The checkpoint is created through the production
import, pairing, provider and reviewed structured-handoff APIs; it is not a fitted
result or a shortcut that assigns real atoms from text labels. No measurement
uncertainties or random noise are supplied. Use unweighted fitting.

## Desktop exercise

1. Open `direct/start.pcsp`. Open the integrated **NMR Workbench**, in a fresh
   session without the earlier FID/CSV teaching data.
2. In **NMR Data**, import both observed-shift CSVs through Preview and Apply.
   Their roles, 1H nucleus and 300 K temperatures are in the files. Verify twelve
   observations in each dataset. These are observed shifts, not imported PCS.
3. In **Shift Analysis**, pair Paramagnetic and Diamagnetic by assignment. Review
   twelve matched rows; δ_para is their difference.
4. In **PCS Preparation**, choose **Direct from Δshift**. This is justified only
   by the explicit zero-contact assumption of this synthetic scenario.
5. Use **Single atom** targets and map H1–H12 to Refs 3–14 using the supplied
   mapping table. Inspect the selected structure/target for each assignment;
   matching labels alone do not establish structural identity.
6. Use **Review & Apply…** for the selected rows. Review the destination document
   and conflicts before transfer. Expect twelve structured PCS observations in
   the main document. Save your practice NMR session and PCS project separately.
7. In main **Fitting**, choose Mode B, select all twelve observations, fit both
   tensor components, use **Unweighted**, no global search and zero-angle
   initialization. Run Fit: expect axial 12, rhombic 3 and RMSD below 0.000001 ppm.
8. Repeat in a fresh session with `contact_corrected/start.pcsp`. Choose
   **FCS-corrected** and enter each known `contact_ppm` from `mapping_and_truth.csv`.
   PCS = δ_para − contact. Review/transfer and recover the same tensor. Direct
   subtraction in this second case includes contact and is not the correct PCS.

To practice fitting alone, open the corresponding `prepared.pcsp` instead.
It preserves PCS provenance and targets, but does not contain the NMR datasets
or a fitted result. Reopening it cannot reconstruct a saved `.nmrsession`.

These known corrections are supplied by the generator, not estimated by the
application or established from measurements. This example does not show that
unknown contact contributions can be separated from one real shift pair.

## Try PCS transfer Undo/Redo

After **Review & Apply…**, use the main Undo arrow beside Help. The destination's
previous PCS values, structured observations, uncertainties and provenance return.
Redo reinstalls the recorded transfer without repeating PCS Preparation. NMR
source datasets and historical transfer receipts remain unchanged.

In a scratch document, also try a main-table δ_Exp edit or Clear δ_Exp and undo
it. This changes the legacy δ_Exp mapping; existing structured-observation
precedence is preserved. Fit reports are retained when input edits are undone.
Saving retains current PCS data, while reopening starts an empty undo stack.

## Reproduce

```shell
python examples/nmr_pcs_fitting_demo/generate_demo.py --output dev_docs/nmr-pcs-review
python examples/nmr_pcs_fitting_demo/generate_demo.py --overwrite
```

See [the public tutorial](../../docs/nmr/tutorial-pcs-fitting.md) and keep this
known-tensor exercise separate from the existing simulated FID/VT examples.
