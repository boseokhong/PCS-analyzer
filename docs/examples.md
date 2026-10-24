# Examples and Practice Data

Use [Quick Start](quick-start.md) to begin with your own structure and PCS data.
The examples below are optional exercises for learning a workflow or checking
how a known dataset behaves.

## Find and open the files

Use the example files supplied with the same PCS Analyzer release as your
application. In a source checkout they are under `examples/`; in the portable
build, `examples/` is beside the executable. Keep each example's directory intact.
The [repository examples directory](https://github.com/boseokhong/PCS-analyzer/tree/main/examples)
contains the same practice inputs. Use the tutorials below for instructions;
no Python generation scripts are needed.

- **Portable application:** extract the full release archive, then open the
  `examples/` directory beside the executable.
- **Source files:** extract the matching release's source archive and use its
  `examples/` directory. For an FID, download the directory contents together;
  a GitHub HTML file preview is not an input file.

Open `.pcsp` files with **File → Open PCS Project...**. They already contain
the saved structure and analysis inputs. Save your practice results to another
path. For FID exercises, open the dataset in NMR Workbench as directed by the
tutorial; a FID directory is not a PCS project.

For a prepared project, do not load its neighbouring XYZ or reimport its CSV
unless the tutorial asks you to. Those files let you inspect or import individual
inputs; the project already contains them. Reloading a structure starts a
different preparation workflow and can replace the restored inputs.

## Choose an exercise

Paths below are relative to `examples/`.
Within each tutorial, short paths such as `noisy/demo.pcsp` are relative to that
tutorial's demo directory. CSV files used alongside a project come from the same
scenario directory, unless another location is explicitly given.

| Learn to… | Starting file or directory | Expected result and tutorial |
| --- | --- | --- |
| Fit tensor orientation and compare rhombicity | `standard_fitting_demo/mode_b/demo.pcsp` | Known axial/rhombic values 12 / 3 E-32 m³; [standard fitting tutorial](fitting/standard-fitting-tutorial.md) also covers Modes A/C and axial/rhombic diagnostic pairs |
| Derive PCS from observed/reference shifts | `nmr_pcs_fitting_demo/direct/start.pcsp` | Twelve reviewed targets; fitted tensor 12 / 3 under the stated contact assumptions; [NMR-to-PCS tutorial](nmr/tutorial-pcs-fitting.md) |
| Compare conformer populations and reliability | `multi_conformer_demo/informative/demo.pcsp` | Compare informative and indistinguishable conformer signatures; [Multi-Conformer tutorial](fitting/multi-conformer-tutorial.md) |
| Fit PCS and RDC together | `joint_pcs_rdc_demo/noisy/demo.pcsp` | Known-tensor recovery with noise; a separate rank-deficient case is rejected; [Joint PCS+RDC tutorial](fitting/joint-pcs-rdc-tutorial.md) |
| Fit a simplified PRE model | `pre_demo/noisy/demo.pcsp` | Effective correlation time near 3 ns; compare a restricted-bound case; [PRE tutorial](fitting/pre-tutorial.md) |
| Compare lanthanides and separate shift contributions | `multi_lanthanide_demo/noisy/demo.pcsp` | Tensor, Reilley and series diagnostics at 300 K; [Multi-Lanthanide tutorial](fitting/multi-lanthanide-tutorial.md) |
| Search a selected torsion | `conformer_search_demo/demo.pcsp` | About +45° rotation with fixed atoms retained; [Conformer Search tutorial](conformer-search/tutorial.md) |
| Inspect averaging and viewer representations | `averaging_visualization_demo/groups.xyz` | Detect the supplied synthetic groups and compare representations; [averaging tutorial](visualization/averaging-demo.md) |
| Process raw NMR or import prepared shifts | `nmr_workflow_demo/` | Five picked peaks and four sample assignments in the FID exercise; [FID/CSV tutorial](nmr/tutorial-fid-csv.md) |
| Track a VT series | `nmr_vt_demo/FID/Paramagnetic/` | Five temperatures, four tracks and Curie-model comparisons; [VT tutorial](nmr/tutorial-vt.md) |
| Inspect a molecular PCS case | `documentation_demo/Nd_iPr2BA3_demo/Nd_iPr2BA3_axial.pcsp` | Supplied structure and experimental shift lists; see the interpretation notes below |
| Practice measured-FID processing | `documentation_demo/Yb_iPr2BA3_demo/PROTON_01.fid/` | A measured proton spectrum; use the [processing recipe below](#yb-measured-fid) |

## Interpret the examples

**Synthetic examples** have known generating assumptions. Their small residuals
or successful recovery test those assumptions and the calculation; they do not
establish accuracy for an experimental molecule. Follow each tutorial's numerical
checkpoints and limitations. The FID/VT teaching signals are simulated spectra,
not measured evidence for the toy structure.

**The Nd molecular project** retains a saved axial tensor and averaging settings.
On opening, its G_ax diagnostic uses all assigned visible points, including iPr
data. That regression need not reproduce the stored tensor. Contact shifts,
conformation and shared/averaged assignments require review before choosing a
scientific fitting subset. The project does not restore a historical subset;
do not select points solely to improve R². See the [dataset notes below](#nd-molecular-project).

**The Yb acquisition** is measured, but its Demo A/B labels are illustrative
annotations, not established molecular assignments. Its processing exercise does
not validate a PCS model or connect the Yb spectrum to the Nd structure. Available
acquisition conditions and processing parameters are recorded [below](#yb-measured-fid).

PCS Workbench PDE calculations need a matching ORCA tensor/geometry output and spin-density grid.
Tensor inspection needs only ORCA; [tensor transfer](pcs-workbench/tensor-transfer.md)
also needs a matching Analyzer document.
That pair is not included in these examples; see its
[input requirements](pcs-workbench/overview.md).

## Nd molecular project

The supplied Nd(*i*Pr₂BA)₃ example contains:

| File | Contents |
| --- | --- |
| `Nd_iPr2BA3.xyz` | Pre-aligned 103-atom structure; Nd is Ref 1. |
| `Nd_iPr2BA3_paramagnetic_shifts.csv` | Paramagnetic shifts in ppm, keyed by structure Ref ID and used as δ_Exp. |
| `Nd_iPr2BA3_axial.pcsp` | Embedded structure, assignments, methyl symmetry averaging and phenyl rotational averaging. |
| `Nd_iPr2BA3_observed_shifts.csv` | Observed Nd sample shifts, δ_obs in ppm. |
| `La_iPr2BA3_observed_shifts.csv` | Diamagnetic La reference shifts, δ_dia, with corresponding Ref IDs. |

The saved project uses axial prediction, Δχ_ax = −3.90816 E-32 m³, and no rhombic
contribution. Three phenyl groups use uniform full rotations with 24 samples
per group. All elements and ligands are visible; original/reference overlays are off.

To inspect this case:

1. Choose **File → Open PCS Project...** and open
   `examples/documentation_demo/Nd_iPr2BA3_demo/Nd_iPr2BA3_axial.pcsp`.
2. Check 103 raw atoms with Nd at Ref 1 and assigned δ_Exp values in the main
   table. Do not reimport the CSVs for this prepared-project exercise.
3. Open **G_ax vs PCS** to inspect the current diagnostic. Its fitted slope is
   calculated from the displayed points; it is not the saved −3.90816 tensor value.
4. Follow [Averaging and Visualization](visualization/averaging-demo.md) to compare
   representations without refitting the tensor. Save any changes to a new path.

The supplied structure and experimental shift lists have no accompanying
publication/CIF identifier or complete experimental-methods record. Establish
sample conditions, assignment evidence and diamagnetic-reference suitability
before treating them as an experimental benchmark.

The observed-shift pair can also be used in [Shift Analysis and PCS Preparation](nmr/non-pcs-analysis.md).
Blank cells are unassigned; numeric zeros remain data. Direct subtraction is a
derivation preview, not evidence that every difference is pure PCS. Inspect
assignments, targets and possible contact contributions before fitting.

## Yb measured FID

Keep `fid` and `procpar` together in `PROTON_01.fid/`. Open that directory through
**NMR Workbench → Open Dataset Folder…**, or its `fid` through **Open FID File…**.
Launch NMR Workbench from the **Tools** Ribbon first, then use its **File** menu.
The measured Varian/Agilent acquisition metadata identifies ¹H, approximately
399.9 MHz, benzene-*d*₆ and 298.15 K. Non-scientific identifiers and system paths
were anonymized; raw FID and scientific acquisition parameters were retained,
with processing parity checked.

For the documentation spectrum, start with a fresh import and Preview/Apply each
operation once in this order:

| Operation | Parameters |
| --- | --- |
| Exponential apodization | 2 Hz |
| Manual phase | P0 = 131.9°, P1 = 1049°, pivot = 0 ppm |
| Baseline | arPLS, lambda = 1e9 |
| Positive peak picking | Height 2%, prominence 2%, minimum separation 100 points |

With these settings, the supplied acquisition gives 17 positive picked peaks.
Peak detection does not assign those peaks to molecular sites.

These are dataset-specific processing choices. The vendor ppm reference is
retained; no independent calibration, publication identifier or validated
molecular assignments are supplied. Demo A/B labels are illustrative only.
This Yb processing exercise is separate from the Nd/La PCS example and does not
transfer Yb observations to the Nd structure. See [FID processing](nmr/raw-processing.md).
