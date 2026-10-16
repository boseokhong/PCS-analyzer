# PCS Analyzer examples

Use examples from a source checkout containing the corresponding feature, or the
examples directory supplied with a newly built portable application. Older binaries
may not contain the newer fitting workflows. Generators and validation commands
require the source Python environment; practice data can be opened in the desktop UI.

| Example | Start here | Purpose and expected outcome | Evidence |
|---|---|---|---|
| [Standard fitting and rhombicity](standard_fitting_demo/README.md) | `standard_fitting_demo/mode_a/demo.pcsp` | Donor-axis A, orientation B, full-tensor/metal C; paired axial/rhombic diagnostics | Synthetic known-tensor probes |
| [Conformer Search](conformer_search_demo/README.md) | `conformer_search_demo/demo.pcsp` | One selected bond, fixed atoms, about 45° rotation; compare/export a candidate | Synthetic teaching fragment |
| [NMR → PCS → fitting](nmr_pcs_fitting_demo/README.md) | `nmr_pcs_fitting_demo/direct/start.pcsp` | Import absolute shifts, pair, map, review PCS transfer, recover 12 / 3 E-32 m³ | Synthetic; contact assumptions explicit |
| [Averaging and visualization](averaging_visualization_demo/README.md) | Existing Nd project; `groups.xyz` for group detection | Averaging comparisons and 2D/3D presentation; 3 methyl, 1 CF₃ and 1 tBu | Synthetic detection fragments; supplied Nd molecular example |
| [Multi-Conformer](multi_conformer_demo/README.md) | `informative/demo.pcsp` | Recover populations; compare ambiguous signatures and bootstrap/CV | Seeded synthetic probes |
| [Joint PCS+RDC](joint_pcs_rdc_demo/README.md) | `noisy/demo.pcsp` | Known tensor recovery; reject rank-deficient data | Independent synthetic equations |
| [PRE](pre_demo/README.md) | `noisy/demo.pcsp` | Effective τ_c near 3 ns; compare upper-bound case | Simplified synthetic proton rates |
| [Multi-Lanthanide](multi_lanthanide_demo/README.md) | `noisy/demo.pcsp` | Six tensor/Reilley/series analyses at 300 K | Synthetic under the application's coefficient convention |
| [NMR FID/CSV workflow](nmr_workflow_demo/README.md) | FID directories **or** CSV files in a fresh session | Five picked peaks, four sample assignments; compare shifts | Simulated teaching signals |
| [VT NMR](nmr_vt_demo/README.md) | `FID/Paramagnetic/` | Five temperatures, four tracks, Curie analysis | Simulated teaching signals |
| [Nd molecular example](documentation_demo/Nd_iPr2BA3_demo/README.md) | `Nd_iPr2BA3_axial.pcsp` | Molecular PCS analysis and interface/visualization practice | Supplied structure/experimental shift lists; interpretation caveats |
| [Yb raw NMR](documentation_demo/Yb_iPr2BA3_demo/README.md) | `PROTON_01.fid/` | Process an actual vendor acquisition | Measured FID; illustrative annotations are not established assignments |

PCS Workbench requires its own matching ORCA tensor/structure output and density
grid. Such a data pair is not bundled; its separate demo remains deferred.

## Regeneration

All synthetic generators accept `--output DIRECTORY` and `--overwrite`.
Existing generated files are rejected unless replacement is explicit. For example:

```shell
python examples/standard_fitting_demo/generate_demo.py --output example-output/standard-review
python examples/standard_fitting_demo/generate_demo.py --overwrite
python examples/joint_pcs_rdc_demo/generate_projects.py --output example-output/joint-review
```

Only the named generated files are replaced; README files and unrelated files are
retained. Seeded numerical inputs are reproducible; serialized runtime identities
and numerical last digits need not be byte-identical across library versions.
Do not regenerate the measured Nd/Yb documentation inputs as synthetic data.
