# Joint PCS+RDC numerical demo

This example includes ready-to-open `.pcsp` projects and independent numerical
validation for the fixed tensor-frame model. It is not experimental validation.
The 24 synthetic N/H probes form 12 bonds of length 1.02 Å. Coordinates are
already metal-centred and tensor-oriented; the metal is at the origin.

Known parameters: Δχ_ax = -18 and Δχ_rh = 5 in E-32 m³ per molecule,
B₀ = 14.1 T and T = 298 K. PCS uncertainty is 0.02 ppm and RDC uncertainty
is 0.5 Hz. The observations are noiseless; the uncertainties specify fit scales,
not added random noise or measured precision.

## Ready-to-open projects

| Project | Purpose | Expected behavior |
| --- | --- | --- |
| `noiseless/demo.pcsp` | Numerical recovery | Current result: Δχ_ax=-18, Δχ_rh=5; residuals near floating-point precision. |
| `noisy/demo.pcsp` | Report and diagnostics | Seeded independent Gaussian noise, σ_PCS=0.02 ppm and σ_RDC=0.5 Hz. Current result close to the known tensor. |
| `ambiguous/demo.pcsp` | Identifiability failure | All probes/bonds lie along z. Run Fit rejects rank below 2; Apply remains disabled. |

Open a project with **Open Project**, then select **Advanced Fitting → Joint
PCS+RDC**. Shared controls remain in the contextual Ribbon. Successful projects
contain recorded inputs/results; reopening neither runs fitting nor applies
the fitted tensor. **Run Fit** explicitly recomputes. Supplied PCS uncertainty
weighting is enabled in these projects. **Export result** writes recorded JSON.

`invalid-rdc.csv` tests unsupported nuclei. Import must reject it without
replacing existing RDC inputs. Scenario folders contain `structure.xyz`,
`pcs.csv`, and `rdc.csv`; the reviewed complete setup is the project. RDC CSVs
use explicit `ref1`/`ref2`. Independent PCS import uses normal application
controls, rather than the RDC loader.

Expected noisy fit: about -18.012246 / 4.990465 E-32 m³; PCS RMSD 0.015990 ppm
and RDC RMSD 0.365648 Hz. These finite synthetic checks are not confidence
intervals. Fresh fits after reload can differ in the last digits because legacy
coordinate tables retain eight decimals. `validation.csv` records outcomes.

## Reproduce and validate

From the repository root:

```bash
python examples/joint_pcs_rdc_demo/generate_demo.py --overwrite
python examples/joint_pcs_rdc_demo/generate_projects.py --overwrite
python examples/joint_pcs_rdc_demo/run_demo.py --scenario noisy
python examples/joint_pcs_rdc_demo/run_demo.py --scenario ambiguous
```

The original generator writes root-level `structure.xyz`, `pcs.csv`, and `rdc.csv`, then checks
recovery of the known scalar values. PCS and RDC observations are generated
from independent Cartesian equations, not the application's forward functions.
Both generators accept `--output DIRECTORY` for a separate review copy and refuse
existing generated files unless `--overwrite` is supplied.
The RDC sign explicitly retains PCS Analyzer's existing negative dipolar
prefactor. Compare signed RDC definitions before importing external software
outputs. CSV RDC labels in this example are N1/H1 through N12/H12; a future
application tutorial must establish the corresponding actual application
labels or explicit Ref mapping rather than assuming an XYZ preserves them.
The scenario projects use explicit raw Ref pairs and avoid that label assumption.

The molecular alignment tensor is
`A = B0² Δχ / (15 μ0 kB T)`. There is no Avogadro factor for molecular
susceptibility in m³. See Orton et al., *Magnetic Resonance* (2020), Eq. 9:
https://pmc.ncbi.nlm.nih.gov/articles/PMC10500712/ .

The tests cover physical scale, field/temperature/bond-length scaling, signed
angular dependence, weighted fitting, unmatched data diagnostics, unsupported
nuclei, invalid conditions, rank-deficient observations, structured reductions,
project round trips and saved-result integrity. Statistical coverage and
full-tensor/metal-position fitting remain outside this demo's model. See the
[Joint PCS+RDC tutorial](../../docs/fitting/joint-pcs-rdc-tutorial.md) for the GUI workflow.
