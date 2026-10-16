# PRE demo

These six proton probes are synthetic, not a molecular structure or experimental
validation. The metal is at the origin; distances are 7.5–19 Å. An independent
SI implementation generates proton Γ₂ at B₀=14.1 T, μ_eff=3.68 μB and effective
τ_c=3 ns. Supplied uncertainties are 5% of the exact rates.

Open a prepared `demo.pcsp` and select **Advanced Fitting → PRE**:

| Folder | Expected result |
| --- | --- |
| `noiseless/` | τ_c=3 ns; rounding-scale residuals. |
| `noisy/` | Fixed perturbations `[0.5,-0.8,0.3,0.9,-0.6,0.2] σ`; τ_c≈3.012644691 ns, RMSD≈39.41814549 s⁻¹, weighted sum of squares≈2.148333333. |
| `boundary/` | Same noisy observations, upper bound 1 ns: upper-bound solution, poor fit; not an estimate of 1 ns supported by the unconstrained data. |

Each folder includes embedded source geometry, observations, conditions and
a verified recorded result in `demo.pcsp`, plus `structure.xyz` and `pre.csv`.
Opening does not refit or change the main susceptibility tensor. To rerun,
select **Run Fit** in the Ribbon. **Input data** collapses both Fit conditions
and its table. Inspect **Distance profile**, **Correlation** or **τ_c scan**.

`invalid-pre.csv` demonstrates rejection of unsupported 13C observations.
Canonical rates are increments Γ₂=R₂,para−R₂,dia, in s⁻¹, not raw R₂.
The simplified proton model fits only effective τ_c, with fixed geometry,
field and moment; it is not full SBM/Curie relaxation or a general-nucleus model.

Reproduce the files and numerical checks from the repository root:

```shell
python examples/pre_demo/generate_demo.py --overwrite
```

Use `--output DIRECTORY` to generate a separate copy; existing generated files
require `--overwrite`. Only named demo outputs are replaced. `validation.csv` records the
expected fit metrics for comparison with the supplied scenarios.
