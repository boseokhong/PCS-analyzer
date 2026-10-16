# Multi-Lanthanide demo

Open `noisy/demo.pcsp` or `noiseless/demo.pcsp` and select
**Advanced Fitting → Multi-Lanthanide**. The Ribbon **Analysis** selector provides
Tensor fitting, Reilley separation, Ln scaling, Nuclei diagnostics, Geometry / CFP
and Parameter table. Each project contains six verified recorded results.

These eight hydrogen probes are synthetic, not a chemically optimized molecule
or experimental validation. Twelve datasets contain separate PCS and total
paramagnetic shifts, δ_para (ppm), for Nd, Eu, Tb, Dy, Er and Yb.

- Tensor PCS values are generated independently from Cartesian susceptibility
  equations with Δχ_ax=0.30 C_D and Δχ_rh=0.055 C_D, in E-32 m³.
- Total shifts use G_eff=80 G_struct and F_eff ranging from −0.03 to +0.045,
  at 300 K. The noisy case adds reproducible Gaussian noise of 0.015 ppm to
  total shifts; supplied errors are 0.03 ppm. The noiseless case verifies exact
  parameter/component recovery. Tensor data remains noiseless in both projects.
- Relative C_D and RHF S(T) coefficients use the application's selected convention.
  This synthetic demo checks numerical recovery under that convention.
  Sm/Eu use fixed 300 K values; their temperature-dependent physics is not added.

`paramagnetic.csv` is the historical wide nucleus-by-Ln input form with explicit
Ref mapping, common geometry factors and an error column. `invalid.csv` must be
rejected, rather than silently converted to missing data.

Click an Ln heading to include/exclude its dataset. Tensor fitting also has a
Dataset view; double-click its first column to change inclusion. Ref mapping
edits apply to the same assigned site across local Ln datasets. Inputs and
conditions and the compact C_D / S(T) coefficient area collapse together. The
coefficient area shows included Ln ions and short Bleaney/Pinkerton references.
Result is a selectable report; splitters control
the report/input heights, and plots fill the remaining viewport.

Review PCS transfer only creates local PCS datasets from a Current Reilley
result. Choose model PCS or measured-minus-contact explicitly and review Ref
mapping. Derived values share fitted-coefficient uncertainty; they are not
independent measured observations. Main PCS and susceptibility tensor are not
overwritten. Missing mapping must be completed before meaningful tensor analysis.

From the repository root:

```shell
python examples/multi_lanthanide_demo/generate_demo.py --overwrite
```

Use `--output DIRECTORY` to generate a separate copy; existing generated files
require `--overwrite`. Only named demo outputs are replaced. Projects embed the source
coordinates, shifts, conditions and recorded results; external source files are
not required to reopen them. Project loading verifies deterministic linear
calculations against recorded outputs; it does not launch an analysis worker,
replace recorded outputs or Apply a main tensor. No nonlinear geometry fitting,
empirical absolute separation or anchor correction is enabled by this demo.
