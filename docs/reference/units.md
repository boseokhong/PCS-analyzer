# Units

This page summarizes units needed to interpret PCS Analyzer inputs and
exports. Scientific definitions remain on the canonical theory pages.

| Quantity | Application/file unit | Notes |
| --- | --- | --- |
| Cartesian coordinates and distances | Å | Main XYZ/ORCA structures, `.pcsp` coordinates, exported XYZ, and displayed distances |
| ORCA coordinates marked `BOHR` | Bohr on input; converted to Å | \(1\ \mathrm{Bohr}=0.529177210903\ \text{Å}\) |
| PCS and chemical shifts | ppm | UI labels δ_Exp and δ_PCS; δ_obs, δ_dia, δ_para, δ_nonPCS, and fitting residuals |
| \(G_{\mathrm{ax}}\) and \(G_{\mathrm{rh}}\) | Å⁻³ | Geometry factors use distances expressed in Å |
| Rotation controls | degrees | Structure XYZ and Tensor Euler ZYZ are distinct representations; `.pcsp` retains equivalent XYZ values |
| Experimental PCS uncertainty σ | ppm | Must be finite and positive for uncertainty-weighted fitting |
| Raw SSE/N | ppm² | Unweighted squared residual per observation, equal to RMSD² |
| Weighted χ² and χ²/N | dimensionless | Residuals divided by σ; χ²/N is per datum, not reduced chi-square |
| Conformer-search angles | degrees | Dihedral variables and rigid-body θ/φ controls |
| Exported PCS plot `Theta` | radians | Applies to **PCS Data** and **Atom Coordinates** exports from **Save plot data...** |
| Rhombicity Table `theta(deg)`, `phi(deg)` | degrees | Units are explicit in the column headings |
| Δχ_ax and Δχ_rh inputs | \(10^{-32}\,\mathrm{m^3}\) per molecule | A displayed value of 1 means \(1\times10^{-32}\,\mathrm{m^3}\) |
| Reconstructed molar susceptibility | \(\mathrm{m^3\,mol^{-1}}\) | Uses the tensor convention and Avogadro conversion described below |
| NMR spectrum position and stick-display FWHM | ppm | Stick broadening changes the display, not measured line shapes |
| NMR FID exponential/Gaussian apodization | Hz | Processing broadening parameters; separate from stick-display width |
| RDC | Hz | Joint PCS+RDC input and prediction |
| PRE Gamma2/R2_para | s⁻¹ | PRE input, prediction, and RMSD |
| Magnetic field | T | Advanced RDC/PRE controls |
| Temperature | K | RDC, Multi-Lanthanide, VT NMR and PCS Workbench inputs |
| PRE correlation time | seconds internally; ns in controls/report | Default bounds are 0.001–1000 ns (\(10^{-12}\)–\(10^{-6}\) s); user bounds are configurable |
| PCS Workbench ORCA tensor | cm³·K·mol⁻¹ on input; Å³ per molecule after conversion | The selected temperature enters the conversion; the Workbench then removes the isotropic trace |
| PCS Workbench `.3d` origin and spacing | Å as interpreted by the current reader | No length-unit conversion is applied to these grid fields |

## VT observed-shift coordinates

NMR Workbench uses effective temperature **T in K**, inverse temperature
**1/T in K⁻¹**, observed shift **δ in ppm** and **δT in ppm·K**. Curie/
Curie-Weiss parameters use offset in ppm, C in ppm·K and θ in K. θ is a
temperature-model parameter, not an orientation angle. Coordinate-linear fit
units depend on the selected axes; raw-shift model metrics and transformed
plot residuals have different domains. See [VT Analysis](../nmr/vt-analysis.md).

## Coordinate and angle exports

XYZ output always contains Cartesian coordinates in Å. The main **Save current
XYZ...** command writes the transformed, metal-centred view rather than the
original absolute input coordinates; see [File Formats](file-formats.md#visible-xyz).

Do not infer an angle's unit from the word `Theta` alone. Main PCS plot-data
exports use radians, while Rhombicity Table headings explicitly use degrees.
Application rotation entries and fitted-angle reports are degrees.

The distinction between δ_Exp, calculated PCS, observed/diamagnetic shifts,
and δ_nonPCS is summarized in
[Mathematical Conventions](mathematical-conventions.md#shift-notation).

## Susceptibility units

PCS Analyzer's Δχ inputs use \(10^{-32}\,\mathrm{m^3}\) per molecule. Tensor
reconstruction may also display SI molar susceptibility in
\(\mathrm{m^3\,mol^{-1}}\). These are not numerically interchangeable without
the documented scaling and Avogadro conversion.

See [Tensor Conventions](../pcs/tensor-conventions.md#units) for the complete
definitions and the cgs-to-SI warning. See
[PCS Theory](../pcs/theory.md) for the PCS prefactor used with the geometry
factors. PCS Workbench's separate ORCA conversion and grid assumptions are
documented under [PCS Workbench Inputs](../pcs-workbench/overview.md#inputs).
