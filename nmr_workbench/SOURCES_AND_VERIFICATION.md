# NMR Workbench Reference Presets — Sources and verification

## Scope

This bundle contains every solvent/reference item requested for the first NMR Workbench
reference-preset milestone:

- 1,4-Dioxane-d8
- Acetic acid-d4
- Acetone-d6
- Acetonitrile-d3
- Benzene-d6
- Chloroform-d
- Cyclohexane-d12
- Deuterium oxide (D2O)
- DMSO-d6
- Ethanol-d6
- Methanol-d4
- Dichloromethane-d2 / methylene chloride-d2
- DMF-d7
- Pyridine-d5
- THF-d8
- Toluene-d8
- Trifluoroacetic acid-d
- TMS (reference standard, deliberately not modeled as a solvent)

Both 1H and 13C entries are represented when a meaningful solvent signal exists.

## Source hierarchy

1. **Fulmer et al., Organometallics 2010** is preferred when its Supporting Information
   directly reports the deuterated-solvent residual signal. It is peer reviewed and provides
   explicit residual-solvent rows for 1H and solvent-signal rows for 13C.
2. **Cambridge Isotope Laboratories NMR Solvent Data Chart** supplies requested solvents
   not covered by Fulmer et al. and provides one consistent experimental chart at 295 K.
3. **Gottlieb et al., J. Org. Chem. 1997** is retained as the foundational literature source
   and cross-check, especially for common solvents and the temperature dependence of HDO.
4. **IUPAC Recommendations 2001** support TMS as the primary reference basis.
5. **Texas A&M's CIL-chart reproduction** is included as an institutional cross-check for
   the CIL-only solvent values.

## Important discrepancies retained rather than hidden

Different legitimate sources do not always give numerically identical solvent shifts. The
resource therefore stores one recommended `ppm` value and preserves alternate measurements
under `verification`.

Examples:

- CDCl3 1H: Fulmer/Gottlieb 7.26 ppm; CIL chart 7.24 ppm.
- CDCl3 13C: Fulmer 77.16 ppm; CIL chart 77.23 ppm.
- THF-d8 1H beta-CH2: Fulmer 1.72 ppm; CIL chart 1.73 ppm.
- THF-d8 13C: Fulmer 67.21/25.31 ppm; CIL chart 67.57/25.37 ppm.
- Toluene-d8: Fulmer gives 2.08, 6.97, 7.01, 7.09 ppm for 1H residual
  signals, while the CIL chart gives 2.09, 6.98, 7.00, 7.09 ppm.
- Methanol-d4 13C: Fulmer 49.00 ppm; CIL chart 49.15 ppm.
- Acetonitrile-d3 13C: Fulmer 1.32/118.26 ppm; CIL chart 1.39/118.69 ppm.

These differences are exactly why the program should store source provenance with a resolved
numeric target rather than treating solvent shifts as timeless universal constants.

## Exchangeable signals

The following signals are included for completeness but marked
`recommended_for_direct_referencing = false`:

- HDO / water in D2O
- exchangeable OH/OD signal in methanol-d4
- exchangeable OH/OD signal in ethanol-d6
- acid proton in acetic acid-d4
- acid proton in trifluoroacetic acid-d

Their positions can change substantially with temperature, pH, concentration, water content,
and sample composition. The UI should surface a caution before such a signal is used as a
reference target.

## Source details

### Fulmer et al. 2010
DOI: 10.1021/om100106e

Supporting Information:
https://scs.illinois.edu/system/files/inline-files/11BFulmer2010NMRSolventsSupplement.pdf

The SI reports, among other values, the following residual-solvent 1H signals:
THF-d8 1.72/3.58; CD2Cl2 5.32; CDCl3 7.26; toluene-d8
2.08/6.97/7.01/7.09; C6D6 7.16; acetone-d6 2.05; DMSO-d6 2.50;
CD3CN 1.94; CD3OD 3.31; D2O 4.79 ppm.

The corresponding 13C solvent-signal row includes:
THF-d8 67.21/25.31; CD2Cl2 53.84; CDCl3 77.16; toluene-d8
137.48/128.87/127.96/125.13/20.43; C6D6 128.06; acetone-d6
29.84/206.26; DMSO-d6 39.52; CD3CN 1.32/118.26; CD3OD 49.00 ppm.

### Cambridge Isotope Laboratories NMR Solvent Data Chart
Branded CIL chart:
https://www.csustan.edu/sites/default/files/2022-02/cambridge_isotope_labs_nmr_solvent_data.pdf

The chart states that the solvent spectra were recorded on a Varian Gemini 200 at 295 K.
It covers all of the requested deuterated solvents except that TMS is a reference standard
rather than a solvent.

### Gottlieb et al. 1997
DOI: 10.1021/jo971176v

Foundational residual-solvent paper. It also explicitly illustrates that HDO chemical shift
depends strongly on temperature.

### IUPAC 2001
DOI: 10.1351/pac200173111795

Recommends the unified chemical-shift scale relative to the 1H resonance of TMS in dilute
solution.

## Intended software behavior

- Store the resolved numeric target in ProcessingHistory/recipe when a reference is applied.
- Also store preset ID, resonance ID, nucleus and source ID.
- Do not re-resolve an old recipe against a newer preset database.
- Use proximity only for ordering signal choices after the user selects the observed peak.
- Never auto-assign the closest solvent signal.
