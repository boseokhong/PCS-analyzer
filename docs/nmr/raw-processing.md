# Raw Data and Processing

In [NMR Workbench](spectrum-viewer.md), choose **Open Dataset Folder…** or
**Open FID File…**. Keep acquisition data and metadata together.

| Vendor input | Select |
|---|---|
| Bruker raw 1D | Experiment folder containing `fid` and `acqus`, or its `fid`. |
| Bruker processed 1D | Processed-data folder containing `1r` and `procs`. |
| Varian/Agilent raw 1D | `.fid` directory containing `fid` and `procpar`, or its `fid`. |

The loader handles supported **1D** datasets, not arbitrary vendor folders
or multidimensional `ser` data. In a Bruker folder containing both raw and
processed markers, automatic loading prefers raw data. Loading raw data
initializes a default Fourier-transformed spectrum; this is a starting point,
not a guarantee of correct phase, baseline, or referencing.

**Raw FID** shows the acquired time-domain data. **Spectrum** shows the
processed frequency-domain trace, normally in ppm when metadata permits.
**Start Reprocessing** starts again from available raw data. A processed-only
dataset cannot recover the original FID or acquire missing resolution.
Applicable Bruker digital-filter correction uses acquisition metadata.

## Processing sequence

Expand the numbered **Processing** sections as needed. **Preview** displays
pending changes, **Apply** commits them, and **Cancel** discards the preview.
Check the committed/preview indicator and **History** before picking signals.

1. **Apodization:** select a window and parameter. Exponential line broadening
   and Gaussian broadening use Hz. Broadening trades resolution for noise
   suppression; it is not a fitted linewidth.
2. **Zero Filling:** choose the transform size. Added points improve digital
   sampling, not intrinsic acquisition resolution.
3. **Fourier Transform:** use **Create Spectrum** when processing from the time
   domain. Reprocessing preserves the original acquisition.
4. **Phase:** expand **Manual adjustment** to adjust **P0/P1**, or select
   **Peak minima** / **ACME** under **Auto** and choose **Run**. **Pivot → Pick…**
   selects the origin of the phase ramp; **Strongest** uses the strongest
   magnitude signal. Pivot is not a chemical-shift reference. Preview changes
   and choose **Apply**. Inspect positive and negative regions before applying.
   Automatic phasing can fail for broad, overlapped, or unusual spectra.
5. **Baseline:** preview correction and inspect weak/broad signals before
   applying. **Auto Baseline** uses arPLS; advanced alternatives expose their
   parameters. Optional processing dependencies may be needed.
6. **Reference:** select an observed signal, choose **Residual solvent**,
   **Reference standard**, or **Custom**, and check observed/target ppm and
   offset. Preview and apply. Presets require a suitable nucleus and selected
   signal; choosing a solvent does not prove a peak's identity.
7. **Peak Picking:** continue with [Peaks and Assignments](peak-assignment.md).

Applying a reference changes peak positions and associated observable shifts.
Use a chemically justified reference for each sample before calculating shift
differences. Retain vendor metadata and record processing choices; inspect
**Metadata** for nucleus, frequency, solvent, temperature, and acquisition
information when available.

The supplied Yb example opens from
`examples/documentation_demo/Yb_iPr2BA3_demo/PROTON_01.fid`.
The canonical image retains its vendor ppm axis; it does not assert an
independently verified solvent calibration or completed quantitative analysis.

For a reproducible introductory exercise with known phase errors, baseline
artifacts and separate TMS offsets, follow the [FID/CSV tutorial](tutorial-fid-csv.md).
Save committed processing in an [NMR session](sessions.md), rather than relying
on a figure export. [Batch Processing](batch-processing.md) also supports reviewed processing of
multiple experiments; its settings and previews do not commit until explicitly
applied.
