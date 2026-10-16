# Yb(*i*Pr₂BA)₃ NMR documentation example

`PROTON_01.fid/` contains a measured Varian/Agilent proton acquisition:
`fid` is the time-domain signal and `procpar` provides acquisition metadata.
Non-scientific identifiers and acquisition-system paths in `procpar` have been
anonymized for public distribution. The raw FID and scientific acquisition
parameters are unchanged; the processing pipeline was checked for numerical parity.
Keep both files together and open the directory with **NMR Workbench →
Open Dataset Folder…**, or select its `fid` with **Open FID File…**.

The metadata identifies ¹H, approximately 399.9 MHz, benzene-*d*₆, and 298.15 K.
The dataset is used for documentation and reproducible screenshots of raw
processing, peak picking, and assignment controls.

The reviewed screenshot recipe uses 2 Hz exponential broadening, Fourier
transformation, manual zero-/first-order phase adjustment, and arPLS baseline
correction before picking positive peaks. Automatic processing alone was not
accepted as the final display. The vendor ppm reference is retained; an
independent calibration and molecular assignments are not established by this
image. Demo A/B are illustrative annotations only.

This standalone Yb NMR session is separate from the Nd/La PCS documentation
examples. No Yb observations are transferred to an Nd structure. Processing
is transient; vendor files and the Nd `.pcsp` are never rewritten. See the
[FID processing tutorial](../../../docs/nmr/tutorial-fid-csv.md).
