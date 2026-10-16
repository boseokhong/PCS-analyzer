# NMR Workbench dependencies

| Package | Purpose | License | Status |
| --- | --- | --- | --- |
| NumPy | Domain arrays and numerical representation | BSD-3-Clause | Existing runtime requirement |
| nmrglue | Bruker and Varian/Agilent I/O, plus Bruker digital-filter correction | BSD-3-Clause | Runtime requirement for supported vendor I/O and Bruker correction |
| pybaselines | Whittaker-family asymmetric least-squares baseline fitting | BSD-3-Clause | Runtime requirement for Whittaker baseline correction |

PCS Analyzer's own source is distributed under the BSD 3-Clause license in the
root `LICENSE`. The two NMR-specific libraries listed above also use
BSD-3-Clause licenses; their notices must be retained when redistributing them.
This list covers the NMR-specific additions, not a complete binary dependency
or third-party license inventory. Dependency objects are confined to
private adapters; NMR Workbench's public API exposes its own domain objects.

