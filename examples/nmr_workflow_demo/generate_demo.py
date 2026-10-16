"""Reproduce the synthetic teaching data; does not change Workbench code.

Requires the application's NumPy and nmrglue dependencies.
Run this file with --overwrite to regenerate only this example's data files.
"""

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import nmrglue as ng


ROOT = Path(__file__).resolve().parent
POINTS = 16384
SW_HZ = 6400.0
FREQUENCY_MHZ = 400.0
CARRIER_PPM = 6.0
TEMPERATURE_K = 298.15
PHASE_DEG = 25.0
TMS_AMPLITUDE = 0.015
TMS_LINEWIDTH_HZ = 3.0
BASELINE_FRACTION = 0.08
EXPERIMENTS = (
    ("Paramagnetic", "paramagnetic", (2.0, 5.0, 8.0, 11.0), 12.0, 0.18),
    ("Diamagnetic", "diamagnetic", (1.2, 3.4, 5.6, 7.8), 6.0, -0.12),
)
AMPLITUDES = (1.0, 0.85, 0.7, 0.6)
STRUCTURE = """9
SYNTHETIC UI teaching cluster; Angstrom; Nd center Ref 1; H1-H4 map to Refs 6-9; not an optimized molecule
Nd  0.000000  0.000000  0.000000
O   2.400000  0.000000  0.000000
O  -1.200000  2.078461  0.000000
O  -1.200000 -2.078461  0.000000
O   0.000000  0.000000  2.400000
H   3.370000  0.000000  0.000000
H  -1.685000  2.918506  0.000000
H  -1.685000 -2.918506  0.000000
H   0.000000  0.000000  3.370000
"""


def write_example(name, role, shifts, linewidth_hz, reference_error_ppm, *, overwrite=False):
    directory = ROOT / "FID" / f"{name}.fid"
    csv_path = ROOT / "CSV" / f"{name}.csv"
    outputs = (directory / "fid", directory / "procpar", directory / "text", csv_path)
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"{name} already exists; use --overwrite to regenerate")
    directory.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # VNMR uses the positive-exponent DFT in the existing SpectrumProcessor.
    # Each causal exponentially decaying oscillation yields one Lorentzian.
    time_s = np.arange(POINTS, dtype=float) / SW_HZ
    fid = sum(amplitude * np.exp(-np.pi * linewidth_hz * time_s)
              * np.exp(-2j * np.pi * (shift + reference_error_ppm - CARRIER_PPM) * FREQUENCY_MHZ * time_s)
              for shift, amplitude in zip(shifts, AMPLITUDES))
    # The reference shares the sample's ppm offset and should be set to 0 ppm.
    fid += TMS_AMPLITUDE * np.exp(-np.pi * TMS_LINEWIDTH_HZ * time_s) * np.exp(
        -2j * np.pi * (reference_error_ppm - CARRIER_PPM) * FREQUENCY_MHZ * time_s)
    # Half-weight t=0 removes the discrete half-sample real baseline pedestal.
    fid[0] *= 0.5
    fid *= 10000.0
    # Deliberately add a smooth real baseline artifact in the unphased spectrum.
    # This numerical teaching artifact is not a model of instrument electronics.
    spectrum = np.fft.fftshift(np.fft.ifft(fid)) * POINTS
    x = np.linspace(-1.0, 1.0, POINTS)
    baseline = BASELINE_FRACTION * float(spectrum.real.max()) * (
        1.0 + 0.6 * x + 0.8 * x**2 + 0.3 * np.cos(np.pi * x))
    fid += np.fft.fft(np.fft.ifftshift(baseline)) / POINTS
    fid *= np.exp(1j * np.deg2rad(PHASE_DEG))

    udic = ng.fileiobase.create_blank_udic(1)
    udic[0].update(size=POINTS, complex=True, time=True, freq=False,
                   sw=SW_HZ, obs=FREQUENCY_MHZ, car=CARRIER_PPM * FREQUENCY_MHZ,
                   label="1H")
    metadata = ng.varian.create_dic(udic)
    metadata["S_COMPLEX"] = 1
    parameters = {
        "acqdim": 1, "sw": SW_HZ, "sfrq": FREQUENCY_MHZ,
        "reffrq": FREQUENCY_MHZ, "rfl": SW_HZ / 2,
        "rfp": CARRIER_PPM * FREQUENCY_MHZ, "tof": 0,
        "temp": TEMPERATURE_K - 273.15, "at": POINTS / SW_HZ,
        "nt": 1, "rp": 0, "lp": 0,
        "tn": "H1", "solvent": "synthetic", "seqfil": "synthetic_tms_baseline_demo",
    }
    for key, value in parameters.items():
        parameter = ng.varian.create_pdic_param(key, [str(value)])
        if isinstance(value, str):
            parameter["basictype"] = "2"
        metadata["procpar"][key] = parameter
    ng.varian.write(str(directory), metadata, fid.astype(np.complex64),
                    repack=True, overwrite=overwrite)
    (directory / "text").write_text(
        f"SYNTHETIC TEACHING DATA — {name}\n"
        "Not an experimental acquisition or structural assignment.\n"
        f"4 Lorentzian resonances; shifts={shifts} ppm; T={TEMPERATURE_K} K.\n"
        f"Plus small TMS, true shift 0 ppm; apparent TMS={reference_error_ppm:+.2f} ppm.\n"
        "All sample peaks share this reference error; reference TMS separately in each dataset.\n"
        f"Smooth baseline artifact scale={BASELINE_FRACTION} of the strongest unphased peak.\n"
        f"Applied phase=+{PHASE_DEG} degrees; correct once with P0=-{PHASE_DEG}, P1=0.\n"
        "H1-H4 are illustrative correspondence labels only.\n", encoding="utf-8")
    evidence = json.dumps({"origin": "synthetic_demo", "synthetic": True,
                           "generator": "nmr_workflow_demo/generate_demo.py"})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("assignment", "shift_ppm", "experiment", "role", "nucleus",
                         "temperature_k", "quantity", "included", "provenance"))
        for index, shift in enumerate(shifts, 1):
            writer.writerow((f"H{index}", f"{shift:.2f}", name, role, "1H",
                             TEMPERATURE_K, "observed_shift", "true", evidence))


def main():
    global ROOT
    from tools.demo_support import generation_args, prepare_output
    args = generation_args(ROOT, __doc__)
    ROOT = args.output.resolve()
    # Check all named destinations before writing either experiment.
    names = ["structure.xyz"]
    for name, *_ in EXPERIMENTS:
        names += [f"CSV/{name}.csv", *[f"FID/{name}.fid/{file}" for file in ("fid", "procpar", "text")]]
    ROOT = prepare_output(ROOT, names, overwrite=args.overwrite)
    for experiment in EXPERIMENTS:
        write_example(*experiment, overwrite=args.overwrite)
    (ROOT / "structure.xyz").write_text(STRUCTURE, encoding="utf-8")
    print(f"Created synthetic FID and CSV examples in {ROOT}")


if __name__ == "__main__":
    main()
