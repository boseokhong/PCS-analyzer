"""Generate synthetic VT teaching FIDs and a ground-truth CSV.

Uses the application's existing NumPy/nmrglue dependencies. No datasets,
tracks, assignments, sessions or fitted results are pre-created in Workbench.
Run with --overwrite to replace only this example's generated files.
"""

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nmrglue as ng
import numpy as np


ROOT = Path(__file__).resolve().parent
POINTS = 16384
SW_HZ = 6400.0
FREQUENCY_MHZ = 400.0
CARRIER_PPM = 6.0
PHASE_DEG = 25.0
TMS_AMPLITUDE = 0.015
TMS_LINEWIDTH_HZ = 3.0
BASELINE_FRACTION = 0.08
AMPLITUDES = (1.0, 0.85, 0.7, 0.6)
OFFSETS_PPM = (1.2, 3.4, 5.6, 7.8)
COEFFICIENTS_PPM_K = (238.4, 476.8, 715.2, 953.6)
# Prefixes make folder-group import order visibly different from temperature order.
ACQUISITIONS = ((298, .18), (278, -.14), (318, .22), (288, .07), (308, -.09))
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


def shifts_at(temperature_k):
    return tuple(offset + coefficient / temperature_k
                 for offset, coefficient in zip(OFFSETS_PPM, COEFFICIENTS_PPM_K))


def experiments():
    for order, (temperature, error) in enumerate(ACQUISITIONS, 1):
        name = f"{order:02d}_Paramagnetic_{temperature}K"
        yield name, "paramagnetic", temperature, error, shifts_at(temperature), 12.0, order, "Paramagnetic"
    yield "Diamagnetic_298K", "diamagnetic", 298, -.12, OFFSETS_PPM, 6.0, None, "Reference"


def write_fid(directory, name, temperature, reference_error, shifts, linewidth, *, overwrite=False):
    directory.mkdir(parents=True, exist_ok=True)
    time_s = np.arange(POINTS, dtype=float) / SW_HZ
    fid = sum(amplitude * np.exp(-np.pi * linewidth * time_s)
              * np.exp(-2j * np.pi * (shift + reference_error - CARRIER_PPM)
                       * FREQUENCY_MHZ * time_s)
              for shift, amplitude in zip(shifts, AMPLITUDES))
    fid += TMS_AMPLITUDE * np.exp(-np.pi * TMS_LINEWIDTH_HZ * time_s) * np.exp(
        -2j * np.pi * (reference_error - CARRIER_PPM) * FREQUENCY_MHZ * time_s)
    fid[0] *= .5
    fid *= 10000.0
    spectrum = np.fft.fftshift(np.fft.ifft(fid)) * POINTS
    x = np.linspace(-1., 1., POINTS)
    baseline = BASELINE_FRACTION * float(spectrum.real.max()) * (
        1. + .6 * x + .8 * x**2 + .3 * np.cos(np.pi * x))
    # Artificial broad real baseline, expressed through the inverse VNMR DFT.
    fid += np.fft.fft(np.fft.ifftshift(baseline)) / POINTS
    fid *= np.exp(1j * np.deg2rad(PHASE_DEG))

    udic = ng.fileiobase.create_blank_udic(1)
    udic[0].update(size=POINTS, complex=True, time=True, freq=False,
                   sw=SW_HZ, obs=FREQUENCY_MHZ, car=CARRIER_PPM * FREQUENCY_MHZ, label="1H")
    metadata = ng.varian.create_dic(udic)
    metadata["S_COMPLEX"] = 1
    parameters = {"acqdim": 1, "sw": SW_HZ, "sfrq": FREQUENCY_MHZ,
                  "reffrq": FREQUENCY_MHZ, "rfl": SW_HZ / 2,
                  "rfp": CARRIER_PPM * FREQUENCY_MHZ, "tof": 0,
                  "temp": temperature - 273.15, "at": POINTS / SW_HZ,
                  "nt": 1, "rp": 0, "lp": 0, "tn": "H1",
                  "solvent": "synthetic", "seqfil": "synthetic_vt_curie_demo"}
    for key, value in parameters.items():
        parameter = ng.varian.create_pdic_param(key, [str(value)])
        if isinstance(value, str):
            parameter["basictype"] = "2"
        metadata["procpar"][key] = parameter
    ng.varian.write(str(directory), metadata, fid.astype(np.complex64),
                    repack=True, overwrite=overwrite)
    (directory / "text").write_text(
        f"SYNTHETIC VT TEACHING DATA — {name}\n"
        f"1H at {temperature} K. Four sample signals plus small TMS; not measured data.\n"
        f"True sample shifts={shifts} ppm. Apparent TMS={reference_error:+.2f} ppm.\n"
        "Phase once with P0=-25 degrees, P1=0; baseline arPLS lambda=1e5.\n"
        "Reference each dataset to its own TMS at 0 ppm before picking/tracking.\n"
        "Sample law: delta=offset+C/T; no PCS/contact or structural interpretation.\n",
        encoding="utf-8")


def main():
    global ROOT
    from tools.demo_support import generation_args, prepare_output
    args = generation_args(ROOT, __doc__)
    ROOT = args.output.resolve()
    rows = tuple(experiments())
    outputs = [ROOT / "CSV" / "Expected_shifts.csv", ROOT / "structure.xyz"]
    outputs += [ROOT / "FID" / group / f"{name}.fid" / file
                for name, _, _, _, _, _, _, group in rows for file in ("fid", "procpar", "text")]
    ROOT = prepare_output(ROOT, [path.relative_to(ROOT) for path in outputs], overwrite=args.overwrite)
    records = []
    for name, role, temperature, error, shifts, linewidth, order, group in rows:
        write_fid(ROOT / "FID" / group / f"{name}.fid", name, temperature, error,
                  shifts, linewidth, overwrite=args.overwrite)
        evidence = json.dumps({"origin": "synthetic_vt_demo", "synthetic": True,
                               "generator": "nmr_vt_demo/generate_demo.py"})
        for index, shift in enumerate(shifts):
            records.append((f"H{index+1}", f"{shift:.12f}", name, role, "1H", temperature,
                            "observed_shift", "true", order if order is not None else "",
                            OFFSETS_PPM[index], COEFFICIENTS_PPM_K[index] if role == "paramagnetic" else "",
                            error, 6+index, evidence))
    (ROOT / "CSV").mkdir(parents=True, exist_ok=True)
    with (ROOT / "CSV" / "Expected_shifts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("assignment", "shift_ppm", "experiment", "role", "nucleus", "temperature_k",
                         "quantity", "included", "demo_acquisition_order", "model_offset_ppm",
                         "model_C_ppm_K", "apparent_tms_ppm", "teaching_structure_ref", "provenance"))
        writer.writerows(records)
    (ROOT / "structure.xyz").write_text(STRUCTURE, encoding="utf-8")
    print(f"Created five Para FIDs, one 298 K Dia reference, expected values and XYZ in {ROOT}")


if __name__ == "__main__":
    main()
