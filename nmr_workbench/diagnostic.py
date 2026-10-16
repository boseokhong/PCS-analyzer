"""Command-line diagnostic for the Bruker-to-domain conversion path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .io import Bruker1DReader


def build_summary(path: str | Path, *, source: str = "auto") -> dict[str, object]:
    """Load a Bruker 1D dataset and return a concise diagnostic summary."""

    dataset = Bruker1DReader().read(path, source=source)
    spectrum = dataset.raw_spectrum or dataset.processed_spectrum
    return {
        "dataset_id": dataset.id,
        "source_path": str(dataset.source_path),
        "vendor": dataset.vendor,
        "state": spectrum.state,
        "points": spectrum.point_count,
        "axis_unit": spectrum.axis_unit,
        "nucleus": dataset.conditions.nucleus,
        "frequency_mhz": dataset.conditions.frequency_mhz,
        "solvent": dataset.conditions.solvent,
        "temperature_k": dataset.conditions.temperature_k,
        "acquisition_parameters": dataset.acquisition_parameters,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Bruker experiment or pdata/N directory")
    parser.add_argument("--source", choices=("auto", "raw", "processed"), default="auto")
    args = parser.parse_args(argv)
    print(json.dumps(build_summary(args.path, source=args.source), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

