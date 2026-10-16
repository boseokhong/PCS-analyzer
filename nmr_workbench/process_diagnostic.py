"""Process a real Bruker 1D FID through the non-GUI NMR pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .io import Bruker1DReader
from .processing import ProcessingResult, SpectrumProcessor


def process_bruker_fid(
    path: str | Path,
    *,
    correct_digital_filter: bool = True,
    lb_hz: float = 0.3,
    target_points: int | None = None,
) -> ProcessingResult:
    """Load and minimally process a real Bruker FID for diagnostic validation."""

    dataset = Bruker1DReader().read(path, source="raw")
    processor = SpectrumProcessor.from_dataset(dataset)
    source = dataset.raw_spectrum
    if source is None:
        raise RuntimeError("Bruker reader did not return a raw spectrum")
    if correct_digital_filter:
        result = processor.correct_digital_filter(source)
        source = result
    result = processor.apodize(source, method="exponential", lb_hz=lb_hz)
    if target_points is not None:
        result = processor.zero_fill(result, target_points=target_points)
    return processor.fourier_transform(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Bruker experiment directory containing fid")
    parser.add_argument("--skip-digital-filter", action="store_true")
    parser.add_argument("--lb-hz", type=float, default=0.3)
    parser.add_argument("--target-points", type=int)
    args = parser.parse_args(argv)
    result = process_bruker_fid(
        args.path,
        correct_digital_filter=not args.skip_digital_filter,
        lb_hz=args.lb_hz,
        target_points=args.target_points,
    )
    summary = {
        "points": result.spectrum.point_count,
        "axis_unit": result.spectrum.axis_unit,
        "axis_start": float(result.spectrum.axis[0]),
        "axis_end": float(result.spectrum.axis[-1]),
        "history": result.history.to_list(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
