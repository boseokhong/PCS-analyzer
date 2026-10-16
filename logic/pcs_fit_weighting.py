"""Standard PCS fit residual-weighting policy."""

from __future__ import annotations

from enum import Enum


class FitWeightingMode(str, Enum):
    UNWEIGHTED = "unweighted"
    EXPERIMENTAL_UNCERTAINTY = "experimental_uncertainty"


_LABELS = {
    FitWeightingMode.UNWEIGHTED: "Unweighted",
    FitWeightingMode.EXPERIMENTAL_UNCERTAINTY: "Experimental uncertainty",
}


def parse_fit_weighting_mode(value) -> FitWeightingMode:
    if isinstance(value, FitWeightingMode):
        return value
    text = str(value or "").strip().lower().replace(" ", "_")
    aliases = {
        "": FitWeightingMode.UNWEIGHTED,
        "unweighted": FitWeightingMode.UNWEIGHTED,
        "experimental_uncertainty": FitWeightingMode.EXPERIMENTAL_UNCERTAINTY,
    }
    if text not in aliases:
        raise ValueError(f"unsupported standard-fit weighting mode {value!r}")
    return aliases[text]


def fit_weighting_label(value) -> str:
    return _LABELS[parse_fit_weighting_mode(value)]


__all__ = [
    "FitWeightingMode",
    "fit_weighting_label",
    "parse_fit_weighting_mode",
]
