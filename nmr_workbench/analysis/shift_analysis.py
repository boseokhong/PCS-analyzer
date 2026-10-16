"""Pair paramagnetic and diamagnetic NMR observables by assignment."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import math


class ShiftAnalysisError(ValueError):
    """Raised when datasets cannot form a valid shift-comparison pair."""


@dataclass(frozen=True)
class ShiftComparisonRow:
    assignment: str
    paramagnetic_ppm: float | None
    diamagnetic_ppm: float | None
    delta_para_ppm: float | None
    status: str
    paramagnetic_observable_id: str | None = None
    diamagnetic_observable_id: str | None = None
    paramagnetic_dataset_id: str | None = None
    diamagnetic_dataset_id: str | None = None


@dataclass(frozen=True)
class ShiftComparisonSummary:
    total: int
    matched: int
    missing_paramagnetic: int
    missing_diamagnetic: int
    ambiguous: int


def normalize_nucleus(value: str | None) -> str | None:
    if value is None:
        return None
    text = (
        str(value).strip().upper()
        .replace("^", "").replace("{", "").replace("}", "")
        .replace("-", "").replace("_", "").replace(" ", "")
    )
    aliases = {
        "H1": "1H",
        "1H": "1H",
        "PROTON": "1H",
        "C13": "13C",
        "13C": "13C",
        "CARBON13": "13C",
    }
    return aliases.get(text, text or None)


def dataset_nucleus(dataset) -> str | None:
    return normalize_nucleus(
        getattr(getattr(dataset, "conditions", None), "nucleus", None)
    )


def roles_are_pairable(first, second) -> bool:
    roles = {
        getattr(first, "experimental_role", "unspecified"),
        getattr(second, "experimental_role", "unspecified"),
    }
    return roles == {"paramagnetic", "diamagnetic"}


def validate_dataset_pair(paramagnetic, diamagnetic) -> None:
    if paramagnetic is diamagnetic or paramagnetic.id == diamagnetic.id:
        raise ShiftAnalysisError("A dataset cannot be paired with itself.")

    if getattr(paramagnetic, "experimental_role", "unspecified") != "paramagnetic":
        raise ShiftAnalysisError("The paramagnetic dataset is not marked Paramagnetic.")
    if getattr(diamagnetic, "experimental_role", "unspecified") != "diamagnetic":
        raise ShiftAnalysisError("The diamagnetic dataset is not marked Diamagnetic.")

    para_nucleus = dataset_nucleus(paramagnetic)
    dia_nucleus = dataset_nucleus(diamagnetic)
    if para_nucleus is None or dia_nucleus is None:
        raise ShiftAnalysisError("Both datasets must have a nucleus before pairing.")
    if para_nucleus != dia_nucleus:
        raise ShiftAnalysisError(
            f"Dataset nuclei do not match ({para_nucleus} vs {dia_nucleus})."
        )


def _assignment_key(value: str | None) -> str | None:
    text = (value or "").strip()
    return text.casefold() if text else None


def _group_observables(dataset):
    grouped = defaultdict(list)
    for observable in dataset.observables:
        if not observable.included:
            continue
        try:
            shift = float(observable.chemical_shift_ppm)
        except (TypeError, ValueError) as exc:
            raise ShiftAnalysisError("Included observables must have finite chemical shifts.") from exc
        if not math.isfinite(shift):
            raise ShiftAnalysisError("Included observables must have finite chemical shifts.")
        if normalize_nucleus(observable.nucleus) != dataset_nucleus(dataset):
            raise ShiftAnalysisError("An included observation's nucleus differs from its experiment. Review it in NMR Data.")
        if observable.source_peak_id:
            peak = next((p for p in dataset.peaks if p.id == observable.source_peak_id), None)
            try:
                position = float(peak.position_ppm) if peak is not None else float("nan")
            except (TypeError, ValueError):
                position = float("nan")
            if not math.isfinite(position) or not math.isclose(position, shift, rel_tol=0, abs_tol=1e-9):
                raise ShiftAnalysisError("An included observation has a missing/changed source peak. Review it in NMR Data.")
        key = _assignment_key(observable.assignment)
        if key is not None:
            grouped[key].append(observable)
    return grouped


def build_shift_comparison(paramagnetic, diamagnetic):
    """Return assignment-matched rows and summary.

    δpara is defined here strictly as:
        δpara = δ(paramagnetic observed) - δ(diamagnetic)

    No claim is made that δpara is identical to PCS.  PCS preparation is a
    separate downstream workflow because FCS/contact-shift corrections may be
    required.
    """
    validate_dataset_pair(paramagnetic, diamagnetic)

    para = _group_observables(paramagnetic)
    dia = _group_observables(diamagnetic)
    keys = sorted(set(para) | set(dia))

    rows: list[ShiftComparisonRow] = []
    for key in keys:
        p_items = para.get(key, [])
        d_items = dia.get(key, [])

        display = (
            (p_items[0].assignment if p_items else None)
            or (d_items[0].assignment if d_items else None)
            or key
        ).strip()

        if len(p_items) > 1 or len(d_items) > 1:
            rows.append(
                ShiftComparisonRow(
                    assignment=display,
                    paramagnetic_ppm=None,
                    diamagnetic_ppm=None,
                    delta_para_ppm=None,
                    status="Ambiguous assignment",
                    paramagnetic_dataset_id=paramagnetic.id,
                    diamagnetic_dataset_id=diamagnetic.id,
                )
            )
            continue

        if not p_items:
            d = d_items[0]
            rows.append(
                ShiftComparisonRow(
                    assignment=display,
                    paramagnetic_ppm=None,
                    diamagnetic_ppm=float(d.chemical_shift_ppm),
                    delta_para_ppm=None,
                    status="Missing paramagnetic",
                    diamagnetic_observable_id=d.id,
                    paramagnetic_dataset_id=paramagnetic.id,
                    diamagnetic_dataset_id=diamagnetic.id,
                )
            )
            continue

        if not d_items:
            p = p_items[0]
            rows.append(
                ShiftComparisonRow(
                    assignment=display,
                    paramagnetic_ppm=float(p.chemical_shift_ppm),
                    diamagnetic_ppm=None,
                    delta_para_ppm=None,
                    status="Missing diamagnetic",
                    paramagnetic_observable_id=p.id,
                    paramagnetic_dataset_id=paramagnetic.id,
                    diamagnetic_dataset_id=diamagnetic.id,
                )
            )
            continue

        p = p_items[0]
        d = d_items[0]
        p_ppm = float(p.chemical_shift_ppm)
        d_ppm = float(d.chemical_shift_ppm)
        rows.append(
            ShiftComparisonRow(
                assignment=display,
                paramagnetic_ppm=p_ppm,
                diamagnetic_ppm=d_ppm,
                delta_para_ppm=p_ppm - d_ppm,
                status="Matched",
                paramagnetic_observable_id=p.id,
                diamagnetic_observable_id=d.id,
                paramagnetic_dataset_id=paramagnetic.id,
                diamagnetic_dataset_id=diamagnetic.id,
            )
        )

    summary = ShiftComparisonSummary(
        total=len(rows),
        matched=sum(row.status == "Matched" for row in rows),
        missing_paramagnetic=sum(row.status == "Missing paramagnetic" for row in rows),
        missing_diamagnetic=sum(row.status == "Missing diamagnetic" for row in rows),
        ambiguous=sum(row.status == "Ambiguous assignment" for row in rows),
    )
    return rows, summary
