"""Compatibility bridge for legacy PCS Analyzer NMR shift state.

Older PCS Analyzer projects store observed and diamagnetic shifts directly in
the host state dictionaries instead of as NMR Workbench datasets. D4e keeps
those projects usable by projecting the dictionaries into read-only synthetic
NMR datasets.

The bridge never rewrites the legacy host dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from nmr_workbench.core import ExperimentalConditions, NMRDataset, NMRObservable
from nmr_workbench.ui.view_model import WorkbenchState


LEGACY_PARAMAGNETIC_DATASET_ID = "pcs-project-observed"
LEGACY_DIAMAGNETIC_DATASET_ID = "pcs-project-diamagnetic"
LEGACY_BRIDGE_MARKER = "pcs_analyzer_legacy_project_shifts"


@dataclass(frozen=True)
class LegacyProjectSyncResult:
    paramagnetic_id: str | None
    diamagnetic_id: str | None
    observed_count: int
    diamagnetic_count: int

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(
            value
            for value in (self.paramagnetic_id, self.diamagnetic_id)
            if value is not None
        )

    @property
    def has_pair(self) -> bool:
        return self.paramagnetic_id is not None and self.diamagnetic_id is not None


def is_legacy_project_dataset(dataset: NMRDataset) -> bool:
    metadata = getattr(dataset, "vendor_metadata", {}) or {}
    return bool(metadata.get(LEGACY_BRIDGE_MARKER))


def _normalized_shift_dict(state: dict, key: str) -> dict[int, float]:
    raw = state.get(key, {}) or {}
    result: dict[int, float] = {}
    for ref_id, value in raw.items():
        try:
            result[int(ref_id)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _assignment_labels(state: dict, ref_ids: set[int]) -> dict[int, str]:
    overrides = state.get("ref_label_overrides", {}) or {}
    raw_labels: dict[int, str] = {}
    counts: dict[str, int] = {}

    for ref_id in ref_ids:
        value = overrides.get(ref_id, overrides.get(str(ref_id), ""))
        text = str(value or "").strip()
        if text:
            raw_labels[ref_id] = text
            counts[text.casefold()] = counts.get(text.casefold(), 0) + 1

    labels: dict[int, str] = {}
    for ref_id in ref_ids:
        candidate = raw_labels.get(ref_id)
        if candidate and counts.get(candidate.casefold(), 0) == 1:
            labels[ref_id] = candidate
        else:
            labels[ref_id] = str(ref_id)
    return labels


def _build_dataset(
    *,
    dataset_id: str,
    display_name: str,
    role: str,
    values: dict[int, float],
    assignments: dict[int, str],
) -> NMRDataset:
    conditions = ExperimentalConditions(nucleus="Legacy shifts")
    dataset = NMRDataset(
        conditions=conditions,
        vendor="PCS Analyzer project",
        id=dataset_id,
        experimental_role=role,
        vendor_metadata={
            LEGACY_BRIDGE_MARKER: True,
            "display_name": display_name,
            "read_only_shift_bridge": True,
        },
    )
    dataset.observables = [
        NMRObservable(
            nucleus="Legacy shifts",
            chemical_shift_ppm=float(values[ref_id]),
            assignment=assignments[ref_id],
            dataset_id=dataset.id,
        )
        for ref_id in sorted(values)
    ]
    return dataset


def sync_legacy_project_datasets(
    model: WorkbenchState,
    state: dict | None,
) -> LegacyProjectSyncResult:
    """Refresh synthetic Workbench datasets from legacy host shift dictionaries.

    Existing real NMR datasets are preserved. Synthetic datasets are replaced
    atomically so repeated calls never duplicate them.
    """
    if not isinstance(state, dict):
        return LegacyProjectSyncResult(None, None, 0, 0)

    observed = _normalized_shift_dict(state, "delta_obs_values")
    diamagnetic = _normalized_shift_dict(state, "delta_dia_values")
    all_ref_ids = set(observed) | set(diamagnetic)
    assignments = _assignment_labels(state, all_ref_ids)

    real_datasets = [
        dataset
        for dataset in model.datasets
        if not is_legacy_project_dataset(dataset)
    ]

    synthetic: list[NMRDataset] = []
    paramagnetic_id = None
    diamagnetic_id = None

    if observed:
        synthetic.append(
            _build_dataset(
                dataset_id=LEGACY_PARAMAGNETIC_DATASET_ID,
                display_name="PCS project observed shifts",
                role="paramagnetic",
                values=observed,
                assignments=assignments,
            )
        )
        paramagnetic_id = LEGACY_PARAMAGNETIC_DATASET_ID

    if diamagnetic:
        synthetic.append(
            _build_dataset(
                dataset_id=LEGACY_DIAMAGNETIC_DATASET_ID,
                display_name="PCS project diamagnetic shifts",
                role="diamagnetic",
                values=diamagnetic,
                assignments=assignments,
            )
        )
        diamagnetic_id = LEGACY_DIAMAGNETIC_DATASET_ID

    selected_before = model.selected_dataset_id
    model.datasets = real_datasets + synthetic

    surviving_ids = {dataset.id for dataset in model.datasets}
    if selected_before in surviving_ids:
        model.selected_dataset_id = selected_before
    elif paramagnetic_id is not None:
        model.selected_dataset_id = paramagnetic_id
    elif diamagnetic_id is not None:
        model.selected_dataset_id = diamagnetic_id
    elif model.datasets:
        model.selected_dataset_id = model.datasets[0].id
    else:
        model.selected_dataset_id = None

    return LegacyProjectSyncResult(
        paramagnetic_id=paramagnetic_id,
        diamagnetic_id=diamagnetic_id,
        observed_count=len(observed),
        diamagnetic_count=len(diamagnetic),
    )
