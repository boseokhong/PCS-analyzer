"""UI-independent VT-series management and explicit definition-file persistence."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable

from nmr_workbench.core import NMRDataset, NMRSeries, resolve_temperature
from nmr_workbench.io import load_nmr_dataset
from nmr_workbench.io.experiment_import import BatchLoadResult, discover_experiments, load_experiments
from .view_model import WorkbenchState


@dataclass(frozen=True)
class SeriesImportResult:
    added_member_ids: tuple[str, ...]
    new_dataset_ids: tuple[str, ...]
    failures: tuple[tuple[str, str], ...]
    skipped: tuple[str, ...]


def _source_key(path) -> str | None:
    if path is None:
        return None
    return os.path.normcase(str(Path(path).resolve()))


class SeriesManager:
    """Operate on Workbench-owned datasets and series without Tk or PCS state."""

    def __init__(self, state: WorkbenchState):
        self.state = state
        self.restoration_diagnostics = ()

    def get_series(self, series_id: str) -> NMRSeries:
        for series in self.state.series:
            if series.id == series_id:
                return series
        raise KeyError(f"unknown series: {series_id}")

    @property
    def active_series(self) -> NMRSeries | None:
        return next((s for s in self.state.series if s.id == self.state.selected_series_id), None)

    def create_series(self, name: str) -> NMRSeries:
        name = self._name(name)
        series = NMRSeries(name=name)
        self.state.series.append(series)
        self.select_series(series.id)
        return series

    @staticmethod
    def _name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Series name must not be blank.")
        return name.strip()

    def rename_series(self, series_id: str, name: str) -> None:
        self.get_series(series_id).name = self._name(name)

    def delete_series(self, series_id: str) -> None:
        series = self.get_series(series_id)
        self.state.series.remove(series)
        if self.state.selected_series_id == series_id:
            self.state.selected_series_id = self.state.series[0].id if self.state.series else None
            self.state._sync_series_member()

    def select_series(self, series_id: str) -> NMRSeries:
        series = self.get_series(series_id)
        self.state.selected_series_id = series.id
        self.state._sync_series_member()
        return series

    def select_member(self, series_id: str, member_id: str):
        series = self.get_series(series_id)
        member = next((m for m in series.members if m.id == member_id), None)
        if member is None:
            raise KeyError(f"unknown member: {member_id}")
        self.state.selected_series_id = series_id
        self.state.select_dataset(member.dataset_id)
        return member

    def add_dataset(self, series_id: str, dataset_id: str):
        dataset = next((d for d in self.state.datasets if d.id == dataset_id), None)
        if dataset is None:
            raise KeyError(f"dataset is not loaded: {dataset_id}")
        self._require_experiment(dataset)
        return self.get_series(series_id).add_member(dataset)

    @staticmethod
    def _require_experiment(dataset):
        from nmr_workbench.integration.legacy_project import is_legacy_project_dataset
        if is_legacy_project_dataset(dataset):
            raise ValueError("Legacy PCS shift projections are not acquired experiments; import a spectrum instead.")

    def remove_member(self, series_id: str, member_id: str) -> None:
        self.get_series(series_id).remove_member(member_id)
        if self.state.selected_series_id == series_id:
            self.state._sync_series_member()

    def edit_member(self, series_id: str, member_id: str, *, label: str | None,
                    included: bool, temperature_override_k: float | None):
        if temperature_override_k is not None:
            if resolve_temperature(None, override_k=temperature_override_k).status != "valid":
                raise ValueError("Temperature override must be finite positive Kelvin, or blank.")
        series = self.get_series(series_id)
        member = next((m for m in series.members if m.id == member_id), None)
        if member is None:
            raise KeyError(f"unknown member: {member_id}")
        updated = replace(member, label=label, included=included,
                          temperature_override_k=temperature_override_k)
        series.replace_member(updated)
        return updated

    def members(self, series_id: str, order: str = "Acquisition"):
        series = self.get_series(series_id)
        if order == "Acquisition":
            return series.members
        if order not in {"Temperature ↑", "Temperature ↓"}:
            raise ValueError("Unknown member ordering.")
        return series.temperature_sorted_members(descending=order == "Temperature ↓")

    def apply_import(self, series_id: str, result: BatchLoadResult) -> SeriesImportResult:
        series = self.get_series(series_id)
        added, new, skipped = [], [], list(result.skipped)
        for path, loaded in result.loaded:
            existing = next((d for d in self.state.datasets if d.id == loaded.id), None)
            if existing is None and loaded.source_path is not None:
                matches = [d for d in self.state.datasets if _source_key(d.source_path) == _source_key(loaded.source_path)]
                # Multiple independent loads of one source must not be guessed.
                if len(matches) > 1:
                    skipped.append(f"{path}: multiple loaded datasets use this source; add one explicitly")
                    continue
                existing = matches[0] if matches else None
            dataset = existing if existing is not None else loaded
            if any(m.dataset_id == dataset.id for m in series.members):
                skipped.append(f"{path}: already in this series")
                continue
            member = series.add_member(dataset)
            if existing is None:
                self.state.datasets.append(dataset)
                new.append(dataset.id)
            added.append(member.id)
        return SeriesImportResult(tuple(added), tuple(new), result.failures, tuple(skipped))

    def import_paths(self, series_id: str, paths, *, loader: Callable = load_nmr_dataset):
        self.get_series(series_id)
        return self.apply_import(series_id, load_experiments(paths, loader=loader))

    def definitions_payload(self) -> dict:
        self.state.update_tracks()
        return {"format": "nmr_series_definitions", "schema_version": 2,
                "series": [s.to_dict() for s in self.state.series]}

    def restore_definitions(self, payload: dict) -> tuple[NMRSeries, ...]:
        """Append validated definitions atomically, resolving only loaded datasets.

        Stable dataset ID wins. A missing ID may resolve to one loaded absolute
        source path; that dataset's existing ID is retained. No files are loaded.
        """
        if not isinstance(payload, dict) or payload.get("format") != "nmr_series_definitions":
            raise ValueError("Not an NMR series definitions file.")
        if type(payload.get("schema_version")) is not int or payload["schema_version"] not in {1, 2}:
            raise ValueError("Unsupported series definitions schema version.")
        rows = payload.get("series")
        if not isinstance(rows, list):
            raise ValueError("series must be an array")
        datasets = {d.id: d for d in self.state.datasets}
        restored = []
        used_ids = {s.id for s in self.state.series}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("members"), list):
                raise ValueError("Invalid series definition.")
            # Copy only plain payload values; domain datasets are never cloned.
            row = dict(row, members=[dict(m) if isinstance(m, dict) else m for m in row["members"]])
            for member in row["members"]:
                if not isinstance(member, dict):
                    raise ValueError("Invalid member definition.")
                if not isinstance(member.get("dataset_id"), str) or not member["dataset_id"].strip():
                    raise ValueError("Invalid member dataset ID.")
                if member.get("dataset_id") not in datasets:
                    source = member.get("source_path")
                    matches = [d for d in self.state.datasets if _source_key(d.source_path) == _source_key(source)] if isinstance(source, str) and Path(source).is_absolute() else []
                    if len(matches) != 1:
                        raise ValueError(f"Load the source dataset first (unique match required): "
                            f"series={row.get('id')}, member={member.get('id')}, "
                            f"dataset={member['dataset_id']}, source={source!r}; "
                            f"{len(matches)} loaded source matches")
                    member["dataset_id"] = matches[0].id
            series = NMRSeries.from_dict(row, datasets)
            for member in series.members:
                self._require_experiment(member.dataset)
            if series.id in used_ids:
                raise ValueError(f"Series is already loaded: {series.name}")
            used_ids.add(series.id)
            restored.append(series)
        self.state.series.extend(restored)
        self.restoration_diagnostics = tuple(
            f"series={series.id}, track={track.id}, member={point.member_id}, "
            f"dataset={point.dataset_id}, peak={point.peak_id}: "
            + ", ".join(dict.fromkeys(([point.reason] if point.reason else []) +
                [issue.code for issue in track.validate(series) if issue.member_id == point.member_id]))
            for series in restored for track in series.tracks for point in track.points
            if point.state == "stale")
        from nmr_workbench.analysis.vt_observables import association_diagnostics
        self.restoration_diagnostics += tuple(
            f"series={series.id}, dataset={dataset_id}: {code}: {message}"
            for series in restored for code, dataset_id, message in association_diagnostics(series))
        if restored:
            self.select_series(restored[0].id)
        return tuple(restored)

    def save_definitions(self, path: str | Path) -> None:
        text = json.dumps(self.definitions_payload(), indent=2, ensure_ascii=False, allow_nan=False)
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             delete=False, suffix=".tmp") as handle:
                temporary = handle.name
                handle.write(text)
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def load_definitions(self, path: str | Path):
        with Path(path).open(encoding="utf-8") as handle:
            return self.restore_definitions(json.load(handle))

    def save_session(self, path, *, processing_session=None):
        from nmr_workbench.io.session import save_session
        from nmr_workbench.integration.legacy_project import is_legacy_project_dataset
        self.state.update_tracks()
        owned = [d for d in self.state.datasets if not is_legacy_project_dataset(d)]
        save_session(path, owned, self.state.series, processing_session=processing_session,
                     batch_presets=self.state.batch_presets)
        return tuple(f"Host-owned PCS shift projection {d.id} was not included; its state belongs to the PCS document."
                     for d in self.state.datasets if is_legacy_project_dataset(d))

    def load_session(self, path, *, source_bindings=None):
        from nmr_workbench.io.session import load_session
        result = load_session(path, source_bindings=source_bindings)
        for dataset in result.datasets:
            self._require_experiment(dataset)
        if {d.id for d in result.datasets} & {d.id for d in self.state.datasets}:
            raise ValueError("Session dataset IDs are already loaded; open into an empty Workbench or a non-overlapping session")
        if {s.id for s in result.series} & {s.id for s in self.state.series}:
            raise ValueError("Session series IDs are already loaded")
        if {r.id for r in result.batch_presets} & {r.id for r in self.state.batch_presets}:
            raise ValueError("Session batch preset IDs are already loaded")
        for collection in ("members", "tracks"):
            existing = {item.id for series in self.state.series for item in getattr(series, collection)}
            incoming = {item.id for series in result.series for item in getattr(series, collection)}
            if existing & incoming:
                raise ValueError(f"Session {collection} IDs are already loaded")
        # No source loading, tree events or preview changes precede this commit.
        self.state.datasets.extend(result.datasets)
        self.state.series.extend(result.series)
        self.state.batch_presets.extend(result.batch_presets)
        if result.series:
            self.select_series(result.series[0].id)
        self.restoration_diagnostics = tuple(f"{d.code}: dataset={d.dataset_id}: {d.message}" for d in result.diagnostics)
        return result
