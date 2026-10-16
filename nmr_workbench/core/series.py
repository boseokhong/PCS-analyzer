"""VT-series membership over existing experiments; no spectrum/session ownership."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any
from uuid import uuid4

from .dataset import NMRDataset
from .metadata import TemperatureResolution, resolve_temperature
from .peak_tracks import PeakTrack, TrackProvenance
from .track_observables import TrackObservableAssociation
from .vt_pcs import VTPCSPreparation, VTPCSHandoffReceipt
from .batch_recipe import BatchRecipe
from .analysis_evidence import VTAnalysisEvidence

NMR_SERIES_SCHEMA_VERSION = 2


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _order(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("acquisition_order must be a non-negative integer")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value


def _temperature_value(value: Any) -> float | None:
    """Strict JSON boundary; malformed/non-finite Kelvin values are rejected."""
    if value is None:
        return None
    resolved = resolve_temperature(None, override_k=value)
    if resolved.status != "valid":
        raise ValueError("serialized temperatures must be finite positive Kelvin or null")
    return resolved.temperature_k


def _descriptive_temperature(value):
    """Keep invalid acquisition values as typed provenance, never effective K."""
    resolution = resolve_temperature(None, override_k=value)
    if resolution.status != "invalid":
        return resolution.temperature_k, None
    if isinstance(value, bool):
        kind, raw = "bool", value
    elif isinstance(value, str):
        kind, raw = "str", value
    elif isinstance(value, Real):
        kind, raw = "float", float(value)
        if not math.isfinite(raw):
            raw = "nan" if math.isnan(raw) else "inf" if raw > 0 else "-inf"
    else:
        raise ValueError("unsupported raw temperature metadata type")
    return None, {"status": "invalid", "raw_type": kind, "raw_value": raw}


def _validate_temperature_provenance(row):
    records = row.get("temperature_provenance", {})
    if not isinstance(records, Mapping):
        raise ValueError("temperature_provenance must be an object")
    for name, record in records.items():
        if name not in {"temperature_k", "calibrated_temperature_k"} or not isinstance(record, Mapping):
            raise ValueError("invalid descriptive temperature provenance")
        if row.get(name) is not None or record.get("status") != "invalid":
            raise ValueError("invalid descriptive temperature provenance")
        kind, raw = record.get("raw_type"), record.get("raw_value")
        if kind == "float":
            if isinstance(raw, str) and raw in {"nan", "inf", "-inf"}:
                value = float(raw)
            elif type(raw) in {int, float} and math.isfinite(raw):
                value = raw
            else:
                raise ValueError("invalid raw temperature value")
        elif (kind == "bool" and type(raw) is bool) or (kind == "str" and isinstance(raw, str)):
            value = raw
        else:
            raise ValueError("invalid raw temperature type")
        if resolve_temperature(None, override_k=value).status != "invalid":
            raise ValueError("raw temperature provenance must describe an invalid value")


@dataclass(frozen=True)
class NMRSeriesMember:
    """Membership metadata referencing the original dataset, without copying it.

    Use dataclasses.replace to edit label/include/override state. Temperature
    validity is reported separately, so incomplete series can be inspected.
    """

    dataset: NMRDataset = field(repr=False, compare=False)
    acquisition_order: int
    label: str | None = None
    included: bool = True
    temperature_override_k: float | None = None
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, NMRDataset):
            raise TypeError("dataset must be an NMRDataset")
        _identifier(self.dataset.id, "dataset_id")
        _identifier(self.id, "member id")
        _order(self.acquisition_order)
        _optional_text(self.label, "label")
        if not isinstance(self.included, bool):
            raise ValueError("included must be a boolean")

    @property
    def dataset_id(self) -> str:
        return self.dataset.id

    def resolve_temperature(self, *, series_override_k: float | None = None) -> TemperatureResolution:
        return resolve_temperature(
            self.dataset.conditions,
            override_k=self.temperature_override_k,
            series_override_k=series_override_k,
        )

    def to_dict(self, *, series_override_k=None) -> dict[str, Any]:
        """Return references and condition snapshots, never dataset-owned arrays."""
        if self.resolve_temperature(series_override_k=series_override_k).status == "invalid":
            raise ValueError("effective temperature must be finite positive Kelvin or missing")
        nominal, nominal_raw = _descriptive_temperature(self.dataset.conditions.temperature_k)
        calibrated, calibrated_raw = _descriptive_temperature(self.dataset.conditions.calibrated_temperature_k)
        payload = {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "source_path": str(self.dataset.source_path) if self.dataset.source_path is not None else None,
            "vendor": self.dataset.vendor,
            "acquisition_order": self.acquisition_order,
            "label": self.label,
            "included": self.included,
            "temperature_override_k": _temperature_value(self.temperature_override_k),
            "temperature_k": nominal,
            "calibrated_temperature_k": calibrated,
        }
        provenance = {name: raw for name, raw in (
            ("temperature_k", nominal_raw), ("calibrated_temperature_k", calibrated_raw)) if raw is not None}
        if provenance:
            payload["temperature_provenance"] = provenance
        return payload


@dataclass(frozen=True)
class SeriesTemperatureIssue:
    """Member-scoped missing/invalid temperatures or an exact duplicate group."""

    code: str
    member_ids: tuple[str, ...]
    temperature_k: float | None = None


@dataclass
class NMRSeries:
    """Independent grouping of datasets; never bound implicitly to a PCS document."""

    name: str = ""
    id: str = field(default_factory=lambda: str(uuid4()))
    temperature_override_k: float | None = None
    _members: list[NMRSeriesMember] = field(default_factory=list, init=False, repr=False)
    _next_acquisition_order: int = field(default=0, init=False, repr=False)
    _tracks: list[PeakTrack] = field(default_factory=list, init=False, repr=False)
    _observable_associations: list[TrackObservableAssociation] = field(default_factory=list, init=False, repr=False)
    _pcs_preparations: list[VTPCSPreparation] = field(default_factory=list, init=False, repr=False)
    _pcs_handoff_receipts: list[VTPCSHandoffReceipt] = field(default_factory=list, init=False, repr=False)
    _batch_recipes: list[BatchRecipe] = field(default_factory=list, init=False, repr=False)
    _analysis_evidence: list[VTAnalysisEvidence] = field(default_factory=list, init=False, repr=False)

    @property
    def analysis_evidence(self):
        return tuple(self._analysis_evidence)

    def add_analysis_evidence(self, evidence):
        if not isinstance(evidence, VTAnalysisEvidence) or evidence.series_id != self.id or any(r.id == evidence.id for r in self.analysis_evidence):
            raise ValueError("Invalid/duplicate analysis evidence")
        self._analysis_evidence.append(evidence)

    def remove_analysis_evidence(self, evidence_id):
        if not any(r.id == evidence_id for r in self.analysis_evidence):
            raise KeyError(evidence_id)
        self._analysis_evidence[:] = [r for r in self._analysis_evidence if r.id != evidence_id]

    @property
    def batch_recipes(self):
        return tuple(self._batch_recipes)

    def set_batch_recipe(self, recipe):
        if not isinstance(recipe, BatchRecipe):
            raise ValueError("Expected a batch recipe")
        self._batch_recipes[:] = [r for r in self._batch_recipes if r.id != recipe.id] + [recipe]

    def remove_batch_recipe(self, recipe_id):
        if not any(r.id == recipe_id for r in self.batch_recipes):
            raise KeyError(recipe_id)
        self._batch_recipes[:] = [r for r in self._batch_recipes if r.id != recipe_id]

    @property
    def pcs_handoff_receipts(self):
        return tuple(self._pcs_handoff_receipts)

    def add_pcs_handoff_receipt(self, receipt):
        if not isinstance(receipt, VTPCSHandoffReceipt) or receipt.series_id != self.id or any(r.id == receipt.id for r in self.pcs_handoff_receipts):
            raise ValueError("invalid/duplicate PCS handoff receipt")
        self._pcs_handoff_receipts.append(receipt)

    @property
    def pcs_preparations(self):
        return tuple(self._pcs_preparations)

    def set_pcs_preparation(self, preparation):
        if not isinstance(preparation, VTPCSPreparation) or preparation.series_id != self.id:
            raise ValueError("PCS preparation belongs to another series")
        self.get_track(preparation.track_id)
        if not any(m.id == preparation.member_id for m in self.members):
            raise ValueError("PCS preparation member is unresolved")
        if any(p.id == preparation.id and (p.track_id, p.member_id) != (preparation.track_id, preparation.member_id) for p in self.pcs_preparations):
            raise ValueError("duplicate PCS preparation id")
        self.remove_pcs_preparation(preparation.track_id, preparation.member_id)
        self._pcs_preparations.append(preparation)

    def remove_pcs_preparation(self, track_id, member_id):
        self._pcs_preparations[:] = [p for p in self.pcs_preparations if (p.track_id, p.member_id) != (track_id, member_id)]

    @property
    def observable_associations(self):
        return tuple(self._observable_associations)

    def observable_association(self, track_id):
        return next((a for a in self.observable_associations if a.track_id == track_id), None)

    def set_observable_association(self, association):
        if not isinstance(association, TrackObservableAssociation) or association.series_id != self.id:
            raise ValueError("association belongs to another series")
        self.get_track(association.track_id)
        if any(a.id == association.id and a.track_id != association.track_id for a in self.observable_associations):
            raise ValueError("duplicate observable association id")
        self.remove_observable_association(association.track_id)
        self._observable_associations.append(association)

    def remove_observable_association(self, track_id):
        self._observable_associations[:] = [a for a in self.observable_associations if a.track_id != track_id]

    @property
    def tracks(self) -> tuple[PeakTrack, ...]:
        return tuple(self._tracks)

    def create_track(self, *, label=None, provenance: TrackProvenance | None = None) -> PeakTrack:
        track = PeakTrack(self.id, label=label, provenance=provenance or TrackProvenance())
        self._tracks.append(track)
        return track

    def delete_track(self, track_id: str) -> PeakTrack:
        track = self.get_track(track_id)
        self._tracks.remove(track)
        self.remove_observable_association(track_id)
        return track

    def get_track(self, track_id: str) -> PeakTrack:
        for track in self._tracks:
            if track.id == track_id:
                return track
        raise KeyError(f"unknown track: {track_id}")

    def update_tracks(self):
        return tuple(issue for track in self._tracks for issue in track.update_stale(self))

    def invalidate_tracks(self, **filters):
        for track in self._tracks:
            track.invalidate(**filters)

    def __post_init__(self) -> None:
        _identifier(self.id, "series id")
        if not isinstance(self.name, str):
            raise ValueError("series name must be a string")

    @property
    def members(self) -> tuple[NMRSeriesMember, ...]:
        """Original acquisition order, including excluded members."""
        return tuple(sorted(self._members, key=lambda member: member.acquisition_order))

    def add_member(
        self,
        dataset: NMRDataset,
        *,
        acquisition_order: int | None = None,
        label: str | None = None,
        included: bool = True,
        temperature_override_k: float | None = None,
    ) -> NMRSeriesMember:
        member = NMRSeriesMember(
            dataset=dataset,
            acquisition_order=self._next_acquisition_order if acquisition_order is None else acquisition_order,
            label=label,
            included=included,
            temperature_override_k=temperature_override_k,
        )
        self._append_member(member)
        return member

    def _append_member(self, member: NMRSeriesMember) -> None:
        if any(item.dataset_id == member.dataset_id for item in self._members):
            raise ValueError("a dataset can appear only once in a series")
        if any(item.id == member.id for item in self._members):
            raise ValueError("duplicate series member id")
        if any(item.acquisition_order == member.acquisition_order for item in self._members):
            raise ValueError("duplicate acquisition_order")
        self._members.append(member)
        self._next_acquisition_order = max(self._next_acquisition_order, member.acquisition_order + 1)

    def remove_member(self, member_id: str) -> NMRSeriesMember:
        for index, member in enumerate(self._members):
            if member.id == member_id:
                self.invalidate_tracks(member_id=member_id, reason="member_removed")
                return self._members.pop(index)
        raise KeyError(f"unknown series member id: {member_id}")

    def replace_member(self, member: NMRSeriesMember) -> None:
        """Apply a metadata edit while retaining dataset identity and original order."""
        if not isinstance(member, NMRSeriesMember):
            raise TypeError("member must be an NMRSeriesMember")
        for index, current in enumerate(self._members):
            if current.id == member.id:
                if member.dataset is not current.dataset or member.acquisition_order != current.acquisition_order:
                    raise ValueError("metadata edits must preserve dataset identity and acquisition order")
                self._members[index] = member
                return
        raise KeyError(f"unknown series member id: {member.id}")

    def temperature_sorted_members(
        self, *, descending: bool = False, included_only: bool = False
    ) -> tuple[NMRSeriesMember, ...]:
        """Non-mutating view; ties retain original order, missing/invalid come last."""
        members = [item for item in self.members if item.included or not included_only]

        def key(member: NMRSeriesMember) -> tuple[bool, float]:
            value = member.resolve_temperature(series_override_k=self.temperature_override_k).temperature_k
            return (value is None, (-value if descending else value) if value is not None else 0.0)

        return tuple(sorted(members, key=key))

    def validate_temperatures(self, *, included_only: bool = False) -> tuple[SeriesTemperatureIssue, ...]:
        """Report effective-temperature problems without dropping members.

        Duplicate detection uses exact Kelvin equality, not an implicit tolerance.
        Excluded members participate unless included_only is explicitly requested.
        """
        issues = []
        groups: dict[float, list[str]] = {}
        for member in self.members:
            if included_only and not member.included:
                continue
            resolved = member.resolve_temperature(series_override_k=self.temperature_override_k)
            if resolved.status != "valid":
                issues.append(SeriesTemperatureIssue(resolved.status, (member.id,)))
            else:
                groups.setdefault(resolved.temperature_k, []).append(member.id)
        issues.extend(
            SeriesTemperatureIssue("duplicate", tuple(ids), value)
            for value, ids in groups.items() if len(ids) > 1
        )
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        self.update_tracks()
        return {
            "schema_version": NMR_SERIES_SCHEMA_VERSION,
            "id": self.id,
            "name": self.name,
            "temperature_override_k": _temperature_value(self.temperature_override_k),
            "next_acquisition_order": self._next_acquisition_order,
            "members": [member.to_dict(series_override_k=self.temperature_override_k) for member in self.members],
            "tracks": [track.to_dict() for track in self.tracks],
            **({"observable_associations": [a.to_dict() for a in self.observable_associations]} if self.observable_associations else {}),
            **({"pcs_preparations": [p.to_dict() for p in self.pcs_preparations]} if self.pcs_preparations else {}),
            **({"pcs_handoff_receipts": [r.to_dict() for r in self.pcs_handoff_receipts]} if self.pcs_handoff_receipts else {}),
            **({"batch_recipes": [r.to_dict() for r in self.batch_recipes]} if self.batch_recipes else {}),
            **({"analysis_evidence": [r.to_dict() for r in self.analysis_evidence]} if self.analysis_evidence else {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], datasets: Mapping[str, NMRDataset]) -> NMRSeries:
        """Resolve references against caller-owned datasets; never load files.

        Source/temperature snapshots are descriptive provenance only. They are
        validated but never written back into the resolved dataset metadata.
        Unknown fields are ignored for additive compatibility.
        """
        if not isinstance(payload, Mapping):
            raise ValueError("series payload must be an object")
        version = payload.get("schema_version")
        if type(version) is not int or version not in {1, NMR_SERIES_SCHEMA_VERSION}:
            raise ValueError(f"unsupported series schema_version: {version}")
        series = cls(
            id=_identifier(payload.get("id"), "series id"),
            name=payload.get("name"),
            temperature_override_k=_temperature_value(payload.get("temperature_override_k")),
        )
        rows = payload.get("members")
        if not isinstance(rows, list):
            raise ValueError("members must be an array")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("member must be an object")
            dataset_id = _identifier(row.get("dataset_id"), "dataset_id")
            dataset = datasets.get(dataset_id)
            if not isinstance(dataset, NMRDataset) or dataset.id != dataset_id:
                raise ValueError(f"unresolved dataset reference: {dataset_id}")
            for name in ("source_path", "vendor"):
                _optional_text(row.get(name), name)
            for name in ("temperature_k", "calibrated_temperature_k"):
                _temperature_value(row.get(name))
            _validate_temperature_provenance(row)
            series._append_member(NMRSeriesMember(
                dataset=dataset,
                id=_identifier(row.get("id"), "member id"),
                acquisition_order=_order(row.get("acquisition_order")),
                label=row.get("label"),
                included=row.get("included", True),
                temperature_override_k=_temperature_value(row.get("temperature_override_k")),
            ))
        next_order = _order(payload.get("next_acquisition_order", series._next_acquisition_order))
        if next_order < series._next_acquisition_order:
            raise ValueError("next_acquisition_order must follow existing members")
        series._next_acquisition_order = next_order
        tracks = payload.get("tracks", []) if version == 2 else []
        if not isinstance(tracks, list):
            raise ValueError("tracks must be an array")
        for row in tracks:
            track = PeakTrack.from_dict(row)
            if track.series_id != series.id:
                raise ValueError("track series association mismatch")
            if any(t.id == track.id for t in series.tracks):
                raise ValueError("duplicate track id")
            series._tracks.append(track)
        associations = payload.get("observable_associations", [])
        if not isinstance(associations, list):
            raise ValueError("observable_associations must be an array")
        for row in associations:
            association = TrackObservableAssociation.from_dict(row)
            if association.series_id != series.id:
                raise ValueError("observable association series mismatch")
            if any(a.track_id == association.track_id or a.id == association.id for a in series.observable_associations):
                raise ValueError("duplicate observable association")
            # Broken track references are preserved for recovery diagnostics.
            series._observable_associations.append(association)
        preparations = payload.get("pcs_preparations", [])
        if not isinstance(preparations, list):
            raise ValueError("pcs_preparations must be an array")
        for row in preparations:
            preparation = VTPCSPreparation.from_dict(row)
            if preparation.series_id != series.id or any(p.id == preparation.id or (p.track_id, p.member_id) == (preparation.track_id, preparation.member_id) for p in series.pcs_preparations):
                raise ValueError("invalid/duplicate PCS preparation association")
            series._pcs_preparations.append(preparation)
        receipts = payload.get("pcs_handoff_receipts", [])
        if not isinstance(receipts, list):
            raise ValueError("pcs_handoff_receipts must be an array")
        for row in receipts:
            series.add_pcs_handoff_receipt(VTPCSHandoffReceipt.from_dict(row))
        recipes = payload.get("batch_recipes", [])
        if not isinstance(recipes, list):
            raise ValueError("batch_recipes must be an array")
        for row in recipes:
            recipe = BatchRecipe.from_dict(row)
            if any(r.id == recipe.id for r in series.batch_recipes):
                raise ValueError("duplicate batch recipe ID")
            series.set_batch_recipe(recipe)
        evidence = payload.get("analysis_evidence", [])
        if not isinstance(evidence, list):
            raise ValueError("analysis_evidence must be an array")
        for row in evidence:
            series.add_analysis_evidence(VTAnalysisEvidence.from_dict(row))
        series.update_tracks()
        return series
