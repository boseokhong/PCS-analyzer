"""Persisted review decisions; snapshots are evidence, not spectrum authority."""
from dataclasses import dataclass, field
import json
from uuid import uuid4


@dataclass(frozen=True)
class VTPCSPreparation:
    series_id: str
    track_id: str
    member_id: str
    evidence_json: str
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        for value in (self.series_id, self.track_id, self.member_id, self.id):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("invalid VT PCS preparation identity")
        evidence = json.loads(self.evidence_json)
        if not isinstance(evidence, dict) or type(evidence.get("schema_version")) is not int or evidence["schema_version"] != 1:
            raise ValueError("unsupported VT PCS evidence schema")
        if any(k not in evidence for k in ("paramagnetic", "reference", "target", "basis", "note", "pcs_value")):
            raise ValueError("incomplete VT PCS preparation evidence")
        if not isinstance(evidence["note"], str) or any(not isinstance(evidence[k], dict) or not isinstance(evidence[k].get("id"), str) for k in ("paramagnetic", "reference")):
            raise ValueError("malformed source observation evidence")
        para = evidence["paramagnetic"]
        if (para.get("series_id"), para.get("track_id"), para.get("member_id")) != (self.series_id, self.track_id, self.member_id):
            raise ValueError("preparation/source identity mismatch")
        json.dumps(evidence, allow_nan=False)

    def to_dict(self):
        return {"schema_version": 1, "id": self.id, "series_id": self.series_id,
            "track_id": self.track_id, "member_id": self.member_id,
            "evidence": json.loads(self.evidence_json)}

    @classmethod
    def from_dict(cls, row):
        if not isinstance(row, dict) or type(row.get("schema_version")) is not int or row["schema_version"] != 1:
            raise ValueError("unsupported VT PCS preparation schema")
        return cls(row["series_id"], row["track_id"], row["member_id"],
            json.dumps(row["evidence"], allow_nan=False, sort_keys=True), row["id"])


@dataclass(frozen=True)
class VTPCSHandoffReceipt:
    """Historical transfer references; PCS values remain in the receiving document."""
    series_id: str
    preparation_ids: tuple[str, ...]
    evidence_digests: tuple[str, ...]
    destination_document_id: str | None
    destination_name: str
    transferred_at: str
    replaced_existing: bool = False
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.id, self.series_id, self.destination_name, self.transferred_at)):
            raise ValueError("invalid handoff receipt identity/context")
        if self.destination_document_id is not None and (not isinstance(self.destination_document_id, str) or not self.destination_document_id.strip()):
            raise ValueError("invalid destination document reference")
        if type(self.replaced_existing) is not bool or not isinstance(self.preparation_ids, tuple) or not isinstance(self.evidence_digests, tuple):
            raise ValueError("invalid handoff receipt fields")
        if not self.preparation_ids or len(self.preparation_ids) != len(self.evidence_digests) or len(set(self.preparation_ids)) != len(self.preparation_ids):
            raise ValueError("invalid handoff preparation references")
        if any(not isinstance(v, str) or not v.strip() for v in self.preparation_ids):
            raise ValueError("invalid handoff preparation id")
        if any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in self.evidence_digests):
            raise ValueError("invalid handoff evidence digest")

    def to_dict(self):
        from dataclasses import asdict
        values = asdict(self)
        values["preparation_ids"] = list(self.preparation_ids)
        values["evidence_digests"] = list(self.evidence_digests)
        return {"schema_version": 1, **values}

    @classmethod
    def from_dict(cls, row):
        if not isinstance(row, dict) or type(row.get("schema_version")) is not int or row["schema_version"] != 1:
            raise ValueError("unsupported handoff receipt schema")
        values = {key: row[key] for key in ("id", "series_id", "destination_document_id", "destination_name", "transferred_at", "replaced_existing")}
        for key in ("preparation_ids", "evidence_digests"):
            if not isinstance(row.get(key), list):
                raise ValueError("handoff receipt references must be arrays")
            values[key] = tuple(row[key])
        return cls(**values)
