"""Explicitly saved historical fit snapshots; never authoritative source data."""
from dataclasses import dataclass, field
import json
from uuid import uuid4


@dataclass(frozen=True)
class VTAnalysisEvidence:
    series_id: str
    evidence_json: str
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.series_id, self.id)):
            raise ValueError("Invalid analysis evidence identity")
        data = json.loads(self.evidence_json)
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1 or data.get("quantity") != "observed_chemical_shift":
            raise ValueError("Unsupported analysis evidence schema/quantity")
        if data.get("series_id") != self.series_id or not isinstance(data.get("tracks"), list) or not data["tracks"]:
            raise ValueError("Analysis evidence requires source tracks")
        ids = []
        for row in data["tracks"]:
            snapshot = row["snapshot"]
            if snapshot["series_id"] != self.series_id or not isinstance(snapshot.get("track_id"), str) or not isinstance(snapshot.get("observations"), list):
                raise ValueError("Invalid analysis evidence source snapshot")
            if not isinstance(row.get("results"), list) or not row["results"]:
                raise ValueError("Analysis evidence requires fit results")
            for result in row["results"]:
                if (not isinstance(result, dict) or result.get("schema_version") != 1
                    or result.get("status") not in {"ok", "invalid"}
                    or type(result.get("point_count")) is not int or result["point_count"] < 0):
                    raise ValueError("Invalid historical fit result")
                fit_source = result.get("coordinates", {}).get("data", {})
                if fit_source.get("series_id") != self.series_id or fit_source.get("track_id") != snapshot["track_id"]:
                    raise ValueError("Fit result source identity conflicts with snapshot")
            ids.append(snapshot["track_id"])
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate analysis evidence source tracks")
        object.__setattr__(self, "evidence_json", json.dumps(data, sort_keys=True, allow_nan=False))

    def to_dict(self):
        return {"schema_version": 1, "id": self.id, "series_id": self.series_id, "evidence": json.loads(self.evidence_json)}

    @classmethod
    def from_dict(cls, row):
        if not isinstance(row, dict) or type(row.get("schema_version")) is not int or row["schema_version"] != 1:
            raise ValueError("Unsupported saved analysis schema")
        return cls(row["series_id"], json.dumps(row["evidence"], allow_nan=False), row["id"])
