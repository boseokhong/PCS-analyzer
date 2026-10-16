"""Reusable batch instructions, separate from each dataset's committed history."""
from dataclasses import dataclass, field
import json
from uuid import uuid4

OPERATIONS = ("apodization", "zero_fill", "fourier_transform", "phase", "baseline", "reference", "peak_pick")


@dataclass(frozen=True)
class BatchRecipe:
    name: str
    configuration_json: str
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.id, self.name)):
            raise ValueError("Batch recipe ID/name must be nonempty")
        config = json.loads(self.configuration_json)
        if not isinstance(config, dict) or set(config) - {"operations", "member_ids", "overrides", "include_excluded", "start_from_raw", "initialize_unprocessed"}:
            raise ValueError("Invalid batch configuration")
        rows = config.get("operations")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Batch recipe requires ordered operations")
        names = []
        for row in rows:
            if not isinstance(row, dict) or set(row) - {"operation", "enabled", "parameters"}:
                raise ValueError("Invalid batch operation")
            if row.get("operation") not in OPERATIONS or type(row.get("enabled", True)) is not bool or not isinstance(row.get("parameters", {}), dict):
                raise ValueError("Invalid batch operation/parameters")
            names.append(row["operation"])
        if len(set(names)) != len(names):
            raise ValueError("Duplicate batch operations")
        ids = config.get("member_ids", [])
        if not isinstance(ids, list) or any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid target member IDs")
        for flag in ("include_excluded", "start_from_raw", "initialize_unprocessed"):
            if type(config.get(flag, False)) is not bool:
                raise ValueError(f"{flag} must be boolean")
        overrides = config.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("Invalid member overrides")
        for member_id, operations in overrides.items():
            if not member_id or not isinstance(operations, dict) or set(operations) - set(names) or any(not isinstance(v, dict) for v in operations.values()):
                raise ValueError("Invalid member operation overrides")
        # Normalize/freeze plain JSON parameters; disallow nonfinite recipe values.
        object.__setattr__(self, "configuration_json", json.dumps(config, sort_keys=True, allow_nan=False))

    @property
    def configuration(self):
        return json.loads(self.configuration_json)

    @classmethod
    def create(cls, name, operations, **configuration):
        return cls(name, json.dumps({"operations": operations, **configuration}, allow_nan=False))

    def to_dict(self):
        return {"schema_version": 1, "id": self.id, "name": self.name, "configuration": self.configuration}

    @classmethod
    def from_dict(cls, row):
        if not isinstance(row, dict) or type(row.get("schema_version")) is not int or row["schema_version"] != 1:
            raise ValueError("Unsupported batch recipe schema")
        return cls(row["name"], json.dumps(row["configuration"], allow_nan=False), row["id"])
