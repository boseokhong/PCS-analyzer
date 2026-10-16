"""Reviewed assignment intent and references; never authoritative observables."""
from dataclasses import asdict, dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class ObservableBinding:
    member_id: str
    dataset_id: str
    peak_id: str
    observable_id: str
    method: str = "explicit_member_confirmation"

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in asdict(self).values()):
            raise ValueError("observable binding references/method must be nonempty strings")


@dataclass(frozen=True)
class TrackObservableAssociation:
    series_id: str
    track_id: str
    assignment: str
    nucleus: str
    note: str = ""
    id: str = field(default_factory=lambda: str(uuid4()))
    method: str = "explicit_assignment_review"
    bindings: tuple[ObservableBinding, ...] = ()

    def __post_init__(self):
        for name in ("series_id", "track_id", "assignment", "nucleus", "id", "method"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"association {name} must be a nonempty string")
        if not isinstance(self.note, str) or not isinstance(self.bindings, tuple):
            raise ValueError("invalid association provenance/bindings")
        if any(not isinstance(b, ObservableBinding) for b in self.bindings):
            raise ValueError("invalid observable binding")
        if len({b.member_id for b in self.bindings}) != len(self.bindings):
            raise ValueError("duplicate association member binding")

    def to_dict(self):
        values = asdict(self)
        values["bindings"] = list(values["bindings"])
        return {"schema_version": 1, **values}

    @classmethod
    def from_dict(cls, row):
        if not isinstance(row, dict) or type(row.get("schema_version")) is not int or row["schema_version"] != 1:
            raise ValueError("unsupported track observable association schema")
        values = {k: row[k] for k in ("series_id", "track_id", "assignment", "nucleus", "id", "method", "note")}
        if not isinstance(row.get("bindings"), (list, tuple)):
            raise ValueError("association bindings must be an array")
        values["bindings"] = tuple(ObservableBinding(**b) for b in row["bindings"])
        return cls(**values)
