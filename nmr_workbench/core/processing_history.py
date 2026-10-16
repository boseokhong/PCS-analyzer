"""Serializable provenance for reproducible spectrum processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ProcessingStep:
    """Description of one ordered, reproducible processing operation."""

    operation: str
    method: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operation = self.operation.strip()
        if not operation:
            raise ValueError("operation must not be empty")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    def to_dict(self) -> dict[str, Any]:
        """Return a persistence-friendly representation."""

        return {
            "operation": self.operation,
            "method": self.method,
            "parameters": dict(self.parameters),
        }


@dataclass
class ProcessingHistory:
    """Ordered processing recipe owned by an NMR dataset."""

    steps: list[ProcessingStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.steps = list(self.steps)

    def append(self, step: ProcessingStep) -> None:
        """Append a processing step without applying it to spectrum arrays."""

        if not isinstance(step, ProcessingStep):
            raise TypeError("step must be a ProcessingStep")
        self.steps.append(step)

    def extend(self, steps: Iterable[ProcessingStep]) -> None:
        """Append multiple steps in order."""

        for step in steps:
            self.append(step)

    def to_list(self) -> list[dict[str, Any]]:
        """Return the ordered recipe in a persistence-friendly form."""

        return [step.to_dict() for step in self.steps]

