"""Immutable scientific inputs for one standard PCS optimizer run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from logic.pcs_forward_api import PCSStructureTarget, TargetReduction


@dataclass(frozen=True)
class FitObservation:
    ref_id: int
    pcs: float
    observable_id: str = ""
    assignment: str = ""
    target: PCSStructureTarget | None = None
    source: str = "legacy_single_atom"
    source_dataset_ids: tuple[str, ...] = ()
    source_observable_ids: tuple[str, ...] = ()
    provenance_json: str = "{}"
    uncertainty_ppm: float | None = None

    def __post_init__(self) -> None:
        ref_id = int(self.ref_id)
        pcs = float(self.pcs)
        observable_id = str(self.observable_id or f"legacy-ref:{ref_id}")
        target = self.target
        if target is None:
            target = PCSStructureTarget(
                target_id=observable_id,
                site_member_ref_ids=((ref_id,),),
                member_reduction=TargetReduction.NONE,
                site_reduction=TargetReduction.NONE,
                assignment=self.assignment or None,
                experimental_pcs_ppm=pcs,
                metadata={"scope": "Single atom", "averaging_hint": "none"},
            )
        object.__setattr__(self, "ref_id", ref_id)
        object.__setattr__(self, "pcs", pcs)
        object.__setattr__(self, "observable_id", observable_id)
        object.__setattr__(self, "assignment", str(self.assignment or ""))
        object.__setattr__(self, "target", target)
        uncertainty = self.uncertainty_ppm
        if uncertainty is None and target is not None:
            uncertainty = target.uncertainty_ppm
        object.__setattr__(
            self,
            "uncertainty_ppm",
            None if uncertainty is None else float(uncertainty),
        )
        object.__setattr__(self, "source_dataset_ids", tuple(map(str, self.source_dataset_ids)))
        object.__setattr__(self, "source_observable_ids", tuple(map(str, self.source_observable_ids)))


@dataclass(frozen=True)
class TorsionGroupSnapshot:
    enabled: bool
    axis_atoms: tuple[int, ...]
    rotating_atoms: tuple[int, ...]
    n_samples: int
    range_mode: str
    range_start_deg: float
    range_end_deg: float

    def as_mapping(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "axis_atoms": self.axis_atoms,
            "rotating_atoms": self.rotating_atoms,
            "n_samples": self.n_samples,
            "range_mode": self.range_mode,
            "range_start_deg": self.range_start_deg,
            "range_end_deg": self.range_end_deg,
        }


@dataclass(frozen=True)
class FitGeometrySnapshot:
    effective_ref_ids: tuple[int, ...]
    effective_coordinates: tuple[tuple[float, float, float], ...]
    raw_ref_ids: tuple[int, ...]
    raw_coordinates: tuple[tuple[float, float, float], ...]
    metal_position: tuple[float, float, float]
    pseudo_members: tuple[tuple[int, tuple[int, ...]], ...]
    torsion_enabled: bool
    torsion_groups: tuple[TorsionGroupSnapshot, ...]

    def effective_array(self) -> np.ndarray:
        return np.asarray(self.effective_coordinates, dtype=float)

    def raw_array(self) -> np.ndarray:
        return np.asarray(self.raw_coordinates, dtype=float)

    def pseudo_member_mapping(self) -> dict[int, tuple[int, ...]]:
        return {ref_id: members for ref_id, members in self.pseudo_members}

    def active_torsion_mappings(self) -> list[dict[str, Any]]:
        if not self.torsion_enabled:
            return []
        return [group.as_mapping() for group in self.torsion_groups]


@dataclass(frozen=True)
class FitTensorSnapshot:
    fit_delta_chi_ax: bool
    fit_delta_chi_rh: bool
    fixed_delta_chi_ax: float
    fixed_delta_chi_rh: float


@dataclass(frozen=True)
class FitOptimizerSnapshot:
    identity: str
    use_global_search: bool
    local_optimizer: str
    de_seed: int | None
    de_polish: bool | None
    de_default_span: float | None
    effective_de_bounds: tuple[tuple[float, float], ...] | None


@dataclass(frozen=True)
class StandardFitInputSnapshot:
    mode: str
    observations: tuple[FitObservation, ...]
    geometry: FitGeometrySnapshot
    donor_ids: tuple[int, ...]
    donor_coordinates: tuple[tuple[float, float, float], ...]
    axis_mode: str | None
    mode_a_angle_z: float
    use_current_tensor: bool
    tensor: FitTensorSnapshot
    free_parameters: tuple[str, ...]
    x0: tuple[float, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    optimizer: FitOptimizerSnapshot
    weighting_mode: str = "unweighted"

    def x0_array(self) -> np.ndarray:
        return np.asarray(self.x0, dtype=float)


def snapshot_torsion_group(group: Any) -> TorsionGroupSnapshot:
    if isinstance(group, Mapping):
        get = group.get
    else:
        get = lambda key, default=None: getattr(group, key, default)
    return TorsionGroupSnapshot(
        enabled=bool(get("enabled", True)),
        axis_atoms=tuple(int(value) for value in get("axis_atoms", ())),
        rotating_atoms=tuple(int(value) for value in get("rotating_atoms", ())),
        n_samples=max(1, int(get("n_samples", 24))),
        range_mode=str(get("range_mode", "full")),
        range_start_deg=float(get("range_start_deg", 0.0)),
        range_end_deg=float(get("range_end_deg", 360.0)),
    )


__all__ = [
    "FitGeometrySnapshot",
    "FitObservation",
    "FitOptimizerSnapshot",
    "FitTensorSnapshot",
    "StandardFitInputSnapshot",
    "TorsionGroupSnapshot",
    "snapshot_torsion_group",
]
