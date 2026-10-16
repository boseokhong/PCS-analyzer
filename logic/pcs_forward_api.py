"""Public, UI-independent contracts for the PCS point forward model.

This module intentionally contains no prediction kernel and reads no application
state.  It freezes the scientific vocabulary used by later migration layers so
that calculators, fitting code, persistence, and NMR consumers can share one
stable boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import cos, isfinite, pi, sin
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
from uuid import uuid4


PCS_FORWARD_API_VERSION = "0.1"
"""Public Python API version.  Follows ``major.minor`` compatibility."""

PCS_FORWARD_SCHEMA_VERSION = 1
"""Serialization schema carried by prediction requests and results."""

ANGSTROM3_PER_DCHI_UNIT = 1.0e-2
"""One ``10^-32 m^3`` susceptibility unit expressed in cubic ångström."""


class CoordinateUnit(str, Enum):
    ANGSTROM = "angstrom"


class SusceptibilityUnit(str, Enum):
    TEN_MINUS_32_M3 = "1e-32_m3"


class AngleUnit(str, Enum):
    DEGREE = "degree"
    RADIAN = "radian"


class EulerConvention(str, Enum):
    """Supported active rotations from structure coordinates to tensor axes."""

    EXTRINSIC_XYZ = "active_extrinsic_xyz"
    INTRINSIC_ZYZ = "active_intrinsic_zyz"


class TargetReduction(str, Enum):
    NONE = "none"
    ARITHMETIC_MEAN = "arithmetic_mean"


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector3(value: Sequence[float], name: str) -> Vector3:
    values = tuple(_finite_float(item, name) for item in value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return values  # type: ignore[return-value]


def _metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _matmul(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def _transpose(matrix: Matrix3) -> Matrix3:
    return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _axis_rotation(axis: str, angle: float) -> Matrix3:
    c, s = cos(angle), sin(angle)
    if axis == "X":
        return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
    if axis == "Y":
        return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


@dataclass(frozen=True)
class PCSFrame:
    """Metal origin and explicit structure-to-tensor orientation.

    ``EXTRINSIC_XYZ`` matches :func:`logic.rotate_align.euler_matrix` and uses
    fixed-axis X, then Y, then Z rotations. ``INTRINSIC_ZYZ`` records the
    distinct convention used by the existing conformer-search path.  Angles
    are never inferred from their numeric range.
    """

    metal_position: Vector3 = (0.0, 0.0, 0.0)
    euler_angles: Vector3 = (0.0, 0.0, 0.0)
    convention: EulerConvention = EulerConvention.EXTRINSIC_XYZ
    angle_unit: AngleUnit = AngleUnit.DEGREE
    coordinate_unit: CoordinateUnit = CoordinateUnit.ANGSTROM

    def __post_init__(self) -> None:
        object.__setattr__(self, "metal_position", _vector3(self.metal_position, "metal_position"))
        object.__setattr__(self, "euler_angles", _vector3(self.euler_angles, "euler_angles"))
        object.__setattr__(self, "convention", EulerConvention(self.convention))
        object.__setattr__(self, "angle_unit", AngleUnit(self.angle_unit))
        object.__setattr__(self, "coordinate_unit", CoordinateUnit(self.coordinate_unit))

    @property
    def rotation_matrix(self) -> Matrix3:
        """Active matrix mapping a structure-frame column vector to tensor axes."""

        angles = self.euler_angles
        if self.angle_unit is AngleUnit.DEGREE:
            angles = tuple(value * pi / 180.0 for value in angles)  # type: ignore[assignment]

        identity: Matrix3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        if self.convention is EulerConvention.EXTRINSIC_XYZ:
            result = identity
            for axis, angle in zip("XYZ", angles):
                result = _matmul(_axis_rotation(axis, angle), result)
            return result

        # Active intrinsic Z-Y-Z: Rz(alpha) @ Ry(beta) @ Rz(gamma).
        first, second, third = angles
        return _matmul(
            _matmul(_axis_rotation("Z", first), _axis_rotation("Y", second)),
            _axis_rotation("Z", third),
        )


@dataclass(frozen=True)
class PCSTensor:
    """Canonical axial/rhombic susceptibility tensor and its spatial frame.

    The rhombic value is retained when disabled so a UI can toggle the term
    without losing user input.  Scientific consumers must use
    :attr:`effective_dchi_rh`, which is exactly zero while disabled.
    """

    dchi_ax: float
    dchi_rh: float = 0.0
    rhombic_enabled: bool = False
    frame: PCSFrame = field(default_factory=PCSFrame)
    susceptibility_unit: SusceptibilityUnit = SusceptibilityUnit.TEN_MINUS_32_M3

    def __post_init__(self) -> None:
        object.__setattr__(self, "dchi_ax", _finite_float(self.dchi_ax, "dchi_ax"))
        object.__setattr__(self, "dchi_rh", _finite_float(self.dchi_rh, "dchi_rh"))
        object.__setattr__(self, "rhombic_enabled", bool(self.rhombic_enabled))
        if not isinstance(self.frame, PCSFrame):
            raise TypeError("frame must be a PCSFrame")
        object.__setattr__(
            self,
            "susceptibility_unit",
            SusceptibilityUnit(self.susceptibility_unit),
        )

    @property
    def effective_dchi_rh(self) -> float:
        return self.dchi_rh if self.rhombic_enabled else 0.0

    @property
    def principal_values(self) -> Vector3:
        """Traceless ``(chi_xx, chi_yy, chi_zz)`` in canonical dchi units."""

        axial = self.dchi_ax
        rhombic = self.effective_dchi_rh
        return (
            -axial / 3.0 + rhombic / 2.0,
            -axial / 3.0 - rhombic / 2.0,
            2.0 * axial / 3.0,
        )

    def tensor_frame_matrix_angstrom3(self) -> Matrix3:
        """Return the diagonal traceless tensor in Å³ for the PDE adapter."""

        xx, yy, zz = self.principal_values
        scale = ANGSTROM3_PER_DCHI_UNIT
        return ((xx * scale, 0.0, 0.0), (0.0, yy * scale, 0.0), (0.0, 0.0, zz * scale))

    def structure_frame_matrix_angstrom3(self) -> Matrix3:
        """Return ``R.T @ chi_tensor @ R`` in the structure coordinate frame."""

        rotation = self.frame.rotation_matrix
        return _matmul(
            _matmul(_transpose(rotation), self.tensor_frame_matrix_angstrom3()),
            rotation,
        )


@dataclass(frozen=True)
class PCSAtom:
    """One addressable structure coordinate used by a prediction request."""

    ref_id: int
    position: Vector3
    label: str = ""
    coordinate_unit: CoordinateUnit = CoordinateUnit.ANGSTROM
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        ref_id = int(self.ref_id)
        if ref_id < 0:
            raise ValueError("ref_id must be non-negative")
        object.__setattr__(self, "ref_id", ref_id)
        object.__setattr__(self, "position", _vector3(self.position, "position"))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "coordinate_unit", CoordinateUnit(self.coordinate_unit))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True)
class PCSStructureTarget:
    """One predicted observation with explicit member/site hierarchy.

    Each inner tuple is one structural site (for example the three protons of
    a methyl).  ``member_reduction`` reduces atoms within a site and
    ``site_reduction`` reduces matching sites to the single observation.
    """

    target_id: str
    site_member_ref_ids: tuple[tuple[int, ...], ...]
    member_reduction: TargetReduction = TargetReduction.NONE
    site_reduction: TargetReduction = TargetReduction.NONE
    assignment: str | None = None
    experimental_pcs_ppm: float | None = None
    uncertainty_ppm: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        target_id = str(self.target_id).strip()
        if not target_id:
            raise ValueError("target_id must not be blank")
        groups = tuple(tuple(int(ref_id) for ref_id in group) for group in self.site_member_ref_ids)
        if not groups or any(not group for group in groups):
            raise ValueError("site_member_ref_ids must contain non-empty site groups")
        flat = tuple(ref_id for group in groups for ref_id in group)
        if any(ref_id < 0 for ref_id in flat):
            raise ValueError("target Ref IDs must be non-negative")
        if len(set(flat)) != len(flat):
            raise ValueError("a Ref ID may occur only once in a target")

        member_reduction = TargetReduction(self.member_reduction)
        site_reduction = TargetReduction(self.site_reduction)
        if member_reduction is TargetReduction.NONE and any(len(group) != 1 for group in groups):
            raise ValueError("member_reduction='none' requires one Ref per site")
        if site_reduction is TargetReduction.NONE and len(groups) != 1:
            raise ValueError("site_reduction='none' requires exactly one site")

        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "site_member_ref_ids", groups)
        object.__setattr__(self, "member_reduction", member_reduction)
        object.__setattr__(self, "site_reduction", site_reduction)
        object.__setattr__(
            self,
            "assignment",
            None if self.assignment is None else str(self.assignment),
        )
        if self.experimental_pcs_ppm is not None:
            object.__setattr__(
                self,
                "experimental_pcs_ppm",
                _finite_float(self.experimental_pcs_ppm, "experimental_pcs_ppm"),
            )
        if self.uncertainty_ppm is not None:
            uncertainty = _finite_float(self.uncertainty_ppm, "uncertainty_ppm")
            if uncertainty < 0.0:
                raise ValueError("uncertainty_ppm must be non-negative")
            object.__setattr__(self, "uncertainty_ppm", uncertainty)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def member_ref_ids(self) -> tuple[int, ...]:
        return tuple(ref_id for group in self.site_member_ref_ids for ref_id in group)


@dataclass(frozen=True)
class PCSPredictionRequest:
    tensor: PCSTensor
    atoms: tuple[PCSAtom, ...]
    targets: tuple[PCSStructureTarget, ...] = ()
    request_id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: int = PCS_FORWARD_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, PCSTensor):
            raise TypeError("tensor must be a PCSTensor")
        atoms = tuple(self.atoms)
        targets = tuple(self.targets)
        if any(not isinstance(atom, PCSAtom) for atom in atoms):
            raise TypeError("atoms must contain only PCSAtom values")
        if any(not isinstance(target, PCSStructureTarget) for target in targets):
            raise TypeError("targets must contain only PCSStructureTarget values")
        atom_ids = [atom.ref_id for atom in atoms]
        if len(set(atom_ids)) != len(atom_ids):
            raise ValueError("atom ref_id values must be unique")
        target_ids = [target.target_id for target in targets]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("target_id values must be unique")
        available = set(atom_ids)
        missing = sorted({ref_id for target in targets for ref_id in target.member_ref_ids} - available)
        if missing:
            raise ValueError("targets reference atoms absent from the request: " + ", ".join(map(str, missing)))
        if int(self.schema_version) != PCS_FORWARD_SCHEMA_VERSION:
            raise ValueError(f"unsupported PCS forward schema_version: {self.schema_version!r}")
        request_id = str(self.request_id).strip()
        if not request_id:
            raise ValueError("request_id must not be blank")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True)
class PCSAtomPrediction:
    ref_id: int
    pcs_ppm: float
    g_ax: float
    g_rh: float
    distance_angstrom: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref_id", int(self.ref_id))
        for name in ("pcs_ppm", "g_ax", "g_rh"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if self.distance_angstrom is not None:
            distance = _finite_float(self.distance_angstrom, "distance_angstrom")
            if distance < 0.0:
                raise ValueError("distance_angstrom must be non-negative")
            object.__setattr__(self, "distance_angstrom", distance)


@dataclass(frozen=True)
class PCSTargetPrediction:
    target_id: str
    pcs_ppm: float
    site_values_ppm: tuple[float, ...]
    member_ref_ids: tuple[int, ...]
    experimental_pcs_ppm: float | None = None
    residual_ppm: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", str(self.target_id))
        object.__setattr__(self, "pcs_ppm", _finite_float(self.pcs_ppm, "pcs_ppm"))
        object.__setattr__(
            self,
            "site_values_ppm",
            tuple(_finite_float(v, "site_values_ppm") for v in self.site_values_ppm),
        )
        object.__setattr__(self, "member_ref_ids", tuple(int(v) for v in self.member_ref_ids))
        if self.experimental_pcs_ppm is not None:
            object.__setattr__(
                self,
                "experimental_pcs_ppm",
                _finite_float(self.experimental_pcs_ppm, "experimental_pcs_ppm"),
            )
        if self.residual_ppm is not None:
            object.__setattr__(self, "residual_ppm", _finite_float(self.residual_ppm, "residual_ppm"))


@dataclass(frozen=True)
class PCSPredictionResult:
    request_id: str
    atom_predictions: tuple[PCSAtomPrediction, ...]
    target_predictions: tuple[PCSTargetPrediction, ...] = ()
    engine_id: str = ""
    engine_version: str = ""
    api_version: str = PCS_FORWARD_API_VERSION
    schema_version: int = PCS_FORWARD_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        atoms = tuple(self.atom_predictions)
        targets = tuple(self.target_predictions)
        if any(not isinstance(item, PCSAtomPrediction) for item in atoms):
            raise TypeError("atom_predictions must contain PCSAtomPrediction values")
        if any(not isinstance(item, PCSTargetPrediction) for item in targets):
            raise TypeError("target_predictions must contain PCSTargetPrediction values")
        if len({item.ref_id for item in atoms}) != len(atoms):
            raise ValueError("atom prediction Ref IDs must be unique")
        if len({item.target_id for item in targets}) != len(targets):
            raise ValueError("target prediction IDs must be unique")
        if str(self.api_version) != PCS_FORWARD_API_VERSION:
            raise ValueError(f"unsupported PCS forward api_version: {self.api_version!r}")
        if int(self.schema_version) != PCS_FORWARD_SCHEMA_VERSION:
            raise ValueError(f"unsupported PCS forward schema_version: {self.schema_version!r}")
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "atom_predictions", atoms)
        object.__setattr__(self, "target_predictions", targets)
        object.__setattr__(self, "engine_id", str(self.engine_id))
        object.__setattr__(self, "engine_version", str(self.engine_version))
        object.__setattr__(self, "api_version", str(self.api_version))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@runtime_checkable
class PCSForwardModel(Protocol):
    """Structured-target-capable public forward-model interface."""

    api_version: str

    def predict(self, request: PCSPredictionRequest) -> PCSPredictionResult:
        ...


@runtime_checkable
class PCSTargetReducer(Protocol):
    """Extension point for reducing atom predictions to one observation."""

    def reduce(
        self,
        target: PCSStructureTarget,
        predictions_by_ref: Mapping[int, PCSAtomPrediction],
    ) -> PCSTargetPrediction:
        ...


__all__ = [
    "ANGSTROM3_PER_DCHI_UNIT",
    "AngleUnit",
    "CoordinateUnit",
    "EulerConvention",
    "Matrix3",
    "PCSAtom",
    "PCSAtomPrediction",
    "PCS_FORWARD_API_VERSION",
    "PCS_FORWARD_SCHEMA_VERSION",
    "PCSForwardModel",
    "PCSFrame",
    "PCSPredictionRequest",
    "PCSPredictionResult",
    "PCSStructureTarget",
    "PCSTargetPrediction",
    "PCSTargetReducer",
    "PCSTensor",
    "SusceptibilityUnit",
    "TargetReduction",
    "Vector3",
]
