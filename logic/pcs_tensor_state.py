"""Canonical application PCS tensor state and legacy-state compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, MutableMapping

from logic.pcs_forward_api import (
    AngleUnit,
    EulerConvention,
    PCSFrame,
    PCSTensor,
    SusceptibilityUnit,
)


PCS_TENSOR_STATE_KEY = "pcs_tensor_state"
PCS_TENSOR_STATE_SCHEMA_VERSION = 1
_LEGACY_CONVENTION_KEY = "pcs_tensor_frame_convention"
_LEGACY_ANGLE_UNIT_KEY = "pcs_tensor_angle_unit"


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ApplicationPCSTensor:
    """One application-owned anisotropic tensor in the molecular frame.

    ``alpha``, ``beta``, and ``gamma`` are interpreted only through
    ``frame_convention`` and ``angle_unit``. They define the active rotation
    from molecular coordinates to the tensor principal-axis frame. Principal
    components retain the explicit x/y/z labels; no value-based ordering is
    performed.
    """

    delta_chi_ax: float
    delta_chi_rh: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    rhombic_enabled: bool = False
    frame_convention: EulerConvention = EulerConvention.EXTRINSIC_XYZ
    angle_unit: AngleUnit = AngleUnit.DEGREE
    susceptibility_unit: SusceptibilityUnit = SusceptibilityUnit.TEN_MINUS_32_M3
    schema_version: int = PCS_TENSOR_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "delta_chi_ax",
            "delta_chi_rh",
            "alpha",
            "beta",
            "gamma",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(
            self,
            "frame_convention",
            EulerConvention(self.frame_convention),
        )
        object.__setattr__(self, "angle_unit", AngleUnit(self.angle_unit))
        object.__setattr__(self, "rhombic_enabled", bool(self.rhombic_enabled))
        object.__setattr__(
            self,
            "susceptibility_unit",
            SusceptibilityUnit(self.susceptibility_unit),
        )
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def euler_angles(self) -> tuple[float, float, float]:
        return (self.alpha, self.beta, self.gamma)

    @property
    def chi_xx(self) -> float:
        return -self.delta_chi_ax / 3.0 + self.delta_chi_rh / 2.0

    @property
    def chi_yy(self) -> float:
        return -self.delta_chi_ax / 3.0 - self.delta_chi_rh / 2.0

    @property
    def chi_zz(self) -> float:
        return 2.0 * self.delta_chi_ax / 3.0

    @property
    def principal_values(self) -> tuple[float, float, float]:
        """Return labeled ``(chi_xx, chi_yy, chi_zz)`` without reordering."""
        return (self.chi_xx, self.chi_yy, self.chi_zz)

    def to_forward_tensor(
        self,
        *,
        metal_position=(0.0, 0.0, 0.0),
        rhombic_enabled: bool | None = None,
    ) -> PCSTensor:
        """Adapt the application tensor to the public L0A forward contract."""
        return PCSTensor(
            dchi_ax=self.delta_chi_ax,
            dchi_rh=self.delta_chi_rh,
            rhombic_enabled=(
                self.rhombic_enabled
                if rhombic_enabled is None
                else bool(rhombic_enabled)
            ),
            frame=PCSFrame(
                metal_position=metal_position,
                euler_angles=self.euler_angles,
                convention=self.frame_convention,
                angle_unit=self.angle_unit,
            ),
            susceptibility_unit=self.susceptibility_unit,
        )


def _read_widget_or_value(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        value = value.get()
    except AttributeError:
        pass
    return fallback if value is None or str(value).strip() == "" else value


def _read_float(value: Any, fallback: float) -> float:
    try:
        return float(_read_widget_or_value(value, fallback))
    except (TypeError, ValueError):
        return float(fallback)


def canonical_tensor_from_legacy(
    state: MutableMapping[str, Any],
) -> ApplicationPCSTensor:
    """Build canonical tensor state from the current split legacy keys."""
    fallback_axial = _read_float(state.get("tensor"), 0.0)
    delta_chi_ax = _read_float(state.get("tensor_entry"), fallback_axial)
    delta_chi_rh = _read_float(state.get("rh_dchi_rh"), 0.0)

    alpha = _read_float(
        state.get("angle_x_var"),
        _read_float(state.get("angle_x"), 0.0),
    )
    beta = _read_float(
        state.get("angle_y_var"),
        _read_float(state.get("angle_y"), 0.0),
    )
    gamma = _read_float(
        state.get("angle_z_var"),
        _read_float(state.get("angle_z"), 0.0),
    )

    canonical = ApplicationPCSTensor(
        delta_chi_ax=delta_chi_ax,
        delta_chi_rh=delta_chi_rh,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        rhombic_enabled=bool(state.get("rh_calc_enabled", False)),
        frame_convention=state.get(
            _LEGACY_CONVENTION_KEY,
            EulerConvention.EXTRINSIC_XYZ,
        ),
        angle_unit=state.get(_LEGACY_ANGLE_UNIT_KEY, AngleUnit.DEGREE),
    )
    state[PCS_TENSOR_STATE_KEY] = canonical
    return canonical


def _set_variable(state: MutableMapping[str, Any], key: str, value: float) -> None:
    variable = state.get(key)
    if variable is not None and hasattr(variable, "set"):
        variable.set(value)


def _set_entry(entry: Any, value: float) -> None:
    if entry is None:
        return
    try:
        entry.delete(0, "end")
        # These editable values are read back by legacy redraw/save paths.
        # Six-digit :g formatting silently rounds a restored tensor on redraw.
        entry.insert(0, str(float(value)))
    except (AttributeError, TypeError):
        return


def apply_canonical_tensor_to_legacy(
    state: MutableMapping[str, Any],
    tensor: ApplicationPCSTensor | None = None,
) -> ApplicationPCSTensor:
    """Project canonical values into legacy keys without removing any keys."""
    canonical = tensor if tensor is not None else state.get(PCS_TENSOR_STATE_KEY)
    if not isinstance(canonical, ApplicationPCSTensor):
        raise TypeError("canonical PCS tensor state is not available")

    state[PCS_TENSOR_STATE_KEY] = canonical
    state["tensor"] = canonical.delta_chi_ax
    state["rh_dchi_rh"] = canonical.delta_chi_rh
    # Legacy X/Y/Z widgets and .pcsp files are always active extrinsic XYZ.
    # Never reinterpret canonical ZYZ angles as those legacy coordinates.
    from logic.pcs_orientation import matrix_to_euler, orientation_matrix

    legacy_angles = matrix_to_euler(
        orientation_matrix(
            canonical.euler_angles,
            canonical.frame_convention,
            canonical.angle_unit,
        ),
        EulerConvention.EXTRINSIC_XYZ,
        AngleUnit.DEGREE,
    )
    state["angle_x"] = legacy_angles[0]
    state["angle_y"] = legacy_angles[1]
    state["angle_z"] = legacy_angles[2]
    state["rh_calc_enabled"] = canonical.rhombic_enabled
    state[_LEGACY_CONVENTION_KEY] = EulerConvention.EXTRINSIC_XYZ.value
    state[_LEGACY_ANGLE_UNIT_KEY] = AngleUnit.DEGREE.value

    _set_entry(state.get("tensor_entry"), canonical.delta_chi_ax)
    _set_variable(state, "rh_dchi_rh_var", canonical.delta_chi_rh)
    _set_variable(state, "angle_x_var", legacy_angles[0])
    _set_variable(state, "angle_y_var", legacy_angles[1])
    _set_variable(state, "angle_z_var", legacy_angles[2])
    # XYZ entries are independent widgets, not textvariables of the sliders.
    # Project both representations so a later FocusOut cannot reapply stale text.
    for axis, angle in zip(("x", "y", "z"), legacy_angles):
        _set_entry(state.get(f"angle_{axis}_entry"), angle)
    _set_variable(state, "rh_calc_enabled_var", canonical.rhombic_enabled)
    return canonical


def get_canonical_tensor_state(
    state: MutableMapping[str, Any],
) -> ApplicationPCSTensor:
    """Return the installed canonical state, importing legacy values if absent."""
    canonical = state.get(PCS_TENSOR_STATE_KEY)
    if isinstance(canonical, ApplicationPCSTensor):
        return canonical
    return canonical_tensor_from_legacy(state)


__all__ = [
    "ApplicationPCSTensor",
    "PCS_TENSOR_STATE_KEY",
    "PCS_TENSOR_STATE_SCHEMA_VERSION",
    "apply_canonical_tensor_to_legacy",
    "canonical_tensor_from_legacy",
    "get_canonical_tensor_state",
]
