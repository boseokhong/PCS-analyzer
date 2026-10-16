"""Rotation-representation adapters for the application PCS tensor.

Both supported representations describe the same active transform from the
molecular frame into the tensor principal-axis frame::

    R_xyz = Rz(z) @ Ry(y) @ Rx(x)
    R_zyz = Rz(alpha) @ Ry(beta) @ Rz(gamma)

Conversions always pass through the rotation matrix.  SciPy's deterministic
Euler decomposition is used: at a gimbal-lock singularity the third angle is
set to zero.  This makes the displayed branch reproducible while preserving
the physical rotation matrix.
"""

from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from logic.pcs_forward_api import AngleUnit, EulerConvention, PCSFrame


def orientation_matrix(
    angles: tuple[float, float, float],
    convention: EulerConvention,
    angle_unit: AngleUnit = AngleUnit.DEGREE,
) -> np.ndarray:
    """Return the canonical active molecular-to-tensor rotation matrix."""
    frame = PCSFrame(
        euler_angles=tuple(float(value) for value in angles),
        convention=EulerConvention(convention),
        angle_unit=AngleUnit(angle_unit),
    )
    return np.asarray(frame.rotation_matrix, dtype=float)


def matrix_to_euler(
    matrix,
    convention: EulerConvention,
    angle_unit: AngleUnit = AngleUnit.DEGREE,
) -> tuple[float, float, float]:
    """Decompose a canonical rotation matrix using a deterministic branch."""
    sequence = (
        "xyz"
        if EulerConvention(convention) is EulerConvention.EXTRINSIC_XYZ
        else "ZYZ"
    )
    degrees = AngleUnit(angle_unit) is AngleUnit.DEGREE
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        values = Rotation.from_matrix(np.asarray(matrix, dtype=float)).as_euler(
            sequence,
            degrees=degrees,
        )
    values[np.abs(values) < 1.0e-12] = 0.0
    return tuple(float(value) for value in values)


def convert_euler(
    angles: tuple[float, float, float],
    source: EulerConvention,
    target: EulerConvention,
    angle_unit: AngleUnit = AngleUnit.DEGREE,
) -> tuple[float, float, float]:
    """Convert Euler coordinates without changing the physical orientation."""
    source = EulerConvention(source)
    target = EulerConvention(target)
    if source is target:
        return tuple(float(value) for value in angles)
    return matrix_to_euler(
        orientation_matrix(angles, source, angle_unit),
        target,
        angle_unit,
    )


def tensors_have_same_physical_state(left, right, *, atol: float = 1.0e-10) -> bool:
    """Compare tensor physics independently of its Euler representation."""
    scalar_fields = (
        "delta_chi_ax",
        "delta_chi_rh",
        "rhombic_enabled",
        "susceptibility_unit",
    )
    if any(getattr(left, name) != getattr(right, name) for name in scalar_fields):
        return False
    return bool(
        np.allclose(
            orientation_matrix(
                left.euler_angles,
                left.frame_convention,
                left.angle_unit,
            ),
            orientation_matrix(
                right.euler_angles,
                right.frame_convention,
                right.angle_unit,
            ),
            rtol=0.0,
            atol=atol,
        )
    )


def tensor_in_convention(tensor, convention: EulerConvention):
    """Return an equivalent tensor expressed in ``convention``."""
    convention = EulerConvention(convention)
    if tensor.frame_convention is convention:
        return tensor
    angles = convert_euler(
        tensor.euler_angles,
        tensor.frame_convention,
        convention,
        tensor.angle_unit,
    )
    return replace(
        tensor,
        alpha=angles[0],
        beta=angles[1],
        gamma=angles[2],
        frame_convention=convention,
    )


__all__ = [
    "convert_euler",
    "matrix_to_euler",
    "orientation_matrix",
    "tensor_in_convention",
    "tensors_have_same_physical_state",
]
