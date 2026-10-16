"""Application policy for ordinary PCS forward predictions.

Tensor state and prediction policy are deliberately separate.  A canonical
tensor may retain an enabled rhombic component while compatibility workflows
continue to request axial predictions.  Consumers that represent the active
prediction model should obtain their forward tensor through this module.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, MutableMapping

from logic.pcs_forward_api import PCSTensor
from logic.pcs_tensor_state import ApplicationPCSTensor


PCS_PREDICTION_MODEL_KEY = "pcs_prediction_model"


class PredictionModel(str, Enum):
    AXIAL = "axial"
    FULL_TENSOR = "full_tensor"


def parse_prediction_model(value: Any) -> PredictionModel:
    """Return a safe model value; unknown project data remains axial."""
    if isinstance(value, PredictionModel):
        return value
    try:
        return PredictionModel(str(value).strip().lower())
    except (TypeError, ValueError):
        return PredictionModel.AXIAL


def get_prediction_model(state: Mapping[str, Any] | None) -> PredictionModel:
    if not isinstance(state, Mapping):
        return PredictionModel.AXIAL
    return parse_prediction_model(
        state.get(PCS_PREDICTION_MODEL_KEY, PredictionModel.AXIAL.value)
    )


def install_prediction_model(
    state: MutableMapping[str, Any],
    value: Any,
) -> PredictionModel:
    """Install normalized project state without publishing an event."""
    model = parse_prediction_model(value)
    state[PCS_PREDICTION_MODEL_KEY] = model.value
    return model


def prediction_uses_rhombic_term(
    model: PredictionModel | str,
    *,
    rhombic_enabled: bool,
) -> bool:
    return (
        parse_prediction_model(model) is PredictionModel.FULL_TENSOR
        and bool(rhombic_enabled)
    )


def prediction_tensor(
    tensor: ApplicationPCSTensor,
    model: PredictionModel | str,
    *,
    metal_position=(0.0, 0.0, 0.0),
) -> PCSTensor:
    """Project canonical tensor state into the selected prediction model."""
    if not isinstance(tensor, ApplicationPCSTensor):
        raise TypeError("tensor must be an ApplicationPCSTensor")
    return tensor.to_forward_tensor(
        metal_position=metal_position,
        rhombic_enabled=prediction_uses_rhombic_term(
            model,
            rhombic_enabled=tensor.rhombic_enabled,
        ),
    )


def prediction_tensor_from_components(
    *,
    dchi_ax: float,
    dchi_rh: float,
    rhombic_enabled: bool,
    model: PredictionModel | str,
) -> PCSTensor:
    """Projection helper for legacy paths whose coordinates are pre-rotated."""
    return PCSTensor(
        dchi_ax=float(dchi_ax),
        dchi_rh=float(dchi_rh),
        rhombic_enabled=prediction_uses_rhombic_term(
            model,
            rhombic_enabled=rhombic_enabled,
        ),
    )


__all__ = [
    "PCS_PREDICTION_MODEL_KEY",
    "PredictionModel",
    "get_prediction_model",
    "install_prediction_model",
    "parse_prediction_model",
    "prediction_tensor",
    "prediction_tensor_from_components",
    "prediction_uses_rhombic_term",
]
