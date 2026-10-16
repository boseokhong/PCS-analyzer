"""Pure unified point-dipole PCS forward model.

The engine consumes only the immutable contracts in :mod:`logic.pcs_forward_api`.
It does not read or mutate application state and intentionally provides no UI,
fitting, persistence, event, or NMR integration adapters.
"""

from __future__ import annotations

from math import acos, atan2, cos, inf, pi, sin, sqrt
from typing import Mapping, Sequence

from logic.pcs_forward_api import (
    Matrix3,
    PCSAtom,
    PCSAtomPrediction,
    PCS_FORWARD_API_VERSION,
    PCSPredictionRequest,
    PCSPredictionResult,
    PCSStructureTarget,
    PCSTargetPrediction,
    PCSTargetReducer,
    PCSTensor,
    TargetReduction,
)


PCS_FORWARD_ENGINE_ID = "pcs-point-dipole"
PCS_FORWARD_ENGINE_VERSION = "0.1"
PCS_SCALE_PPM = 1.0e4 / (12.0 * pi)


def _reduce_values(values: tuple[float, ...], mode: TargetReduction) -> float:
    if mode is TargetReduction.NONE:
        if len(values) != 1:
            raise ValueError("reduction='none' requires exactly one value")
        return values[0]
    if mode is TargetReduction.ARITHMETIC_MEAN:
        if not values:
            raise ValueError("cannot average an empty value collection")
        return sum(values) / len(values)
    raise ValueError(f"unsupported PCS target reduction: {mode!r}")


class ArithmeticPCSTargetReducer:
    """Reduce atom predictions according to the L0A target hierarchy."""

    def reduce(
        self,
        target: PCSStructureTarget,
        predictions_by_ref: Mapping[int, PCSAtomPrediction],
    ) -> PCSTargetPrediction:
        site_values: list[float] = []
        for member_ids in target.site_member_ref_ids:
            try:
                member_values = tuple(
                    predictions_by_ref[ref_id].pcs_ppm for ref_id in member_ids
                )
            except KeyError as exc:
                raise KeyError(
                    f"target {target.target_id!r} is missing prediction for "
                    f"Ref {int(exc.args[0])}"
                ) from exc
            site_values.append(
                _reduce_values(member_values, target.member_reduction)
            )

        predicted = _reduce_values(tuple(site_values), target.site_reduction)
        experimental = target.experimental_pcs_ppm
        residual = None if experimental is None else predicted - experimental
        return PCSTargetPrediction(
            target_id=target.target_id,
            pcs_ppm=predicted,
            site_values_ppm=tuple(site_values),
            member_ref_ids=target.member_ref_ids,
            experimental_pcs_ppm=experimental,
            residual_ppm=residual,
        )


class PCSForwardEngine:
    """Canonical axial/rhombic PCS point forward engine.

    Coordinates enter in the structure frame.  For each atom the model
    subtracts the metal position, applies the L0A structure-to-tensor rotation,
    evaluates the established ``Gax`` and ``Grh`` factors, and uses

    ``PCS = (dchi_ax * Gax + effective_dchi_rh * Grh) * 1e4 / (12*pi)``.

    A stored rhombic value contributes only when ``rhombic_enabled`` is true.
    At the metal origin both geometry factors are zero, matching the shared
    ``logic.symmetry_geometry`` behavior used by current table calculations.
    """

    api_version = PCS_FORWARD_API_VERSION
    engine_id = PCS_FORWARD_ENGINE_ID
    engine_version = PCS_FORWARD_ENGINE_VERSION

    def __init__(self, target_reducer: PCSTargetReducer | None = None) -> None:
        if target_reducer is not None and not isinstance(target_reducer, PCSTargetReducer):
            raise TypeError("target_reducer must implement PCSTargetReducer")
        self._target_reducer = (
            ArithmeticPCSTargetReducer()
            if target_reducer is None
            else target_reducer
        )

    def predict(self, request: PCSPredictionRequest) -> PCSPredictionResult:
        return self._predict(request, effective_geometry_by_ref=None)

    def predict_with_effective_geometry(
        self,
        request: PCSPredictionRequest,
        effective_geometry_by_ref: Mapping[int, Sequence[float]],
    ) -> PCSPredictionResult:
        """Predict with caller-supplied effective ``(Gax, Grh)`` factors.

        This compatibility boundary preserves existing pseudo-atom and
        torsional-ensemble geometry averaging.  PCS scaling remains owned by
        this engine; callers supply geometry factors, not calculated PCS.
        """

        if not isinstance(request, PCSPredictionRequest):
            raise TypeError("request must be a PCSPredictionRequest")
        if not isinstance(effective_geometry_by_ref, Mapping):
            raise TypeError("effective_geometry_by_ref must be a mapping")
        available = {atom.ref_id for atom in request.atoms}
        unknown = sorted(
            int(ref_id)
            for ref_id in effective_geometry_by_ref
            if int(ref_id) not in available
        )
        if unknown:
            raise ValueError(
                "effective geometry references atoms absent from the request: "
                + ", ".join(map(str, unknown))
            )
        normalized: dict[int, tuple[float, float]] = {}
        for ref_id, factors in effective_geometry_by_ref.items():
            values = tuple(float(value) for value in factors)
            if len(values) != 2:
                raise ValueError("effective geometry must contain exactly Gax and Grh")
            normalized[int(ref_id)] = (values[0], values[1])
        return self._predict(request, effective_geometry_by_ref=normalized)

    def _predict(
        self,
        request: PCSPredictionRequest,
        effective_geometry_by_ref: Mapping[int, tuple[float, float]] | None,
    ) -> PCSPredictionResult:
        if not isinstance(request, PCSPredictionRequest):
            raise TypeError("request must be a PCSPredictionRequest")

        tensor = request.tensor
        rotation = tensor.frame.rotation_matrix
        atom_predictions = tuple(
            self._predict_atom(
                atom,
                tensor,
                rotation,
                None
                if effective_geometry_by_ref is None
                else effective_geometry_by_ref.get(atom.ref_id),
            )
            for atom in request.atoms
        )
        predictions_by_ref = {
            prediction.ref_id: prediction for prediction in atom_predictions
        }
        target_predictions = tuple(
            self._target_reducer.reduce(target, predictions_by_ref)
            for target in request.targets
        )

        return PCSPredictionResult(
            request_id=request.request_id,
            atom_predictions=atom_predictions,
            target_predictions=target_predictions,
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            metadata={
                "equation": "axial_rhombic_point_dipole",
                "rhombic_enabled": request.tensor.rhombic_enabled,
                "effective_geometry": effective_geometry_by_ref is not None,
            },
        )

    @staticmethod
    def _predict_atom(
        atom: PCSAtom,
        tensor: PCSTensor,
        rotation: Matrix3,
        effective_geometry: tuple[float, float] | None,
    ) -> PCSAtomPrediction:
        frame = tensor.frame
        relative = tuple(
            atom.position[index] - frame.metal_position[index]
            for index in range(3)
        )
        x, y, z = tuple(
            sum(rotation[row][column] * relative[column] for column in range(3))
            for row in range(3)
        )

        distance = sqrt(x * x + y * y + z * z)
        distance_safe = inf if distance == 0.0 else distance
        if effective_geometry is None:
            cos_theta = max(-1.0, min(1.0, z / distance_safe))
            theta = acos(cos_theta)
            phi = atan2(y, x)
            inverse_r3 = 1.0 / (distance_safe**3)
            g_ax = (3.0 * cos(theta) ** 2 - 1.0) * inverse_r3
            g_rh = 1.5 * sin(theta) ** 2 * cos(2.0 * phi) * inverse_r3
        else:
            g_ax, g_rh = effective_geometry

        pcs_ppm = PCSForwardEngine._pcs_from_geometry_factors(
            tensor,
            g_ax,
            g_rh,
        )

        return PCSAtomPrediction(
            ref_id=atom.ref_id,
            pcs_ppm=pcs_ppm,
            g_ax=g_ax,
            g_rh=g_rh,
            distance_angstrom=distance,
        )

    @staticmethod
    def _pcs_from_geometry_factors(
        tensor: PCSTensor,
        g_ax: float,
        g_rh: float,
    ) -> float:
        return (
            tensor.dchi_ax * g_ax
            + tensor.effective_dchi_rh * g_rh
        ) * PCS_SCALE_PPM


def predict_pcs(request: PCSPredictionRequest) -> PCSPredictionResult:
    """Predict PCS with the default stateless engine."""

    return PCSForwardEngine().predict(request)


__all__ = [
    "ArithmeticPCSTargetReducer",
    "PCS_FORWARD_ENGINE_ID",
    "PCS_FORWARD_ENGINE_VERSION",
    "PCS_SCALE_PPM",
    "PCSForwardEngine",
    "predict_pcs",
]
