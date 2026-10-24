"""UI-independent calculation snapshots and result lifetime for PCS Workbench."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np


CALCULATION_KEYS = (
    "temperature", "fft_pad_factor", "normalize_density", "normalization_target",
    "normalization_stage",
)


def calculation_signature(params):
    return tuple(params.get(key) for key in CALCULATION_KEYS)


def validate_calculation_params(params):
    padding = params.get("fft_pad_factor", 2)
    if not np.isfinite(padding) or int(padding) != padding or padding < 0:
        raise ValueError("Padding factor must be a non-negative integer.")
    target = params.get("normalization_target", 1.0)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Normalization target must be positive and finite.")
    temperature = params.get("temperature")
    if temperature is not None and (not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("Temperature must be positive and finite.")
    if params.get("normalization_stage", "before_padding") not in ("before_padding", "after_padding"):
        raise ValueError("Select a supported normalization stage.")


def frozen_array(value):
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CalculationSnapshot:
    revision: int
    atoms: tuple
    origin: np.ndarray
    spacing: tuple
    rho: np.ndarray
    chi: np.ndarray
    chi_raw: np.ndarray
    temperature: float
    params: dict
    orca_path: str
    dens_path: str
    source_origin: np.ndarray
    source_shape: tuple
    signed_density: np.ndarray | None
    tensor_source: dict

    def compute_kwargs(self):
        return dict(atoms=self.atoms, origin=self.origin, spacing=self.spacing,
                    rho=self.rho, chi_tensor=self.chi, params=deepcopy(self.params),
                    chi_raw=self.chi_raw, temperature=self.temperature)


class AppSession:
    def __init__(self):
        self.orca_path = self.dens_path = None
        self.orca = self.dens = None
        self.temperature = self.chi_raw = self.chi = None
        self.last_result = self.last_view_params = self.viewer_plotter = None
        self.revision = 0
        self.status = "Input required"
        self.last_error = ""
        self.last_traceback = ""

    def invalidate(self, reason):
        self.revision += 1
        self.last_result = None
        self.last_error = self.last_traceback = ""
        self.status = reason

    def snapshot(self, params, temperature, chi_raw, chi):
        validate_calculation_params(params)
        if self.orca is None or self.dens is None:
            raise ValueError("Load an ORCA output and a density grid first.")
        signed = self.dens.get("rho_signed")
        return CalculationSnapshot(
            self.revision, tuple(tuple(atom) for atom in self.orca["atoms"]),
            frozen_array(self.dens["origin"]), tuple(self.dens["spacing"]),
            frozen_array(self.dens["rho"]), frozen_array(chi), frozen_array(chi_raw),
            float(temperature), deepcopy(params), self.orca_path, self.dens_path,
            frozen_array(self.dens["origin"]), tuple(self.dens["rho"].shape),
            frozen_array(signed) if signed is not None else None,
            {"section": self.orca.get("selected_tensor_section"),
             "line": self.orca.get("tensor_sections", [{}])[
                 self.orca.get("selected_tensor_section", 1) - 1
             ].get("line") if self.orca.get("tensor_sections") else None},
        )

    def accept_result(self, snapshot, result):
        if snapshot.revision != self.revision:
            return False
        result.update(
            input_files={"orca": snapshot.orca_path, "density": snapshot.dens_path},
            calculation_params=deepcopy(snapshot.params),
            source_origin=snapshot.source_origin, source_shape=snapshot.source_shape,
            source_spacing=snapshot.spacing, source_signed_density=snapshot.signed_density,
            tensor_source=snapshot.tensor_source,
        )
        self.last_result = result
        self.status = "Result current"
        self.last_error = self.last_traceback = ""
        return True


def estimate_calculation_memory(shape, padding):
    """Conservative working-array estimate; excludes renderer and FFT plan caches."""
    if int(padding) != padding or padding < 0:
        raise ValueError("Padding factor must be a non-negative integer.")
    padded = tuple(int(n) * (2 * int(padding) + 1) for n in shape)
    full = int(np.prod(padded, dtype=object))
    spectrum = padded[0] * padded[1] * (padded[2] // 2 + 1)
    return padded, full * 8 * 5 + spectrum * 16 * 7
