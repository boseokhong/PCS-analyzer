from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from scipy.signal import find_peaks


class InteractionMode(str, Enum):
    NAVIGATE = "navigate"
    ZOOM = "zoom"
    REFERENCE_PEAK = "reference_peak"
    PEAK_PICK = "peak_pick"


@dataclass(frozen=True)
class PeakCandidate:
    index: int
    x: float
    y: float


@dataclass
class InteractionState:
    mode: InteractionMode = InteractionMode.NAVIGATE
    hovered_peak: Optional[PeakCandidate] = None
    selected_peak: Optional[PeakCandidate] = None


class PeakCandidateCache:
    """
    Lightweight candidate cache for interactive hover selection.

    This is NOT the scientific peak-picking model.

    Candidates are rebuilt only when the displayed/committed spectrum
    changes. Mouse-motion handling should query this cache instead of
    repeatedly running scipy.signal.find_peaks().
    """

    def __init__(self) -> None:
        self._candidates: list[PeakCandidate] = []
        self._x_values = np.empty(0, dtype=float)

    @property
    def candidates(self) -> tuple[PeakCandidate, ...]:
        return tuple(self._candidates)

    def clear(self) -> None:
        self._candidates.clear()
        self._x_values = np.empty(0, dtype=float)

    def rebuild(
        self,
        axis: np.ndarray,
        intensities: np.ndarray,
        *,
        prominence: Optional[float] = None,
        distance: Optional[int] = None,
    ) -> None:
        axis = np.asarray(axis, dtype=float)
        values = np.asarray(intensities)

        if axis.ndim != 1 or values.ndim != 1:
            raise ValueError("Peak hover cache requires 1D spectrum data.")

        if axis.size != values.size:
            raise ValueError(
                "Spectrum axis and intensity arrays must have the same length."
            )

        if axis.size < 3:
            self.clear()
            return

        # Reference interaction follows the displayed physical spectrum.
        # For a normal phased spectrum this is the REAL component,
        # not magnitude mode.
        y = np.real(values).astype(float, copy=False)

        finite = np.isfinite(axis) & np.isfinite(y)
        if not np.all(finite):
            valid_indices = np.flatnonzero(finite)
            if valid_indices.size < 3:
                self.clear()
                return

            working_y = y[finite]
            working_axis = axis[finite]

            peaks, _ = find_peaks(
                working_y,
                prominence=prominence,
                distance=distance,
            )

            original_indices = valid_indices[peaks]

            candidates = [
                PeakCandidate(
                    index=int(original_index),
                    x=float(axis[original_index]),
                    y=float(y[original_index]),
                )
                for original_index in original_indices
            ]
        else:
            peaks, _ = find_peaks(
                y,
                prominence=prominence,
                distance=distance,
            )

            candidates = [
                PeakCandidate(
                    index=int(index),
                    x=float(axis[index]),
                    y=float(y[index]),
                )
                for index in peaks
            ]

        # Spectrum1D axis is normally ascending, but do not depend on
        # acquisition/display direction here.
        candidates.sort(key=lambda candidate: candidate.x)

        self._candidates = candidates
        self._x_values = np.asarray(
            [candidate.x for candidate in candidates],
            dtype=float,
        )

    def candidates_near_x(
        self,
        x: float,
        *,
        neighbours: int = 4,
    ) -> tuple[PeakCandidate, ...]:
        """
        Return a small set of candidates surrounding x.

        This is intentionally NOT the final snap decision. The Matplotlib
        adapter will later transform these candidates to display coordinates
        and apply a pixel-distance threshold.
        """
        if not self._candidates:
            return ()

        x = float(x)

        insert_at = int(np.searchsorted(self._x_values, x))

        start = max(0, insert_at - neighbours)
        stop = min(len(self._candidates), insert_at + neighbours + 1)

        return tuple(self._candidates[start:stop])


class SpectrumInteractionController:
    """
    Renderer-independent interaction state for the spectrum view.

    No Matplotlib objects should enter this public API.
    """

    def __init__(self) -> None:
        self.state = InteractionState()
        self.peak_cache = PeakCandidateCache()

    @property
    def mode(self) -> InteractionMode:
        return self.state.mode

    @property
    def hovered_peak(self) -> Optional[PeakCandidate]:
        return self.state.hovered_peak

    @property
    def selected_peak(self) -> Optional[PeakCandidate]:
        return self.state.selected_peak

    def set_mode(self, mode: InteractionMode | str) -> None:
        mode = InteractionMode(mode)

        if mode == self.state.mode:
            return

        self.state.mode = mode
        self.state.hovered_peak = None

    def enter_reference_mode(self) -> None:
        self.set_mode(InteractionMode.REFERENCE_PEAK)

    def enter_peak_pick_mode(self) -> None:
        self.set_mode(InteractionMode.PEAK_PICK)

    def enter_zoom_mode(self) -> None:
        self.set_mode(InteractionMode.ZOOM)

    def enter_navigate_mode(self) -> None:
        self.set_mode(InteractionMode.NAVIGATE)

    def set_hovered_peak(
        self,
        candidate: Optional[PeakCandidate],
    ) -> None:
        if self.state.mode not in {
            InteractionMode.REFERENCE_PEAK,
            InteractionMode.PEAK_PICK,
        }:
            self.state.hovered_peak = None
            return

        self.state.hovered_peak = candidate

    def confirm_hovered_peak(self) -> Optional[PeakCandidate]:
        """Confirm the highlighted local maximum for the active interaction mode.

        Reference selection is one-shot and returns to navigation. Peak-picking
        mode stays active so multiple peaks can be added consecutively.
        """
        if self.state.mode not in {
            InteractionMode.REFERENCE_PEAK,
            InteractionMode.PEAK_PICK,
        }:
            return None

        candidate = self.state.hovered_peak
        if candidate is None:
            return None

        if self.state.mode == InteractionMode.REFERENCE_PEAK:
            self.state.selected_peak = candidate
            self.state.mode = InteractionMode.NAVIGATE

        self.state.hovered_peak = None
        return candidate

    def clear_hover(self) -> None:
        self.state.hovered_peak = None

    def clear_selection(self) -> None:
        self.state.selected_peak = None
        self.state.hovered_peak = None

    def reset(self) -> None:
        """
        Reset transient interaction state when switching datasets/views.
        """
        self.state = InteractionState()
        self.peak_cache.clear()