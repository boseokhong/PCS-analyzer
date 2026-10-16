from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib.patches import Rectangle

from .spectrum_interaction import (
    InteractionMode,
    PeakCandidate,
    SpectrumInteractionController,
)


@dataclass(frozen=True)
class HoverMatch:
    candidate: PeakCandidate
    distance_px: float


class MatplotlibSpectrumInteractionAdapter:
    """Matplotlib bridge for renderer-independent spectrum interaction state.

    The adapter owns renderer-specific mouse behavior:
    - hover/click peak snapping for Peak Pick / Reference modes
    - left-drag horizontal zoom in Zoom mode
    - middle/right-drag pan
    - mouse-wheel vertical intensity scaling
    - Shift + mouse-wheel horizontal zoom

    The controller remains free of Matplotlib objects.
    """

    def __init__(
        self,
        *,
        axes,
        canvas,
        controller: SpectrumInteractionController,
        hover_threshold_px: float = 10.0,
        on_peak_selected=None,
        on_peak_picked=None,
        on_view_changed=None,
        on_navigate_click=None,
    ) -> None:
        self.axes = axes
        self.canvas = canvas
        self.controller = controller
        self.hover_threshold_px = float(hover_threshold_px)
        self.on_peak_selected = on_peak_selected
        self.on_peak_picked = on_peak_picked
        self.on_view_changed = on_view_changed
        self.on_navigate_click = on_navigate_click

        self._motion_cid: Optional[int] = None
        self._leave_cid: Optional[int] = None
        self._press_cid: Optional[int] = None
        self._release_cid: Optional[int] = None
        self._scroll_cid: Optional[int] = None

        self._hover_marker = None
        self._hover_annotation = None
        self._selected_marker = None
        self._selected_annotation = None

        self._zoom_start_x: float | None = None
        self._zoom_start_px: float | None = None
        self._zoom_rect = None

        self._pan_button: int | None = None
        self._pan_start_x: float | None = None
        self._pan_start_y: float | None = None
        self._pan_xlim: tuple[float, float] | None = None
        self._pan_ylim: tuple[float, float] | None = None

    def connect(self) -> None:
        if self._motion_cid is None:
            self._motion_cid = self.canvas.mpl_connect(
                "motion_notify_event", self._on_motion
            )
        if self._leave_cid is None:
            self._leave_cid = self.canvas.mpl_connect(
                "axes_leave_event", self._on_axes_leave
            )
        if self._press_cid is None:
            self._press_cid = self.canvas.mpl_connect(
                "button_press_event", self._on_button_press
            )
        if self._release_cid is None:
            self._release_cid = self.canvas.mpl_connect(
                "button_release_event", self._on_button_release
            )
        if self._scroll_cid is None:
            self._scroll_cid = self.canvas.mpl_connect(
                "scroll_event", self._on_scroll
            )

    def disconnect(self) -> None:
        for name in (
            "_motion_cid",
            "_leave_cid",
            "_press_cid",
            "_release_cid",
            "_scroll_cid",
        ):
            cid = getattr(self, name)
            if cid is not None:
                self.canvas.mpl_disconnect(cid)
                setattr(self, name, None)

        self.cancel_drag()
        self.clear_hover_artists()
        self.clear_selection_artists()

    def _toolbar_active(self) -> bool:
        toolbar = getattr(self.canvas, "toolbar", None)
        if toolbar is None:
            return False
        mode = getattr(toolbar, "mode", "")
        return bool(mode)

    def _on_button_press(self, event) -> None:
        if event.inaxes is not self.axes:
            return
        if self._toolbar_active():
            return

        button = getattr(event, "button", None)

        # Middle or right drag pans in any custom mode.
        if button in (2, 3):
            if event.xdata is None or event.ydata is None:
                return
            self._pan_button = int(button)
            self._pan_start_x = float(event.xdata)
            self._pan_start_y = float(event.ydata)
            self._pan_xlim = tuple(map(float, self.axes.get_xlim()))
            self._pan_ylim = tuple(map(float, self.axes.get_ylim()))
            return

        if button != 1:
            return

        mode = self.controller.mode

        if mode == InteractionMode.ZOOM:
            if event.xdata is None:
                return
            self._zoom_start_x = float(event.xdata)
            self._zoom_start_px = float(event.x)
            self._begin_zoom_rectangle(self._zoom_start_x)
            return

        if mode == InteractionMode.NAVIGATE:
            if (
                self.on_navigate_click is not None
                and event.xdata is not None
                and event.ydata is not None
            ):
                self.on_navigate_click(event)
            return

        if mode not in {InteractionMode.REFERENCE_PEAK, InteractionMode.PEAK_PICK}:
            return

        selected = self.controller.confirm_hovered_peak()
        if selected is None:
            return

        self._hide_hover_artists()

        if mode == InteractionMode.REFERENCE_PEAK:
            self._show_selected(selected)
            if self.on_peak_selected is not None:
                self.on_peak_selected(selected)
        elif self.on_peak_picked is not None:
            self.on_peak_picked(selected)

    def _on_button_release(self, event) -> None:
        if self._pan_button is not None:
            self._clear_pan_state()
            self._notify_view_changed()
            return

        if self._zoom_start_x is None:
            return

        start_x = self._zoom_start_x
        start_px = self._zoom_start_px
        end_x = float(event.xdata) if event.inaxes is self.axes and event.xdata is not None else None
        end_px = float(event.x) if getattr(event, "x", None) is not None else None

        self._clear_zoom_rectangle()

        if end_x is None or end_px is None or start_px is None:
            return
        if abs(end_px - start_px) < 5.0:
            return

        lo, hi = sorted((start_x, end_x))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return

        current_left, current_right = self.axes.get_xlim()
        if current_left > current_right:
            self.axes.set_xlim(hi, lo)
        else:
            self.axes.set_xlim(lo, hi)

        self.canvas.draw_idle()
        self._notify_view_changed()

    def _on_motion(self, event) -> None:
        if self._pan_button is not None:
            self._update_pan(event)
            return

        if self._zoom_start_x is not None:
            if event.inaxes is self.axes and event.xdata is not None:
                self._update_zoom_rectangle(float(event.xdata))
            return

        if event.inaxes is not self.axes:
            self._clear_hover()
            return

        if event.xdata is None or event.ydata is None:
            self._clear_hover()
            return

        if self.controller.mode not in {
            InteractionMode.REFERENCE_PEAK,
            InteractionMode.PEAK_PICK,
        }:
            self._clear_hover()
            return

        match = self.find_hover_match(
            xdata=float(event.xdata),
            mouse_x_px=float(event.x),
            mouse_y_px=float(event.y),
        )

        if match is None:
            self._clear_hover()
            return

        self.controller.set_hovered_peak(match.candidate)
        self._show_hover(match.candidate)

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.axes or self._toolbar_active():
            return

        step = getattr(event, "step", 0)
        if step == 0:
            return
        factor = 0.80 if step > 0 else 1.25

        key = str(getattr(event, "key", "") or "").lower()
        if "shift" in key:
            # NMR-style navigation: Shift + wheel changes the horizontal
            # spectral window around the pointer.
            self._scale_x(factor, event.xdata)
        else:
            # Plain wheel changes vertical intensity scaling.
            self._scale_y(factor, event.ydata)

        self.canvas.draw_idle()
        self._notify_view_changed()

    def _scale_x(self, factor: float, anchor: float | None) -> None:
        left, right = map(float, self.axes.get_xlim())
        if anchor is None or not np.isfinite(anchor):
            anchor = (left + right) / 2.0
        self.axes.set_xlim(
            anchor + (left - anchor) * factor,
            anchor + (right - anchor) * factor,
        )

    def _scale_y(self, factor: float, anchor: float | None) -> None:
        bottom, top = map(float, self.axes.get_ylim())
        if anchor is None or not np.isfinite(anchor):
            anchor = (bottom + top) / 2.0
        self.axes.set_ylim(
            anchor + (bottom - anchor) * factor,
            anchor + (top - anchor) * factor,
        )

    def _update_pan(self, event) -> None:
        if (
            event.inaxes is not self.axes
            or event.xdata is None
            or event.ydata is None
            or self._pan_start_x is None
            or self._pan_start_y is None
            or self._pan_xlim is None
            or self._pan_ylim is None
        ):
            return

        dx = float(event.xdata) - self._pan_start_x
        dy = float(event.ydata) - self._pan_start_y
        self.axes.set_xlim(self._pan_xlim[0] - dx, self._pan_xlim[1] - dx)
        self.axes.set_ylim(self._pan_ylim[0] - dy, self._pan_ylim[1] - dy)
        self.canvas.draw_idle()

    def _clear_pan_state(self) -> None:
        self._pan_button = None
        self._pan_start_x = None
        self._pan_start_y = None
        self._pan_xlim = None
        self._pan_ylim = None

    def _begin_zoom_rectangle(self, x: float) -> None:
        self._clear_zoom_artist_only()
        ymin, ymax = self.axes.get_ylim()
        self._zoom_rect = Rectangle(
            (x, ymin),
            0.0,
            ymax - ymin,
            fill=False,
            linestyle="--",
            linewidth=1.0,
            edgecolor="0.35",
            zorder=12,
        )
        self.axes.add_patch(self._zoom_rect)
        self.canvas.draw_idle()

    def _update_zoom_rectangle(self, x: float) -> None:
        if self._zoom_rect is None or self._zoom_start_x is None:
            return
        ymin, ymax = self.axes.get_ylim()
        lo = min(self._zoom_start_x, x)
        hi = max(self._zoom_start_x, x)
        self._zoom_rect.set_x(lo)
        self._zoom_rect.set_width(hi - lo)
        self._zoom_rect.set_y(ymin)
        self._zoom_rect.set_height(ymax - ymin)
        self.canvas.draw_idle()

    def _clear_zoom_artist_only(self) -> None:
        if self._zoom_rect is not None:
            try:
                self._zoom_rect.remove()
            except ValueError:
                pass
            self._zoom_rect = None

    def _clear_zoom_rectangle(self) -> None:
        self._clear_zoom_artist_only()
        self._zoom_start_x = None
        self._zoom_start_px = None
        self.canvas.draw_idle()

    def cancel_drag(self) -> None:
        self._clear_pan_state()
        self._clear_zoom_rectangle()

    def _notify_view_changed(self) -> None:
        if self.on_view_changed is not None:
            self.on_view_changed()

    def _on_axes_leave(self, _event) -> None:
        if self._pan_button is None and self._zoom_start_x is None:
            self._clear_hover()

    def find_hover_match(
        self,
        *,
        xdata: float,
        mouse_x_px: float,
        mouse_y_px: float,
    ) -> Optional[HoverMatch]:
        nearby = self.controller.peak_cache.candidates_near_x(
            xdata,
            neighbours=6,
        )
        if not nearby:
            return None

        best: Optional[HoverMatch] = None
        for candidate in nearby:
            peak_px = self.axes.transData.transform((candidate.x, candidate.y))
            dx = float(peak_px[0]) - mouse_x_px
            dy = float(peak_px[1]) - mouse_y_px
            distance = float(np.hypot(dx, dy))
            if distance > self.hover_threshold_px:
                continue
            if best is None or distance < best.distance_px:
                best = HoverMatch(candidate=candidate, distance_px=distance)
        return best

    def _show_hover(self, candidate: PeakCandidate) -> None:
        if self._hover_marker is None:
            (self._hover_marker,) = self.axes.plot(
                [candidate.x],
                [candidate.y],
                marker="o",
                linestyle="None",
                markersize=6,
                zorder=10,
            )
        else:
            self._hover_marker.set_data([candidate.x], [candidate.y])
            self._hover_marker.set_visible(True)

        label = f"{candidate.x:.4f} ppm"
        if self._hover_annotation is None:
            self._hover_annotation = self.axes.annotate(
                label,
                xy=(candidate.x, candidate.y),
                xytext=(8, 10),
                textcoords="offset points",
                annotation_clip=True,
            )
        else:
            self._hover_annotation.xy = (candidate.x, candidate.y)
            self._hover_annotation.set_text(label)
            self._hover_annotation.set_visible(True)
        self.canvas.draw_idle()

    def _show_selected(self, candidate: PeakCandidate) -> None:
        if self._selected_marker is None:
            (self._selected_marker,) = self.axes.plot(
                [candidate.x],
                [candidate.y],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor="none",
                markeredgewidth=1.5,
                zorder=11,
                label="_nolegend_",
            )
        else:
            self._selected_marker.set_data([candidate.x], [candidate.y])
            self._selected_marker.set_visible(True)

        label = f"Observed: {candidate.x:.4f} ppm"
        if self._selected_annotation is None:
            self._selected_annotation = self.axes.annotate(
                label,
                xy=(candidate.x, candidate.y),
                xytext=(8, 12),
                textcoords="offset points",
                annotation_clip=True,
            )
        else:
            self._selected_annotation.xy = (candidate.x, candidate.y)
            self._selected_annotation.set_text(label)
            self._selected_annotation.set_visible(True)
        self.canvas.draw_idle()

    def _hide_hover_artists(self) -> None:
        changed = False
        if self._hover_marker is not None and self._hover_marker.get_visible():
            self._hover_marker.set_visible(False)
            changed = True
        if self._hover_annotation is not None and self._hover_annotation.get_visible():
            self._hover_annotation.set_visible(False)
            changed = True
        if changed:
            self.canvas.draw_idle()

    def _clear_hover(self) -> None:
        self.controller.clear_hover()
        self._hide_hover_artists()

    def clear_hover_artists(self) -> None:
        if self._hover_marker is not None:
            try:
                self._hover_marker.remove()
            except ValueError:
                pass
            self._hover_marker = None
        if self._hover_annotation is not None:
            try:
                self._hover_annotation.remove()
            except ValueError:
                pass
            self._hover_annotation = None
        self.controller.clear_hover()
        self.canvas.draw_idle()

    def detach_selection_artists(self) -> None:
        if self._selected_marker is not None:
            try:
                self._selected_marker.remove()
            except ValueError:
                pass
            self._selected_marker = None
        if self._selected_annotation is not None:
            try:
                self._selected_annotation.remove()
            except ValueError:
                pass
            self._selected_annotation = None

    def restore_selection_artists(self) -> None:
        candidate = self.controller.selected_peak
        if candidate is not None:
            self._show_selected(candidate)

    def clear_selection_artists(self) -> None:
        self.detach_selection_artists()
        self.controller.clear_selection()
        self.canvas.draw_idle()
