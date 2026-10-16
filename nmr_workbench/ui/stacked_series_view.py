"""Dedicated VT renderer and member-aware navigation, separate from processing."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

from .presentation import WorkbenchFigureCanvas
from .series_manager import SeriesManager
from .series_traces import prepare_series_traces
from .view_model import dataset_display_name
from .actions import action, divider


class StackedSeriesView(ttk.Frame):
    def __init__(self, parent, model, *, on_member_selected, on_series_selected,
                 on_peak_selected=None, on_tracking_requested=None, on_analysis_requested=None, on_batch_requested=None):
        super().__init__(parent)
        self.manager = SeriesManager(model)
        self.on_member_selected = on_member_selected
        self.on_series_selected = on_series_selected
        self.on_peak_selected = on_peak_selected
        self._peak_targets = []
        self._peak_artists = []
        self._tracking_key = None
        self.order = tk.StringVar(self, value="Acquisition")
        self.spacing = tk.StringVar(self, value="1.2")
        self.normalize = tk.BooleanVar(self, value=True)
        self.status = tk.StringVar(self)
        self._series_ids = []
        self._lines = []
        self._layout_key = None
        self._data_key = None
        self._highlighted_member = None
        self._labels = {}
        self._press = None
        self.prepared = None
        controls = ttk.Frame(self)
        controls.pack(fill="x", pady=(0, 4))
        self.selector = ttk.Combobox(controls, width=16, state="readonly")
        self.selector.pack(side="left", fill="x", expand=True)
        self.selector.bind("<<ComboboxSelected>>", self._select_series)
        display = ttk.Frame(self)
        display.pack(fill="x", pady=(0, 6))
        order = ttk.Combobox(display, width=15, state="readonly", textvariable=self.order,
                             values=("Acquisition", "Temperature ↑", "Temperature ↓"))
        order.pack(side="left", padx=4)
        order.bind("<<ComboboxSelected>>", lambda event: self.refresh())
        ttk.Label(display, text="Spacing ×").pack(side="left")
        spacing = ttk.Spinbox(display, from_=0, to=20, increment=.2, width=5,
                              textvariable=self.spacing, command=self.refresh)
        spacing.pack(side="left", padx=4)
        spacing.bind("<Return>", lambda event: self.refresh())
        spacing.bind("<FocusOut>", lambda event: self.refresh())
        ttk.Checkbutton(display, text="Normalize display", variable=self.normalize,
                        command=self.refresh).pack(side="left")
        if on_tracking_requested:
            divider(controls)
            action(controls, "Tracks", "point_select", on_tracking_requested).pack(side="left", padx=3)
        if on_analysis_requested:
            action(controls, "Analysis", "graph", on_analysis_requested).pack(side="left", padx=3)
        if on_batch_requested:
            action(controls, "Batch", "run", on_batch_requested, description="Batch processing").pack(side="left", padx=3)
        self.figure = Figure(figsize=(8, 5))
        self.figure.subplots_adjust(left=.105, right=.985, bottom=.13, top=.965)
        self.axes = self.figure.add_subplot(111)
        self.canvas = WorkbenchFigureCanvas(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.pack(fill="x")
        ttk.Label(self, textvariable=self.status, wraplength=650,
                  style="NMR.Muted.TLabel").pack(fill="x")
        self._cids = [self.canvas.mpl_connect("button_press_event", self._on_press),
                      self.canvas.mpl_connect("button_release_event", self._on_release)]

    def _select_series(self, event=None):
        index = self.selector.current()
        if index >= 0:
            self.manager.select_series(self._series_ids[index])
            self.on_series_selected()
            self.refresh()

    def refresh(self, *, preserve_view=True):
        state = self.manager.state
        self._series_ids = [s.id for s in state.series]
        self.selector.configure(values=[s.name for s in state.series])
        self.selector.set("")
        if state.selected_series_id in self._series_ids:
            self.selector.current(self._series_ids.index(state.selected_series_id))
        series = self.manager.active_series
        try:
            spacing = float(self.spacing.get())
            if not np.isfinite(spacing) or spacing < 0:
                raise ValueError("Spacing must be a finite non-negative number.")
            data_key = (state.selected_series_id, self.order.get(), spacing, self.normalize.get(),
                        tuple(self._member_signature(m, series) for m in series.members) if series else ())
            if preserve_view and data_key == self._data_key:
                # Preview/UI refreshes cannot change committed spectrum data.
                # Reuse display arrays and update only member highlighting.
                if self._highlighted_member != state.selected_member_id:
                    for trace, line in self._lines:
                        active = trace.member_id == state.selected_member_id
                        line.set_linewidth(1.8 if active else .8)
                        line.set_color("#D97706" if active else "#2459A6")
                        self._labels[trace.member_id].set_fontweight("bold" if active else "normal")
                    self._highlighted_member = state.selected_member_id
                    self.canvas.draw_idle()
                self._refresh_status()
                self._refresh_tracking()
                return
            prepared = prepare_series_traces(series, order=self.order.get(), spacing=spacing,
                                               normalize=self.normalize.get()) if series else None
        except (ValueError, OverflowError) as exc:
            self.status.set(str(exc))
            return  # Keep the previous valid rendering for incomplete entry edits.
        key = (state.selected_series_id, self.order.get(), spacing, self.normalize.get(),
               tuple(t.member_id for t in prepared.traces) if prepared else ())
        same_layout = key == self._layout_key
        limits = (self.axes.get_xlim(), self.axes.get_ylim()) if same_layout and preserve_view else None
        self.axes.clear()
        self._peak_artists = []
        self._tracking_key = None
        self._lines = []
        self._labels = {}
        self._press = None
        self.prepared = prepared
        self._layout_key = key
        self._data_key = data_key
        self._highlighted_member = state.selected_member_id
        if prepared is None or not prepared.traces:
            message = "Select a VT series. Use VT Series… to create or manage one." if series is None else "No included members have committed ppm spectra."
            self.axes.text(.5, .5, message, ha="center", va="center", transform=self.axes.transAxes)
            self.axes.set_axis_off()
        else:
            self.axes.set_axis_on()
            self.axes.set_xlabel("Chemical shift / ppm")
            self.axes.set_ylabel("Normalized intensity + offset" if self.normalize.get() else "Intensity (a.u.) + offset")
            for trace in prepared.traces:
                active = trace.member_id == state.selected_member_id
                line, = self.axes.plot(trace.axis_ppm, trace.display_values,
                                       linewidth=1.8 if active else .8,
                                       color="#D97706" if active else "#2459A6",
                                       pickradius=6)
                self._lines.append((trace, line))
                self._labels[trace.member_id] = self.axes.text(
                    .01, trace.offset, trace.label, transform=self.axes.get_yaxis_transform(),
                    va="bottom", fontsize="small", fontweight="bold" if active else "normal")
            self.axes.grid(True, alpha=.15)
            self.axes.margins(x=.01, y=.08)
            self.axes.relim()
            self.axes.autoscale_view()
            self.axes.invert_xaxis()
        if limits:
            self.axes.set_xlim(limits[0])
            self.axes.set_ylim(limits[1])
        if not same_layout:
            self.toolbar.update()
            self.toolbar.push_current()
        self._refresh_status()
        self._refresh_tracking()
        self.canvas.draw_idle()

    def _refresh_tracking(self):
        from .tracking_manager import TrackingManager
        self.manager.state.update_tracks()
        series = self.manager.active_series
        track = TrackingManager(self.manager.state).active_track
        key = (repr(track.to_dict()) if track else None, self.manager.state.selected_peak_id,
               self.manager.state.selected_member_id, id(self.prepared),
               tuple((m.id, tuple((p.id, p.position_ppm) for p in m.dataset.peaks)) for m in series.members) if series else ())
        if key == self._tracking_key:
            return
        self._tracking_key = key
        for artist in self._peak_artists:
            artist.remove()
        self._peak_artists, self._peak_targets = [], []
        members = {m.id: m for m in series.members} if series else {}
        for trace in self.prepared.traces if self.prepared else ():
            member = members[trace.member_id]
            finite = np.isfinite(trace.axis_ppm) & np.isfinite(trace.display_values)
            indices = np.flatnonzero(finite)
            if not indices.size:
                continue
            point = track.get_point(member.id) if track else None
            for peak in member.dataset.peaks:
                if not np.isfinite(peak.position_ppm) or not trace.axis_ppm[finite].min() <= peak.position_ppm <= trace.axis_ppm[finite].max():
                    continue
                index = indices[np.argmin(np.abs(trace.axis_ppm[indices] - peak.position_ppm))]
                y = trace.display_values[index]  # native sample for display only
                linked = point and point.peak_id == peak.id and point.dataset_id == member.dataset_id
                selected = member.id == self.manager.state.selected_member_id and peak.id == self.manager.state.selected_peak_id
                color = "#B45309" if linked and point.state == "stale" else "#15803D" if linked else "#6B7280"
                marker = "x" if linked and point.state == "stale" else "o"
                artist, = self.axes.plot([peak.position_ppm], [y], linestyle="none", marker=marker,
                    color=color, markersize=7 if linked else 4, zorder=5)
                self._peak_artists.append(artist)
                if selected:
                    ring, = self.axes.plot([peak.position_ppm], [y], linestyle="none", marker="o",
                        markerfacecolor="none", color="#7C3AED", markersize=11, zorder=6)
                    self._peak_artists.append(ring)
                self._peak_targets.append((member.id, member.dataset_id, peak.id, peak.position_ppm, y))
        self.canvas.draw_idle()

    def peak_hit_test(self, event):
        if event.inaxes is not self.axes:
            return None
        candidates = []
        for target in self._peak_targets:
            x, y = self.axes.transData.transform(target[3:])
            distance = np.hypot(event.x - x, event.y - y)
            if distance <= 8:
                candidates.append((distance, target))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    @staticmethod
    def _member_signature(member, series):
        spectrum = member.dataset.processed_spectrum
        temperature = member.resolve_temperature(series_override_k=series.temperature_override_k)
        return (member.id, member.included, member.label, member.acquisition_order,
                member.dataset.spectral_revision,
                dataset_display_name(member.dataset), temperature,
                id(spectrum), id(spectrum.axis) if spectrum else None,
                id(spectrum.intensities) if spectrum else None,
                (spectrum.domain, spectrum.axis_unit) if spectrum else None)

    def _refresh_status(self):
        unavailable = self.prepared.unavailable if self.prepared else ()
        self.status.set("Committed spectra only · Click a trace/peak to select. Grey: peaks · Green: linked · Amber ×: stale · Purple: selected. " + " · ".join(unavailable))

    def reset_view(self):
        self.refresh(preserve_view=False)

    def autoscale_y(self):
        low, high = sorted(self.axes.get_xlim())
        ranges = []
        for trace, _ in self._lines:
            values = trace.display_values[(trace.axis_ppm >= low) & (trace.axis_ppm <= high)]
            values = values[np.isfinite(values)]
            if values.size:
                ranges.extend((float(values.min()), float(values.max())))
        if ranges:
            low, high = min(ranges), max(ranges)
            pad = (high - low) * .08 or 1.0
            self.axes.set_ylim(low - pad, high + pad)
            self.canvas.draw_idle()

    def hit_test(self, event):
        """Nearest visible trace segment in screen space, retaining member identity."""
        if event.inaxes is not self.axes:
            return None
        candidates = []
        mouse = np.array([event.x, event.y])
        for trace, line in self._lines:
            contains, details = line.contains(event)
            if not contains:
                continue
            indices = np.asarray(details["ind"], dtype=int)
            if not indices.size:
                continue
            # Segment geometry is used solely for picking, not spectral resampling.
            next_indices = np.minimum(indices + 1, trace.axis_ppm.size - 1)
            start = self.axes.transData.transform(np.column_stack((trace.axis_ppm[indices], trace.display_values[indices])))
            end = self.axes.transData.transform(np.column_stack((trace.axis_ppm[next_indices], trace.display_values[next_indices])))
            valid = np.isfinite(start).all(axis=1)
            start, end = start[valid], end[valid]
            if not start.size:
                continue
            # A finite point beside a NaN gap remains a valid hit target.
            end = np.where(np.isfinite(end).all(axis=1)[:, None], end, start)
            vector = end - start
            length_sq = np.sum(vector * vector, axis=1)
            fraction = np.divide(np.sum((mouse - start) * vector, axis=1), length_sq,
                                 out=np.zeros_like(length_sq), where=length_sq > 0)
            nearest = start + np.clip(fraction, 0, 1)[:, None] * vector
            distance = float(np.min(np.linalg.norm(nearest - mouse, axis=1)))
            candidates.append((distance, trace))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _on_press(self, event):
        self._press = (event.x, event.y) if event.button == 1 and event.inaxes is self.axes and not self.toolbar.mode else None

    def _on_release(self, event):
        press, self._press = self._press, None
        if press is None or event.button != 1 or self.toolbar.mode:
            return
        if np.hypot(event.x - press[0], event.y - press[1]) > 4:
            return
        peak = self.peak_hit_test(event)
        if peak and self.on_peak_selected:
            self.manager.select_member(self.manager.active_series.id, peak[0])
            self.on_peak_selected(peak[1], peak[2])
            self.refresh()
            return
        trace = self.hit_test(event)
        if trace:
            self.manager.select_member(trace.series_id, trace.member_id)
            self.on_member_selected(trace.dataset_id)
            self.refresh()

    def destroy(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        super().destroy()
