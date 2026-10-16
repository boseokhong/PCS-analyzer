"""Compact assignment intent and selected-member confirmation, without fitting."""
import tkinter as tk
from tkinter import messagebox, ttk

from ..analysis.vt_observables import associate_track, unassociate_track, extract_vt_observable_series, confirm_member_associations
from .tracking_manager import TrackingManager
from .actions import action, polish


class VTObservableWindow(tk.Toplevel):
    def __init__(self, parent, state, *, on_activate, on_changed, on_committed):
        super().__init__(parent)
        self.title("VT Observable / Assignment Review")
        self.transient(parent)
        self.geometry("980x470")
        self.manager = TrackingManager(state)
        self.on_activate, self.on_changed = on_activate, on_changed
        self.on_committed = on_committed
        self._context, self.review = None, None
        self.assignment, self.nucleus, self.note = (tk.StringVar(self) for _ in range(3))
        self.status = tk.StringVar(self)
        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        self.title_label = ttk.Label(body)
        self.title_label.pack(anchor="w")
        ttk.Label(body, text="Associate sets reviewed assignment intent for this track. Confirm selected suggestions changes only those dataset observables.",
                  wraplength=940).pack(anchor="w")
        form = ttk.Frame(body)
        form.pack(fill="x", pady=8)
        for name, variable, width in (("Assignment", self.assignment, 22), ("Nucleus", self.nucleus, 8), ("Review note", self.note, 25)):
            ttk.Label(form, text=name).pack(side="left", padx=3)
            ttk.Entry(form, textvariable=variable, width=width).pack(side="left")
        action(form, "Associate", "apply", self._associate,
               description="Explicitly associate or change assignment intent for this track").pack(side="left", padx=5)
        action(form, "Remove", "remove", self._unassociate, description="Remove association").pack(side="left")
        self.points = ttk.Treeview(body, columns=("member", "K", "shift", "assignment", "observable", "action", "diagnostics"), show="headings", selectmode="extended")
        for name, width in (("member", 105), ("K", 60), ("shift", 75), ("assignment", 100), ("observable", 105), ("action", 125), ("diagnostics", 280)):
            titles = {"member": "Member", "K": "T / K", "shift": "δ / ppm", "assignment": "Assignment",
                      "observable": "Observable ID", "action": "Review action", "diagnostics": "Diagnostics"}
            self.points.heading(name, text=titles[name])
            self.points.column(name, width=width)
        self.points.pack(fill="both", expand=True)
        self.points.bind("<Double-1>", self._navigate)
        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=8)
        action(actions, "Refresh", "update", self.refresh, description="Refresh review").pack(side="left")
        action(actions, "Use selected", "use_existing_data", self._use_assignment, description="Use selected assignment").pack(side="left", padx=5)
        action(actions, "Show spectrum", "nmr_workbench", self._navigate).pack(side="left", padx=5)
        action(actions, "Confirm selected", "apply", self._confirm, primary=True,
               description="Confirm only the reviewed selected member association suggestions").pack(side="right")
        ttk.Label(body, textvariable=self.status, wraplength=940).pack(fill="x")
        polish(self)
        self.refresh()

    def refresh(self):
        series, track = self.manager.active_series, self.manager.active_track
        context = (series.id, track.id) if series and track else None
        association = series.observable_association(track.id) if context else None
        key = (context, association)
        if key != self._context:
            self.assignment.set(association.assignment if association else "")
            member = next((m for m in series.members if m.id == self.manager.state.selected_member_id), None) if series else None
            self.nucleus.set(association.nucleus if association else (member.dataset.conditions.nucleus or "" if member else ""))
            self.note.set(association.note if association else "")
            self._context = key
        selected = self.points.selection()
        self.points.delete(*self.points.get_children())
        self.review = extract_vt_observable_series(series, track.id) if context else None
        self.title_label.configure(text=f"Track: {track.label or track.id} · {association.method if association else 'No association'}" if track else "Select a track in tracking or analysis first.")
        if self.review:
            for row in self.review.points:
                obs = row.observation
                self.points.insert("", "end", iid=obs.identity.member_id, values=(obs.member_label or obs.identity.dataset_id[:8],
                    obs.temperature_k if obs.temperature_k is not None else obs.temperature_status,
                    obs.chemical_shift_ppm, row.assignment or "—", row.observable_id or "—", row.action or "blocked / no intent",
                    ", ".join(row.diagnostics)))
        self.points.selection_set([s for s in selected if self.points.exists(s)])
        self.status.set("Association is reviewed intent, not proof of assignment or structural identity. Ctrl/Shift selects members; confirmation changes only selected safe proposals.")

    def _run(self, action):
        try:
            action()
        except (KeyError, ValueError) as exc:
            self.status.set(str(exc))
            return
        self.on_changed()
        self.refresh()

    def _associate(self):
        def action():
            series, track = self._active()
            associate_track(series, track.id, assignment=self.assignment.get(), nucleus=self.nucleus.get(), note=self.note.get())
        self._run(action)

    def _use_assignment(self):
        if not self.review or not self.points.selection():
            return
        point = next(p for p in self.review.points if p.observation.identity.member_id == self.points.selection()[0])
        if not point.assignment or not point.nucleus:
            self.status.set("Selected member has no unique assigned observable to review.")
            return
        self.assignment.set(point.assignment)
        self.nucleus.set(point.nucleus)
        self.status.set("Assignment copied for review. Associate explicitly, then confirm individual member suggestions.")

    def _unassociate(self):
        def action():
            series, track = self._active()
            unassociate_track(series, track.id)
        self._run(action)

    def _active(self):
        if not self.manager.active_series or not self.manager.active_track:
            raise ValueError("Select a series track first")
        return self.manager.active_series, self.manager.active_track

    def _confirm(self):
        if not self.review or not self.points.selection():
            self.status.set("Select member proposals to review and confirm.")
            return
        selected = self.points.selection()
        review = self.review
        series = self.manager.active_series
        rows = [p for p in review.points if p.observation.identity.member_id in selected]
        description = "\n".join(f"{p.observation.identity.dataset_id[:8]}: {p.action or 'BLOCKED'} → {review.association.assignment if review.association else 'no intent'}" for p in rows)
        if any(p.action is None for p in rows):
            self.status.set("Selection includes blocked proposals. Review conflicts before confirming.")
            return
        if messagebox.askyesno("Confirm member observable associations", description + "\n\nCreate/update these dataset observables and save reviewed references?", parent=self):
            def commit():
                if self.manager.active_series is not series or not self.manager.active_track or self.manager.active_track.id != review.track_id:
                    raise ValueError("Active review changed; review again before confirming")
                confirm_member_associations(series, review, selected, confirmed=True)
                self.on_committed([(p.observation.identity.dataset_id, p.observation.identity.peak_id) for p in rows])
            self._run(commit)

    def _navigate(self, event=None):
        if not self.review or not self.points.selection():
            return
        row = next(p for p in self.review.points if p.observation.identity.member_id == self.points.selection()[0])
        def action():
            identity = row.observation.identity
            member = self.manager.select_peak(identity.member_id, identity.peak_id)
            self.on_activate(member.dataset_id, identity.peak_id)
        self._run(action)
