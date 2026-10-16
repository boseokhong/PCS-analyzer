"""Compact manual-first VT correspondence review, separate from processing."""

import tkinter as tk
from tkinter import messagebox, ttk

from .tracking_manager import TrackingManager
from .view_model import dataset_display_name
from .actions import action, action_menu, polish


class TrackingWindow(tk.Toplevel):
    def __init__(self, parent, model, *, on_activate, on_changed, order):
        super().__init__(parent)
        self.title("VT Peak Tracking — NMR Workbench")
        self.transient(parent)
        self.geometry("940x600")
        self.minsize(780, 480)
        self.manager = TrackingManager(model)
        self.on_activate, self.on_changed, self.order = on_activate, on_changed, order
        self._refreshing = False
        self._track_ids, self._peak_ids, self._suggestions = [], [], ()
        self.label = tk.StringVar(self)
        self.window_ppm = tk.StringVar(self, value="0.15")
        self.status = tk.StringVar(self, value="Choose a member and existing peak. Link explicitly, or mark missing/ambiguous.")
        self.provenance = tk.StringVar(self)
        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        top = ttk.Frame(body)
        top.pack(fill="x")
        ttk.Label(top, text="Track").pack(side="left")
        self.selector = ttk.Combobox(top, state="readonly", width=20)
        self.selector.pack(side="left", padx=5)
        self.selector.bind("<<ComboboxSelected>>", self._select_track)
        ttk.Label(top, text="Label").pack(side="left", padx=(4, 0))
        ttk.Entry(top, textvariable=self.label, width=22).pack(side="left", fill="x", expand=True)
        action(top, "New", "add", self._new, description="New track").pack(side="left", padx=3)
        action(top, "Save", "save", self._rename, description="Save track label").pack(side="left", padx=3)
        action_menu(top, "Workflow", "toolbox", (("Assignment review…", parent._open_vt_observables),
            ("VT analysis…", parent._open_vt_analysis), ("PCS preparation…", parent._open_vt_pcs), None,
            ("Delete track…", self._delete))).pack(side="left", padx=3)
        ttk.Label(body, textvariable=self.provenance, style="NMR.Muted.TLabel").pack(fill="x", pady=4)
        ttk.Label(body, text="Track = correspondence across temperatures, not an assignment. Stale links need review; suggestions require acceptance.",
                  wraplength=900, style="NMR.Muted.TLabel").pack(anchor="w")
        table = ttk.Frame(body)
        table.pack(fill="both", expand=True)
        self.points = ttk.Treeview(table, columns=("member", "temperature", "state", "ppm", "source", "reason"),
                                   show="headings", height=8, selectmode="browse")
        for column, title, width in (("member", "Member", 160), ("temperature", "K", 65),
                ("state", "State", 85), ("ppm", "Linked ppm", 85), ("source", "Decision", 120),
                ("reason", "Review / diagnostics", 230)):
            self.points.heading(column, text=title)
            self.points.column(column, width=width, stretch=column in {"member", "reason"})
        self.points.tag_configure("stale", foreground="#B45309")
        self.points.tag_configure("ambiguous", foreground="#7C3AED")
        self.points.tag_configure("confirmed", foreground="#15803D")
        self.points.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.points.yview)
        scroll.pack(side="right", fill="y")
        self.points.configure(yscrollcommand=scroll.set)
        self.points.bind("<<TreeviewSelect>>", self._select_member)
        manual = ttk.Frame(body)
        manual.pack(fill="x", pady=6)
        ttk.Label(manual, text="Existing peak").pack(side="left")
        self.peaks = ttk.Combobox(manual, width=21, state="readonly")
        self.peaks.pack(side="left", padx=4)
        self.peaks.bind("<<ComboboxSelected>>", self._select_peak)
        self.link_button = action(manual, "Link peak", "apply", self._link, primary=True,
                                  description="Explicitly link or reconfirm this existing peak in the selected member")
        self.link_button.pack(side="left", padx=3)
        action_menu(manual, "Point state", "settings", (("Mark missing", lambda: self._mark("missing")),
            ("Mark ambiguous", lambda: self._mark("ambiguous")), None,
            ("Unlink peak", lambda: self._mark("unlinked")), ("Remove point", self._remove_point))).pack(side="left", padx=3)
        assisted = ttk.Frame(body)
        assisted.pack(fill="x")
        ttk.Label(assisted, text="Candidate window / ppm").pack(side="left")
        ttk.Entry(assisted, textvariable=self.window_ppm, width=7).pack(side="left", padx=4)
        action(assisted, "Suggest", "search", self._suggest).pack(side="left", padx=3)
        self.accept_button = action(assisted, "Accept candidate", "check", self._accept,
                                    description="Accept only the selected candidate; suggestions never establish links automatically")
        self.accept_button.pack(side="left", padx=3)
        self.candidates = ttk.Treeview(body, columns=("ppm", "score", "reason"), show="headings", height=3, selectmode="browse")
        for column, width in (("ppm", 90), ("score", 80), ("reason", 650)):
            self.candidates.heading(column, text="δ / ppm" if column == "ppm" else column.capitalize())
            self.candidates.column(column, width=width, stretch=column == "reason")
        self.candidates.pack(fill="x", pady=5)
        self.candidates.bind("<<TreeviewSelect>>", self._candidate_detail)
        ttk.Label(body, textvariable=self.status, wraplength=900, style="NMR.Muted.TLabel").pack(fill="x")
        polish(self)
        self.refresh()

    def _run(self, action):
        try:
            action()
        except (KeyError, ValueError) as exc:
            self.status.set(str(exc))
            return False
        self._suggestions = ()
        self.refresh()
        self.on_changed()
        return True

    def refresh(self):
        self.manager.state.update_tracks()
        self._refreshing = True
        try:
            series, track = self.manager.active_series, self.manager.active_track
            self._track_ids = [t.id for t in series.tracks] if series else []
            self.selector.configure(values=[t.label or f"Track {t.id[:8]}" for t in series.tracks] if series else [])
            self.selector.set("")
            if track:
                self.selector.current(self._track_ids.index(track.id))
            self.label.set(track.label or "" if track else "")
            self.provenance.set(f"{track.provenance.origin} · {track.provenance.method or 'manual correspondence'} · Track identity does not assign atoms." if track else "Select/create a track. No assignment or PCS propagation.")
            self.points.delete(*self.points.get_children())
            issues = track.validate(series) if track else ()
            members = self.manager.members(series.id, self.order()) if series else ()
            for member in members:
                point = track.get_point(member.id) if track else None
                peak = next((p for p in member.dataset.peaks if point and p.id == point.peak_id and point.dataset_id == member.dataset_id), None)
                temp = member.resolve_temperature(series_override_k=series.temperature_override_k)
                state = point.state if point else "unreviewed"
                reason = ", ".join(dict.fromkeys(([point.reason] if point and point.reason else []) + [i.code for i in issues if i.member_id == member.id]))
                self.points.insert("", "end", iid=member.id, values=(member.label or dataset_display_name(member.dataset),
                    f"{temp.temperature_k:g}" if temp.temperature_k is not None else temp.status,
                    state, f"{peak.position_ppm:.6g}" if peak else "—",
                    point.provenance.method or point.provenance.origin if point else "—", reason), tags=(state,))
            if track:
                for point in track.points:
                    if not any(m.id == point.member_id for m in members):
                        self.points.insert("", "end", iid=point.member_id, values=(f"Removed {point.member_id[:8]}", "—", point.state, "—", point.provenance.origin, point.reason), tags=(point.state,))
            selected = self.manager.state.selected_member_id
            if selected and self.points.exists(selected):
                self.points.selection_set(selected)
                self.points.see(selected)
            member = next((m for m in members if m.id == selected), None)
            peaks = member.dataset.peaks if member else []
            self._peak_ids = [p.id for p in peaks]
            self.peaks.configure(values=[f"{p.position_ppm:.6g} ppm · {p.id[:8]}" for p in peaks])
            self.peaks.set("")
            if self.manager.state.selected_peak_id in self._peak_ids:
                self.peaks.current(self._peak_ids.index(self.manager.state.selected_peak_id))
            self.candidates.delete(*self.candidates.get_children())
            for index, candidate in enumerate(self._suggestions):
                peak = next((p for p in peaks if p.id == candidate.peak_id), None)
                self.candidates.insert("", "end", iid=str(index), values=(f"{peak.position_ppm:.6g}" if peak else "deleted", f"{candidate.score:.3f}", candidate.reasons[0]))
            self.link_button.configure(state="normal" if track and member and self.peaks.current() >= 0 else "disabled")
            self.accept_button.configure(state="normal" if self._suggestions else "disabled")
        finally:
            self._refreshing = False

    def _new(self):
        self._run(lambda: self.manager.create_track(self.label.get().strip() or None))

    def _rename(self):
        self._run(lambda: self.manager.rename_track(self.label.get().strip() or None))

    def _delete(self):
        if self.manager.active_track and messagebox.askyesno("Delete track", "Delete this correspondence track? Dataset peaks are retained.", parent=self):
            self._run(self.manager.delete_track)

    def _select_track(self, event=None):
        index = self.selector.current()
        if index >= 0:
            self._run(lambda: self.manager.select_track(self._track_ids[index]))

    def _select_member(self, event=None):
        if self._refreshing or not self.points.selection():
            return
        member_id = self.points.selection()[0]
        if member_id == self.manager.state.selected_member_id:
            return
        series = self.manager.active_series
        if series is None or not any(m.id == member_id for m in series.members):
            self.status.set("This member was removed. Use Remove point, or select a current member to review.")
            return
        point = self.manager.active_track.get_point(member_id) if self.manager.active_track else None
        member = next(m for m in series.members if m.id == member_id)
        peak_id = point.peak_id if point and point.dataset_id == member.dataset_id and sum(p.id == point.peak_id for p in member.dataset.peaks) == 1 else None
        self._run(lambda: self._activate(member_id, peak_id))

    def _activate(self, member_id, peak_id):
        member = self.manager.select_peak(member_id, peak_id)
        self.on_activate(member.dataset_id, peak_id)

    def _select_peak(self, event=None):
        index = self.peaks.current()
        if index >= 0:
            self._run(lambda: self._activate(self.manager.state.selected_member_id, self._peak_ids[index]))

    def _link(self):
        if self.points.selection() != (self.manager.state.selected_member_id,):
            self.status.set("Select a current member before linking.")
            return
        if self._run(self.manager.link_selected):
            self.status.set("Linked explicitly. This does not establish a chemical assignment.")

    def _mark(self, state):
        if self.points.selection() != (self.manager.state.selected_member_id,):
            self.status.set("Select a current member, or Remove point for a removed member.")
            return
        self._run(lambda: self.manager.mark(state))

    def _remove_point(self):
        if self.points.selection() and self.manager.active_track:
            self._run(lambda: self.manager.active_track.remove_point(self.points.selection()[0]))

    def _suggest(self):
        try:
            self._suggestions = self.manager.suggestions(window_ppm=float(self.window_ppm.get()))
            self.status.set(f"{len(self._suggestions)} suggestions; none linked. Multiple peaks may be plausible. Select one to review reasons, then accept explicitly." if self._suggestions else "No candidates: add a valid confirmed anchor or review/link a peak manually.")
        except (KeyError, ValueError) as exc:
            self._suggestions = ()
            self.status.set(str(exc))
        self.refresh()

    def _candidate_detail(self, event=None):
        if not self._refreshing and self.candidates.selection():
            self.status.set(" · ".join(self._suggestions[int(self.candidates.selection()[0])].reasons))

    def _accept(self):
        if not self.candidates.selection():
            self.status.set("Select a suggestion to review and accept.")
            return
        candidate = self._suggestions[int(self.candidates.selection()[0])]
        def accept():
            self.manager.accept(candidate)
            member = next(m for m in self.manager.active_series.members if m.id == candidate.member_id)
            self.on_activate(member.dataset_id, candidate.peak_id)
        if self._run(accept):
            self.status.set("Suggestion accepted explicitly; score and input provenance saved.")
