"""VT preparation and explicit canonical handoff review; no fitting."""
import json
import tkinter as tk
from tkinter import messagebox, ttk

from ..analysis.pcs_derivation import builtin_pcs_providers
from ..analysis.vt_pcs import vt_observations, reference_observations, review_pair, prepare_pair, pcs_ready_row, all_observations, preparation_diagnostics
from .tracking_manager import TrackingManager
from ..integration.vt_pcs_handoff import destination_context, preview_vt_transfer, apply_vt_transfer
from .actions import action, action_menu, polish


class VTPCSWindow(tk.Toplevel):
    def __init__(self, parent, state, *, targets, on_changed, destination=None, rebind=None):
        super().__init__(parent)
        self.title("VT PCS Pairing / Preparation")
        self.transient(parent)
        self.geometry("1060x590")
        self.manager = TrackingManager(state)
        self.target_choices, self.on_changed = targets, on_changed
        self.destination = destination or (lambda: None)
        self.rebind_destination = rebind
        self._transfer_preview = None
        self.replace_existing = tk.BooleanVar(self, value=False)
        self.destination_label = tk.StringVar(self)
        self._para, self._references, self._targets = (), (), ()
        self._context = None
        self.prepared_row = None  # transient canonical preview, never transferred
        self.provider = tk.StringVar(self, value="shift_difference_only")
        self.correction, self.manual, self.note = (tk.StringVar(self) for _ in range(3))
        self.status = tk.StringVar(self)
        self.pages = ttk.Notebook(self)
        self.pages.pack(fill="both", expand=True)
        body = ttk.Frame(self.pages, padding=10)
        self.pages.add(body, text="Prepare / pair")
        transfer = ttk.Frame(self.pages, padding=10)
        self.pages.add(transfer, text="Reviewed transfer")
        controls = ttk.Frame(body)
        controls.pack(fill="x")
        ttk.Label(controls, text="VT observation").pack(side="left")
        self.member = ttk.Combobox(controls, state="readonly", width=42)
        self.member.pack(side="left", padx=5)
        self.member.bind("<<ComboboxSelected>>", lambda e: self._review())
        action(controls, "Refresh", "update", self.refresh, description="Refresh observations").pack(side="left")
        self.references = ttk.Treeview(body, columns=("dataset", "observable", "temperature", "shift", "status"), show="headings", height=7, selectmode="browse")
        for name, width in (("dataset", 160), ("observable", 180), ("temperature", 90), ("shift", 100), ("status", 390)):
            titles = {"dataset": "Reference dataset", "observable": "Observable", "temperature": "T / K", "shift": "δ / ppm", "status": "Pairing diagnostics"}
            self.references.heading(name, text=titles[name])
            self.references.column(name, width=width)
        self.references.pack(fill="both", expand=True, pady=8)
        self.references.bind("<<TreeviewSelect>>", lambda e: self._display_selected())
        row = ttk.Frame(body)
        row.pack(fill="x")
        ttk.Label(row, text="Reviewed structural target").pack(side="left")
        self.target = ttk.Combobox(row, state="readonly", width=75)
        self.target.pack(side="left", padx=4)
        self.target.bind("<<ComboboxSelected>>", lambda e: self._review())
        method = ttk.Frame(body)
        method.pack(fill="x", pady=7)
        ttk.Combobox(method, textvariable=self.provider, state="readonly", width=24,
            values=("shift_difference_only", *(p.provider_id for p in builtin_pcs_providers()))).pack(side="left")
        for label, var in (("Manual contact / ppm", self.correction), ("Imported PCS / ppm", self.manual)):
            ttk.Label(method, text=label).pack(side="left", padx=4)
            ttk.Entry(method, textvariable=var, width=9).pack(side="left")
        ttk.Label(body, text="Scientific basis / review note (required for PCS)").pack(anchor="w")
        ttk.Entry(body, textvariable=self.note).pack(fill="x")
        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=7)
        action(actions, "Prepare reviewed pair", "apply", self._prepare, primary=True).pack(side="left")
        action_menu(actions, "Preparation", "settings", (("Inspect saved provenance", self._inspect), None,
            ("Remove saved preparation", self._remove))).pack(side="right")
        self.details = tk.Text(body, height=6, wrap="word", state="disabled")
        self.details.pack(fill="x")
        ttk.Label(body, textvariable=self.status, wraplength=1000).pack(fill="x", pady=5)
        self._build_transfer_tab(transfer)
        polish(self)
        self.refresh()

    def _target(self):
        index = self.target.current()
        return self._targets[index - 1] if index > 0 else None

    def refresh(self):
        series, track = self.manager.active_series, self.manager.active_track
        context = (series.id, track.id) if series and track else None
        if context != self._context:
            self.note.set("")
            self.correction.set("")
            self.manual.set("")
            self._context = context
        self._para = vt_observations(series, track.id) if context else ()
        self._references = reference_observations(self.manager.state.datasets, self.manager.state.series)
        self._targets = tuple(self.target_choices())
        previous = self.target.get()
        self.target.configure(values=("Unresolved", *(t.display for t in self._targets)))
        self.target.set(previous if previous in self.target["values"] else "Unresolved")
        previous_member = self.member.current()
        self.member.configure(values=[f"{o.member_id[:8]} · {o.temperature_k} K · δ {o.observed_shift_ppm}" for o in self._para])
        if self._para:
            selected = next((i for i, o in enumerate(self._para) if o.member_id == self.manager.state.selected_member_id), max(0, min(previous_member, len(self._para)-1)))
            self.member.current(selected)
        else:
            self.member.set("")
        self.prepared_row = None
        self._review()
        self._refresh_transfer_rows()

    def _build_transfer_tab(self, body):
        top = ttk.Frame(body)
        top.pack(fill="x")
        ttk.Label(top, textvariable=self.destination_label).pack(side="left")
        action(top, "Choose document", "settings", self._rebind_transfer, description="Choose active PCS document").pack(side="right")
        ttk.Label(body, text="Active series preparations — Ctrl/Shift selects a subset. Nothing is selected or transferred automatically.").pack(anchor="w", pady=6)
        self.transfer_rows = ttk.Treeview(body, columns=("temperature", "track", "member", "pcs", "target", "state"),
            show="headings", selectmode="extended", height=8)
        for name, width in (("temperature", 85), ("track", 125), ("member", 95), ("pcs", 85), ("target", 180), ("state", 390)):
            titles = {"temperature": "T / K", "track": "Track", "member": "Member", "pcs": "PCS / ppm", "target": "Structural target", "state": "Transfer readiness"}
            self.transfer_rows.heading(name, text=titles[name])
            self.transfer_rows.column(name, width=width)
        self.transfer_rows.pack(fill="both", expand=True)
        self.transfer_rows.bind("<<TreeviewSelect>>", self._invalidate_transfer_preview)
        controls = ttk.Frame(body)
        controls.pack(fill="x", pady=7)
        action(controls, "Select temperature", "select_check", self._select_temperature, description="Select current temperature").pack(side="left")
        action(controls, "Preview transfer", "preview", self._preview_transfer).pack(side="left", padx=5)
        self.apply_transfer_button = action(controls, "Transfer reviewed", "apply", self._apply_transfer, primary=True, state="disabled")
        self.apply_transfer_button.pack(side="right")
        ttk.Checkbutton(body, text="Allow the explicitly listed destination replacements (unrelated data remains intact)",
            variable=self.replace_existing).pack(anchor="w")
        self.transfer_details = tk.Text(body, height=9, wrap="word", state="disabled")
        self.transfer_details.pack(fill="x", pady=7)
        ttk.Label(body, textvariable=self.status, wraplength=1000).pack(fill="x")

    def _invalidate_transfer_preview(self, event=None):
        if event is not None and self._transfer_preview is not None and self.transfer_rows.selection() == self._transfer_preview.preparation_ids:
            return
        self._transfer_preview = None
        self.apply_transfer_button.configure(state="disabled")

    def _refresh_transfer_rows(self):
        self._invalidate_transfer_preview()
        selected = self.transfer_rows.selection()
        self.transfer_rows.delete(*self.transfer_rows.get_children())
        try:
            context = destination_context(self.destination())
            self.destination_label.set(f"Destination: {context[1]} · {context[0] or 'legacy context'}")
        except (ValueError, RuntimeError) as exc:
            self.destination_label.set(f"Destination unavailable: {exc}")
        series = self.manager.active_series
        if not series:
            return
        observations = all_observations(self.manager.state.datasets, self.manager.state.series)
        for record in series.pcs_preparations:
            evidence = json.loads(record.evidence_json)
            try:
                pcs_ready_row(record, observations, target_choices=self._targets)
                state = "ready for explicit review"
            except (ValueError, KeyError) as exc:
                state = str(exc)
            value = evidence["pcs_value"]
            self.transfer_rows.insert("", "end", iid=record.id, values=(evidence["paramagnetic"].get("temperature_k"),
                record.track_id[:8], record.member_id[:8], value.get("pcs_ppm") if value else "undefined",
                evidence["target"].get("display") if evidence["target"] else "unresolved", state))
        self.transfer_rows.selection_set([i for i in selected if self.transfer_rows.exists(i)])

    def _set_transfer_details(self, text):
        self.transfer_details.configure(state="normal")
        self.transfer_details.delete("1.0", "end")
        self.transfer_details.insert("1.0", text)
        self.transfer_details.configure(state="disabled")

    def _select_temperature(self):
        series = self.manager.active_series
        member = next((m for m in series.members if m.id == self.manager.state.selected_member_id), None) if series else None
        temperature = member.resolve_temperature(series_override_k=series.temperature_override_k).temperature_k if member else None
        if temperature is None:
            self.status.set("Select a member with a valid effective temperature first.")
            return
        ids = [p.id for p in series.pcs_preparations if json.loads(p.evidence_json)["paramagnetic"].get("temperature_k") == temperature]
        self.transfer_rows.selection_set(ids)
        self._invalidate_transfer_preview()

    def _rebind_transfer(self):
        if self.rebind_destination:
            try:
                self.rebind_destination()
                self.refresh()
                self.status.set("Destination binding reviewed. Preview selected observations again.")
            except (ValueError, RuntimeError) as exc:
                self.status.set(str(exc))

    def _preview_transfer(self):
        self._invalidate_transfer_preview()
        try:
            preview = preview_vt_transfer(self.manager.active_series, self.manager.state.datasets, self.manager.state.series,
                self.transfer_rows.selection(), self.destination(), tuple(self.target_choices()))
            self._transfer_preview = preview
            rows = [f"{r.value_id}: {r.assignment}, {r.temperature_K:g} K, {r.pcs_ppm:g} ppm → {r.target.display}; "
                f"groups={r.target.site_member_ref_ids}; observables={r.source_observable_ids}" for r in preview.rows]
            conflicts = [f"{c.code}: {c.message}" for c in preview.conflicts]
            self._set_transfer_details(f"Destination: {preview.destination_name}\n"
                f"{preview.summary.added} added, {preview.summary.replaced} replaced; {preview.summary.single_atom} single atom / {preview.summary.structured} grouped.\n"
                + "\n".join(rows + conflicts))
            self.apply_transfer_button.configure(state="normal" if all(c.replaceable for c in preview.conflicts) else "disabled")
            self.status.set("Preview ready; approve any listed replacements, then Apply explicitly. No fitting will run.")
        except (ValueError, RuntimeError, KeyError) as exc:
            self.status.set(str(exc))

    def _apply_transfer(self):
        preview = self._transfer_preview
        if preview is None:
            self.status.set("Preview selected observations and destination first.")
            return
        replace = self.replace_existing.get()
        if preview.conflicts and not replace:
            self.status.set("Review destination conflicts and explicitly approve replacements first.")
            return
        if not messagebox.askyesno("Transfer reviewed VT PCS", f"Transfer {len(preview.rows)} reviewed observations to {preview.destination_name}?\n"
            + ("The listed destination values will be replaced.\n" if preview.conflicts else "") + "No fitting will run.", parent=self):
            return
        try:
            if self._transfer_preview is not preview or self.transfer_rows.selection() != preview.preparation_ids:
                raise ValueError("Transfer selection changed; preview again")
            result, receipt = apply_vt_transfer(preview, self.manager.active_series, self.manager.state.datasets,
                self.manager.state.series, self.destination(), tuple(self.target_choices()), confirmed=True, replace_existing=replace)
            self._invalidate_transfer_preview()
            refresh_error = None
            try:
                self.on_changed()
            except Exception as exc:
                refresh_error = str(exc)
                self._invalidate_transfer_preview()
            self._set_transfer_details(f"Transferred {result.applied} observations ({result.structured} grouped). Receipt {receipt.id}.\n"
                "PCS document owns transferred values; later NMR source changes do not modify them.")
            self.status.set(f"Transferred {result.applied} to {receipt.destination_name}; {result.added} added / {result.replaced} replaced."
                + (f" Display refresh failed: {refresh_error}" if refresh_error else ""))
        except Exception as exc:
            self._invalidate_transfer_preview()
            self.status.set(f"Transfer failed: {exc}")

    def _review(self):
        self.references.delete(*self.references.get_children())
        index = self.member.current()
        if index < 0:
            self.status.set("Select a VT track in tracking/analysis. No reference is chosen automatically.")
            return
        review = review_pair(self._para[index], self._references, target=self._target())
        for i, candidate in enumerate(review.candidates):
            r = candidate.reference
            self.references.insert("", "end", iid=str(i), values=(r.dataset_id, r.observable_id, r.temperature_k,
                r.observed_shift_ppm, ", ".join(candidate.diagnostics) or "compatible — select explicitly"))
        self.status.set("; ".join(review.diagnostics))
        self._set_details("Observed δ remains distinct from reference δ, Δshift, contact correction and PCS. Select a reference, structural target and derivation basis explicitly.")

    def _set_details(self, text):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def _selected_review(self):
        if self.member.current() < 0 or not self.references.selection():
            raise ValueError("Select a VT observation and reference explicitly")
        reference = self._references[int(self.references.selection()[0])]
        return review_pair(self._para[self.member.current()], self._references, reference_id=reference.id, target=self._target())

    def _display_selected(self):
        try:
            review = self._selected_review()
        except ValueError:
            return
        self.status.set("; ".join(review.diagnostics) or "Explicit compatible pair; PCS basis and target still require review.")
        self._set_details(f"Observed δ: {review.paramagnetic.observed_shift_ppm} ppm\nReference δ: {review.selected_reference.observed_shift_ppm if review.selected_reference else 'unresolved'} ppm\nShift difference: {review.shift_difference_ppm} ppm\nSource observation: {review.paramagnetic.id}")

    def _prepare(self):
        try:
            review = self._selected_review()
            # Re-read scientific sources; a refresh never authorizes a new selection.
            observations = all_observations(self.manager.state.datasets, self.manager.state.series)
            provider = None if self.provider.get() == "shift_difference_only" else self.provider.get()
            existing = next((p for p in self.manager.active_series.pcs_preparations if p.track_id == review.paramagnetic.track_id and p.member_id == review.paramagnetic.member_id), None)
            record = prepare_pair(review, target=self._target(), provider_id=provider,
                correction_ppm=float(self.correction.get()) if self.correction.get().strip() else None,
                manual_pcs_ppm=float(self.manual.get()) if self.manual.get().strip() else None, note=self.note.get(),
                preparation_id=existing.id if existing else None)
            problems = preparation_diagnostics(record, observations)
            if any(p not in {"pcs_not_defined", "unresolved_target"} for p in problems):
                raise ValueError("; ".join(problems))
            prepared = pcs_ready_row(record, observations, target_choices=tuple(self.target_choices())) if provider else None
            series = self.manager.active_series
            series.set_pcs_preparation(record)
            self.on_changed()
            self.prepared_row = prepared
            evidence = json.loads(record.evidence_json)
            self._set_details(json.dumps(evidence, indent=2))
            self.status.set("PCS-ready canonical observation prepared; no transfer or fitting performed." if provider else "Reviewed shift difference saved; PCS is not defined.")
        except (KeyError, ValueError) as exc:
            self.status.set(str(exc))

    def _inspect(self):
        if not self.manager.active_series or self.member.current() < 0:
            return
        obs = self._para[self.member.current()]
        saved = next((p for p in self.manager.active_series.pcs_preparations if p.track_id == obs.track_id and p.member_id == obs.member_id), None)
        if not saved:
            self.status.set("No saved preparation for this member.")
            return
        self._set_details(json.dumps(json.loads(saved.evidence_json), indent=2))
        self.status.set("; ".join(preparation_diagnostics(saved, all_observations(self.manager.state.datasets, self.manager.state.series))) or "Saved sources unchanged; inspect target and derivation basis before use.")

    def _remove(self):
        series, track = self.manager.active_series, self.manager.active_track
        if series and track and self.member.current() >= 0:
            series.remove_pcs_preparation(track.id, self._para[self.member.current()].member_id)
            self.on_changed()
            self.status.set("Saved preparation removed; source observables retained.")
