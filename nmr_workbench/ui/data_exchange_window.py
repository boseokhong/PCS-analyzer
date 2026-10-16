"""Compact reviewed assignment exchange and historical analysis inspection."""
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from nmr_workbench.io.assignment_exchange import export_assignments, review_assignments, apply_assignments
from nmr_workbench.io.analysis_exchange import load_analysis_evidence, review_analysis_evidence, retain_analysis_evidence
from nmr_workbench.io.exchange_files import atomic_text_export
from .workflow_help import show_workflow_help
from .actions import action, action_menu, polish


class DataExchangeWindow(tk.Toplevel):
    def __init__(self, parent, state, *, on_assignments, on_changed):
        super().__init__(parent)
        self.title("NMR assignments & saved analysis")
        self.transient(parent)
        self.geometry("1000x650")
        self.minsize(980, 550)
        self.state_model, self.on_assignments, self.on_changed = state, on_assignments, on_changed
        self.plan = self.evidence = None
        self.status = tk.StringVar(self, value="Save an NMR session for complete recovery; CSV/JSON exchange has a narrower purpose.")
        self.replace_conflicts = tk.BooleanVar(self, value=False)
        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True, padx=10, pady=10)
        assignments, results = ttk.Frame(tabs, padding=8), ttk.Frame(tabs, padding=8)
        tabs.add(assignments, text="Assignments — CSV")
        tabs.add(results, text="Saved analysis — JSON / session")
        ttk.Label(assignments, text="Import → review rows → Apply. Keep dataset/peak IDs; reopen the original NMR session first.",
                  wraplength=920, style="NMR.Muted.TLabel").pack(anchor="w", pady=(0, 8))
        buttons = ttk.Frame(assignments)
        buttons.pack(fill="x")
        action(buttons, "Import CSV", "import", lambda: self._run(self._review_assignments),
               description="Import an assignment table for read-only review").pack(side="left")
        action(buttons, "Export CSV", "export", lambda: self._run(self._export_assignments),
               description="Export assignments from all loaded datasets").pack(side="left", padx=5)
        self.apply_button = action(buttons, "Apply reviewed rows", "apply", lambda: self._run(self._apply_assignments), primary=True, state="disabled")
        self.apply_button.pack(side="right")
        ttk.Checkbutton(assignments, text="Explicitly replace conflicting names / uncertainties on Apply", variable=self.replace_conflicts).pack(anchor="w", pady=6)
        assignment_table = ttk.Frame(assignments)
        assignment_table.pack(fill="both", expand=True)
        self.assignment_rows = ttk.Treeview(assignment_table, columns=("row", "dataset", "peak", "assignment", "status", "reason"), show="headings")
        for key, label, width in (("row", "CSV row", 55), ("dataset", "Dataset ID", 100), ("peak", "Peak ID", 100),
                ("assignment", "Assignment", 150), ("status", "Review state", 90), ("reason", "Action / reason", 370)):
            self.assignment_rows.heading(key, text=label)
            self.assignment_rows.column(key, width=width)
        assignment_scroll = ttk.Scrollbar(assignment_table, command=self.assignment_rows.yview)
        self.assignment_rows.configure(yscrollcommand=assignment_scroll.set)
        assignment_scroll.pack(side="right", fill="y")
        self.assignment_rows.pack(fill="both", expand=True)
        ttk.Label(results, text="Historical results: inspect → retain → Save session. Live fits and source data stay unchanged.",
                  wraplength=920, style="NMR.Muted.TLabel").pack(anchor="w", pady=(0, 8))
        result_buttons = ttk.Frame(results)
        result_buttons.pack(fill="x")
        self.saved = ttk.Combobox(result_buttons, state="readonly", width=26)
        self.saved.pack(side="left", fill="x", expand=True)
        self.saved.bind("<<ComboboxSelected>>", self._select_saved)
        action(result_buttons, "Refresh", "update", self.refresh_saved, description="Refresh saved results").pack(side="left", padx=5)
        action(result_buttons, "Import JSON", "import", lambda: self._run(self._import_results)).pack(side="left")
        action(result_buttons, "Export JSON", "export", lambda: self._run(self._export_results), description="Export selected result").pack(side="left", padx=5)
        action(result_buttons, "Keep in session", "save", lambda: self._run(self._retain), primary=True).pack(side="left")
        text_frame = ttk.Frame(results)
        text_frame.pack(fill="both", expand=True, pady=8)
        self.result_text = tk.Text(text_frame, wrap="word")
        scroll = ttk.Scrollbar(text_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.result_text.pack(fill="both", expand=True)
        self.result_text.configure(state="disabled")
        footer = ttk.Frame(self, padding=10)
        footer.pack(fill="x")
        action_menu(footer, "Guide", "toolbox", (("Workflow & file guide…", lambda: show_workflow_help(self)),)).pack(side="right")
        ttk.Label(footer, textvariable=self.status, wraplength=770).pack(side="left", fill="x", expand=True)
        self.refresh_saved()
        polish(self)

    def _run(self, action):
        try:
            action()
        except (ValueError, KeyError, OSError) as exc:
            self.status.set(str(exc))

    def _export_assignments(self):
        path = filedialog.asksaveasfilename(parent=self, title="Export assignments from all loaded datasets", defaultextension=".csv", filetypes=(("Assignment table", "*.csv"),))
        if path:
            count = export_assignments(path, self.state_model.datasets)
            self.status.set(f"Exported {count} assignments. Save .nmrsession too to preserve source IDs and spectra.")

    def _review_assignments(self):
        # A failed/cancelled new review must never leave an old Apply enabled.
        self.plan = None
        self.apply_button.configure(state="disabled")
        self.assignment_rows.delete(*self.assignment_rows.get_children())
        self.replace_conflicts.set(False)
        path = filedialog.askopenfilename(parent=self, title="Review exported assignment CSV", filetypes=(("Assignment table", "*.csv"),))
        if not path:
            return
        self.plan = review_assignments(path, self.state_model.datasets)
        for row in self.plan.rows:
            self.assignment_rows.insert("", "end", values=(row.row_number, row.dataset_id[:8], row.peak_id[:8], row.assignment, row.status, row.message))
        counts = {s: sum(r.status == s for r in self.plan.rows) for s in ("ready", "conflict", "blocked", "unchanged")}
        self.status.set(" · ".join(f"{n} {s}" for s, n in counts.items()) + ". Nothing changed; review before Apply.")
        self.apply_button.configure(state="normal" if counts["ready"] + counts["conflict"] else "disabled")

    def _apply_assignments(self):
        if self.plan is None:
            raise ValueError("Review an assignment file first")
        selected = [r for r in self.plan.rows if r.status == "ready" or (r.status == "conflict" and self.replace_conflicts.get())]
        if not selected:
            raise ValueError("No eligible rows. Conflicts require explicit replacement approval.")
        if not messagebox.askyesno("Apply reviewed assignments", f"Apply {len(selected)} reviewed rows?\nBlocked/unchanged rows remain untouched. No track assignments are propagated.", parent=self):
            return
        count = apply_assignments(self.plan, self.state_model.datasets, replace_conflicts=self.replace_conflicts.get())
        self.plan = None
        self.apply_button.configure(state="disabled")
        self.on_assignments(tuple((r.dataset_id, r.peak_id) for r in selected))
        self.on_changed()
        self.status.set(f"Applied {count} assignments; observable IDs preserved. Save session to retain changes.")

    def refresh_saved(self):
        self._saved = [(s, e) for s in self.state_model.series for e in s.analysis_evidence]
        self.saved.configure(values=[f"{s.name} · result {e.id[:8]}" for s, e in self._saved])
        self.saved.set("")

    def _select_saved(self, event=None):
        if self.saved.current() >= 0:
            self.evidence = self._saved[self.saved.current()][1]
            self._run(self._show_result)

    def _import_results(self):
        path = filedialog.askopenfilename(parent=self, title="Inspect historical VT analysis results", filetypes=(("Analysis results JSON", "*.json"),))
        if path:
            self.evidence = load_analysis_evidence(path)
            self.saved.set("")
            self._show_result()

    def _show_result(self):
        warnings = review_analysis_evidence(self.evidence, self.state_model.series)
        data = json.loads(self.evidence.evidence_json)
        lines = [f"Historical analysis {self.evidence.id}", f"Created: {data.get('created_at', 'unknown')}",
            f"Quantity: observed chemical shift / ppm · coordinate: {data.get('coordinate')}",
            "Source review: " + (" | ".join(warnings) if warnings else "Matches loaded scientific sources"), ""]
        for row in data["tracks"]:
            snapshot = row["snapshot"]
            lines.append(f"Track: {snapshot.get('track_label') or snapshot['track_id']}")
            for result in row["results"]:
                config = result.get("config", {})
                lines.append(f"  Model: {config.get('model_id', 'linear coordinate')} · status: {result['status']} · n: {result['point_count']} · R²: {result.get('r_squared')}")
                for key in ("slope", "intercept", "parameters", "diagnostics"):
                    if key in result:
                        lines.append(f"  {key}: {json.dumps(result[key], ensure_ascii=False)}")
            lines.append("")
        lines.extend(("Complete saved evidence (identities, masks, fitted values, residuals, provenance):", json.dumps(data, indent=2, ensure_ascii=False)))
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "\n".join(lines))
        self.result_text.configure(state="disabled")
        self.status.set("Historical result inspected; live fits and source data unchanged.")

    def _retain(self):
        if self.evidence is None:
            raise ValueError("Select or import a result first")
        added = retain_analysis_evidence(self.evidence, self.state_model.series)
        self.on_changed()
        self.refresh_saved()
        self.status.set("Historical result retained; Save session to keep it." if added else "This result is already retained in the session.")

    def _export_results(self):
        if self.evidence is None:
            raise ValueError("Select or import a result first")
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".json", filetypes=(("Analysis results JSON", "*.json"),))
        if path:
            with atomic_text_export(path) as handle:
                json.dump(self.evidence.to_dict(), handle, indent=2, allow_nan=False)
            self.status.set("Exported selected historical result JSON.")
