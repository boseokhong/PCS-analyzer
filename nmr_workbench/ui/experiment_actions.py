"""Compact Tk experiment manager actions; import/science stay in services."""
import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from pathlib import Path
from queue import Queue, Empty
from threading import Thread

from ..io.experiment_import import load_experiments, source_key
from .experiment_manager import ExperimentManager
from .view_model import dataset_display_name, dataset_summary


class ExperimentActions:
    def _setup_experiment_actions(self):
        self.experiments = ExperimentManager(self.model)
        self._experiment_status = tk.StringVar(self, value="0 experiments · 0 selected")
        ttk.Label(self._experiment_tree.master.master, textvariable=self._experiment_status,
                  wraplength=220).pack(side="bottom", before=self._dataset_role_box.master.master, fill="x")
        tree = self._experiment_tree
        tree.bind("<Control-a>", self._select_all_experiments)
        tree.bind("<Delete>", lambda e: self._remove_experiments())
        tree.bind("<Return>", self._activate_experiment)
        tree.bind("<Double-1>", self._activate_experiment)
        tree.bind("<Button-3>", self._experiment_menu)
        tree.bind("<ButtonPress-1>", self._experiment_drag_start, add="+")
        tree.bind("<B1-Motion>", self._experiment_drag_motion, add="+")
        tree.bind("<ButtonRelease-1>", self._experiment_drag_end, add="+")
        self._experiment_drag = None
        self._experiment_tree.tag_configure("active", foreground="#2459A6")
        self._import_poll_id = None
        self._import_busy = False
        self._drop_widgets = []
        self._enable_experiment_drop(self)

    def _enable_experiment_drop(self, widget):
        try:
            from tkinterdnd2 import DND_FILES
            # TkinterDnD's root normally owns tkdnd. Standalone Tk roots may load it.
            if not widget.tk.call("info", "commands", "tkdnd::drop_target"):
                from tkinterdnd2.TkinterDnD import _require
                _require(widget)
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self._drop_experiments)
            self._drop_widgets.append(widget)
        except (ImportError, tk.TclError, RuntimeError) as exc:
            self._experiment_status.set(f"Drop unavailable ({exc}); use Open or Import")

    def _drop_experiments(self, event):
        try:
            paths = self.tk.splitlist(event.data)  # Tcl list handles spaces/braces/Unicode.
            self._start_experiment_import(paths)
        except (ValueError, tk.TclError) as exc:
            self._experiment_status.set(f"Drop failed: {exc}")
        return "copy"

    def _start_experiment_import(self, paths):
        if not paths:
            return
        if self._import_busy:
            self._experiment_status.set("Import already running; drop/open again when complete")
            return
        self._import_busy = True
        sources = tuple(d.source_path for d in self.model.datasets if d.source_path)
        queue = Queue()
        def worker():
            try:
                queue.put(load_experiments(paths, existing_sources=sources))
            except Exception as exc:
                queue.put(exc)
        Thread(target=worker, daemon=True).start()
        self._experiment_status.set("Importing experiments…")
        def poll():
            self._import_poll_id = None
            try:
                result = queue.get_nowait()
            except Empty:
                self._import_poll_id = self.after(70, poll)
                return
            self._import_busy = False
            if isinstance(result, Exception):
                self._experiment_status.set(f"Import failed: {result}")
                return
            self._publish_experiment_import(result)
        self._import_poll_id = self.after(70, poll)

    def _publish_experiment_import(self, result):
        imported, skipped = [], list(result.skipped)
        for path, dataset in result.loaded:
            if any(d.id == dataset.id or (d.source_path is not None and dataset.source_path is not None
                   and source_key(d.source_path) == source_key(dataset.source_path)) for d in self.model.datasets):
                skipped.append(f"{path}: already loaded")
                continue
            self.model.datasets.append(dataset)
            imported.append(dataset.id)
        if imported:
            self.experiments.select(imported)
        self._refresh_series_datasets()
        if imported:
            self._select_dataset(imported[-1])
        self._experiment_status.set(f"{len(imported)} imported · {len(skipped)} skipped · {len(result.failures)} failed")
        if skipped or result.failures:
            messagebox.showwarning("NMR import results", self._experiment_status.get()+"\n\n"+
                "\n".join(skipped + [f"{path}: {error}" for path,error in result.failures]), parent=self)
        return tuple(imported)

    def _selection_experiment_ids(self):
        tree = self._experiment_tree
        ids = tuple(dict.fromkeys(self._tree_dataset_ids[i] for i in tree.selection() if i in self._tree_dataset_ids))
        self.experiments.select(ids)
        self._batch_selected_button.configure(state="normal" if ids else "disabled")
        active = self.model.selected_dataset_id
        self._experiment_status.set(f"{len(self.model.datasets)} experiments · {len(ids)} selected" +
            (f" · Active: {dataset_display_name(self.model.current_dataset)}" if active and self.model.current_dataset else ""))
        return self.experiments.selected_ids

    def _select_all_experiments(self, event=None):
        self._experiment_tree.selection_set(self._experiment_tree.get_children())
        self._selection_experiment_ids()
        return "break"

    def _activate_experiment(self, event=None):
        tree = self._experiment_tree
        item = tree.identify_row(event.y) if event is not None and hasattr(event,"y") and event.type != tk.EventType.KeyPress else tree.focus()
        dataset_id = self._tree_dataset_ids.get(item)
        if dataset_id and dataset_id != self.model.selected_dataset_id:
            self._select_dataset(dataset_id)
        return "break"

    def _remove_experiments(self, ids=None):
        ids = tuple(ids or self._selection_experiment_ids())
        if not ids:
            return "break"
        self._cancel_phase_timer()
        active_removed = self.model.selected_dataset_id in ids
        self.experiments.remove(ids)
        for dataset_id in ids:
            self._assignment_drafts_by_dataset.pop(dataset_id, None)
        self._dataset_pair_ids = {key:value for key,value in self._dataset_pair_ids.items()
                                  if key not in ids and value not in ids}
        self._refresh_series_datasets()
        if active_removed:
            if self.model.current_dataset is not None:
                self._select_dataset(self.model.current_dataset.id)
            else:
                self.processing_session = self.processing_session.__class__()
                self._dataset_text.set("No experiment selected")
                self._view_selector.configure(values=(),state="disabled")
                self._view_choice.set("")
                self._refresh()
                self._refresh_dataset_role_controls()
                self._refresh_peak_list()
                self._history_text.configure(state="normal")
                self._history_text.delete("1.0", "end")
                self._history_text.configure(state="disabled")
                self._metadata_tree.delete(*self._metadata_tree.get_children())
                self._state_text.set("No spectrum")
        self._refresh_stacked_series()
        return "break"

    def _rename_experiment(self, dataset_id):
        dataset = next(d for d in self.model.datasets if d.id == dataset_id)
        label = simpledialog.askstring("Experiment label", "Display label:", initialvalue=dataset_display_name(dataset), parent=self)
        if label is not None:
            try:
                self.experiments.rename(dataset_id,label)
                self._refresh_series_datasets()
            except ValueError as exc:
                messagebox.showerror("Experiment label",str(exc),parent=self)

    def _experiment_properties(self, dataset_id):
        from dataclasses import asdict
        dataset = next(d for d in self.model.datasets if d.id == dataset_id)
        window = tk.Toplevel(self)
        window.title("Experiment metadata")
        text = tk.Text(window, width=80, height=22, wrap="word")
        text.pack(fill="both",expand=True)
        memberships = [s.name for s in self.model.series if any(m.dataset_id == dataset_id for m in s.members)]
        text.insert("1.0", f"{dataset_display_name(dataset)}\nID: {dataset.id}\nSource: {dataset.source_path}\nVendor: {dataset.vendor}\n{dataset_summary(dataset)}\nSeries: {', '.join(memberships) or 'none'}\nSpectral revision: {dataset.spectral_revision}\n\n"+
                    "\n".join(f"{key}: {value}" for key,value in asdict(dataset.conditions).items()))
        text.configure(state="disabled")

    def _reveal_experiment(self, dataset_id):
        dataset = next(d for d in self.model.datasets if d.id == dataset_id)
        source = Path(dataset.source_path) if dataset.source_path else None
        if source is None or not source.exists():
            messagebox.showwarning("Experiment source", "Source location is unavailable; committed session data is retained.",parent=self)
            return
        path = str(source if source.is_dir() else source.parent)
        try:
            if sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])
        except OSError as exc:
            messagebox.showwarning("Experiment source", str(exc), parent=self)

    def _experiment_menu(self, event):
        tree = self._experiment_tree
        item = tree.identify_row(event.y)
        dataset_id = self._tree_dataset_ids.get(item)
        if not dataset_id:
            return
        selected = self._selection_experiment_ids()
        if dataset_id not in selected:
            tree.selection_set(item)
        tree.focus(item)
        menu = tk.Menu(tree, tearoff=False)
        menu.add_command(label="Remove", command=lambda:self._remove_experiments((dataset_id,)))
        menu.add_command(label="Remove selected  Del", command=self._remove_experiments)
        menu.add_command(label="Rename/Edit label…",command=lambda:self._rename_experiment(dataset_id))
        menu.add_command(label="Properties/Metadata…",command=lambda:self._experiment_properties(dataset_id))
        menu.add_command(label="Reveal source location", command=lambda:self._reveal_experiment(dataset_id))
        menu.add_command(label="Batch selected…", command=self._batch_selected_experiments)
        ordering = tk.Menu(menu,tearoff=False)
        for order in ("Import order","Name","Temperature","Acquisition time"):
            ordering.add_command(label=order,command=lambda order=order:self._sort_experiments(order))
        menu.add_cascade(label="Display order",menu=ordering)
        try:
            menu.tk_popup(event.x_root,event.y_root)
        finally:
            menu.grab_release()

    def _sort_experiments(self, order):
        self.experiments.sort(order)
        self._rebuild_experiment_tree()

    def _experiment_drag_start(self, event):
        item = self._experiment_tree.identify_row(event.y)
        self._experiment_drag = (item,event.x,event.y,False)

    def _experiment_drag_motion(self, event):
        if self._experiment_drag:
            item,x,y,_ = self._experiment_drag
            self._experiment_drag = (item,x,y,abs(event.x-x)+abs(event.y-y)>8)

    def _experiment_drag_end(self, event):
        drag,self._experiment_drag = self._experiment_drag,None
        if not drag or not drag[3]:
            return
        tree = self._experiment_tree
        moved = self._selection_experiment_ids()
        if not moved:
            moved = (self._tree_dataset_ids.get(drag[0]),)
        target = self._tree_dataset_ids.get(tree.identify_row(event.y))
        self.experiments.reorder(moved,target)
        self._rebuild_experiment_tree()

    def _batch_selected_experiments(self, operation=None, parameters=None):
        self._selection_experiment_ids()
        if not self.experiments.selected_ids:
            return
        self._open_batch_processing(context=self.experiments.batch_context(), operation=operation, parameters=parameters)

    def _dispatch_experiment_batch(self, operation, parameters=None):
        if len(self._selection_experiment_ids()) < 2:
            return False
        try:
            if callable(parameters):
                parameters = parameters()
            self._batch_selected_experiments(operation,parameters)
        except (ValueError, TypeError, KeyError) as exc:
            self._show_processing_error(exc)
        return True

    def _cleanup_experiment_actions(self):
        if self._import_poll_id is not None:
            self.after_cancel(self._import_poll_id)
        for widget in self._drop_widgets:
            try:
                widget.drop_target_unregister()
            except tk.TclError:
                pass
