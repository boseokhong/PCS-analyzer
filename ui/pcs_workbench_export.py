"""Export a fixed PCS calculation snapshot without blocking the Tk event loop."""

from pathlib import Path
import queue
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from logic.pcs_volume_export import (
    VOLUME_FORMATS, SURFACE_FORMATS, export_volume, export_surface, prepare_volume,
)
from ui.widget_style import ui_label_style, ui_native_colors, ui_font_name
from ui.dialog_layout import DialogForm, dialog_actions, fit_dialog


FIELDS = {"PCS field": "pcs", "Density used by solver": "density_used",
          "Original signed density": "density_signed"}
REGIONS = {"Full calculation grid": "full", "Input density region": "input",
           "Custom bounds (Angstrom)": "custom"}


class VolumeExportDialog(tk.Toplevel):
    def __init__(self, parent, result, initial_format=".cube"):
        super().__init__(parent)
        self.title("Export 3D data")
        self.transient(parent)
        self.result = result
        self._queue = queue.SimpleQueue()
        self._cancel = threading.Event()
        self._running = False
        self._poll_id = None
        self._field_labels = {}
        self._advanced_open = False
        self.field = tk.StringVar(value="PCS field")
        self.region = tk.StringVar(value="Full calculation grid")
        self.kind = tk.StringVar(value="Volume")
        self.format = tk.StringVar(value=initial_format)
        self.precision = tk.StringVar(value="float64")
        self.compressed = tk.BooleanVar(value=True)
        self.smooth = tk.BooleanVar(value=False)
        self.sigma = tk.StringVar(value="1.0")
        self.level = tk.StringVar(value="1.0")
        ext = result["ext"]
        self.bounds = tk.StringVar(value=" ".join(f"{v:.8g}" for v in ext))
        self.status = tk.StringVar(value="Ready. Calculation arrays are exported unchanged by default.")
        form = DialogForm(self)
        actions = dialog_actions(self)
        actions.pack(side="bottom", fill="x")
        form.pack(fill="both", expand=True)
        body = form.content
        body.columnconfigure(1, weight=1)
        files = result.get("input_files", {})
        summary = (f"Calculation snapshot: {result.get('temperature', 0):g} K\n"
                   f"ORCA: {files.get('orca', 'Not recorded')}\nDensity: {files.get('density', 'Not recorded')}")
        ttk.Label(body, text=summary, wraplength=560, justify="left").grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self._combo(body, 1, "Data", self.field, tuple(FIELDS))
        self._combo(body, 2, "Region", self.region, tuple(REGIONS))
        self._bounds_entry = self._entry(body, 3, "Bounds: xmin xmax ymin ymax zmin zmax (Å)", self.bounds)
        self._combo(body, 4, "Export", self.kind, ("Volume", "Isosurface"))
        self._format_combo = self._combo(body, 5, "Format", self.format, VOLUME_FORMATS)
        self._level_entry = self._entry(body, 6, "Surface isovalue (field units; signed)", self.level)
        self._advanced_toggle = ttk.Button(body, text="▸ Advanced options", command=self._toggle_advanced)
        self._advanced_toggle.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(6, 3))
        self._advanced_frame = ttk.Frame(body, padding=0)
        self._advanced_frame.columnconfigure(1, weight=1)
        self._advanced_frame.grid(row=8, column=0, columnspan=2, sticky="ew")
        advanced = self._advanced_frame
        self._precision_combo = self._combo(advanced, 0, "Precision (NPY / NPZ / NRRD)", self.precision, ("float64", "float32"))
        self._compression_check = ttk.Checkbutton(advanced, text="Compress NPZ / NRRD", variable=self.compressed)
        self._compression_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=3)
        self._smooth_check = ttk.Checkbutton(advanced, text="Smooth PCS export (Gaussian;\noriginal calculation is retained)",
                                             variable=self.smooth)
        self._smooth_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=3)
        self._sigma_entry = self._entry(advanced, 3, "Smoothing sigma (voxels)", self.sigma)
        self._advanced_frame.grid_remove()
        ttk.Label(body, text="MRC / CCP4 / MAP use Float32; text formats use Float64.\n"
                  "Exports include a JSON sidecar with units, coordinates and calculation provenance.",
                  style=ui_label_style(body, "muted"), wraplength=560, justify="left").grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=4)
        self._estimate = tk.StringVar()
        ttk.Label(body, textvariable=self._estimate, wraplength=560, justify="left").grid(
            row=12, column=0, columnspan=2, sticky="ew", pady=5)
        self._progress = ttk.Progressbar(body, mode="determinate", maximum=100)
        self._progress.grid(row=13, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(body, textvariable=self.status, wraplength=560, justify="left").grid(
            row=14, column=0, columnspan=2, sticky="ew", pady=5)
        self._details = ui_native_colors(
            tk.Text(body, width=52, height=5, wrap="word", state="disabled", font=ui_font_name(body, "report")),
            background="field", foreground="text",
        )
        self._details.grid(row=15, column=0, columnspan=2, sticky="nsew", pady=5)
        body.rowconfigure(15, weight=1)
        self._export = ttk.Button(actions, text="Export…", command=self._start)
        self._export.pack(side="left")
        ttk.Button(actions, text="Cancel export / Close", command=self._close).pack(side="right")
        self._traces = [
            (var, var.trace_add("write", self._options_changed))
            for var in (self.kind, self.field, self.region, self.precision, self.format, self.smooth)
        ]
        self._options_changed()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Destroy>", self._on_destroy, add="+")
        fit_dialog(self, form=form)

    def _combo(self, body, row, label, var, values):
        caption = ttk.Label(body, text=label, wraplength=260)
        caption.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
        widget = ttk.Combobox(body, textvariable=var, values=values, state="readonly")
        widget.grid(row=row, column=1, sticky="ew", pady=3)
        self._field_labels[widget] = caption
        return widget

    def _entry(self, body, row, label, var):
        caption = ttk.Label(body, text=label, wraplength=260)
        caption.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
        widget = ttk.Entry(body, textvariable=var)
        widget.grid(row=row, column=1, sticky="ew", pady=3)
        self._field_labels[widget] = caption
        return widget

    def _set_row_visible(self, widget, visible):
        for item in (self._field_labels[widget], widget):
            item.grid() if visible else item.grid_remove()

    def _toggle_advanced(self):
        self._advanced_open = not self._advanced_open
        self._advanced_frame.grid() if self._advanced_open else self._advanced_frame.grid_remove()
        self._update_advanced_caption()

    def _update_advanced_caption(self):
        self._advanced_toggle.configure(text=("▾" if self._advanced_open else "▸") + " Advanced options"
                                        + (" · smoothing on" if self.smooth.get() else ""))

    def _options_changed(self, *_):
        formats = SURFACE_FORMATS if self.kind.get() == "Isosurface" else VOLUME_FORMATS
        self._format_combo.configure(values=formats)
        if self.format.get() not in formats:
            self.format.set(formats[0])
        self._set_row_visible(self._bounds_entry, self.region.get() == "Custom bounds (Angstrom)")
        self._set_row_visible(self._level_entry, self.kind.get() == "Isosurface")
        self._set_row_visible(self._precision_combo, self.format.get() in (".npy", ".npz", ".nrrd"))
        (self._compression_check.grid if self.format.get() in (".npz", ".nrrd") else self._compression_check.grid_remove)()
        self._bounds_entry.configure(state="normal" if self.region.get() == "Custom bounds (Angstrom)" else "disabled")
        self._level_entry.configure(state="normal" if self.kind.get() == "Isosurface" else "disabled")
        self._precision_combo.configure(state="readonly" if self.format.get() in (".npy", ".npz", ".nrrd") else "disabled")
        self._compression_check.configure(state="normal" if self.format.get() in (".npz", ".nrrd") else "disabled")
        is_pcs = self.field.get() == "PCS field"
        self._smooth_check.configure(state="normal" if is_pcs else "disabled")
        self._sigma_entry.configure(state="normal" if is_pcs and self.smooth.get() else "disabled")
        if not is_pcs and self.smooth.get():
            self.smooth.set(False)
        self._set_row_visible(self._sigma_entry, is_pcs and self.smooth.get())
        (self._smooth_check.grid if is_pcs else self._smooth_check.grid_remove)()
        self._update_advanced_caption()
        shape = self.result["pcs_field"].shape if self.region.get() != "Input density region" else self.result["source_shape"]
        count = 1
        for n in shape:
            count *= int(n)
        size = count * (4 if self.precision.get() == "float32" else 8)
        self._estimate.set(f"Grid: {' × '.join(map(str, shape))}; raw array estimate: {size / 1024 ** 2:.1f} MiB.\n"
                           "Text files can be larger; custom bounds, compression and isosurfaces change output size."
                           + ("\nSmall NRRD grids may be interpreted as multi-component images by VTK readers; use VTI for these grids."
                              if self.format.get() == ".nrrd" and shape[0] <= 9 else ""))

    def _start(self):
        if self._running:
            return
        try:
            field = FIELDS[self.field.get()]
            region = REGIONS[self.region.get()]
            bounds = [float(x) for x in self.bounds.get().replace(",", " ").split()] if region == "custom" else None
            sigma = float(self.sigma.get()) if self.smooth.get() else 0.0
            if self.smooth.get() and (field != "pcs" or not 0 < sigma < float("inf")):
                raise ValueError("Smoothing requires PCS and a positive finite sigma.")
            level = float(self.level.get()) if self.kind.get() == "Isosurface" else None
            extension = self.format.get()
            dtype, compressed = self.precision.get(), self.compressed.get()
        except (ValueError, KeyError) as exc:
            messagebox.showerror("Export settings", str(exc), parent=self)
            return
        filename = filedialog.asksaveasfilename(parent=self, title="Export 3D data",
            defaultextension=extension, filetypes=[(extension[1:].upper(), "*" + extension)])
        if not filename:
            return
        path = Path(filename)
        if path.suffix.lower() != extension:
            messagebox.showerror("Export filename", f"Use the selected {extension} extension.", parent=self)
            return
        self._running = True
        self._cancel.clear()
        self._export.configure(state="disabled")
        self._progress.configure(value=0)
        self.status.set("Preparing data…")
        result, events, cancel = self.result, self._queue, self._cancel

        def worker():
            try:
                volume = prepare_volume(result, field, region, bounds, sigma)
                last = [-1]
                def progress(fraction):
                    percent = int(fraction * 100)
                    if percent != last[0]:
                        events.put(("progress", percent, ""))
                        last[0] = percent
                if level is None:
                    export_volume(path, volume, dtype, compressed, progress, cancel)
                else:
                    export_surface(path, volume, level, progress, cancel)
                events.put(("done", str(path), ""))
            except Exception as exc:
                events.put(("error", str(exc) or type(exc).__name__, traceback.format_exc()))
        threading.Thread(target=worker, daemon=True).start()
        self._poll_id = self.after(80, self._poll)

    def _poll(self):
        self._poll_id = None
        while True:
            try:
                kind, value, details = self._queue.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                self._progress.configure(value=value)
                self.status.set(f"Exporting… {value}%")
            else:
                self._running = False
                self._export.configure(state="normal")
                self.status.set(f"Exported: {value}" if kind == "done" else value)
                self._details.configure(state="normal")
                self._details.delete("1.0", "end")
                self._details.insert("end", details)
                self._details.configure(state="disabled")
                if kind == "done":
                    self._progress.configure(value=100)
        if self._running:
            self._poll_id = self.after(80, self._poll)

    def _close(self):
        if self._running:
            self._cancel.set()
            self.status.set("Cancellation requested. Waiting for the current writer step…")
        else:
            self.destroy()

    def _on_destroy(self, event):
        if event.widget is self:
            self._cancel.set()
            if self._poll_id is not None:
                self.after_cancel(self._poll_id)
            for var, token in self._traces:
                try:
                    var.trace_remove("write", token)
                except tk.TclError:
                    continue  # Interpreter shutdown can remove the variable first.
            self._traces = []
            # Release variables while still on the Tk thread, rather than when a
            # worker allocation happens to collect a destroyed window's cycles.
            for name, value in list(vars(self).items()):
                if isinstance(value, tk.Variable):
                    setattr(self, name, None)
