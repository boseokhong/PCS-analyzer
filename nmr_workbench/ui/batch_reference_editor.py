"""Human-readable batch reference choices; scientific parameters remain recipe data."""
import tkinter as tk
from tkinter import ttk
from ..reference_presets import load_reference_preset_library


class BatchReferenceEditor(ttk.Frame):
    TYPES = {"Reference standard": "reference_standard", "Residual solvent": "residual_solvent", "Custom target": "custom"}

    def __init__(self, parent, values, nuclei, on_change):
        super().__init__(parent)
        self.library = load_reference_preset_library()
        self.nuclei = set(nuclei)
        self.fields = {key: tk.StringVar(self, value="" if values.get(key) is None else str(values[key]))
                       for key in ("preset_id", "resonance_id", "target_ppm", "window_low_ppm", "window_high_ppm", "observed_ppm")}
        self.kind = tk.StringVar(self, value="Custom target" if values.get("target_ppm") is not None and not values.get("preset_id") else "Reference standard")
        self.preset_name = tk.StringVar(self)
        self.signal_name = tk.StringVar(self)
        self.note = tk.StringVar(self)
        self.presets = {}
        self.signals = {}
        self.selectors = {}
        for row, (label, variable, key, choices) in enumerate((
                ("Reference type", self.kind, "type", tuple(self.TYPES)),
                ("Reference compound", self.preset_name, "compound", ()),
                ("Reference signal", self.signal_name, "signal", ()))):
            grid_row, column = ((0, 0), (0, 2), (1, 0))[row]
            ttk.Label(self, text=label).grid(row=grid_row, column=column, sticky="w")
            widget = ttk.Combobox(self, textvariable=variable, values=choices, state="readonly", width=23)
            widget.grid(row=grid_row, column=column+1, sticky="ew", padx=5, pady=2)
            self.selectors[key] = widget
        self.selectors["type"].bind("<<ComboboxSelected>>", lambda event: self._type_changed())
        self.selectors["compound"].bind("<<ComboboxSelected>>", lambda event: self._compound_changed())
        self.selectors["signal"].bind("<<ComboboxSelected>>", lambda event: self._signal_changed())
        ttk.Label(self, text="Target position (ppm)").grid(row=1, column=2, sticky="w")
        self.target = ttk.Entry(self, textvariable=self.fields["target_ppm"], width=12)
        self.target.grid(row=1, column=3, sticky="w", padx=5)
        window = ttk.Frame(self)
        window.grid(row=2, column=0, columnspan=4, sticky="w", pady=4)
        ttk.Label(window, text="Search region (ppm), from").pack(side="left")
        ttk.Entry(window, textvariable=self.fields["window_low_ppm"], width=9).pack(side="left", padx=4)
        ttk.Label(window, text="to").pack(side="left")
        ttk.Entry(window, textvariable=self.fields["window_high_ppm"], width=9).pack(side="left", padx=4)
        ttk.Label(self, text="Drag a region on the ppm plot below, or enter its bounds.\nEach experiment must have exactly one detected candidate.", wraplength=600).grid(row=3, column=0, columnspan=4, sticky="w")
        ttk.Label(self, textvariable=self.note, wraplength=600).grid(row=4, column=0, columnspan=4, sticky="w", pady=4)
        self.manual = ttk.Frame(self)
        ttk.Label(self.manual, text="Observed position (ppm)").pack(side="left")
        ttk.Entry(self.manual, textvariable=self.fields["observed_ppm"], width=12).pack(side="left", padx=5)
        ttk.Label(self.manual, text="Use Override selected members only").pack(side="left")
        self.manual_visible = False
        ttk.Button(self, text="Manual position override…", command=self._toggle_manual).grid(row=5, column=0, columnspan=4, sticky="w")
        existing = self.fields["preset_id"].get()
        if existing:
            preset = next((p for p in self.library.presets if p.id == existing), None)
            if preset is not None:
                self.kind.set(next(name for name, category in self.TYPES.items() if category == preset.category))
        self._type_changed(restore=True)
        for variable in self.fields.values():
            variable.trace_add("write", lambda *args: on_change())
        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, weight=1)

    def _eligible(self, preset):
        return len(self.nuclei) == 1 and all(self.library.resonances_for(preset.id, nucleus, include_nonpreferred=False) for nucleus in self.nuclei)

    def _type_changed(self, restore=False):
        category = self.TYPES[self.kind.get()]
        custom = category == "custom"
        self.target.configure(state="normal" if custom else "readonly")
        for key in ("compound", "signal"):
            self.selectors[key].configure(state="disabled" if custom else "readonly")
        self.presets = {p.display_name: p for p in self.library.presets if p.category == category and self._eligible(p)}
        self.selectors["compound"].configure(values=tuple(self.presets))
        self.preset_name.set("")
        self.signal_name.set("")
        self.signals = {}
        self.selectors["signal"].configure(values=())
        if custom:
            if not restore:
                self.fields["preset_id"].set("")
                self.fields["resonance_id"].set("")
            self.note.set("Custom: enter a scientifically justified target. Search bounds are observed coordinates, not the target.")
        else:
            match = next((name for name, p in self.presets.items() if p.id == self.fields["preset_id"].get()), None) if restore else None
            if match:
                self.preset_name.set(match)
                self._compound_changed(restore=True)
            else:
                if not restore:
                    self.fields["preset_id"].set("")
                    self.fields["resonance_id"].set("")
                    self.fields["target_ppm"].set("")
                self.note.set("Choose a compound and signal." if self.presets else
                              "No compatible references: check nucleus metadata and use separate batches for different nuclei.")

    def _compound_changed(self, restore=False):
        preset = self.presets.get(self.preset_name.get())
        if preset is None:
            return
        self.fields["preset_id"].set(preset.id)
        self.signals = {}
        # Mixed-nucleus batches cannot share one resonance ID/target.
        if len(self.nuclei) == 1:
            nucleus = next(iter(self.nuclei))
            for resonance in self.library.resonances_for(preset.id, nucleus, include_nonpreferred=False):
                name = f"{resonance.label} — {resonance.ppm:g} ppm"
                self.signals[name] = resonance
        self.selectors["signal"].configure(values=tuple(self.signals))
        selected = next((name for name, r in self.signals.items() if r.id == self.fields["resonance_id"].get()), None) if restore else None
        self.signal_name.set(selected or "")
        if selected:
            self._signal_changed()
        elif not restore and len(self.signals) == 1:
            self.signal_name.set(next(iter(self.signals)))
            self._signal_changed()
        else:
            if not restore:
                self.fields["resonance_id"].set("")
                self.fields["target_ppm"].set("")
            self.note.set("Choose the reference signal; use separate batches for different nuclei.")

    def _signal_changed(self):
        resonance = self.signals.get(self.signal_name.get())
        if resonance is None:
            return
        self.fields["resonance_id"].set(resonance.id)
        self.fields["target_ppm"].set(f"{resonance.ppm:g}")
        condition = f"Tabulated at {resonance.temperature_K:g} K. " if resonance.temperature_K is not None else "No temperature calibration specified. "
        self.note.set(condition + "For VT, review target applicability at each temperature; values are not temperature-corrected automatically.")

    def _toggle_manual(self):
        self.manual_visible = not self.manual_visible
        if self.manual_visible:
            self.manual.grid(row=6, column=0, columnspan=4, sticky="w", pady=4)
        else:
            self.manual.grid_remove()
            self.fields["observed_ppm"].set("")

    def validate_choice(self):
        if self.kind.get() != "Custom target" and (self.preset_name.get() not in self.presets or self.signal_name.get() not in self.signals):
            raise ValueError("Choose a compatible reference compound and signal")
