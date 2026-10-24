"""Review and apply an ORCA tensor through the host's existing model commands."""

import tkinter as tk
from tkinter import ttk, messagebox

from logic.document_edit_history import fingerprint
from logic.pcs_document import active_document_id, active_document, capture_active_document
from logic.pcs_tensor_state import get_canonical_tensor_state
from logic.pcs_orientation import tensor_in_convention
from logic.pcs_forward_api import EulerConvention
from logic.orca_tensor_transfer import (
    prepare_tensor_transfer, apply_tensor_transfer, source_details,
)
from ui.widget_style import ui_font_name, ui_text_color
from ui.dialog_layout import fit_dialog


class TensorTransferDialog(tk.Toplevel):
    def __init__(self, parent, state, payload):
        # Validate before opening a window; there is no partial apply path.
        self.preview = prepare_tensor_transfer(payload, state)
        if state.get("conformer_preview"):
            raise ValueError("Apply or discard the current conformer preview before importing a tensor.")
        super().__init__(parent)
        self.state_values = state
        get_canonical_tensor_state(state)
        self.target_id = active_document_id(state)
        self.target_context = fingerprint(state)
        self.title("Send tensor to PCS Analyzer")
        self.transient(parent)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        body = ttk.Frame(self, padding=14)
        body.grid(row=0, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        report = tk.Text(body, width=62, height=27, wrap="word", relief="flat", borderwidth=0,
                         font=ui_font_name(body, "report"), background=ui_text_color(body, "field"),
                         foreground=ui_text_color(body, "text"), padx=10, pady=8)
        report.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(body, command=report.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        report.configure(yscrollcommand=scrollbar.set)
        source, tensor = self.preview.provenance, self.preview.tensor
        current = get_canonical_tensor_state(state)
        current_zyz = tensor_in_convention(current, EulerConvention.INTRINSIC_ZYZ)
        target = active_document(state)
        report.insert("1.0", (
            f"Source: {source['file']}\nSection: {source.get('section')} · line {source.get('line')}\n"
            f"Temperature: {source['temperature']:g} K\n"
            f"Target: {target.title if target else 'Current PCS document'}\n\n"
            "                         Current        ORCA\n"
            f"Δχ_ax (×10^-32 m³)  {current.delta_chi_ax:12.6g}  {tensor.delta_chi_ax:12.6g}\n"
            f"Δχ_rh (×10^-32 m³)  {current.delta_chi_rh:12.6g}  {tensor.delta_chi_rh:12.6g}\n\n"
            f"Current orientation (intrinsic ZYZ, degrees):\n"
            f"  α {current_zyz.alpha:.6g} · β {current_zyz.beta:.6g} · γ {current_zyz.gamma:.6g}\n"
            f"New orientation (intrinsic ZYZ, degrees):\n"
            f"  α {tensor.alpha:.6g} · β {tensor.beta:.6g} · γ {tensor.gamma:.6g}\n\n"
            f"Structure: ordered atom correspondence / proper rigid alignment\n"
            f"Alignment RMSD: {self.preview.alignment_rmsd:.6g} Å\n"
            f"Maximum coordinate error: {self.preview.alignment_max_error:.6g} Å\n"
            f"Metal: Ref {source['target']['metal_ref']} (existing centre retained)\n"
            f"Point PCS agreement: max difference {self.preview.point_max_error:.6g} ppm\n\n"
            "Prediction model after Apply: Full tensor (Axial + Rhombic)\n"
            "The current fit preview will be cleared. Recorded fit results and experimental χ_mol remain unchanged.\n"
            "Apply can be reverted with one Undo."
        ))
        report.configure(state="disabled")
        self.confirm_order = tk.BooleanVar(value=False)
        ttk.Checkbutton(body, text="Confirm matching atom order",
                        variable=self.confirm_order, command=self._refresh_apply).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(12, 0))
        actions = ttk.Frame(body)
        actions.grid(row=2, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(actions, text="Cancel", command=self.destroy).pack(side="left", padx=6)
        self.apply_button = ttk.Button(actions, text="Apply", command=self.apply, state="disabled")
        self.apply_button.pack(side="left")
        self.bind("<Escape>", lambda _e: self.destroy())
        self.bind("<Destroy>", self._release_variables, add="+")
        fit_dialog(self)

    def _release_variables(self, event):
        if event.widget is self:
            self.confirm_order = None

    def _refresh_apply(self):
        self.apply_button.configure(state="normal" if self.confirm_order.get() else "disabled")

    def apply(self):
        if not self.confirm_order.get():
            return
        state = self.state_values
        if active_document_id(state) != self.target_id or fingerprint(state) != self.target_context:
            messagebox.showwarning("Target changed", "The target document or its inputs changed. Open a new preview.", parent=self)
            return
        try:
            errors = apply_tensor_transfer(state, self.preview)
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Tensor import", str(exc), parent=self)
            return
        capture_active_document(state)
        if errors:
            messagebox.showwarning("Tensor applied", "Tensor applied with view refresh warnings. Use Update to refresh the views.", parent=self)
        self.destroy()


def review_tensor_transfer(state, payload, parent):
    for key in ("finish_parameter_gesture", "check_edit_history_pending_inputs"):
        callback = state.get(key)
        if callable(callback):
            callback()
    try:
        return TensorTransferDialog(parent, state, payload)
    except (ValueError, TypeError, KeyError) as exc:
        messagebox.showerror("Cannot send tensor", str(exc), parent=parent)
        return None


def show_tensor_details(state):
    text = source_details(state)
    if not text:
        state["plots_nb"].select(state["rhtab"])
        return
    window = tk.Toplevel(state["root"])
    window.title("Tensor details")
    window.transient(state["root"])
    window.rowconfigure(0, weight=1)
    window.columnconfigure(0, weight=1)
    report = tk.Text(window, width=62, height=min(22, max(8, text.count("\n") + 2)),
                     wrap="word", font=ui_font_name(window, "report"),
                     background=ui_text_color(window, "field"), foreground=ui_text_color(window, "text"),
                     padx=12, pady=10, relief="flat", borderwidth=0)
    report.insert("1.0", text)
    report.configure(state="disabled")
    report.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(window, command=report.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    report.configure(yscrollcommand=scrollbar.set)
    actions = ttk.Frame(window, padding=8)
    actions.grid(row=1, column=0, columnspan=2, sticky="e")
    ttk.Button(actions, text="Open Rhombicity Table", command=lambda: (
        state["plots_nb"].select(state["rhtab"]), window.destroy())).pack(side="left", padx=6)
    ttk.Button(actions, text="Close", command=window.destroy).pack(side="left")

    fit_dialog(window)
