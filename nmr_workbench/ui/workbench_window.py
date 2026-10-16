"""Tkinter NMR Workbench window with embedded Matplotlib spectrum view."""

from __future__ import annotations

from pathlib import Path
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

from ui.icons import IconManager, TOOLBAR_ICON_SIZE
from ui.ribbon.icon_service import nearest_packaged_icon_size, tk_scale_factor
from ui.style import get_app_fonts, table_row_height
from ui.tooltips import ToolTip
from .presentation import (
    AccordionHeader, WorkbenchFigureCanvas, WorkbenchPanedWindow, WorkbenchPresentation,
)
from .workspace import WorkbenchWorkspace, INSPECTOR_PREFERRED_WIDTH, WORKSPACE_OUTER_X

from .spectrum_interaction import (
    InteractionMode,
    SpectrumInteractionController,
)
from .matplotlib_interaction import MatplotlibSpectrumInteractionAdapter
from ..core import NMRPeak
from ..core.metadata import resolve_temperature
from ..analysis import (
    ObservableSyncError,
    PeakPickingError,
    PeakPickingParameters,
    assignments_from_observables,
    observable_for_peak,
    pick_positive_peaks_1d,
    sync_dataset_observables,
    sync_peak_observable,
    ShiftAnalysisError,
    build_shift_comparison,
    dataset_nucleus,
    roles_are_pairable,
    PCSDerivationContext,
    PCSDerivationError,
    builtin_pcs_providers,
    build_absolute_spectrum_positions,
    build_predicted_paramagnetic_peaks,
    build_shift_diagnostic_quantities,
    build_shift_diagnostics,
)
from ..integration.pcs_exp_transfer import (
    extract_pcs_targets,
    suggest_target_ref_id,
)
from ..integration.pcs_structured_handoff import (
    PCSStructuredHandoffError,
    STATE_KEY as PCS_OBSERVATIONS_STATE_KEY,
    apply_structured_pcs_handoff,
    build_handoff_rows,
    preview_handoff_summary,
)
from ..integration.structure_aware_pcs_targets import (
    PCSStructureTargetError,
    SCOPE_CUSTOM,
    SCOPE_MATCHING,
    SCOPE_NMR_SITE,
    SCOPE_SINGLE,
    TARGET_SCOPES,
    build_structure_target_catalog_from_state,
    parse_custom_ref_choice,
)

from ..integration.shared_selection import (
    publish_workbench_selection,
    subscribe_workbench_selection,
    unsubscribe_workbench_selection,
)
from ..integration.assignment_structure_link import (
    AssignmentStructureSelection,
    matching_site_selection,
    nmr_site_selection,
    resolve_assignment_structure_selection,
)
from ..integration.legacy_project import (
    LEGACY_DIAMAGNETIC_DATASET_ID,
    LEGACY_PARAMAGNETIC_DATASET_ID,
    sync_legacy_project_datasets,
)

from ..reference_presets import (
    ReferencePreset,
    ReferenceResonance,
    load_reference_preset_library,
)

from .processing_session import (
    ProcessingSession,
    ProcessingSessionError,
    suggested_zero_fill_sizes,
)
from .view_model import (
    WorkbenchState,
    dataset_display_name,
    dataset_summary,
    format_metadata_rows,
    format_processing_history,
    processing_display_state_label,
    should_invert_display_axis,
)
from .experiment_actions import ExperimentActions
from ..io.experiment_import import load_experiments
from ..version import __version__


def _mapping_state_signature(state: dict, key: str) -> tuple:
    value = state.get(key, {}) or {}
    if not isinstance(value, dict):
        return (("value", type(value).__name__, repr(value)),)
    return tuple(
        sorted(
            (
                type(item_key).__name__,
                repr(item_key),
                type(item_value).__name__,
                repr(item_value),
            )
            for item_key, item_value in value.items()
        )
    )


def _pcs_derived_state_signature(state: dict | None) -> tuple | None:
    """Return the host inputs that can change Workbench PCS diagnostics.

    The model revision deduplicates canonical event delivery.  The projected
    dictionaries remain part of the signature because legacy geometry and
    structure callbacks can rebuild ``pcs_by_id`` without advancing that
    revision.
    """
    if not isinstance(state, dict):
        return None

    selected = state.get("current_selected_ids", ()) or ()
    try:
        selected_signature = tuple(
            (type(value).__name__, repr(value)) for value in selected
        )
    except TypeError:
        selected_signature = ((type(selected).__name__, repr(selected)),)

    return (
        state.get("pcs_model_revision", 0),
        _mapping_state_signature(state, "pcs_by_id"),
        _mapping_state_signature(state, "atom_by_id"),
        _mapping_state_signature(state, "ref_label_overrides"),
        selected_signature,
        tuple(
            repr(record)
            for record in (
                state.get(PCS_OBSERVATIONS_STATE_KEY, ()) or ()
            )
        ),
    )


def _legacy_shift_state_signature(state: dict | None) -> tuple | None:
    """Return legacy shift inputs, or ``None`` when that bridge is unused."""
    if not isinstance(state, dict):
        return None
    if "delta_obs_values" not in state and "delta_dia_values" not in state:
        return None

    return (
        _mapping_state_signature(state, "delta_obs_values"),
        _mapping_state_signature(state, "delta_dia_values"),
        _mapping_state_signature(state, "ref_label_overrides"),
    )


# Presentation-only spacing; keep processing and interaction constants separate.
_PANEL_PADDING = 4
_SECTION_PADDING = (8, 4, 4, 6)
_CONTROL_GAP = 6
_ROW_GAP = 3
_ACTION_GAP = 4
_HEADER_GAP = 2
_GROUP_GAP = 8


class _Accordion(ttk.Frame):
    """Small single-open-section accordion suitable for future controls."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, padding=0)
        self._sections: list[tuple[ttk.Button, ttk.Frame, str]] = []
        self._open: ttk.Frame | None = None

    def add(self, title: str, *, initially_open: bool = False) -> ttk.Frame:
        body = ttk.Frame(self, padding=_SECTION_PADDING)
        button = AccordionHeader(
            self,
            text=f"▸  {title}",
            style="NMR.Section.TButton",
            command=lambda: self._toggle(button, body, title),
        )
        button.pack(fill="x", pady=(0, _HEADER_GAP))
        self._sections.append((button, body, title))
        if initially_open:
            self.after_idle(lambda: self._toggle(button, body, title))
        return body

    def _toggle(self, button: ttk.Button, body: ttk.Frame, title: str) -> None:
        if self._open is body:
            body.pack_forget()
            button.configure(text=f"▸  {title}")
            self._open = None
            return
        for other_button, other_body, other_title in self._sections:
            other_body.pack_forget()
            other_button.configure(text=f"▸  {other_title}")
        body.pack(fill="x", after=button)
        button.configure(text=f"▾  {title}")
        self._open = body


class _VerticalScrolledFrame(ttk.Frame):
    """A ttk container with a stable-width vertically scrollable interior."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, padding=0)

        self._canvas = tk.Canvas(
            self,
            highlightthickness=0,
            borderwidth=0,
            relief="flat",
        )
        self._scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self._canvas.yview,
        )
        self.interior = ttk.Frame(self._canvas, padding=0)

        self._window_id = self._canvas.create_window(
            (0, 0),
            window=self.interior,
            anchor="nw",
        )
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar.pack(side="right", fill="y")

        self.interior.bind("<Configure>", self._on_interior_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_interior_configure(self, _event=None) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        # Keep the processing controls at the viewport width so the user only
        # scrolls vertically; no horizontal scrollbar is needed.
        self._canvas.itemconfigure(self._window_id, width=event.width)

    def bind_mousewheel_recursive(self) -> None:
        """Make wheel scrolling work while the pointer is over any child control."""

        def bind_widget(widget) -> None:
            widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_mousewheel, add="+")
            widget.bind("<Button-5>", self._on_mousewheel, add="+")
            for child in widget.winfo_children():
                bind_widget(child)

        bind_widget(self)

    def _on_mousewheel(self, event):
        # Windows/macOS report event.delta; X11 commonly reports Button-4/5.
        if getattr(event, "num", None) == 4:
            units = -1
        elif getattr(event, "num", None) == 5:
            units = 1
        else:
            delta = int(getattr(event, "delta", 0) or 0)
            if delta == 0:
                return None
            units = -1 if delta > 0 else 1

        bbox = self._canvas.bbox("all")
        if not bbox:
            return None

        content_height = bbox[3] - bbox[1]
        viewport_height = self._canvas.winfo_height()
        if content_height <= viewport_height:
            return None

        self._canvas.yview_scroll(units * 3, "units")
        return "break"


class NMRWorkbenchWindow(ExperimentActions, tk.Toplevel):
    """Workflow-oriented shell for loading, inspecting, and processing 1D NMR."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        pcs_state: dict | None = None,
    ) -> None:
        super().__init__(parent)
        self.title("NMR Workbench")
        self.geometry("1280x820")
        self.minsize(900, 620)
        fonts = getattr(parent._root(), "_app_fonts", {}) or {}
        self._title_font = fonts.get("title", ("Segoe UI", 13, "bold"))
        self._presentation = WorkbenchPresentation(self)

        self.model = WorkbenchState()
        self.processing_session = ProcessingSession()
        # Reserved for the later PCS Preparation -> δ_exp bridge.
        # Shift Analysis itself stays inside NMR Workbench.
        self._pcs_state = pcs_state or self._discover_pcs_state(parent)
        self._dataset_pair_ids: dict[str, str] = {}
        self._pair_display_to_id: dict[str, str] = {}
        self._pcs_providers = {
            provider.display_name: provider
            for provider in builtin_pcs_providers()
        }
        self._pcs_corrections_ppm: dict[str, float] = {}
        self._pcs_manual_values_ppm: dict[str, float] = {}
        self._pcs_uncertainties_ppm: dict[str, float] = {}
        self._prepared_pcs_values: dict[str, object] = {}
        self._pcs_target_by_assignment: dict[str, int] = {}
        self._pcs_excluded_assignments: set[str] = set()
        self._pcs_target_display_to_id: dict[str, int] = {}
        # D5c/D5d: structure-aware target planning is kept separately from
        # the legacy single-Ref mapping so grouped sites are never flattened
        # into repeated delta_exp values during handoff.
        self._pcs_target_scope_by_assignment: dict[str, str] = {}
        self._pcs_target_choice_key_by_assignment: dict[str, str] = {}
        self._pcs_custom_target_text_by_assignment: dict[str, str] = {}
        self._pcs_target_display_to_key: dict[str, str] = {}
        self._pcs_structure_target_catalog = None
        self._pcs_structure_target_graph_identity: int | None = None
        # D5e: structure navigation from assignments is preview-only until the
        # user explicitly chooses "Use in PCS Preparation".
        self._structure_preview_choice_key_by_assignment: dict[str, str] = {}

        self._reference_library = load_reference_preset_library()
        self._reference_preset_by_display: dict[str, ReferencePreset] = {}
        self._reference_resonance_by_display: dict[str, ReferenceResonance] = {}
        self._tree_dataset_ids: dict[str, str] = {}
        self._view_choice = tk.StringVar(value="")
        self._coordinate = tk.StringVar(value="")
        self._state_text = tk.StringVar(value="No dataset loaded")
        self._dataset_text = tk.StringVar(value="Open an NMR experiment to begin.")
        self._interaction_mode_text = tk.StringVar(value="Mode: Navigate")
        self._pcs_binding_text = tk.StringVar(value="PCS: not bound")

        # Assignment is intentionally kept separate from NMRPeak.  A picked
        # peak is a spectral feature; an assignment is interpretation layered
        # on top of it.  Drafts are keyed by dataset id, then peak id.
        self._assignment_drafts_by_dataset: dict[str, dict[str, str]] = {}
        self._assignment_filter = tk.StringVar(value="All")
        self._assignment_editor: ttk.Entry | None = None
        self._assignment_editor_peak_id: str | None = None

        # Renderer-independent spectrum interaction state.  The Matplotlib
        # adapter is connected only after the plot canvas/axes exist.
        self._spectrum_interaction = SpectrumInteractionController()
        self._interaction_cache_source_id: int | None = None

        self._build_toolbar()
        self._build_workspace()

        # Old .pcsp projects may contain only host-side shift dictionaries.
        # Establish that compatibility projection once before subscriptions;
        # later PCS-only events do not rebuild these datasets.
        self.refresh_from_pcs_state()

        # D4d.2: semantic NMR selection now uses the same shared SelectionHub
        # as PCS table, Chemical Map, 3D and PCS/polar views.
        self._shared_selection_syncing = False
        self._shared_selection_subscription = subscribe_workbench_selection(
            self._pcs_state,
            self._on_shared_selection_context,
            fire_immediately=True,
        )

        self._mpl_interaction = MatplotlibSpectrumInteractionAdapter(
            axes=self._axes,
            canvas=self._canvas,
            controller=self._spectrum_interaction,
            hover_threshold_px=10.0,
            on_peak_selected=self._on_reference_peak_selected,
            on_peak_picked=self._on_peak_picked,
            on_view_changed=self._on_spectrum_view_changed,
            on_navigate_click=self._on_spectrum_peak_click,
        )
        self._mpl_interaction.connect()

        # Mnova-style keyboard interaction.  Bind to this Toplevel only so
        # shortcuts do not leak into the rest of PCS Analyzer.
        self.bind("<KeyPress>", self._on_workbench_key, add="+")

        self._build_statusbar()
        self._setup_experiment_actions()
        self._enable_experiment_drop(self._canvas.get_tk_widget())
        self._draw_empty_plot()
        self._build_menubar()
        self._presentation.refresh(force=True)

    @staticmethod
    def _discover_pcs_state(parent: tk.Misc) -> dict | None:
        candidates = []
        root = getattr(parent, "_root", lambda: None)()
        for obj in (parent, root):
            if obj is None:
                continue
            for name in ("state", "_state", "_app_state", "app_state"):
                value = getattr(obj, name, None)
                if isinstance(value, dict):
                    candidates.append(value)
        markers = {
            "current_selected_ids", "atom_by_id", "delta_obs_values",
            "delta_dia_values", "delta_exp_values", "tree",
        }
        for value in candidates:
            if markers.intersection(value):
                return value
        return None

    @property
    def current_dataset(self):
        return self.model.current_dataset

    def _pcs_target_state(self) -> dict | None:
        """Resolve the launch-time PCS target without silently retargeting."""
        affinity = getattr(self, "_pcs_document_affinity", None)
        if affinity is not None:
            return affinity.resolve_state()
        return self._pcs_state

    def refresh_pcs_document_binding(self) -> None:
        """Refresh pinned PCS identity presentation without changing affinity."""
        from logic.pcs_document import active_document, document_display_name

        affinity = getattr(self, "_pcs_document_affinity", None)
        document = None
        if affinity is not None:
            try:
                document = affinity.resolve_document()
            except RuntimeError:
                pass

        if document is None:
            label = "PCS document closed" if affinity is not None else "PCS: not bound"
            self.title(f"NMR Workbench — {label}")
            self._pcs_binding_text.set(label)
        else:
            name = document_display_name(document)
            self.title(f"NMR Workbench — PCS: {name}")
            self._pcs_binding_text.set(f"PCS: {name}")

        active = active_document(self._pcs_state) if self._pcs_state is not None else None
        can_rebind = active is not None and (
            affinity is None or active.document_id != affinity.document_id
        )
        button = getattr(self, "_pcs_rebind_button", None)
        if button is not None:
            button.configure(state="normal" if can_rebind else "disabled")

        pcs_actions_available = False
        if affinity is not None:
            try:
                pcs_actions_available = affinity.resolve_state() is not None
            except RuntimeError:
                pass
        review = getattr(self, "_pcs_review_apply_button", None)
        if review is not None and not pcs_actions_available:
            review.configure(state="disabled")

    def _rebind_to_active_pcs_document(self) -> None:
        """User-triggered affinity change; NMR datasets remain untouched."""
        from nmr_workbench.integration.pcs_document_affinity import (
            bind_to_active_pcs_document,
        )

        affinity = getattr(self, "_pcs_document_affinity", None)
        if affinity is None:
            affinity = bind_to_active_pcs_document(self._pcs_state)
            if affinity is None:
                return
            self._pcs_document_affinity = affinity
        else:
            affinity.rebind_to_active()
        self.bound_pcs_document_id = affinity.document_id
        for attribute in ("_pcs_refresh_signature", "_legacy_shift_refresh_signature"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        self.refresh_pcs_document_binding()
        self.refresh_from_pcs_state()

    def load_path(self, path: str | Path) -> None:
        """Load a path through the public vendor-neutral loader."""

        result = load_experiments((path,))
        if not result.loaded and result.failures:
            raise ValueError(result.failures[0][1])
        self._publish_experiment_import(result)

    def _rebuild_experiment_tree(self) -> None:
        """Rebuild legacy-compatible experiment rows and preserve selection."""
        tree = getattr(self, "_experiment_tree", None)
        if tree is None:
            return

        selected_dataset_id = self.model.selected_dataset_id
        for item in tree.get_children(""):
            tree.delete(item)
        self._tree_dataset_ids.clear()

        parent_for_dataset: dict[str, str] = {}
        datasets = self.experiments.datasets() if hasattr(self,"experiments") else self.model.datasets
        for dataset in datasets:
            parent_item = tree.insert(
                "",
                "end",
                text=dataset_display_name(dataset),
                open=True,
                tags=("active",) if dataset.id == selected_dataset_id else (),
            )
            detail_item = tree.insert(
                parent_item,
                "end",
                text=dataset_summary(dataset),
            )
            self._tree_dataset_ids[parent_item] = dataset.id
            self._tree_dataset_ids[detail_item] = dataset.id
            parent_for_dataset[dataset.id] = parent_item

        selected_ids = self.experiments.selected_ids if hasattr(self,"experiments") else ()
        selected_item = parent_for_dataset.get(selected_dataset_id)
        tree.selection_set([parent_for_dataset[i] for i in selected_ids if i in parent_for_dataset] or ([selected_item] if selected_item else []))
        if hasattr(self, "_batch_selected_button"):
            self._batch_selected_button.configure(state="normal" if tree.selection() else "disabled")
        if selected_item is not None:
            tree.focus(selected_item)
            tree.see(selected_item)

    def _build_menubar(self) -> None:
        """Additional access to existing actions; menu posting reads live UI state."""
        menubar = tk.Menu(self, tearoff=False)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open Dataset Folder...", command=self._open_dataset_folder)
        file_menu.add_command(label="Open FID File...", command=self._open_fid_file)
        file_menu.add_separator()
        file_menu.add_command(label="Assignments / Saved Analysis...", command=self._open_data_exchange)
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self._close_from_menu)
        menubar.add_cascade(label="File", menu=file_menu)

        self._view_menu = tk.Menu(menubar, tearoff=False, postcommand=self._sync_menubar)
        for choice in ("Spectrum", "Raw FID"):
            self._view_menu.add_radiobutton(
                label=choice, value=choice, variable=self._view_choice,
                command=self._on_view_selection,
            )
        self._view_menu.add_separator()
        self._view_menu.add_command(label="Reset View", accelerator="F", command=self._reset_view)
        self._view_menu.add_command(label="Auto Scale", accelerator="H", command=self._auto_scale_y)
        menubar.add_cascade(label="View", menu=self._view_menu)

        self._processing_menu = tk.Menu(menubar, tearoff=False, postcommand=self._sync_menubar)
        self._processing_menu.add_command(label="Create Spectrum", command=self._raw_processing_action)
        self._processing_menu.add_separator()
        for section, index in (("Phase", 3), ("Baseline", 4), ("Reference", 5), ("Peak Picking", 6)):
            self._processing_menu.add_command(
                label=section, command=lambda index=index: self._show_processing_section(index)
            )
        menubar.add_cascade(label="Processing", menu=self._processing_menu)

        self._interaction_menu = tk.Menu(menubar, tearoff=False, postcommand=self._sync_menubar)
        self._interaction_menu.add_command(label="Navigate", accelerator="Esc", command=self._enter_navigate_mode)
        self._interaction_menu.add_command(label="Zoom", accelerator="Z", command=self._toggle_zoom_mode)
        self._interaction_menu.add_command(label="Peak Picking", accelerator="K", command=self._begin_peak_picking)
        self._interaction_menu.add_command(label="Reference", accelerator="L", command=self._begin_reference_peak_selection)
        menubar.add_cascade(label="Interaction", menu=self._interaction_menu)

        series_menu = tk.Menu(menubar, tearoff=False)
        for label, command in (("Manage VT Series...", self._open_series_manager),
                               ("Stacked Spectra", self._show_vt_stack),
                               ("Batch Processing...", self._open_batch_processing),
                               ("Peak Tracking...", self._open_tracking),
                               ("VT Analysis...", self._open_vt_analysis)):
            series_menu.add_command(label=label, command=command)
        menubar.add_cascade(label="VT Series", menu=series_menu)

        self._pcs_menu = tk.Menu(menubar, tearoff=False, postcommand=self._sync_menubar)
        self._pcs_menu.add_command(label="Rebind to Active PCS Document", command=self._rebind_to_active_pcs_document)
        menubar.add_cascade(label="PCS", menu=self._pcs_menu)
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_keyboard_shortcuts)
        help_menu.add_command(label="Workflow & File Guide...", command=self._open_workflow_help)
        help_menu.add_separator()
        help_menu.add_command(label=f"NMR Workbench v{__version__}", state="disabled")
        menubar.add_cascade(label="Help", menu=help_menu)
        self.configure(menu=menubar)
        self._menubar = menubar
        self._sync_menubar()

    def _sync_menubar(self) -> None:
        """Mirror existing availability and controller state, without owning it."""
        choices = self._view_selector.cget("values")
        for index, choice in enumerate(("Spectrum", "Raw FID")):
            self._view_menu.entryconfigure(index, state="normal" if choice in choices else "disabled")
        # The accordion action uses the same label/eligibility as the conditional
        # toolbar action, and is accessible even when that toolbar button is hidden.
        self._processing_menu.entryconfigure(
            0, label=self._flow_ft_button.cget("text"),
            state=self._flow_ft_button.cget("state"),
        )
        mode = self._spectrum_interaction.mode
        for index, (label, value) in enumerate((
            ("Navigate", InteractionMode.NAVIGATE), ("Zoom", InteractionMode.ZOOM),
            ("Peak Picking", InteractionMode.PEAK_PICK), ("Reference", InteractionMode.REFERENCE_PEAK),
        )):
            self._interaction_menu.entryconfigure(index, label=f"✓ {label}" if mode == value else label)
        picking = mode == InteractionMode.PEAK_PICK
        referencing = mode == InteractionMode.REFERENCE_PEAK
        self._interaction_menu.entryconfigure(
            2, command=self._stop_peak_picking if picking else self._begin_peak_picking,
            state="normal" if picking else self._peak_pick_start_button.cget("state"),
        )
        self._interaction_menu.entryconfigure(
            3, command=self._enter_navigate_mode if referencing else self._begin_reference_peak_selection,
            state="normal" if referencing else self._reference_pick_button.cget("state"),
        )
        self._pcs_menu.entryconfigure(0, state=self._pcs_rebind_button.cget("state"))

    def _show_processing_section(self, index: int) -> None:
        button, body, _title = self._processing_accordion._sections[index]
        if self._processing_accordion._open is not body:
            button.invoke()
        self.update_idletasks()
        canvas = self._processing_scroller._canvas
        bounds = canvas.bbox("all")
        if bounds and bounds[3] > 0:
            canvas.yview_moveto(button.winfo_y() / bounds[3])

    def _close_from_menu(self) -> None:
        # The launcher installs host-state cleanup after construction. Look up
        # that live protocol rather than capturing or duplicating its callback.
        close = self.protocol("WM_DELETE_WINDOW")
        if close:
            self.tk.call("eval", close)
        else:
            self.destroy()

    def _show_keyboard_shortcuts(self) -> None:
        messagebox.showinfo(
            "NMR Workbench — Keyboard Shortcuts",
            "Z — Zoom\nF — Full / reset view\nH — Fit intensity height\n"
            "K — Peak picking\nL — Reference\nEscape — Navigate\n\n"
            "Wheel — Height\nShift+Wheel — Horizontal zoom\n"
            "Left drag in Zoom mode — Horizontal zoom\nMiddle/right drag — Pan\n\n"
            "Single-letter shortcuts are suppressed while editing text fields.",
            parent=self,
        )

    def _build_toolbar(self) -> None:
        root = self._root()
        icons = getattr(root, "_icon_manager", None)
        if icons is None:
            icons = IconManager(root)
        icon_size = nearest_packaged_icon_size(TOOLBAR_ICON_SIZE, tk_scale_factor(root))
        fonts = get_app_fonts(root)
        ttk.Style(self).configure(
            "NMR.Toolbar.TButton", padding=(_ACTION_GAP, _HEADER_GAP), width=0
        )
        toolbar = ttk.Frame(self, padding=(_CONTROL_GAP, _ROW_GAP))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="NMR", font=fonts["ui_bold"]).pack(
            side="left", padx=(0, _GROUP_GAP)
        )
        open_folder = ttk.Button(
            toolbar, text="Open Folder…", command=self._open_dataset_folder,
            style="NMR.Toolbar.TButton",
            image=icons.get("open_folder", size=icon_size), compound="left",
        )
        open_folder.pack(side="left")
        ToolTip(open_folder, "Open an NMR experiment or a parent folder containing multiple experiments.")
        open_fid = ttk.Button(
            toolbar, text="Open FID…", command=self._open_fid_file, style="NMR.Toolbar.TButton",
            image=icons.get("open_fid", size=icon_size), compound="left",
        )
        open_fid.pack(side="left", padx=(_ACTION_GAP, 0))
        ToolTip(open_fid, "Open a raw FID file from an NMR experiment.")
        ttk.Separator(toolbar, orient="vertical").pack(
            side="left", fill="y", padx=_GROUP_GAP
        )
        reset_view = ttk.Button(
            toolbar, text="Reset View", command=self._reset_view,
            style="NMR.Toolbar.TButton",
            image=icons.get("reset", size=icon_size), compound="left",
        )
        reset_view.pack(side="left")
        ToolTip(reset_view, "Restore the full spectrum view (F).")
        auto_scale = ttk.Button(
            toolbar, text="Auto Scale", command=self._auto_scale_y, style="NMR.Toolbar.TButton",
            image=icons.get("auto_scale", size=icon_size), compound="left",
        )
        auto_scale.pack(
            side="left", padx=(_CONTROL_GAP, 0)
        )
        ToolTip(auto_scale, "Fit the intensity height to the visible spectrum (H).")
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=_GROUP_GAP)
        from .actions import action
        self._workflow_toolbar = toolbar
        self._series_toolbar_button = action(toolbar, "VT Series", "nmr_workbench", self._open_series_manager)
        self._series_toolbar_button.pack(side="left")
        self._exchange_toolbar_button = action(toolbar, "Assignments / Analysis", "toolbox", self._open_data_exchange,
            description="Review assignments and import/export saved analysis results")
        self._exchange_toolbar_button.pack(side="left", padx=_ACTION_GAP)
        self._guide_toolbar_button = action(toolbar, "Guide", "search", self._open_workflow_help,
            description="Open the NMR workflow and file guide")
        self._guide_toolbar_button.pack(side="left")
        # Keep dataset/destination context separate from the primary tools so
        # the tool row remains usable at the workspace's minimum width.
        information = ttk.Frame(self, padding=(_CONTROL_GAP, 0, _CONTROL_GAP, _ROW_GAP))
        information.pack(fill="x")
        self._pcs_rebind_button = ttk.Button(
            information,
            text="Rebind PCS",
            style="NMR.Toolbar.TButton",
            command=self._rebind_to_active_pcs_document,
        )
        self._pcs_rebind_button.pack(side="left", padx=(0, 8))
        ToolTip(self._pcs_rebind_button, "Bind this Workbench to the active PCS document.")
        ttk.Label(information, text="View:").pack(side="left")
        self._view_selector = ttk.Combobox(
            information, textvariable=self._view_choice, values=(), width=12, state="disabled"
        )
        self._view_selector.pack(side="left", padx=(_ACTION_GAP, 0))
        self._view_selector.bind("<<ComboboxSelected>>", self._on_view_selection)
        ToolTip(self._view_selector, "Display Spectrum or Raw FID. Switching to Raw FID cancels an active preview.")
        self._reprocess_button = ttk.Button(
            toolbar, text="Start Reprocessing", command=self._raw_processing_action,
            style="NMR.Toolbar.TButton",
        )
        ToolTip(self._reprocess_button, "Create a spectrum from the prepared FID, or restart processing from the original FID.")
        # A fixed information row keeps variable-length status out of the action row.
        ttk.Label(information, textvariable=self._state_text, font=fonts["ui_small"]).pack(side="left", padx=_GROUP_GAP)
        ttk.Label(information, textvariable=self._pcs_binding_text, font=fonts["ui_small"]).pack(
            side="right", padx=(_GROUP_GAP, 0)
        )

    def _build_workspace(self) -> None:
        """Build the primary three-column NMR workflow.

        Left   : experiments + processing
        Center : spectrum
        Right  : peak/shift/PCS inspector
        """
        main = WorkbenchWorkspace(self)
        self._workspace = main
        main.pack(fill="both", expand=True, padx=WORKSPACE_OUTER_X, pady=(0, 4))

        left = WorkbenchPanedWindow(main, orient="vertical", width=main.left_width)
        # Keep titles clear of the boundary gutter's narrow pane-edge overhang.
        center = ttk.Frame(main, width=700, padding=(_PANEL_PADDING, 0, 0, 0))
        right = ttk.Frame(main, width=INSPECTOR_PREFERRED_WIDTH,
                          padding=(_PANEL_PADDING, 0, 0, 0))

        main.add(left, weight=0)
        main.add(center, weight=1)
        main.add(right, weight=0)

        experiments = ttk.LabelFrame(left, text="Experiments", padding=_PANEL_PADDING)
        processing = ttk.Frame(left, padding=_PANEL_PADDING)
        left.add(experiments, weight=1)
        left.add(processing, weight=1)
        self._left_sections = left
        self._processing_header = ttk.Frame(processing)
        self._processing_header.pack(fill="x", pady=(0, _ROW_GAP))
        ttk.Label(self._processing_header, text="Processing", font=get_app_fonts(self)["ui_bold"]).pack(side="left")
        from .actions import action
        self._batch_selected_button = action(self._processing_header, "Batch Processing…", "run",
            self._batch_selected_experiments, state="disabled",
            description="Preview a processing recipe for the selected experiments, then explicitly apply it")
        self._batch_selected_button.pack(side="right")

        # Inherit theme colors/label styling; only lighten the surrounding border.
        ttk.Style(self).configure("NMR.Spectrum.TLabelframe", borderwidth=0, relief="flat")
        plot_frame = ttk.LabelFrame(
            center, text="Spectrum", padding=_HEADER_GAP, style="NMR.Spectrum.TLabelframe"
        )
        plot_frame.pack(fill="both", expand=True)

        inspector = ttk.LabelFrame(right, text="Inspector", padding=_PANEL_PADDING)
        inspector.pack(fill="both", expand=True)

        self._build_experiments(experiments)

        self._processing_scroller = _VerticalScrolledFrame(processing)
        self._processing_scroller.pack(fill="both", expand=True)
        self._build_processing(self._processing_scroller.interior)
        self._processing_scroller.bind_mousewheel_recursive()

        self._spectrum_tabs = ttk.Notebook(plot_frame, style="NMR.Inspector.TNotebook")
        self._spectrum_tabs.pack(fill="both", expand=True)
        single = ttk.Frame(self._spectrum_tabs)
        self._spectrum_tabs.add(single, text="Single spectrum")
        self._build_plot(single)
        from .stacked_series_view import StackedSeriesView
        self._stacked_view = StackedSeriesView(
            self._spectrum_tabs, self.model,
            on_member_selected=self._activate_series_dataset,
            on_series_selected=self._refresh_series_manager,
            on_peak_selected=self._activate_tracking_peak,
            on_tracking_requested=self._open_tracking,
            on_analysis_requested=self._open_vt_analysis,
            on_batch_requested=self._open_batch_processing,
        )
        self._spectrum_tabs.add(self._stacked_view, text="VT stacked")
        self._spectrum_tabs.bind("<<NotebookTabChanged>>", lambda event: self._refresh_stacked_series())
        self._build_inspector(inspector)
        main.attach_panes(left, center, right)

    def _build_experiments(self, parent: ttk.Frame) -> None:
        fonts = get_app_fonts(self)
        ttk.Style(self).configure(
            "NMR.Experiments.Treeview",
            rowheight=table_row_height(self, fonts["table"], fonts["table_rowheight"]) + 2,
            indent=16
        )
        ttk.Style(self).configure("NMR.Role.TLabelframe", borderwidth=0, relief="flat")
        ttk.Label(
            parent, textvariable=self._dataset_text, wraplength=220, justify="left",
            font=fonts["ui_small_bold"],
        ).pack(
            fill="x", pady=(0, _ROW_GAP)
        )
        tree_frame = ttk.Frame(parent, padding=0)
        tree_frame.pack(fill="both", expand=True)
        self._experiment_tree = ttk.Treeview(
            tree_frame, show="tree", selectmode="extended", height=8,
            style="NMR.Experiments.Treeview",
        )
        scrollbar = ttk.Scrollbar(tree_frame, command=self._experiment_tree.yview)
        self._experiment_tree.configure(yscrollcommand=scrollbar.set)
        self._experiment_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._experiment_tree.bind("<<TreeviewSelect>>", self._on_tree_selection)

        role_frame = ttk.LabelFrame(
            parent, text="Dataset role", padding=_HEADER_GAP, style="NMR.Role.TLabelframe"
        )
        # Reserve the role controls before the elastic experiment list.
        role_frame.pack(side="bottom", before=tree_frame, fill="x", pady=(_CONTROL_GAP, 0))

        role_row = ttk.Frame(role_frame, padding=0)
        role_row.pack(fill="x")
        ttk.Label(role_row, text="Role", width=8, anchor="w").pack(side="left")
        self._dataset_role = tk.StringVar(value="Unspecified")
        self._dataset_role_box = ttk.Combobox(
            role_row,
            textvariable=self._dataset_role,
            values=("Unspecified", "Paramagnetic", "Diamagnetic"),
            state="disabled",
            width=16,
        )
        self._dataset_role_box.pack(side="right", fill="x", expand=True, padx=(_CONTROL_GAP, 0))
        self._dataset_role_box.bind(
            "<<ComboboxSelected>>",
            self._on_dataset_role_changed,
        )

        pair_row = ttk.Frame(role_frame, padding=0)
        pair_row.pack(fill="x", pady=(_ROW_GAP, 0))
        ttk.Label(pair_row, text="Pair with", width=8, anchor="w").pack(side="left")
        self._dataset_pair = tk.StringVar(value="")
        self._dataset_pair_box = ttk.Combobox(
            pair_row,
            textvariable=self._dataset_pair,
            values=(),
            state="disabled",
            width=16,
        )
        self._dataset_pair_box.pack(side="right", fill="x", expand=True, padx=(_CONTROL_GAP, 0))
        self._dataset_pair_box.bind(
            "<<ComboboxSelected>>",
            self._on_dataset_pair_changed,
        )

        self._dataset_pair_status = tk.StringVar(
            value="Mark datasets as Paramagnetic or Diamagnetic to compare shifts."
        )
        ttk.Label(
            role_frame,
            textvariable=self._dataset_pair_status,
            wraplength=220,
            justify="left",
        ).pack(fill="x", pady=(_ROW_GAP, 0))

    def _open_series_manager(self) -> None:
        from .series_window import SeriesManagementWindow
        window = getattr(self, "_series_window", None)
        if window is not None and window.winfo_exists():
            window.refresh()
            window.lift()
            return

        self._series_window = SeriesManagementWindow(
            self, self.model, on_activate=self._activate_series_dataset,
            on_datasets_changed=self._refresh_series_datasets,
            on_changed=self._refresh_stacked_series,
        )

    def _open_batch_processing(self, *, context=None, operation=None, parameters=None):
        window = getattr(self, "_batch_window", None)
        if window is not None and window.winfo_exists():
            if context is None and window.context is None:
                window.refresh()
                window.lift()
                return
            window.destroy()
        from .batch_window import BatchProcessingWindow
        def committed_spectra():
            session = self.processing_session
            return {session.dataset.id: session.committed_spectrum} if session.dataset else {}
        def blocked_members():
            session = self.processing_session
            contexts = [context] if context is not None else self.model.series
            return [m.id for s in contexts for m in s.members
                    if session.has_preview and session.dataset is m.dataset]
        self._batch_window = BatchProcessingWindow(self, self.model, on_committed=self._batch_committed,
            committed_spectra=committed_spectra, blocked_members=blocked_members, context=context)
        if context is not None:
            first = next((m.id for m in context.members if m.included), None)
            if first is not None:
                self._batch_window.members.selection_set(first)
                self._batch_window.members.focus(first)
        if operation is not None:
            row = next(r for r in self._batch_window.rows if r["operation"] == operation)
            row["enabled"] = True
            if parameters is not None:
                row["parameters"] = parameters
            self._batch_window._draw_operations()
            self._batch_window.operations.selection_set(operation)
            self._batch_window._edit_parameters()

    def _batch_committed(self, results):
        changed = {r.dataset_id for r in results if r.status == "succeeded"}
        for dataset in self.model.datasets:
            if dataset.id in changed:
                drafts = self._assignment_drafts_by_dataset.get(dataset.id, {})
                live = {p.id for p in dataset.peaks}
                self._assignment_drafts_by_dataset[dataset.id] = {k: v for k, v in drafts.items() if k in live}
        if self.current_dataset is not None and self.current_dataset.id in changed:
            self._cancel_phase_timer()
            self.processing_session.select_dataset(self.current_dataset)
            self._reset_spectrum_interaction()
            self._update_zero_fill_choices()
            self._view_selector.configure(values=self._available_view_choices(), state="readonly")
            self._view_choice.set("Spectrum" if self.current_dataset.processed_spectrum is not None else "Raw FID")
        self._refresh(preserve_view=True)
        self._refresh_series_manager()

    def _activate_series_dataset(self, dataset_id) -> None:
        # Shared M2/M3 selection path; repeated selection leaves previews intact.
        single_selection = len(self.experiments.selected_ids) <= 1
        dataset = self.processing_session.dataset
        if dataset is None or dataset.id != dataset_id:
            self._select_dataset(dataset_id)
        if single_selection:
            self.experiments.select((dataset_id,))
        selected = self._experiment_tree.selection()
        tree_dataset_id = self._tree_dataset_ids.get(selected[0]) if selected else None
        if tree_dataset_id != dataset_id:
            self._refresh_series_datasets()
        self._selection_experiment_ids()
        self._refresh_series_manager()

    def _stacked_series_visible(self) -> bool:
        view = getattr(self, "_stacked_view", None)
        tabs = getattr(self, "_spectrum_tabs", None)
        return view is not None and tabs is not None and tabs.select() == str(view)

    def _refresh_stacked_series(self) -> None:
        if self._stacked_series_visible():
            self._stacked_view.refresh()
        window = getattr(self, "_tracking_window", None)
        if window is not None and window.winfo_exists():
            window.refresh()

        analysis = getattr(self, "_vt_analysis_window", None)
        if analysis is not None and analysis.winfo_exists():
            analysis.refresh()
        review = getattr(self, "_vt_observable_window", None)
        if review is not None and review.winfo_exists():
            review.refresh()
        pcs = getattr(self, "_vt_pcs_window", None)
        if pcs is not None and pcs.winfo_exists():
            pcs.refresh()

    def _vt_pcs_target_choices(self):
        self._refresh_structure_target_catalog()
        catalog = self._pcs_structure_target_catalog
        choices = (*catalog.single_choices, *catalog.nmr_site_choices, *catalog.matching_choices) if catalog else ()
        return choices

    def _open_vt_pcs(self):
        window = getattr(self, "_vt_pcs_window", None)
        if window is not None and window.winfo_exists():
            window.refresh()
            window.lift()
            return
        from .vt_pcs_window import VTPCSWindow
        self._vt_pcs_window = VTPCSWindow(self, self.model,
            targets=self._vt_pcs_target_choices, on_changed=self._refresh_stacked_series,
            destination=self._pcs_target_state, rebind=self._rebind_to_active_pcs_document)

    def _open_vt_analysis(self) -> None:
        window = getattr(self, "_vt_analysis_window", None)
        if window is not None and window.winfo_exists():
            window.refresh()
            window.deiconify()
            window.lift()
            return
        from .vt_analysis_window import VTAnalysisWindow
        self._vt_analysis_window = VTAnalysisWindow(self, self.model,
            on_activate=self._activate_tracking_peak, on_changed=self._refresh_series_manager,
            on_show_stack=self._show_vt_stack)

    def _open_workflow_help(self):
        from .workflow_help import show_workflow_help
        show_workflow_help(self)

    def _open_data_exchange(self):
        self._finish_assignment_edit(commit=True)
        window = getattr(self, "_data_exchange_window", None)
        if window is not None and window.winfo_exists():
            window.refresh_saved()
            window.lift()
            return
        from .data_exchange_window import DataExchangeWindow
        self._data_exchange_window = DataExchangeWindow(self, self.model,
            on_assignments=self._refresh_vt_observables, on_changed=self._refresh_stacked_series)

    def _open_vt_observables(self) -> None:
        window = getattr(self, "_vt_observable_window", None)
        if window is not None and window.winfo_exists():
            window.refresh()
            window.lift()
            return
        from .vt_observable_window import VTObservableWindow
        self._vt_observable_window = VTObservableWindow(self, self.model,
            on_activate=self._activate_tracking_peak, on_changed=self._refresh_stacked_series,
            on_committed=self._refresh_vt_observables)

    def _refresh_vt_observables(self, committed_points) -> None:
        # Reconcile only explicitly committed peaks; retain unrelated drafts.
        for dataset_id, peak_id in committed_points:
            dataset = next(d for d in self.model.datasets if d.id == dataset_id)
            drafts = self._assignment_drafts_by_dataset.get(dataset.id)
            if drafts is not None:
                drafts[peak_id] = assignments_from_observables(dataset)[peak_id]
        self._refresh_peak_list()
        self._refresh_peak_markers()

    def _show_vt_stack(self) -> None:
        self._spectrum_tabs.select(self._stacked_view)
        self.lift()

    def _open_tracking(self) -> None:
        window = getattr(self, "_tracking_window", None)
        if window is not None and window.winfo_exists():
            window.refresh()
            window.lift()
            return
        from .tracking_window import TrackingWindow
        self._tracking_window = TrackingWindow(self, self.model,
            on_activate=self._activate_tracking_peak, on_changed=self._refresh_stacked_series,
            order=lambda: self._stacked_view.order.get())
        self._refresh_stacked_series()

    def _activate_tracking_peak(self, dataset_id, peak_id) -> None:
        self._activate_series_dataset(dataset_id)
        dataset = self.model.current_dataset
        valid = dataset is not None and sum(p.id == peak_id for p in dataset.peaks) == 1
        self.model.selected_peak_id = peak_id if valid else None
        tree = self._peak_tree
        self._begin_peak_selection_projection()
        if valid and tree.exists(peak_id):
            tree.selection_set(peak_id)
            tree.see(peak_id)
        else:
            tree.selection_remove(*tree.selection())
        self._refresh_selected_peak_highlights()
        self._refresh_stacked_series()

    def _refresh_series_datasets(self) -> None:
        # Ignore generated selection events while adding imported experiment rows;
        # batch import alone must not discard a single-spectrum preview.
        self._series_tree_refresh = True
        self._rebuild_experiment_tree()
        self.after_idle(lambda: setattr(self, "_series_tree_refresh", False))

    def _refresh_series_manager(self) -> None:
        window = getattr(self, "_series_window", None)
        if window is not None and window.winfo_exists():
            window.refresh()
        else:
            self._refresh_stacked_series()

    def _build_processing(self, parent: ttk.Frame) -> None:
        fonts = get_app_fonts(self)
        style = ttk.Style(self)
        for role in ("Primary", "Secondary", "Danger"):
            style.configure(
                f"NMR.{role}.TButton", width=0,
                padding=(_ACTION_GAP, _HEADER_GAP),
                font=fonts["ui_bold"] if role == "Primary" else fonts["ui"],
            )
        style.configure(
            "NMR.Section.TButton", anchor="w", width=0,
            padding=(_CONTROL_GAP, _ROW_GAP), font=fonts["ui_bold"],
        )
        style.configure(
            "NMR.Advanced.TButton", anchor="w", width=0,
            padding=(_ACTION_GAP, _HEADER_GAP), font=fonts["ui_small_bold"],
        )
        style.configure("NMR.Reference.TLabelframe", borderwidth=0, relief="flat")
        accordion = _Accordion(parent)
        self._processing_accordion = accordion
        accordion.pack(fill="both", expand=True)

        apodization = accordion.add("1 · Apodization", initially_open=True)
        self._apod_method = tk.StringVar(value="Exponential")
        self._apod_parameter_label = tk.StringVar(value="LB (Hz)")
        self._apod_value = tk.StringVar(value="0.3")
        ttk.Label(apodization, text="Method").grid(row=0, column=0, sticky="w")
        self._apod_method_selector = ttk.Combobox(
            apodization,
            textvariable=self._apod_method,
            values=("Exponential", "Gaussian"),
            state="readonly",
            width=14,
        )
        self._apod_method_selector.grid(row=0, column=1, sticky="ew", padx=(_CONTROL_GAP, 0))
        self._apod_method_selector.bind(
            "<<ComboboxSelected>>", self._on_apodization_method
        )
        ttk.Label(apodization, textvariable=self._apod_parameter_label).grid(
            row=1, column=0, sticky="w", pady=(_ROW_GAP, 0)
        )
        self._apod_entry = ttk.Entry(
            apodization, textvariable=self._apod_value, width=12
        )
        self._apod_entry.grid(row=1, column=1, sticky="ew", padx=(_CONTROL_GAP, 0), pady=(_ROW_GAP, 0))
        apodization.columnconfigure(1, weight=1)
        apod_buttons = ttk.Frame(apodization, padding=0)
        apod_buttons.grid(row=2, column=0, columnspan=2, sticky="w", pady=(_CONTROL_GAP, 0))
        self._apod_preview_button = ttk.Button(
            apod_buttons, text="Preview", command=self._preview_apodization
        )
        self._apod_preview_button.pack(side="left")
        self._apod_apply_button = ttk.Button(
            apod_buttons, text="Apply", command=self._apply_preview
        )
        self._apod_apply_button.pack(side="left", padx=(_ACTION_GAP, 0))
        self._apod_cancel_button = ttk.Button(
            apod_buttons, text="Cancel Preview", command=self._cancel_preview
        )
        self._apod_cancel_button.pack(side="left", padx=(_ACTION_GAP, 0))

        zero_fill = accordion.add("2 · Zero Filling")
        self._zero_target = tk.StringVar(value="")
        ttk.Label(zero_fill, text="Target points").pack(anchor="w")
        self._zero_selector = ttk.Combobox(
            zero_fill, textvariable=self._zero_target, state="normal", width=18
        )
        self._zero_selector.pack(fill="x", pady=(3, 0))
        zero_buttons = ttk.Frame(zero_fill, padding=0)
        zero_buttons.pack(anchor="w", pady=(_CONTROL_GAP, 0))
        self._zero_preview_button = ttk.Button(
            zero_buttons, text="Preview", command=self._preview_zero_fill
        )
        self._zero_preview_button.pack(side="left")
        self._zero_apply_button = ttk.Button(
            zero_buttons, text="Apply", command=self._apply_preview
        )
        self._zero_apply_button.pack(side="left", padx=(_ACTION_GAP, 0))
        self._zero_cancel_button = ttk.Button(
            zero_buttons, text="Cancel Preview", command=self._cancel_preview
        )
        self._zero_cancel_button.pack(side="left", padx=(_ACTION_GAP, 0))

        fourier = accordion.add("3 · Fourier Transform")
        ttk.Label(
            fourier,
            text=(
                "Create the frequency-domain spectrum after time-domain "
                "preparation, or restart processing from the original FID."
            ),
            wraplength=220,
            justify="left",
        ).pack(anchor="w")
        self._flow_ft_button = ttk.Button(
            fourier,
            text="Create Spectrum",
            command=self._raw_processing_action,
        )
        self._flow_ft_button.pack(anchor="w", pady=(_CONTROL_GAP, 0))

        phase = accordion.add("4 · Phase")
        self._phase_syncing = False
        self._phase_after_id = None
        self._p0_text = tk.StringVar(value="0")
        self._p1_text = tk.StringVar(value="0")
        self._pivot_text = tk.StringVar(value="")
        self._pivot_label = tk.StringVar(value="Pivot (ppm)")
        self._p0_slider_value = tk.DoubleVar(value=0.0)
        self._p1_slider_value = tk.DoubleVar(value=0.0)
        self._pivot_slider_value = tk.DoubleVar(value=0.0)
        self._auto_phase_algorithm = tk.StringVar(value="ACME")
        self._phase_entries: list[ttk.Entry] = []
        self._phase_drags = []
        phase_auto = ttk.Frame(phase, padding=0)
        phase_auto.pack(fill="x")
        ttk.Label(phase_auto, text="Auto", width=6).grid(row=0, column=0, sticky="w")
        self._auto_phase_selector = ttk.Combobox(
            phase_auto,
            textvariable=self._auto_phase_algorithm,
            values=("ACME", "Peak minima"),
            state="readonly",
            width=12,
        )
        self._auto_phase_selector.grid(row=0, column=1, padx=(_CONTROL_GAP, _ACTION_GAP), sticky="w")
        self._auto_phase_button = ttk.Button(
            phase_auto, text="Run", command=self._run_auto_phase
        )
        self._auto_phase_button.grid(row=0, column=2, sticky="w")
        self._phase_pivot_pick = False
        pivot_controls = ttk.Frame(phase)
        pivot_controls.pack(fill="x", pady=(_ROW_GAP, 0))

        pivot_row = ttk.Frame(pivot_controls)
        pivot_row.pack(fill="x")
        pivot_label = ttk.Label(pivot_row, text="Pivot", width=6)
        pivot_label.grid(row=0, column=0, sticky="w")
        ToolTip(pivot_label, "Phase pivot: ppm for a ppm spectrum; sample index for an index-axis spectrum.")
        self._pivot_entry = ttk.Entry(pivot_row, textvariable=self._pivot_text, width=8)
        self._pivot_entry.grid(row=0, column=1, columnspan=2, padx=(_CONTROL_GAP, 0), sticky="ew")
        self._pivot_entry.bind("<Return>", self._on_pivot_commit)
        self._pivot_entry.bind("<FocusOut>", self._on_pivot_commit)
        self._phase_pick_button = ttk.Button(pivot_row, text="Pick…", command=self._phase_pick_pivot)
        self._phase_pick_button.grid(row=1, column=1, padx=(_CONTROL_GAP, _ACTION_GAP), pady=(_ROW_GAP, 0), sticky="ew")
        self._biggest_peak_button = ttk.Button(pivot_row, text="Strongest", command=self._phase_biggest_peak)
        self._biggest_peak_button.grid(row=1, column=2, pady=(_ROW_GAP, 0), sticky="ew")
        pivot_row.columnconfigure((1, 2), weight=1)
        ToolTip(self._phase_pick_button, "Click a pivot position on the single spectrum. Escape cancels selection.")
        ToolTip(self._biggest_peak_button, "Use the strongest peak as the phase pivot for automatic or manual adjustment.")
        # Keep the existing coordinate synchronization control, without a second
        # visible pivot slider competing with the entry/pick controls.
        self._pivot_slider = ttk.Scale(pivot_controls, variable=self._pivot_slider_value, command=self._on_pivot_slider, style="PCS.Horizontal.TScale")
        manual = ttk.Frame(phase)
        manual.pack(fill="x", pady=(_ROW_GAP, 0))
        self._phase_advanced = ttk.Frame(manual)
        self._phase_advanced_button = ttk.Button(manual, text="▸  Manual adjustment",
            command=lambda: self._toggle_subsection(self._phase_advanced, self._phase_advanced_button, "Manual adjustment"))
        self._phase_advanced_button.pack(fill="x")
        self._add_phase_row(self._phase_advanced, 0, "P0 / °", self._p0_text, "p0")
        self._add_phase_row(self._phase_advanced, 1, "P1 / °", self._p1_text, "p1")
        self._phase_advanced.columnconfigure(4, weight=1)
        phase_actions = ttk.Frame(phase)
        phase_actions.pack(fill="x", pady=(_ROW_GAP, 0))
        self._phase_apply_button = ttk.Button(phase_actions, text="Apply", command=self._apply_preview)
        self._phase_reset_button = ttk.Button(phase_actions, text="Reset", command=self._phase_reset)
        self._phase_cancel_button = ttk.Button(phase_actions, text="Cancel", command=self._cancel_preview)
        for column, button in enumerate((self._phase_apply_button, self._phase_reset_button, self._phase_cancel_button)):
            phase_actions.columnconfigure(column * 2, weight=1, uniform="phase_actions")
            if column:
                phase_actions.columnconfigure(column * 2 - 1, minsize=_ACTION_GAP)
            button.grid(row=0, column=column * 2, sticky="ew")
        ToolTip(self._phase_reset_button, "Reset P0 and P1 to zero while keeping the pivot. Apply to commit.")

        baseline = accordion.add("5 · Baseline")
        self._baseline_method = tk.StringVar(value="Bernstein")
        self._baseline_order = tk.StringVar(value="8")
        self._baseline_lambda = tk.StringVar(value="1e6")
        self._baseline_p = tk.StringVar(value="0.001")
        self._show_fitted_baseline = tk.BooleanVar(value=True)
        self._show_baseline_original = tk.BooleanVar(value=True)
        self._auto_baseline_button = ttk.Button(
            baseline, text="Auto Baseline (arPLS)", command=self._preview_auto_baseline
        )
        self._auto_baseline_button.pack(anchor="w")
        baseline_actions = ttk.Frame(baseline, padding=0)
        self._baseline_preview_button = ttk.Button(
            baseline_actions, text="Preview", command=self._preview_baseline
        )
        self._baseline_preview_button.pack(side="left")
        self._baseline_apply_button = ttk.Button(
            baseline_actions, text="Apply", command=self._apply_preview
        )
        self._baseline_apply_button.pack(side="left", padx=(_ACTION_GAP, 0))
        self._baseline_cancel_button = ttk.Button(
            baseline_actions, text="Cancel Preview", command=self._cancel_preview
        )
        self._baseline_cancel_button.pack(side="left", padx=(_ACTION_GAP, 0))
        baseline_manual = ttk.Frame(baseline)
        baseline_manual.pack(fill="x", pady=(_CONTROL_GAP, 0))
        self._baseline_advanced = ttk.Frame(baseline_manual, padding=(_CONTROL_GAP, _ROW_GAP, 0, 0))
        self._baseline_advanced_button = ttk.Button(
            baseline_manual,
            text="▸  Manual adjustment",
            command=lambda: self._toggle_subsection(
                self._baseline_advanced,
                self._baseline_advanced_button,
                "Manual adjustment",
            ),
        )
        self._baseline_advanced_button.pack(fill="x")
        ttk.Label(self._baseline_advanced, text="Method").grid(
            row=0, column=0, sticky="w"
        )
        self._baseline_method_selector = ttk.Combobox(
            self._baseline_advanced,
            textvariable=self._baseline_method,
            values=("Bernstein", "Whittaker / AsLS", "arPLS"),
            state="readonly",
            width=18,
        )
        self._baseline_method_selector.grid(
            row=0, column=1, sticky="ew", padx=(_CONTROL_GAP, 0)
        )
        self._baseline_method_selector.bind(
            "<<ComboboxSelected>>", self._on_baseline_method
        )
        self._bernstein_parameters = ttk.Frame(self._baseline_advanced, padding=0)
        ttk.Label(self._bernstein_parameters, text="Order").pack(side="left")
        self._baseline_order_entry = ttk.Entry(
            self._bernstein_parameters,
            textvariable=self._baseline_order,
            width=12,
        )
        self._baseline_order_entry.pack(side="right")
        self._asls_parameters = ttk.Frame(self._baseline_advanced, padding=0)
        ttk.Label(self._asls_parameters, text="Lambda").grid(row=0, column=0, sticky="w")
        self._baseline_lambda_entry = ttk.Entry(
            self._asls_parameters,
            textvariable=self._baseline_lambda,
            width=12,
        )
        self._baseline_lambda_entry.grid(row=0, column=1, sticky="ew", padx=(_CONTROL_GAP, 0))
        ttk.Label(self._asls_parameters, text="p").grid(
            row=1, column=0, sticky="w", pady=(_ROW_GAP, 0)
        )
        self._baseline_p_entry = ttk.Entry(
            self._asls_parameters, textvariable=self._baseline_p, width=12
        )
        self._baseline_p_entry.grid(
            row=1, column=1, sticky="ew", padx=(_CONTROL_GAP, 0), pady=(_ROW_GAP, 0)
        )
        self._asls_parameters.columnconfigure(1, weight=1)

        self._arpls_parameters = ttk.Frame(self._baseline_advanced, padding=0)
        ttk.Label(self._arpls_parameters, text="Lambda").grid(
            row=0, column=0, sticky="w"
        )
        self._baseline_arpls_lambda_entry = ttk.Entry(
            self._arpls_parameters,
            textvariable=self._baseline_lambda,
            width=12,
        )
        self._baseline_arpls_lambda_entry.grid(
            row=0, column=1, sticky="ew", padx=(_CONTROL_GAP, 0)
        )
        self._arpls_parameters.columnconfigure(1, weight=1)

        self._baseline_advanced.columnconfigure(1, weight=1)
        overlay_options = ttk.Frame(self._baseline_advanced, padding=0)
        overlay_options.grid(row=2, column=0, columnspan=2, sticky="w", pady=(_CONTROL_GAP, 0))
        self._baseline_fit_toggle = ttk.Checkbutton(
            overlay_options,
            text="Show fitted baseline",
            variable=self._show_fitted_baseline,
            command=self._refresh_baseline_overlays,
        )
        self._baseline_fit_toggle.pack(anchor="w")
        self._baseline_original_toggle = ttk.Checkbutton(
            overlay_options,
            text="Show original spectrum",
            variable=self._show_baseline_original,
            command=self._refresh_baseline_overlays,
        )
        self._baseline_original_toggle.pack(anchor="w")
        self._on_baseline_method()
        baseline_actions.pack(fill="x", pady=(_CONTROL_GAP, 0))

        reference = accordion.add("6 · Reference")
        self._reference_observed = tk.StringVar(value="")
        self._reference_status = tk.StringVar(value="")

        step1 = ttk.LabelFrame(
            reference,
            text="1. Select observed signal",
            padding=_HEADER_GAP,
            style="NMR.Reference.TLabelframe",
        )
        step1.pack(fill="x")

        observed_row = ttk.Frame(step1, padding=0)
        observed_row.pack(fill="x")

        ttk.Label(observed_row, text="Observed").pack(side="left")
        self._reference_observed_entry = ttk.Entry(
            observed_row,
            textvariable=self._reference_observed,
            state="readonly",
            width=12,
        )
        self._reference_observed_entry.pack(side="left", padx=(6, 3))
        ttk.Label(observed_row, text="ppm").pack(side="left")

        self._reference_pick_button = ttk.Button(
            observed_row,
            text="Pick…",
            command=self._begin_reference_peak_selection,
        )
        self._reference_pick_button.pack(side="left", padx=(_ACTION_GAP, 0))
        ToolTip(self._reference_pick_button, "Select an observed reference peak on the spectrum.")

        reference_feedback = ttk.Label(
            step1,
            textvariable=self._reference_status,
            wraplength=220,
        )
        # Keep actual reference feedback, without a permanent startup instruction
        # or an empty label row when no reference operation is active.
        def show_reference_feedback(*_args):
            if self._reference_status.get():
                reference_feedback.pack(anchor="w", pady=(_ROW_GAP, 0))
            else:
                reference_feedback.pack_forget()
        self._reference_status.trace_add("write", show_reference_feedback)


        step2 = ttk.LabelFrame(
            reference,
            text="2. Assign reference",
            padding=_HEADER_GAP,
            style="NMR.Reference.TLabelframe",
        )
        step2.pack(fill="x", pady=(8, 0))

        self._reference_type = tk.StringVar(value="Residual solvent")
        self._reference_preset = tk.StringVar(value="")
        self._reference_signal = tk.StringVar(value="")
        self._reference_target = tk.StringVar(value="")
        self._reference_offset = tk.StringVar(value="")

        ttk.Label(step2, text="Type").grid(row=0, column=0, sticky="w")
        self._reference_type_selector = ttk.Combobox(
            step2,
            textvariable=self._reference_type,
            values=("Residual solvent", "Reference standard", "Custom"),
            state="disabled",
            width=18,
        )
        self._reference_type_selector.grid(
            row=0, column=1, sticky="ew", padx=(_CONTROL_GAP, 0)
        )
        self._reference_type_selector.bind(
            "<<ComboboxSelected>>", self._on_reference_type_changed
        )

        self._reference_preset_label = ttk.Label(step2, text="Solvent")
        self._reference_preset_label.grid(
            row=1, column=0, sticky="w", pady=(_ROW_GAP, 0)
        )
        self._reference_preset_selector = ttk.Combobox(
            step2,
            textvariable=self._reference_preset,
            values=(),
            state="disabled",
            width=22,
        )
        self._reference_preset_selector.grid(
            row=1, column=1, sticky="ew", padx=(_CONTROL_GAP, 0), pady=(_ROW_GAP, 0)
        )
        self._reference_preset_selector.bind(
            "<<ComboboxSelected>>", self._on_reference_preset_changed
        )

        ttk.Label(step2, text="Signal").grid(
            row=2, column=0, sticky="w", pady=(_ROW_GAP, 0)
        )
        self._reference_signal_selector = ttk.Combobox(
            step2,
            textvariable=self._reference_signal,
            values=(),
            state="disabled",
            width=22,
        )
        self._reference_signal_selector.grid(
            row=2, column=1, sticky="ew", padx=(_CONTROL_GAP, 0), pady=(_ROW_GAP, 0)
        )
        self._reference_signal_selector.bind(
            "<<ComboboxSelected>>", self._on_reference_signal_changed
        )

        ttk.Label(step2, text="Target").grid(
            row=3, column=0, sticky="w", pady=(_ROW_GAP, 0)
        )
        self._reference_target_entry = ttk.Entry(
            step2,
            textvariable=self._reference_target,
            width=14,
            state="disabled",
        )
        self._reference_target_entry.grid(
            row=3, column=1, sticky="ew", padx=(_CONTROL_GAP, 0), pady=(_ROW_GAP, 0)
        )
        self._reference_target_entry.bind(
            "<Return>", self._on_reference_target_changed
        )
        self._reference_target_entry.bind(
            "<FocusOut>", self._on_reference_target_changed
        )

        ttk.Label(step2, text="Offset").grid(
            row=4, column=0, sticky="w", pady=(_ROW_GAP, 0)
        )
        self._reference_offset_entry = ttk.Entry(
            step2,
            textvariable=self._reference_offset,
            width=14,
            state="readonly",
        )
        self._reference_offset_entry.grid(
            row=4, column=1, sticky="ew", padx=(_CONTROL_GAP, 0), pady=(_ROW_GAP, 0)
        )

        reference_actions = ttk.Frame(step2, padding=0)
        reference_actions.grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(8, 0),
        )

        self._reference_preview_button = ttk.Button(
            reference_actions,
            text="Preview",
            command=self._preview_reference,
        )
        self._reference_preview_button.pack(side="left")

        self._reference_apply_button = ttk.Button(
            reference_actions,
            text="Apply",
            command=self._apply_preview,
        )
        self._reference_apply_button.pack(side="left", padx=(_ACTION_GAP, 0))

        self._reference_cancel_button = ttk.Button(
            reference_actions,
            text="Cancel Preview",
            command=self._cancel_preview,
        )
        self._reference_cancel_button.pack(side="left", padx=(_ACTION_GAP, 0))

        step2.columnconfigure(1, weight=1)

        peak_picking = accordion.add("7 · Peak Picking")
        self._peak_pick_status = tk.StringVar(value="Ready to pick peaks.")
        self._auto_peak_height_pct = tk.StringVar(value="1.0")
        self._auto_peak_prominence_pct = tk.StringVar(value="0.5")
        self._auto_peak_distance_points = tk.StringVar(value="3")

        peak_actions = ttk.Frame(peak_picking, padding=0)
        peak_actions.pack(fill="x")
        self._peak_pick_start_button = ttk.Button(
            peak_actions,
            text="Start Picking",
            command=self._begin_peak_picking,
        )
        self._peak_pick_start_button.pack(side="left")
        self._peak_pick_stop_button = ttk.Button(
            peak_actions,
            text="Stop",
            command=self._stop_peak_picking,
        )
        self._peak_pick_stop_button.pack(side="left", padx=(_ACTION_GAP, 0))
        self._auto_peak_button = ttk.Button(
            peak_actions,
            text="Auto Pick",
            command=self._auto_pick_peaks,
        )
        self._auto_peak_button.pack(side="left", padx=(8, 0))

        self._peak_pick_advanced = ttk.Frame(peak_picking, padding=(_CONTROL_GAP, _ROW_GAP, 0, 0))
        self._peak_pick_advanced_button = ttk.Button(
            peak_picking,
            text="▸  Advanced",
            command=lambda: self._toggle_subsection(
                self._peak_pick_advanced,
                self._peak_pick_advanced_button,
                "Advanced",
            ),
        )
        self._peak_pick_advanced_button.pack(fill="x", pady=(_CONTROL_GAP, 0))

        ttk.Label(self._peak_pick_advanced, text="Min height (% max)").grid(
            row=0, column=0, sticky="w"
        )
        self._auto_peak_height_entry = ttk.Entry(
            self._peak_pick_advanced,
            textvariable=self._auto_peak_height_pct,
            width=9,
        )
        self._auto_peak_height_entry.grid(row=0, column=1, sticky="ew", padx=(_CONTROL_GAP, 0))

        ttk.Label(self._peak_pick_advanced, text="Prominence (% range)").grid(
            row=1, column=0, sticky="w", pady=(_ROW_GAP, 0)
        )
        self._auto_peak_prominence_entry = ttk.Entry(
            self._peak_pick_advanced,
            textvariable=self._auto_peak_prominence_pct,
            width=9,
        )
        self._auto_peak_prominence_entry.grid(
            row=1, column=1, sticky="ew", padx=(_CONTROL_GAP, 0), pady=(_ROW_GAP, 0)
        )

        ttk.Label(self._peak_pick_advanced, text="Min distance (points)").grid(
            row=2, column=0, sticky="w", pady=(_ROW_GAP, 0)
        )
        self._auto_peak_distance_entry = ttk.Entry(
            self._peak_pick_advanced,
            textvariable=self._auto_peak_distance_points,
            width=9,
        )
        self._auto_peak_distance_entry.grid(
            row=2, column=1, sticky="ew", padx=(_CONTROL_GAP, 0), pady=(_ROW_GAP, 0)
        )

        peak_clear_actions = ttk.Frame(self._peak_pick_advanced, padding=0)
        peak_clear_actions.grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(_CONTROL_GAP, 0)
        )
        self._clear_auto_peaks_button = ttk.Button(
            peak_clear_actions,
            text="Clear Auto Peaks",
            command=self._clear_auto_peaks,
        )
        self._clear_auto_peaks_button.pack(side="left")

        ttk.Label(
            self._peak_pick_advanced,
            text="Auto Pick replaces the previous auto-picked result; manual peaks are preserved.",
            wraplength=220,
        ).grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(_ROW_GAP, 0),
        )
        self._peak_pick_advanced.columnconfigure(1, weight=1)

        ttk.Label(
            peak_picking,
            textvariable=self._peak_pick_status,
            wraplength=220,
        ).pack(anchor="w", pady=(_ROW_GAP, 0))
        self._style_processing_actions()

    def _style_processing_actions(self) -> None:
        """Style existing controls once; never recreate controls or own eligibility."""
        root = self._root()
        icons = root._icon_manager  # The toolbar has initialized the shared manager.
        size = nearest_packaged_icon_size(TOOLBAR_ICON_SIZE, tk_scale_factor(root))
        for button in (
            self._apod_apply_button, self._zero_apply_button, self._phase_apply_button,
            self._baseline_apply_button, self._reference_apply_button,
        ):
            button.configure(style="NMR.Primary.TButton", image=icons.get("apply", size=size), compound="left")
        for button in (
            self._flow_ft_button, self._auto_peak_button,
        ):
            button.configure(style="NMR.Primary.TButton")
        for button in (
            self._apod_preview_button, self._zero_preview_button, self._baseline_preview_button,
            self._reference_preview_button,
        ):
            button.configure(style="NMR.Secondary.TButton", image=icons.get("preview", size=size), compound="left")
        for button in (
            self._biggest_peak_button, self._reference_pick_button, self._auto_phase_button,
            self._auto_baseline_button, self._peak_pick_start_button,
        ):
            button.configure(style="NMR.Secondary.TButton")
        for button in (
            self._apod_cancel_button, self._zero_cancel_button, self._phase_cancel_button,
            self._baseline_cancel_button, self._reference_cancel_button,
        ):
            button.configure(style="NMR.Danger.TButton", text="Cancel", image=icons.get("cancel", size=size), compound="left")
            ToolTip(button, "Discard the current processing preview without applying it.")
        self._phase_reset_button.configure(
            style="NMR.Secondary.TButton", image=icons.get("reset", size=size), compound="left"
        )
        self._clear_auto_peaks_button.configure(style="NMR.Danger.TButton")
        self._peak_pick_stop_button.configure(style="NMR.Secondary.TButton")
        for button in (
            self._phase_advanced_button, self._baseline_advanced_button, self._peak_pick_advanced_button,
        ):
            button.configure(style="NMR.Advanced.TButton")
        ToolTip(self._auto_phase_button, "Preview automatic phase correction using the selected algorithm.")
        for button, role in (
            (self._auto_phase_button, "Secondary"), (self._biggest_peak_button, "Secondary"),
            (self._phase_pick_button, "Secondary"), (self._phase_flip_button, "Secondary"),
            (self._phase_apply_button, "Primary"), (self._phase_reset_button, "Secondary"),
            (self._phase_cancel_button, "Danger"), (self._phase_advanced_button, "Advanced"),
            (self._baseline_advanced_button, "Advanced"),
        ):
            button.configure(style=f"NMR.Phase.{role}.TButton")

    def _add_phase_row(self, parent, row, label, text_variable, kind) -> None:
        from .phase_drag import PhaseDrag
        ttk.Label(parent, text=label, width=6).grid(row=row, column=0, sticky="w", pady=1)
        entry = ttk.Entry(parent, textvariable=text_variable, width=8)
        entry.grid(row=row, column=1, padx=(_CONTROL_GAP, _ACTION_GAP), pady=1, sticky="w")
        entry.bind("<Return>", lambda _event, name=kind: self._on_phase_entry(name))
        entry.bind("<FocusOut>", lambda _event, name=kind: self._on_phase_entry(name))
        drag = PhaseDrag(parent, read_value=text_variable.get,
            change_value=lambda value: self._change_phase_angle(kind, value), report_error=self._show_processing_error)
        drag.grid(row=row, column=2, padx=(0, _ACTION_GAP), pady=1)
        if kind == "p0":
            self._phase_flip_button = ttk.Button(parent, text="Flip", command=self._phase_flip)
            self._phase_flip_button.grid(row=row, column=3, pady=1)
            ToolTip(self._phase_flip_button, "Invert phase by 180°. Press again to restore the equivalent phase. P1 and pivot stay unchanged; Apply commits.")
        self._phase_entries.append(entry)
        self._phase_drags.append(drag)

    def _build_plot(self, parent: ttk.Frame) -> None:
        # Keep axes geometry fixed.  constrained_layout dynamically changes
        # the drawable spectrum width when y-axis tick labels gain/lose digits,
        # which is visually distracting during wheel scaling.
        figure = Figure(figsize=(8, 5), constrained_layout=False)
        figure.subplots_adjust(
            left=0.105,
            right=0.985,
            bottom=0.13,
            top=0.965,
        )
        self._axes = figure.add_subplot(111)
        self._canvas = WorkbenchFigureCanvas(figure, master=parent)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        nav = NavigationToolbar2Tk(self._canvas, parent, pack_toolbar=False)
        nav.update()
        nav.pack(fill="x")
        self._canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self._full_xlim: tuple[float, float] | None = None
        self._full_ylim: tuple[float, float] | None = None

    def _on_nmr_data_changed(self) -> None:
        # Reconcile edited peak assignments through the same draft projection
        # used by Peaks; independent observations never acquire a fake peak.
        for dataset in self.model.datasets:
            drafts = self._assignment_drafts_by_dataset.setdefault(dataset.id, {})
            drafts.update(assignments_from_observables(dataset))
        self._refresh_series_datasets()
        self._refresh_peak_list()
        self._refresh_peak_markers()
        self._refresh_shift_analysis()
        self._refresh_series_manager()

    def _select_nmr_observation(self, dataset_id, observable_id) -> None:
        if observable_id and observable_id.startswith("peak:"):
            self._activate_tracking_peak(dataset_id, observable_id[5:])
            dataset = self.current_dataset
            item = observable_for_peak(dataset, observable_id[5:]) if dataset else None
            self.model.selected_observable_id = item.id if item else None
            self._publish_peak_selection()
            return
        dataset = next((d for d in self.model.datasets if d.id == dataset_id), None)
        observation = next((o for o in dataset.observables if o.id == observable_id), None) if dataset else None
        if observation is None:
            return
        if observation.source_peak_id:
            self._activate_tracking_peak(dataset_id, observation.source_peak_id)
        else:
            self._activate_series_dataset(dataset_id)
            self.model.selected_peak_id = None
            self._begin_peak_selection_projection()
            self._peak_tree.selection_remove(*self._peak_tree.selection())
            self._refresh_selected_peak_highlights()
        self.model.selected_observable_id = observation.id
        if not self._shared_selection_syncing:
            selection = self._structure_selection_for_assignment(observation.assignment or "")
            publish_workbench_selection(self._pcs_state, assignment=observation.assignment,
                dataset_id=dataset.id, peak_ids=(observation.source_peak_id,) if observation.source_peak_id else (),
                ref_ids=selection.ref_ids if selection else (), source="nmr_data_table")

    def _import_legacy_shifts(self, kind, paste=False) -> None:
        # Reuse the original Excel/CSV mapping and review/apply UI verbatim.
        from logic.table_utils import import_delta_file, import_delta_from_clipboard
        document_id = getattr(self, "bound_pcs_document_id", None)
        def validate_destination():
            if getattr(self, "bound_pcs_document_id", None) != document_id:
                raise RuntimeError("PCS destination changed after import review. Reopen the import for the intended document.")
            self._pcs_target_state()  # existing affinity rejects an inactive document
        try:
            validate_destination()
            state = self._pcs_target_state()
            callback = state.get("plot_cartesian_graph_fn")
            action = import_delta_from_clipboard if paste else import_delta_file
            action(state, kind=kind, plot_cartesian_graph_fn=callback, before_apply=validate_destination)
            self.refresh_from_pcs_state()
        except RuntimeError as exc:
            messagebox.showerror("PCS project import", str(exc), parent=self)

    def _edit_data_peak_assignment(self, dataset_id, peak_id, assignment):
        self._activate_tracking_peak(dataset_id, peak_id)
        self._set_assignment_for_peak(peak_id, assignment)
        self._on_nmr_data_changed()
        if assignment and observable_for_peak(self.current_dataset, peak_id) is None:
            raise ValueError(self._assignment_warning_text.get() or "Set the experiment nucleus before assigning this peak.")

    def _remove_data_peak(self, dataset_id, peak_id):
        self._activate_tracking_peak(dataset_id, peak_id)
        self._delete_peak_by_id(peak_id)

    def _show_shift_analysis(self):
        self._inspector_notebook.select(self._shift_tab)
        self._refresh_shift_analysis()

    def _refresh_comparison_choices(self):
        if not hasattr(self, "_comparison_para_box"):
            return
        choices = {f"{dataset_display_name(d)} · {d.id[:8]}": d.id for d in self.model.datasets}
        self._comparison_ids = choices
        for box in (self._comparison_para_box, self._comparison_dia_box):
            box.configure(values=tuple(choices))
        pair = self._current_shift_pair()
        signature = tuple(d.id if d else None for d in pair)
        if signature != getattr(self, "_comparison_pair_signature", None):
            self._comparison_pair_signature = signature
            if all(pair):
                for variable, dataset in zip((self._comparison_para, self._comparison_dia), pair):
                    variable.set(next((label for label, key in choices.items() if key == dataset.id), ""))
        for variable in (self._comparison_para, self._comparison_dia):
            if variable.get() not in choices:
                variable.set("")
        if not self._comparison_para.get() and self.current_dataset:
            active = self.current_dataset
            variable = self._comparison_dia if active.experimental_role == "diamagnetic" else self._comparison_para
            if not variable.get():
                variable.set(next((label for label, key in choices.items() if key == active.id), ""))

    def _compare_selected_datasets(self):
        from ..analysis.nmr_data import read_only
        para = self._dataset_by_id(self._comparison_ids.get(self._comparison_para.get()))
        dia = self._dataset_by_id(self._comparison_ids.get(self._comparison_dia.get()))
        if para is None or dia is None or para.id == dia.id:
            messagebox.showinfo("Shift comparison", "Choose two different observed/reference experiments.", parent=self)
            return
        if not dataset_nucleus(para) or dataset_nucleus(para) != dataset_nucleus(dia):
            messagebox.showerror("Shift comparison", "Both experiments need the same known nucleus.", parent=self)
            return
        changes = [(d, role) for d, role in ((para, "paramagnetic"), (dia, "diamagnetic")) if d.experimental_role != role]
        if any(read_only(d) for d, _ in changes):
            messagebox.showerror("Shift comparison", "PCS project experiment roles are read-only. Choose the corresponding observed/reference datasets.", parent=self)
            return
        if changes and not messagebox.askyesno("Set experiment roles", "Use these roles for comparison?\n\n" +
            "\n".join(f"{dataset_display_name(d)} → {role.title()}" for d, role in changes), parent=self):
            return
        for dataset, role in changes:
            self._clear_dataset_pair(dataset.id)
            dataset.experimental_role = role
        self._activate_series_dataset(para.id)
        self._refresh_dataset_role_controls()
        label = next((label for label, key in self._pair_display_to_id.items() if key == dia.id), None)
        if label:
            self._dataset_pair.set(label)
            self._on_dataset_pair_changed()

    def _toggle_shift_display_options(self):
        self._shift_display_expanded = not self._shift_display_expanded
        self._shift_inspector_layout_signature = None
        self._sync_shift_inspector_layout()

    def _build_inspector(self, parent: ttk.Frame) -> None:
        fonts = get_app_fonts(self)
        style = ttk.Style(self)
        style.configure("NMR.Inspector.TNotebook", padding=0)
        style.configure(
            "NMR.Inspector.TNotebook.Tab", padding=(_ACTION_GAP, _HEADER_GAP),
            font=fonts["ui_small"],
        )
        notebook = ttk.Notebook(parent, style="NMR.Inspector.TNotebook")
        notebook.pack(fill="both", expand=True)
        self._inspector_notebook = notebook
        history_tab = ttk.Frame(notebook, padding=0)
        metadata_tab = ttk.Frame(notebook, padding=0)
        peaks_tab = ttk.Frame(notebook, padding=0)
        data_tab = ttk.Frame(notebook, padding=0)
        shift_tab = ttk.Frame(notebook, padding=0)
        pcs_tab = ttk.Frame(notebook, padding=0)
        self._pcs_tab = pcs_tab
        notebook.add(history_tab, text="History")
        notebook.add(metadata_tab, text="Metadata")
        # Keep the existing peak controller projection private while presenting
        # one unified table. Peak picking/history/invalidation APIs stay intact.
        notebook.add(data_tab, text="NMR Data")
        notebook.add(shift_tab, text="Shift Analysis")
        notebook.add(pcs_tab, text="PCS Preparation")
        # A shared color accent groups analysis tabs without replacing native
        # notebook navigation or changing the spectrum notebook's appearance.
        self._inspector_analysis_tabs = (data_tab, shift_tab, pcs_tab)
        self._shift_tab = shift_tab
        from .nmr_data_panel import NMRDataPanel
        self._nmr_data_panel = NMRDataPanel(data_tab, self.model,
            on_changed=self._on_nmr_data_changed, on_select=self._select_nmr_observation,
            on_map=self._show_assignment_in_chemical_map,
            on_peak_assignment=self._edit_data_peak_assignment, on_peak_remove=self._remove_data_peak,
            on_clear_peaks=self._clear_all_peaks,
            peak_assignment_draft=lambda dataset_id, peak_id: self._assignment_drafts_by_dataset.get(dataset_id, {}).get(peak_id, ""),
            on_compare=self._show_shift_analysis,
            on_legacy=self._import_legacy_shifts if isinstance(self._pcs_state, dict) else None,
            structure_actions=(("Preview NMR site", self._select_assignment_nmr_site),
                ("Preview matching ligand sites", self._select_assignment_matching_sites),
                ("Use reviewed target in PCS Preparation", self._use_assignment_structure_in_pcs_preparation)))
        self._nmr_data_panel.pack(fill="both", expand=True)

        self._history_text = tk.Text(
            history_tab, height=6, wrap="word", relief="flat", padx=_ACTION_GAP, pady=_HEADER_GAP,
            font=fonts["report_small"]
        )
        self._history_scrollbar = ttk.Scrollbar(history_tab, command=self._history_text.yview)
        self._history_text.configure(yscrollcommand=self._history_scrollbar.set)
        self._history_scrollbar.pack(side="right", fill="y")
        self._history_text.pack(side="left", fill="both", expand=True)
        self._history_text.configure(state="disabled")

        self._metadata_tree = ttk.Treeview(
            metadata_tab, columns=("field", "value"), show="headings", height=8
        )
        self._metadata_tree.heading("field", text="Field")
        self._metadata_tree.heading("value", text="Value")
        self._metadata_tree.column("field", width=140, minwidth=140, stretch=False)
        self._metadata_tree.column("value", width=320, minwidth=320, stretch=True)
        scrollbar = ttk.Scrollbar(metadata_tab, command=self._metadata_tree.yview)
        self._metadata_tree.configure(yscrollcommand=scrollbar.set)
        self._metadata_x_scrollbar = ttk.Scrollbar(
            metadata_tab, orient="horizontal", command=self._metadata_tree.xview
        )
        self._metadata_tree.configure(xscrollcommand=self._metadata_x_scrollbar.set)
        self._metadata_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._metadata_x_scrollbar.grid(row=1, column=0, sticky="ew")
        metadata_tab.rowconfigure(0, weight=1)
        metadata_tab.columnconfigure(0, weight=1)

        peak_header = ttk.Frame(peaks_tab, padding=0)
        peak_header.pack(fill="x", pady=(0, 4))

        ttk.Label(
            peak_header,
            text="Peak List",
            font=fonts["ui_bold"],
        ).pack(side="left")

        self._peak_count_text = tk.StringVar(value="0 peaks")
        self._peak_header_count_label = ttk.Label(
            peak_header,
            textvariable=self._peak_count_text,
        )
        self._peak_header_count_label.pack(side="right")
        peak_header.bind("<Configure>", self._layout_peak_header)

        filter_row = ttk.Frame(peak_header, padding=0)
        filter_row.pack(side="left", padx=(10, 0))
        ttk.Label(filter_row, text="Show:").pack(side="left")
        self._assignment_filter_box = ttk.Combobox(
            filter_row,
            textvariable=self._assignment_filter,
            values=("All", "Assigned", "Unassigned"),
            state="readonly",
            width=11,
        )
        self._assignment_filter_box.pack(side="left", padx=(4, 0))
        self._assignment_filter_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._refresh_peak_list(),
        )
        self._assignment_warning_text = tk.StringVar(value="")
        self._assignment_warning_label = ttk.Label(peaks_tab, textvariable=self._assignment_warning_text,
                                                   style="NMR.Muted.TLabel", wraplength=260)
        self._assignment_warning_label.bind("<Configure>", self._wrap_inspector_label)
        self._assignment_warning_text.trace_add("write", self._show_assignment_warning)

        peak_table = ttk.Frame(peaks_tab, padding=0)
        peak_table.pack(fill="both", expand=True)
        self._peak_tree = ttk.Treeview(
            peak_table,
            columns=("delete", "number", "ppm", "intensity", "source", "assignment", "observable"),
            show="headings",
            selectmode="extended",
            height=8,
        )
        self._peak_tree.heading("delete", text="×")
        self._peak_tree.heading("number", text="#")
        self._peak_tree.heading(
            "ppm",
            text="ppm",
            command=lambda: self._sort_peak_table("ppm"),
        )
        self._peak_tree.heading(
            "intensity",
            text="Intensity",
            command=lambda: self._sort_peak_table("intensity"),
        )
        self._peak_tree.heading(
            "source",
            text="Source",
            command=lambda: self._sort_peak_table("source"),
        )
        self._peak_tree.heading("assignment", text="Assignment")
        self._peak_tree.heading("observable", text="Obs")

        self._peak_tree.column("delete", width=30, anchor="center", stretch=False)
        self._peak_tree.column("number", width=42, anchor="center", stretch=False)
        self._peak_tree.column("ppm", width=95, anchor="e", stretch=False)
        self._peak_tree.column("intensity", width=120, anchor="e", stretch=False)
        self._peak_tree.column("source", width=70, anchor="center", stretch=False)
        self._peak_tree.column("assignment", width=150, minwidth=150, anchor="w", stretch=False)
        self._peak_tree.column("observable", width=48, anchor="center", stretch=False)

        peak_scroll = ttk.Scrollbar(peak_table, command=self._peak_tree.yview)
        self._peak_tree.configure(yscrollcommand=peak_scroll.set)
        self._peak_x_scrollbar = ttk.Scrollbar(
            peak_table, orient="horizontal", command=self._peak_tree.xview
        )
        self._peak_tree.configure(xscrollcommand=self._on_peak_horizontal_scroll)
        self._peak_tree.grid(row=0, column=0, sticky="nsew")
        peak_scroll.grid(row=0, column=1, sticky="ns")
        self._peak_x_scrollbar.grid(row=1, column=0, sticky="ew")
        peak_table.rowconfigure(0, weight=1)
        peak_table.columnconfigure(0, weight=1)
        self._peak_empty_hint = ttk.Label(peak_table, text="No peaks yet.\nUse Auto Pick or pick a peak on the spectrum.",
                                          style="NMR.Muted.TLabel", justify="center", wraplength=260)
        ToolTip(self._peak_tree, "Double-click Assignment or press Enter to edit. Tab / Shift+Tab: next / previous. Delete: remove. Right-click: structure.")

        self._peak_tree.bind("<Button-1>", self._on_peak_tree_click, add="+")
        self._peak_tree.bind("<Configure>", self._fit_peak_assignment_column, add="+")
        self._peak_tree.bind("<Double-1>", self._on_peak_tree_double_click)
        self._peak_tree.bind("<<TreeviewSelect>>", self._on_peak_tree_selection)
        self._peak_tree.bind("<Delete>", self._on_peak_delete_key)
        self._peak_tree.bind("<BackSpace>", self._on_peak_delete_key)
        self._peak_tree.bind("<Control-a>", self._on_peak_select_all)
        self._peak_tree.bind("<Control-A>", self._on_peak_select_all)
        self._peak_tree.bind("<Return>", self._on_peak_assignment_enter)
        self._peak_tree.bind("<KP_Enter>", self._on_peak_assignment_enter)
        self._peak_tree.bind(
            "<Button-3>",
            lambda event: self._show_structure_context_menu(event, source_tree="peak"),
            add="+",
        )

        self._peak_sort_column = "ppm"
        self._peak_sort_reverse = False

        peak_buttons = ttk.Frame(peaks_tab, padding=(0, _ROW_GAP, 0, 0))
        peak_buttons.pack(fill="x")
        self._delete_peak_button = ttk.Button(
            peak_buttons,
            text="Delete Selected",
            command=self._delete_selected_peaks,
        )
        self._delete_peak_button.pack(side="left")
        self._clear_peaks_button = ttk.Button(
            peak_buttons,
            text="Clear All",
            command=self._clear_all_peaks,
        )
        self._clear_peaks_button.pack(side="left", padx=(4, 0))
        ttk.Label(peaks_tab, text="Double-click Assignment to edit · Enter / Tab: edit / next · Delete: remove",
                  style="NMR.Muted.TLabel", wraplength=320).pack(fill="x", pady=(0, 3))

        shift_header = ttk.Frame(shift_tab, padding=0)
        shift_header.pack(fill="x", padx=6, pady=(6, 4))
        ttk.Label(
            shift_header,
            text="Shift Analysis",
            font=fonts["ui_bold"],
        ).pack(side="left")
        self._shift_analysis_status = tk.StringVar(value="No dataset pair selected")
        ttk.Label(
            shift_header,
            textvariable=self._shift_analysis_status,
        ).pack(side="right")

        self._shift_pair_text = tk.StringVar(
            value="Choose observed and reference experiments below, then Compare."
        )
        shift_pair_label = ttk.Label(
            shift_tab,
            textvariable=self._shift_pair_text,
            wraplength=260,
            justify="left",
        )
        shift_pair_label.pack(fill="x", padx=6, pady=(0, 5))
        shift_pair_label.bind("<Configure>", self._wrap_inspector_label)

        pair_controls = ttk.Frame(shift_tab)
        pair_controls.pack(fill="x", padx=6, pady=(0, 5))
        self._comparison_para = tk.StringVar()
        self._comparison_dia = tk.StringVar()
        ttk.Label(pair_controls, text="Observed / Para").grid(row=0, column=0, sticky="w")
        ttk.Label(pair_controls, text="Reference / Dia").grid(row=1, column=0, sticky="w")
        self._comparison_para_box = ttk.Combobox(pair_controls, textvariable=self._comparison_para, state="readonly", width=21)
        self._comparison_dia_box = ttk.Combobox(pair_controls, textvariable=self._comparison_dia, state="readonly", width=21)
        self._comparison_para_box.grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        self._comparison_dia_box.grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(pair_controls, text="Compare", command=self._compare_selected_datasets,
                   style="NMR.Secondary.TButton").grid(row=1, column=2, sticky="e")
        pair_controls.columnconfigure(1, weight=1)

        # View controls replace the old separate Shift Viewer workflow.
        shift_viewbar = ttk.Frame(shift_tab, padding=0)
        shift_viewbar.pack(fill="x", padx=6, pady=(0, 5))

        ttk.Label(shift_viewbar, text="View").grid(row=0, column=0, sticky="w")
        self._shift_view_mode = tk.StringVar(value="Table")
        self._shift_view_selector = ttk.Combobox(
            shift_viewbar,
            textvariable=self._shift_view_mode,
            values=("Table", "Overlay", "Stacked", "Shift Diagnostics"),
            state="readonly",
            width=18,
        )
        self._shift_view_selector.grid(row=0, column=1, sticky="w", padx=(_ACTION_GAP, 0))
        self._shift_view_selector.bind(
            "<<ComboboxSelected>>",
            self._on_shift_view_changed,
        )
        self._shift_display_expanded = False
        self._shift_display_button = ttk.Button(shift_viewbar, text="Display ▾", command=self._toggle_shift_display_options,
                                                style="NMR.Secondary.TButton")
        self._shift_display_button.grid(row=0, column=2, sticky="e", padx=6)
        self._shift_display_options = ttk.Frame(shift_viewbar)
        self._shift_display_options.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self._shift_normalize = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self._shift_display_options,
            text="Normalize",
            variable=self._shift_normalize,
            command=self._refresh_shift_visualization,
        ).grid(row=0, column=0, sticky="w")

        self._shift_show_labels = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self._shift_display_options,
            text="Labels",
            variable=self._shift_show_labels,
            command=self._refresh_shift_visualization,
        ).grid(row=0, column=1, sticky="w", padx=4)

        self._shift_broaden_sticks = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self._shift_display_options,
            text="Broaden sticks",
            variable=self._shift_broaden_sticks,
            command=self._refresh_shift_visualization,
        ).grid(row=1, column=0, sticky="w", pady=4)

        self._shift_broadening_ppm = tk.StringVar(value="0.03")
        self._shift_broadening_entry = ttk.Entry(
            self._shift_display_options,
            textvariable=self._shift_broadening_ppm,
            width=6,
        )
        self._shift_broadening_entry.grid(row=1, column=1, sticky="w", pady=4)
        self._shift_broadening_entry.bind(
            "<Return>",
            lambda _event: self._refresh_shift_visualization(),
        )
        ttk.Label(self._shift_display_options, text="ppm FWHM").grid(row=1, column=2, sticky="w", padx=4)

        layers = ttk.Frame(shift_tab, padding=0)
        self._shift_spectrum_options = layers
        layers.pack(fill="x", padx=6, pady=(0, 2))
        ttk.Label(layers, text="Spectrum layers").grid(row=0, column=0, columnspan=2, sticky="w")

        self._shift_show_paramag = tk.BooleanVar(value=True)
        self._shift_show_diamag = tk.BooleanVar(value=True)
        self._shift_show_predicted_paramag = tk.BooleanVar(value=True)
        self._shift_show_delta_para = tk.BooleanVar(value=True)
        self._shift_show_pcs_calc = tk.BooleanVar(value=True)
        self._shift_show_fixed_correction = tk.BooleanVar(value=True)
        self._shift_show_non_pcs = tk.BooleanVar(value=True)

        for index, (label, variable) in enumerate((
            ("Obs paramag", self._shift_show_paramag),
            ("Obs diamag", self._shift_show_diamag),
            ("Predicted paramag", self._shift_show_predicted_paramag),
        )):
            ttk.Checkbutton(
                layers,
                text=label,
                variable=variable,
                command=self._refresh_shift_visualization,
            ).grid(row=1 + index // 2, column=index % 2, sticky="w", padx=(0, _ACTION_GAP), pady=(_HEADER_GAP, 0))

        diagnostic_layers = ttk.Frame(shift_tab, padding=0)
        self._shift_diagnostic_options = diagnostic_layers
        diagnostic_layers.pack(fill="x", padx=6, pady=(0, 5))
        ttk.Label(diagnostic_layers, text="Diagnostic layers").grid(row=0, column=0, columnspan=2, sticky="w")
        for index, (label, variable) in enumerate((
            ("δpara(obs)", self._shift_show_delta_para),
            ("PCS(calc)", self._shift_show_pcs_calc),
            ("Fixed corr.", self._shift_show_fixed_correction),
            ("Non-PCS", self._shift_show_non_pcs),
        )):
            ttk.Checkbutton(
                diagnostic_layers,
                text=label,
                variable=variable,
                command=self._refresh_shift_visualization,
            ).grid(row=1 + index // 2, column=index % 2, sticky="w", padx=(0, _ACTION_GAP), pady=(_HEADER_GAP, 0))

        self._shift_content = ttk.Frame(shift_tab, padding=0)
        self._shift_content.pack(fill="both", expand=True, padx=6)
        self._shift_empty_hint = ttk.Label(self._shift_content, text="Choose observed and reference experiments\nabove, then Compare.",
                                           style="NMR.Muted.TLabel", justify="center", anchor="center", wraplength=260)

        # ---- Table view -------------------------------------------------
        self._shift_table_frame = ttk.Frame(self._shift_content, padding=0)
        self._shift_table_frame.pack(fill="both", expand=True)

        self._shift_tree = ttk.Treeview(
            self._shift_table_frame,
            columns=(
                "assignment",
                "para",
                "dia",
                "delta_para",
                "pcs_calc",
                "fixed_correction",
                "non_pcs",
                "status",
            ),
            show="headings",
            selectmode="browse",
            height=8,
        )
        for column, label, width, anchor_name in (
            ("assignment", "Assignment", 110, "w"),
            ("para", "Paramag observed", 108, "e"),
            ("dia", "Diamag observed", 108, "e"),
            ("delta_para", "δpara(obs)", 88, "e"),
            ("pcs_calc", "PCS(calc)", 82, "e"),
            ("fixed_correction", "Fixed corr.", 82, "e"),
            ("non_pcs", "Implied non-PCS", 108, "e"),
            ("status", "Status", 120, "w"),
        ):
            self._shift_tree.heading(column, text=label)
            self._shift_tree.column(
                column,
                width=width,
                anchor=anchor_name,
                stretch=column in {"assignment", "status"},
            )

        shift_y_scroll = ttk.Scrollbar(
            self._shift_table_frame,
            command=self._shift_tree.yview,
        )
        shift_x_scroll = ttk.Scrollbar(
            self._shift_table_frame,
            orient="horizontal",
            command=self._shift_tree.xview,
        )
        self._shift_tree.configure(
            yscrollcommand=shift_y_scroll.set,
            xscrollcommand=shift_x_scroll.set,
        )
        self._shift_tree.grid(row=0, column=0, sticky="nsew")
        shift_y_scroll.grid(row=0, column=1, sticky="ns")
        shift_x_scroll.grid(row=1, column=0, sticky="ew")
        self._shift_table_frame.rowconfigure(0, weight=1)
        self._shift_table_frame.columnconfigure(0, weight=1)
        self._shift_tree.bind(
            "<<TreeviewSelect>>",
            self._on_shift_row_selected,
        )
        self._shift_tree.bind(
            "<Button-3>",
            lambda event: self._show_structure_context_menu(event, source_tree="shift"),
            add="+",
        )

        # ---- Visualization view ----------------------------------------
        self._shift_plot_frame = ttk.Frame(self._shift_content, padding=0)

        shift_figure = Figure(figsize=(5.3, 3.2), dpi=100)
        self._shift_axes = shift_figure.add_subplot(111)
        self._shift_canvas = WorkbenchFigureCanvas(
            shift_figure,
            master=self._shift_plot_frame,
        )
        self._shift_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._shift_plot_candidates = []
        self._shift_hover_annotation = None
        self._shift_hover_assignment = None
        self._shift_canvas.mpl_connect(
            "motion_notify_event",
            self._on_shift_plot_motion,
        )
        self._shift_canvas.mpl_connect(
            "button_press_event",
            self._on_shift_plot_click,
        )
        self._shift_canvas.mpl_connect(
            "figure_leave_event",
            self._on_shift_plot_leave,
        )

        shift_plot_footer = ttk.Frame(self._shift_plot_frame, padding=0)
        shift_plot_footer.pack(fill="x", pady=(3, 0))
        self._shift_plot_note = tk.StringVar(
            value=(
                "PCS(calc) is a calculated difference-space quantity, not "
                "PCS(exp) or an absolute chemical shift."
            )
        )
        ttk.Label(
            shift_plot_footer,
            textvariable=self._shift_plot_note,
            wraplength=260,
            justify="left",
        ).pack(side="left")

        shift_actions = ttk.Frame(shift_tab, padding=0)
        shift_actions.pack(fill="x", padx=6, pady=(7, 6))
        ttk.Button(
            shift_actions,
            text="Refresh",
            command=self._refresh_shift_analysis,
        ).pack(side="left")
        ToolTip(self._shift_tree, "δpara = δparamagnetic − δdiamagnetic (ppm). Right-click a row to inspect structure.")
        self._pcs_prepare_button = ttk.Button(
            shift_actions,
            text="Prepare PCS →",
            state="disabled",
            command=self._open_pcs_preparation,
        )
        self._pcs_prepare_button.pack(side="right")

        self._shift_diagnostic_rows = []
        self._shift_analysis_status.trace_add("write", lambda *args: self._sync_shift_inspector_layout())
        self._refresh_shift_analysis()

        pcs_header = ttk.Frame(pcs_tab, padding=0)
        pcs_header.pack(fill="x", padx=6, pady=(6, 4))
        ttk.Label(
            pcs_header,
            text="PCS Preparation",
            font=fonts["ui_bold"],
        ).pack(side="left")
        self._pcs_prep_status = tk.StringVar(value="No valid shift pair")
        ttk.Label(
            pcs_header,
            textvariable=self._pcs_prep_status,
        ).pack(side="right")

        self._pcs_source_text = tk.StringVar(
            value="Pair paramagnetic and diamagnetic datasets in Shift Analysis first."
        )
        pcs_source_label = ttk.Label(
            pcs_tab,
            textvariable=self._pcs_source_text,
            wraplength=260,
            justify="left",
        )
        pcs_source_label.pack(fill="x", padx=6, pady=(0, 6))
        pcs_source_label.bind("<Configure>", self._wrap_inspector_label)

        method_frame = ttk.LabelFrame(pcs_tab, text="Derivation method", padding=_PANEL_PADDING)
        method_frame.pack(fill="x", padx=6, pady=(0, 6))

        method_row = ttk.Frame(method_frame, padding=0)
        method_row.pack(fill="x")
        ttk.Label(method_row, text="Method").pack(side="left")
        self._pcs_method = tk.StringVar(value="Choose derivation method…")
        self._pcs_method_box = ttk.Combobox(
            method_row,
            textvariable=self._pcs_method,
            values=("Choose derivation method…", *tuple(self._pcs_providers)),
            state="readonly",
            width=20,
        )
        self._pcs_method_box.pack(side="left", padx=(_CONTROL_GAP, 0))
        self._pcs_method_box.bind(
            "<<ComboboxSelected>>",
            self._on_pcs_method_changed,
        )

        self._pcs_method_description = tk.StringVar(value="")
        method_description_label = ttk.Label(
            method_frame,
            textvariable=self._pcs_method_description,
            wraplength=260,
            justify="left",
        )
        method_description_label.pack(fill="x", pady=(5, 0))
        method_description_label.bind("<Configure>", self._wrap_inspector_label)

        self._pcs_detail_panes = WorkbenchPanedWindow(pcs_tab, orient="vertical")
        self._pcs_detail_panes.pack(fill="both", expand=True, padx=6)
        prep_table_frame = ttk.Frame(self._pcs_detail_panes, padding=0)
        self._pcs_detail_panes.add(prep_table_frame, weight=1)
        self._pcs_tree = ttk.Treeview(
            prep_table_frame,
            columns=(
                "use",
                "assignment",
                "delta",
                "correction",
                "pcs",
                "uncertainty",
                "target",
                "scope",
                "status",
            ),
            show="headings",
            selectmode="browse",
            height=8,
        )
        for column, label, width, anchor_name in (
            ("use", "Use", 44, "center"),
            ("assignment", "Assignment", 100, "w"),
            ("delta", "Δshift", 82, "e"),
            ("correction", "Correction", 86, "e"),
            ("pcs", "PCS", 82, "e"),
            ("uncertainty", "σ", 66, "e"),
            ("target", "Target", 150, "w"),
            ("scope", "Target scope", 120, "w"),
            ("status", "Status", 125, "w"),
        ):
            self._pcs_tree.heading(column, text=label)
            self._pcs_tree.column(
                column,
                width=width,
                anchor=anchor_name,
                stretch=column in {"assignment", "status"},
            )
        prep_scroll = ttk.Scrollbar(
            prep_table_frame,
            command=self._pcs_tree.yview,
        )
        self._pcs_tree.configure(yscrollcommand=prep_scroll.set)
        self._pcs_x_scrollbar = ttk.Scrollbar(
            prep_table_frame, orient="horizontal", command=self._pcs_tree.xview
        )
        self._pcs_tree.configure(xscrollcommand=self._pcs_x_scrollbar.set)
        self._pcs_tree.grid(row=0, column=0, sticky="nsew")
        prep_scroll.grid(row=0, column=1, sticky="ns")
        self._pcs_x_scrollbar.grid(row=1, column=0, sticky="ew")
        prep_table_frame.rowconfigure(0, weight=1)
        prep_table_frame.columnconfigure(0, weight=1)
        self._pcs_tree.bind("<<TreeviewSelect>>", self._on_pcs_row_selected)
        self._pcs_tree.bind("<Double-1>", self._toggle_pcs_use_from_click)

        self._pcs_detail_host = ttk.Frame(self._pcs_detail_panes, padding=(0, 4, 0, 0))
        self._pcs_detail_panes.add(self._pcs_detail_host, weight=0)
        self._pcs_detail_expanded = False
        self._pcs_detail_assignment = None
        self._pcs_detail_hint = ttk.Button(self._pcs_detail_host, text="Selected assignment — select a row",
            style="NMR.Secondary.TButton", command=self._toggle_pcs_detail)
        self._pcs_detail_hint.pack(fill="x", pady=(0, 4))
        settings = self._pcs_detail_form = ttk.LabelFrame(
            self._pcs_detail_host,
            text="Selected assignment",
            padding=_PANEL_PADDING,
        )
        settings.pack(fill="x")
        settings.columnconfigure(3, weight=1)

        self._pcs_selected_assignment = tk.StringVar(value="—")
        ttk.Label(settings, text="Assignment").grid(row=0, column=0, sticky="w")
        ttk.Label(
            settings,
            textvariable=self._pcs_selected_assignment,
        ).grid(row=0, column=1, sticky="w", padx=(6, 0))

        ttk.Label(settings, text="Correction / FCS").grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )
        self._pcs_correction = tk.StringVar(value="")
        correction_row = ttk.Frame(settings)
        correction_row.grid(row=1, column=1, sticky="w", padx=_CONTROL_GAP, pady=(_ROW_GAP, 0))
        self._pcs_correction_entry = ttk.Entry(
            correction_row,
            textvariable=self._pcs_correction,
            width=14,
            state="disabled",
        )
        self._pcs_correction_entry.pack(side="left")

        ttk.Label(settings, text="Manual PCS").grid(
            row=2, column=0, sticky="w", pady=(4, 0)
        )
        self._pcs_manual_value = tk.StringVar(value="")
        manual_row = ttk.Frame(settings)
        manual_row.grid(row=2, column=1, sticky="w", padx=_CONTROL_GAP, pady=(_ROW_GAP, 0))
        self._pcs_manual_entry = ttk.Entry(
            manual_row,
            textvariable=self._pcs_manual_value,
            width=14,
            state="disabled",
        )
        self._pcs_manual_entry.pack(side="left")

        ttk.Label(settings, text="Target scope").grid(
            row=3, column=0, sticky="w", pady=(4, 0)
        )
        self._pcs_target_scope = tk.StringVar(value=SCOPE_SINGLE)
        self._pcs_target_scope_box = ttk.Combobox(
            settings,
            textvariable=self._pcs_target_scope,
            values=TARGET_SCOPES,
            state="disabled",
            width=20,
        )
        self._pcs_target_scope_box.grid(
            row=3, column=1, sticky="w", padx=(6, 0), pady=(4, 0)
        )
        self._pcs_target_scope_box.bind(
            "<<ComboboxSelected>>",
            self._on_pcs_target_scope_changed,
        )

        ttk.Label(settings, text="PCS target").grid(
            row=4, column=0, sticky="w", pady=(4, 0)
        )
        self._pcs_target = tk.StringVar(value="")
        self._pcs_target_box = ttk.Combobox(
            settings,
            textvariable=self._pcs_target,
            values=(),
            state="disabled",
            width=24,
        )
        self._pcs_target_box.grid(
            row=4, column=1, sticky="w", padx=(6, 0), pady=(4, 0)
        )
        self._pcs_target_box.bind(
            "<<ComboboxSelected>>",
            self._on_pcs_target_changed,
        )

        ttk.Label(settings, text="Uncertainty σ").grid(
            row=5, column=0, sticky="w", pady=(4, 0)
        )
        self._pcs_uncertainty = tk.StringVar(value="")
        uncertainty_row = ttk.Frame(settings)
        uncertainty_row.grid(row=5, column=1, sticky="w", padx=_CONTROL_GAP, pady=(_ROW_GAP, 0))
        self._pcs_uncertainty_entry = ttk.Entry(
            uncertainty_row,
            textvariable=self._pcs_uncertainty,
            width=14,
            state="disabled",
        )
        self._pcs_uncertainty_entry.pack(side="left")
        for row in (correction_row, manual_row, uncertainty_row):
            ttk.Label(row, text="ppm", style="NMR.Muted.TLabel").pack(side="left", padx=4)

        self._pcs_selected_status = tk.StringVar(value="")
        selected_status_label = ttk.Label(
            settings,
            textvariable=self._pcs_selected_status,
            wraplength=260,
            justify="left",
        )
        selected_status_label.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        selected_status_label.bind("<Configure>", self._wrap_inspector_label)

        self._pcs_update_button = ttk.Button(
            settings,
            text="Update selected",
            style="NMR.Secondary.TButton",
            command=self._update_selected_pcs_inputs,
            state="disabled",
        )
        self._pcs_update_button.grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        prep_actions = ttk.Frame(pcs_tab, padding=0)
        self._pcs_actions = prep_actions
        prep_actions.pack(fill="x", padx=6, pady=(8, 6))
        ttk.Button(
            prep_actions,
            text="Refresh",
            command=self._refresh_pcs_preparation,
        ).pack(side="left")
        self._pcs_review_apply_button = ttk.Button(
            prep_actions,
            text="Review & Apply…",
            style="NMR.Primary.TButton",
            command=self._review_and_apply_pcs,
            state="disabled",
        )
        self._pcs_review_apply_button.pack(side="right")

        self._update_pcs_method_description()
        self._refresh_pcs_preparation()

    def _wrap_inspector_label(self, event) -> None:
        width = max(120, event.width - 4)
        if int(event.widget.cget("wraplength")) != width:
            event.widget.configure(wraplength=width)

    def _layout_peak_header(self, event) -> None:
        compact = event.width < round(460 * tk_scale_factor(self._root()))
        if compact != getattr(self, "_peak_header_compact", None):
            self._peak_header_compact = compact
            self._peak_header_count_label.pack_configure(side="bottom" if compact else "right",
                                                         anchor="w", pady=(3, 0) if compact else 0)

    def _fit_peak_assignment_column(self, event) -> None:
        tree = self._peak_tree
        fixed_width = sum(tree.column(column, "width") for column in tree["columns"] if column != "assignment")
        width = min(320, max(150, event.width - fixed_width - 4))
        if tree.column("assignment", "width") != width:
            tree.column("assignment", width=width)

    def _show_assignment_warning(self, *args) -> None:
        label = self._assignment_warning_label
        if self._assignment_warning_text.get():
            if not label.winfo_manager():
                label.pack(fill="x", pady=(0, 4), before=self._peak_tree.master)
        elif label.winfo_manager():
            label.pack_forget()

    def _sync_shift_inspector_layout(self) -> None:
        mode = self._shift_view_mode.get()
        pair = self._current_shift_pair()
        valid = all(dataset is not None for dataset in pair) and self._shift_analysis_status.get() != "Pair error"
        expanded = self._shift_display_expanded
        self._shift_display_button.configure(state="disabled" if mode == "Table" else "normal",
                                             text="Display ▴" if expanded else "Display ▾")
        signature = (mode, valid, expanded)
        if signature == getattr(self, "_shift_inspector_layout_signature", None):
            return
        self._shift_inspector_layout_signature = signature
        for frame in (self._shift_spectrum_options, self._shift_diagnostic_options):
            if frame.winfo_manager():
                frame.pack_forget()
        if mode != "Table" and expanded:
            self._shift_display_options.grid()
            options = self._shift_diagnostic_options if mode == "Shift Diagnostics" else self._shift_spectrum_options
            options.pack(fill="x", padx=6, pady=(0, 4), before=self._shift_content)
            for group in (self._shift_display_options, options):
                for widget in group.winfo_children():
                    if isinstance(widget, (ttk.Checkbutton, ttk.Entry)):
                        widget.state(["!disabled" if valid else "disabled"])
        else:
            self._shift_display_options.grid_remove()
        desired = self._shift_table_frame if mode == "Table" else (self._shift_plot_frame if valid else self._shift_empty_hint)
        self._shift_empty_hint.place_forget()
        for frame in (self._shift_empty_hint, self._shift_table_frame, self._shift_plot_frame):
            if frame is not desired and frame.winfo_manager():
                frame.pack_forget()
        if not desired.winfo_manager():
            desired.pack(fill="both", expand=True)
        if not valid and mode == "Table":
            self._shift_empty_hint.place(in_=self._shift_table_frame, relx=.5, rely=.5, anchor="center")

    def _on_peak_horizontal_scroll(self, first, last) -> None:
        """Track horizontal overflow without changing an active assignment draft."""
        self._peak_x_scrollbar.set(first, last)
        editor = self._assignment_editor
        peak_id = self._assignment_editor_peak_id
        if editor is not None and peak_id is not None:
            bbox = self._peak_tree.bbox(peak_id, "assignment")
            if bbox:
                x, y, width, height = bbox
                editor.place_configure(x=x, y=y, width=max(width, 80), height=height)

    def _current_shift_rows(self):
        paramagnetic, diamagnetic = self._current_shift_pair()
        if paramagnetic is None or diamagnetic is None:
            return None, None, [], None
        try:
            rows, summary = build_shift_comparison(paramagnetic, diamagnetic)
        except ShiftAnalysisError:
            return paramagnetic, diamagnetic, [], None
        return paramagnetic, diamagnetic, rows, summary

    def _pair_temperature_K(self, paramagnetic, diamagnetic) -> float | None:
        for dataset in (paramagnetic, diamagnetic):
            resolved = resolve_temperature(getattr(dataset, "conditions", None))
            if resolved.status == "invalid":
                return None
            if resolved.temperature_k is not None:
                return resolved.temperature_k
        return None

    def _pcs_context(self, paramagnetic, diamagnetic) -> PCSDerivationContext:
        return PCSDerivationContext(
            paramagnetic_dataset_id=paramagnetic.id,
            diamagnetic_dataset_id=diamagnetic.id,
            temperature_K=self._pair_temperature_K(paramagnetic, diamagnetic),
            corrections_ppm=dict(self._pcs_corrections_ppm),
            manual_pcs_ppm=dict(self._pcs_manual_values_ppm),
            uncertainties_ppm=dict(self._pcs_uncertainties_ppm),
        )

    def _pcs_targets(self):
        return extract_pcs_targets(self._pcs_state)

    def _pcs_structure_ref_ids_and_labels(self) -> tuple[list[int], dict[int, str]]:
        state = self._pcs_state if isinstance(self._pcs_state, dict) else {}
        atom_data = state.get("atom_data_raw") or state.get("atom_data") or []
        atom_ids = state.get("atom_ids_raw") or state.get("atom_ids") or []
        overrides = state.get("ref_label_overrides", {}) or {}

        if atom_data and not atom_ids:
            atom_ids = list(range(1, len(atom_data) + 1))

        ids = [int(value) for value in atom_ids]
        labels: dict[int, str] = {}
        for index, ref_id in enumerate(ids):
            element = ""
            if index < len(atom_data):
                try:
                    element = str(atom_data[index][0])
                except Exception:
                    element = ""
            labels[ref_id] = str(overrides.get(ref_id, element) or "")

        if not ids:
            targets = self._pcs_targets()
            ids = [target.ref_id for target in targets]
            labels = {target.ref_id: target.label for target in targets}
        return ids, labels

    def _refresh_structure_target_catalog(self) -> None:
        ref_ids, labels = self._pcs_structure_ref_ids_and_labels()
        self._pcs_structure_target_catalog = build_structure_target_catalog_from_state(
            self._pcs_state,
            allowed_ref_ids=ref_ids,
            labels_by_ref=labels,
        )
        graph = self._pcs_state.get("structure_graph") if isinstance(self._pcs_state, dict) else None
        self._pcs_structure_target_graph_identity = id(graph) if graph is not None else None

    def _pcs_target_scope_for_assignment(self, assignment: str) -> str:
        scope = self._pcs_target_scope_by_assignment.get(assignment, SCOPE_SINGLE)
        return scope if scope in TARGET_SCOPES else SCOPE_SINGLE

    def _refresh_pcs_target_choices(self) -> None:
        assignment = self._pcs_selected_assignment_name()
        scope = (
            self._pcs_target_scope_for_assignment(assignment)
            if assignment
            else SCOPE_SINGLE
        )
        catalog = self._pcs_structure_target_catalog
        if catalog is None:
            self._refresh_structure_target_catalog()
            catalog = self._pcs_structure_target_catalog

        self._pcs_target_display_to_key = {}
        self._pcs_target_display_to_id = {}
        if catalog is not None and scope != SCOPE_CUSTOM:
            choices = catalog.choices_for_scope(scope)
            self._pcs_target_display_to_key = {
                choice.display: choice.key for choice in choices
            }
            self._pcs_target_display_to_id = {
                choice.display: int(choice.primary_ref_id)
                for choice in choices
                if choice.primary_ref_id is not None
            }

        if hasattr(self, "_pcs_target_box"):
            values = ("", *tuple(self._pcs_target_display_to_key))
            self._pcs_target_box.configure(values=values)

    def _pcs_target_choice_for_assignment(self, assignment: str):
        scope = self._pcs_target_scope_for_assignment(assignment)
        catalog = self._pcs_structure_target_catalog
        if catalog is None:
            self._refresh_structure_target_catalog()
            catalog = self._pcs_structure_target_catalog

        if scope == SCOPE_CUSTOM:
            text = self._pcs_custom_target_text_by_assignment.get(assignment, "")
            if not text:
                return None
            try:
                ref_ids, labels = self._pcs_structure_ref_ids_and_labels()
                return parse_custom_ref_choice(
                    text,
                    allowed_ref_ids=ref_ids,
                    labels_by_ref=labels,
                )
            except PCSStructureTargetError:
                return None

        if catalog is None:
            return None

        key = self._pcs_target_choice_key_by_assignment.get(assignment)
        choice = catalog.choice_by_key(key)
        if choice is not None and choice.scope == scope:
            return choice

        targets = self._pcs_targets()
        ref_id = self._pcs_target_by_assignment.get(assignment)
        if ref_id is None:
            ref_id = suggest_target_ref_id(assignment, targets)
            if ref_id is not None:
                self._pcs_target_by_assignment[assignment] = int(ref_id)

        choice = catalog.choice_for_ref(scope, ref_id)
        if choice is not None:
            self._pcs_target_choice_key_by_assignment[assignment] = choice.key
            if choice.primary_ref_id is not None:
                self._pcs_target_by_assignment[assignment] = int(choice.primary_ref_id)
        return choice

    def _pcs_target_display_for_assignment(self, assignment: str) -> str:
        scope = self._pcs_target_scope_for_assignment(assignment)
        if scope == SCOPE_CUSTOM:
            return self._pcs_custom_target_text_by_assignment.get(assignment, "")
        choice = self._pcs_target_choice_for_assignment(assignment)
        return "" if choice is None else choice.display

    def _on_pcs_target_scope_changed(self, _event=None) -> None:
        assignment = self._pcs_selected_assignment_name()
        if not assignment:
            return
        scope = self._pcs_target_scope.get().strip()
        if scope not in TARGET_SCOPES:
            scope = SCOPE_SINGLE
        self._pcs_target_scope_by_assignment[assignment] = scope
        self._pcs_target_choice_key_by_assignment.pop(assignment, None)
        self._refresh_shift_analysis()
        self._refresh_pcs_preparation(preserve_assignment=assignment)

    def _on_pcs_target_changed(self, _event=None) -> None:
        assignment = self._pcs_selected_assignment_name()
        if not assignment:
            return
        scope = self._pcs_target_scope_for_assignment(assignment)
        display = self._pcs_target.get().strip()

        if scope == SCOPE_CUSTOM:
            self._pcs_custom_target_text_by_assignment[assignment] = display
            if display:
                try:
                    ref_ids, labels = self._pcs_structure_ref_ids_and_labels()
                    choice = parse_custom_ref_choice(
                        display,
                        allowed_ref_ids=ref_ids,
                        labels_by_ref=labels,
                    )
                except PCSStructureTargetError as exc:
                    self._pcs_selected_status.set(str(exc))
                    return
                if choice.primary_ref_id is not None:
                    self._pcs_target_by_assignment[assignment] = int(choice.primary_ref_id)
            self._refresh_shift_analysis()
            self._refresh_pcs_preparation(preserve_assignment=assignment)
            return

        if not display:
            self._pcs_target_choice_key_by_assignment.pop(assignment, None)
            if scope == SCOPE_SINGLE:
                self._pcs_target_by_assignment.pop(assignment, None)
        else:
            key = self._pcs_target_display_to_key.get(display)
            catalog = self._pcs_structure_target_catalog
            choice = catalog.choice_by_key(key) if catalog is not None else None
            if choice is not None:
                self._pcs_target_choice_key_by_assignment[assignment] = choice.key
                if choice.primary_ref_id is not None:
                    self._pcs_target_by_assignment[assignment] = int(choice.primary_ref_id)
        self._refresh_shift_analysis()
        self._refresh_pcs_preparation(preserve_assignment=assignment)

    def _toggle_pcs_use_from_click(self, event) -> None:
        region = self._pcs_tree.identify_region(event.x, event.y)
        column = self._pcs_tree.identify_column(event.x)
        row_id = self._pcs_tree.identify_row(event.y)
        if region != "cell" or column != "#1" or not row_id:
            return

        values = self._pcs_tree.item(row_id, "values")
        if len(values) < 2:
            return
        assignment = str(values[1])
        if assignment in self._pcs_excluded_assignments:
            self._pcs_excluded_assignments.discard(assignment)
        else:
            self._pcs_excluded_assignments.add(assignment)
        self._refresh_pcs_preparation(preserve_assignment=assignment)

    def _pcs_structured_pending_assignments(self) -> list[str]:
        # Compatibility helper retained for callers that want to list grouped
        # targets.  D5d can now hand these rows off; they are no longer blocked.
        pending = []
        for assignment in self._prepared_pcs_values:
            if assignment in self._pcs_excluded_assignments:
                continue
            choice = self._pcs_target_choice_for_assignment(assignment)
            if choice is not None and choice.requires_structured_handoff:
                pending.append(assignment)
        return pending

    def _pcs_handoff_rows(self):
        target_choices = {
            assignment: self._pcs_target_choice_for_assignment(assignment)
            for assignment in self._prepared_pcs_values
        }
        return build_handoff_rows(
            self._prepared_pcs_values,
            target_choices,
            excluded_assignments=self._pcs_excluded_assignments,
        )

    def _update_pcs_method_description(self) -> None:
        provider = self._pcs_providers.get(self._pcs_method.get())
        if provider is None:
            self._pcs_method_description.set(
                "Choose how experimental PCS values are derived before they are handed off to PCS Analyzer."
            )
        else:
            self._pcs_method_description.set(provider.description)

    def _on_pcs_method_changed(self, _event=None) -> None:
        self._update_pcs_method_description()
        self._refresh_pcs_preparation()
        self._on_pcs_row_selected()

    def _pcs_selected_assignment_name(self) -> str | None:
        tree = getattr(self, "_pcs_tree", None)
        if tree is None:
            return None
        selected = tree.selection()
        if not selected:
            return None
        values = tree.item(selected[0], "values")
        return str(values[1]) if len(values) > 1 else None

    @staticmethod
    def _parse_optional_float(text: str) -> float | None:
        value = text.strip()
        return None if not value else float(value)

    def _update_selected_pcs_inputs(self) -> None:
        assignment = self._pcs_selected_assignment_name()
        if not assignment:
            return
        try:
            correction = self._parse_optional_float(self._pcs_correction.get())
            manual = self._parse_optional_float(self._pcs_manual_value.get())
            uncertainty = self._parse_optional_float(self._pcs_uncertainty.get())
        except ValueError:
            self._pcs_selected_status.set(
                "Correction, PCS, and uncertainty must be numeric values."
            )
            return

        if correction is None:
            self._pcs_corrections_ppm.pop(assignment, None)
        else:
            self._pcs_corrections_ppm[assignment] = correction

        if manual is None:
            self._pcs_manual_values_ppm.pop(assignment, None)
        else:
            self._pcs_manual_values_ppm[assignment] = manual

        if uncertainty is None:
            self._pcs_uncertainties_ppm.pop(assignment, None)
        else:
            self._pcs_uncertainties_ppm[assignment] = uncertainty

        if self._pcs_target_scope_for_assignment(assignment) == SCOPE_CUSTOM:
            custom_text = self._pcs_target.get().strip()
            if custom_text:
                try:
                    ref_ids, labels = self._pcs_structure_ref_ids_and_labels()
                    custom_choice = parse_custom_ref_choice(
                        custom_text,
                        allowed_ref_ids=ref_ids,
                        labels_by_ref=labels,
                    )
                except PCSStructureTargetError as exc:
                    self._pcs_selected_status.set(str(exc))
                    return
                self._pcs_custom_target_text_by_assignment[assignment] = custom_text
                if custom_choice.primary_ref_id is not None:
                    self._pcs_target_by_assignment[assignment] = int(custom_choice.primary_ref_id)
            else:
                self._pcs_custom_target_text_by_assignment.pop(assignment, None)

        if self._pcs_target_scope_for_assignment(assignment) == SCOPE_CUSTOM:
            self._refresh_shift_analysis()
        self._refresh_pcs_preparation(preserve_assignment=assignment)
        self._refresh_predicted_pcs_markers()

    def _toggle_pcs_detail(self) -> None:
        if self._pcs_selected_assignment_name():
            self._pcs_detail_expanded = not self._pcs_detail_expanded
            self._sync_pcs_detail_layout()

    def _sync_pcs_detail_layout(self) -> None:
        assignment = self._pcs_selected_assignment_name()
        self._pcs_detail_hint.configure(
            text=f"{'▾' if self._pcs_detail_expanded else '▸'} Selected assignment: {assignment}" if assignment else "Selected assignment — select a row",
            state="normal" if assignment else "disabled")
        if assignment and self._pcs_detail_expanded:
            if not self._pcs_detail_form.winfo_manager():
                self._pcs_detail_form.pack(fill="x")
        else:
            self._pcs_detail_form.pack_forget()

    def _on_pcs_row_selected(self, _event=None) -> None:
        assignment = self._pcs_selected_assignment_name()
        if assignment != self._pcs_detail_assignment:
            self._pcs_detail_assignment = assignment
            self._pcs_detail_expanded = bool(assignment)
        self._sync_pcs_detail_layout()
        if not assignment:
            self._pcs_selected_assignment.set("—")
            self._pcs_correction.set("")
            self._pcs_manual_value.set("")
            self._pcs_uncertainty.set("")
            self._pcs_target_scope.set(SCOPE_SINGLE)
            self._pcs_target.set("")
            self._pcs_correction_entry.configure(state="disabled")
            self._pcs_manual_entry.configure(state="disabled")
            self._pcs_uncertainty_entry.configure(state="disabled")
            self._pcs_target_scope_box.configure(state="disabled")
            self._pcs_target_box.configure(state="disabled")
            self._pcs_update_button.configure(state="disabled")
            return

        self._pcs_selected_assignment.set(assignment)
        self._pcs_correction.set(
            "" if assignment not in self._pcs_corrections_ppm
            else f"{self._pcs_corrections_ppm[assignment]:.8g}"
        )
        self._pcs_manual_value.set(
            "" if assignment not in self._pcs_manual_values_ppm
            else f"{self._pcs_manual_values_ppm[assignment]:.8g}"
        )
        self._pcs_uncertainty.set(
            "" if assignment not in self._pcs_uncertainties_ppm
            else f"{self._pcs_uncertainties_ppm[assignment]:.8g}"
        )

        scope = self._pcs_target_scope_for_assignment(assignment)
        self._pcs_target_scope.set(scope)
        self._refresh_pcs_target_choices()
        self._pcs_target.set(
            self._pcs_target_display_for_assignment(assignment)
        )

        method = self._pcs_method.get()
        self._pcs_correction_entry.configure(
            state="normal" if method == "FCS-corrected" else "disabled"
        )
        self._pcs_manual_entry.configure(
            state="normal" if method == "Manual / Imported PCS" else "disabled"
        )
        self._pcs_uncertainty_entry.configure(state="normal")
        self._pcs_target_scope_box.configure(
            state="readonly" if self._pcs_state is not None else "disabled"
        )
        if self._pcs_state is None:
            target_state = "disabled"
        elif scope == SCOPE_CUSTOM:
            target_state = "normal"
        else:
            target_state = "readonly"
        self._pcs_target_box.configure(state=target_state)
        self._pcs_update_button.configure(state="normal")
        use_text = (
            "Excluded from transfer."
            if assignment in self._pcs_excluded_assignments
            else "Included in transfer."
        )
        choice = self._pcs_target_choice_for_assignment(assignment)
        if choice is not None and choice.requires_structured_handoff:
            target_note = (
                f" Structured target: {choice.site_count} site(s), "
                f"{choice.member_count} Ref member(s). The grouping is preserved "
                "during handoff and is not flattened into repeated delta_exp values."
            )
        elif scope == SCOPE_CUSTOM:
            target_note = " Enter comma-separated Ref IDs for a custom grouped target."
        else:
            target_note = ""
        self._pcs_selected_status.set(
            "Edit the derivation inputs and target mapping. "
            f"{use_text} Double-click the Use cell to toggle inclusion."
            + target_note
        )

    def _refresh_pcs_preparation(
        self,
        *,
        preserve_assignment: str | None = None,
    ) -> None:
        tree = getattr(self, "_pcs_tree", None)
        if tree is None:
            return

        if preserve_assignment is None:
            preserve_assignment = self._pcs_selected_assignment_name()

        for item in tree.get_children(""):
            tree.delete(item)
        self._prepared_pcs_values = {}
        self._refresh_structure_target_catalog()

        paramagnetic, diamagnetic, rows, summary = self._current_shift_rows()
        if paramagnetic is None or diamagnetic is None or summary is None:
            self._pcs_prep_status.set("No valid pair")
            self._pcs_source_text.set(
                "Pair paramagnetic and diamagnetic datasets in Shift Analysis first."
            )
            self._pcs_review_apply_button.configure(state="disabled")
            self._on_pcs_row_selected()
            return

        nucleus = dataset_nucleus(paramagnetic) or "?"
        self._pcs_source_text.set(
            f"Paramagnetic: {dataset_display_name(paramagnetic)}   |   "
            f"Diamagnetic: {dataset_display_name(diamagnetic)}   |   "
            f"Nucleus: {nucleus}"
        )

        provider = self._pcs_providers.get(self._pcs_method.get())
        context = self._pcs_context(paramagnetic, diamagnetic)
        ready = 0
        structured_pending = 0
        matched = 0

        for index, row in enumerate(rows):
            if row.status != "Matched":
                continue
            matched += 1

            correction = ""
            if self._pcs_method.get() == "FCS-corrected":
                value = self._pcs_corrections_ppm.get(row.assignment)
                correction = "—" if value is None else f"{value:.6f}"
            elif self._pcs_method.get() == "Direct from Δshift":
                correction = "0"
            else:
                correction = "—"

            uncertainty = self._pcs_uncertainties_ppm.get(row.assignment)
            uncertainty_text = "—" if uncertainty is None else f"{uncertainty:.6f}"

            derived = None
            status = "Choose method" if provider is None else "Needs input"
            if provider is not None:
                try:
                    derived = provider.derive(row, context)
                    self._prepared_pcs_values[row.assignment] = derived
                    status = "Ready"
                except PCSDerivationError:
                    status = "Needs input"

            if derived is not None:
                pcs_text = f"{derived.pcs_ppm:.6f}"
            elif self._pcs_method.get() == "Manual / Imported PCS":
                manual = self._pcs_manual_values_ppm.get(row.assignment)
                pcs_text = "—" if manual is None else f"{manual:.6f}"
            else:
                pcs_text = "—"

            scope = self._pcs_target_scope_for_assignment(row.assignment)
            target_display = self._pcs_target_display_for_assignment(row.assignment)
            target_choice = self._pcs_target_choice_for_assignment(row.assignment)
            included = row.assignment not in self._pcs_excluded_assignments
            if derived is not None and included:
                if self._pcs_state is None:
                    status = "PCS Analyzer unavailable"
                elif not target_display or target_choice is None:
                    status = "Needs mapping"
                elif target_choice.requires_structured_handoff:
                    structured_pending += 1
                    ready += 1
                    status = "Ready (structured)"
                else:
                    ready += 1
                    status = "Ready"
            elif derived is not None and not included:
                status = "Excluded"

            tree.insert(
                "",
                "end",
                iid=f"pcs-{index}",
                values=(
                    "☑" if included else "☐",
                    row.assignment,
                    self._format_shift_value(row.delta_para_ppm),
                    correction,
                    pcs_text,
                    uncertainty_text,
                    target_display or "—",
                    scope,
                    status,
                ),
            )

        try:
            pcs_target_available = self._pcs_target_state() is not None
        except RuntimeError:
            pcs_target_available = False
        if not pcs_target_available:
            self._pcs_prep_status.set(
                f"{ready}/{matched} ready · bound PCS document unavailable"
            )
            review_state = "disabled"
        else:
            structured_text = (
                f" · {structured_pending} structured" if structured_pending else ""
            )
            self._pcs_prep_status.set(
                f"{ready}/{matched} ready{structured_text}"
            )
            review_state = "normal" if ready else "disabled"
        self._pcs_review_apply_button.configure(state=review_state)

        if preserve_assignment:
            for item in tree.get_children(""):
                values = tree.item(item, "values")
                if len(values) > 1 and str(values[1]) == preserve_assignment:
                    tree.selection_set(item)
                    tree.focus(item)
                    tree.see(item)
                    break
        self._on_pcs_row_selected()

    def _review_and_apply_pcs(self) -> None:
        try:
            pcs_target_state = self._pcs_target_state()
        except RuntimeError as exc:
            self._show_processing_error(exc)
            return
        if pcs_target_state is None:
            self._show_processing_error(
                ValueError("PCS Analyzer state is not available.")
            )
            return

        rows = self._pcs_handoff_rows()
        try:
            summary = preview_handoff_summary(pcs_target_state, rows)
        except PCSStructuredHandoffError as exc:
            self._show_processing_error(exc)
            return

        selected_rows = [row for row in rows if row.included]

        top = tk.Toplevel(self)
        top.title("Review Experimental PCS Handoff")
        top.transient(self)
        top.grab_set()
        top.geometry("900x540")

        paramagnetic, diamagnetic = self._current_shift_pair()
        source_text = (
            f"Paramagnetic: {dataset_display_name(paramagnetic)}\n"
            f"Diamagnetic: {dataset_display_name(diamagnetic)}\n"
            f"Method: {self._pcs_method.get()}\n"
            f"Temperature: "
            f"{self._pair_temperature_K(paramagnetic, diamagnetic) or '—'} K"
        )
        ttk.Label(
            top,
            text=source_text,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(10, 6))

        summary_parts = [
            f"{summary.applied} selected",
            f"{summary.single_atom} single atom",
            f"{summary.structured} structured",
            f"{summary.added} new",
            f"{summary.replaced} replace",
        ]
        if summary.delta_exp_overwritten:
            summary_parts.append(
                f"{summary.delta_exp_overwritten} delta_exp overwrite"
            )
        ttk.Label(
            top,
            text=" · ".join(summary_parts),
            font=self._title_font,
        ).pack(anchor="w", padx=10, pady=(0, 8))

        columns = ("assignment", "pcs", "scope", "target", "action")
        tree = ttk.Treeview(
            top,
            columns=columns,
            show="headings",
            height=12,
        )
        for name, label, width, anchor_name in (
            ("assignment", "Assignment", 150, "w"),
            ("pcs", "PCS / ppm", 110, "e"),
            ("scope", "Target scope", 150, "w"),
            ("target", "Structural target", 330, "w"),
            ("action", "Action", 90, "center"),
        ):
            tree.heading(name, text=label)
            tree.column(name, width=width, anchor=anchor_name)
        tree.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        existing_records = [
            record
            for record in (
                pcs_target_state.get(PCS_OBSERVATIONS_STATE_KEY, []) or []
            )
            if isinstance(record, dict) and record.get("assignment")
        ]
        existing_delta = pcs_target_state.get("delta_exp_values", {}) or {}

        def record_matches_row(record, row) -> bool:
            row_observables = tuple(row.source_observable_ids)
            record_observables = tuple(
                str(value)
                for value in (record.get("source_observable_ids", ()) or ())
            )
            if row_observables and record_observables:
                return row_observables == record_observables
            row_datasets = tuple(row.source_dataset_ids)
            record_datasets = tuple(
                str(value)
                for value in (record.get("source_dataset_ids", ()) or ())
            )
            if row_datasets and record_datasets:
                return (
                    row_datasets == record_datasets
                    and str(record.get("assignment", "")) == row.assignment
                )
            return (
                not record_observables
                and not record_datasets
                and str(record.get("assignment", "")) == row.assignment
            )

        for row in selected_rows:
            target = row.target
            if target is None:
                continue
            action = (
                "Replace"
                if any(record_matches_row(record, row) for record in existing_records)
                else "New"
            )
            if (
                action == "New"
                and target.is_legacy_single_atom
                and target.primary_ref_id is not None
                and int(target.primary_ref_id) in existing_delta
            ):
                action = "Overwrite"

            tree.insert(
                "",
                "end",
                values=(
                    row.assignment,
                    f"{row.pcs_ppm:.6f}",
                    target.scope,
                    target.display,
                    action,
                ),
            )

        note = (
            "This step only hands validated experimental PCS values and their "
            "structural target definitions to PCS Analyzer. Structured targets "
            "remain grouped metadata and are not expanded into repeated "
            "delta_exp entries. Single-atom targets also update delta_exp_values "
            "for compatibility with the current PCS Analyzer workflow. No PCS "
            "calculation, residual evaluation, or fitting is performed here."
        )
        ttk.Label(
            top,
            text=note,
            wraplength=860,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 8))

        buttons = ttk.Frame(top)
        buttons.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(
            buttons,
            text="Back",
            command=top.destroy,
        ).pack(side="right")

        def apply_now():
            try:
                result = apply_structured_pcs_handoff(pcs_target_state, rows)
            except PCSStructuredHandoffError as exc:
                messagebox.showerror(
                    "Apply Experimental PCS",
                    str(exc),
                    parent=top,
                )
                return
            except Exception as exc:
                messagebox.showerror(
                    "Apply Experimental PCS",
                    f"PCS Analyzer handoff failed:\n{exc}",
                    parent=top,
                )
                return

            top.destroy()
            self._pcs_prep_status.set(
                f"Handed off {result.applied} PCS values "
                f"({result.single_atom} single atom, "
                f"{result.structured} structured)."
            )
            self._refresh_pcs_preparation()
            self._refresh_shift_analysis()

        ttk.Button(
            buttons,
            text="Apply to PCS Analyzer",
            command=apply_now,
        ).pack(side="right", padx=(0, 6))

    def _open_pcs_preparation(self) -> None:
        self._refresh_pcs_preparation()
        # Select the PCS Preparation notebook tab without coupling the backend
        # to a particular tab index.
        notebook = getattr(self, "_inspector_notebook", None)
        if notebook is not None and hasattr(self, "_pcs_tab"):
            notebook.select(self._pcs_tab)

    def _dataset_by_id(self, dataset_id: str | None):
        if dataset_id is None:
            return None
        return next(
            (dataset for dataset in self.model.datasets if dataset.id == dataset_id),
            None,
        )

    def _dataset_pair_display(self, dataset) -> str:
        nucleus = dataset_nucleus(dataset) or "?"
        role = getattr(dataset, "experimental_role", "unspecified")
        return f"{dataset_display_name(dataset)} · {nucleus} · {role.title()}"

    def _opposite_dataset_role(self, role: str) -> str | None:
        if role == "paramagnetic":
            return "diamagnetic"
        if role == "diamagnetic":
            return "paramagnetic"
        return None

    def _refresh_dataset_role_controls(self) -> None:
        dataset = self.current_dataset
        if dataset is None:
            self._dataset_role.set("Unspecified")
            self._dataset_role_box.configure(state="disabled")
            self._dataset_pair.set("")
            self._dataset_pair_box.configure(values=(), state="disabled")
            return

        role = getattr(dataset, "experimental_role", "unspecified")
        self._dataset_role.set(role.title())
        self._dataset_role_box.configure(state="readonly")

        opposite = self._opposite_dataset_role(role)
        current_nucleus = dataset_nucleus(dataset)
        candidates = []
        if opposite is not None and current_nucleus is not None:
            for candidate in self.model.datasets:
                if candidate.id == dataset.id:
                    continue
                if getattr(candidate, "experimental_role", "unspecified") != opposite:
                    continue
                if dataset_nucleus(candidate) != current_nucleus:
                    continue
                candidates.append(candidate)

        self._pair_display_to_id = {
            self._dataset_pair_display(candidate): candidate.id
            for candidate in candidates
        }
        values = tuple(self._pair_display_to_id)
        paired_id = self._dataset_pair_ids.get(dataset.id)
        paired_display = next(
            (
                display
                for display, candidate_id in self._pair_display_to_id.items()
                if candidate_id == paired_id
            ),
            "",
        )
        self._dataset_pair.set(paired_display)
        self._dataset_pair_box.configure(
            values=values,
            state="readonly" if values else "disabled",
        )

        if role == "unspecified":
            self._dataset_pair_status.set(
                "Choose Paramagnetic or Diamagnetic before pairing."
            )
        elif current_nucleus is None:
            self._dataset_pair_status.set(
                "This dataset needs a nucleus before pairing."
            )
        elif paired_display:
            self._dataset_pair_status.set(f"Paired with {paired_display}")
        elif values:
            self._dataset_pair_status.set(
                f"Choose a {opposite.title()} {current_nucleus} dataset."
            )
        else:
            self._dataset_pair_status.set(
                f"No compatible {opposite.title()} {current_nucleus} dataset loaded."
            )

    def _clear_dataset_pair(self, dataset_id: str) -> None:
        other_id = self._dataset_pair_ids.pop(dataset_id, None)
        if other_id is not None and self._dataset_pair_ids.get(other_id) == dataset_id:
            self._dataset_pair_ids.pop(other_id, None)

    def _on_dataset_role_changed(self, _event=None) -> None:
        dataset = self.current_dataset
        if dataset is None:
            return

        new_role = self._dataset_role.get().strip().lower()
        if new_role not in {"unspecified", "paramagnetic", "diamagnetic"}:
            return

        if dataset.experimental_role != new_role:
            self._clear_dataset_pair(dataset.id)
            dataset.experimental_role = new_role

        self._refresh_dataset_role_controls()
        self._refresh_shift_analysis()

    def _on_dataset_pair_changed(self, _event=None) -> None:
        dataset = self.current_dataset
        if dataset is None:
            return

        target_id = self._pair_display_to_id.get(self._dataset_pair.get())
        target = self._dataset_by_id(target_id)
        if target is None:
            self._clear_dataset_pair(dataset.id)
            self._refresh_shift_analysis()
            return

        if not roles_are_pairable(dataset, target):
            self._dataset_pair_status.set("The selected dataset roles are not pairable.")
            return
        if dataset_nucleus(dataset) != dataset_nucleus(target):
            self._dataset_pair_status.set("Dataset nuclei do not match.")
            return

        self._clear_dataset_pair(dataset.id)
        self._clear_dataset_pair(target.id)
        self._dataset_pair_ids[dataset.id] = target.id
        self._dataset_pair_ids[target.id] = dataset.id

        self._refresh_dataset_role_controls()
        self._refresh_shift_analysis()

    def _current_shift_pair(self):
        dataset = self.current_dataset
        if dataset is None:
            return None, None

        paired = self._dataset_by_id(self._dataset_pair_ids.get(dataset.id))
        if paired is None:
            return None, None

        if dataset.experimental_role == "paramagnetic":
            return dataset, paired
        if dataset.experimental_role == "diamagnetic":
            return paired, dataset
        return None, None

    @staticmethod
    def _format_shift_value(value: float | None) -> str:
        return "—" if value is None else f"{value:.6f}"

    def _refresh_shift_analysis(self) -> None:
        self._refresh_comparison_choices()
        tree = getattr(self, "_shift_tree", None)
        if tree is None:
            return

        selected_assignment = None
        selected = tree.selection()
        if selected:
            values = tree.item(selected[0], "values")
            if values:
                selected_assignment = str(values[0])

        for item in tree.get_children(""):
            tree.delete(item)

        self._shift_diagnostic_rows = []

        paramagnetic, diamagnetic = self._current_shift_pair()
        if paramagnetic is None or diamagnetic is None:
            self._shift_analysis_status.set("No valid pair")
            self._shift_pair_text.set(
                "Choose observed / paramagnetic and reference / diamagnetic "
                "experiments above, then Compare."
            )
            if hasattr(self, "_pcs_prepare_button"):
                self._pcs_prepare_button.configure(state="disabled")
            self._refresh_shift_visualization()
            self._refresh_predicted_pcs_markers()
            return

        try:
            rows, summary = build_shift_comparison(paramagnetic, diamagnetic)
        except ShiftAnalysisError as exc:
            self._shift_analysis_status.set("Pair error")
            self._shift_pair_text.set(str(exc))
            self._pcs_prepare_button.configure(state="disabled")
            self._refresh_shift_visualization()
            self._refresh_predicted_pcs_markers()
            return

        explicit_scopes = getattr(
            self, "_pcs_target_scope_by_assignment", {}
        )
        explicit_choices = getattr(
            self, "_pcs_target_choice_key_by_assignment", {}
        )
        explicit_custom = getattr(
            self, "_pcs_custom_target_text_by_assignment", {}
        )
        self._shift_diagnostic_rows = build_shift_diagnostics(
            rows,
            self._pcs_state,
            structured_target_choices={
                row.assignment: choice
                for row in rows
                if (
                    row.assignment in explicit_scopes
                    or row.assignment in explicit_choices
                    or row.assignment in explicit_custom
                )
                and (choice := self._pcs_target_choice_for_assignment(row.assignment))
                is not None
            },
        )

        nucleus = dataset_nucleus(paramagnetic) or "?"
        self._shift_pair_text.set(
            f"Paramagnetic: {dataset_display_name(paramagnetic)}   |   "
            f"Diamagnetic: {dataset_display_name(diamagnetic)}   |   "
            f"Nucleus: {nucleus}"
        )

        diagnostic_count = 0
        for index, row in enumerate(self._shift_diagnostic_rows):
            if row.non_pcs_ppm is not None:
                diagnostic_count += 1
            tree.insert(
                "",
                "end",
                iid=f"shift-{index}",
                values=(
                    row.assignment,
                    self._format_shift_value(row.paramagnetic_ppm),
                    self._format_shift_value(row.diamagnetic_ppm),
                    self._format_shift_value(row.delta_para_ppm),
                    self._format_shift_value(row.pcs_calculated_ppm),
                    self._format_shift_value(
                        self._pcs_corrections_ppm.get(row.assignment, 0.0)
                    ),
                    self._format_shift_value(row.non_pcs_ppm),
                    row.status,
                ),
            )

        status = f"{summary.matched}/{summary.total} matched"
        if summary.ambiguous:
            status += f" · {summary.ambiguous} ambiguous"
        if diagnostic_count:
            status += f" · {diagnostic_count} PCS diagnostics"
        self._shift_analysis_status.set(status)

        self._pcs_prepare_button.configure(
            state="normal" if summary.matched else "disabled"
        )

        if selected_assignment:
            for item in tree.get_children(""):
                values = tree.item(item, "values")
                if values and str(values[0]) == selected_assignment:
                    tree.selection_set(item)
                    tree.focus(item)
                    tree.see(item)
                    break

        self._refresh_shift_visualization()

        if hasattr(self, "_pcs_tree"):
            self._refresh_pcs_preparation()
        self._refresh_predicted_pcs_markers()

    def refresh_from_pcs_state(self) -> bool:
        """Refresh only Workbench data derived from the host PCS model.

        ``push_layers_to_nmr_if_open`` is the compatibility endpoint used by
        both the typed PCS refresh consumer and legacy callback paths.  Those
        paths can converge more than once for one logical update, so this
        method rejects re-entrant and unchanged deliveries before rebuilding
        Shift Analysis diagnostics.  Ordinary PCS/model updates do not modify
        datasets, assignment data, the active dataset, or spectrum/peak
        selection. Legacy synthetic datasets are synchronized only when their
        source ``delta_obs``/``delta_dia`` dictionaries actually change.

        Returns ``True`` when a refresh ran and ``False`` for a duplicate or
        unavailable host state.
        """
        try:
            pcs_target_state = self._pcs_target_state()
        except RuntimeError:
            # A future document switch must not silently retarget this window.
            return False
        pcs_signature = _pcs_derived_state_signature(pcs_target_state)
        if pcs_signature is None:
            return False
        if getattr(self, "_pcs_refresh_active", False):
            return False

        legacy_signature = _legacy_shift_state_signature(pcs_target_state)
        pcs_changed = pcs_signature != getattr(
            self, "_pcs_refresh_signature", None
        )
        legacy_was_initialized = hasattr(
            self, "_legacy_shift_refresh_signature"
        )
        legacy_changed = legacy_signature != getattr(
            self, "_legacy_shift_refresh_signature", None
        )
        if not pcs_changed and not legacy_changed:
            return False

        self._pcs_refresh_active = True
        try:
            if legacy_changed and (
                legacy_signature is not None or legacy_was_initialized
            ):
                result = sync_legacy_project_datasets(
                    self.model,
                    pcs_target_state,
                )

                for legacy_id in (
                    LEGACY_PARAMAGNETIC_DATASET_ID,
                    LEGACY_DIAMAGNETIC_DATASET_ID,
                ):
                    self._clear_dataset_pair(legacy_id)

                if result.has_pair:
                    self._dataset_pair_ids[result.paramagnetic_id] = (
                        result.diamagnetic_id
                    )
                    self._dataset_pair_ids[result.diamagnetic_id] = (
                        result.paramagnetic_id
                    )

                self._rebuild_experiment_tree()
                selected_id = self.model.selected_dataset_id
                active_id = getattr(
                    getattr(self.processing_session, "dataset", None),
                    "id",
                    None,
                )
                if selected_id is not None and (
                    active_id != selected_id or selected_id in result.dataset_ids
                ):
                    self._select_dataset(selected_id)
                else:
                    # A real spectrum or preview remains active while only
                    # host-side compatibility datasets are synchronized.
                    self._refresh_dataset_role_controls()
                    self._refresh_shift_analysis()
                    if selected_id is None:
                        self._refresh()
            else:
                self._refresh_shift_analysis()
        except Exception:
            # Leave the previous signature intact so a later compatibility
            # notification can retry after a transient UI failure.
            raise
        else:
            self._pcs_refresh_signature = pcs_signature
            self._legacy_shift_refresh_signature = legacy_signature
            return True
        finally:
            self._pcs_refresh_active = False

    @staticmethod
    def _displayable_frequency_spectrum(dataset):
        spectrum = getattr(dataset, "processed_spectrum", None)
        if (
            spectrum is None
            or getattr(spectrum, "domain", None) != "frequency"
            or str(getattr(spectrum, "axis_unit", "")).lower() != "ppm"
        ):
            return None
        return spectrum

    @staticmethod
    def _normalized_real_values(spectrum, normalize: bool):
        values = np.real(np.asarray(spectrum.intensities, dtype=complex))
        if not normalize:
            return values
        finite = np.abs(values[np.isfinite(values)])
        if finite.size == 0:
            return values
        scale = float(np.max(finite))
        if not np.isfinite(scale) or scale <= 0:
            return values
        return values / scale

    def _on_shift_view_changed(self, _event=None) -> None:
        self._sync_shift_inspector_layout()
        self._refresh_shift_visualization()

    def _selected_shift_assignment(self) -> str | None:
        tree = getattr(self, "_shift_tree", None)
        if tree is None:
            return None
        selected = tree.selection()
        if not selected:
            return None
        values = tree.item(selected[0], "values")
        return str(values[0]) if values else None

    def _on_shift_row_selected(self, _event=None) -> None:
        assignment = self._selected_shift_assignment()
        if assignment and not self._shared_selection_syncing:
            self._publish_assignment_selection(
                assignment,
                source="nmr_shift_analysis",
            )
        self._refresh_shift_visualization()

    def _show_structure_context_menu(self, event, *, source_tree: str) -> str | None:
        """Show D5e structure actions for one assigned peak/shift row."""
        tree = self._peak_tree if source_tree == "peak" else self._shift_tree
        row_id = tree.identify_row(event.y)
        if not row_id:
            return None

        try:
            tree.selection_set(row_id)
            tree.focus(row_id)
        except Exception:
            pass

        values = tree.item(row_id, "values")
        assignment = ""
        if source_tree == "peak":
            if len(values) > 5:
                assignment = str(values[5]).strip()
        elif values:
            assignment = str(values[0]).strip()

        if assignment and isinstance(self._pcs_state, dict):
            # Rebuild once per explicit structure-menu request so in-place
            # connectivity edits are reflected without repeating the analysis
            # for every menu item.
            self._refresh_structure_target_catalog()

        menu = tk.Menu(self, tearoff=False)
        if not assignment:
            menu.add_command(label="Assign this peak first", state="disabled")
        else:
            current = self._structure_selection_for_assignment(assignment)
            nmr_site = self._nmr_site_selection_for_assignment(assignment)
            matching = self._matching_site_selection_for_assignment(assignment)
            preview = self._preview_structure_selection_for_assignment(assignment)
            use_choice = preview or current

            menu.add_command(
                label="Show in Chemical Map",
                state="normal" if current is not None else "disabled",
                command=lambda a=assignment: self._show_assignment_in_chemical_map(a),
            )
            menu.add_command(
                label="Select NMR site",
                state="normal" if nmr_site is not None else "disabled",
                command=lambda a=assignment: self._select_assignment_nmr_site(a),
            )
            menu.add_command(
                label="Select matching ligand sites",
                state="normal" if matching is not None else "disabled",
                command=lambda a=assignment: self._select_assignment_matching_sites(a),
            )
            menu.add_separator()
            menu.add_command(
                label="Use current structure target in PCS Preparation",
                state="normal" if use_choice is not None else "disabled",
                command=lambda a=assignment: self._use_assignment_structure_in_pcs_preparation(a),
            )

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            try:
                menu.grab_release()
            except Exception:
                pass
        return "break"

    def _structure_target_catalog_for_actions(self):
        if not isinstance(self._pcs_state, dict):
            return None
        graph = self._pcs_state.get("structure_graph")
        graph_identity = id(graph) if graph is not None else None
        if (
            self._pcs_structure_target_catalog is None
            or graph_identity != self._pcs_structure_target_graph_identity
        ):
            self._refresh_structure_target_catalog()
        return self._pcs_structure_target_catalog

    def _preview_structure_selection_for_assignment(
        self,
        assignment: str,
    ) -> AssignmentStructureSelection | None:
        catalog = self._structure_target_catalog_for_actions()
        if catalog is None:
            return None
        key = self._structure_preview_choice_key_by_assignment.get(str(assignment))
        if not key:
            return None
        choice = catalog.choice_by_key(key)
        if choice is None:
            self._structure_preview_choice_key_by_assignment.pop(str(assignment), None)
            return None
        return AssignmentStructureSelection(
            assignment=str(assignment),
            choice=choice,
            source="preview",
        )

    def _structure_selection_for_assignment(
        self,
        assignment: str,
    ) -> AssignmentStructureSelection | None:
        """Resolve the current structural target without changing PCS Preparation."""
        assignment = str(assignment)
        preview = self._preview_structure_selection_for_assignment(assignment)
        if preview is not None:
            return preview

        catalog = self._structure_target_catalog_for_actions()
        if catalog is None:
            return None

        # Respect only target state that already exists.  Do not call the D5c
        # convenience resolver here because it may populate suggested target
        # mappings as a side effect during PCS Preparation refresh.
        scope = self._pcs_target_scope_by_assignment.get(assignment)
        key = self._pcs_target_choice_key_by_assignment.get(assignment)
        if key:
            prepared_choice = catalog.choice_by_key(key)
            if prepared_choice is not None:
                return AssignmentStructureSelection(
                    assignment=assignment,
                    choice=prepared_choice,
                    source="pcs_preparation",
                )

        if scope == SCOPE_CUSTOM:
            text = self._pcs_custom_target_text_by_assignment.get(assignment, "")
            if text:
                try:
                    ref_ids, labels = self._pcs_structure_ref_ids_and_labels()
                    prepared_choice = parse_custom_ref_choice(
                        text,
                        allowed_ref_ids=ref_ids,
                        labels_by_ref=labels,
                    )
                except PCSStructureTargetError:
                    prepared_choice = None
                if prepared_choice is not None:
                    return AssignmentStructureSelection(
                        assignment=assignment,
                        choice=prepared_choice,
                        source="pcs_preparation",
                    )

        fallback_ref = self._target_ref_for_assignment(assignment)
        return resolve_assignment_structure_selection(
            assignment,
            catalog,
            fallback_ref_id=fallback_ref,
        )

    def _nmr_site_selection_for_assignment(
        self,
        assignment: str,
    ) -> AssignmentStructureSelection | None:
        catalog = self._structure_target_catalog_for_actions()
        return nmr_site_selection(
            assignment,
            catalog,
            fallback_ref_id=self._target_ref_for_assignment(assignment),
        )

    def _matching_site_selection_for_assignment(
        self,
        assignment: str,
    ) -> AssignmentStructureSelection | None:
        catalog = self._structure_target_catalog_for_actions()
        return matching_site_selection(
            assignment,
            catalog,
            fallback_ref_id=self._target_ref_for_assignment(assignment),
        )

    def _assignment_peak_ids(self, assignment: str) -> tuple[str, ...]:
        dataset = self.current_dataset
        if dataset is None:
            return ()
        return tuple(
            peak.id
            for peak in dataset.peaks
            if self._assignment_for_peak(peak.id).strip() == str(assignment).strip()
        )

    def _open_chemical_map_and_publish(
        self,
        selection: AssignmentStructureSelection,
        *,
        source: str,
    ) -> bool:
        if not isinstance(self._pcs_state, dict):
            messagebox.showinfo(
                "Structure selection",
                "PCS Analyzer structure state is unavailable in standalone NMR Workbench mode.",
                parent=self,
            )
            return False
        try:
            # Import lazily so NMR Workbench can still be imported standalone.
            from ui.chemical_map_window import open_chemical_map_window

            open_chemical_map_window(self._pcs_state)
        except Exception as exc:
            messagebox.showerror(
                "Chemical Map",
                f"Could not open Chemical Map:\n{exc}",
                parent=self,
            )
            return False

        dataset = self.current_dataset
        publish_workbench_selection(
            self._pcs_state,
            assignment=selection.assignment,
            peak_ids=self._assignment_peak_ids(selection.assignment),
            dataset_id=(dataset.id if dataset is not None else None),
            ref_ids=selection.ref_ids,
            source=source,
        )
        return True

    def _show_assignment_in_chemical_map(self, assignment: str) -> None:
        selection = self._structure_selection_for_assignment(assignment)
        if selection is None:
            messagebox.showinfo(
                "Chemical Map",
                f"No structural target is available for assignment {assignment!r}.",
                parent=self,
            )
            return
        self._open_chemical_map_and_publish(
            selection,
            source="nmr_structure_map",
        )

    def _select_assignment_nmr_site(self, assignment: str) -> None:
        selection = self._nmr_site_selection_for_assignment(assignment)
        if selection is None:
            messagebox.showinfo(
                "NMR site",
                "No grouped NMR site is available for this assignment. "
                "CH2/NH2 and symmetry-related aromatic protons are not auto-collapsed.",
                parent=self,
            )
            return
        self._structure_preview_choice_key_by_assignment[str(assignment)] = (
            selection.choice.key
        )
        self._open_chemical_map_and_publish(
            selection,
            source="nmr_structure_site",
        )

    def _select_assignment_matching_sites(self, assignment: str) -> None:
        selection = self._matching_site_selection_for_assignment(assignment)
        if selection is None:
            messagebox.showinfo(
                "Matching ligand sites",
                "No unique matching ligand-site group is available for this assignment. "
                "Symmetry-ambiguous mappings are intentionally not selected automatically.",
                parent=self,
            )
            return
        self._structure_preview_choice_key_by_assignment[str(assignment)] = (
            selection.choice.key
        )
        self._open_chemical_map_and_publish(
            selection,
            source="nmr_structure_matching",
        )

    def _use_assignment_structure_in_pcs_preparation(self, assignment: str) -> None:
        """Copy the previewed structure target into PCS Preparation only."""
        assignment = str(assignment)
        selection = (
            self._preview_structure_selection_for_assignment(assignment)
            or self._structure_selection_for_assignment(assignment)
        )
        if selection is None:
            messagebox.showinfo(
                "PCS Preparation",
                f"No structural target is available for assignment {assignment!r}.",
                parent=self,
            )
            return

        choice = selection.choice
        self._pcs_target_scope_by_assignment[assignment] = choice.scope
        if choice.scope == SCOPE_CUSTOM:
            # A pre-existing custom target is already represented by its text;
            # D5e never manufactures a custom group from a map selection.
            pass
        else:
            self._pcs_target_choice_key_by_assignment[assignment] = choice.key
        if choice.primary_ref_id is not None:
            self._pcs_target_by_assignment[assignment] = int(choice.primary_ref_id)

        self._structure_preview_choice_key_by_assignment.pop(assignment, None)
        self._refresh_pcs_preparation(preserve_assignment=assignment)
        notebook = getattr(self, "_inspector_notebook", None)
        if notebook is not None and hasattr(self, "_pcs_tab"):
            notebook.select(self._pcs_tab)
        self._pcs_selected_status.set(
            f"Structure target linked from NMR assignment: {choice.display}. "
            "Review & Apply is still required for PCS Analyzer handoff."
        )

    def _target_ref_for_assignment(self, assignment: str | None) -> int | None:
        # Resolve an NMR assignment to a PCS Analyzer Ref without mutating state.
        if not assignment:
            return None
        assignment = str(assignment)

        explicit = self._pcs_target_by_assignment.get(assignment)
        if explicit is not None:
            try:
                return int(explicit)
            except (TypeError, ValueError):
                pass

        for row in getattr(self, "_shift_diagnostic_rows", ()) or ():
            if row.assignment == assignment and row.target_ref_id is not None:
                try:
                    return int(row.target_ref_id)
                except (TypeError, ValueError):
                    pass

        if isinstance(self._pcs_state, dict):
            try:
                targets = extract_pcs_targets(self._pcs_state)
                ref_id = suggest_target_ref_id(assignment, targets)
                return int(ref_id) if ref_id is not None else None
            except Exception:
                pass
        return None

    def _publish_assignment_selection(
        self,
        assignment: str,
        *,
        peak_ids=(),
        source: str,
    ) -> None:
        dataset = self.current_dataset
        target_ref = self._target_ref_for_assignment(assignment)
        publish_workbench_selection(
            self._pcs_state,
            assignment=assignment,
            peak_ids=peak_ids,
            dataset_id=(dataset.id if dataset is not None else None),
            ref_ids=(() if target_ref is None else (target_ref,)),
            source=source,
        )

    def _publish_peak_selection(self) -> None:
        dataset = self.current_dataset
        if dataset is None:
            return

        selected_ids = self._selected_peak_ids()
        # A filtered-out model selection still has semantic identity.
        if not selected_ids and any(p.id == self.model.selected_peak_id for p in dataset.peaks):
            selected_ids = (self.model.selected_peak_id,)
        if not selected_ids:
            observation = next((o for o in dataset.observables if o.id == self.model.selected_observable_id), None)
            if observation is not None and observation.source_peak_id is None:
                # A generated empty Peaks selection must not clear a valid
                # spectrum-free observation's Chemical Map selection.
                selection = self._structure_selection_for_assignment(observation.assignment or "")
                publish_workbench_selection(self._pcs_state, assignment=observation.assignment,
                    dataset_id=dataset.id, ref_ids=selection.ref_ids if selection else (), source="nmr_data_table")
                return
        assignments = {
            self._assignment_for_peak(peak_id).strip()
            for peak_id in selected_ids
            if self._assignment_for_peak(peak_id).strip()
        }

        assignment = next(iter(assignments)) if len(assignments) == 1 else None
        target_ref = self._target_ref_for_assignment(assignment)

        publish_workbench_selection(
            self._pcs_state,
            assignment=assignment,
            peak_ids=selected_ids,
            dataset_id=dataset.id,
            ref_ids=(() if target_ref is None else (target_ref,)),
            source="nmr_peak_table",
        )

    def _assignment_for_shared_ref(self, ref_id: int | None) -> str | None:
        if ref_id is None:
            return None

        for row in getattr(self, "_shift_diagnostic_rows", ()) or ():
            try:
                target_ref = (
                    int(row.target_ref_id)
                    if row.target_ref_id is not None
                    else None
                )
            except (TypeError, ValueError):
                target_ref = None
            if target_ref == int(ref_id):
                return str(row.assignment)

        for assignment, target_ref in self._pcs_target_by_assignment.items():
            try:
                if int(target_ref) == int(ref_id):
                    return str(assignment)
            except (TypeError, ValueError):
                continue
        # Independent/manual observations do not appear in the Peaks tree.
        # Resolve only an unambiguous existing mapping for structure navigation.
        assignments = set()
        for dataset in self.model.datasets:
            for item in dataset.observables:
                if item.assignment:
                    selection = self._structure_selection_for_assignment(item.assignment)
                    if selection and int(ref_id) in selection.ref_ids:
                        assignments.add(item.assignment)
        if len(assignments) == 1:
            return next(iter(assignments))
        return None

    def _select_peak_assignment_from_shared_context(
        self,
        assignment: str | None,
        *,
        peak_ids=(),
        dataset_id: str | None = None,
    ) -> None:
        tree = getattr(self, "_peak_tree", None)
        dataset = self.current_dataset
        if tree is None or dataset is None:
            return

        selected = []
        if dataset_id is not None and str(dataset.id) == str(dataset_id):
            selected = [
                str(peak_id)
                for peak_id in peak_ids
                if tree.exists(str(peak_id))
            ]

        if not selected and assignment:
            selected = [
                peak.id
                for peak in dataset.peaks
                if self._assignment_for_peak(peak.id) == assignment
                and tree.exists(peak.id)
            ]

        if selected:
            tree.selection_set(selected)
            tree.focus(selected[0])
            tree.see(selected[0])
            self._refresh_selected_peak_highlights()

    def _on_shared_selection_context(self, context) -> None:
        # Receive selection originating in PCS/structure views.
        if getattr(context, "source", None) in {
            "nmr_peak_table",
            "nmr_shift_analysis",
            "nmr_shift_plot",
            "nmr_structure_map",
            "nmr_structure_site",
            "nmr_structure_matching",
            "nmr_data_table",
        }:
            return

        assignment = getattr(context, "assignment", None)
        if not assignment:
            assignment = self._assignment_for_shared_ref(
                getattr(context, "primary_ref_id", None)
            )

        peak_ids = getattr(context, "peak_ids", ()) or ()
        dataset_id = getattr(context, "dataset_id", None)

        self._shared_selection_syncing = True
        try:
            if assignment:
                tree = getattr(self, "_shift_tree", None)
                if tree is not None:
                    for item in tree.get_children(""):
                        values = tree.item(item, "values")
                        if values and str(values[0]) == str(assignment):
                            tree.selection_set(item)
                            tree.focus(item)
                            tree.see(item)
                            break

            self._select_peak_assignment_from_shared_context(
                assignment,
                peak_ids=peak_ids,
                dataset_id=dataset_id,
            )
            panel = getattr(self, "_nmr_data_panel", None)
            if panel is not None:
                panel.select_identity(dataset_id=dataset_id or self.model.selected_dataset_id,
                    peak_id=peak_ids[0] if len(peak_ids) == 1 else None, assignment=assignment)
                if assignment and not dataset_id:
                    panel.select_identity(assignment=assignment)
            self._refresh_shift_visualization()
        finally:
            self._shared_selection_syncing = False

    def destroy(self) -> None:
        if hasattr(self, "experiments"):
            self._cleanup_experiment_actions()
        pending = getattr(self, "_peak_selection_projection_idle", None)
        if pending is not None:
            self.after_cancel(pending)
            self._peak_selection_projection_idle = None
        unsubscribe_workbench_selection(
            getattr(self, "_shared_selection_subscription", None)
        )
        super().destroy()

    def _shift_broadening_value(self) -> float:
        try:
            value = float(self._shift_broadening_ppm.get())
        except (TypeError, ValueError):
            return 0.03
        if not np.isfinite(value) or value <= 0:
            return 0.03
        return value

    def _register_shift_candidate(
        self,
        *,
        assignment: str,
        x: float,
        y: float,
        layer: str,
        value: float,
    ) -> None:
        self._shift_plot_candidates.append(
            {
                "assignment": str(assignment),
                "x": float(x),
                "y": float(y),
                "layer": str(layer),
                "value": float(value),
            }
        )

    def _nearest_shift_candidate(self, event, max_pixels: float = 12.0):
        axes = getattr(self, "_shift_axes", None)
        if (
            axes is None
            or event.inaxes is not axes
            or event.x is None
            or event.y is None
            or not self._shift_plot_candidates
        ):
            return None

        best = None
        best_distance = float(max_pixels)
        for candidate in self._shift_plot_candidates:
            px, py = axes.transData.transform((candidate["x"], candidate["y"]))
            distance = math.hypot(float(event.x) - px, float(event.y) - py)
            if distance <= best_distance:
                best = candidate
                best_distance = distance
        return best

    def _clear_shift_hover_annotation(self) -> None:
        annotation = getattr(self, "_shift_hover_annotation", None)
        if annotation is not None:
            try:
                annotation.remove()
            except Exception:
                pass
        self._shift_hover_annotation = None
        self._shift_hover_assignment = None

    def _on_shift_plot_motion(self, event) -> None:
        candidate = self._nearest_shift_candidate(event)
        if candidate is None:
            if self._shift_hover_annotation is not None:
                self._clear_shift_hover_annotation()
                self._shift_canvas.draw_idle()
            return

        identity = (
            candidate["assignment"],
            candidate["layer"],
            candidate["value"],
        )
        if self._shift_hover_assignment == identity:
            return

        self._clear_shift_hover_annotation()
        text = (
            f"{candidate['assignment']}\n"
            f"{candidate['layer']}: {candidate['value']:.4f} ppm"
        )
        self._shift_hover_annotation = self._shift_axes.annotate(
            text,
            xy=(candidate["x"], candidate["y"]),
            xytext=(8, 10),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "alpha": 0.65},
            fontsize="small",
        )
        self._shift_hover_assignment = identity
        self._shift_canvas.draw_idle()

    def _on_shift_plot_leave(self, _event=None) -> None:
        if self._shift_hover_annotation is not None:
            self._clear_shift_hover_annotation()
            self._shift_canvas.draw_idle()

    def _select_shift_assignment(self, assignment: str) -> bool:
        tree = getattr(self, "_shift_tree", None)
        if tree is None:
            return False
        for item in tree.get_children(""):
            values = tree.item(item, "values")
            if values and str(values[0]) == str(assignment):
                tree.selection_set(item)
                tree.focus(item)
                tree.see(item)
                self._on_shift_row_selected()
                return True
        return False

    def _on_shift_plot_click(self, event) -> None:
        if getattr(event, "button", None) != 1:
            return
        candidate = self._nearest_shift_candidate(event, max_pixels=14.0)
        if candidate is None:
            return
        self._select_shift_assignment(candidate["assignment"])

    @staticmethod
    def _gaussian_profile(x, center: float, fwhm: float) -> np.ndarray:
        sigma = float(fwhm) / 2.3548200450309493
        if sigma <= 0:
            return np.zeros_like(x, dtype=float)
        return np.exp(-0.5 * ((x - float(center)) / sigma) ** 2)

    def _plot_broadened_shift_layer(
        self,
        axis,
        *,
        values,
        labels,
        y_base: float,
        height: float,
        legend_label: str,
        selected_assignment: str | None,
    ) -> None:
        finite_values = [
            float(value)
            for value in values
            if value is not None and np.isfinite(float(value))
        ]
        if not finite_values:
            return

        fwhm = self._shift_broadening_value()
        xmin = min(finite_values) - max(0.5, 8 * fwhm)
        xmax = max(finite_values) + max(0.5, 8 * fwhm)
        x = np.linspace(xmin, xmax, 1800)

        total = np.zeros_like(x)
        for value in finite_values:
            total += self._gaussian_profile(x, value, fwhm)
        maximum = float(np.max(total)) if total.size else 0.0
        if maximum > 0:
            total = total / maximum

        axis.plot(
            x,
            y_base + height * total,
            linewidth=1.0,
            label=legend_label,
        )

        for value, assignment in zip(values, labels):
            if value is None:
                continue
            value = float(value)
            y = y_base + height
            self._register_shift_candidate(
                assignment=assignment,
                x=value,
                y=y,
                layer=legend_label,
                value=value,
            )
            if assignment == selected_assignment:
                axis.axvline(
                    value,
                    ymin=0.0,
                    ymax=1.0,
                    linewidth=1.5,
                    alpha=0.45,
                )
            if self._shift_show_labels.get():
                axis.text(
                    value,
                    y_base + height * 1.02,
                    assignment,
                    rotation=90,
                    va="bottom",
                    ha="center",
                    fontsize="x-small",
                )

    def _plot_shift_sticks(
        self,
        axis,
        *,
        y_base: float,
        height: float,
        values,
        labels,
        selected_assignment: str | None,
        legend_label: str,
    ) -> None:
        if self._shift_broaden_sticks.get():
            self._plot_broadened_shift_layer(
                axis,
                values=values,
                labels=labels,
                y_base=y_base,
                height=height,
                legend_label=legend_label,
                selected_assignment=selected_assignment,
            )
            return

        first = True
        for value, assignment in zip(values, labels):
            if value is None:
                continue
            value = float(value)
            linewidth = 2.4 if assignment == selected_assignment else 1.0
            alpha = 1.0 if assignment == selected_assignment else 0.72
            axis.vlines(
                value,
                y_base,
                y_base + height,
                linewidth=linewidth,
                alpha=alpha,
                label=legend_label if first else "_nolegend_",
            )
            self._register_shift_candidate(
                assignment=assignment,
                x=value,
                y=y_base + height,
                layer=legend_label,
                value=value,
            )
            if self._shift_show_labels.get():
                axis.text(
                    value,
                    y_base + height * 1.02,
                    assignment,
                    rotation=90,
                    va="bottom",
                    ha="center",
                    fontsize="x-small",
                )
            first = False

    def _refresh_shift_visualization(self) -> None:
        axes = getattr(self, "_shift_axes", None)
        canvas = getattr(self, "_shift_canvas", None)
        if axes is None or canvas is None:
            return

        view_mode = getattr(self, "_shift_view_mode", None)
        mode = view_mode.get() if view_mode is not None else "Table"
        if mode == "Table":
            return

        self._clear_shift_hover_annotation()
        self._shift_plot_candidates = []
        axes.clear()
        paramagnetic, diamagnetic = self._current_shift_pair()
        if paramagnetic is None or diamagnetic is None:
            axes.text(
                0.5,
                0.5,
                "Pair paramagnetic and diamagnetic datasets first.",
                ha="center",
                va="center",
                transform=axes.transAxes,
            )
            axes.set_axis_off()
            canvas.draw_idle()
            return

        axes.set_axis_on()
        selected_assignment = self._selected_shift_assignment()
        normalize = bool(self._shift_normalize.get())

        para_spec = self._displayable_frequency_spectrum(paramagnetic)
        dia_spec = self._displayable_frequency_spectrum(diamagnetic)

        plotted_spectra = False
        spectrum_layers = {}
        if mode in {"Overlay", "Stacked"}:
            spectra = []
            if self._shift_show_paramag.get() and para_spec is not None:
                spectra.append(
                    (
                        "observed_paramagnetic",
                        "Paramagnetic spectrum",
                        para_spec,
                        self._normalized_real_values(para_spec, normalize),
                    )
                )
            if self._shift_show_diamag.get() and dia_spec is not None:
                spectra.append(
                    (
                        "observed_diamagnetic",
                        "Diamagnetic spectrum",
                        dia_spec,
                        self._normalized_real_values(dia_spec, normalize),
                    )
                )

            if mode == "Stacked" and spectra:
                for index, (kind, label, spectrum, values) in enumerate(spectra):
                    offset = float(len(spectra) - index - 1) * 1.35
                    axes.plot(
                        spectrum.axis,
                        values + offset,
                        linewidth=0.9,
                        label=label,
                    )
                    spectrum_layers[kind] = (spectrum, values, offset)
                plotted_spectra = True
            else:
                for kind, label, spectrum, values in spectra:
                    axes.plot(
                        spectrum.axis,
                        values,
                        linewidth=0.9,
                        label=label,
                    )
                    spectrum_layers[kind] = (spectrum, values, 0.0)
                plotted_spectra = bool(spectra)

        diagnostic_rows = self._shift_diagnostic_rows
        assignments = [row.assignment for row in diagnostic_rows]
        absolute_positions = build_absolute_spectrum_positions(
            diagnostic_rows,
            fixed_corrections_ppm=self._pcs_corrections_ppm,
        )
        diagnostic_quantities = build_shift_diagnostic_quantities(
            diagnostic_rows,
            fixed_corrections_ppm=self._pcs_corrections_ppm,
        )

        if mode in {"Overlay", "Stacked"}:
            layer_names = {
                "observed_paramagnetic": "Observed paramagnetic",
                "observed_diamagnetic": "Observed diamagnetic",
            }
            for position in absolute_positions:
                if position.kind == "predicted_paramagnetic":
                    continue
                layer = spectrum_layers.get(position.kind)
                if layer is None:
                    continue
                spectrum, signal_values, stack_offset = layer
                axis_values = np.asarray(spectrum.axis, dtype=float)
                if axis_values.size == 0:
                    continue
                index = int(
                    np.argmin(
                        np.abs(axis_values - position.position_ppm)
                    )
                )
                y = float(signal_values[index]) + stack_offset
                self._register_shift_candidate(
                    assignment=position.assignment,
                    x=position.position_ppm,
                    y=y,
                    layer=layer_names[position.kind],
                    value=position.position_ppm,
                )

            if self._shift_show_predicted_paramag.get() and plotted_spectra:
                for position in absolute_positions:
                    if position.kind != "predicted_paramagnetic":
                        continue
                    axes.axvline(
                        position.position_ppm,
                        ymin=0.82,
                        ymax=0.98,
                        color="#D97706",
                        linestyle=(0, (3, 2)),
                        linewidth=(
                            2.0
                            if position.assignment == selected_assignment
                            else 1.1
                        ),
                        alpha=0.9,
                    )
                    y0, y1 = axes.get_ylim()
                    y = y0 + 0.9 * (y1 - y0)
                    self._register_shift_candidate(
                        assignment=position.assignment,
                        x=position.position_ppm,
                        y=y,
                        layer="Predicted paramagnetic",
                        value=position.position_ppm,
                    )
                    if self._shift_show_labels.get():
                        axes.text(
                            position.position_ppm,
                            y,
                            position.assignment,
                            rotation=90,
                            va="top",
                            ha="center",
                            fontsize="x-small",
                            color="#A65300",
                        )

        if mode in {"Shift Diagnostics", "Sticks"}:
            y_base = 0.0
            diagnostic_layers = (
                (
                    "delta_para_observed",
                    self._shift_show_delta_para,
                    "δpara(observed)",
                ),
                (
                    "pcs_calculated",
                    self._shift_show_pcs_calc,
                    "PCS(calc)",
                ),
                (
                    "fixed_correction",
                    self._shift_show_fixed_correction,
                    "Fixed correction",
                ),
                (
                    "implied_non_pcs",
                    self._shift_show_non_pcs,
                    "Implied non-PCS = δpara(obs) − PCS(calc)",
                ),
            )
            quantity_by_key = {
                (quantity.assignment, quantity.kind): quantity.value_ppm
                for quantity in diagnostic_quantities
            }
            for kind, visible, label in diagnostic_layers:
                if not visible.get():
                    continue
                values = [
                    quantity_by_key.get((assignment, kind))
                    for assignment in assignments
                ]
                if not any(value is not None for value in values):
                    continue
                self._plot_shift_sticks(
                    axes,
                    y_base=y_base,
                    height=0.9,
                    values=values,
                    labels=assignments,
                    selected_assignment=selected_assignment,
                    legend_label=label,
                )
                y_base += 1.15
            axes.set_ylim(-0.1, max(y_base, 1.0))

        elif mode in {"Overlay", "Stacked"}:
            if not plotted_spectra:
                axes.text(
                    0.5,
                    0.5,
                    "Processed ppm spectra are unavailable.\n"
                    "Use Shift Diagnostics for difference-space quantities.",
                    ha="center",
                    va="center",
                    transform=axes.transAxes,
                )

        difference_mode = mode in {"Shift Diagnostics", "Sticks"}
        axes.set_xlabel(
            "Shift difference / ppm"
            if difference_mode
            else "Absolute chemical shift / ppm"
        )
        axes.set_ylabel(
            "Normalized intensity"
            if normalize and not difference_mode
            else ("Intensity" if not difference_mode else "Diagnostic layer")
        )

        if not difference_mode:
            # Conventional NMR display applies only to absolute spectra.
            xlim = axes.get_xlim()
            if xlim[0] < xlim[1]:
                axes.invert_xaxis()

        handles, labels = axes.get_legend_handles_labels()
        if handles:
            axes.legend(loc="best", fontsize="small")

        axes.set_title(
            {
                "Overlay": "Paired spectra overlay",
                "Stacked": "Paired spectra stacked",
                "Shift Diagnostics": "Shift Diagnostics (difference space)",
                "Sticks": "Shift Diagnostics (difference space)",
            }.get(mode, "Shift Analysis")
        )

        if difference_mode:
            note = (
                "Difference-space diagnostics only: δpara(observed), PCS(calc), "
                "fixed correction, and implied non-PCS. PCS(calc) is not "
                "PCS(exp) and is not an absolute chemical shift."
            )
        else:
            note = (
                "Absolute ppm only: observed paramagnetic, observed diamagnetic, "
                "and predicted paramagnetic positions. Hover or click an "
                "assignment marker to follow its identity."
            )
        self._shift_plot_note.set(note)
        canvas.draw_idle()

    def _build_statusbar(self) -> None:
        status = ttk.Frame(self, padding=(_CONTROL_GAP, _HEADER_GAP))
        status.pack(fill="x")

        ttk.Label(status, textvariable=self._coordinate).pack(side="left")

        ttk.Label(
            status,
            textvariable=self._interaction_mode_text,
            style="NMR.Mode.TLabel",
            font=get_app_fonts(self)["ui_small_bold"],
        ).pack(side="left", padx=(16, 0))

        ttk.Label(
            status,
            text="Wheel Height   Shift+Wheel X Zoom   Z Drag Zoom   F Full   H Fit   K Peaks   L Ref",
            style="NMR.Muted.TLabel",
            font=get_app_fonts(self)["ui_small"],
        ).pack(side="right")

    def _interaction_mode_label(self) -> str:
        if getattr(self, "_phase_pivot_pick", False) and self._spectrum_interaction.mode == InteractionMode.NAVIGATE:
            return "Phase pivot"
        labels = {
            InteractionMode.NAVIGATE: "Navigate",
            InteractionMode.ZOOM: "Zoom",
            InteractionMode.PEAK_PICK: "Peak Pick",
            InteractionMode.REFERENCE_PEAK: "Reference",
        }
        return labels.get(self._spectrum_interaction.mode, "Navigate")

    def _sync_interaction_mode_ui(self) -> None:
        if self._spectrum_interaction.mode != InteractionMode.NAVIGATE:
            self._phase_pivot_pick = False
        self._interaction_mode_text.set(
            f"Mode: {self._interaction_mode_label()}"
        )
        self._update_processing_controls()

    def _enter_navigate_mode(self, *, message: str | None = None) -> None:
        self._phase_pivot_pick = False
        self._spectrum_interaction.enter_navigate_mode()
        adapter = getattr(self, "_mpl_interaction", None)
        if adapter is not None:
            adapter.cancel_drag()
            adapter.clear_hover_artists()
        if message is not None:
            self._coordinate.set(message)
        self._sync_interaction_mode_ui()

    def _toggle_zoom_mode(self) -> None:
        if self._stacked_series_visible():
            self._stacked_view.toolbar.zoom()
            return
        if self._spectrum_interaction.mode == InteractionMode.ZOOM:
            self._enter_navigate_mode(message="Navigate")
            return

        spectrum = self._current_display_spectrum()
        if spectrum is None or self._view_choice.get() != "Spectrum":
            return
        if self.processing_session.has_preview:
            self._show_processing_error(
                ValueError("Apply or cancel the current processing preview first.")
            )
            return

        self._mpl_interaction.clear_hover_artists()
        self._spectrum_interaction.enter_zoom_mode()
        self._coordinate.set("Zoom mode — drag horizontally across the spectrum")
        self._sync_interaction_mode_ui()

    def _on_spectrum_view_changed(self) -> None:
        # Renderer callback after drag-pan / wheel / drag-zoom.
        # Keep this lightweight; no processing or peak-cache rebuild is needed.
        self._coordinate.set(
            f"{self._interaction_mode_label()} · "
            f"x-range {self._axes.get_xlim()[0]:.6g} to {self._axes.get_xlim()[1]:.6g}"
        )

    def _shortcut_target_is_editable(self, widget) -> bool:
        """Do not steal single-letter shortcuts while the user edits text."""
        editable_classes = {
            "Entry",
            "TEntry",
            "Text",
            "TCombobox",
            "Spinbox",
            "TSpinbox",
        }
        try:
            widget_class = widget.winfo_class()
        except Exception:
            return False
        return widget_class in editable_classes

    def _on_workbench_key(self, event) -> str | None:
        if self._shortcut_target_is_editable(getattr(event, "widget", None)):
            return None

        key = str(getattr(event, "keysym", "") or "").lower()

        if self._stacked_series_visible():
            if key == "f":
                self._reset_view()
            elif key == "h":
                self._auto_scale_y()
            elif key == "z":
                self._toggle_zoom_mode()
            elif key == "escape":
                toolbar = self._stacked_view.toolbar
                if toolbar.mode.name == "PAN":
                    toolbar.pan()
                elif toolbar.mode.name == "ZOOM":
                    toolbar.zoom()
            else:
                return None
            return "break"

        if key == "z":
            self._toggle_zoom_mode()
        elif key == "f":
            self._reset_view()
            self._coordinate.set("Full spectrum")
        elif key == "h":
            self._auto_scale_y()
            self._coordinate.set("Height fitted to visible spectrum")
        elif key == "k":
            if self._spectrum_interaction.mode == InteractionMode.PEAK_PICK:
                self._stop_peak_picking()
            else:
                self._begin_peak_picking()
        elif key == "l":
            if self._spectrum_interaction.mode == InteractionMode.REFERENCE_PEAK:
                self._enter_navigate_mode(message="Reference selection cancelled")
            else:
                self._begin_reference_peak_selection()
        elif key == "escape":
            self._enter_navigate_mode(message="Navigate")
        else:
            return None

        return "break"

    def _open_dataset_folder(self) -> None:
        path = filedialog.askdirectory(title="Select an NMR experiment or parent folder")
        self._load_selected_path(path)

    def _open_fid_file(self) -> None:
        path = filedialog.askopenfilenames(
            title="Select raw NMR fid files (multiple allowed)",
            filetypes=(("Raw NMR FID", "fid"), ("All files", "*")),
        )
        self._load_selected_path(path)

    def _load_selected_path(self, path: str) -> None:
        if not path:
            return
        try:
            self._start_experiment_import((path,) if isinstance(path,str) else path)
        except Exception as exc:
            messagebox.showerror("NMR Workbench", str(exc), parent=self)

    def _insert_dataset(self, dataset) -> None:
        parent_item = self._experiment_tree.insert(
            "", "end", text=dataset_display_name(dataset), open=True
        )
        detail_item = self._experiment_tree.insert(
            parent_item, "end", text=dataset_summary(dataset)
        )
        self._tree_dataset_ids[parent_item] = dataset.id
        self._tree_dataset_ids[detail_item] = dataset.id
        self._experiment_tree.selection_set(parent_item)
        self._experiment_tree.focus(parent_item)
        if hasattr(self, "_dataset_role_box"):
            self._refresh_dataset_role_controls()
        self._refresh_series_manager()

    def _on_tree_selection(self, _event=None) -> None:
        if getattr(self, "_series_tree_refresh", False):
            return
        ids = self._selection_experiment_ids()
        if len(ids) > 1:
            self._cancel_phase_timer()
        if len(ids) == 1 and (ids[0] != self.model.selected_dataset_id or self.processing_session.dataset is not self.model.current_dataset):
            self._select_dataset(ids[0])
        self._update_processing_controls()

    def _select_dataset(self, dataset_id: str) -> None:
        self._cancel_phase_timer()
        self._reset_spectrum_interaction()
        dataset = self.model.select_dataset(dataset_id)
        try:
            self.processing_session.select_dataset(dataset)
            if (
                dataset.raw_spectrum is not None
                and dataset.processed_spectrum is None
                and not dataset.processing_history.steps
            ):
                self.processing_session.initialize_default_spectrum()
        except ProcessingSessionError as exc:
            messagebox.showerror("NMR Workbench", str(exc), parent=self)
        choices = self._available_view_choices()
        self._view_selector.configure(
            values=choices, state="readonly" if choices else "disabled"
        )
        self._view_choice.set(
            "Spectrum" if "Spectrum" in choices else (choices[0] if choices else "")
        )
        suffix = "Spectrum" if dataset.processed_spectrum is not None else "Raw FID" if dataset.raw_spectrum is not None else "Observed shifts"
        self._dataset_text.set(
            f"{dataset_display_name(dataset)} · {dataset_summary(dataset)} · {suffix}"
        )
        self._update_zero_fill_choices()
        self._initialize_phase_controls()

        # Existing project/dataset observables are the persistent source used
        # to restore assignment drafts when revisiting the dataset.
        self._assignment_drafts_by_dataset[dataset.id] = (
            assignments_from_observables(dataset)
        )
        self._refresh_dataset_role_controls()
        self._refresh_shift_analysis()
        self._refresh()

        self._refresh_series_manager()
        self._selection_experiment_ids()
        for item in self._experiment_tree.get_children():
            self._experiment_tree.item(item, tags=("active",) if self._tree_dataset_ids.get(item)==dataset.id else ())

    def _on_view_selection(self, _event=None) -> None:
        if self._view_choice.get() == "Raw FID":
            self._cancel_phase_timer()
            self._reset_spectrum_interaction()
            if self.processing_session.has_preview:
                phase_preview = self.processing_session.preview.operation == "phase"
                self.processing_session.cancel_preview()
                if phase_preview:
                    self._set_phase_values(0.0, 0.0)
        self._refresh(preserve_view=True)

    def _raw_processing_action(self) -> None:
        spectrum = self.processing_session.committed_spectrum
        try:
            if spectrum is not None and spectrum.domain == "frequency":
                self.processing_session.start_reprocessing_from_raw()
                self._view_choice.set("Raw FID")
                self._update_zero_fill_choices()
                self._initialize_phase_controls()
                self._refresh()
            else:
                self._apply_fourier_transform()
        except ProcessingSessionError as exc:
            self._show_processing_error(exc)

    def _refresh(self, *, preserve_view: bool = False) -> None:
        self.model.update_tracks()
        self._refresh_stacked_series()
        dataset = self.current_dataset
        if dataset is None:
            self._draw_empty_plot()
            return
        spectrum = self._current_display_spectrum()
        self._draw_spectrum(spectrum, preserve_view=preserve_view)
        visible_state = (
            "Observed shifts" if spectrum is None and dataset.observables else "Raw FID" if self._view_choice.get() == "Raw FID" else "Spectrum"
        )
        self._dataset_text.set(
            f"{dataset_display_name(dataset)} · {dataset_summary(dataset)} · "
            f"{visible_state}"
        )
        self._state_text.set(
            processing_display_state_label(
                spectrum,
                preview=self.processing_session.has_preview,
                committed=(
                    spectrum is self.processing_session.committed_spectrum
                    and bool(dataset.processing_history.steps)
                ),
            )
        )
        self._refresh_history(dataset)
        self._refresh_metadata(dataset)
        self._update_processing_controls()
        self._update_pivot_marker()
        self._refresh_interaction_peak_cache()
        self._refresh_peak_list()
        self._refresh_peak_markers()

    def _current_display_spectrum(self):
        session = self.processing_session
        if session.has_preview:
            return session.preview_spectrum
        dataset = self.current_dataset
        if dataset is None:
            return None
        if self._view_choice.get() == "Raw FID":
            return dataset.raw_spectrum
        return session.committed_spectrum

    def _available_view_choices(self) -> tuple[str, ...]:
        dataset = self.current_dataset
        if dataset is None:
            return ()
        choices: list[str] = []
        committed = self.processing_session.committed_spectrum
        if committed is not None and committed.domain == "frequency":
            choices.append("Spectrum")
        if dataset.raw_spectrum is not None:
            choices.append("Raw FID")
        return tuple(choices)

    def _on_apodization_method(self, _event=None) -> None:
        gaussian = self._apod_method.get().strip().lower() == "gaussian"
        self._apod_parameter_label.set("GB (Hz)" if gaussian else "LB (Hz)")

    def _toggle_subsection(
        self, frame: ttk.Frame, button: ttk.Button, title: str
    ) -> None:
        if frame.winfo_manager():
            frame.pack_forget()
            button.configure(text=f"▸  {title}")
        else:
            frame.pack(fill="x", pady=(4, 0))
            button.configure(text=f"▾  {title}")

    def _on_baseline_method(self, _event=None) -> None:
        self._bernstein_parameters.grid_forget()
        self._asls_parameters.grid_forget()
        self._arpls_parameters.grid_forget()

        method = self._baseline_method.get()
        if method == "Bernstein":
            self._bernstein_parameters.grid(
                row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0)
            )
        elif method == "arPLS":
            self._arpls_parameters.grid(
                row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0)
            )
        else:
            self._asls_parameters.grid(
                row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0)
            )

    def _on_phase_slider(self, kind: str, value: str) -> None:
        if self._phase_syncing:
            return
        number = float(value)
        target = self._p0_text if kind == "p0" else self._p1_text
        target.set(f"{number:.2f}")
        self._queue_phase_preview()

    def _change_phase_angle(self, kind, value):
        if not np.isfinite(value):
            raise ValueError("Phase angle must be finite.")
        p0 = value if kind == "p0" else float(self._p0_text.get())
        p1 = value if kind == "p1" else float(self._p1_text.get())
        if not np.isfinite(p0) or not np.isfinite(p1):
            raise ValueError("Phase angles must be finite.")
        self._set_phase_values(p0, p1)
        self._queue_phase_preview()

    def _on_phase_entry(self, kind: str) -> None:
        variable = self._p0_text if kind == "p0" else self._p1_text
        try:
            self._change_phase_angle(kind, float(variable.get()))
        except ValueError as exc:
            self._show_processing_error(exc)

    def _change_phase_pivot(self, pivot):
        from ..processing import ppm_to_fractional_index
        from ..processing.phasing import rebase_phase_pivot
        spectrum = self.processing_session.committed_spectrum
        old = getattr(self, "_phase_pivot_coordinate", None)
        if spectrum is not None and old is not None:
            to_index = (lambda value: ppm_to_fractional_index(spectrum, value)) if spectrum.axis_unit.lower() == "ppm" else float
            p0 = rebase_phase_pivot(float(self._p0_text.get()), float(self._p1_text.get()),
                                    to_index(old), to_index(pivot), spectrum.point_count)
            self._set_phase_values(p0, float(self._p1_text.get()))
        self._phase_pivot_coordinate = pivot
        self._pivot_text.set(f"{pivot:.12g}")

    def _on_pivot_commit(self, _event=None) -> None:
        try:
            pivot = float(self._pivot_text.get())
            self._change_phase_pivot(pivot)
        except (ValueError, RuntimeError) as exc:
            self._show_processing_error(exc)
            return
        self._phase_syncing = True
        try:
            self._pivot_slider_value.set(pivot)
        finally:
            self._phase_syncing = False
        self._update_pivot_marker()
        self._queue_phase_preview()

    def _on_pivot_slider(self, value: str) -> None:
        if self._phase_syncing:
            return
        pivot = float(value)
        try:
            self._change_phase_pivot(pivot)
        except (ValueError, RuntimeError) as exc:
            self._show_processing_error(exc)
            return
        self._update_pivot_marker()
        self._queue_phase_preview()

    def _queue_phase_preview(self) -> None:
        if len(self._selection_experiment_ids()) > 1:
            self._cancel_phase_timer()
            self._experiment_status.set(f"{len(self.experiments.selected_ids)} experiments selected · Batch Processing… previews phase")
            return
        if (
            not self.processing_session.can_phase
            or self._view_choice.get() != "Spectrum"
        ):
            return
        self._cancel_phase_timer()
        self._phase_after_id = self.after(35, self._run_phase_preview)

    def _cancel_phase_timer(self) -> None:
        if getattr(self, "_phase_after_id", None) is not None:
            try:
                self.after_cancel(self._phase_after_id)
            except tk.TclError:
                pass
            self._phase_after_id = None

    def _run_phase_preview(self) -> None:
        self._phase_after_id = None
        if len(self._selection_experiment_ids()) > 1:
            return
        spectrum = self.processing_session.committed_spectrum
        if spectrum is None:
            return
        try:
            kwargs = {
                "p0_deg": self._p0_text.get(),
                "p1_deg": self._p1_text.get(),
            }
            if spectrum.axis_unit.lower() == "ppm":
                kwargs["pivot_ppm"] = self._pivot_text.get()
            else:
                kwargs["pivot_index"] = self._pivot_text.get()
            self.processing_session.preview_phase(**kwargs)
        except ProcessingSessionError as exc:
            self._show_processing_error(exc)
            return
        self._view_choice.set("Spectrum")
        self._view_selector.configure(
            values=self._available_view_choices(), state="readonly"
        )
        self._refresh_phase_preview()

    def _refresh_phase_preview(self) -> None:
        spectrum = self.processing_session.preview_spectrum
        signature = None if spectrum is None else (spectrum.domain, spectrum.axis_unit)
        line = getattr(self, "_spectrum_line", None)
        if (
            spectrum is None
            or line is None
            or getattr(self, "_display_signature", None) != signature
            or len(line.get_ydata()) != spectrum.point_count
        ):
            self._refresh(preserve_view=True)
            return
        line.set_ydata(np.real(spectrum.intensities))
        self._state_text.set(
            processing_display_state_label(spectrum, preview=True, committed=False)
        )
        self._update_processing_controls()
        self._canvas.draw_idle()

    def _run_auto_phase(self) -> None:
        algorithm = (
            "peak_minima"
            if self._auto_phase_algorithm.get() == "Peak minima"
            else "acme"
        )
        if self._dispatch_experiment_batch("phase", {"mode": "auto", "pivot_mode": "biggest_peak", "algorithm": algorithm}):
            return
        spectrum = self.processing_session.committed_spectrum
        if spectrum is None:
            return
        try:
            if spectrum.axis_unit.lower() == "ppm":
                estimate = self.processing_session.preview_auto_phase(
                    algorithm=algorithm, pivot_ppm=self._pivot_text.get()
                )
            else:
                estimate = self.processing_session.preview_auto_phase(
                    algorithm=algorithm, pivot_index=self._pivot_text.get()
                )
        except ProcessingSessionError as exc:
            self._show_processing_error(exc)
            return
        self._set_phase_values(estimate.p0_deg, estimate.p1_deg)
        self._view_choice.set("Spectrum")
        self._refresh_phase_preview()
        if estimate.diagnostics:
            self._state_text.set(self._state_text.get() + " · " + "; ".join(estimate.diagnostics))

    def _phase_biggest_peak(self) -> None:
        try:
            pivot = self.processing_session.biggest_peak_pivot()
            self._change_phase_pivot(pivot)
        except (ValueError, RuntimeError) as exc:
            self._show_processing_error(exc)
            return
        self._pivot_text.set(f"{pivot:.12g}")
        self._phase_pivot_coordinate = pivot
        self._phase_syncing = True
        try:
            self._pivot_slider_value.set(pivot)
        finally:
            self._phase_syncing = False
        self._update_pivot_marker()
        self._queue_phase_preview()

    def _phase_reset(self) -> None:
        self._set_phase_values(0.0, 0.0)
        self._queue_phase_preview()

    def _phase_flip(self) -> None:
        try:
            p0, p1 = float(self._p0_text.get()), float(self._p1_text.get())
            if not np.isfinite(p0) or not np.isfinite(p1):
                raise ValueError("Phase angles must be finite.")
            flipped = (p0 % 360.) - 180.
            self._set_phase_values(180. if flipped == -180. else flipped, p1)
            self._queue_phase_preview()
        except ValueError as exc:
            self._show_processing_error(exc)

    def _phase_pick_pivot(self) -> None:
        if str(self._phase_pick_button.cget("state")) == "disabled":
            return
        self._spectrum_tabs.select(0)
        self._enter_navigate_mode()
        self._phase_pivot_pick = True
        toolbar = getattr(self._canvas, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            if toolbar.mode.name == "PAN":
                toolbar.pan()
            elif toolbar.mode.name == "ZOOM":
                toolbar.zoom()
        self._sync_interaction_mode_ui()
        self._coordinate.set("Click the phase pivot on the spectrum; Escape cancels")

    def _set_phase_values(self, p0: float, p1: float) -> None:
        self._phase_syncing = True
        try:
            self._p0_text.set(f"{p0:.12g}")
            self._p1_text.set(f"{p1:.12g}")
            self._p0_slider_value.set(p0)
            self._p1_slider_value.set(p1)
        finally:
            self._phase_syncing = False

    def _initialize_phase_controls(self) -> None:
        self._phase_pivot_pick = False
        for drag in self._phase_drags:
            drag.cancel_drag()
        self._cancel_phase_timer()
        spectrum = self.processing_session.committed_spectrum
        self._set_phase_values(0.0, 0.0)
        self._phase_pivot_coordinate = None
        if spectrum is None or spectrum.domain != "frequency":
            self._pivot_text.set("")
            self._update_pivot_marker()
            return
        if spectrum.axis_unit.lower() == "ppm":
            self._pivot_label.set("Pivot (ppm)")
            pivot = (float(spectrum.axis[0]) + float(spectrum.axis[-1])) / 2.0
            self._pivot_slider.configure(
                from_=float(spectrum.axis[-1]), to=float(spectrum.axis[0])
            )
        else:
            self._pivot_label.set("Pivot (index)")
            pivot = (spectrum.point_count - 1) / 2.0
            self._pivot_slider.configure(from_=0.0, to=spectrum.point_count - 1)
        self._pivot_text.set(f"{pivot:.12g}")
        self._phase_pivot_coordinate = pivot
        self._phase_syncing = True
        try:
            self._pivot_slider_value.set(pivot)
        finally:
            self._phase_syncing = False
        self._update_pivot_marker()

    def _update_pivot_marker(self) -> None:
        marker = getattr(self, "_pivot_marker", None)
        if (
            not self.processing_session.can_phase
            or self._view_choice.get() != "Spectrum"
        ):
            if marker is not None:
                try:
                    marker.remove()
                except ValueError:
                    pass
            self._pivot_marker = None
            return
        try:
            pivot = float(self._pivot_text.get())
        except ValueError:
            return
        xlim, ylim = self._axes.get_xlim(), self._axes.get_ylim()
        if marker is None:
            self._pivot_marker = self._axes.axvline(
                pivot,
                color="#7A3E9D",
                linestyle="--",
                linewidth=0.9,
                label="Phase pivot",
            )
        else:
            marker.set_xdata([pivot, pivot])
            marker.set_visible(True)
        self._axes.set_xlim(xlim)
        self._axes.set_ylim(ylim)
        self._canvas.draw_idle()

    def _preview_apodization(self) -> None:
        if self._dispatch_experiment_batch("apodization", {"method": self._apod_method.get().lower(), "lb_hz": self._apod_value.get() if self._apod_method.get().lower() != "gaussian" else None, "gb_hz": self._apod_value.get() if self._apod_method.get().lower() == "gaussian" else None}):
            return
        method = self._apod_method.get().strip().lower()
        try:
            if self._view_choice.get() == "Spectrum":
                if method == "gaussian":
                    self.processing_session.preview_apodization_from_spectrum(
                        method=method,
                        gb_hz=self._apod_value.get(),
                    )
                else:
                    self.processing_session.preview_apodization_from_spectrum(
                        method=method,
                        lb_hz=self._apod_value.get(),
                    )
                self._view_choice.set("Spectrum")
            else:
                if method == "gaussian":
                    self.processing_session.preview_apodization(
                        method=method,
                        gb_hz=self._apod_value.get(),
                    )
                else:
                    self.processing_session.preview_apodization(
                        method=method,
                        lb_hz=self._apod_value.get(),
                    )
                self._view_choice.set("Raw FID")
        except ProcessingSessionError as exc:
            self._show_processing_error(exc)
            return

        self._view_selector.configure(
            values=self._available_view_choices(),
            state="readonly",
        )
        self._refresh(preserve_view=True)

    def _preview_zero_fill(self) -> None:
        if self._dispatch_experiment_batch("zero_fill", {"target_points": self._zero_target.get()}):
            return
        try:
            if self._view_choice.get() == "Spectrum":
                self.processing_session.preview_zero_fill_from_spectrum(
                    self._zero_target.get()
                )
                self._view_choice.set("Spectrum")
            else:
                self.processing_session.preview_zero_fill(
                    self._zero_target.get()
                )
                self._view_choice.set("Raw FID")
        except ProcessingSessionError as exc:
            self._show_processing_error(exc)
            return

        self._view_selector.configure(
            values=self._available_view_choices(),
            state="readonly",
        )
        self._refresh(preserve_view=True)

    def _preview_baseline(self) -> None:
        if self._dispatch_experiment_batch("baseline", {"method": {"Bernstein":"bernstein", "arPLS":"arpls"}.get(self._baseline_method.get(),"asls"), "order": self._baseline_order.get(), "lam": self._baseline_lambda.get(), "p": self._baseline_p.get()}):
            return
        try:
            method = self._baseline_method.get()
            if method == "Bernstein":
                self.processing_session.preview_baseline_correction(
                    method="bernstein",
                    order=self._baseline_order.get(),
                )
            elif method == "arPLS":
                self.processing_session.preview_baseline_correction(
                    method="arpls",
                    lam=self._baseline_lambda.get(),
                )
            else:
                self.processing_session.preview_baseline_correction(
                    method="asls",
                    lam=self._baseline_lambda.get(),
                    p=self._baseline_p.get(),
                )
        except ProcessingSessionError as exc:
            self._show_processing_error(exc)
            return
        self._view_choice.set("Spectrum")
        self._view_selector.configure(
            values=self._available_view_choices(), state="readonly"
        )
        self._refresh_baseline_preview()

    def _preview_auto_baseline(self) -> None:
        if self._dispatch_experiment_batch("baseline"):
            return
        try:
            self.processing_session.preview_auto_baseline()
        except ProcessingSessionError as exc:
            self._show_processing_error(exc)
            return
        self._view_choice.set("Spectrum")
        self._view_selector.configure(
            values=self._available_view_choices(), state="readonly"
        )
        self._refresh_baseline_preview()

    def _refresh_baseline_preview(self) -> None:
        spectrum = self.processing_session.preview_spectrum
        baseline = self.processing_session.preview_baseline
        line = getattr(self, "_spectrum_line", None)
        signature = None if spectrum is None else (spectrum.domain, spectrum.axis_unit)
        if (
            spectrum is None
            or baseline is None
            or line is None
            or getattr(self, "_display_signature", None) != signature
            or len(line.get_ydata()) != spectrum.point_count
        ):
            self._refresh(preserve_view=True)
        else:
            line.set_ydata(np.real(spectrum.intensities))
            self._state_text.set(
                processing_display_state_label(spectrum, preview=True, committed=False)
            )
            self._refresh_history(self.current_dataset)
            self._update_processing_controls()
        self._refresh_baseline_overlays()

    def _refresh_baseline_overlays(self) -> None:
        baseline = self.processing_session.preview_baseline
        committed = self.processing_session.committed_spectrum
        if baseline is None or committed is None or self._spectrum_line is None:
            self._clear_baseline_overlays()
            self._canvas.draw_idle()
            return
        self._spectrum_line.set_label("Corrected preview")
        if self._baseline_line is None:
            (self._baseline_line,) = self._axes.plot(
                committed.axis,
                baseline,
                linestyle="--",
                linewidth=1.0,
                color="#B24A33",
                label="Fitted baseline",
            )
        else:
            self._baseline_line.set_data(committed.axis, baseline)
        self._baseline_line.set_visible(self._show_fitted_baseline.get())
        original = np.real(committed.intensities)
        if self._baseline_original_line is None:
            (self._baseline_original_line,) = self._axes.plot(
                committed.axis,
                original,
                linestyle=":",
                linewidth=0.9,
                color="0.4",
                label="Committed spectrum",
            )
        else:
            self._baseline_original_line.set_data(committed.axis, original)
        self._baseline_original_line.set_visible(self._show_baseline_original.get())
        self._axes.legend(loc="upper right")
        self._canvas.draw_idle()

    def _clear_baseline_overlays(self) -> None:
        for attribute in ("_baseline_line", "_baseline_original_line"):
            line = getattr(self, attribute, None)
            if line is not None:
                try:
                    line.remove()
                except ValueError:
                    pass
            setattr(self, attribute, None)
        if getattr(self, "_spectrum_line", None) is not None:
            self._spectrum_line.set_label("_nolegend_")
        legend = self._axes.get_legend()
        if legend is not None:
            legend.remove()

    def _apply_preview(self) -> None:
        operation = self.processing_session.preview.operation if self.processing_session.preview else "phase"
        if self._dispatch_experiment_batch(operation, {"p0_deg": self._p0_text.get(), "p1_deg": self._p1_text.get(), "pivot_ppm": self._pivot_text.get()} if operation=="phase" else None):
            return
        operation = (
            self.processing_session.preview.operation
            if self.processing_session.preview is not None
            else None
        )
        self._cancel_phase_timer()
        try:
            self.processing_session.apply_preview()
        except ProcessingSessionError as exc:
            self._show_processing_error(exc)
            return
        if operation == "phase":
            self._set_phase_values(0.0, 0.0)

        if operation is not None:
            applied_offset = (
                self._reference_offset.get()
                if operation == "reference"
                and hasattr(self, "_reference_offset")
                else ""
            )
            if operation == "reference" and self.current_dataset is not None:
                try:
                    shift = float(applied_offset)
                except (TypeError, ValueError):
                    shift = 0.0
                if shift:
                    for peak in self.current_dataset.peaks:
                        peak.position_ppm += shift
                    try:
                        sync_dataset_observables(
                            self.current_dataset,
                            self._assignment_map(),
                        )
                    except ObservableSyncError as exc:
                        if hasattr(self, "_assignment_warning_text"):
                            self._assignment_warning_text.set(str(exc))
            self._reset_spectrum_interaction()
            if operation == "reference" and hasattr(self, "_reference_status"):
                self._reference_status.set(
                    f"Reference applied ({applied_offset} ppm)."
                    if applied_offset
                    else "Reference applied."
                )

        committed = self.processing_session.committed_spectrum
        self._view_choice.set(
            "Spectrum"
            if committed is not None and committed.domain == "frequency"
            else "Raw FID"
        )
        self._view_selector.configure(
            values=self._available_view_choices(), state="readonly"
        )
        self._update_zero_fill_choices()
        self._refresh(preserve_view=True)

    def _cancel_preview(self) -> None:
        operation = (
            self.processing_session.preview.operation
            if self.processing_session.preview is not None
            else None
        )
        self._cancel_phase_timer()
        self.processing_session.cancel_preview()
        if operation == "phase":
            self._set_phase_values(0.0, 0.0)
        choices = self._available_view_choices()
        self._view_choice.set(
            "Spectrum" if "Spectrum" in choices else (choices[0] if choices else "")
        )
        self._view_selector.configure(
            values=choices, state="readonly" if choices else "disabled"
        )
        self._refresh(preserve_view=True)

    def _reset_spectrum_interaction(self) -> None:
        """Clear transient hover/selection state and cached peak candidates."""
        self._phase_pivot_pick = False
        for drag in getattr(self, "_phase_drags", ()):
            drag.cancel_drag()
        adapter = getattr(self, "_mpl_interaction", None)
        if adapter is not None:
            adapter.clear_hover_artists()
            adapter.clear_selection_artists()
        self._spectrum_interaction.reset()
        self._interaction_cache_source_id = None
        if hasattr(self, "_interaction_mode_text"):
            self._interaction_mode_text.set("Mode: Navigate")

        if hasattr(self, "_reference_observed"):
            self._reference_observed.set("")
        if hasattr(self, "_reference_status"):
            self._reference_status.set("")

        if hasattr(self, "_reference_preset"):
            self._reference_preset.set("")
        if hasattr(self, "_reference_signal"):
            self._reference_signal.set("")
        if hasattr(self, "_reference_target"):
            self._reference_target.set("")
        if hasattr(self, "_reference_offset"):
            self._reference_offset.set("")
        if hasattr(self, "_reference_type_selector"):
            self._reference_type_selector.configure(state="disabled")
        if hasattr(self, "_reference_preset_selector"):
            self._reference_preset_selector.configure(values=(), state="disabled")
        if hasattr(self, "_reference_signal_selector"):
            self._reference_signal_selector.configure(values=(), state="disabled")
        if hasattr(self, "_reference_target_entry"):
            self._reference_target_entry.configure(state="disabled")

        self._reference_preset_by_display.clear()
        self._reference_resonance_by_display.clear()
        if hasattr(self, "_peak_pick_status"):
            self._peak_pick_status.set("Ready to pick peaks.")

    def _refresh_interaction_peak_cache(self) -> None:
        """Rebuild lightweight hover candidates only for a new committed ppm spectrum.

        Preview redraws deliberately keep the existing cache.  This prevents
        expensive ``find_peaks`` work while a phase slider is moving.
        """
        if self._view_choice.get() != "Spectrum":
            self._spectrum_interaction.peak_cache.clear()
            self._interaction_cache_source_id = None
            return

        spectrum = self.processing_session.committed_spectrum
        if (
            spectrum is None
            or spectrum.domain != "frequency"
            or spectrum.axis_unit.lower() != "ppm"
        ):
            self._spectrum_interaction.peak_cache.clear()
            self._interaction_cache_source_id = None
            return

        source_id = id(spectrum)
        if self._interaction_cache_source_id == source_id:
            return

        axis = np.asarray(spectrum.axis, dtype=float)
        intensities = np.asarray(spectrum.intensities)
        if axis.ndim != 1 or intensities.ndim != 1:
            self._spectrum_interaction.peak_cache.clear()
            self._interaction_cache_source_id = None
            return
        if axis.size != intensities.size or axis.size < 3:
            self._spectrum_interaction.peak_cache.clear()
            self._interaction_cache_source_id = None
            return

        real_values = np.real(intensities)
        finite_values = real_values[np.isfinite(real_values)]
        if finite_values.size == 0:
            self._spectrum_interaction.peak_cache.clear()
            self._interaction_cache_source_id = None
            return

        # Interaction-only noise suppression.  This is intentionally not a
        # scientific peak-picking threshold.
        amplitude = float(np.nanmax(finite_values) - np.nanmin(finite_values))
        prominence = amplitude * 0.002 if np.isfinite(amplitude) and amplitude > 0 else None

        self._spectrum_interaction.peak_cache.rebuild(
            axis,
            intensities,
            prominence=prominence,
        )
        self._interaction_cache_source_id = source_id

    def _peak_picking_available(self) -> bool:
        spectrum = self.processing_session.committed_spectrum
        return (
            spectrum is not None
            and spectrum.domain == "frequency"
            and spectrum.axis_unit.lower() == "ppm"
            and self._view_choice.get() == "Spectrum"
            and not self.processing_session.has_preview
        )

    def _begin_peak_picking(self) -> None:
        if not self._peak_picking_available():
            self._show_processing_error(
                ValueError("Peak picking requires a committed ppm-domain spectrum.")
            )
            return
        self._refresh_interaction_peak_cache()
        if not self._spectrum_interaction.peak_cache.candidates:
            self._show_processing_error(
                ValueError("No suitable peak candidates were found in the current spectrum.")
            )
            return
        self._mpl_interaction.cancel_drag()
        self._mpl_interaction.clear_selection_artists()
        self._spectrum_interaction.enter_peak_pick_mode()
        self._interaction_mode_text.set("Mode: Peak Pick")
        self._peak_pick_status.set(
            "Move near a peak and click the highlighted maximum. Picking stays active."
        )
        self._coordinate.set("Peak picking active")
        self._update_processing_controls()

    def _stop_peak_picking(self) -> None:
        if self._spectrum_interaction.mode == InteractionMode.PEAK_PICK:
            self._spectrum_interaction.enter_navigate_mode()
        self._mpl_interaction.clear_hover_artists()
        self._peak_pick_status.set("Peak picking stopped.")
        self._coordinate.set("")
        self._interaction_mode_text.set("Mode: Navigate")
        self._update_processing_controls()

    def _peak_duplicate_tolerance_ppm(self) -> float:
        spectrum = self.processing_session.committed_spectrum
        if spectrum is None or spectrum.point_count < 2:
            return 1.0e-6
        spacing = np.abs(np.diff(np.asarray(spectrum.axis, dtype=float)))
        spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
        if spacing.size == 0:
            return 1.0e-6
        return max(float(np.median(spacing)) * 1.5, 1.0e-6)

    def _auto_peak_parameters(self) -> PeakPickingParameters:
        try:
            height_pct = float(self._auto_peak_height_pct.get())
            prominence_pct = float(self._auto_peak_prominence_pct.get())
        except ValueError as exc:
            raise PeakPickingError(
                "Peak height and prominence must be numeric percentages."
            ) from exc

        distance_text = self._auto_peak_distance_points.get().strip()
        try:
            distance_points = int(distance_text)
        except ValueError as exc:
            raise PeakPickingError("Minimum distance must be an integer number of points.") from exc

        if not np.isfinite(height_pct) or not 0.0 <= height_pct <= 100.0:
            raise PeakPickingError("Minimum height must be between 0 and 100%.")
        if not np.isfinite(prominence_pct) or not 0.0 <= prominence_pct <= 100.0:
            raise PeakPickingError("Prominence must be between 0 and 100%.")

        return PeakPickingParameters(
            height_fraction=height_pct / 100.0,
            prominence_fraction=prominence_pct / 100.0,
            distance_points=distance_points,
        )

    @staticmethod
    def _is_auto_peak(peak) -> bool:
        return str(getattr(peak, "source", "") or "").strip().lower() == "auto"

    def _remove_auto_peaks(self) -> int:
        """Remove only auto-picked peaks and their linked assignment/observables.

        Manual peaks and independent observables are deliberately preserved.
        """
        dataset = self.current_dataset
        if dataset is None:
            return 0

        auto_peak_ids = {
            peak.id
            for peak in dataset.peaks
            if self._is_auto_peak(peak)
        }
        if not auto_peak_ids:
            return 0

        dataset.peaks[:] = [
            peak
            for peak in dataset.peaks
            if peak.id not in auto_peak_ids
        ]
        for series in self.model.series:
            series.invalidate_tracks(dataset_id=dataset.id, peak_ids=auto_peak_ids,
                                     reason="auto_peaks_removed")

        assignment_map = self._assignment_map()
        for peak_id in auto_peak_ids:
            assignment_map.pop(peak_id, None)

        try:
            sync_dataset_observables(dataset, assignment_map)
        except ObservableSyncError:
            # The cleanup itself is still valid even when observable creation
            # cannot proceed (for example, if nucleus metadata is incomplete).
            pass

        return len(auto_peak_ids)

    def _clear_auto_peaks(self) -> None:
        removed = self._remove_auto_peaks()
        if removed:
            self._peak_pick_status.set(
                f"Removed {removed} auto-picked peak"
                f"{'s' if removed != 1 else ''}; manual peaks were preserved."
            )
        else:
            self._peak_pick_status.set("There are no auto-picked peaks to remove.")

        self._refresh_peak_list()
        self._refresh_peak_markers()
        self._update_processing_controls()

    def _auto_pick_peaks(self) -> None:
        if self._dispatch_experiment_batch("peak_pick", lambda: {"height_fraction": self._auto_peak_parameters().height_fraction, "prominence_fraction": self._auto_peak_parameters().prominence_fraction, "distance_points": self._auto_peak_parameters().distance_points}):
            return
        if not self._peak_picking_available():
            self._show_processing_error(
                ValueError("Automatic peak picking requires a committed ppm-domain spectrum.")
            )
            return

        dataset = self.current_dataset
        spectrum = self.processing_session.committed_spectrum
        if dataset is None or spectrum is None:
            return

        try:
            parameters = self._auto_peak_parameters()
            detected = pick_positive_peaks_1d(spectrum, parameters=parameters)
        except (PeakPickingError, ValueError) as exc:
            self._show_processing_error(exc)
            return

        # Auto Pick is a recalculation, not an additive operation:
        # discard the previous auto result while preserving manually curated peaks.
        replaced = self._remove_auto_peaks()

        tolerance = self._peak_duplicate_tolerance_ppm()
        added = 0
        skipped_manual_duplicates = 0
        for peak in detected:
            if any(
                abs(existing.position_ppm - peak.position_ppm) <= tolerance
                for existing in dataset.peaks
            ):
                skipped_manual_duplicates += 1
                continue
            dataset.peaks.append(peak)
            added += 1

        dataset.peaks.sort(key=lambda peak: peak.position_ppm)

        status = (
            f"Auto Pick found {len(detected)} peaks; "
            f"replaced {replaced} previous auto peak"
            f"{'s' if replaced != 1 else ''}; "
            f"added {added} new auto peak"
            f"{'s' if added != 1 else ''}."
        )
        if skipped_manual_duplicates:
            status += (
                f" {skipped_manual_duplicates} result"
                f"{'s' if skipped_manual_duplicates != 1 else ''} overlapped "
                "manual peaks and were kept manual."
            )
        self._peak_pick_status.set(status)
        self._refresh_peak_list()
        self._refresh_peak_markers()
        self._update_processing_controls()

    def _on_peak_picked(self, candidate) -> None:
        dataset = self.current_dataset
        if dataset is None:
            return
        tolerance = self._peak_duplicate_tolerance_ppm()
        if any(abs(peak.position_ppm - candidate.x) <= tolerance for peak in dataset.peaks):
            self._peak_pick_status.set(
                f"Peak at {candidate.x:.4f} ppm is already in the peak list."
            )
            return
        dataset.peaks.append(
            NMRPeak(
                position_ppm=float(candidate.x),
                height=float(candidate.y),
                source="manual",
            )
        )
        self._peak_pick_status.set(
            f"Added peak at {candidate.x:.4f} ppm. Continue picking or press Stop."
        )
        self._refresh_peak_list()
        self._refresh_peak_markers()

    def _assignment_map(self) -> dict[str, str]:
        dataset = self.current_dataset
        if dataset is None:
            return {}

        if dataset.id not in self._assignment_drafts_by_dataset:
            self._assignment_drafts_by_dataset[dataset.id] = (
                assignments_from_observables(dataset)
            )
        return self._assignment_drafts_by_dataset[dataset.id]

    def _assignment_for_peak(self, peak_id: str) -> str:
        return self._assignment_map().get(peak_id, "")

    def _set_assignment_for_peak(self, peak_id: str, value: str) -> None:
        assignment = value.strip()
        mapping = self._assignment_map()
        if assignment:
            mapping[peak_id] = assignment
        else:
            mapping.pop(peak_id, None)

        dataset = self.current_dataset
        if dataset is None:
            return

        try:
            sync_peak_observable(
                dataset,
                peak_id=peak_id,
                assignment=assignment,
            )
        except ObservableSyncError as exc:
            # Keep the assignment draft: the user should not lose annotation
            # merely because metadata required for an observable is incomplete.
            self._assignment_warning_text.set(str(exc))

    def _cleanup_assignment_drafts(self) -> None:
        dataset = self.current_dataset
        if dataset is None:
            return
        mapping = self._assignment_drafts_by_dataset.get(dataset.id)
        if not mapping:
            return
        live_ids = {peak.id for peak in dataset.peaks}
        for peak_id in tuple(mapping):
            if peak_id not in live_ids:
                mapping.pop(peak_id, None)

    def _filtered_peaks_for_table(self, peaks: list[NMRPeak]) -> list[NMRPeak]:
        mode = self._assignment_filter.get()
        if mode == "Assigned":
            return [peak for peak in peaks if self._assignment_for_peak(peak.id)]
        if mode == "Unassigned":
            return [peak for peak in peaks if not self._assignment_for_peak(peak.id)]
        return peaks

    def _update_assignment_summary(self) -> None:
        dataset = self.current_dataset
        if dataset is None:
            self._assignment_warning_text.set("")
            return

        assignments = [
            self._assignment_for_peak(peak.id)
            for peak in dataset.peaks
            if self._assignment_for_peak(peak.id)
        ]
        duplicates = sorted({
            value
            for value in assignments
            if assignments.count(value) > 1
        })
        if duplicates:
            preview = ", ".join(duplicates[:3])
            if len(duplicates) > 3:
                preview += ", …"
            self._assignment_warning_text.set(
                f"Duplicate assignments: {preview}"
            )
        else:
            self._assignment_warning_text.set("")

    def _visible_peak_ids(self) -> list[str]:
        tree = getattr(self, "_peak_tree", None)
        if tree is None:
            return []
        return list(tree.get_children(""))

    def _edit_selected_assignment(self) -> None:
        selected = self._selected_peak_ids()
        if not selected:
            return
        self._begin_assignment_edit(selected[0])

    def _on_peak_assignment_enter(self, _event=None):
        self._edit_selected_assignment()
        return "break"

    def _on_peak_tree_double_click(self, event) -> None:
        row_id = self._peak_tree.identify_row(event.y)
        column = self._peak_tree.identify_column(event.x)
        # Assignment is the sixth displayed column.
        if row_id and column == "#6":
            self._begin_assignment_edit(row_id)

    def _begin_assignment_edit(self, peak_id: str) -> None:
        tree = self._peak_tree
        if not tree.exists(peak_id):
            return

        self._finish_assignment_edit(commit=True)

        bbox = tree.bbox(peak_id, "assignment")
        if not bbox:
            tree.see(peak_id)
            self.after_idle(lambda: self._begin_assignment_edit(peak_id))
            return

        x, y, width, height = bbox
        editor = ttk.Entry(tree)
        editor.insert(0, self._assignment_for_peak(peak_id))
        editor.select_range(0, "end")
        editor.place(x=x, y=y, width=max(width, 80), height=height)

        self._assignment_editor = editor
        self._assignment_editor_peak_id = peak_id

        editor.bind("<Return>", self._on_assignment_editor_commit)
        editor.bind("<KP_Enter>", self._on_assignment_editor_commit)
        editor.bind("<Escape>", self._on_assignment_editor_cancel)
        editor.bind("<Tab>", self._on_assignment_editor_tab)
        editor.bind("<Shift-Tab>", self._on_assignment_editor_shift_tab)
        editor.bind("<ISO_Left_Tab>", self._on_assignment_editor_shift_tab)
        editor.bind("<Up>", self._on_assignment_editor_up)
        editor.bind("<Down>", self._on_assignment_editor_down)
        editor.bind("<FocusOut>", self._on_assignment_editor_focus_out)
        editor.focus_set()

    def _finish_assignment_edit(self, *, commit: bool) -> str | None:
        editor = self._assignment_editor
        peak_id = self._assignment_editor_peak_id
        if editor is None:
            return peak_id

        value = editor.get()
        try:
            editor.place_forget()
            editor.destroy()
        except tk.TclError:
            pass

        self._assignment_editor = None
        self._assignment_editor_peak_id = None

        if commit and peak_id is not None:
            self._set_assignment_for_peak(peak_id, value)
            self._refresh_peak_list()
            self._update_assignment_summary()
            if self._peak_tree.exists(peak_id):
                self._peak_tree.selection_set(peak_id)
                self._peak_tree.focus(peak_id)

        return peak_id

    def _move_assignment_editor(self, current_peak_id: str | None, delta: int) -> None:
        visible = self._visible_peak_ids()
        if not visible:
            return

        if current_peak_id not in visible:
            target = visible[0 if delta >= 0 else -1]
        else:
            index = visible.index(current_peak_id)
            target = visible[(index + delta) % len(visible)]

        self._peak_tree.selection_set(target)
        self._peak_tree.focus(target)
        self._peak_tree.see(target)
        self.after_idle(lambda: self._begin_assignment_edit(target))

    def _on_assignment_editor_commit(self, _event=None):
        peak_id = self._finish_assignment_edit(commit=True)
        if peak_id is not None:
            self._peak_tree.focus_set()
        return "break"

    def _on_assignment_editor_cancel(self, _event=None):
        peak_id = self._finish_assignment_edit(commit=False)
        if peak_id and self._peak_tree.exists(peak_id):
            self._peak_tree.selection_set(peak_id)
            self._peak_tree.focus(peak_id)
        self._peak_tree.focus_set()
        return "break"

    def _on_assignment_editor_tab(self, _event=None):
        peak_id = self._finish_assignment_edit(commit=True)
        self._move_assignment_editor(peak_id, +1)
        return "break"

    def _on_assignment_editor_shift_tab(self, _event=None):
        peak_id = self._finish_assignment_edit(commit=True)
        self._move_assignment_editor(peak_id, -1)
        return "break"

    def _on_assignment_editor_up(self, _event=None):
        peak_id = self._finish_assignment_edit(commit=True)
        self._move_assignment_editor(peak_id, -1)
        return "break"

    def _on_assignment_editor_down(self, _event=None):
        peak_id = self._finish_assignment_edit(commit=True)
        self._move_assignment_editor(peak_id, +1)
        return "break"

    def _on_assignment_editor_focus_out(self, _event=None):
        self._finish_assignment_edit(commit=True)

    def _peak_sort_value(self, peak: NMRPeak, column: str):
        if column == "ppm":
            return float(peak.position_ppm)
        if column == "intensity":
            return float(peak.height) if peak.height is not None else float("-inf")
        if column == "source":
            return (peak.source or "").lower()
        return float(peak.position_ppm)

    def _sorted_peaks_for_table(self) -> list[NMRPeak]:
        dataset = self.current_dataset
        if dataset is None:
            return []
        return sorted(
            dataset.peaks,
            key=lambda peak: self._peak_sort_value(
                peak, getattr(self, "_peak_sort_column", "ppm")
            ),
            reverse=bool(getattr(self, "_peak_sort_reverse", False)),
        )

    def _sort_peak_table(self, column: str) -> None:
        if getattr(self, "_peak_sort_column", None) == column:
            self._peak_sort_reverse = not self._peak_sort_reverse
        else:
            self._peak_sort_column = column
            self._peak_sort_reverse = False
        self._refresh_peak_list()

    def _refresh_peak_list(self) -> None:
        panel = getattr(self, "_nmr_data_panel", None)
        if panel is not None:
            panel.refresh()
        self.model.update_tracks()
        tree = getattr(self, "_peak_tree", None)
        if tree is None:
            return

        self._begin_peak_selection_projection()
        selected_ids = tuple(tree.selection())
        dataset = self.current_dataset
        selected_peak = self.model.selected_peak_id
        if dataset is not None and any(p.id == selected_peak for p in dataset.peaks):
            selected_ids = (selected_peak,)
        elif selected_peak is not None:
            self.model.selected_peak_id = None
        for item in tree.get_children(""):
            tree.delete(item)

        dataset = self.current_dataset
        if dataset is None:
            if hasattr(self, "_peak_count_text"):
                self._peak_count_text.set("0 peaks")
            self._peak_empty_hint.configure(text="Load an NMR experiment to inspect peaks.")
            self._peak_empty_hint.place(relx=.5, rely=.5, anchor="center")
            self._assignment_warning_text.set("")
            return

        self._cleanup_assignment_drafts()
        all_peaks = self._sorted_peaks_for_table()
        peaks = self._filtered_peaks_for_table(all_peaks)
        if peaks:
            self._peak_empty_hint.place_forget()
        else:
            self._peak_empty_hint.configure(text="No peaks match this filter.\nChange Show to inspect other peaks." if all_peaks
                                            else "No peaks yet.\nUse Auto Pick or pick a peak on the spectrum.")
            self._peak_empty_hint.place(relx=.5, rely=.5, anchor="center")

        if hasattr(self, "_peak_count_text"):
            count = len(all_peaks)
            assigned_count = sum(
                bool(self._assignment_for_peak(peak.id))
                for peak in all_peaks
            )
            observable_count = sum(
                observable_for_peak(dataset, peak.id) is not None
                for peak in all_peaks
            )
            self._peak_count_text.set(
                f"{count} peaks · {assigned_count} assigned · {observable_count} obs"
                + (f" · {sum(p.source == 'manual' for p in dataset.peaks)} manual"
                   f" · {sum(p.source == 'auto' for p in dataset.peaks)} auto" if count else "")
            )

        for number, peak in enumerate(peaks, 1):
            intensity = "—" if peak.height is None else f"{peak.height:.6g}"
            source = {
                "manual": "Manual",
                "auto": "Auto",
            }.get(peak.source or "", peak.source or "—")
            tree.insert(
                "",
                "end",
                iid=peak.id,
                values=(
                    "×",
                    number,
                    f"{peak.position_ppm:.6f}",
                    intensity,
                    source,
                    self._assignment_for_peak(peak.id),
                    "✓" if observable_for_peak(dataset, peak.id) is not None else "",
                ),
            )

        surviving = [peak_id for peak_id in selected_ids if tree.exists(peak_id)]
        if surviving:
            tree.selection_set(surviving)

        self._update_assignment_summary()
        if hasattr(self, "_shift_tree"):
            self._refresh_shift_analysis()

        self._refresh_stacked_series()

    def _refresh_peak_markers(self) -> None:
        for artist in getattr(self, "_picked_peak_artists", []):
            try:
                artist.remove()
            except ValueError:
                pass
        self._picked_peak_artists = []

        self._clear_peak_highlight(draw=False)

        dataset = self.current_dataset
        spectrum = self.processing_session.committed_spectrum
        if (
            dataset is None
            or spectrum is None
            or self._view_choice.get() != "Spectrum"
            or self.processing_session.has_preview
            or spectrum.domain != "frequency"
            or spectrum.axis_unit.lower() != "ppm"
        ):
            self._refresh_predicted_pcs_markers(draw=False)
            self._canvas.draw_idle()
            return

        for peak in dataset.peaks:
            y = self._peak_height_for_display(peak)
            (artist,) = self._axes.plot(
                [peak.position_ppm],
                [y],
                marker="x",
                linestyle="None",
                markersize=6,
                markeredgewidth=1.2,
                zorder=9,
                label="_nolegend_",
            )
            self._picked_peak_artists.append(artist)

        self._refresh_selected_peak_highlights(draw=False)
        self._refresh_predicted_pcs_markers(draw=False)
        self._canvas.draw_idle()

    def _clear_predicted_pcs_markers(self) -> None:
        for artist, _prediction in getattr(
            self, "_predicted_pcs_marker_records", ()
        ):
            try:
                artist.remove()
            except ValueError:
                pass
        self._predicted_pcs_marker_records = []

    def _refresh_predicted_pcs_markers(self, *, draw: bool = True) -> None:
        """Replace the derived paramagnetic-position overlay in place."""
        axes = getattr(self, "_axes", None)
        canvas = getattr(self, "_canvas", None)
        self._clear_predicted_pcs_markers()

        predictions = build_predicted_paramagnetic_peaks(
            getattr(self, "_shift_diagnostic_rows", ()) or (),
            fixed_corrections_ppm=getattr(
                self, "_pcs_corrections_ppm", {}
            ),
        )
        self._predicted_paramagnetic_peaks = predictions

        dataset = self.current_dataset
        spectrum = getattr(self.processing_session, "committed_spectrum", None)
        if (
            axes is None
            or canvas is None
            or dataset is None
            or getattr(dataset, "experimental_role", "unspecified")
            != "paramagnetic"
            or spectrum is None
            or self._view_choice.get() != "Spectrum"
            or self.processing_session.has_preview
            or spectrum.domain != "frequency"
            or spectrum.axis_unit.lower() != "ppm"
        ):
            if draw and canvas is not None:
                canvas.draw_idle()
            return

        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        try:
            for prediction in predictions:
                artist = axes.axvline(
                    prediction.position_ppm,
                    ymin=0.82,
                    ymax=0.98,
                    color="#D97706",
                    linestyle=(0, (3, 2)),
                    linewidth=1.35,
                    alpha=0.9,
                    zorder=10,
                    label="_nolegend_",
                )
                identity = (
                    f"target-{prediction.structured_target_id}:obs-"
                    f"{prediction.paramagnetic_observable_id or prediction.diamagnetic_observable_id}"
                    if prediction.structured_target_id is not None
                    else (
                        f"ref-{prediction.target_ref_id}"
                        if prediction.target_ref_id is not None
                        else prediction.assignment
                    )
                )
                artist.set_gid(f"predicted-paramagnetic:{identity}")
                self._predicted_pcs_marker_records.append(
                    (artist, prediction)
                )
        finally:
            # Artist replacement must not autoscale away from the user's
            # current zoom/pan window.
            axes.set_xlim(xlim)
            axes.set_ylim(ylim)

        if draw:
            canvas.draw_idle()

    def _peak_height_for_display(self, peak: NMRPeak) -> float:
        if peak.height is not None and np.isfinite(peak.height):
            return float(peak.height)
        spectrum = self.processing_session.committed_spectrum
        if spectrum is None:
            return 0.0
        axis = np.asarray(spectrum.axis, dtype=float)
        index = int(np.argmin(np.abs(axis - peak.position_ppm)))
        return float(np.real(spectrum.intensities[index]))

    def _on_peak_tree_click(self, event) -> None:
        tree = self._peak_tree
        row_id = tree.identify_row(event.y)
        column = tree.identify_column(event.x)
        if row_id and column == "#1":
            self.after_idle(lambda peak_id=row_id: self._delete_peak_by_id(peak_id))

    def _begin_peak_selection_projection(self) -> None:
        """Ignore queued Treeview events while projecting model/filter changes.

        Tk delivers generated selection events before idle callbacks. Once the
        projection settles, explicit user selection/deselection is authoritative.
        """
        self._peak_selection_projecting = True
        pending = getattr(self, "_peak_selection_projection_idle", None)
        if pending is not None:
            self.after_cancel(pending)

        def settled():
            self._peak_selection_projecting = False
            self._peak_selection_projection_idle = None
            self._refresh_selected_peak_highlights()
            self._update_processing_controls()
            if not self._shared_selection_syncing:
                self._publish_peak_selection()

        self._peak_selection_projection_idle = self.after_idle(settled)

    def _on_peak_tree_selection(self, _event=None) -> None:
        if getattr(self, "_peak_selection_projecting", False):
            return
        ids = self._selected_peak_ids()
        selected = ids[0] if len(ids) == 1 else None
        if self.model.selected_peak_id != selected:
            self.model.selected_peak_id = selected
            self._refresh_stacked_series()
        dataset = self.current_dataset
        observation = observable_for_peak(dataset, selected) if dataset is not None and selected else None
        self.model.selected_observable_id = observation.id if observation is not None else None
        self._refresh_selected_peak_highlights()
        self._update_processing_controls()
        if not self._shared_selection_syncing:
            self._publish_peak_selection()

    def _selected_peak_ids(self) -> tuple[str, ...]:
        tree = getattr(self, "_peak_tree", None)
        return tuple(tree.selection()) if tree is not None else ()

    def _refresh_selected_peak_highlights(self, *, draw: bool = True) -> None:
        self._clear_peak_highlight(draw=False)

        dataset = self.current_dataset
        if dataset is None:
            if draw:
                self._canvas.draw_idle()
            return

        panel = getattr(self, "_nmr_data_panel", None)
        if panel is not None and self.model.selected_peak_id:
            panel.select_identity(dataset_id=dataset.id, peak_id=self.model.selected_peak_id)

        selected_ids = set(self._selected_peak_ids())
        selected_peaks = [peak for peak in dataset.peaks if peak.id in selected_ids]

        self._peak_highlight_artists = []
        for peak in selected_peaks:
            y = self._peak_height_for_display(peak)
            (artist,) = self._axes.plot(
                [peak.position_ppm],
                [y],
                marker="o",
                linestyle="None",
                markersize=9,
                markerfacecolor="none",
                markeredgewidth=1.6,
                zorder=12,
                label="_nolegend_",
            )
            self._peak_highlight_artists.append(artist)

        if len(selected_peaks) == 1:
            self._coordinate.set(
                f"Selected peak = {selected_peaks[0].position_ppm:.6f} ppm"
            )
        elif len(selected_peaks) > 1:
            self._coordinate.set(f"{len(selected_peaks)} peaks selected")

        if draw:
            self._canvas.draw_idle()

    def _clear_peak_highlight(self, *, draw: bool = True) -> None:
        for artist in getattr(self, "_peak_highlight_artists", []):
            try:
                artist.remove()
            except ValueError:
                pass
        self._peak_highlight_artists = []

        old_artist = getattr(self, "_peak_highlight_artist", None)
        if old_artist is not None:
            try:
                old_artist.remove()
            except ValueError:
                pass
        self._peak_highlight_artist = None

        if draw:
            self._canvas.draw_idle()

    def _delete_peak_by_id(self, peak_id: str) -> None:
        dataset = self.current_dataset
        if dataset is None:
            return
        before = len(dataset.peaks)
        dataset.peaks[:] = [peak for peak in dataset.peaks if peak.id != peak_id]
        self._assignment_map().pop(peak_id, None)
        try:
            sync_dataset_observables(dataset, self._assignment_map())
        except ObservableSyncError:
            pass
        if len(dataset.peaks) != before:
            self._peak_pick_status.set("Peak removed.")
        self._refresh_peak_list()
        self._refresh_peak_markers()
        self._update_processing_controls()

    def _delete_selected_peaks(self) -> None:
        dataset = self.current_dataset
        selected_ids = set(self._selected_peak_ids())
        if dataset is None or not selected_ids:
            return

        before = len(dataset.peaks)
        dataset.peaks[:] = [
            peak for peak in dataset.peaks if peak.id not in selected_ids
        ]
        assignment_map = self._assignment_map()
        for peak_id in selected_ids:
            assignment_map.pop(peak_id, None)
        try:
            sync_dataset_observables(dataset, assignment_map)
        except ObservableSyncError:
            pass
        removed = before - len(dataset.peaks)
        if removed:
            self._peak_pick_status.set(
                f"Removed {removed} selected peak{'s' if removed != 1 else ''}."
            )

        self._refresh_peak_list()
        self._refresh_peak_markers()
        self._update_processing_controls()

    def _on_peak_delete_key(self, _event=None):
        self._delete_selected_peaks()
        return "break"

    def _on_peak_select_all(self, _event=None):
        children = self._peak_tree.get_children("")
        if children:
            self._peak_tree.selection_set(children)
        return "break"

    def _clear_all_peaks(self) -> None:
        dataset = self.current_dataset
        if dataset is None or not dataset.peaks:
            return
        if not messagebox.askyesno(
            "Clear Peaks",
            f"Remove all {len(dataset.peaks)} picked peaks?",
            parent=self,
        ):
            return

        dataset.peaks.clear()
        self._assignment_map().clear()
        try:
            sync_dataset_observables(dataset, self._assignment_map())
        except ObservableSyncError:
            pass
        self._peak_pick_status.set("All picked peaks were removed.")
        self._refresh_peak_list()
        self._refresh_peak_markers()
        self._update_processing_controls()

    def _on_spectrum_peak_click(self, event) -> None:
        """Select an existing picked peak by clicking close to its marker."""
        if getattr(self, "_phase_pivot_pick", False):
            if event.inaxes is not self._axes or event.xdata is None or not np.isfinite(event.xdata):
                return
            if str(self._phase_pick_button.cget("state")) == "disabled":
                self._enter_navigate_mode()
                return
            self._pivot_text.set(f"{float(event.xdata):.12g}")
            self._on_pivot_commit()
            self._enter_navigate_mode(message="Phase pivot selected")
            return
        dataset = self.current_dataset
        tree = getattr(self, "_peak_tree", None)
        if (
            dataset is None
            or tree is None
            or not dataset.peaks
            or event.x is None
            or event.y is None
        ):
            return

        mouse_x = float(event.x)
        mouse_y = float(event.y)
        best_peak = None
        best_distance = None

        for peak in dataset.peaks:
            y = self._peak_height_for_display(peak)
            peak_x_px, peak_y_px = self._axes.transData.transform(
                (peak.position_ppm, y)
            )
            distance = float(np.hypot(peak_x_px - mouse_x, peak_y_px - mouse_y))
            if distance <= 10.0 and (
                best_distance is None or distance < best_distance
            ):
                best_peak = peak
                best_distance = distance

        if best_peak is None or not tree.exists(best_peak.id):
            return

        self._finish_assignment_edit(commit=True)
        tree.selection_set(best_peak.id)
        tree.focus(best_peak.id)
        tree.see(best_peak.id)
        self._refresh_selected_peak_highlights()

    def _begin_reference_peak_selection(self) -> None:
        """Enter hover-assisted observed-peak selection for ppm spectra."""
        spectrum = self.processing_session.committed_spectrum

        if (
            spectrum is None
            or spectrum.domain != "frequency"
            or spectrum.axis_unit.lower() != "ppm"
            or self._view_choice.get() != "Spectrum"
        ):
            self._show_processing_error(
                ValueError(
                    "Reference peak selection requires a ppm-domain spectrum."
                )
            )
            return

        if self.processing_session.has_preview:
            self._show_processing_error(
                ValueError(
                    "Apply or cancel the current processing preview first."
                )
            )
            return

        self._refresh_interaction_peak_cache()

        if not self._spectrum_interaction.peak_cache.candidates:
            self._show_processing_error(
                ValueError(
                    "No suitable peak candidates were found in the current spectrum."
                )
            )
            return

        self._mpl_interaction.cancel_drag()
        self._spectrum_interaction.enter_reference_mode()
        self._interaction_mode_text.set("Mode: Reference")
        self._reference_status.set(
            "Move the pointer near a peak and click the highlighted maximum."
        )
        self._coordinate.set("Reference peak selection active")

    def _on_reference_peak_selected(self, candidate) -> None:
        """Receive a confirmed local maximum from the Matplotlib adapter."""
        self._reference_observed.set(f"{candidate.x:.6f}")
        self._reference_status.set(
            f"Observed peak selected at {candidate.x:.4f} ppm."
        )
        self._coordinate.set(
            f"Observed reference peak = {candidate.x:.6f} ppm"
        )
        self._enable_reference_assignment()
        self._update_reference_offset()
        self._interaction_mode_text.set("Mode: Navigate")
        self._update_processing_controls()

    def _current_reference_nucleus(self) -> str | None:
        """Return the current nucleus in the preset-library convention."""
        dataset = self.current_dataset
        if dataset is None:
            return None

        nucleus = None

        conditions = getattr(dataset, "conditions", None)
        if conditions is not None:
            nucleus = getattr(conditions, "nucleus", None)

        if nucleus is None:
            metadata = getattr(dataset, "metadata", None)
            if metadata is not None:
                nucleus = getattr(metadata, "nucleus", None)

        if nucleus is None:
            nucleus = getattr(dataset, "nucleus", None)

        if nucleus is None:
            return None

        normalized = (
            str(nucleus)
            .strip()
            .upper()
            .replace("^", "")
            .replace("{", "")
            .replace("}", "")
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )

        aliases = {
            "H1": "1H",
            "1H": "1H",
            "PROTON": "1H",
            "C13": "13C",
            "13C": "13C",
            "CARBON13": "13C",
        }
        return aliases.get(normalized)

    def _enable_reference_assignment(self) -> None:
        nucleus = self._current_reference_nucleus()

        if nucleus not in {"1H", "13C"}:
            self._reference_type_selector.configure(state="disabled")
            self._reference_status.set(
                "Reference presets currently support 1H and 13C datasets."
            )
            return

        self._reference_type_selector.configure(state="readonly")
        self._populate_reference_presets()

    def _on_reference_type_changed(self, _event=None) -> None:
        reference_type = self._reference_type.get()

        self._reference_preset.set("")
        self._reference_signal.set("")
        self._reference_target.set("")
        self._reference_offset.set("")
        self._reference_preset_by_display.clear()
        self._reference_resonance_by_display.clear()

        if reference_type == "Custom":
            self._reference_preset_label.configure(text="Source")
            self._reference_preset_selector.configure(
                values=(),
                state="disabled",
            )
            self._reference_signal_selector.configure(
                values=(),
                state="disabled",
            )
            self._reference_target_entry.configure(state="normal")
            self._update_processing_controls()
            return

        self._reference_target_entry.configure(state="readonly")
        self._reference_preset_label.configure(
            text=(
                "Standard"
                if reference_type == "Reference standard"
                else "Solvent"
            )
        )
        self._populate_reference_presets()
        self._update_processing_controls()

    def _populate_reference_presets(self) -> None:
        if not self._reference_observed.get():
            return

        nucleus = self._current_reference_nucleus()
        if nucleus not in {"1H", "13C"}:
            return

        reference_type = self._reference_type.get()
        if reference_type == "Custom":
            return

        category = (
            "reference_standard"
            if reference_type == "Reference standard"
            else "residual_solvent"
        )

        presets = self._reference_library.presets_for_nucleus(
            nucleus,
            category=category,
        )

        self._reference_preset_by_display = {
            preset.display_name: preset
            for preset in presets
        }

        values = tuple(
            preset.display_name
            for preset in presets
        )

        self._reference_preset_selector.configure(
            values=values,
            state="readonly" if values else "disabled",
        )
        self._reference_signal_selector.configure(
            values=(),
            state="disabled",
        )

        self._reference_preset.set("")
        self._reference_signal.set("")
        self._reference_target.set("")
        self._reference_offset.set("")

        if reference_type == "Residual solvent":
            self._suggest_dataset_solvent()
        elif reference_type == "Reference standard":
            self._auto_select_reference_standard()

        self._update_processing_controls()

    def _dataset_solvent_name(self) -> str | None:
        dataset = self.current_dataset
        if dataset is None:
            return None

        conditions = getattr(dataset, "conditions", None)
        if conditions is not None:
            solvent = getattr(conditions, "solvent", None)
            if solvent:
                return str(solvent)

        metadata = getattr(dataset, "metadata", None)
        if metadata is not None:
            solvent = getattr(metadata, "solvent", None)
            if solvent:
                return str(solvent)

        solvent = getattr(dataset, "solvent", None)
        return str(solvent) if solvent else None

    def _suggest_dataset_solvent(self) -> None:
        if self._reference_type.get() != "Residual solvent":
            return

        solvent = self._dataset_solvent_name()
        if not solvent:
            return

        preset = self._reference_library.resolve_alias(solvent)
        if preset is None or preset.category != "residual_solvent":
            return

        if preset.display_name not in self._reference_preset_by_display:
            return

        self._reference_preset.set(preset.display_name)
        self._on_reference_preset_changed()

    def _auto_select_reference_standard(self) -> None:
        """Choose TMS automatically for Reference standard mode.

        Reference-standard presets normally contain only TMS.  If multiple
        standards are added later, prefer an explicit TMS/tetramethylsilane
        match and otherwise auto-select only when there is exactly one choice.
        """
        if self._reference_type.get() != "Reference standard":
            return

        values = tuple(self._reference_preset_selector.cget("values"))
        if not values:
            return

        preferred = None
        for display_name in values:
            folded = str(display_name).casefold()
            if "tms" in folded or "tetramethylsilane" in folded:
                preferred = display_name
                break
        if preferred is None and len(values) == 1:
            preferred = values[0]
        if preferred is None:
            return

        self._reference_preset.set(preferred)
        self._on_reference_preset_changed()

    def _on_reference_preset_changed(self, _event=None) -> None:
        display_name = self._reference_preset.get()
        preset = self._reference_preset_by_display.get(display_name)

        if preset is None:
            self._reference_signal_selector.configure(
                values=(),
                state="disabled",
            )
            self._reference_signal.set("")
            self._reference_target.set("")
            self._reference_offset.set("")
            self._update_processing_controls()
            return

        nucleus = self._current_reference_nucleus()
        if nucleus not in {"1H", "13C"}:
            return

        try:
            observed = float(self._reference_observed.get())
        except ValueError:
            observed = None

        resonances = self._reference_library.sorted_resonances_for(
            preset.id,
            nucleus,
            observed_ppm=observed,
        )

        self._reference_resonance_by_display.clear()
        display_values: list[str] = []

        for resonance in resonances:
            suffix = (
                "  [condition-sensitive]"
                if not resonance.recommended_for_direct_referencing
                else ""
            )
            label = (
                f"{resonance.label} — "
                f"{resonance.ppm:.4f} ppm"
                f"{suffix}"
            )
            self._reference_resonance_by_display[label] = resonance
            display_values.append(label)

        self._reference_signal_selector.configure(
            values=tuple(display_values),
            state="readonly" if display_values else "disabled",
        )

        self._reference_signal.set("")
        self._reference_target.set("")
        self._reference_offset.set("")

        # Residual-solvent resonances still require an explicit user choice.
        # A reference standard such as TMS usually has one defined reference
        # signal, so avoid a redundant dropdown click.
        if (
            self._reference_type.get() == "Reference standard"
            and len(display_values) == 1
        ):
            self._reference_signal.set(display_values[0])
            self._on_reference_signal_changed()
            return

        self._update_processing_controls()

    def _on_reference_signal_changed(self, _event=None) -> None:
        label = self._reference_signal.get()
        resonance = self._reference_resonance_by_display.get(label)

        if resonance is None:
            self._reference_target.set("")
            self._reference_offset.set("")
            self._update_processing_controls()
            return

        self._reference_target.set(f"{resonance.ppm:.6f}")
        self._update_reference_offset()

        if not resonance.recommended_for_direct_referencing:
            self._reference_status.set(
                "Selected reference signal is condition-sensitive; "
                "verify temperature and sample conditions before applying."
            )
        else:
            source = self._reference_library.source_for(resonance)
            self._reference_status.set(
                f"Target selected from {source.id}: {resonance.ppm:.4f} ppm."
            )

        self._update_processing_controls()

    def _on_reference_target_changed(self, _event=None) -> None:
        if self._reference_type.get() != "Custom":
            return
        self._update_reference_offset()
        self._update_processing_controls()

    def _update_reference_offset(self) -> None:
        try:
            observed = float(self._reference_observed.get())
            target = float(self._reference_target.get())
        except (TypeError, ValueError):
            self._reference_offset.set("")
            return

        offset = target - observed
        self._reference_offset.set(f"{offset:+.6f}")

    def _reference_provenance(self) -> dict[str, str | None]:
        """Return provenance for the currently resolved reference target."""
        reference_type = self._reference_type.get()

        if reference_type == "Custom":
            return {
                "source_type": "custom",
                "preset_id": None,
                "resonance_id": None,
                "source_id": None,
            }

        preset = self._reference_preset_by_display.get(
            self._reference_preset.get()
        )
        resonance = self._reference_resonance_by_display.get(
            self._reference_signal.get()
        )

        if preset is None or resonance is None:
            return {
                "source_type": None,
                "preset_id": None,
                "resonance_id": None,
                "source_id": None,
            }

        source_type = (
            "reference_standard"
            if reference_type == "Reference standard"
            else "residual_solvent"
        )
        return {
            "source_type": source_type,
            "preset_id": preset.id,
            "resonance_id": resonance.id,
            "source_id": resonance.source_id,
        }

    def _reference_values_ready(self) -> bool:
        try:
            observed = float(self._reference_observed.get())
            target = float(self._reference_target.get())
        except (TypeError, ValueError):
            return False
        return np.isfinite(observed) and np.isfinite(target)

    def _preview_reference(self) -> None:
        if self._dispatch_experiment_batch("reference", None):
            return
        if not self._reference_values_ready():
            self._show_processing_error(
                ValueError(
                    "Select an observed peak and a valid reference target first."
                )
            )
            return

        reference_type = self._reference_type.get()
        if reference_type != "Custom":
            if self._reference_preset.get() not in self._reference_preset_by_display:
                self._show_processing_error(
                    ValueError("Select a solvent or reference standard first.")
                )
                return
            if self._reference_signal.get() not in self._reference_resonance_by_display:
                self._show_processing_error(
                    ValueError("Select the reference signal to use.")
                )
                return

        provenance = self._reference_provenance()

        try:
            self.processing_session.preview_reference(
                observed_ppm=self._reference_observed.get(),
                target_ppm=self._reference_target.get(),
                **provenance,
            )
        except ProcessingSessionError as exc:
            self._show_processing_error(exc)
            return

        self._view_choice.set("Spectrum")
        self._view_selector.configure(
            values=self._available_view_choices(),
            state="readonly",
        )
        self._reference_status.set(
            f"Previewing reference shift {self._reference_offset.get()} ppm."
        )
        self._refresh(preserve_view=True)

    def _debug_enable_reference_hover(self) -> None:
        """Developer hook retained for compatibility with earlier testing."""
        self._begin_reference_peak_selection()

    def _update_zero_fill_choices(self) -> None:
        spectrum = self.processing_session.committed_spectrum
        if spectrum is None:
            self._zero_selector.configure(values=())
            return
        if spectrum.domain != "time":
            dataset = self.current_dataset
            spectrum = None if dataset is None else dataset.raw_spectrum
        if spectrum is None or spectrum.domain != "time":
            self._zero_selector.configure(values=())
            return
        sizes = tuple(
            size
            for size in suggested_zero_fill_sizes(spectrum.point_count)
            if size <= self.processing_session.max_points
        )
        values = tuple(str(size) for size in sizes)
        self._zero_selector.configure(values=values)
        suggested = next((size for size in sizes if size > spectrum.point_count), None)
        self._zero_target.set(str(suggested or spectrum.point_count))

    def _update_processing_controls(self) -> None:
        session = self.processing_session
        time_domain = session.can_process_time_domain
        preview_operation = session.preview.operation if session.preview else None

        dataset = self.current_dataset
        raw_available = dataset is not None and dataset.raw_spectrum is not None
        preprocess_available = (
            raw_available
            and self._view_choice.get() in {"Spectrum", "Raw FID"}
            and preview_operation in {None, "apodization", "zero_fill"}
        )
        preview_state = "normal" if preprocess_available else "disabled"
        for widget in (
            self._apod_preview_button,
            self._zero_preview_button,
            self._apod_entry,
            self._zero_selector,
        ):
            widget.configure(state=preview_state)
        self._apod_method_selector.configure(
            state="readonly" if preprocess_available else "disabled"
        )
        self._apod_apply_button.configure(
            state="normal" if preview_operation == "apodization" else "disabled"
        )
        self._apod_cancel_button.configure(
            state="normal" if preview_operation == "apodization" else "disabled"
        )
        self._zero_apply_button.configure(
            state="normal" if preview_operation == "zero_fill" else "disabled"
        )
        self._zero_cancel_button.configure(
            state="normal" if preview_operation == "zero_fill" else "disabled"
        )
        if self._view_choice.get() == "Raw FID":
            action_text = "Create Spectrum" if time_domain else "Start Reprocessing"
            action_state = "disabled" if session.has_preview else "normal"
            self._reprocess_button.configure(
                text=action_text,
                state=action_state,
            )
            if not self._reprocess_button.winfo_manager():
                self._reprocess_button.pack(side="left", padx=(6, 0))
        else:
            action_text = "Start Reprocessing"
            action_state = "disabled" if session.has_preview else "normal"
            self._reprocess_button.pack_forget()

        if hasattr(self, "_flow_ft_button"):
            self._flow_ft_button.configure(
                text=action_text,
                state=action_state,
            )
        phase_available = (
            session.can_phase
            and self._view_choice.get() == "Spectrum"
            and preview_operation in {None, "phase"}
        )
        phase_state = "normal" if phase_available else "disabled"
        for widget in (
            *self._phase_entries,
            *self._phase_drags,
            self._pivot_entry,
            self._pivot_slider,
            self._biggest_peak_button,
            self._phase_reset_button,
            self._phase_flip_button,
            self._phase_pick_button,
            self._auto_phase_button,
        ):
            widget.configure(state=phase_state)
        self._auto_phase_selector.configure(
            state="readonly" if phase_available else "disabled"
        )
        self._phase_apply_button.configure(
            state="normal" if preview_operation == "phase" or (hasattr(self,"experiments") and len(self.experiments.selected_ids)>1 and phase_available) else "disabled"
        )
        self._phase_cancel_button.configure(
            state="normal" if preview_operation == "phase" else "disabled"
        )
        baseline_available = (
            session.can_baseline
            and self._view_choice.get() == "Spectrum"
            and preview_operation in {None, "baseline"}
        )
        baseline_state = "normal" if baseline_available else "disabled"
        self._baseline_method_selector.configure(
            state="readonly" if baseline_available else "disabled"
        )
        for widget in (
            self._baseline_order_entry,
            self._baseline_lambda_entry,
            self._baseline_arpls_lambda_entry,
            self._baseline_p_entry,
            self._baseline_preview_button,
            self._auto_baseline_button,
        ):
            widget.configure(state=baseline_state)
        overlay_state = "normal" if preview_operation == "baseline" else "disabled"
        self._baseline_fit_toggle.configure(state=overlay_state)
        self._baseline_original_toggle.configure(state=overlay_state)
        self._baseline_apply_button.configure(
            state="normal" if preview_operation == "baseline" else "disabled"
        )
        self._baseline_cancel_button.configure(
            state="normal" if preview_operation == "baseline" else "disabled"
        )

        committed = session.committed_spectrum
        reference_available = (
            committed is not None
            and committed.domain == "frequency"
            and committed.axis_unit.lower() == "ppm"
            and self._view_choice.get() == "Spectrum"
            and not session.has_preview
        )
        if hasattr(self, "_reference_pick_button"):
            reference_pick_available = (
                reference_available
                and self._spectrum_interaction.mode == InteractionMode.NAVIGATE
            )
            self._reference_pick_button.configure(
                state="normal" if reference_pick_available else "disabled"
            )

        if hasattr(self, "_reference_type_selector"):
            observed_selected = bool(self._reference_observed.get())
            nucleus_supported = self._current_reference_nucleus() in {"1H", "13C"}
            type_enabled = (
                reference_available
                and observed_selected
                and nucleus_supported
            )
            self._reference_type_selector.configure(
                state="readonly" if type_enabled else "disabled"
            )

        if hasattr(self, "_reference_preview_button"):
            reference_preview_allowed = (
                reference_available
                and self._reference_values_ready()
                and preview_operation in {None, "reference"}
            )
            self._reference_preview_button.configure(
                state="normal" if reference_preview_allowed else "disabled"
            )
            self._reference_apply_button.configure(
                state="normal" if preview_operation == "reference" else "disabled"
            )
            self._reference_cancel_button.configure(
                state="normal" if preview_operation == "reference" else "disabled"
            )

        if hasattr(self, "_peak_pick_start_button"):
            peak_available = self._peak_picking_available()
            picking = self._spectrum_interaction.mode == InteractionMode.PEAK_PICK
            interaction_idle = self._spectrum_interaction.mode == InteractionMode.NAVIGATE
            self._peak_pick_start_button.configure(
                state="normal" if (peak_available and interaction_idle) else "disabled"
            )
            self._peak_pick_stop_button.configure(
                state="normal" if picking else "disabled"
            )
            self._auto_peak_button.configure(
                state="normal" if (peak_available and interaction_idle) else "disabled"
            )
            auto_parameter_state = (
                "normal" if (peak_available and interaction_idle) else "disabled"
            )
            for widget in (
                self._auto_peak_height_entry,
                self._auto_peak_prominence_entry,
                self._auto_peak_distance_entry,
                self._clear_auto_peaks_button,
            ):
                widget.configure(state=auto_parameter_state)

            has_peaks = bool(self.current_dataset and self.current_dataset.peaks)
            self._delete_peak_button.configure(
                state="normal" if has_peaks else "disabled"
            )
            self._clear_peaks_button.configure(
                state="normal" if has_peaks else "disabled"
            )

        choices = self._available_view_choices()
        self._view_selector.configure(
            values=choices,
            state="readonly" if choices else "disabled",
        )

    def _show_processing_error(self, error: Exception) -> None:
        messagebox.showerror("NMR Processing", str(error), parent=self)

    def _draw_empty_plot(self) -> None:
        adapter = getattr(self, "_mpl_interaction", None)
        if adapter is not None:
            adapter.clear_hover_artists()
            adapter.clear_selection_artists()
        self._predicted_pcs_marker_records = []
        self._predicted_paramagnetic_peaks = []
        self._axes.clear()
        self._spectrum_line = None
        self._baseline_line = None
        self._baseline_original_line = None
        self._pivot_marker = None
        self._axes.text(
            0.5,
            0.5,
            "Open a Bruker or Varian/Agilent 1D experiment",
            ha="center",
            va="center",
            transform=self._axes.transAxes,
            color="0.45",
        ).set_gid("nmr-empty-hint")
        self._axes.set_axis_off()
        self._display_signature = None
        self._canvas.draw_idle()

    def _draw_spectrum(self, spectrum, *, preserve_view: bool = False) -> None:
        previous_xlim = self._axes.get_xlim() if preserve_view else None
        previous_ylim = self._axes.get_ylim() if preserve_view else None
        previous_signature = getattr(self, "_display_signature", None)
        adapter = getattr(self, "_mpl_interaction", None)
        if adapter is not None:
            adapter.clear_hover_artists()
            adapter.detach_selection_artists()
        self._clear_peak_highlight(draw=False)
        self._picked_peak_artists = []
        self._predicted_pcs_marker_records = []
        self._axes.clear()
        self._baseline_line = None
        self._baseline_original_line = None
        self._pivot_marker = None
        self._axes.set_axis_on()
        if spectrum is None:
            self._draw_empty_plot()
            return
        values = np.real(spectrum.intensities)
        (self._spectrum_line,) = self._axes.plot(
            spectrum.axis, values, color="#2459A6", linewidth=0.8
        )
        self._axes.set_xlabel(spectrum.axis_unit)
        self._axes.set_ylabel("Intensity (a.u.)")

        # Compact y tick text plus fixed subplot margins keeps the physical
        # spectrum viewport stable while intensity scale changes.
        try:
            self._axes.ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(-3, 3),
                useOffset=True,
            )
        except AttributeError:
            pass

        self._axes.grid(True, alpha=0.16, linewidth=0.5)
        self._axes.margins(x=0.01)
        if should_invert_display_axis(spectrum):
            self._axes.invert_xaxis()
        self._axes.relim()
        self._axes.autoscale_view()
        self._full_xlim = self._axes.get_xlim()
        self._full_ylim = self._axes.get_ylim()
        signature = (spectrum.domain, spectrum.axis_unit)
        if preserve_view and previous_signature == signature:
            self._axes.set_xlim(previous_xlim)
            self._axes.set_ylim(previous_ylim)
        self._display_signature = signature

        if (
            adapter is not None
            and self._view_choice.get() == "Spectrum"
            and not self.processing_session.has_preview
            and spectrum is self.processing_session.committed_spectrum
        ):
            adapter.restore_selection_artists()

        self._refresh_peak_markers()
        self._canvas.draw_idle()

    def _reset_view(self) -> None:
        if self._stacked_series_visible():
            self._stacked_view.reset_view()
            return
        adapter = getattr(self, "_mpl_interaction", None)
        if adapter is not None:
            adapter.cancel_drag()
        if self._full_xlim is not None:
            self._axes.set_xlim(self._full_xlim)
        if self._full_ylim is not None:
            self._axes.set_ylim(self._full_ylim)
        self._canvas.draw_idle()

    def _auto_scale_y(self) -> None:
        if self._stacked_series_visible():
            self._stacked_view.autoscale_y()
            return
        spectrum = self._current_display_spectrum()
        if spectrum is None:
            return
        low, high = sorted(self._axes.get_xlim())
        mask = (spectrum.axis >= low) & (spectrum.axis <= high)
        values = np.real(spectrum.intensities[mask])
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        minimum, maximum = float(values.min()), float(values.max())
        padding = max(
            (maximum - minimum) * 0.08,
            max(abs(minimum), abs(maximum), 1.0) * 0.01,
        )
        self._axes.set_ylim(minimum - padding, maximum + padding)
        self._canvas.draw_idle()

    def _on_plot_motion(self, event) -> None:
        if event.inaxes is not self._axes or event.xdata is None:
            self._coordinate.set("")
            return
        self._coordinate.set(f"x = {event.xdata:.6g} {self._axes.get_xlabel() or 'x'}")

    def _apply_fourier_transform(self) -> None:
        try:
            self.processing_session.apply_fourier_transform()
        except ProcessingSessionError as exc:
            self._show_processing_error(exc)
            return
        self._view_selector.configure(
            values=self._available_view_choices(), state="readonly"
        )
        self._view_choice.set("Spectrum")
        self._initialize_phase_controls()
        self._refresh()

    def _refresh_history(self, dataset) -> None:
        self._history_text.configure(state="normal")
        self._history_text.delete("1.0", "end")
        self._history_text.insert(
            "1.0", "\n".join(format_processing_history(dataset.processing_history))
        )
        self._history_text.configure(state="disabled")

    def _refresh_metadata(self, dataset) -> None:
        for item in self._metadata_tree.get_children(""):
            self._metadata_tree.delete(item)
        for label, value in format_metadata_rows(dataset):
            self._metadata_tree.insert("", "end", values=(label, value))
