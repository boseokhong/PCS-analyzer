"""Independent Chemical Map window for PCS Analyzer D4c."""

from __future__ import annotations

from ui.widget_style import ui_font_name
from ui.style import get_app_fonts
import math
import tkinter as tk
from tkinter import colorchooser, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from logic.chemical_map import (
    ENGINE_COORDGEN,
    ENGINE_GRAPH_2D,
    ENGINE_RDKIT_2D,
    LAYOUT_2D_CHEMICAL,
    LAYOUT_GEOMETRY_PROJECTION,
    build_chemical_map_model,
    find_model_for_atom,
    layout_chemical_map,
)
from logic.ligand_correspondence import analyze_ligand_correspondence
from logic.nmr_sites import analyze_nmr_sites
from logic.ligand_grouping_state import refresh_ligand_analysis_in_state
from logic.chemical_structure_editor import (
    BOND_AROMATIC,
    BOND_DELOCALIZED,
    BOND_DOUBLE,
    BOND_SINGLE,
    BOND_TRIPLE,
    BOND_UNKNOWN,
    add_bond_in_state,
    cleanup_structure_2d,
    get_2d_override,
    get_2d_override_source,
    is_manual_2d_override_source,
    remove_bond_in_state,
    reset_2d_layout,
    reset_bond_override_in_state,
    restore_editor_state,
    set_atom_2d_position,
    set_bond_type_in_state,
    snapshot_editor_state,
    transform_2d_layout,
)
from logic.selection_context import (
    ensure_selection_hub,
    publish_ref_selection,
)


_ELEMENT_COLORS = {
    "H": "#F7F7F7",
    "C": "#505050",
    "N": "#3A63C7",
    "O": "#D84343",
    "F": "#62B36F",
    "P": "#D88A2D",
    "S": "#C7AA32",
    "Cl": "#4DAA55",
    "Br": "#8F4A35",
    "I": "#7553A6",
}
_DEFAULT_COLOR = "#8B8B8B"
_DEFAULT_BACKGROUND_COLOR = "#FFFFFF"


def _normalize_hex_color(value: object, default: str = _DEFAULT_BACKGROUND_COLOR) -> str:
    text = str(value or "").strip()
    if len(text) == 7 and text.startswith("#"):
        try:
            int(text[1:], 16)
        except ValueError:
            return default
        return text.upper()
    return default


def _contrast_palette(background: str) -> tuple[str, str]:
    """Return high-contrast foreground and muted-line colors for a background."""
    color = _normalize_hex_color(background)
    r = int(color[1:3], 16) / 255.0
    g = int(color[3:5], 16) / 255.0
    b = int(color[5:7], 16) / 255.0
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if luminance < 0.45:
        return "#F3F4F6", "#A3A3A3"
    return "#222222", "#6B7280"


class ChemicalMapWindow(tk.Toplevel):
    """Separate ligand-centric 2D structure view synchronized through Ref IDs."""

    def __init__(self, master, *, state: dict):
        super().__init__(master)
        self.state = state
        from ui.document_integrations import active_document_title

        self.title(active_document_title(state, "Chemical Map"))
        self.geometry("980x720")
        self.minsize(780, 560)

        self.graph = None
        self.analysis = None
        self.correspondence_analysis = None
        self.nmr_site_analysis = None
        self.model = None
        self.layout = None
        self.background_color = _normalize_hex_color(
            self.state.get("chemical_map_background"),
            _DEFAULT_BACKGROUND_COLOR,
        )
        self.state["chemical_map_background"] = self.background_color

        self._group_lookup: dict[str, int] = {}
        self._instance_lookup: dict[str, int] = {}
        self._atom_xy: dict[int, tuple[float, float]] = {}
        self._selected_atom_id: int | None = None
        self._selected_atom_ids: tuple[int, ...] = ()
        self._selected_assignment: str | None = None
        self._tree_bind_id = None
        self._hover_annotation = None
        self._selected_bond: tuple[int, int] | None = None
        self._drag_atom_id: int | None = None
        self._drag_base_coordinates: dict[int, tuple[float, float]] = {}
        self._edit_undo_snapshot = None

        self._build_ui()
        self._last_layout_engine_label = str(self.layout_engine_var.get() or "CoordGen")
        self.selection_hub = ensure_selection_hub(self.state)
        self._selection_token = self.selection_hub.subscribe(
            self._on_shared_selection,
            fire_immediately=False,
        )
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.refresh_structure()

    def _build_ui(self) -> None:
        """Build a compact toolbar-first UI with the molecular view as the focus."""
        # One compact control language for toolbar actions.  ttk.Button,
        # ttk.Menubutton and mode-style Checkbuttons otherwise inherit
        # noticeably different vertical padding on Windows.
        style = ttk.Style(self)
        style.configure("ChemicalMap.Toolbar.TButton", padding=(7, 4))
        style.configure("ChemicalMap.Toolbar.TMenubutton", padding=(7, 4))
        style.configure("ChemicalMap.Toolbar.TCheckbutton", padding=(5, 4))

        outer = ttk.Frame(self, padding=(7, 6, 7, 5))
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        # Keep controls compact: two fixed toolbar rows, an optional contextual
        # edit row, a large resizable canvas, then a thin status area.
        toolbar = ttk.Frame(outer)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        toolbar.columnconfigure(0, weight=1)

        primary = ttk.Frame(toolbar)
        primary.grid(row=0, column=0, sticky="ew")
        primary.columnconfigure(1, weight=3)
        primary.columnconfigure(3, weight=2)
        primary.columnconfigure(8, weight=1)

        ttk.Label(primary, text="Ligand").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.group_var = tk.StringVar()
        self.group_combo = ttk.Combobox(
            primary, textvariable=self.group_var, state="readonly", width=25
        )
        self.group_combo.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.group_combo.bind("<<ComboboxSelected>>", self._on_group_changed)

        ttk.Label(primary, text="Instance").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.instance_var = tk.StringVar()
        self.instance_combo = ttk.Combobox(
            primary, textvariable=self.instance_var, state="readonly", width=20
        )
        self.instance_combo.grid(row=0, column=3, sticky="ew", padx=(0, 10))
        self.instance_combo.bind("<<ComboboxSelected>>", self._on_instance_changed)

        ttk.Label(primary, text="View").grid(row=0, column=4, sticky="w", padx=(0, 4))
        self.view_mode_var = tk.StringVar(value="Chemical")
        self.view_mode_combo = ttk.Combobox(
            primary,
            textvariable=self.view_mode_var,
            state="readonly",
            width=18,
            values=("Chemical", "Atom Map", "Geometry Projection"),
        )
        self.view_mode_combo.grid(row=0, column=5, sticky="w", padx=(0, 10))
        self.view_mode_combo.bind("<<ComboboxSelected>>", self._on_layout_changed)

        ttk.Label(primary, text="Layout").grid(row=0, column=6, sticky="w", padx=(0, 4))
        self.layout_engine_var = tk.StringVar(value="CoordGen")
        self.layout_engine_combo = ttk.Combobox(
            primary,
            textvariable=self.layout_engine_var,
            state="readonly",
            width=11,
            values=("CoordGen", "RDKit 2D", "Graph 2D"),
        )
        self.layout_engine_combo.grid(row=0, column=7, sticky="w")
        self.layout_engine_combo.bind("<<ComboboxSelected>>", self._on_layout_changed)

        ttk.Button(
            primary,
            text="Refresh",
            width=9,
            style="ChemicalMap.Toolbar.TButton",
            command=self.refresh_structure,
        ).grid(row=0, column=9, sticky="e", padx=(8, 0))

        secondary = ttk.Frame(toolbar)
        secondary.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        secondary.columnconfigure(11, weight=1)

        self.show_h_var = tk.BooleanVar(value=False)
        self.show_ids_var = tk.BooleanVar(value=False)
        self.show_labels_var = tk.BooleanVar(value=False)
        self.show_donors_var = tk.BooleanVar(value=True)
        self.show_nmr_sites_var = tk.BooleanVar(value=True)
        self.show_matching_var = tk.BooleanVar(value=True)
        self.bond_edit_var = tk.BooleanVar(value=False)
        self.edit_2d_var = tk.BooleanVar(value=False)
        self.propagate_matching_var = tk.BooleanVar(value=True)

        # Frequently used overlays stay directly accessible.
        for column, text, variable in (
            (0, "H", self.show_h_var),
            (1, "Ref IDs", self.show_ids_var),
            (2, "NMR sites", self.show_nmr_sites_var),
            (3, "Matching", self.show_matching_var),
        ):
            ttk.Checkbutton(
                secondary,
                text=text,
                variable=variable,
                style="ChemicalMap.Toolbar.TCheckbutton",
                command=self.redraw,
            ).grid(row=0, column=column, sticky="w", padx=(0, 8))

        ttk.Separator(secondary, orient=tk.VERTICAL).grid(
            row=0, column=4, sticky="ns", padx=(2, 8)
        )

        self.bond_edit_check = ttk.Checkbutton(
            secondary,
            text="Bond Edit",
            width=10,
            style="ChemicalMap.Toolbar.TCheckbutton",
            variable=self.bond_edit_var,
            command=lambda: self._on_editor_mode_changed("bond"),
        )
        self.bond_edit_check.grid(row=0, column=5, sticky="w", padx=(0, 4))

        self.edit_2d_check = ttk.Checkbutton(
            secondary,
            text="2D Edit",
            width=10,
            style="ChemicalMap.Toolbar.TCheckbutton",
            variable=self.edit_2d_var,
            command=lambda: self._on_editor_mode_changed("2d"),
        )
        self.edit_2d_check.grid(row=0, column=6, sticky="w", padx=(0, 4))

        self.cleanup_button = ttk.Button(
            secondary,
            text="Cleanup",
            width=10,
            style="ChemicalMap.Toolbar.TButton",
            command=self._cleanup_structure,
        )
        self.cleanup_button.grid(row=0, column=7, sticky="w", padx=(0, 4))

        self.undo_button = ttk.Button(
            secondary,
            text="Undo",
            width=10,
            style="ChemicalMap.Toolbar.TButton",
            command=self._undo_last_edit,
        )
        self.undo_button.grid(row=0, column=8, sticky="w", padx=(0, 4))

        more_menu = ttk.Menubutton(
            secondary,
            text="More",
            width=8,
            style="ChemicalMap.Toolbar.TMenubutton",
        )
        more_popup = tk.Menu(more_menu, tearoff=False)
        more_popup.add_checkbutton(
            label="PCS labels", variable=self.show_labels_var, command=self.redraw
        )
        more_popup.add_checkbutton(
            label="Donor markers", variable=self.show_donors_var, command=self.redraw
        )
        more_popup.add_separator()
        more_popup.add_command(label="Choose background…", command=self._choose_background_color)
        more_menu["menu"] = more_popup
        more_menu.grid(row=0, column=9, sticky="w", padx=(0, 6))

        background_frame = ttk.Frame(secondary)
        background_frame.grid(row=0, column=10, sticky="w")
        ttk.Label(background_frame, text="BG").pack(side=tk.LEFT, padx=(0, 3))
        self.background_swatch = tk.Label(
            background_frame,
            width=2,
            height=1,
            relief=tk.SUNKEN,
            borderwidth=1,
            background=self.background_color,
            cursor="hand2",
        )
        self.background_swatch.pack(side=tk.LEFT)
        self.background_swatch.bind(
            "<Button-1>", lambda event: self._choose_background_color()
        )

        # Context controls appear only while an edit mode is active.
        self.editor_context = ttk.Frame(outer, padding=(4, 3))
        self.editor_context.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        self.editor_context.columnconfigure(10, weight=1)

        self.context_label = ttk.Label(self.editor_context, text="", width=14)
        self.context_label.grid(row=0, column=0, sticky="w", padx=(0, 6))

        self.add_bond_button = ttk.Button(
            self.editor_context,
            text="Add bond",
            width=11,
            style="ChemicalMap.Toolbar.TButton",
            command=self._add_bond_from_selection,
        )
        self.add_bond_button.grid(row=0, column=1, sticky="w", padx=(0, 5))

        self.tools_menu = ttk.Menubutton(
            self.editor_context,
            text="2D Tools",
            width=11,
            style="ChemicalMap.Toolbar.TMenubutton",
        )
        tools_popup = tk.Menu(self.tools_menu, tearoff=False)
        tools_popup.add_command(label="Rotate +30°", command=lambda: self._transform_2d(30.0))
        tools_popup.add_command(label="Rotate -30°", command=lambda: self._transform_2d(-30.0))
        tools_popup.add_separator()
        tools_popup.add_command(
            label="Flip horizontal", command=lambda: self._transform_2d(0.0, flip_horizontal=True)
        )
        tools_popup.add_command(
            label="Flip vertical", command=lambda: self._transform_2d(0.0, flip_vertical=True)
        )
        self.tools_menu["menu"] = tools_popup
        self.tools_menu.grid(row=0, column=2, sticky="w", padx=(0, 5))

        self.reset_2d_button = ttk.Button(
            self.editor_context,
            text="Reset 2D",
            width=11,
            style="ChemicalMap.Toolbar.TButton",
            command=self._reset_2d_layout,
        )
        self.reset_2d_button.grid(row=0, column=3, sticky="w", padx=(0, 8))

        self.match_edits_check = ttk.Checkbutton(
            self.editor_context,
            text="Apply to matching ligands",
            variable=self.propagate_matching_var,
            style="ChemicalMap.Toolbar.TCheckbutton",
        )
        self.match_edits_check.grid(row=0, column=4, sticky="w")

        canvas_frame = ttk.Frame(outer, relief=tk.SUNKEN, borderwidth=1)
        canvas_frame.grid(row=2, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        fig = Figure(figsize=(7.8, 5.8), dpi=100, facecolor=self.background_color)
        self.figure = fig
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor(self.background_color)
        self.ax.set_axis_off()

        self.canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.canvas.mpl_connect("button_press_event", self._on_plot_click)
        self.canvas.mpl_connect("button_release_event", self._on_plot_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.canvas.mpl_connect("axes_leave_event", self._on_plot_leave)

        status_frame = ttk.Frame(outer)
        status_frame.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        status_frame.columnconfigure(0, weight=1)

        self.summary_var = tk.StringVar(value="")
        ttk.Label(
            status_frame, textvariable=self.summary_var, anchor="w", font=ui_font_name(status_frame, "ui_small")
        ).grid(row=0, column=0, sticky="ew")

        self.status_var = tk.StringVar(
            value="Hover for atom/Ref details; click an atom to select it."
        )
        ttk.Label(status_frame, textvariable=self.status_var, anchor="w").grid(
            row=1, column=0, sticky="ew"
        )

        self._sync_view_control_states()
        self._sync_editor_context_bar()

    def _sync_view_control_states(self) -> None:
        """Keep controls relevant to the active representation without changing behavior."""
        is_2d = self._layout_mode() != LAYOUT_GEOMETRY_PROJECTION
        self.layout_engine_combo.configure(state="readonly" if is_2d else "disabled")
        depiction_state = "normal" if is_2d else "disabled"
        for widget_name in (
            "cleanup_button",
            "edit_2d_check",
            "tools_menu",
            "reset_2d_button",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.configure(state=depiction_state)
        self._sync_editor_context_bar()

    def _sync_editor_context_bar(self) -> None:
        """Show only the controls needed by the active edit mode."""
        frame = getattr(self, "editor_context", None)
        if frame is None:
            return

        bond_mode = bool(self.bond_edit_var.get())
        edit_2d_mode = bool(self.edit_2d_var.get()) and (
            self._layout_mode() != LAYOUT_GEOMETRY_PROJECTION
        )

        if not bond_mode and not edit_2d_mode:
            frame.grid_remove()
            return

        frame.grid()
        self.context_label.configure(
            text="Bond editing" if bond_mode else "2D depiction"
        )

        if bond_mode:
            self.add_bond_button.grid()
            self.tools_menu.grid_remove()
            self.reset_2d_button.grid_remove()
        else:
            self.add_bond_button.grid_remove()
            self.tools_menu.grid()
            self.reset_2d_button.grid()

    def _choose_background_color(self) -> None:
        _rgb, color = colorchooser.askcolor(
            color=self.background_color,
            parent=self,
            title="Chemical Map background",
        )
        if not color:
            return
        self.background_color = _normalize_hex_color(color, self.background_color)
        self.state["chemical_map_background"] = self.background_color
        try:
            self.background_swatch.configure(background=self.background_color)
        except Exception:
            pass
        self.redraw()

    def _plot_foreground(self) -> str:
        return _contrast_palette(self.background_color)[0]

    def _plot_muted(self) -> str:
        return _contrast_palette(self.background_color)[1]

    def _apply_plot_background(self) -> None:
        self.figure.patch.set_facecolor(self.background_color)
        self.ax.set_facecolor(self.background_color)

    def _bind_main_tree(self) -> None:
        tree = self.state.get("tree")
        if tree is None:
            return
        try:
            self._tree_bind_id = tree.bind(
                "<<TreeviewSelect>>",
                self._on_main_tree_select,
                add="+",
            )
        except Exception:
            self._tree_bind_id = None

    def _on_close(self) -> None:
        try:
            self.selection_hub.unsubscribe(self._selection_token)
        except Exception:
            pass
        self.state["chemical_map_win"] = None
        self.destroy()

    def refresh_structure(self) -> None:
        from ui.document_integrations import active_document_title

        self.title(active_document_title(self.state, "Chemical Map"))
        selected_before = self._selected_ref_from_main_tree()

        try:
            analysis = refresh_ligand_analysis_in_state(self.state)
            graph = self.state.get("structure_graph")
        except Exception as exc:
            self.graph = None
            self.analysis = None
            self.correspondence_analysis = None
            self.nmr_site_analysis = None
            self.model = None
            self._group_lookup.clear()
            self.group_combo["values"] = ()
            self.instance_combo["values"] = ()
            self._draw_message(f"Chemical Map unavailable:\n{exc}")
            self.status_var.set("Load a molecular structure first.")
            return

        self.graph = graph
        self.analysis = analysis
        try:
            self.correspondence_analysis = analyze_ligand_correspondence(
                graph, analysis
            )
            self.nmr_site_analysis = analyze_nmr_sites(
                graph,
                analysis,
                correspondence=self.correspondence_analysis,
            )
        except Exception:
            # Chemical Map must remain usable even when optional D5a semantic
            # analysis cannot be built for a hand-edited structure.
            self.correspondence_analysis = None
            self.nmr_site_analysis = None

        self._group_lookup = {}
        group_labels = []
        for group in analysis.groups:
            label = (
                f"G{group.group_id} — {group.formula}"
                f"  ×{group.multiplicity}"
            )
            group_labels.append(label)
            self._group_lookup[label] = group.group_id

        self.group_combo["values"] = group_labels

        if not group_labels:
            self.group_var.set("")
            self.instance_var.set("")
            self.instance_combo["values"] = ()
            self._draw_message(
                "No coordinating ligand fragments were detected.\n"
                "Check StructureGraph connectivity or coordination overrides."
            )
            self.status_var.set("No coordinating ligand fragments.")
            return

        preferred = (
            find_model_for_atom(analysis, selected_before)
            if selected_before is not None
            else None
        )

        preferred_group = preferred[0] if preferred else analysis.groups[0].group_id
        preferred_fragment = (
            preferred[1]
            if preferred
            else analysis.groups[0].representative_fragment_id
        )

        self._set_group_and_instance(
            preferred_group,
            preferred_fragment,
            redraw=True,
        )
        self._selected_atom_id = selected_before
        self.redraw()

    def _set_group_and_instance(
        self,
        group_id: int,
        fragment_id: int | None,
        *,
        redraw: bool,
    ) -> None:
        if self.analysis is None:
            return

        group = next(
            (g for g in self.analysis.groups if g.group_id == int(group_id)),
            None,
        )
        if group is None:
            return

        group_label = next(
            (
                label
                for label, gid in self._group_lookup.items()
                if gid == group.group_id
            ),
            None,
        )
        if group_label is not None:
            self.group_var.set(group_label)

        self._instance_lookup = {}
        instance_labels = []
        for index, member_fragment_id in enumerate(
            group.member_fragment_ids,
            start=1,
        ):
            label = (
                f"Ligand {index}/{group.multiplicity}"
                f"  (fragment {member_fragment_id})"
            )
            instance_labels.append(label)
            self._instance_lookup[label] = member_fragment_id

        self.instance_combo["values"] = instance_labels

        if fragment_id not in group.member_fragment_ids:
            fragment_id = group.representative_fragment_id

        instance_label = next(
            (
                label
                for label, fid in self._instance_lookup.items()
                if fid == fragment_id
            ),
            instance_labels[0],
        )
        self.instance_var.set(instance_label)

        if redraw:
            self._rebuild_model()

    def _view_mode(self) -> str:
        return str(self.view_mode_var.get() or "Chemical")

    def _layout_mode(self) -> str:
        if self._view_mode() == "Geometry Projection":
            return LAYOUT_GEOMETRY_PROJECTION
        return LAYOUT_2D_CHEMICAL

    def _layout_engine(self) -> str:
        label = str(self.layout_engine_var.get() or "CoordGen")
        if label == "RDKit 2D":
            return ENGINE_RDKIT_2D
        if label == "Graph 2D":
            return ENGINE_GRAPH_2D
        return ENGINE_COORDGEN

    def _is_chemical_view(self) -> bool:
        return self._view_mode() == "Chemical"

    def _is_atom_map_view(self) -> bool:
        return self._view_mode() == "Atom Map"

    def _on_layout_changed(self, event=None) -> None:
        new_engine_label = str(self.layout_engine_var.get() or "CoordGen")
        engine_changed = new_engine_label != self._last_layout_engine_label

        if engine_changed and not self._prepare_layout_engine_change(
            self._last_layout_engine_label,
            new_engine_label,
        ):
            # ComboboxSelected fires after the displayed value changes, so put
            # the previous engine back when the user keeps manual 2D edits.
            self.layout_engine_var.set(self._last_layout_engine_label)
            return

        if engine_changed:
            self._last_layout_engine_label = new_engine_label

        self._sync_view_control_states()
        self.redraw()

    def _prepare_layout_engine_change(
        self,
        old_label: str,
        new_label: str,
    ) -> bool:
        """Resolve stored 2D depiction state before changing base engines.

        Cleanup-generated coordinates are disposable layout state and are
        cleared automatically.  True manual edits are protected by a warning.
        No chemistry, connectivity, bond order, or XYZ coordinate is changed.
        """
        if self.analysis is None or self.model is None:
            return True
        if self._layout_mode() == LAYOUT_GEOMETRY_PROJECTION:
            return True

        try:
            group = self.analysis.group_for_fragment_id(self.model.fragment_id)
            fragment_ids = tuple(int(value) for value in group.member_fragment_ids)
        except Exception:
            fragment_ids = (int(self.model.fragment_id),)

        stored = []
        manual = []
        for fragment_id in fragment_ids:
            coords = get_2d_override(self.state, fragment_id)
            if not coords:
                continue
            source = get_2d_override_source(self.state, fragment_id)
            stored.append(fragment_id)
            # Missing metadata can come from an older patch.  Treat that as a
            # manual layout so changing engines never destroys unknown user data
            # without confirmation.
            if source is None or is_manual_2d_override_source(source):
                manual.append(fragment_id)

        if manual:
            noun = "ligand" if len(manual) == 1 else "ligands"
            if not messagebox.askyesno(
                "Change 2D layout engine",
                f"Changing 2D layout from {old_label} to {new_label} will reset "
                f"manual 2D adjustments for {len(manual)} {noun} in this matching "
                "ligand group.\n\nChemistry and XYZ coordinates are not changed. Continue?",
                parent=self,
            ):
                return False

        if stored:
            self._push_edit_undo()
            # The base 2D engine is group-wide for the currently displayed
            # ligand family.  Clear cleanup/manual coordinate overlays so the
            # newly selected engine can actually become visible immediately.
            reset_2d_layout(
                self.state,
                self.analysis,
                fragment_id=self.model.fragment_id,
                include_matching=True,
            )

        if manual:
            self.status_var.set(
                f"Changed 2D layout to {new_label}; manual 2D adjustments were reset."
            )
        elif stored:
            self.status_var.set(
                f"Changed 2D layout to {new_label}; cleanup coordinates were rebuilt from the new engine."
            )
        return True

    def _on_editor_mode_changed(self, activated: str | None = None) -> None:
        # Editing modes are intentionally exclusive.  Bond editing changes the
        # chemical graph; 2D editing changes depiction coordinates only.
        if self.bond_edit_var.get() and self.edit_2d_var.get():
            if activated == "2d":
                self.bond_edit_var.set(False)
            else:
                self.edit_2d_var.set(False)
        if self._layout_mode() == LAYOUT_GEOMETRY_PROJECTION and self.edit_2d_var.get():
            self.edit_2d_var.set(False)
            self.status_var.set("2D Edit is available only in 2D Chemical/Atom Map views.")
            return
        self._sync_editor_context_bar()
        if self.bond_edit_var.get():
            self.status_var.set("Bond Edit: click a bond to set its type, remove it, or reset inference.")
        elif self.edit_2d_var.get():
            self.status_var.set("2D Edit: drag atoms to adjust depiction only; XYZ coordinates are unchanged.")
        else:
            self._update_selection_status()

    def _push_edit_undo(self) -> None:
        self._edit_undo_snapshot = snapshot_editor_state(self.state)

    def _undo_last_edit(self) -> None:
        if self._edit_undo_snapshot is None:
            self.status_var.set("Nothing to undo.")
            return
        snapshot = self._edit_undo_snapshot
        self._edit_undo_snapshot = None
        restore_editor_state(self.state, snapshot)
        self.refresh_structure()
        self.status_var.set("Last Chemical Map edit was undone.")

    def _on_group_changed(self, event=None) -> None:
        group_id = self._group_lookup.get(self.group_var.get())
        if group_id is None or self.analysis is None:
            return
        group = next(g for g in self.analysis.groups if g.group_id == group_id)
        self._set_group_and_instance(
            group_id,
            group.representative_fragment_id,
            redraw=True,
        )

    def _on_instance_changed(self, event=None) -> None:
        self._rebuild_model()

    def _rebuild_model(self) -> None:
        if self.graph is None or self.analysis is None:
            return

        group_id = self._group_lookup.get(self.group_var.get())
        fragment_id = self._instance_lookup.get(self.instance_var.get())
        if group_id is None or fragment_id is None:
            return

        try:
            self.model = build_chemical_map_model(
                self.graph,
                self.analysis,
                group_id=group_id,
                fragment_id=fragment_id,
                label_overrides=self.state.get("ref_label_overrides", {}) or {},
            )
            self.layout = layout_chemical_map(
                self.graph,
                self.model,
                mode=self._layout_mode(),
                engine=self._layout_engine(),
            )
        except Exception as exc:
            self.model = None
            self.layout = None
            self._draw_message(f"Could not build Chemical Map:\n{exc}")
            self.status_var.set(str(exc))
            return

        self.redraw()

    def redraw(self) -> None:
        if self.graph is None or self.analysis is None:
            return
        if self.model is None:
            self._rebuild_model()
            return

        self.layout = layout_chemical_map(
            self.graph,
            self.model,
            mode=self._layout_mode(),
            engine=self._layout_engine(),
        )

        ax = self.ax
        ax.clear()
        self._hover_annotation = None
        self._apply_plot_background()
        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="box")

        coordinates = dict(self.layout.coordinates)
        if self._layout_mode() == LAYOUT_2D_CHEMICAL and self.model is not None:
            override = get_2d_override(self.state, self.model.fragment_id)
            if override:
                coordinates.update(override)
        labels = self.state.get("ref_label_overrides", {}) or {}
        chemical_view = self._is_chemical_view()

        shared_context = getattr(getattr(self, "selection_hub", None), "context", None)
        shared_refs = tuple(getattr(shared_context, "ref_ids", ()) or ())
        if shared_refs:
            self._selected_atom_ids = tuple(int(value) for value in shared_refs)
            self._selected_atom_id = self._selected_atom_ids[0]
            self._selected_assignment = getattr(shared_context, "assignment", None)
        else:
            selected = self._selected_ref_from_main_tree()
            if selected is not None:
                self._selected_atom_id = selected
                self._selected_atom_ids = (int(selected),)

        selected_ids = set(self._selected_atom_ids)
        matching_current = (
            self._matching_atoms_in_current_fragment(self._selected_atom_id)
            if bool(self.show_matching_var.get())
            else set()
        ) - selected_ids

        visible = set()
        for atom in self.model.atoms:
            keep = (
                atom.element != "H"
                or bool(self.show_h_var.get())
                or atom.atom_id == self._selected_atom_id
                or (bool(self.show_labels_var.get()) and atom.atom_id in labels)
            )
            if keep:
                visible.add(atom.atom_id)

        for bond in self.model.bonds:
            if bond.atom1_id not in visible or bond.atom2_id not in visible:
                continue
            self._draw_bond(
                coordinates[bond.atom1_id],
                coordinates[bond.atom2_id],
                bond.order,
                bond_type=getattr(bond, "bond_type", "unknown"),
                chemical=chemical_view,
            )

        self._atom_xy = {}
        for atom in self.model.atoms:
            if atom.atom_id not in visible:
                continue
            x, y = coordinates[atom.atom_id]
            self._atom_xy[atom.atom_id] = (x, y)

            color = _ELEMENT_COLORS.get(atom.element, _DEFAULT_COLOR)
            is_matching = atom.atom_id in matching_current
            is_selected = atom.atom_id in selected_ids
            is_primary = atom.atom_id == self._selected_atom_id
            is_donor = bool(self.show_donors_var.get()) and (
                atom.is_donor or atom.is_haptic
            )

            if chemical_view:
                # True skeletal-style rendering: carbon atoms are vertices, not
                # filled nodes.  Heteroatoms are text glyphs and all semantic
                # information is layered as quiet halos around the RDKit layout.
                if is_matching:
                    ax.scatter(
                        [x], [y], s=120, facecolors="none",
                        edgecolors="#2563EB", linewidths=1.6, zorder=2.7,
                    )
                if is_donor:
                    ax.scatter(
                        [x], [y], s=145, facecolors="none",
                        edgecolors="#6D5BD0", linewidths=1.2, zorder=2.8,
                    )
                if is_selected:
                    ax.scatter(
                        [x], [y], s=175, facecolors="none",
                        edgecolors="#D97706", linewidths=2.0, zorder=2.9,
                    )
                if is_primary:
                    ax.scatter(
                        [x], [y], s=215, facecolors="none",
                        edgecolors="#B45309", linewidths=2.4, zorder=3.0,
                    )

                atom_text = self._display_atom_text(atom)
                if atom_text:
                    ax.text(
                        x, y, atom_text,
                        fontsize=10.5 if atom.element != "H" else 8.5,
                        fontweight="bold" if atom.element != "H" else "normal",
                        color=(
                            self._plot_foreground()
                            if atom.element in {"C", "H"}
                            else color
                        ),
                        ha="center", va="center", zorder=5,
                        bbox=dict(
                            boxstyle="round,pad=0.10",
                            facecolor=self.background_color, edgecolor="none", alpha=0.92,
                        ),
                    )

                detail_parts = []
                if self.show_labels_var.get() and atom.atom_id in labels:
                    detail_parts.append(str(labels[atom.atom_id]))
                if self.show_ids_var.get():
                    detail_parts.append(f"#{atom.atom_id}")
                if detail_parts:
                    ax.text(
                        x, y - 0.040, " ".join(detail_parts),
                        fontsize=7, ha="center", va="top", zorder=6,
                        bbox=dict(
                            boxstyle="round,pad=0.08",
                            facecolor=self.background_color, edgecolor="none", alpha=0.85,
                        ),
                    )
                if atom.is_haptic and is_donor:
                    ax.text(
                        x + 0.024, y + 0.032, "η",
                        fontsize=7, fontweight="bold",
                        ha="left", va="bottom", zorder=6,
                    )
                continue

            # Atom Map and Geometry Projection intentionally retain explicit
            # colored atom nodes for Ref-centric inspection.
            if self._layout_mode() == LAYOUT_2D_CHEMICAL:
                if atom.element == "C":
                    size = 42
                elif atom.element == "H":
                    size = 72
                else:
                    size = 135
                linewidth = 1.0
            else:
                size = 180 if atom.element != "H" else 110
                linewidth = 1.2

            edgecolor = self._plot_foreground()
            if is_matching:
                size *= 1.35
                linewidth = 2.2
                edgecolor = "#2563EB"
            if is_selected:
                size *= 1.45
                linewidth = 2.5
                edgecolor = "#D97706"
            if is_primary:
                size *= 1.15
                linewidth = 3.2
                edgecolor = "#B45309"

            ax.scatter(
                [x], [y], s=size, c=[color], edgecolors=edgecolor,
                linewidths=linewidth, zorder=3,
            )

            if is_donor:
                ax.scatter(
                    [x], [y], s=size * 1.75, facecolors="none",
                    edgecolors="#6D5BD0", linewidths=1.2, zorder=2.9,
                )
                if atom.is_haptic:
                    ax.text(
                        x + 0.025, y + 0.035, "η", fontsize=7,
                        fontweight="bold", color=self._plot_foreground(),
                        ha="left", va="bottom", zorder=5,
                    )

            text_parts = []
            if self.show_labels_var.get() and atom.atom_id in labels:
                text_parts.append(str(labels[atom.atom_id]))
            else:
                atom_text = self._display_atom_text(atom)
                if atom_text:
                    text_parts.append(atom_text)
            if self.show_ids_var.get():
                text_parts.append(f"#{atom.atom_id}")
            if text_parts:
                ax.text(
                    x, y - 0.042, " ".join(text_parts),
                    fontsize=8, color=self._plot_foreground(),
                    ha="center", va="top", zorder=5,
                )

        if bool(self.show_nmr_sites_var.get()):
            self._draw_nmr_site_markers(coordinates, visible, selected_ids)

        if self._atom_xy:
            xs = [xy[0] for xy in self._atom_xy.values()]
            ys = [xy[1] for xy in self._atom_xy.values()]
            margin = 0.20
            ax.set_xlim(min(xs) - margin, max(xs) + margin)
            ax.set_ylim(min(ys) - margin, max(ys) + margin)

        summary = (
            f"{self.model.formula} ×{self.model.multiplicity}"
            f" | κ{self.model.denticity}"
            f" | η{self.model.hapticity}"
            f" | metals: {', '.join(map(str, self.model.attached_metal_ids)) or '-'}"
            f" | view: {self._view_mode()}"
            f" | layout: {self.layout.backend}"
        )
        if (
            self.layout.backend == "Graph 2D"
            and self._layout_mode() == LAYOUT_2D_CHEMICAL
            and self._layout_engine() != ENGINE_GRAPH_2D
        ):
            summary += " | RDKit unavailable: using fallback layout (Graph 2D)"
        if self.model.ambiguous_correspondence:
            summary += " | mapping ambiguous"
        self.summary_var.set(summary)
        self._update_selection_status()

        self.canvas.draw_idle()

    def _display_atom_text(self, atom) -> str:
        if self._is_chemical_view():
            # Skeletal convention: carbon is implied at vertices/endpoints.
            if atom.element == "C":
                return ""
            return atom.element
        return atom.element

    @staticmethod
    def _unicode_subscript(value: int) -> str:
        table = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return str(int(value)).translate(table)

    def _draw_nmr_site_markers(self, coordinates, visible, selected_ids=None) -> None:
        if self.nmr_site_analysis is None or self.model is None:
            return
        for site in self.nmr_site_analysis.sites_for_fragment(self.model.fragment_id):
            if site.site_type not in {"methyl_H", "tert_butyl_H"}:
                continue
            anchor = site.anchor_ref_id
            if anchor is None or anchor not in visible or anchor not in coordinates:
                continue
            x, y = coordinates[anchor]
            is_tbu = site.site_type == "tert_butyl_H"
            selected_ids = set(selected_ids or ())
            site_selected = bool(site.member_ref_ids) and set(site.member_ref_ids).issubset(selected_ids)
            if self._is_chemical_view():
                marker_size = (165 if is_tbu else 135) if site_selected else (135 if is_tbu else 110)
            else:
                marker_size = (260 if is_tbu else 210) if site_selected else (215 if is_tbu else 170)
            self.ax.scatter(
                [x],
                [y],
                s=marker_size,
                marker="s" if is_tbu else "o",
                facecolors="none",
                edgecolors="#D97706" if site_selected else "#0F766E",
                linewidths=2.2 if site_selected else 1.2,
                zorder=2.8,
            )
            label = "H₉" if is_tbu else "H₃"
            self.ax.text(
                x + 0.027,
                y + 0.038,
                label,
                fontsize=6.5,
                fontweight="bold",
                color=self._plot_foreground(),
                ha="left",
                va="bottom",
                zorder=6,
            )

    def _matching_atoms_in_current_fragment(self, atom_id) -> set[int]:
        if (
            atom_id is None
            or self.correspondence_analysis is None
            or self.model is None
        ):
            return set()
        try:
            result = self.correspondence_analysis.matching_atoms(int(atom_id))
        except Exception:
            return set()
        for item in result.candidates:
            if item.fragment_id == self.model.fragment_id:
                return set(item.atom_ids) - {int(atom_id)}
        return set()

    def _nmr_sites_for_atom(self, atom_id: int):
        if self.nmr_site_analysis is None or self.model is None:
            return ()
        result = []
        for site in self.nmr_site_analysis.sites_for_fragment(self.model.fragment_id):
            if site.site_type == "atom":
                continue
            if site.anchor_ref_id == atom_id or atom_id in site.member_ref_ids:
                result.append(site)
        return tuple(result)

    def _update_selection_status(self) -> None:
        atom_id = self._selected_atom_id
        if atom_id is None:
            self.status_var.set("Hover for atom/Ref details; click an atom to select it.")
            return

        parts = []
        if self._selected_assignment:
            parts.append(f"Assignment {self._selected_assignment}")

        selected_count = len(self._selected_atom_ids)
        if selected_count > 1:
            parts.append(f"{selected_count} structural Refs selected (primary Ref {atom_id})")
        else:
            parts.append(f"Selected Ref {atom_id}")

        sites = self._nmr_sites_for_atom(atom_id)
        if sites:
            labels = []
            for site in sites:
                if site.site_type == "methyl_H":
                    labels.append("CH3-H3 site")
                elif site.site_type == "tert_butyl_H":
                    labels.append("tBu-H9 site")
            if labels:
                parts.append(", ".join(labels))

        if self.correspondence_analysis is not None:
            try:
                match = self.correspondence_analysis.matching_atoms(atom_id)
                other_fragments = [
                    item
                    for item in match.candidates
                    if self.model is not None
                    and item.fragment_id != self.model.fragment_id
                ]
                candidate_count = sum(len(item.atom_ids) for item in other_fragments)
                if candidate_count:
                    qualifier = "ambiguous" if match.ambiguous else "matching"
                    parts.append(
                        f"{candidate_count} {qualifier} position(s) on other ligand(s)"
                    )
            except Exception:
                pass
        self.status_var.set(" | ".join(parts))

    def _draw_bond(
        self,
        p1,
        p2,
        order,
        *,
        bond_type: str = "unknown",
        chemical: bool = False,
    ) -> None:
        x1, y1 = p1
        x2, y2 = p2
        try:
            numeric_order = float(order) if order is not None else 1.0
        except Exception:
            numeric_order = 1.0

        bond_type = str(bond_type or "unknown").lower()
        if bond_type == "triple":
            line_count = 3
        elif bond_type == "double":
            line_count = 2
        elif bond_type in {"aromatic", "delocalized"}:
            line_count = 2
        else:
            line_count = 3 if numeric_order >= 2.5 else 2 if numeric_order >= 1.75 else 1

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy) or 1.0
        nx = -dy / length
        ny = dx / length
        offsets = {
            1: (0.0,),
            2: (-0.008, 0.008),
            3: (-0.012, 0.0, 0.012),
        }[line_count]

        for index, offset in enumerate(offsets):
            linestyle = "-"
            linewidth = 1.55 if chemical else 1.35
            alpha = 1.0
            if bond_type == "aromatic" and index == len(offsets) - 1:
                linestyle = (0, (3.0, 2.0))
                linewidth *= 0.85
            elif bond_type == "delocalized" and index == len(offsets) - 1:
                linestyle = (0, (1.4, 2.2))
                linewidth *= 0.78
                alpha = 0.85
            self.ax.plot(
                [x1 + nx * offset, x2 + nx * offset],
                [y1 + ny * offset, y2 + ny * offset],
                color=self._plot_foreground() if chemical else self._plot_muted(),
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                solid_capstyle="round",
                zorder=1,
            )

    def _nearest_atom(self, event) -> int | None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return None
        if not self._atom_xy:
            return None

        nearest = None
        nearest_dist = float("inf")
        for atom_id, (x, y) in self._atom_xy.items():
            dist = math.hypot(event.xdata - x, event.ydata - y)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = atom_id

        return nearest if nearest_dist <= 0.09 else None

    def _nearest_bond(self, event) -> tuple[int, int] | None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return None
        if self.model is None:
            return None
        px, py = float(event.xdata), float(event.ydata)
        best = None
        best_dist = float("inf")
        for bond in self.model.bonds:
            a = self._atom_xy.get(bond.atom1_id)
            b = self._atom_xy.get(bond.atom2_id)
            if a is None or b is None:
                continue
            x1, y1 = a
            x2, y2 = b
            dx, dy = x2 - x1, y2 - y1
            denom = dx * dx + dy * dy
            if denom <= 1e-16:
                dist = math.hypot(px - x1, py - y1)
            else:
                t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
                qx, qy = x1 + t * dx, y1 + t * dy
                dist = math.hypot(px - qx, py - qy)
            if dist < best_dist:
                best_dist = dist
                best = tuple(sorted((bond.atom1_id, bond.atom2_id)))
        return best if best_dist <= 0.045 else None

    def _on_plot_click(self, event) -> None:
        if self.model is None:
            return

        if self.bond_edit_var.get():
            bond = self._nearest_bond(event)
            if bond is not None:
                self._selected_bond = bond
                self._show_bond_edit_menu(event, bond)
                return

        atom_id = self._nearest_atom(event)
        if atom_id is None:
            return

        if self.edit_2d_var.get() and self._layout_mode() == LAYOUT_2D_CHEMICAL:
            self._push_edit_undo()
            self._drag_atom_id = int(atom_id)
            base = dict(self.layout.coordinates) if self.layout is not None else {}
            existing = get_2d_override(self.state, self.model.fragment_id) or {}
            base.update(existing)
            self._drag_base_coordinates = base
            return

        self._selected_atom_id = atom_id
        self._selected_atom_ids = (int(atom_id),)
        self._selected_assignment = None
        publish_ref_selection(
            self.state,
            [atom_id],
            source="chemical_map",
        )
        self.redraw()

    def _on_plot_release(self, event) -> None:
        if self._drag_atom_id is not None:
            self._drag_atom_id = None
            self._drag_base_coordinates = {}
            self.status_var.set("2D depiction position updated. XYZ coordinates were not changed.")

    def _on_plot_motion(self, event) -> None:
        if (
            self._drag_atom_id is not None
            and self.model is not None
            and event.inaxes is self.ax
            and event.xdata is not None
            and event.ydata is not None
        ):
            set_atom_2d_position(
                self.state,
                self.model.fragment_id,
                self._drag_base_coordinates or self._atom_xy,
                self._drag_atom_id,
                float(event.xdata),
                float(event.ydata),
            )
            self.redraw()
            return

        atom_id = self._nearest_atom(event)
        if atom_id is None or self.model is None:
            self._hide_hover()
            return
        self._update_hover_for_atom(atom_id)

    def _update_hover_for_atom(self, atom_id: int) -> None:
        atom = next(
            (item for item in self.model.atoms if item.atom_id == atom_id),
            None,
        )
        if atom is None:
            self._hide_hover()
            return

        role = []
        if atom.is_donor:
            role.append("donor")
        if atom.is_haptic:
            role.append("haptic")
        role_text = f" | {', '.join(role)}" if role else ""
        site_text = ""
        grouped_sites = self._nmr_sites_for_atom(atom_id)
        if grouped_sites:
            names = []
            for site in grouped_sites:
                if site.site_type == "methyl_H":
                    names.append("CH3-H3")
                elif site.site_type == "tert_butyl_H":
                    names.append("tBu-H9")
            if names:
                site_text = f" | NMR: {', '.join(names)}"

        if self._hover_annotation is None:
            self._hover_annotation = self.ax.annotate(
                "",
                xy=(0, 0),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="round",
                    fc=self.background_color,
                    ec=self._plot_muted(),
                    alpha=0.94,
                ),
                arrowprops=dict(arrowstyle="->", color=self._plot_foreground()),
                color=self._plot_foreground(),
                fontsize=8,
                zorder=10,
            )

        x, y = self._atom_xy[atom_id]
        self._hover_annotation.xy = (x, y)
        self._hover_annotation.set_text(
            f"{atom.display_label} | Ref {atom.atom_id}{role_text}{site_text}"
        )
        self._hover_annotation.set_visible(True)
        self.canvas.draw_idle()

    def _show_bond_edit_menu(self, event, bond: tuple[int, int]) -> None:
        menu = tk.Menu(self, tearoff=False)
        a, b = bond
        menu.add_command(label=f"Bond Ref {a}–{b}", state="disabled")
        menu.add_separator()
        for label, bond_type in (
            ("Single", BOND_SINGLE),
            ("Double", BOND_DOUBLE),
            ("Triple", BOND_TRIPLE),
            ("Aromatic", BOND_AROMATIC),
            ("Delocalized", BOND_DELOCALIZED),
            ("Unknown", BOND_UNKNOWN),
        ):
            menu.add_command(
                label=label,
                command=lambda value=bond_type: self._apply_bond_type(bond, value),
            )
        menu.add_separator()
        menu.add_command(label="Remove bond", command=lambda: self._remove_bond(bond))
        menu.add_command(label="Reset to inferred", command=lambda: self._reset_bond(bond))
        try:
            gui = getattr(event, "guiEvent", None)
            x_root = int(getattr(gui, "x_root", self.winfo_pointerx()))
            y_root = int(getattr(gui, "y_root", self.winfo_pointery()))
            menu.tk_popup(x_root, y_root)
        finally:
            try:
                menu.grab_release()
            except Exception:
                pass

    def _apply_bond_type(self, bond: tuple[int, int], bond_type: str) -> None:
        if self.analysis is None or self.model is None:
            return
        self._push_edit_undo()
        outcome = set_bond_type_in_state(
            self.state,
            self.analysis,
            fragment_id=self.model.fragment_id,
            atom1_id=bond[0],
            atom2_id=bond[1],
            bond_type=bond_type,
            propagate_matching=bool(self.propagate_matching_var.get()),
        )
        self.refresh_structure()
        message = f"Set {len(outcome.edited_pairs)} bond(s) to {bond_type}."
        if outcome.warning:
            message += f" {outcome.warning}"
        self.status_var.set(message)

    def _remove_bond(self, bond: tuple[int, int]) -> None:
        if self.analysis is None or self.model is None:
            return
        self._push_edit_undo()
        outcome = remove_bond_in_state(
            self.state,
            self.analysis,
            fragment_id=self.model.fragment_id,
            atom1_id=bond[0],
            atom2_id=bond[1],
            propagate_matching=bool(self.propagate_matching_var.get()),
        )
        self.refresh_structure()
        message = f"Removed {len(outcome.edited_pairs)} bond(s)."
        if outcome.warning:
            message += f" {outcome.warning}"
        self.status_var.set(message)

    def _reset_bond(self, bond: tuple[int, int]) -> None:
        if self.analysis is None or self.model is None:
            return
        self._push_edit_undo()
        outcome = reset_bond_override_in_state(
            self.state,
            self.analysis,
            fragment_id=self.model.fragment_id,
            atom1_id=bond[0],
            atom2_id=bond[1],
            propagate_matching=bool(self.propagate_matching_var.get()),
        )
        self.refresh_structure()
        message = f"Reset {len(outcome.edited_pairs)} bond override(s) to inferred topology."
        if outcome.warning:
            message += f" {outcome.warning}"
        self.status_var.set(message)

    def _add_bond_from_selection(self) -> None:
        if self.analysis is None or self.model is None:
            return
        refs = tuple(int(value) for value in self._selected_atom_ids)
        if len(refs) != 2:
            messagebox.showinfo(
                "Add bond",
                "Select exactly two atoms first (for example through shared selection), then try again.",
                parent=self,
            )
            return
        if not set(refs).issubset(set(self.model.atom_ids)):
            messagebox.showwarning(
                "Add bond",
                "Both selected atoms must belong to the currently displayed ligand instance.",
                parent=self,
            )
            return
        self._push_edit_undo()
        outcome = add_bond_in_state(
            self.state,
            self.analysis,
            fragment_id=self.model.fragment_id,
            atom1_id=refs[0],
            atom2_id=refs[1],
            bond_type=BOND_SINGLE,
            propagate_matching=bool(self.propagate_matching_var.get()),
        )
        self.refresh_structure()
        self.status_var.set(f"Added {len(outcome.edited_pairs)} single bond(s).")

    def _cleanup_structure(self) -> None:
        if self.graph is None or self.analysis is None or self.model is None:
            return
        if self._layout_mode() != LAYOUT_2D_CHEMICAL:
            messagebox.showinfo(
                "Cleanup Structure",
                "Cleanup Structure affects only the 2D depiction. Switch to Chemical or Atom Map first.",
                parent=self,
            )
            return
        existing = get_2d_override(self.state, self.model.fragment_id)
        if existing and not messagebox.askyesno(
            "Cleanup Structure",
            "Rebuild the 2D layout from the current chemical graph?\n\nManual 2D adjustments for the affected ligand(s) will be reset. Chemistry and XYZ coordinates are not changed.",
            parent=self,
        ):
            return
        self._push_edit_undo()
        outcome = cleanup_structure_2d(
            self.state,
            self.graph,
            self.analysis,
            group_id=self.model.group_id,
            fragment_id=self.model.fragment_id,
            engine=self._layout_engine(),
            propagate_matching=bool(self.propagate_matching_var.get()),
        )
        self.redraw()
        message = f"Cleaned 2D depiction for {len(outcome.fragment_ids)} ligand instance(s)."
        if outcome.warning:
            message += f" {outcome.warning}"
        self.status_var.set(message)

    def _transform_2d(
        self,
        rotate_degrees: float = 0.0,
        *,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
    ) -> None:
        if self.model is None or self.layout is None:
            return
        if self._layout_mode() != LAYOUT_2D_CHEMICAL:
            self.status_var.set("2D transforms are available only in 2D views.")
            return
        self._push_edit_undo()
        base = dict(self.layout.coordinates)
        existing = get_2d_override(self.state, self.model.fragment_id) or {}
        base.update(existing)
        transform_2d_layout(
            self.state,
            self.model.fragment_id,
            base,
            rotate_degrees=rotate_degrees,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
        )
        self.redraw()
        self.status_var.set("Updated 2D depiction transform; XYZ coordinates are unchanged.")

    def _reset_2d_layout(self) -> None:
        if self.analysis is None or self.model is None:
            return
        self._push_edit_undo()
        targets = reset_2d_layout(
            self.state,
            self.analysis,
            fragment_id=self.model.fragment_id,
            include_matching=bool(self.propagate_matching_var.get()),
        )
        self.redraw()
        self.status_var.set(f"Reset stored 2D layout for {len(targets)} ligand instance(s).")

    def _on_plot_leave(self, event=None) -> None:
        self._hide_hover()

    def _hide_hover(self) -> None:
        if self._hover_annotation is not None and self._hover_annotation.get_visible():
            self._hover_annotation.set_visible(False)
            self.canvas.draw_idle()

    def _selected_ref_from_main_tree(self) -> int | None:
        tree = self.state.get("tree")
        if tree is None:
            return None
        try:
            selection = tree.selection()
            if not selection:
                return None
            values = tree.item(selection[0], "values")
            return int(values[0])
        except Exception:
            return None

    def _select_ref_in_main_table(self, ref_id: int) -> None:
        tree = self.state.get("tree")
        row_by_id = self.state.get("row_by_id", {}) or {}
        item = row_by_id.get(int(ref_id))

        if tree is None or item is None:
            self.status_var.set(
                f"Ref {ref_id} is not currently present in the PCS table."
            )
            return

        try:
            tree.selection_set(item)
            tree.focus(item)
            tree.see(item)
            self.status_var.set(f"Selected Ref {ref_id} in PCS Analyzer.")
        except Exception as exc:
            self.status_var.set(str(exc))

    def _on_shared_selection(self, context) -> None:
        if self.analysis is None:
            return

        refs = tuple(int(value) for value in (getattr(context, "ref_ids", ()) or ()))
        self._selected_atom_ids = refs
        self._selected_assignment = getattr(context, "assignment", None)

        ref_id = context.primary_ref_id
        if ref_id is None:
            self._selected_atom_id = None
            self.redraw()
            return

        self._selected_atom_id = int(ref_id)
        target = find_model_for_atom(self.analysis, ref_id)
        if target is not None:
            group_id, fragment_id = target
            current_group = self._group_lookup.get(self.group_var.get())
            current_fragment = self._instance_lookup.get(self.instance_var.get())
            if current_group != group_id or current_fragment != fragment_id:
                self._set_group_and_instance(
                    group_id,
                    fragment_id,
                    redraw=False,
                )
                self._rebuild_model()
                return

        self.redraw()

    def _draw_message(self, message: str) -> None:
        self.ax.clear()
        self._apply_plot_background()
        self.ax.set_axis_off()
        self.ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=self.ax.transAxes,
            wrap=True,
            color=self._plot_foreground(),
        )
        self.canvas.draw_idle()


def open_chemical_map_window(state: dict):
    """Open or focus the independent Chemical Map window."""
    existing = state.get("chemical_map_win")
    if existing is not None:
        try:
            if existing.winfo_exists():
                existing.lift()
                existing.focus_force()
                existing.refresh_structure()
                return existing
        except Exception:
            pass

    root = state.get("root")
    if root is None:
        raise RuntimeError("PCS Analyzer root window is not available")

    win = ChemicalMapWindow(root, state=state)
    state["chemical_map_win"] = win
    return win


__all__ = ["ChemicalMapWindow", "open_chemical_map_window"]
