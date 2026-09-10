# ui/averaging_settings_window.py
# PCS_PATCH_AVERAGING_SETTINGS_PHASE1
# PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE2
# PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE3
# PCS_PATCH_TORSIONAL_ENSEMBLE_PHASE4

from __future__ import annotations

from copy import deepcopy
import tkinter as tk
from tkinter import ttk, messagebox

from logic.func_group_collapse import build_bond_graph, find_ax3_groups
from logic.torsional_ensemble import (
    DEFAULT_PLANARITY_THRESHOLD,
    build_manual_torsion_group,
    detect_planar_rotors,
    merge_group_settings,
    validate_independent_groups,
)


_SAMPLE_PRESETS = {
    "Fast · 12 points": 12,
    "Normal · 24 points": 24,
    "Fine · 72 points": 72,
}


def _get_bool(state: dict, key: str, default: bool) -> bool:
    var = state.get(key)
    if var is None:
        return bool(default)
    try:
        return bool(var.get())
    except Exception:
        return bool(default)


def _get_text(state: dict, key: str, default: str) -> str:
    var = state.get(key)
    if var is None:
        return str(default)
    try:
        return str(var.get())
    except Exception:
        return str(default)


def _get_float(state: dict, key: str, default: float) -> float:
    var = state.get(key)
    if var is None:
        try:
            return float(state.get(key, default))
        except Exception:
            return float(default)
    try:
        return float(var.get())
    except Exception:
        return float(default)


def _set_bool(state: dict, key: str, value: bool) -> None:
    var = state.get(key)
    if var is None:
        state[key] = tk.BooleanVar(master=state.get("root"), value=bool(value))
        return
    try:
        var.set(bool(value))
    except Exception:
        pass


def _set_text(state: dict, key: str, value: str) -> None:
    var = state.get(key)
    if var is None:
        state[key] = tk.StringVar(master=state.get("root"), value=str(value))
        return
    try:
        var.set(str(value))
    except Exception:
        pass


def _set_float(state: dict, key: str, value: float) -> None:
    var = state.get(key)
    if var is None:
        state[key] = tk.DoubleVar(master=state.get("root"), value=float(value))
        return
    try:
        var.set(float(value))
    except Exception:
        pass


def _detect_local_groups(state: dict) -> list[dict]:
    """Return read-only CH3 / CF3 detections from the current raw structure."""
    atom_data = state.get("atom_data_raw") or state.get("atom_data") or []
    if not atom_data:
        return []

    ref_ids = state.get("atom_ids_raw") or list(range(1, len(atom_data) + 1))
    if len(ref_ids) != len(atom_data):
        ref_ids = list(range(1, len(atom_data) + 1))

    neigh = build_bond_graph(atom_data, scale=1.10)
    groups: list[dict] = []

    specs = (("CH₃", "H", 1), ("CF₃", "F", 1))
    for label, ligand_elem, nonligand_count in specs:
        detected = find_ax3_groups(
            atom_data,
            neigh,
            center_elem="C",
            ligand_elem=ligand_elem,
            require_center_nonligand_count=nonligand_count,
        )
        for group in detected:
            groups.append(
                {
                    "type": label,
                    "center_ref": int(ref_ids[group.a_idx]),
                    "member_refs": tuple(int(ref_ids[i]) for i in group.x_idx),
                }
            )

    groups.sort(key=lambda g: (g["center_ref"], g["type"]))
    return groups


def _format_ref_sequence(refs) -> str:
    vals = sorted({int(x) for x in refs})
    if not vals:
        return "—"
    if len(vals) >= 2 and vals == list(range(vals[0], vals[-1] + 1)):
        return f"{vals[0]}–{vals[-1]}"
    return ",".join(str(x) for x in vals)


def _format_ring(group: dict) -> str:
    cycles = group.get("ring_cycles") or ()
    if len(cycles) > 1:
        return " / ".join(_format_ref_sequence(c) for c in cycles)
    return _format_ref_sequence(group.get("ring_atoms") or ())


def _sampling_label(n: int) -> str:
    n = int(n)
    for label, count in _SAMPLE_PRESETS.items():
        if count == n:
            return label
    return "Custom"


def open_averaging_settings_window(state: dict):
    """Open/focus the single Averaging Settings window."""
    root = state.get("root")
    if root is None:
        return None

    existing = state.get("averaging_settings_window")
    if existing is not None:
        try:
            if existing.winfo_exists():
                existing.deiconify()
                existing.lift()
                existing.focus_force()
                return existing
        except Exception:
            pass

    top = tk.Toplevel(root)
    state["averaging_settings_window"] = top
    top.title("Averaging Settings")
    top.geometry("820x760")
    top.minsize(720, 650)
    try:
        top.transient(root)
    except Exception:
        pass

    # Transactional dialog variables: state changes only when Apply is pressed.
    local_enabled = tk.BooleanVar(value=_get_bool(state, "symavg_enabled_var", False))
    methyl_enabled = tk.BooleanVar(value=_get_bool(state, "symavg_methyl_enabled_var", True))
    cf3_enabled = tk.BooleanVar(value=_get_bool(state, "symavg_cf3_enabled_var", True))
    keep_original = tk.BooleanVar(value=_get_bool(state, "symavg_keep_original_var", False))

    rot_enabled = tk.BooleanVar(value=_get_bool(state, "torsion_avg_enabled_var", False))
    rot_keep_reference = tk.BooleanVar(value=_get_bool(state, "torsion_avg_keep_reference_var", False))
    detection_mode = tk.StringVar(value=_get_text(state, "torsion_avg_detection_mode_var", "auto"))
    planarity = tk.StringVar(value=f"{_get_float(state, 'torsion_avg_planarity_var', DEFAULT_PLANARITY_THRESHOLD):.3g}")

    rot_groups: list[dict] = deepcopy(state.get("torsion_avg_groups", []) or [])
    selected_iid: str | None = None
    updating_detail = False

    outer = ttk.Frame(top, padding=10)
    outer.pack(fill="both", expand=True)
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(0, weight=1)

    body = ttk.Frame(outer)
    body.grid(row=0, column=0, sticky="nsew")
    body.columnconfigure(0, weight=1)
    body.rowconfigure(1, weight=1)

    # ------------------------------------------------------------------
    # Local symmetry averaging
    # ------------------------------------------------------------------
    local_box = ttk.LabelFrame(body, text="Local symmetry averaging", padding=8)
    local_box.grid(row=0, column=0, sticky="ew")
    local_box.columnconfigure(1, weight=1)

    ttk.Checkbutton(local_box, text="Enable", variable=local_enabled).grid(row=0, column=0, sticky="w")
    local_opts = ttk.Frame(local_box)
    local_opts.grid(row=0, column=1, sticky="w", padx=(12, 0))
    ttk.Label(local_opts, text="Groups:").pack(side="left")
    ttk.Checkbutton(local_opts, text="CH₃", variable=methyl_enabled).pack(side="left", padx=(8, 2))
    ttk.Checkbutton(local_opts, text="CF₃", variable=cf3_enabled).pack(side="left", padx=2)
    ttk.Checkbutton(local_box, text="Keep original atoms", variable=keep_original).grid(
        row=0, column=2, sticky="e", padx=(10, 0)
    )

    local_tree = ttk.Treeview(
        local_box,
        columns=("type", "center", "members"),
        show="headings",
        height=3,
    )
    local_tree.heading("type", text="Type")
    local_tree.heading("center", text="Center")
    local_tree.heading("members", text="Members")
    local_tree.column("type", width=80, stretch=False)
    local_tree.column("center", width=90, stretch=False)
    local_tree.column("members", width=360, stretch=True)
    local_tree.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(7, 0))

    local_status = tk.StringVar(value="")
    ttk.Label(local_box, textvariable=local_status).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

    # ------------------------------------------------------------------
    # Rotational ensemble averaging
    # ------------------------------------------------------------------
    rot_box = ttk.LabelFrame(body, text="Rotational ensemble averaging", padding=8)
    rot_box.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
    rot_box.columnconfigure(0, weight=1)
    rot_box.rowconfigure(3, weight=1)

    header = ttk.Frame(rot_box)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(4, weight=1)
    ttk.Checkbutton(header, text="Enable", variable=rot_enabled).grid(row=0, column=0, sticky="w")
    ttk.Label(header, text="Detection:").grid(row=0, column=1, sticky="w", padx=(16, 4))
    ttk.Radiobutton(header, text="Auto planar rings", variable=detection_mode, value="auto").grid(row=0, column=2, sticky="w")
    ttk.Radiobutton(header, text="Manual torsion", variable=detection_mode, value="manual").grid(row=0, column=3, sticky="w", padx=(8, 0))
    ttk.Label(header, text="Planarity RMS ≤").grid(row=0, column=5, sticky="e", padx=(12, 4))
    ttk.Entry(header, textvariable=planarity, width=6).grid(row=0, column=6, sticky="e")
    ttk.Label(header, text="Å").grid(row=0, column=7, sticky="w", padx=(3, 0))

    manual = ttk.Frame(rot_box)
    manual.grid(row=1, column=0, sticky="ew", pady=(6, 0))
    ttk.Label(manual, text="Manual axis A:").pack(side="left")
    man_a = ttk.Entry(manual, width=7)
    man_a.pack(side="left", padx=(4, 8))
    ttk.Label(manual, text="B:").pack(side="left")
    man_b = ttk.Entry(manual, width=7)
    man_b.pack(side="left", padx=(4, 8))
    ttk.Label(manual, text="Moving-side Ref:").pack(side="left")
    man_side = ttk.Entry(manual, width=7)
    man_side.pack(side="left", padx=(4, 8))

    rot_status = tk.StringVar(value="")

    tree_frame = ttk.Frame(rot_box)
    tree_frame.grid(row=3, column=0, sticky="nsew", pady=(7, 0))
    tree_frame.columnconfigure(0, weight=1)
    tree_frame.rowconfigure(0, weight=1)

    rot_tree = ttk.Treeview(
        tree_frame,
        columns=("use", "group", "label", "ring", "axis"),
        show="headings",
        height=7,
        selectmode="browse",
    )
    rot_tree.heading("use", text="Use")
    rot_tree.heading("group", text="Group")
    rot_tree.heading("label", text="Type")
    rot_tree.heading("ring", text="Ring")
    rot_tree.heading("axis", text="Axis")
    rot_tree.column("use", width=42, stretch=False, anchor="center")
    rot_tree.column("group", width=62, stretch=False, anchor="center")
    rot_tree.column("label", width=220, stretch=True, anchor="w")
    rot_tree.column("ring", width=180, stretch=True, anchor="w")
    rot_tree.column("axis", width=90, stretch=False, anchor="center")
    rot_tree.grid(row=0, column=0, sticky="nsew")
    rot_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=rot_tree.yview)
    rot_scroll.grid(row=0, column=1, sticky="ns")
    rot_tree.configure(yscrollcommand=rot_scroll.set)

    # Selected group controls stay in this same settings window (no nested dialog).
    detail = ttk.LabelFrame(rot_box, text="Selected group", padding=7)
    detail.grid(row=4, column=0, sticky="ew", pady=(7, 0))
    detail.columnconfigure(7, weight=1)

    selected_label = tk.StringVar(value="No group selected")
    ttk.Label(detail, textvariable=selected_label).grid(row=0, column=0, columnspan=8, sticky="w", pady=(0, 5))

    range_mode = tk.StringVar(value="full")
    ttk.Radiobutton(detail, text="Full rotation", variable=range_mode, value="full").grid(row=1, column=0, sticky="w")
    ttk.Radiobutton(detail, text="Restricted", variable=range_mode, value="restricted").grid(row=1, column=1, sticky="w", padx=(8, 0))
    ttk.Label(detail, text="Range:").grid(row=1, column=2, sticky="e", padx=(14, 4))
    range_start = ttk.Entry(detail, width=7)
    range_start.grid(row=1, column=3, sticky="w")
    ttk.Label(detail, text="to").grid(row=1, column=4, padx=4)
    range_end = ttk.Entry(detail, width=7)
    range_end.grid(row=1, column=5, sticky="w")
    ttk.Label(detail, text="°").grid(row=1, column=6, sticky="w", padx=(2, 0))

    ttk.Label(detail, text="Sampling:").grid(row=2, column=0, sticky="w", pady=(6, 0))
    sample_combo = ttk.Combobox(
        detail,
        state="readonly",
        width=21,
        values=list(_SAMPLE_PRESETS) + ["Custom"],
    )
    sample_combo.grid(row=2, column=1, columnspan=2, sticky="w", pady=(6, 0), padx=(4, 0))
    ttk.Label(detail, text="Points:").grid(row=2, column=3, sticky="e", pady=(6, 0), padx=(8, 4))
    sample_points = ttk.Entry(detail, width=7)
    sample_points.grid(row=2, column=4, sticky="w", pady=(6, 0))

    rotating_label = tk.StringVar(value="Rotating atoms: —")
    ttk.Label(detail, textvariable=rotating_label).grid(row=3, column=0, columnspan=8, sticky="w", pady=(6, 0))

    footer_opts = ttk.Frame(rot_box)
    footer_opts.grid(row=5, column=0, sticky="ew", pady=(7, 0))
    footer_opts.columnconfigure(1, weight=1)
    ttk.Checkbutton(footer_opts, text="Keep reference structure", variable=rot_keep_reference).grid(row=0, column=0, sticky="w")
    ttk.Label(footer_opts, textvariable=rot_status).grid(row=0, column=1, sticky="e")

    phase_note = ttk.Label(
        rot_box,
        text="Averaged coordinates are used for structural plots; PCS uses ensemble-averaged G factors.",
    )
    phase_note.grid(row=6, column=0, sticky="w", pady=(5, 0))

    def refresh_local():
        local_tree.delete(*local_tree.get_children())
        groups = _detect_local_groups(state)
        for g in groups:
            local_tree.insert("", "end", values=(g["type"], f"C{g['center_ref']}", ", ".join(str(v) for v in g["member_refs"])))
        if groups:
            n_me = sum(g["type"] == "CH₃" for g in groups)
            n_cf = sum(g["type"] == "CF₃" for g in groups)
            local_status.set(f"Detected: CH₃ {n_me} · CF₃ {n_cf}")
        elif state.get("atom_data_raw") or state.get("atom_data"):
            local_status.set("No CH₃ / CF₃ groups detected.")
        else:
            local_status.set("No structure loaded.")

    def _group_index_from_iid(iid: str | None):
        if not iid:
            return None
        try:
            return int(iid.split("_")[-1])
        except Exception:
            return None

    def _update_range_state():
        st = "normal" if range_mode.get() == "restricted" else "disabled"
        range_start.configure(state=st)
        range_end.configure(state=st)

    def _update_sample_state():
        sample_points.configure(state="normal" if sample_combo.get() == "Custom" else "disabled")

    def _commit_detail():
        nonlocal updating_detail
        if updating_detail:
            return
        iid = rot_tree.focus() or (rot_tree.selection()[0] if rot_tree.selection() else "")
        idx = _group_index_from_iid(iid)
        if idx is None or idx < 0 or idx >= len(rot_groups):
            return
        g = rot_groups[idx]
        g["range_mode"] = range_mode.get()
        try:
            g["range_start_deg"] = float(range_start.get())
            g["range_end_deg"] = float(range_end.get())
        except Exception:
            pass
        label = sample_combo.get()
        if label in _SAMPLE_PRESETS:
            g["n_samples"] = int(_SAMPLE_PRESETS[label])
        else:
            try:
                g["n_samples"] = max(1, int(sample_points.get()))
            except Exception:
                pass
        _update_range_state()
        _update_sample_state()
        refresh_rot_status()

    def _load_detail(iid: str | None):
        nonlocal updating_detail, selected_iid
        idx = _group_index_from_iid(iid)
        selected_iid = iid
        updating_detail = True
        try:
            if idx is None or idx < 0 or idx >= len(rot_groups):
                selected_label.set("No group selected")
                rotating_label.set("Rotating atoms: —")
                return
            g = rot_groups[idx]
            selected_label.set(
                f"{g.get('group_id', f'rot{idx+1}')} · {g.get('label', 'Planar group')} · "
                f"Axis {_format_ref_sequence(g.get('axis_atoms', ())) }"
            )
            range_mode.set(str(g.get("range_mode", "full")))
            range_start.delete(0, "end")
            range_start.insert(0, f"{float(g.get('range_start_deg', 0.0)):g}")
            range_end.delete(0, "end")
            range_end.insert(0, f"{float(g.get('range_end_deg', 360.0)):g}")
            n = int(g.get("n_samples", 24))
            sample_combo.set(_sampling_label(n))
            sample_points.configure(state="normal")
            sample_points.delete(0, "end")
            sample_points.insert(0, str(n))
            refs = list(g.get("rotating_atoms") or ())
            txt = _format_ref_sequence(refs)
            if len(txt) > 90:
                txt = txt[:87] + "..."
            rotating_label.set(f"Rotating atoms: {txt}")
            _update_range_state()
            _update_sample_state()
        finally:
            updating_detail = False

    def refresh_rot_status():
        enabled_count = sum(bool(g.get("enabled", True)) for g in rot_groups)
        auto_count = sum(str(g.get("source", "auto")) == "auto" for g in rot_groups)
        manual_count = sum(str(g.get("source", "auto")) == "manual" for g in rot_groups)
        rot_status.set(f"Groups: {len(rot_groups)} · active {enabled_count} · auto {auto_count} · manual {manual_count}")

    def refresh_rot_tree(select_index: int | None = None):
        rot_tree.delete(*rot_tree.get_children())
        for idx, g in enumerate(rot_groups):
            ring_text = _format_ring(g)
            if not (g.get("ring_atoms") or g.get("ring_cycles")):
                ring_text = "Manual"
            rot_tree.insert(
                "",
                "end",
                iid=f"grp_{idx}",
                values=(
                    "✓" if bool(g.get("enabled", True)) else "",
                    g.get("group_id", f"rot{idx+1}"),
                    g.get("label", "Planar group"),
                    ring_text,
                    _format_ref_sequence(g.get("axis_atoms") or ()),
                ),
            )
        refresh_rot_status()
        if rot_groups:
            idx = 0 if select_index is None else min(max(0, select_index), len(rot_groups) - 1)
            iid = f"grp_{idx}"
            rot_tree.selection_set(iid)
            rot_tree.focus(iid)
            rot_tree.see(iid)
            _load_detail(iid)
        else:
            _load_detail(None)

    def refresh_auto_detection():
        nonlocal rot_groups
        atom_data = state.get("atom_data_raw") or state.get("atom_data") or []
        ref_ids = state.get("atom_ids_raw") or list(range(1, len(atom_data) + 1))
        if not atom_data:
            rot_groups = [g for g in rot_groups if str(g.get("source", "auto")) == "manual"]
            refresh_rot_tree()
            rot_status.set("No structure loaded.")
            return
        try:
            threshold = float(planarity.get())
            if threshold <= 0:
                raise ValueError
        except Exception:
            messagebox.showwarning("Averaging", "Planarity RMS threshold must be a positive number.", parent=top)
            return

        if detection_mode.get() == "auto":
            detected = detect_planar_rotors(
                atom_data,
                ref_ids,
                planarity_threshold=threshold,
            )
            rot_groups = merge_group_settings(detected, rot_groups)
        refresh_rot_tree()

    def add_manual_group():
        nonlocal rot_groups
        atom_data = state.get("atom_data_raw") or state.get("atom_data") or []
        ref_ids = state.get("atom_ids_raw") or list(range(1, len(atom_data) + 1))
        if not atom_data:
            messagebox.showwarning("Averaging", "No structure loaded.", parent=top)
            return
        try:
            a = int(man_a.get())
            b = int(man_b.get())
            s = man_side.get().strip()
            side = int(s) if s else None
            group = build_manual_torsion_group(atom_data, ref_ids, a, b, moving_side_ref=side).to_dict()
        except Exception as exc:
            messagebox.showwarning("Manual torsion", str(exc), parent=top)
            return
        rot_groups.append(group)
        for i, g in enumerate(rot_groups, start=1):
            g["group_id"] = f"rot{i}"
        refresh_rot_tree(select_index=len(rot_groups) - 1)

    def remove_selected_manual():
        nonlocal rot_groups
        iid = rot_tree.focus() or (rot_tree.selection()[0] if rot_tree.selection() else "")
        idx = _group_index_from_iid(iid)
        if idx is None or idx < 0 or idx >= len(rot_groups):
            return
        if str(rot_groups[idx].get("source", "auto")) != "manual":
            messagebox.showinfo("Rotational averaging", "Auto-detected groups can be disabled; only manual groups are removed.", parent=top)
            return
        del rot_groups[idx]
        for i, g in enumerate(rot_groups, start=1):
            g["group_id"] = f"rot{i}"
        refresh_rot_tree(select_index=max(0, idx - 1))

    add_manual_btn = ttk.Button(manual, text="Add manual", command=add_manual_group)
    add_manual_btn.pack(side="left", padx=(4, 4))
    ttk.Button(manual, text="Remove selected", command=remove_selected_manual).pack(side="left", padx=(4, 0))

    def update_manual_state(*_):
        st = "normal" if detection_mode.get() == "manual" else "disabled"
        for widget in (man_a, man_b, man_side, add_manual_btn):
            widget.configure(state=st)

    detection_mode.trace_add("write", update_manual_state)
    update_manual_state()

    def on_tree_select(_event=None):
        _load_detail(rot_tree.focus() or (rot_tree.selection()[0] if rot_tree.selection() else None))

    def on_tree_click(event):
        if rot_tree.identify("region", event.x, event.y) != "cell":
            return
        if rot_tree.identify_column(event.x) != "#1":
            return
        iid = rot_tree.identify_row(event.y)
        idx = _group_index_from_iid(iid)
        if idx is None or idx >= len(rot_groups):
            return
        rot_groups[idx]["enabled"] = not bool(rot_groups[idx].get("enabled", True))
        refresh_rot_tree(select_index=idx)

    rot_tree.bind("<<TreeviewSelect>>", on_tree_select)
    rot_tree.bind("<Button-1>", on_tree_click, add="+")
    range_mode.trace_add("write", lambda *_: _commit_detail())
    sample_combo.bind("<<ComboboxSelected>>", lambda _e: _commit_detail())
    range_start.bind("<FocusOut>", lambda _e: _commit_detail())
    range_end.bind("<FocusOut>", lambda _e: _commit_detail())
    sample_points.bind("<FocusOut>", lambda _e: _commit_detail())

    refresh_local()
    if detection_mode.get() == "auto":
        refresh_auto_detection()
    else:
        refresh_rot_tree()

    # ------------------------------------------------------------------
    # Window footer
    # ------------------------------------------------------------------
    footer = ttk.Frame(outer)
    footer.grid(row=1, column=0, sticky="ew", pady=(8, 0))
    footer.columnconfigure(1, weight=1)

    ttk.Button(footer, text="Refresh detection", command=lambda: (refresh_local(), refresh_auto_detection())).grid(row=0, column=0, sticky="w")

    def apply_changes():
        _commit_detail()
        try:
            threshold = float(planarity.get())
            if threshold <= 0:
                raise ValueError
        except Exception:
            messagebox.showwarning("Averaging", "Planarity RMS threshold must be a positive number.", parent=top)
            return

        overlaps = validate_independent_groups(rot_groups)
        if overlaps and rot_enabled.get():
            a, b, common = overlaps[0]
            messagebox.showwarning(
                "Rotational averaging",
                f"Enabled groups {a} and {b} have overlapping rotating fragments.\n"
                f"Common Ref IDs: {_format_ref_sequence(common)}\n\n"
                "Disable one group or redefine the manual torsion before applying.",
                parent=top,
            )
            return

        _set_bool(state, "symavg_enabled_var", local_enabled.get())
        _set_bool(state, "symavg_methyl_enabled_var", methyl_enabled.get())
        _set_bool(state, "symavg_cf3_enabled_var", cf3_enabled.get())
        _set_bool(state, "symavg_keep_original_var", keep_original.get())

        _set_bool(state, "torsion_avg_enabled_var", rot_enabled.get())
        _set_bool(state, "torsion_avg_keep_reference_var", rot_keep_reference.get())
        _set_text(state, "torsion_avg_detection_mode_var", detection_mode.get())
        _set_float(state, "torsion_avg_planarity_var", threshold)
        state["torsion_avg_groups"] = deepcopy(rot_groups)
        state["torsion_avg_phase"] = 4

        callback = state.get("apply_averaging_settings")
        if callable(callback):
            callback()
        else:
            rebuild = state.get("apply_symavg_to_state")
            if callable(rebuild):
                rebuild(state)
            update = state.get("update_graph")
            if callable(update):
                update()

        refresh_local()
        refresh_rot_status()

    def close_window():
        try:
            top.destroy()
        finally:
            state["averaging_settings_window"] = None

    ttk.Button(footer, text="Apply", command=apply_changes).grid(row=0, column=2, padx=(8, 4))
    ttk.Button(footer, text="Close", command=close_window).grid(row=0, column=3, padx=(4, 0))

    top.protocol("WM_DELETE_WINDOW", close_window)
    return top
