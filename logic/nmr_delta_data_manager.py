# logic/nmr_delta_data_manager.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

@dataclass
class NMRLayer:
    """Container for one spectrum 'layer' to be rendered in the NMR window."""
    name: str
    shifts: np.ndarray
    intensities: np.ndarray
    labels: List[str]
    ref_ids: List[int]

def ensure_delta_dicts(state: dict) -> None:
    """
    Ensure required dictionaries exist in state.
    IMPORTANT: Do NOT alias dict objects between different delta kinds.
    """
    state.setdefault("delta_obs_values", {})
    state.setdefault("delta_dia_values", {})
    state.setdefault("delta_para_values", {})  # cache
    state.setdefault("nmr_layer_show", {"PCS": True, "OBS": False, "DIA": False, "PARA": False})
    state.setdefault("nmr_layer_mode", "stacked")  # 'stacked' | 'overlay'

    # Legacy migration: old projects stored δ_Exp in 'delta_values'
    if "delta_exp_values" not in state and isinstance(state.get("delta_values"), dict):
        state["delta_exp_values"] = dict(state["delta_values"])

def recompute_delta_para(state: dict) -> None:
    """
    δ_para = δ_obs - δ_dia
    Stored in state['delta_para_values'] as {ref_id: float}
    Only computed when BOTH obs and dia exist for the same ref_id.
    """
    obs = state.get("delta_obs_values", {}) or {}
    dia = state.get("delta_dia_values", {}) or {}

    para = {}
    # only where both exist
    common_ids = set(obs.keys()) & set(dia.keys())
    for rid in common_ids:
        try:
            o = obs.get(rid, None)
            d = dia.get(rid, None)
            if o is None or d is None:
                continue
            para[int(rid)] = float(o) - float(d)
        except Exception:
            continue

    state["delta_para_values"] = para

def compute_nonpcs_by_id(state: dict) -> Dict[int, float]:
    """
    Return δ_nonPCS = δ_para - δ_pcs (where available).
    Note: This is a diagnostic residual-like quantity, not a guaranteed physical separation.
    """
    ensure_delta_dicts(state)
    recompute_delta_para(state)

    pcs_by_id: Dict[int, float] = state.get("pcs_by_id", {}) or {}
    para_by_id: Dict[int, float] = state.get("delta_para_values", {}) or {}

    out: Dict[int, float] = {}
    for rid, para in para_by_id.items():
        if rid in pcs_by_id:
            out[rid] = float(para) - float(pcs_by_id[rid])
    return out

def build_analysis_rows(state: dict):
    """
    Build analysis rows for nuclei where PCS / obs / dia are available.

    Returns:
        List[dict] with keys:
        ref_id, atom,
        pcs, obs, dia, para,
        non_pcs, abs_non_pcs, ratio_non_pcs
    """
    pcs_by_id  = state.get("pcs_by_id", {}) or {}
    obs_by_id  = state.get("delta_obs_values", {}) or {}
    dia_by_id  = state.get("delta_dia_values", {}) or {}
    atom_by_id = state.get("atom_by_id", {}) or {}

    # ensure δ_para up to date
    recompute_delta_para(state)
    para_by_id = state.get("delta_para_values", {}) or {}

    rows = []

    # union of all ids
    all_ids = set(pcs_by_id) | set(obs_by_id) | set(dia_by_id)

    for rid in sorted(all_ids):
        pcs  = pcs_by_id.get(rid)
        obs  = obs_by_id.get(rid)
        dia  = dia_by_id.get(rid)
        para = para_by_id.get(rid)

        non_pcs = None
        ratio   = None

        if para is not None and pcs is not None:
            try:
                non_pcs = float(para) - float(pcs)
                if abs(float(para)) > 1e-12:
                    ratio = non_pcs / float(para)
            except Exception:
                non_pcs = None
                ratio = None

        rows.append({
            "ref_id": rid,
            "atom": atom_by_id.get(rid, ""),
            "pcs": pcs,
            "obs": obs,
            "dia": dia,
            "para": para,
            "non_pcs": non_pcs,
            "abs_non_pcs": abs(non_pcs) if non_pcs is not None else None,
            "ratio_non_pcs": ratio,
        })

    return rows

def _ordered_ids_for_display(state: dict) -> List[int]:
    """Use current selection order if present; otherwise sort known ids."""
    pcs_by_id = state.get("pcs_by_id", {}) or {}
    atom_by_id = state.get("atom_by_id", {}) or {}

    ids = state.get("current_selected_ids", []) or []
    if ids:
        known = set(pcs_by_id.keys()) | set(atom_by_id.keys())
        return [i for i in ids if i in known]
    known = set(pcs_by_id.keys()) | set(atom_by_id.keys())
    return sorted(known)


def _build_layer_from_dict(
    name: str,
    values_by_id: Dict[int, float],
    ordered_ids: Sequence[int],
    atom_by_id: Dict[int, str],
) -> Optional[NMRLayer]:
    ids = [rid for rid in ordered_ids if rid in values_by_id]
    if not ids:
        return None

    shifts = np.asarray([values_by_id[rid] for rid in ids], dtype=float)
    intensities = np.ones_like(shifts, dtype=float)
    labels = [f"{rid}, {atom_by_id.get(rid, '')}".strip(", ") for rid in ids]
    return NMRLayer(name=name, shifts=shifts, intensities=intensities, labels=labels, ref_ids=list(ids))


def build_nmr_layers(state: dict) -> Tuple[List[NMRLayer], str]:
    """
    Build layers for NMR rendering based on current state and show flags.
    Returns (layers, mode), where mode is 'stacked' or 'overlay'.
    """
    ensure_delta_dicts(state)
    recompute_delta_para(state)

    show = state.get("nmr_layer_show", {}) or {}
    mode = state.get("nmr_layer_mode", "stacked")

    pcs_by_id = state.get("pcs_by_id", {}) or {}
    obs_by_id = state.get("delta_obs_values", {}) or {}
    dia_by_id = state.get("delta_dia_values", {}) or {}
    para_by_id = state.get("delta_para_values", {}) or {}
    atom_by_id = state.get("atom_by_id", {}) or {}

    ordered_ids = _ordered_ids_for_display(state)

    layers: List[NMRLayer] = []
    if show.get("PCS", True):
        lay = _build_layer_from_dict("PCS", pcs_by_id, ordered_ids, atom_by_id)
        if lay: layers.append(lay)
    if show.get("PARA", False):
        lay = _build_layer_from_dict("δ_para", para_by_id, ordered_ids, atom_by_id)
        if lay: layers.append(lay)
    if show.get("OBS", False):
        lay = _build_layer_from_dict("δ_obs", obs_by_id, ordered_ids, atom_by_id)
        if lay: layers.append(lay)
    if show.get("DIA", False):
        lay = _build_layer_from_dict("δ_dia", dia_by_id, ordered_ids, atom_by_id)
        if lay: layers.append(lay)

    return layers, mode


def push_layers_to_nmr_if_open(state: dict) -> None:
    """
    Push current layer data to the NMR spectrum window if it exists and supports set_layers().
    Falls back to set_data() with PCS only.
    Also supports state['schedule_nmr_update'] debounce mechanism.
    """
    win = state.get("nmr_win", None)
    if win is None:
        return

    try:
        if hasattr(win, "winfo_exists") and not win.winfo_exists():
            state["nmr_win"] = None
            return
    except Exception:
        return

    # Optional debounce: scheduler should call state["_pending_nmr_push"]()
    sched = state.get("schedule_nmr_update", None)
    if callable(sched):
        state["_pending_nmr_push"] = lambda: _push_now(state)
        sched()
        return

    _push_now(state)

def build_pcs_layer_only(state: dict):
    """
    Build PCS-only layer for the MAIN canvas.
    This must IGNORE any drawer layer toggles (PCS checkbox etc.).
    """
    pcs_by_id = state.get("pcs_by_id", {}) or {}
    atom_by_id = state.get("atom_by_id", {}) or {}

    # If you have current_selected_ids, prefer that ordering; else fall back to sorted keys
    ids = state.get("current_selected_ids", None)
    if ids:
        ref_ids = [int(i) for i in ids if int(i) in pcs_by_id]
    else:
        ref_ids = sorted(int(k) for k in pcs_by_id.keys())

    if not ref_ids:
        return None

    shifts = [float(pcs_by_id[rid]) for rid in ref_ids]
    intensities = [1.0] * len(ref_ids)

    # labels: keep consistent with what you used before (atom or ref)
    labels = []
    for rid in ref_ids:
        at = atom_by_id.get(rid, "")
        labels.append(f"{rid}:{at}" if at else str(rid))

    # Use the same layer class you already use in build_nmr_layers()
    return NMRLayer(
        name="PCS",
        shifts=shifts,
        intensities=intensities,
        labels=labels,
        ref_ids=ref_ids,
    )

def _push_now(state: dict) -> None:
    win = state.get("nmr_win", None)
    if win is None:
        return

    # 1) MAIN: ALWAYS PCS only (ignore drawer toggles)
    pcs_layer = build_pcs_layer_only(state)
    if pcs_layer is not None:
        try:
            win.set_data(
                pcs_layer.shifts,
                pcs_layer.intensities,
                labels=pcs_layer.labels,
                ref_ids=pcs_layer.ref_ids,
            )
        except TypeError:
            win.set_data(pcs_layer.shifts, pcs_layer.intensities)

    # 2) DRAWER: layers based on toggles (can exclude PCS)
    layers, mode = build_nmr_layers(state)

    if hasattr(win, "set_layers"):
        # Even if empty, push so drawer can clear itself cleanly
        win.set_layers(layers or [], mode=mode)




# Old push pcs to nmr
# def _push_pcs_to_nmr_if_open(state):
#     """
#     Push current PCS values to the NMR spectrum window (if open).
#     Labels are formatted as "Ref Atom" for quick identification.
#     """
#     import numpy as np
#     win = state.get('nmr_win', None)
#     if win is None:
#         return
#     try:
#         if hasattr(win, "winfo_exists") and not win.winfo_exists():
#             state['nmr_win'] = None
#             return
#     except Exception:
#         return
#
#     pcs_by_id = state.get('pcs_by_id', {})
#     atom_by_id = state.get('atom_by_id', {})
#     if not pcs_by_id:
#         return
#
#     ids = state.get('current_selected_ids', [])
#     if ids:
#         ordered_ids = [i for i in ids if i in pcs_by_id]
#     else:
#         ordered_ids = sorted(pcs_by_id.keys())
#
#     if not ordered_ids:
#         return
#
#     shifts = np.asarray([pcs_by_id[i] for i in ordered_ids], dtype=float)
#     intensities = np.ones_like(shifts, dtype=float)
#     # labels like "12, H" or "12, H16"
#     labels = [f"{i}, {atom_by_id.get(i, '')}".strip(", ") for i in ordered_ids]
#     ref_ids = ordered_ids
#
#     # Debounce if available, else direct
#     sched = state.get('schedule_nmr_update', None)
#     if callable(sched):
#         sched()
#     else:
#         try:
#             win.set_data(shifts, intensities, labels=labels, ref_ids=ref_ids)
#         except TypeError:
#             win.set_data(shifts, intensities, labels=labels, ref_ids=ref_ids)