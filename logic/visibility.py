"""Document-owned atom visibility for PCS Analyzer.

Visibility is a filter over the existing PCS Analyzer Ref IDs.  It never
renumbers or removes atoms from the document model.  Element and ligand filters
are combined here so table, plots, fitting, and viewers share one definition of
"visible".

Ligand identity is based on the sorted member Ref IDs of the coordinating
fragment rather than its display order.  This keeps persisted visibility stable
when ligand display numbering changes while preserving the application's
existing atom identity system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from logic.ligand_grouping_state import refresh_ligand_analysis_in_state


ELEMENT_SELECTION_KEY = "_visibility_selected_elements"
LIGAND_SELECTION_KEY = "_visibility_selected_ligands"


@dataclass(frozen=True)
class LigandVisibilityRecord:
    """UI/persistence projection of one detected coordinating fragment."""

    key: str
    fragment_id: int
    label: str
    formula: str
    atom_ref_ids: tuple[int, ...]

    @property
    def atom_count(self) -> int:
        return len(self.atom_ref_ids)


def ligand_visibility_key(atom_ref_ids: Iterable[int]) -> str:
    """Return a stable ligand key built only from authoritative atom Ref IDs."""
    refs = tuple(sorted({int(value) for value in atom_ref_ids}))
    return "refs:" + ",".join(str(value) for value in refs)


def _element_from_label(label: object) -> str:
    """Normalize a PCS atom/pseudo-atom label to an element symbol."""
    text = str(label or "").strip()
    if text.startswith("MeH@"):
        return "H"
    if text.startswith("CF3F@"):
        return "F"
    if len(text) >= 2 and text[1].islower():
        return text[:2]
    return text[:1]


def _cached_or_refresh_ligand_analysis(state: dict):
    analysis = state.get("ligand_analysis")
    if analysis is not None:
        return analysis
    atom_data = state.get("atom_data_raw") or state.get("atom_data") or []
    if not atom_data:
        return None
    try:
        return refresh_ligand_analysis_in_state(state)
    except Exception as exc:
        # Ligand visibility must never make the core PCS views unusable when
        # topology perception fails.  Element visibility remains available.
        state["ligand_visibility_error"] = exc
        return None


def ligand_visibility_records(state: dict, *, refresh: bool = False) -> tuple[LigandVisibilityRecord, ...]:
    """Return coordinating ligand fragments using existing Chemical Map logic."""
    if refresh:
        state.pop("ligand_analysis", None)
        state.pop("ligand_analysis_summary", None)
    analysis = _cached_or_refresh_ligand_analysis(state)
    if analysis is None:
        return ()

    ligands = sorted(
        analysis.ligands,
        key=lambda item: (min(item.atom_ids) if item.atom_ids else 10**12, item.fragment_id),
    )
    records = []
    for display_index, ligand in enumerate(ligands, start=1):
        refs = tuple(sorted(int(value) for value in ligand.atom_ids))
        records.append(
            LigandVisibilityRecord(
                key=ligand_visibility_key(refs),
                fragment_id=int(ligand.fragment_id),
                label=f"L{display_index}",
                formula=str(ligand.formula or ""),
                atom_ref_ids=refs,
            )
        )
    return tuple(records)


def selected_elements(state: Mapping, available: Iterable[str] | None = None) -> set[str]:
    """Return the selected element set; ``None`` means default/all visible."""
    explicit = state.get(ELEMENT_SELECTION_KEY)
    if explicit is not None:
        return {str(value) for value in explicit}

    check_vars = state.get("check_vars", {}) or {}
    if isinstance(check_vars, Mapping) and check_vars:
        selected = set()
        for element, variable in check_vars.items():
            try:
                if bool(variable.get()):
                    selected.add(str(element))
            except Exception:
                pass
        return selected

    return {str(value) for value in (available or ())}


def selected_ligands(
    state: Mapping,
    records: Sequence[LigandVisibilityRecord] | None = None,
) -> set[str]:
    """Return selected stable ligand keys; ``None`` means default/all visible."""
    explicit = state.get(LIGAND_SELECTION_KEY)
    if explicit is not None:
        return {str(value) for value in explicit}

    check_vars = state.get("ligand_check_vars", {}) or {}
    if isinstance(check_vars, Mapping) and check_vars:
        selected = set()
        for key, variable in check_vars.items():
            try:
                if bool(variable.get()):
                    selected.add(str(key))
            except Exception:
                pass
        return selected

    return {record.key for record in (records or ())}


def sync_visibility_state_from_vars(state: dict) -> None:
    """Commit Tk visibility variables into plain document-owned state values."""
    check_vars = state.get("check_vars", {}) or {}
    if isinstance(check_vars, Mapping) and check_vars:
        chosen = []
        for element, variable in check_vars.items():
            try:
                if bool(variable.get()):
                    chosen.append(str(element))
            except Exception:
                pass
        state[ELEMENT_SELECTION_KEY] = tuple(sorted(chosen))

    ligand_vars = state.get("ligand_check_vars", {}) or {}
    if isinstance(ligand_vars, Mapping) and ligand_vars:
        chosen = []
        for key, variable in ligand_vars.items():
            try:
                if bool(variable.get()):
                    chosen.append(str(key))
            except Exception:
                pass
        state[LIGAND_SELECTION_KEY] = tuple(sorted(chosen))


def _ligand_membership(records: Sequence[LigandVisibilityRecord]) -> dict[int, str]:
    membership: dict[int, str] = {}
    for record in records:
        for ref_id in record.atom_ref_ids:
            membership[int(ref_id)] = record.key
    return membership


def _ligand_keys_for_ref(
    state: Mapping,
    ref_id: int,
    membership: Mapping[int, str],
) -> set[str]:
    ref_id = int(ref_id)
    direct = membership.get(ref_id)
    if direct is not None:
        return {direct}

    # Effective pseudo atoms retain their own Ref ID.  Resolve ligand ownership
    # through the original member Ref IDs instead of changing pseudo identities.
    pseudo_members = state.get("symavg_members_by_pseudo_id", {}) or {}
    members = pseudo_members.get(ref_id, ()) if isinstance(pseudo_members, Mapping) else ()
    return {
        membership[int(member)]
        for member in members
        if int(member) in membership
    }


def visible_ref_ids(
    state: dict,
    *,
    atom_data: Sequence[tuple] | None = None,
    ref_ids: Sequence[int] | None = None,
    refresh_ligands: bool = False,
) -> tuple[int, ...]:
    """Return visible Ref IDs in input order without changing their identity.

    Element and ligand filters are AND-combined for coordinating-ligand atoms.
    Metals and atoms that are not assigned to a coordinating ligand are affected
    only by the element filter.
    """
    if atom_data is None:
        atom_data = state.get("atom_data_eff") or state.get("atom_data") or []
    atom_data = list(atom_data or [])
    if not atom_data:
        return ()

    if ref_ids is None:
        ref_ids = state.get("atom_ids_eff") or state.get("atom_ids")
    if not ref_ids or len(ref_ids) != len(atom_data):
        ref_ids = list(range(1, len(atom_data) + 1))
    else:
        ref_ids = [int(value) for value in ref_ids]

    available_elements = {_element_from_label(row[0]) for row in atom_data if row}
    allowed_elements = selected_elements(state, available_elements)

    records = ligand_visibility_records(state, refresh=refresh_ligands)
    membership = _ligand_membership(records)
    allowed_ligands = selected_ligands(state, records)

    output = []
    for row, ref_id in zip(atom_data, ref_ids):
        element = _element_from_label(row[0] if row else "")
        if element not in allowed_elements:
            continue

        ligand_keys = _ligand_keys_for_ref(state, ref_id, membership)
        if ligand_keys and not ligand_keys.issubset(allowed_ligands):
            continue
        output.append(int(ref_id))
    return tuple(output)


def visible_ref_id_set(state: dict, **kwargs) -> set[int]:
    return set(visible_ref_ids(state, **kwargs))


__all__ = [
    "ELEMENT_SELECTION_KEY",
    "LIGAND_SELECTION_KEY",
    "LigandVisibilityRecord",
    "ligand_visibility_key",
    "ligand_visibility_records",
    "selected_elements",
    "selected_ligands",
    "sync_visibility_state_from_vars",
    "visible_ref_ids",
    "visible_ref_id_set",
]
