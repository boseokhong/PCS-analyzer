"""Assignment-to-structure helpers for NMR Workbench D5e.

This module keeps the NMR Workbench side of structure navigation deliberately
lightweight.  It resolves an assignment to one of the D5c structure-target
choices and returns the semantic Ref set that should be published through the
shared SelectionHub.

It does *not* derive PCS values and it does not perform PCS calculation or
fitting.  The only mutable step in D5e is an explicit user action that copies a
resolved target choice into PCS Preparation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .structure_aware_pcs_targets import (
    PCSStructureTargetCatalog,
    PCSStructureTargetChoice,
    SCOPE_MATCHING,
    SCOPE_NMR_SITE,
    SCOPE_SINGLE,
)


@dataclass(frozen=True)
class AssignmentStructureSelection:
    """A semantic structural selection associated with one NMR assignment."""

    assignment: str
    choice: PCSStructureTargetChoice
    source: str

    @property
    def ref_ids(self) -> tuple[int, ...]:
        """Unique Ref IDs in stable site/member order."""
        seen: set[int] = set()
        ordered: list[int] = []
        for ref_id in self.choice.member_ref_ids:
            value = int(ref_id)
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return tuple(ordered)

    @property
    def primary_ref_id(self) -> int | None:
        if self.choice.primary_ref_id is not None:
            return int(self.choice.primary_ref_id)
        return self.ref_ids[0] if self.ref_ids else None



def resolve_assignment_structure_selection(
    assignment: str,
    catalog: PCSStructureTargetCatalog | None,
    *,
    requested_scope: str | None = None,
    preferred_choice_key: str | None = None,
    fallback_ref_id: int | None = None,
) -> AssignmentStructureSelection | None:
    """Resolve an assignment to a safe D5c target choice.

    Resolution order:

    1. An explicit ``preferred_choice_key`` when it exists and, if supplied,
       matches ``requested_scope``.
    2. The most specific choice in ``requested_scope`` containing
       ``fallback_ref_id``.
    3. With no requested scope, a single-atom choice for ``fallback_ref_id``.

    No ambiguous matching group is manufactured here; D5c already excludes
    those groups from the catalog.
    """

    text = str(assignment or "").strip()
    if not text or catalog is None:
        return None

    if preferred_choice_key:
        preferred = catalog.choice_by_key(preferred_choice_key)
        if preferred is not None and (
            requested_scope is None or preferred.scope == requested_scope
        ):
            return AssignmentStructureSelection(
                assignment=text,
                choice=preferred,
                source="preferred",
            )

    if fallback_ref_id is None:
        return None

    ref_id = int(fallback_ref_id)
    scope = requested_scope or SCOPE_SINGLE
    choice = catalog.choice_for_ref(scope, ref_id)
    if choice is None:
        return None

    return AssignmentStructureSelection(
        assignment=text,
        choice=choice,
        source="ref",
    )



def nmr_site_selection(
    assignment: str,
    catalog: PCSStructureTargetCatalog | None,
    *,
    fallback_ref_id: int | None,
) -> AssignmentStructureSelection | None:
    return resolve_assignment_structure_selection(
        assignment,
        catalog,
        requested_scope=SCOPE_NMR_SITE,
        fallback_ref_id=fallback_ref_id,
    )



def matching_site_selection(
    assignment: str,
    catalog: PCSStructureTargetCatalog | None,
    *,
    fallback_ref_id: int | None,
) -> AssignmentStructureSelection | None:
    return resolve_assignment_structure_selection(
        assignment,
        catalog,
        requested_scope=SCOPE_MATCHING,
        fallback_ref_id=fallback_ref_id,
    )


__all__ = [
    "AssignmentStructureSelection",
    "matching_site_selection",
    "nmr_site_selection",
    "resolve_assignment_structure_selection",
]
