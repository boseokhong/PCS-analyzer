"""Structure-aware PCS target planning for NMR Workbench.

D5c deliberately separates *target planning* from downstream PCS Analyzer
interpretation.  The resolver can describe single atoms, grouped NMR sites
(CH3/tBu), matching positions across equivalent ligands, and user-defined Ref
groups.

Grouped/matching plans preserve one experimental observation as one target plan;
they are not flattened into repeated ``delta_exp_values``.  D5d hands those
plans to PCS Analyzer as structured experimental PCS metadata without carrying
out averaging, residual calculation, or fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from logic.ligand_correspondence import analyze_ligand_correspondence
from logic.ligand_grouping_state import refresh_ligand_analysis_in_state
from logic.nmr_sites import (
    AVERAGING_MEAN_MEMBERS,
    AVERAGING_NONE,
    NMRSiteAnalysis,
    NMRTargetSite,
    analyze_nmr_sites,
)


SCOPE_SINGLE = "Single atom"
SCOPE_NMR_SITE = "NMR site"
SCOPE_MATCHING = "Matching ligand sites"
SCOPE_CUSTOM = "Custom refs"
TARGET_SCOPES = (
    SCOPE_SINGLE,
    SCOPE_NMR_SITE,
    SCOPE_MATCHING,
    SCOPE_CUSTOM,
)


class PCSStructureTargetError(ValueError):
    """Raised when a structure-aware PCS target cannot be built safely."""


@dataclass(frozen=True)
class PCSStructureTargetChoice:
    """One selectable structure-aware PCS target plan.

    ``site_member_ref_ids`` keeps the hierarchy explicit.  For example:

    * one atom: ``((17,),)``
    * one methyl: ``((18, 19, 20),)``
    * matching methyls on three ligands:
      ``((18,19,20), (41,42,43), (64,65,66))``

    D5c does not interpret the latter as three experimental observations.
    """

    key: str
    scope: str
    display: str
    site_member_ref_ids: tuple[tuple[int, ...], ...]
    site_ids: tuple[str, ...] = ()
    primary_ref_id: int | None = None
    averaging: str = AVERAGING_NONE
    matching_status: str | None = None
    source: str = "structure"

    @property
    def member_ref_ids(self) -> tuple[int, ...]:
        return tuple(
            ref_id
            for group in self.site_member_ref_ids
            for ref_id in group
        )

    @property
    def site_count(self) -> int:
        return len(self.site_member_ref_ids)

    @property
    def member_count(self) -> int:
        return len(self.member_ref_ids)

    @property
    def is_legacy_single_atom(self) -> bool:
        return (
            self.scope == SCOPE_SINGLE
            and self.site_member_ref_ids
            and len(self.site_member_ref_ids) == 1
            and len(self.site_member_ref_ids[0]) == 1
        )

    @property
    def requires_structured_handoff(self) -> bool:
        return not self.is_legacy_single_atom

    @property
    def requires_structured_fitting(self) -> bool:
        """Backward-compatible alias retained for D5c callers."""
        return self.requires_structured_handoff


@dataclass(frozen=True)
class PCSStructureTargetCatalog:
    single_choices: tuple[PCSStructureTargetChoice, ...]
    nmr_site_choices: tuple[PCSStructureTargetChoice, ...]
    matching_choices: tuple[PCSStructureTargetChoice, ...]
    warnings: tuple[str, ...] = ()

    def choices_for_scope(self, scope: str) -> tuple[PCSStructureTargetChoice, ...]:
        if scope == SCOPE_SINGLE:
            return self.single_choices
        if scope == SCOPE_NMR_SITE:
            return self.nmr_site_choices
        if scope == SCOPE_MATCHING:
            return self.matching_choices
        if scope == SCOPE_CUSTOM:
            return ()
        return ()

    def choice_by_key(self, key: str | None) -> PCSStructureTargetChoice | None:
        if not key:
            return None
        for choice in (
            *self.single_choices,
            *self.nmr_site_choices,
            *self.matching_choices,
        ):
            if choice.key == key:
                return choice
        return None

    def choice_for_ref(
        self,
        scope: str,
        ref_id: int | None,
    ) -> PCSStructureTargetChoice | None:
        if ref_id is None:
            return None
        ref_id = int(ref_id)
        candidates = [
            choice
            for choice in self.choices_for_scope(scope)
            if ref_id in choice.member_ref_ids
        ]
        if len(candidates) == 1:
            return candidates[0]

        # Prefer the most specific grouped site over a broader parent (e.g.
        # methyl over tBu) when both contain the selected proton.
        if candidates:
            return min(
                candidates,
                key=lambda choice: (choice.member_count, choice.site_count, choice.key),
            )
        return None


def build_structure_target_catalog_from_state(
    state: dict | None,
    *,
    allowed_ref_ids: Iterable[int] | None = None,
    labels_by_ref: Mapping[int, str] | None = None,
) -> PCSStructureTargetCatalog:
    """Build selectable target plans from the current PCS Analyzer structure.

    Failure to build chemical topology is intentionally non-fatal for the
    caller: it is reported as a catalog warning while single-atom choices can
    still be supplied from the state.
    """

    if not isinstance(state, dict):
        return PCSStructureTargetCatalog((), (), (), ("PCS Analyzer state unavailable.",))

    allowed = None if allowed_ref_ids is None else {int(v) for v in allowed_ref_ids}
    labels = {int(k): str(v) for k, v in dict(labels_by_ref or {}).items()}

    single_choices = _single_choices_from_state(state, allowed, labels)

    try:
        ligand_analysis = refresh_ligand_analysis_in_state(state)
        graph = state["structure_graph"]
        correspondence = analyze_ligand_correspondence(graph, ligand_analysis)
        site_analysis = analyze_nmr_sites(
            graph,
            ligand_analysis,
            correspondence=correspondence,
        )
    except Exception as exc:
        return PCSStructureTargetCatalog(
            single_choices=single_choices,
            nmr_site_choices=(),
            matching_choices=(),
            warnings=(f"Structure-aware target analysis unavailable: {exc}",),
        )

    nmr_site_choices = _nmr_site_choices(site_analysis, allowed, labels)
    matching_choices, matching_warnings = _matching_choices(
        site_analysis,
        allowed,
        labels,
    )

    return PCSStructureTargetCatalog(
        single_choices=single_choices,
        nmr_site_choices=nmr_site_choices,
        matching_choices=matching_choices,
        warnings=matching_warnings,
    )


def parse_custom_ref_choice(
    text: str,
    *,
    allowed_ref_ids: Iterable[int] | None = None,
    labels_by_ref: Mapping[int, str] | None = None,
) -> PCSStructureTargetChoice:
    """Parse a comma/space/semicolon separated custom Ref group."""

    normalized = str(text or "").replace(";", ",").replace(" ", ",")
    tokens = [token.strip() for token in normalized.split(",") if token.strip()]
    if not tokens:
        raise PCSStructureTargetError("Enter at least one Ref ID.")

    refs: list[int] = []
    for token in tokens:
        if not token.isdigit():
            raise PCSStructureTargetError(
                f"Custom targets must be integer Ref IDs; got {token!r}."
            )
        ref_id = int(token)
        if ref_id not in refs:
            refs.append(ref_id)

    allowed = None if allowed_ref_ids is None else {int(v) for v in allowed_ref_ids}
    if allowed is not None:
        invalid = [ref_id for ref_id in refs if ref_id not in allowed]
        if invalid:
            raise PCSStructureTargetError(
                "Custom target contains unavailable Ref IDs: "
                + ", ".join(map(str, invalid))
            )

    labels = {int(k): str(v) for k, v in dict(labels_by_ref or {}).items()}
    if len(refs) == 1:
        display = _ref_display(refs[0], labels)
        averaging = AVERAGING_NONE
    else:
        display = f"{len(refs)} custom Refs: " + ", ".join(map(str, refs))
        averaging = AVERAGING_MEAN_MEMBERS

    return PCSStructureTargetChoice(
        key="custom:" + ",".join(map(str, refs)),
        scope=SCOPE_CUSTOM,
        display=display,
        site_member_ref_ids=(tuple(refs),),
        primary_ref_id=refs[0],
        averaging=averaging,
        source="user",
    )


def _single_choices_from_state(
    state: dict,
    allowed: set[int] | None,
    labels: Mapping[int, str],
) -> tuple[PCSStructureTargetChoice, ...]:
    atom_by_id = state.get("atom_by_id", {}) or {}
    overrides = state.get("ref_label_overrides", {}) or {}
    ids = sorted(
        {int(value) for value in atom_by_id}
        | {int(value) for value in overrides}
        | (allowed or set())
    )
    if allowed is not None:
        ids = [ref_id for ref_id in ids if ref_id in allowed]

    merged_labels = {
        ref_id: str(
            labels.get(
                ref_id,
                overrides.get(ref_id, atom_by_id.get(ref_id, "")),
            )
            or ""
        )
        for ref_id in ids
    }
    return tuple(
        PCSStructureTargetChoice(
            key=f"single:{ref_id}",
            scope=SCOPE_SINGLE,
            display=_ref_display(ref_id, merged_labels),
            site_member_ref_ids=((ref_id,),),
            primary_ref_id=ref_id,
            averaging=AVERAGING_NONE,
        )
        for ref_id in ids
    )


def _nmr_site_choices(
    site_analysis: NMRSiteAnalysis,
    allowed: set[int] | None,
    labels: Mapping[int, str],
) -> tuple[PCSStructureTargetChoice, ...]:
    choices: list[PCSStructureTargetChoice] = []
    for site in site_analysis.sites:
        if site.site_type == "atom" or not site.is_grouped:
            continue
        if site.nucleus not in {None, "1H"}:
            continue
        members = tuple(int(v) for v in site.member_ref_ids)
        if allowed is not None and not set(members).issubset(allowed):
            continue
        choices.append(
            PCSStructureTargetChoice(
                key=f"site:{site.site_id}",
                scope=SCOPE_NMR_SITE,
                display=_site_display(site, labels),
                site_member_ref_ids=(members,),
                site_ids=(site.site_id,),
                primary_ref_id=(members[0] if members else site.anchor_ref_id),
                averaging=site.averaging,
            )
        )
    return tuple(sorted(choices, key=lambda item: (item.member_count, item.display, item.key)))


def _matching_choices(
    site_analysis: NMRSiteAnalysis,
    allowed: set[int] | None,
    labels: Mapping[int, str],
) -> tuple[tuple[PCSStructureTargetChoice, ...], tuple[str, ...]]:
    choices: list[PCSStructureTargetChoice] = []
    ambiguous_count = 0
    incomplete_count = 0

    for group_index, group in enumerate(site_analysis.matching_groups):
        if len(group.candidates) <= 1:
            continue
        if not group.complete:
            incomplete_count += 1
            continue
        if group.ambiguous:
            ambiguous_count += 1
            continue

        site_ids = tuple(item.site_ids[0] for item in group.candidates)
        sites = tuple(site_analysis.site(site_id) for site_id in site_ids)
        # Matching PCS assignment is currently meaningful for proton targets;
        # atom sites are included only when the atom itself is H.
        if any(not _site_is_proton_target(site) for site in sites):
            continue

        member_groups = tuple(
            tuple(int(value) for value in site.member_ref_ids)
            for site in sites
        )
        flat = {ref_id for members in member_groups for ref_id in members}
        if allowed is not None and not flat.issubset(allowed):
            continue

        representative = sites[0]
        label = _site_display(representative, labels)
        display = f"{label} (+{len(sites) - 1} matching ligand site"
        if len(sites) - 1 != 1:
            display += "s"
        display += ")"
        choices.append(
            PCSStructureTargetChoice(
                key=f"matching:{group_index}",
                scope=SCOPE_MATCHING,
                display=display,
                site_member_ref_ids=member_groups,
                site_ids=site_ids,
                primary_ref_id=(member_groups[0][0] if member_groups[0] else None),
                averaging=AVERAGING_MEAN_MEMBERS,
                matching_status="unique",
            )
        )

    warnings: list[str] = []
    if ambiguous_count:
        warnings.append(
            f"{ambiguous_count} symmetry-ambiguous matching group(s) were not auto-selectable."
        )
    if incomplete_count:
        warnings.append(
            f"{incomplete_count} incomplete matching group(s) were not auto-selectable."
        )

    return (
        tuple(sorted(choices, key=lambda item: (item.display, item.key))),
        tuple(warnings),
    )


def _site_is_proton_target(site: NMRTargetSite) -> bool:
    return site.nucleus == "1H" or site.site_type in {"methyl_H", "tert_butyl_H"}


def _site_display(site: NMRTargetSite, labels: Mapping[int, str]) -> str:
    if site.site_type == "methyl_H":
        return f"CH3-H3 · anchor Ref {site.anchor_ref_id} · {site.member_count} H"
    if site.site_type == "tert_butyl_H":
        return f"tBu-H9 · anchor Ref {site.anchor_ref_id} · {site.member_count} H"
    if site.site_type == "atom" and site.member_ref_ids:
        return _ref_display(site.member_ref_ids[0], labels)
    return site.label or site.site_id


def _ref_display(ref_id: int, labels: Mapping[int, str]) -> str:
    label = str(labels.get(int(ref_id), "") or "").strip()
    return f"{int(ref_id)} · {label}" if label else str(int(ref_id))


__all__ = [
    "PCSStructureTargetCatalog",
    "PCSStructureTargetChoice",
    "PCSStructureTargetError",
    "SCOPE_CUSTOM",
    "SCOPE_MATCHING",
    "SCOPE_NMR_SITE",
    "SCOPE_SINGLE",
    "TARGET_SCOPES",
    "build_structure_target_catalog_from_state",
    "parse_custom_ref_choice",
]
