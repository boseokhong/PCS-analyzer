"""Hierarchical structure-aware NMR target sites for PCS preparation.

D5a introduces an explicit layer between structure atoms and experimental NMR
resonances.  One resonance does not necessarily correspond to one Cartesian
atom: rapidly averaging groups such as CH3 are represented as one target site
whose calculated PCS is the mean of its member atoms.

Automatic detection is intentionally conservative:
* individual atom sites are always available for ligand atoms;
* CH3-H3 sites are detected from explicit H connectivity;
* tert-butyl-H9 sites are detected as three methyl groups on one quaternary C;
* CH2/NH2/aromatic symmetry positions are NOT automatically averaged;
* matching positions on repeated ligands are structural candidates only, not a
  declaration of experimental NMR equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from logic.ligand_correspondence import (
    LigandCorrespondenceAnalysis,
    analyze_ligand_correspondence,
)
from logic.ligand_grouping import LigandAnalysis
from logic.structure_graph import StructureGraph


AVERAGING_NONE = "none"
AVERAGING_MEAN_MEMBERS = "mean_members"


@dataclass(frozen=True)
class NMRTargetSite:
    """One structural target corresponding to one possible NMR observation."""

    site_id: str
    fragment_id: int
    site_type: str
    nucleus: str | None
    member_ref_ids: tuple[int, ...]
    anchor_ref_id: int
    child_site_ids: tuple[str, ...] = ()
    averaging: str = AVERAGING_NONE
    label: str = ""
    structural_key: str = ""
    source: str = "detected"

    @property
    def member_count(self) -> int:
        return len(self.member_ref_ids)

    @property
    def is_grouped(self) -> bool:
        return self.member_count > 1

    def aggregate_member_values(self, values_by_ref: Mapping[int, float]) -> float:
        """Reduce per-atom calculated values to this NMR observation target.

        The method intentionally returns one scalar for one target site, so a
        CH3 or tBu resonance cannot accidentally acquire 3x/9x fitting weight.
        """
        try:
            values = [float(values_by_ref[ref_id]) for ref_id in self.member_ref_ids]
        except KeyError as exc:
            raise KeyError(
                f"missing calculated value for Ref {int(exc.args[0])}"
            ) from exc

        if self.averaging == AVERAGING_NONE:
            if len(values) != 1:
                raise ValueError(
                    "non-averaged NMR site must contain exactly one member"
                )
            return values[0]
        if self.averaging == AVERAGING_MEAN_MEMBERS:
            if not values:
                raise ValueError("cannot average an empty NMR site")
            return sum(values) / len(values)
        raise ValueError(f"unsupported NMR-site averaging mode: {self.averaging}")


@dataclass(frozen=True)
class FragmentSiteCandidates:
    fragment_id: int
    site_ids: tuple[str, ...]


@dataclass(frozen=True)
class NMRMatchingSiteGroup:
    """Matching structural NMR-site candidates across one ligand group."""

    ligand_group_id: int
    site_type: str
    structural_key: str
    candidates: tuple[FragmentSiteCandidates, ...]

    @property
    def complete(self) -> bool:
        return bool(self.candidates) and all(item.site_ids for item in self.candidates)

    @property
    def ambiguous(self) -> bool:
        return any(len(item.site_ids) != 1 for item in self.candidates)

    def candidates_for_fragment(self, fragment_id: int) -> tuple[str, ...]:
        fragment_id = int(fragment_id)
        for item in self.candidates:
            if item.fragment_id == fragment_id:
                return item.site_ids
        return ()


@dataclass(frozen=True)
class NMRMatchingSiteResult:
    source_site_id: str
    source_fragment_id: int
    ligand_group_id: int
    site_type: str
    candidates: tuple[FragmentSiteCandidates, ...]
    status: str

    @property
    def ambiguous(self) -> bool:
        return self.status == "ambiguous"

    @property
    def matching_site_ids(self) -> tuple[str, ...]:
        return tuple(
            site_id
            for item in self.candidates
            for site_id in item.site_ids
            if site_id != self.source_site_id
        )


@dataclass(frozen=True)
class NMRSiteAnalysis:
    sites: tuple[NMRTargetSite, ...]
    matching_groups: tuple[NMRMatchingSiteGroup, ...]
    site_to_matching_group_index: Mapping[str, int]
    atom_site_ids: Mapping[int, str]

    def site(self, site_id: str) -> NMRTargetSite:
        for item in self.sites:
            if item.site_id == site_id:
                return item
        raise KeyError(f"unknown NMR site {site_id!r}")

    def atom_site(self, atom_id: int) -> NMRTargetSite:
        try:
            site_id = self.atom_site_ids[int(atom_id)]
        except KeyError as exc:
            raise KeyError(f"atom {atom_id} has no NMR atom site") from exc
        return self.site(site_id)

    def sites_for_fragment(self, fragment_id: int) -> tuple[NMRTargetSite, ...]:
        fragment_id = int(fragment_id)
        return tuple(site for site in self.sites if site.fragment_id == fragment_id)

    def sites_for_anchor(self, atom_id: int) -> tuple[NMRTargetSite, ...]:
        atom_id = int(atom_id)
        return tuple(site for site in self.sites if site.anchor_ref_id == atom_id)

    def matching_sites(self, site_id: str) -> NMRMatchingSiteResult:
        source = self.site(site_id)
        try:
            index = self.site_to_matching_group_index[site_id]
        except KeyError as exc:
            raise KeyError(f"site {site_id!r} has no ligand correspondence") from exc

        group = self.matching_groups[index]
        if len(group.candidates) <= 1:
            status = "single_ligand"
        elif not group.complete:
            status = "incomplete"
        elif group.ambiguous:
            status = "ambiguous"
        else:
            status = "unique"

        return NMRMatchingSiteResult(
            source_site_id=source.site_id,
            source_fragment_id=source.fragment_id,
            ligand_group_id=group.ligand_group_id,
            site_type=group.site_type,
            candidates=group.candidates,
            status=status,
        )


def analyze_nmr_sites(
    graph: StructureGraph,
    ligand_analysis: LigandAnalysis,
    *,
    correspondence: LigandCorrespondenceAnalysis | None = None,
) -> NMRSiteAnalysis:
    """Detect hierarchical NMR target sites and cross-ligand matches."""

    if correspondence is None:
        correspondence = analyze_ligand_correspondence(graph, ligand_analysis)

    sites: list[NMRTargetSite] = []
    atom_site_ids: dict[int, str] = {}

    # Build atom sites first so grouped sites can refer to them as children.
    for ligand in ligand_analysis.ligands:
        for atom_id in ligand.atom_ids:
            atom = graph.atom(atom_id)
            site_id = _atom_site_id(ligand.fragment_id, atom_id)
            atom_site_ids[atom_id] = site_id
            sites.append(
                NMRTargetSite(
                    site_id=site_id,
                    fragment_id=ligand.fragment_id,
                    site_type="atom",
                    nucleus=_default_nucleus(atom.element),
                    member_ref_ids=(atom_id,),
                    anchor_ref_id=atom_id,
                    averaging=AVERAGING_NONE,
                    label=f"{atom.element} Ref {atom_id}",
                    structural_key=correspondence.rooted_signature(atom_id),
                )
            )

    methyl_by_fragment_anchor: dict[tuple[int, int], NMRTargetSite] = {}

    # Conservative explicit-H methyl detection.
    for ligand in ligand_analysis.ligands:
        fragment_set = set(ligand.atom_ids)
        for atom_id in ligand.atom_ids:
            if graph.atom(atom_id).element != "C":
                continue
            neighbours = _covalent_neighbors_in_fragment(graph, atom_id, fragment_set)
            hydrogen_ids = tuple(
                sorted(n for n in neighbours if graph.atom(n).element == "H")
            )
            heavy_ids = tuple(
                sorted(n for n in neighbours if graph.atom(n).element != "H")
            )
            if len(hydrogen_ids) != 3 or len(heavy_ids) != 1 or len(neighbours) != 4:
                continue

            site = NMRTargetSite(
                site_id=_methyl_site_id(ligand.fragment_id, atom_id),
                fragment_id=ligand.fragment_id,
                site_type="methyl_H",
                nucleus="1H",
                member_ref_ids=hydrogen_ids,
                anchor_ref_id=atom_id,
                child_site_ids=tuple(atom_site_ids[h] for h in hydrogen_ids),
                averaging=AVERAGING_MEAN_MEMBERS,
                label=f"CH3-H3 @ Ref {atom_id}",
                structural_key=correspondence.rooted_signature(atom_id),
            )
            methyl_by_fragment_anchor[(ligand.fragment_id, atom_id)] = site
            sites.append(site)

    # tert-Butyl = quaternary carbon with exactly three neighbouring methyl C
    # atoms and one remaining non-H substituent.  This avoids auto-merging CH2
    # or arbitrary graph-symmetry positions.
    for ligand in ligand_analysis.ligands:
        fragment_set = set(ligand.atom_ids)
        methyl_anchors = {
            anchor: site
            for (fragment_id, anchor), site in methyl_by_fragment_anchor.items()
            if fragment_id == ligand.fragment_id
        }

        for atom_id in ligand.atom_ids:
            if graph.atom(atom_id).element != "C":
                continue
            neighbours = _covalent_neighbors_in_fragment(graph, atom_id, fragment_set)
            if len(neighbours) != 4:
                continue
            if any(graph.atom(n).element == "H" for n in neighbours):
                continue

            methyl_neighbours = tuple(
                sorted(n for n in neighbours if n in methyl_anchors)
            )
            if len(methyl_neighbours) != 3:
                continue
            non_methyl_neighbours = tuple(
                n for n in neighbours if n not in methyl_anchors
            )
            if len(non_methyl_neighbours) != 1:
                continue

            children = tuple(methyl_anchors[n] for n in methyl_neighbours)
            member_ids = tuple(
                sorted(
                    atom_id_member
                    for child in children
                    for atom_id_member in child.member_ref_ids
                )
            )
            if len(member_ids) != 9:
                continue

            sites.append(
                NMRTargetSite(
                    site_id=_tert_butyl_site_id(ligand.fragment_id, atom_id),
                    fragment_id=ligand.fragment_id,
                    site_type="tert_butyl_H",
                    nucleus="1H",
                    member_ref_ids=member_ids,
                    anchor_ref_id=atom_id,
                    child_site_ids=tuple(child.site_id for child in children),
                    averaging=AVERAGING_MEAN_MEMBERS,
                    label=f"tBu-H9 @ Ref {atom_id}",
                    structural_key=correspondence.rooted_signature(atom_id),
                )
            )

    matching_groups, site_to_group = _build_matching_site_groups(
        sites,
        ligand_analysis,
        correspondence,
    )

    return NMRSiteAnalysis(
        sites=tuple(sites),
        matching_groups=matching_groups,
        site_to_matching_group_index=site_to_group,
        atom_site_ids=dict(atom_site_ids),
    )


def _build_matching_site_groups(
    sites: list[NMRTargetSite],
    ligand_analysis: LigandAnalysis,
    correspondence: LigandCorrespondenceAnalysis,
) -> tuple[tuple[NMRMatchingSiteGroup, ...], dict[str, int]]:
    sites_by_fragment: dict[int, list[NMRTargetSite]] = {}
    for site in sites:
        sites_by_fragment.setdefault(site.fragment_id, []).append(site)

    groups: list[NMRMatchingSiteGroup] = []
    site_to_group: dict[str, int] = {}

    for ligand_group in ligand_analysis.groups:
        keys = sorted(
            {
                (site.site_type, site.structural_key)
                for fragment_id in ligand_group.member_fragment_ids
                for site in sites_by_fragment.get(fragment_id, ())
            }
        )

        for site_type, structural_key in keys:
            candidates = tuple(
                FragmentSiteCandidates(
                    fragment_id=fragment_id,
                    site_ids=tuple(
                        sorted(
                            site.site_id
                            for site in sites_by_fragment.get(fragment_id, ())
                            if site.site_type == site_type
                            and site.structural_key == structural_key
                        )
                    ),
                )
                for fragment_id in ligand_group.member_fragment_ids
            )
            group_index = len(groups)
            groups.append(
                NMRMatchingSiteGroup(
                    ligand_group_id=ligand_group.group_id,
                    site_type=site_type,
                    structural_key=structural_key,
                    candidates=candidates,
                )
            )
            for item in candidates:
                for site_id in item.site_ids:
                    site_to_group[site_id] = group_index

    return tuple(groups), site_to_group


def _covalent_neighbors_in_fragment(
    graph: StructureGraph,
    atom_id: int,
    fragment_atom_ids: set[int],
) -> tuple[int, ...]:
    return tuple(
        neighbour
        for neighbour in graph.neighbors(atom_id, kinds={"covalent"})
        if neighbour in fragment_atom_ids
    )


def _default_nucleus(element: str) -> str | None:
    # Keep this intentionally minimal.  Other nuclei can be assigned by the
    # experimental dataset later; D5a only needs an explicit 1H identity for
    # proton averaging groups.
    return "1H" if element == "H" else None


def _atom_site_id(fragment_id: int, atom_id: int) -> str:
    return f"f{int(fragment_id)}:atom:{int(atom_id)}"


def _methyl_site_id(fragment_id: int, carbon_id: int) -> str:
    return f"f{int(fragment_id)}:methyl_H:{int(carbon_id)}"


def _tert_butyl_site_id(fragment_id: int, central_carbon_id: int) -> str:
    return f"f{int(fragment_id)}:tert_butyl_H:{int(central_carbon_id)}"


__all__ = [
    "AVERAGING_MEAN_MEMBERS",
    "AVERAGING_NONE",
    "FragmentSiteCandidates",
    "NMRMatchingSiteGroup",
    "NMRMatchingSiteResult",
    "NMRSiteAnalysis",
    "NMRTargetSite",
    "analyze_nmr_sites",
]
