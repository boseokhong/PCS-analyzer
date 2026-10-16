"""PCS Analyzer state adapter for D4b ligand grouping."""

from __future__ import annotations

from logic.ligand_grouping import analyze_ligands
from logic.structure_graph_state import refresh_structure_graph_in_state


def refresh_ligand_analysis_in_state(
    state: dict,
    *,
    include_coordination_role: bool = True,
):
    """Rebuild structure topology and cache transient ligand analysis.

    The dataclass result is intentionally transient in D4b; only the summary is
    JSON-friendly. Persistent user ligand labels/overrides belong to a later
    project-schema milestone.
    """
    graph = refresh_structure_graph_in_state(state)
    analysis = analyze_ligands(
        graph,
        include_coordination_role=include_coordination_role,
    )
    state["ligand_analysis"] = analysis
    state["ligand_analysis_summary"] = analysis.summary()
    return analysis


__all__ = ["refresh_ligand_analysis_in_state"]
