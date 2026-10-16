"""Integration adapters and launch helpers for NMR Workbench."""

# Preserve the original public launcher API used by PCS Analyzer:
#
#     from nmr_workbench.integration import open_nmr_workbench
#
# Do not remove this re-export when adding new integration modules.
from .launcher import open_nmr_workbench

# PCS transfer helpers introduced by the transfer-review milestone.
from .pcs_transfer import (
    PCSTarget,
    PCSTransferError,
    PCSTransferRow,
    PCSTransferSummary,
    apply_transfer_rows,
    build_transfer_rows,
    extract_pcs_targets,
    suggest_target_ref_id,
)

__all__ = [
    "open_nmr_workbench",
    "PCSTarget",
    "PCSTransferError",
    "PCSTransferRow",
    "PCSTransferSummary",
    "apply_transfer_rows",
    "build_transfer_rows",
    "extract_pcs_targets",
    "suggest_target_ref_id",
]
