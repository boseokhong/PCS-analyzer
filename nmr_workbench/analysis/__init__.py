"""Scientific analysis helpers for NMR Workbench."""

from .vt_analysis import (
    VTPointIdentity, VTDiagnostic, VTObservation, VTTrackData, VTFitDomain,
    VTCoordinate, VTCoordinates, VTLinearFitResult,
    extract_vt_track, prepare_vt_coordinates, fit_vt_linear,
)
from .vt_models import (VTModelConfig, VTModelParameter, VTModelResult, evaluate_vt_model,
    fit_vt_model, compare_vt_models, predict_vt_model, model_curve)

from .peak_picking import (
    PeakPickingError,
    PeakPickingParameters,
    pick_positive_peaks_1d,
)
from .observable_sync import (
    ObservableSyncError,
    ObservableSyncSummary,
    assignments_from_observables,
    observable_for_peak,
    sync_dataset_observables,
    sync_peak_observable,
)

from .shift_analysis import (
    ShiftAnalysisError,
    ShiftComparisonRow,
    ShiftComparisonSummary,
    build_shift_comparison,
    dataset_nucleus,
    normalize_nucleus,
    roles_are_pairable,
    validate_dataset_pair,
)

from .pcs_derivation import (
    DirectDeltaShiftProvider,
    ManualFCSProvider,
    ManualPCSProvider,
    PCSDerivationContext,
    PCSDerivationError,
    PCSDerivationProvider,
    builtin_pcs_providers,
)

from .shift_diagnostics import (
    AbsoluteSpectrumPosition,
    PredictedParamagneticPeak,
    ShiftDiagnosticQuantity,
    ShiftDiagnosticRow,
    build_absolute_spectrum_positions,
    build_predicted_paramagnetic_peaks,
    build_shift_diagnostic_quantities,
    build_shift_diagnostics,
    suggest_ref_id,
)

__all__ = [
    "VTModelConfig", "VTModelParameter", "VTModelResult", "evaluate_vt_model",
    "fit_vt_model", "compare_vt_models", "predict_vt_model", "model_curve",
    "VTPointIdentity", "VTDiagnostic", "VTObservation", "VTTrackData", "VTFitDomain",
    "VTCoordinate", "VTCoordinates", "VTLinearFitResult",
    "extract_vt_track", "prepare_vt_coordinates", "fit_vt_linear",
    "PeakPickingError",
    "PeakPickingParameters",
    "pick_positive_peaks_1d",
    "ObservableSyncError",
    "ObservableSyncSummary",
    "assignments_from_observables",
    "observable_for_peak",
    "sync_dataset_observables",
    "sync_peak_observable",
    "ShiftAnalysisError",
    "ShiftComparisonRow",
    "ShiftComparisonSummary",
    "build_shift_comparison",
    "dataset_nucleus",
    "normalize_nucleus",
    "roles_are_pairable",
    "validate_dataset_pair",
    "DirectDeltaShiftProvider",
    "ManualFCSProvider",
    "ManualPCSProvider",
    "PCSDerivationContext",
    "PCSDerivationError",
    "PCSDerivationProvider",
    "builtin_pcs_providers",
    "AbsoluteSpectrumPosition",
    "PredictedParamagneticPeak",
    "ShiftDiagnosticQuantity",
    "ShiftDiagnosticRow",
    "build_absolute_spectrum_positions",
    "build_predicted_paramagnetic_peaks",
    "build_shift_diagnostic_quantities",
    "build_shift_diagnostics",
    "suggest_ref_id",
]
