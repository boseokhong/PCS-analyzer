"""Record explicit prediction-policy selection through the existing model command."""
from logic.document_edit_history import begin_edit, finish_edit
from logic.pcs_model_state import update_prediction_model
from logic.pcs_prediction_model import PredictionModel, parse_prediction_model


def select_prediction_model(state, requested):
    finish_gesture = state.get("finish_parameter_gesture")
    if callable(finish_gesture):
        finish_gesture()
    check = state.get("check_edit_history_pending_inputs")
    if callable(check):
        check()
    model = parse_prediction_model(requested)
    token = begin_edit(state, ("pcs_prediction_model",))
    try:
        return update_prediction_model(state, model, source="control-panel:prediction-model")
    finally:
        # The model command commits before dispatch. A failed view must leave
        # that accepted change reversible rather than losing its history.
        label = "Axial" if model is PredictionModel.AXIAL else "Full tensor"
        finish_edit(state, token, f"Prediction model: {label}")
