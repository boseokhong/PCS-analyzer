"""Detached ORCA-to-Analyzer tensor conversion and conservative frame matching."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from logic.logic_pcs_pde import point_pcs_from_tensor, rank2_chi
from logic.pcs_forward_api import AngleUnit, EulerConvention, PCSAtom, PCSPredictionRequest
from logic.pcs_forward_model import predict_pcs
from logic.pcs_orientation import matrix_to_euler
from logic.pcs_prediction_model import get_prediction_model, PredictionModel
from logic.pcs_tensor_state import ApplicationPCSTensor, get_canonical_tensor_state


SOURCE_KEY = "pcs_tensor_source"


@dataclass(frozen=True)
class TensorTransferPreview:
    tensor: ApplicationPCSTensor
    provenance: dict
    alignment_rmsd: float
    alignment_max_error: float
    point_max_error: float
    rotation: tuple


def target_signature(state):
    """Only raw structure values and the assigned centre define this mapping."""
    atoms = state.get("atom_data_raw") or []
    return dict(atoms=[list(atom) for atom in atoms], refs=list(state.get("atom_ids_raw") or []),
                metal_ref=state.get("metal_ref_id"),
                metal_xyz=[float(state.get(key, 0.0)) for key in ("x0", "y0", "z0")])


def prepare_tensor_transfer(payload, state, *, coordinate_tolerance=1e-4):
    """Match ordered raw atoms using a proper rigid transform; never deform data.

    The preview UI must explicitly confirm atom order. Rank-deficient linear
    structures cannot define a unique orientation and are rejected.
    """
    source_atoms = tuple(tuple(atom) for atom in payload["atoms"])
    target = target_signature(state)
    target_atoms = target["atoms"]
    if len(source_atoms) < 3 or len(source_atoms) != len(target_atoms):
        raise ValueError("Load the matching structure in PCS Analyzer (at least three atoms).")
    if [a[0] for a in source_atoms] != [a[0] for a in target_atoms]:
        raise ValueError("Atom elements/order differ. Automatic atom remapping is not supported.")
    refs = target["refs"]
    if len(refs) != len(target_atoms) or len(set(refs)) != len(refs):
        raise ValueError("PCS Analyzer raw atom references are missing or ambiguous.")
    if target["metal_ref"] not in refs:
        raise ValueError("Assign the corresponding metal centre in PCS Analyzer first.")
    centre = refs.index(target["metal_ref"])
    if centre != int(payload["metal_index"]):
        raise ValueError("The assigned metal centre differs from the Workbench centre. Assign the matching atom first.")
    source = np.asarray([a[1:] for a in source_atoms], dtype=float)
    dest = np.asarray([a[1:] for a in target_atoms], dtype=float)
    if source.shape != (len(source_atoms), 3) or dest.shape != source.shape or not (
            np.all(np.isfinite(source)) and np.all(np.isfinite(dest))):
        raise ValueError("Structure coordinates must be finite XYZ values in angstrom.")
    if not np.allclose(dest[centre], target["metal_xyz"], rtol=0, atol=1e-8):
        raise ValueError("The assigned metal centre does not match its raw atom coordinates.")
    a, b = source - source.mean(axis=0), dest - dest.mean(axis=0)
    if np.linalg.matrix_rank(a, tol=1e-8) < 2 or np.linalg.matrix_rank(b, tol=1e-8) < 2:
        raise ValueError("Collinear structures cannot determine the tensor coordinate frame.")
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    row_rotation = u @ correction @ vt
    errors = np.linalg.norm(a @ row_rotation - b, axis=1)
    maximum = float(np.max(errors))
    if maximum > coordinate_tolerance:
        raise ValueError(f"Structure/order mismatch: maximum alignment error {maximum:.6g} Å "
                         f"exceeds {coordinate_tolerance:g} Å. Use the same geometry and atom order.")
    chi = np.asarray(payload["chi_converted"], dtype=float)
    if chi.shape != (3, 3) or not np.all(np.isfinite(chi)):
        raise ValueError("ORCA susceptibility must be a finite 3 by 3 tensor.")
    if not np.allclose(chi, chi.T, rtol=1e-10, atol=1e-12):
        raise ValueError("ORCA susceptibility tensor is not symmetric.")
    temperature = float(payload["temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Select a positive finite ORCA temperature.")
    raw = np.asarray(payload["chi_raw"], dtype=float)
    if raw.shape != (3, 3) or not np.all(np.isfinite(raw)):
        raise ValueError("Raw ORCA susceptibility must be a finite 3 by 3 tensor.")
    chi_source = rank2_chi(chi)
    chi_target = row_rotation.T @ chi_source @ row_rotation
    values, axes = np.linalg.eigh(chi_target)
    order = np.argsort(np.abs(values), kind="stable")
    values, axes = values[order], axes[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    angles = matrix_to_euler(axes.T, EulerConvention.INTRINSIC_ZYZ, AngleUnit.DEGREE)
    tensor = ApplicationPCSTensor(
        delta_chi_ax=float(1.5 * values[2] * 100),
        delta_chi_rh=float((values[0] - values[1]) * 100),
        alpha=angles[0], beta=angles[1], gamma=angles[2], rhombic_enabled=True,
        frame_convention=EulerConvention.INTRINSIC_ZYZ)
    reconstructed = np.asarray(tensor.to_forward_tensor().structure_frame_matrix_angstrom3())
    if not np.allclose(reconstructed, chi_target, rtol=1e-10, atol=1e-12):
        raise ValueError("Tensor conversion failed its matrix reconstruction check.")
    # Verify two independent PCS implementations, excluding the singular centre.
    # Use the exact rigid image, so coordinate rounding is measured separately.
    mapped = a @ row_rotation + dest.mean(axis=0)
    forward = tensor.to_forward_tensor(metal_position=tuple(mapped[centre]))
    request = PCSPredictionRequest(tensor=forward,
                                  atoms=tuple(PCSAtom(i + 1, tuple(xyz)) for i, xyz in enumerate(mapped)))
    main = np.array([p.pcs_ppm for p in predict_pcs(request).atom_predictions])
    expected = point_pcs_from_tensor(source, source[centre], chi_source)
    mask = np.isfinite(expected)
    if not np.any(mask) or not np.allclose(main[mask], expected[mask], rtol=1e-9, atol=1e-8):
        raise ValueError("Tensor conversion failed the independent Point PCS agreement check.")
    point_error = float(np.max(np.abs(main[mask] - expected[mask])))
    actual_request = PCSPredictionRequest(
        tensor=tensor.to_forward_tensor(metal_position=tuple(dest[centre])),
        atoms=tuple(PCSAtom(i + 1, tuple(xyz)) for i, xyz in enumerate(dest)))
    actual = np.array([p.pcs_ppm for p in predict_pcs(actual_request).atom_predictions])
    noncentre = np.arange(len(source_atoms)) != centre
    if not np.all(np.isfinite(expected[noncentre])) or not np.allclose(
            actual[noncentre], expected[noncentre], rtol=1e-6, atol=1e-6):
        raise ValueError("Target coordinates do not preserve Point PCS within tolerance. Use the exact same geometry.")
    point_error = float(np.max(np.abs(actual[noncentre] - expected[noncentre])))
    provenance = dict(schema_version=1, kind="orca", file=str(payload["file"]),
                      temperature=temperature, section=payload.get("section"), line=payload.get("line"),
                      raw_tensor=raw.tolist(),
                      raw_tensor_unit="cm3 K/mol (CGS molar Curie)",
                      converted_tensor_unit="angstrom3/molecule",
                      converted_tensor=chi.tolist(), applied_matrix_angstrom3=reconstructed.tolist(),
                      delta_chi_ax=tensor.delta_chi_ax, delta_chi_rh=tensor.delta_chi_rh,
                      euler_zyz_degrees=list(angles), target=target,
                      prediction_model="full_tensor", alignment_rmsd=float(np.sqrt(np.mean(errors**2))),
                      alignment_max_error=maximum, point_max_error=point_error,
                      source_to_target_rotation=row_rotation.T.tolist())
    return TensorTransferPreview(tensor, provenance, provenance["alignment_rmsd"], maximum,
                                 point_error, tuple(tuple(row) for row in row_rotation.T))


def validate_source_record(value):
    """Validate optional project provenance without trusting its contents as code."""
    if not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("kind") != "orca":
        raise ValueError("Unsupported ORCA tensor provenance.")
    if not isinstance(value.get("file"), str) or not value["file"]:
        raise ValueError("ORCA tensor provenance has no source filename.")
    for key in ("temperature", "delta_chi_ax", "delta_chi_rh", "alignment_rmsd", "alignment_max_error", "point_max_error"):
        if (isinstance(value[key], bool) or not isinstance(value[key], (int, float))
                or not np.isfinite(float(value[key]))):
            raise ValueError("ORCA tensor provenance contains a nonfinite number.")
    if value["temperature"] <= 0 or value.get("prediction_model") != "full_tensor":
        raise ValueError("Invalid ORCA tensor provenance conditions.")
    for key in ("raw_tensor", "converted_tensor", "applied_matrix_angstrom3", "source_to_target_rotation"):
        array = np.asarray(value[key])
        if array.shape != (3, 3) or array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
            raise ValueError("Invalid matrix in ORCA tensor provenance.")
    angles = np.asarray(value["euler_zyz_degrees"])
    if angles.shape != (3,) or angles.dtype.kind not in "fiu" or not np.all(np.isfinite(angles)):
        raise ValueError("Invalid orientation in ORCA tensor provenance.")
    target = value["target"]
    if (not isinstance(target, dict) or len(target["atoms"]) < 3
            or len(target["refs"]) != len(target["atoms"])
            or len(set(target["refs"])) != len(target["refs"])
            or target["metal_ref"] not in target["refs"]):
        raise ValueError("Invalid structure in ORCA tensor provenance.")
    coordinates = np.asarray([atom[1:] for atom in target["atoms"]], dtype=float)
    centre = np.asarray(target["metal_xyz"], dtype=float)
    if coordinates.shape != (len(target["atoms"]), 3) or centre.shape != (3,) or not (
            np.all(np.isfinite(coordinates)) and np.all(np.isfinite(centre))):
        raise ValueError("Invalid coordinates in ORCA tensor provenance.")
    return deepcopy(value)


def source_is_current(state):
    source = state.get(SOURCE_KEY)
    if not isinstance(source, dict) or source.get("kind") != "orca":
        return False
    if state.get("fit_override"):
        return False
    try:
        matrix = get_canonical_tensor_state(state).to_forward_tensor().structure_frame_matrix_angstrom3()
        return (np.allclose(matrix, source["applied_matrix_angstrom3"], rtol=1e-10, atol=1e-12)
                and target_signature(state) == source["target"]
                and get_prediction_model(state) is PredictionModel.FULL_TENSOR)
    except (KeyError, TypeError, ValueError):
        return False


def tensor_state_label(state, canonical):
    label = "Axial + Rhombic" if canonical.rhombic_enabled else "Axial"
    source = state.get(SOURCE_KEY)
    if isinstance(source, dict) and source.get("kind") == "orca":
        suffix = f"ORCA, {source['temperature']:g} K" if source_is_current(state) else "Modified from ORCA"
        label += " · " + suffix
    return label


def source_details(state):
    source = state.get(SOURCE_KEY)
    if not isinstance(source, dict) or source.get("kind") != "orca":
        return ""
    return (f"Tensor source: {'ORCA' if source_is_current(state) else 'Modified from ORCA'}\n"
            f"File: {source['file']}\nSection: {source.get('section')} · line {source.get('line')}\n"
            f"Temperature: {source['temperature']:g} K\n\n"
            f"Imported Δχ_ax: {source['delta_chi_ax']:.10g} ×10^-32 m³\n"
            f"Imported Δχ_rh: {source['delta_chi_rh']:.10g} ×10^-32 m³\n"
            f"Imported ZYZ angles (degrees): {source['euler_zyz_degrees']}\n"
            f"Metal Ref: {source['target']['metal_ref']}\n"
            f"Alignment RMSD: {source['alignment_rmsd']:.6g} Å\n"
            f"Point PCS agreement: max difference {source['point_max_error']:.6g} ppm\n\n"
            "Import applies the anisotropic tensor only. Experimental χ_mol is unchanged.")


def apply_tensor_transfer(state, preview):
    """One reversible edit through existing model commands; no widget ownership."""
    from logic.document_edit_history import FIT_KEYS, begin_edit, finish_edit
    from logic.pcs_model_state import update_canonical_tensor, update_prediction_model
    from logic.pcs_events import PCSDispatchError
    if target_signature(state) != preview.provenance["target"]:
        raise ValueError("The target structure or assigned centre changed. Open a new preview.")
    if state.get("conformer_preview"):
        raise ValueError("Apply or discard the current conformer preview before importing a tensor.")
    token = begin_edit(state, FIT_KEYS + (SOURCE_KEY, "pcs_prediction_model"))
    errors = []
    try:
        state[SOURCE_KEY] = deepcopy(preview.provenance)
        state.pop("fit_override", None)
        try:
            update_prediction_model(state, PredictionModel.FULL_TENSOR, source="orca:tensor-import")
        except PCSDispatchError as exc:
            errors.extend(exc.failures)
        try:
            update_canonical_tensor(state, preview.tensor, source="orca:tensor-import",
                                    legacy_callback=state.get("update_graph"))
        except PCSDispatchError as exc:
            errors.extend(exc.failures)
        for key in ("refresh_orientation_editor", "refresh_prediction_model_ui", "rh_refresh_table",
                    "refresh_fit_result_validity_ui", "refresh_fit_result_apply_ui"):
            callback = state.get(key)
            if callable(callback):
                try:
                    callback()
                except Exception as exc:
                    errors.append(f"{key}: {exc}")
    finally:
        finish_edit(state, token, f"Import ORCA tensor · {preview.provenance['temperature']:g} K")
    return errors
