"""Reviewed observed-shift table exchange; no PCS derivation or peak inference."""
import csv
import io
import json
from copy import copy, deepcopy
from dataclasses import dataclass
from uuid import uuid4

from ..analysis.nmr_data import finite_number, new_data_dataset, read_only
from ..analysis.shift_analysis import normalize_nucleus
from ..core import NMRObservable


COLUMNS = ("schema_version", "quantity", "dataset_id", "experiment", "role", "nucleus", "temperature_k",
    "calibrated_temperature_k", "observable_id", "source_peak_id", "spectral_revision",
    "assignment", "shift_ppm", "uncertainty_ppm", "included", "provenance")
ALIASES = {
    "assignment": ("assignment", "ref", "ref_id", "id", "label", "atom"),
    "shift_ppm": ("shift_ppm", "chemical_shift_ppm", "ppm", "shift", "value", "delta_obs", "δ_obs", "obs", "delta_dia", "δ_dia", "dia"),
}


def table_reader(text):
    text = text.lstrip("\ufeff").strip()
    if not text:
        raise ValueError("No data found")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    headers = tuple(reader.fieldnames or ())
    if not headers or len({h.strip().casefold() for h in headers}) != len(headers):
        raise ValueError("A table needs unique column headers")
    rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError("Some rows have more cells than the header")
    return headers, rows


def default_mapping(headers):
    lookup = {h.strip().casefold(): h for h in headers}
    return {key: next((lookup[a] for a in aliases if a in lookup), "")
            for key, aliases in ALIASES.items()}


@dataclass(frozen=True)
class ImportObservation:
    dataset_id: str
    experiment: str
    role: str
    nucleus: str
    temperature_k: float | None
    calibrated_temperature_k: float | None
    observable_id: str
    peak_id: str | None
    spectral_revision: int | None
    assignment: str | None
    shift_ppm: float
    uncertainty_ppm: float | None
    included: bool
    provenance: dict


def parse_observation_table(text, *, name="Imported shifts", nucleus="1H", role="unspecified", mapping=None, override_nucleus=False):
    headers, raw_rows = table_reader(text)
    fields = {h.strip().casefold(): h for h in headers}
    if "peak_id" in fields and "quantity" not in fields:
        raise ValueError("This is an assignment-review CSV. Use Assignments / Analysis → Review import to preserve its original peak links.")
    mapping = default_mapping(headers) if mapping is None else mapping
    assignment_col, shift_col = mapping.get("assignment"), mapping.get("shift_ppm")
    if not assignment_col or not shift_col:
        raise ValueError("Choose the Assignment and observed Shift columns")
    if assignment_col not in headers or shift_col not in headers or assignment_col == shift_col:
        raise ValueError("Assignment and Shift must be different existing columns")
    if shift_col.strip().casefold() in {"δ_para", "delta_para", "para", "pcs", "δ_exp", "delta_exp", "delta_shift", "Δshift".casefold()}:
        raise ValueError("This column is a derived shift/PCS quantity, not an absolute observed shift")
    obs_col = next((fields[k] for k in ("δ_obs", "delta_obs", "obs") if k in fields), None)
    dia_col = next((fields[k] for k in ("δ_dia", "delta_dia", "dia") if k in fields), None)
    # Legacy wide tables explicitly contain two absolute observations. δ_para
    # is derived downstream, not imported as another absolute observation.
    wide = obs_col is not None and dia_col is not None and "quantity" not in fields
    result = []
    for index, row in enumerate(raw_rows, 2):
        def cell(key, default=""):
            value = row.get(fields.get(key, ""))
            return (str(value).strip() or default) if value is not None else default
        quantity = cell("quantity", "observed_shift")
        if cell("schema_version", "1") != "1":
            raise ValueError("Unsupported observed-shift table version")
        if quantity != "observed_shift":
            raise ValueError(f"Row {index}: {quantity} is not an observed chemical shift; use the existing PCS import for PCS values")
        values = ((obs_col, "paramagnetic"), (dia_col, "diamagnetic")) if wide else ((shift_col, cell("role", role)),)
        for column, row_role in values:
            row_role = {"para": "paramagnetic", "obs": "paramagnetic", "dia": "diamagnetic"}.get(row_role.casefold(), row_role.casefold())
            raw_shift = row.get(column)
            if raw_shift is None or str(raw_shift).strip() == "":
                if wide:
                    continue
                raise ValueError(f"Row {index}: missing observed shift")
            if row_role not in {"paramagnetic", "diamagnetic", "unspecified"}:
                raise ValueError(f"Row {index}: unknown experiment role {row_role}")
            original_nucleus = cell("nucleus", nucleus)
            row_nucleus = normalize_nucleus(nucleus if override_nucleus else original_nucleus)
            if not row_nucleus:
                raise ValueError(f"Row {index}: nucleus is required")
            use = cell("included", "true").casefold()
            if use not in {"true", "false", "yes", "no", "1", "0"}:
                raise ValueError(f"Row {index}: Use must be true/false")
            evidence = json.loads(cell("provenance", "{}"))
            if not isinstance(evidence, dict):
                raise ValueError(f"Row {index}: provenance must be an object")
            if override_nucleus and normalize_nucleus(original_nucleus) != row_nucleus:
                evidence["reviewed_nucleus_override"] = {"original": original_nucleus, "selected": row_nucleus}
            revision = cell("spectral_revision")
            revision = int(revision) if revision else None
            if revision is not None and revision < 0:
                raise ValueError(f"Row {index}: invalid spectral revision")
            experiment = cell("experiment", name) + (f" ({row_role})" if wide else "")
            result.append(ImportObservation(cell("dataset_id"), experiment, row_role, row_nucleus,
                finite_number(cell("temperature_k"), "Temperature (K)", optional=True, positive=True),
                finite_number(cell("calibrated_temperature_k"), "Calibrated temperature (K)", optional=True, positive=True),
                cell("observable_id"), cell("source_peak_id") or None, revision,
                str(row.get(assignment_col) or "").strip() or None,
                finite_number(raw_shift, f"Row {index} shift"),
                finite_number(cell("uncertainty_ppm"), "Uncertainty (ppm)", optional=True, positive=True),
                use in {"true", "yes", "1"}, evidence))
    if not result:
        raise ValueError("No observations found")
    return tuple(result)


def observation_state_signature(model):
    """A review snapshot, including in-place edits to peak/observation records."""
    return repr([(d.id, d.spectral_revision, vars(d.conditions), d.experimental_role,
        [(p.id, p.position_ppm) for p in d.peaks], [vars(o) for o in d.observables]) for d in model.datasets])


def apply_observation_import(model, records, *, source="CSV", update_existing=False, detach=False, reviewed_signature=None):
    """Stage the entire import off-state, then publish without replacing datasets.

    Exact IDs are the only update key. Spectrum references require matching
    revisions and measurements. Detachment must be explicitly requested and
    records the original identities instead of pretending they remain linked.
    """
    if reviewed_signature is not None and reviewed_signature != observation_state_signature(model):
        raise ValueError("NMR data changed after preview. Preview and review the import again.")
    loaded = {d.id: d for d in model.datasets}
    staged, seen = {}, set()
    occupied = {o.id for d in model.datasets for o in d.observables}
    for record in records:
        key = record.dataset_id or (record.experiment, record.role, record.nucleus,
                                   record.temperature_k, record.calibrated_temperature_k)
        if key not in staged:
            existing = loaded.get(record.dataset_id) if not detach else None
            if existing is not None:
                if not update_existing or read_only(existing):
                    raise ValueError("This experiment is already loaded. Enable reviewed ID updates, or import a detached copy.")
                dataset = copy(existing)
                dataset.observables = deepcopy(existing.observables)
            else:
                dataset = new_data_dataset(record.experiment, record.nucleus, record.role, record.temperature_k)
                dataset.conditions.calibrated_temperature_k = record.calibrated_temperature_k
                if record.dataset_id and not detach:
                    dataset.id = record.dataset_id
                dataset.source_provenance.update(origin="imported_observations", table_source=str(source))
            staged[key] = (existing, dataset)
        existing, dataset = staged[key]
        if (normalize_nucleus(dataset.conditions.nucleus), dataset.experimental_role,
            dataset.conditions.temperature_k, dataset.conditions.calibrated_temperature_k) != (
                record.nucleus, record.role, record.temperature_k, record.calibrated_temperature_k):
            raise ValueError("Conflicting experiment metadata; acquisition conditions will not be overwritten")
        if record.peak_id and not detach:
            if not record.assignment:
                raise ValueError("A spectrum-linked observation needs an assignment; clear it in Peaks instead")
            peak = next((p for p in dataset.peaks if p.id == record.peak_id), None)
            if peak is None or record.spectral_revision != dataset.spectral_revision or float(peak.position_ppm) != record.shift_ppm:
                raise ValueError("Spectrum link is missing/stale. Open its original session, or explicitly import detached values.")
        item_id = str(uuid4()) if detach or not record.observable_id else record.observable_id
        if item_id in seen:
            raise ValueError("Duplicate observation IDs in the import")
        seen.add(item_id)
        previous = next((o for o in dataset.observables if o.id == item_id), None)
        if item_id in occupied and previous is None:
            raise ValueError("Observation ID belongs to a different loaded experiment")
        if previous and previous.source_peak_id != record.peak_id:
            raise ValueError("An import cannot change an existing observation's peak identity")
        evidence = dict(previous.provenance) if previous else {}
        evidence.update(record.provenance)
        if detach:
            evidence["detached_source"] = {"dataset_id": record.dataset_id,
                "observable_id": record.observable_id, "peak_id": record.peak_id,
                "spectral_revision": record.spectral_revision}
        evidence.setdefault("origin", "Imported")
        evidence["table_source"] = str(source)
        item = NMRObservable(record.nucleus, record.shift_ppm, record.assignment,
            record.uncertainty_ppm, dataset.id, None if detach else record.peak_id,
            item_id, record.included, evidence)
        if previous:
            dataset.observables[dataset.observables.index(previous)] = item
        else:
            dataset.observables.append(item)
        if item.source_peak_id and sum(o.source_peak_id == item.source_peak_id for o in dataset.observables) > 1:
            raise ValueError("A source peak already has a different observation ID")
    added = []
    for existing, dataset in staged.values():
        if existing is None:
            model.datasets.append(dataset)
            added.append(dataset)
        else:
            original = {o.id: o for o in existing.observables}
            for item in dataset.observables:
                if item.id in original:
                    original[item.id].__dict__.update(item.__dict__)
                else:
                    existing.observables.append(item)
    return tuple(added)


def export_observation_table(datasets):
    """Export authoritative observations, including identities and exclusions."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=COLUMNS)
    writer.writeheader()
    for dataset in datasets:
        conditions = dataset.conditions
        for item in dataset.observables:
            writer.writerow(dict(schema_version=1, quantity="observed_shift", dataset_id=dataset.id,
                experiment=dataset.vendor_metadata.get("display_name") or str(dataset.source_path or dataset.id),
                role=dataset.experimental_role, nucleus=item.nucleus,
                temperature_k=conditions.temperature_k, calibrated_temperature_k=conditions.calibrated_temperature_k,
                observable_id=item.id, source_peak_id=item.source_peak_id,
                spectral_revision=dataset.spectral_revision, assignment=item.assignment,
                shift_ppm=item.chemical_shift_ppm, uncertainty_ppm=item.uncertainty_ppm,
                included=item.included, provenance=json.dumps(item.provenance, ensure_ascii=False, allow_nan=False)))
    return output.getvalue()
