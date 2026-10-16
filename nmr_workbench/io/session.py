"""Recoverable, independent NMR sessions: strict JSON and non-pickled NumPy data."""

from dataclasses import asdict, dataclass, fields
from datetime import datetime
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
from numbers import Real
import tempfile
import zipfile

import numpy as np

from nmr_workbench.core import (ExperimentalConditions, NMRDataset, NMRObservable,
    NMRPeak, NMRSeries, ProcessingHistory, ProcessingStep, Spectrum1D)
from nmr_workbench.core.batch_recipe import BatchRecipe


@dataclass(frozen=True)
class SessionDiagnostic:
    code: str
    dataset_id: str | None
    message: str


@dataclass(frozen=True)
class RecoveredSession:
    datasets: tuple[NMRDataset, ...]
    series: tuple[NMRSeries, ...]
    diagnostics: tuple[SessionDiagnostic, ...]
    batch_presets: tuple[BatchRecipe, ...] = ()


def _id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("scientific IDs must be nonempty strings")
    return value


def _unique(objects, label):
    ids = [_id(o.id) for o in objects]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate {label} IDs")


def _encode(value):
    """Typed metadata codec: preserve nonfinite acquisition values safely."""
    if isinstance(value, np.generic):
        return {"type": "numpy_scalar", "dtype": str(value.dtype), "value": _encode(value.item())}
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"type": "float", "value": repr(value)}
    if isinstance(value, complex):
        return {"type": "complex", "value": [_encode(value.real), _encode(value.imag)]}
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("metadata keys must be strings")
        return {"type": "mapping", "value": {k: _encode(v) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return {"type": "tuple" if isinstance(value, tuple) else "list", "value": [_encode(v) for v in value]}
    if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
        return {"type": "array", "dtype": str(value.dtype), "value": _encode(value.tolist())}
    raise ValueError(f"unsupported persisted metadata type: {type(value).__name__}")


def _decode(value):
    if not isinstance(value, dict):
        if value is None or type(value) in {str, bool, int, float}:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("nonfinite JSON number requires typed encoding")
            return value
        raise ValueError("invalid typed metadata")
    kind, raw = value.get("type"), value.get("value")
    if kind == "mapping" and isinstance(raw, dict):
        return {k: _decode(v) for k, v in raw.items()}
    if kind in {"list", "tuple"} and isinstance(raw, list):
        decoded = [_decode(v) for v in raw]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "float" and raw in {"nan", "inf", "-inf"}:
        return float(raw)
    if kind == "complex" and isinstance(raw, list) and len(raw) == 2:
        return complex(*(_decode(v) for v in raw))
    if kind == "datetime":
        return datetime.fromisoformat(raw)
    if kind == "path" and isinstance(raw, str):
        return Path(raw)
    if kind == "bytes":
        return base64.b64decode(raw, validate=True)
    if kind in {"array", "numpy_scalar"}:
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "biufc":
            raise ValueError("nonnumeric metadata dtype")
        decoded = _decode(raw)
        return np.asarray(decoded, dtype=dtype) if kind == "array" else dtype.type(decoded)
    raise ValueError("unknown or malformed typed metadata")


def fingerprint_source(path):
    """Hash vendor scientific files only; no recursive unrelated folder hashing."""
    if path is None:
        return None
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(str(root))
    if root.is_symlink():
        raise ValueError("source symlinks are not fingerprinted")
    if root.is_file():
        files = [root]
        base = root.parent
    else:
        base = root
        names = {"fid", "acqus", "acqu", "procpar", "1r", "1i", "procs", "proc"}
        files = sorted((p for p in root.iterdir() if p.name in names and p.is_file()), key=lambda p: p.name)
        pdata = root / "pdata"
        if pdata.is_dir() and not pdata.is_symlink():
            for directory in sorted(pdata.iterdir()):
                if directory.is_dir() and not directory.is_symlink():
                    files.extend(p for p in sorted(directory.iterdir()) if p.name in names and p.is_file())
    if not files:
        raise ValueError("no supported vendor source files to fingerprint")
    manifest = []
    for path in files:
        if path.is_symlink():
            raise ValueError("source symlinks are not fingerprinted")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest.append({"path": path.relative_to(base).as_posix(), "size": path.stat().st_size,
                         "sha256": digest.hexdigest()})
    return {"algorithm": "sha256-vendor-files-v1", "files": manifest}


def _source_record(dataset):
    record = dict(dataset.source_provenance)
    record.setdefault("original_path", str(dataset.source_path) if dataset.source_path else None)
    if not record.get("fingerprint"):
        try:
            record["fingerprint"] = fingerprint_source(dataset.source_path)
        except (OSError, ValueError) as exc:
            record["fingerprint"] = None
            record["fingerprint_error"] = str(exc)
    return record


def rebind_source(dataset, path):
    """Explicit locator change only: embedded data/IDs/revisions stay untouched."""
    record = _source_record(dataset)
    dataset.source_path = Path(path).expanduser().resolve()
    dataset.source_provenance = record
    return _check_source(dataset, relocated=True)


def _check_source(dataset, *, relocated=False):
    if (dataset.source_path is None and dataset.raw_spectrum is None and dataset.processed_spectrum is None
        and not dataset.processing_history.steps and dataset.source_provenance.get("origin") in {"manual_observations", "imported_observations"}):
        dataset.source_provenance["recovery_status"] = "observations_only"
        return SessionDiagnostic("observations_only", dataset.id, "Dataset-owned observations restored; no vendor spectrum is required")
    expected = dataset.source_provenance.get("fingerprint")
    try:
        actual = fingerprint_source(dataset.source_path)
        code = "source_unverified" if expected is None else "source_changed" if actual != expected else "source_moved" if relocated else "source_unchanged"
    except FileNotFoundError:
        code = "source_missing"
    except (OSError, ValueError):
        code = "source_unreadable"
    dataset.source_provenance["recovery_status"] = code
    return SessionDiagnostic(code, dataset.id, f"{dataset.source_path}: committed embedded data retained")


def _write_spectrum(archive, spectrum, prefix):
    if spectrum is None:
        return None
    result = {"axis_unit": spectrum.axis_unit, "domain": spectrum.domain, "state": spectrum.state}
    for name, array in (("axis", spectrum.axis), ("intensities", spectrum.intensities)):
        if array.dtype.kind not in ("biuf" if name == "axis" else "biufc") or array.ndim != 1:
            raise ValueError("spectra require a real 1D axis and numeric 1D intensities")
        stream = io.BytesIO()
        np.save(stream, array, allow_pickle=False)
        data = stream.getvalue()
        entry = f"{prefix}/{name}.npy"
        archive.writestr(entry, data)
        result[name] = {"entry": entry, "sha256": hashlib.sha256(data).hexdigest()}
    return result


def _read_spectrum(archive, record):
    if record is None:
        return None
    arrays = []
    for name in ("axis", "intensities"):
        reference = record[name]
        info = archive.getinfo(reference["entry"])
        if info.file_size > 512 * 1024 * 1024:
            raise ValueError("spectrum entry exceeds 512 MiB limit")
        data = archive.read(info)
        if hashlib.sha256(data).hexdigest() != reference["sha256"]:
            raise ValueError("spectrum checksum mismatch")
        array = np.load(io.BytesIO(data), allow_pickle=False)
        if array.dtype.kind not in ("biuf" if name == "axis" else "biufc") or array.ndim != 1:
            raise ValueError("spectrum entry requires a real 1D axis and numeric 1D intensities")
        arrays.append(array)
    return Spectrum1D(*arrays, record["axis_unit"], record["domain"], record["state"])


def save_session(path, datasets, series, *, processing_session=None, batch_presets=()):
    """Atomic archive save; only dataset state and the committed session are read."""
    datasets, series = tuple(datasets), tuple(series)
    batch_presets = tuple(batch_presets)
    _unique(batch_presets, "batch preset")
    _unique(datasets, "dataset")
    _unique(series, "series")
    owned = {d.id: d for d in datasets}
    for item in series:
        if any(owned.get(m.dataset_id) is not m.dataset for m in item.members):
            raise ValueError("series member dataset must be owned by this session")
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = handle.name
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            rows = []
            for index, dataset in enumerate(datasets):
                _unique(dataset.peaks, "peak")
                _unique(dataset.observables, "observable")
                committed = dataset.restored_committed_spectrum or dataset.processed_spectrum
                if processing_session is not None and processing_session.dataset is dataset:
                    committed = processing_session.committed_spectrum
                if committed is None and dataset.raw_spectrum is not None:
                    if dataset.processing_history.steps:
                        from nmr_workbench.processing import SpectrumProcessor
                        committed = SpectrumProcessor.from_dataset(dataset).replay(dataset.raw_spectrum, dataset.processing_history).spectrum
                    else:
                        committed = dataset.raw_spectrum
                rows.append({"id": dataset.id, "source_path": str(dataset.source_path) if dataset.source_path else None,
                    "source_provenance": _encode(_source_record(dataset)), "vendor": dataset.vendor,
                    "experimental_role": dataset.experimental_role, "spectral_revision": dataset.spectral_revision,
                    "conditions": _encode(asdict(dataset.conditions)), "acquisition_parameters": _encode(dataset.acquisition_parameters),
                    "vendor_metadata": _encode(dataset.vendor_metadata), "history": _encode(dataset.processing_history.to_list()),
                    "peaks": _encode([asdict(p) for p in dataset.peaks]), "observables": _encode([asdict(o) for o in dataset.observables]),
                    "raw": _write_spectrum(archive, dataset.raw_spectrum, f"datasets/{index}/raw"),
                    "committed": _write_spectrum(archive, committed, f"datasets/{index}/committed")})
            manifest = {"format": "nmr_workbench_session", "schema_version": 1,
                        "datasets": rows, "series": [s.to_dict() for s in series]}
            if batch_presets:
                manifest["batch_presets"] = [preset.to_dict() for preset in batch_presets]
            archive.writestr("manifest.json", json.dumps(manifest, allow_nan=False, ensure_ascii=False))
        # Validate the produced recovery graph before replacing any existing file.
        load_session(temporary)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def load_session(path, *, source_bindings=None):
    """Parse entirely off-state; callers publish the complete result atomically."""
    diagnostics, datasets = [], []
    bindings = source_bindings or {}
    try:
        with zipfile.ZipFile(path) as archive:
            if len(set(archive.namelist())) != len(archive.namelist()):
                raise ValueError("duplicate archive entries")
            if archive.getinfo("manifest.json").file_size > 64 * 1024 * 1024:
                raise ValueError("session manifest exceeds 64 MiB limit")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("format") != "nmr_workbench_session" or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
                raise ValueError("unsupported NMR session schema")
            if not isinstance(manifest.get("datasets"), list) or not isinstance(manifest.get("series"), list):
                raise ValueError("datasets and series must be arrays")
            for row in manifest["datasets"]:
                if (row.get("vendor") is not None and not isinstance(row["vendor"], str)) or (row.get("source_path") is not None and not isinstance(row["source_path"], str)):
                    raise ValueError("invalid dataset vendor/source locator")
                metadata = {name: _decode(row[name]) for name in (
                    "acquisition_parameters", "vendor_metadata", "source_provenance")}
                if any(not isinstance(value, dict) for value in metadata.values()):
                    raise ValueError("dataset metadata/provenance must be mappings")
                condition_values = _decode(row["conditions"])
                if not isinstance(condition_values, dict) or set(condition_values) != {f.name for f in fields(ExperimentalConditions)}:
                    raise ValueError("invalid experimental conditions record")
                conditions = ExperimentalConditions()
                for name, value in condition_values.items():
                    if name in {"nucleus", "solvent", "reference_compound"}:
                        if value is not None and not isinstance(value, str):
                            raise ValueError(f"invalid condition: {name}")
                    elif name == "acquisition_date":
                        if value is not None and not isinstance(value, datetime):
                            raise ValueError("invalid acquisition date")
                    elif name not in {"temperature_k", "calibrated_temperature_k"}:
                        if value is not None and (isinstance(value, bool) or not isinstance(value, Real)):
                            raise ValueError(f"invalid numeric condition: {name}")
                    setattr(conditions, name, value)  # preserve invalid acquisition metadata for diagnostics
                history_rows = _decode(row["history"])
                if not isinstance(history_rows, list):
                    raise ValueError("processing history must be an array")
                steps = []
                for step in history_rows:
                    if not isinstance(step, dict) or not isinstance(step.get("operation"), str) or not isinstance(step.get("parameters"), dict) or (step.get("method") is not None and not isinstance(step["method"], str)):
                        raise ValueError("malformed processing history step")
                    steps.append(ProcessingStep(**step))
                raw, committed = _read_spectrum(archive, row["raw"]), _read_spectrum(archive, row["committed"])
                dataset = NMRDataset(id=_id(row["id"]), conditions=conditions, raw_spectrum=raw,
                    processed_spectrum=committed if committed and committed.domain == "frequency" else None,
                    processing_history=ProcessingHistory(steps), spectral_revision=row["spectral_revision"],
                    vendor=row["vendor"], source_path=row["source_path"], experimental_role=row["experimental_role"],
                    acquisition_parameters=metadata["acquisition_parameters"], vendor_metadata=metadata["vendor_metadata"],
                    peaks=[NMRPeak(**p) for p in _decode(row["peaks"])],
                    observables=[NMRObservable(**o) for o in _decode(row["observables"])],
                    source_provenance=metadata["source_provenance"], restored_committed_spectrum=committed)
                _unique(dataset.peaks, "peak")
                _unique(dataset.observables, "observable")
                for peak in dataset.peaks:
                    for name in ("position_ppm", "height", "integral", "linewidth_hz"):
                        value = getattr(peak, name)
                        if value is not None and (isinstance(value, bool) or not isinstance(value, Real)):
                            raise ValueError(f"invalid peak measurement: {name}")
                    if peak.source is not None and not isinstance(peak.source, str):
                        raise ValueError("invalid peak source")
                for observable in dataset.observables:
                    if type(observable.included) is not bool or not isinstance(observable.provenance, dict):
                        raise ValueError("invalid observable inclusion/provenance")
                    if not isinstance(observable.nucleus, str) or (observable.assignment is not None and not isinstance(observable.assignment, str)):
                        raise ValueError("invalid observable nucleus/assignment")
                    for value in (observable.chemical_shift_ppm, observable.uncertainty_ppm):
                        if value is not None and (isinstance(value, bool) or not isinstance(value, Real)):
                            raise ValueError("invalid observable measurement")
                    if observable.source_peak_id is not None:
                        _id(observable.source_peak_id)
                    if observable.source_peak_id and not any(p.id == observable.source_peak_id for p in dataset.peaks):
                        diagnostics.append(SessionDiagnostic("observable_peak_missing", dataset.id, f"observable={observable.id}, peak={observable.source_peak_id}"))
                if committed is None and not (raw is None and not steps and
                    (dataset.observables or dataset.source_provenance.get("origin") in {"manual_observations", "imported_observations"})):
                    diagnostics.append(SessionDiagnostic("committed_data_missing", dataset.id, "No embedded committed spectrum; raw data/recipe may be available"))
                if any(s.operation not in {"digital_filter", "apodization", "zero_fill", "fourier_transform", "phase", "baseline", "reference"} for s in steps):
                    diagnostics.append(SessionDiagnostic("recipe_unsupported", dataset.id, "Saved committed data retained; recipe contains unsupported operations"))
                elif raw is not None and steps:
                    from nmr_workbench.processing import SpectrumProcessor, ProcessingError
                    try:
                        replayed = SpectrumProcessor.from_dataset(dataset).replay(raw, dataset.processing_history).spectrum
                        if committed is not None and (replayed.domain != committed.domain or
                            replayed.axis.shape != committed.axis.shape or
                            not np.allclose(replayed.axis, committed.axis, equal_nan=True) or
                            not np.allclose(replayed.intensities, committed.intensities, equal_nan=True)):
                            diagnostics.append(SessionDiagnostic("recipe_snapshot_mismatch", dataset.id,
                                "Recipe replay differs; embedded committed data remains authoritative"))
                    except (ProcessingError, ValueError) as exc:
                        diagnostics.append(SessionDiagnostic("recipe_unreplayable", dataset.id, str(exc)))
                if dataset.id in bindings:
                    diagnostics.append(rebind_source(dataset, bindings[dataset.id]))
                else:
                    diagnostics.append(_check_source(dataset))
                datasets.append(dataset)
        _unique(datasets, "dataset")
        if set(bindings) - {d.id for d in datasets}:
            raise ValueError("source binding references an unknown dataset")
        by_id = {d.id: d for d in datasets}
        series = []
        for row in manifest["series"]:
            for member in row["members"]:
                dataset_id = _id(member["dataset_id"])
                if dataset_id not in by_id:
                    placeholder = NMRDataset(id=dataset_id)
                    by_id[dataset_id] = placeholder
                    datasets.append(placeholder)
                    diagnostics.append(SessionDiagnostic("member_dataset_missing", dataset_id,
                        f"series={row.get('id')}, member={member.get('id')}: unresolved placeholder, no data inferred"))
            item = NMRSeries.from_dict(row, by_id)
            for issue in item.update_tracks():
                point = item.get_track(issue.track_id).get_point(issue.member_id)
                diagnostics.append(SessionDiagnostic(issue.code, point.dataset_id,
                    f"series={item.id}, track={issue.track_id}, member={issue.member_id}, peak={point.peak_id}"))
            from ..analysis.vt_observables import association_diagnostics
            for code, dataset_id, message in association_diagnostics(item):
                diagnostics.append(SessionDiagnostic(code, dataset_id, message))
            series.append(item)
        _unique(series, "series")
        _unique([t for s in series for t in s.tracks], "track")
        _unique([a for s in series for a in s.observable_associations], "observable association")
        _unique([p for s in series for p in s.pcs_preparations], "PCS preparation")
        _unique([r for s in series for r in s.pcs_handoff_receipts], "PCS handoff receipt")
        _unique([r for s in series for r in s.analysis_evidence], "analysis evidence")
        _unique([m for s in series for m in s.members], "member")
        if any(s.pcs_preparations for s in series):
            from ..analysis.vt_pcs import recovery_diagnostics
            for code, dataset_id, message in recovery_diagnostics(datasets, series):
                diagnostics.append(SessionDiagnostic(code, dataset_id, message))
        if any(s.pcs_handoff_receipts for s in series):
            from ..integration.vt_pcs_handoff import receipt_diagnostics
            for item in series:
                for code, message in receipt_diagnostics(item):
                    diagnostics.append(SessionDiagnostic(code, None, message))
        if any(s.analysis_evidence for s in series):
            from ..analysis.vt_models import analysis_evidence_diagnostics
            for item in series:
                for evidence in item.analysis_evidence:
                    for code, message in analysis_evidence_diagnostics(item, evidence):
                        diagnostics.append(SessionDiagnostic(code, None, message))
        if any(d.code in {"member_dataset_missing", "committed_data_missing"} for d in diagnostics):
            diagnostics.append(SessionDiagnostic("partial_recovery", None,
                "Some dataset records or committed spectra are unavailable; review individual diagnostics"))
        preset_rows = manifest.get("batch_presets", [])
        if not isinstance(preset_rows, list):
            raise ValueError("batch_presets must be an array")
        presets = tuple(BatchRecipe.from_dict(row) for row in preset_rows)
        _unique(presets, "batch preset")
        return RecoveredSession(tuple(datasets), tuple(series), tuple(diagnostics), presets)
    except (KeyError, TypeError, AttributeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"malformed NMR session: {exc}") from exc
