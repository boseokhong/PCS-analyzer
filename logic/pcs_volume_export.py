"""Volumetric export services. Coordinates are Angstrom unless a format requires Bohr."""

from dataclasses import dataclass, replace, field as dataclass_field
from pathlib import Path
import gzip
import json
import os
import struct
import tempfile
from xml.sax.saxutils import quoteattr

import numpy as np


VOLUME_FORMATS = (".cube", ".cub", ".vti", ".vtk", ".dx", ".mrc", ".ccp4", ".map",
                  ".nrrd", ".npz", ".npy")
SURFACE_FORMATS = (".vtp", ".ply", ".stl", ".obj")
BOHR_ANGSTROM = 0.529177210903
ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
            "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
            "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
            "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()


@dataclass(frozen=True)
class Volume:
    values: np.ndarray
    origin: np.ndarray
    spacing: tuple
    name: str
    units: str
    atoms: tuple
    metadata: dict
    extra_arrays: dict = dataclass_field(default_factory=dict)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode metadata value: {type(value).__name__}")


def prepare_volume(result, field="pcs", region="full", bounds=None, smoothing_sigma=0.0):
    if field not in ("pcs", "density_used", "density_signed"):
        raise ValueError("Select PCS, density used, or original signed density.")
    if not np.isfinite(smoothing_sigma) or smoothing_sigma < 0 or (smoothing_sigma and field != "pcs"):
        raise ValueError("A positive smoothing sigma is supported only for PCS.")
    origin = np.array(result["origin"], dtype=float)
    spacing = tuple(result["spacing"])
    units = "ppm" if field == "pcs" else (
        "angstrom^-3" if field == "density_used" and result.get("normalize_density") else "source_units"
    )
    if field == "density_signed":
        values = result.get("source_signed_density")
        if values is None:
            raise ValueError("Original signed density is not available. Reload the density and recompute.")
        offset = np.rint((np.asarray(result["source_origin"]) - origin) / spacing).astype(int)
        tail = np.array(result["pcs_field"].shape) - np.array(values.shape) - offset
        values = np.pad(values, tuple(zip(offset, tail))) if region != "input" else values
        if region == "input":
            origin = np.asarray(result["source_origin"], dtype=float)
    else:
        values = result["pcs_field" if field == "pcs" else "rho_used"]
        if smoothing_sigma:
            if field != "pcs" or not np.isfinite(smoothing_sigma) or smoothing_sigma < 0:
                raise ValueError("A positive smoothing sigma is supported only for PCS.")
            from scipy.ndimage import gaussian_filter
            values = gaussian_filter(values, sigma=smoothing_sigma)
        if region == "input":
            offset = np.rint((np.asarray(result["source_origin"]) - origin) / spacing).astype(int)
            slices = tuple(slice(int(a), int(a + n)) for a, n in zip(offset, result["source_shape"]))
            values = values[slices]
            origin = np.asarray(result["source_origin"], dtype=float)
    if region not in ("full", "input", "custom"):
        raise ValueError("Unknown export region.")
    if region == "custom":
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (6,) or not np.all(np.isfinite(bounds)):
            raise ValueError("Bounds must be xmin xmax ymin ymax zmin zmax in Angstrom.")
        starts = np.ceil((bounds[::2] - origin) / spacing - 1e-9).astype(int)
        ends = np.floor((bounds[1::2] - origin) / spacing + 1e-9).astype(int)
        if np.any(starts < 0) or np.any(ends >= values.shape) or np.any(ends < starts):
            raise ValueError("Custom bounds must select samples within the grid.")
        values = values[tuple(slice(int(a), int(b + 1)) for a, b in zip(starts, ends))]
        origin = origin + starts * spacing
    metadata = {key: result.get(key) for key in (
        "temperature", "input_files", "calculation_params", "chi_raw", "chi_converted", "chi_rank2",
        "fft_pad_factor", "normalize_density", "normalization_target", "normalization_stage",
        "normalization_info", "tensor_source", "padding_convergence",
    )}
    metadata.update(field=field, region=region, smoothing_sigma=float(smoothing_sigma),
                    coordinate_units="angstrom", value_units=units)
    extra = {}
    if field == "pcs":
        extra["pcs_field"] = values
    if field == "pcs" and region == "full" and not smoothing_sigma:
        extra.update({key: result[key] for key in (
            "source_term", "rho_used", "source_signed_density", "source_origin", "source_shape",
            "source_spacing", "temperature", "chi_raw", "chi_converted", "chi_rank2",
            "fft_pad_factor", "normalize_density", "normalization_target", "normalization_stage",
        ) if result.get(key) is not None})
    extra["atom_coordinates"] = np.array([atom[1:] for atom in result["atoms"]])
    extra["atom_elements"] = np.array([atom[0] for atom in result["atoms"]])
    return Volume(np.asarray(values), origin, spacing, field, units, tuple(result["atoms"]), metadata, extra)


def validate_volume(volume):
    if volume.values.ndim != 3 or min(volume.values.shape) < 1 or not np.all(np.isfinite(volume.values)):
        raise ValueError("Volume must contain a non-empty finite 3D array.")
    if np.shape(volume.origin) != (3,) or not np.all(np.isfinite(volume.origin)):
        raise ValueError("Volume origin must contain three finite values.")
    if len(volume.spacing) != 3 or not np.all(np.isfinite(volume.spacing)) or min(volume.spacing) <= 0:
        raise ValueError("Volume spacing must contain three positive finite values.")
    for symbol, *xyz in volume.atoms:
        if symbol not in ELEMENTS or len(xyz) != 3 or not np.all(np.isfinite(xyz)):
            raise ValueError(f"Cannot export atom {symbol!r}: invalid element or coordinates.")


def _chunks(values, order="C", size=32768):
    # nditer avoids a whole-volume Fortran-order copy for large fields.
    yield from np.nditer(values, flags=["external_loop", "buffered"],
                        op_flags=["readonly"], order=order, buffersize=size)


def _write_ascii(fh, values, order="C", factor=1.0, progress=None, cancel=None):
    completed = 0
    for chunk in _chunks(values, order):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Export cancelled.")
        tokens = [f"{float(value) * factor:.17g}" for value in chunk]
        fh.write("\n".join(" ".join(tokens[i:i + 6]) for i in range(0, len(tokens), 6)) + "\n")
        completed += chunk.size
        if progress:
            progress(completed / values.size)


def _write_volume(path, volume, dtype, compressed, progress, cancel):
    ext = path.suffix.lower()
    values, origin, spacing = volume.values, volume.origin, np.asarray(volume.spacing)
    shape = values.shape
    if ext in (".npz", ".npy"):
        with path.open("wb") as fh:
            converted = values.astype(dtype, copy=False)
            if not np.all(np.isfinite(converted)):
                raise ValueError("Values cannot be represented with the selected precision.")
            if ext == ".npy":
                np.save(fh, converted, allow_pickle=False)
            else:
                save = np.savez_compressed if compressed else np.savez
                extra_arrays = dict(volume.extra_arrays)
                if "pcs_field" in extra_arrays:
                    extra_arrays["pcs_field"] = converted
                save(fh, values=converted, origin=origin, spacing=spacing,
                     atoms=np.array(volume.atoms, dtype=str),
                     metadata_json=np.array(json.dumps(volume.metadata, default=_json_value)),
                     **extra_arrays)
        return
    if ext in (".mrc", ".ccp4", ".map"):
        header = bytearray(1024)
        struct.pack_into("<4i", header, 0, *shape, 2)
        struct.pack_into("<3i", header, 28, *shape)
        struct.pack_into("<3f", header, 40, *(np.array(shape) * spacing))
        struct.pack_into("<3f", header, 52, 90, 90, 90)
        struct.pack_into("<3i", header, 64, 1, 2, 3)
        struct.pack_into("<3f", header, 76, float(values.min()), float(values.max()), float(values.mean()))
        struct.pack_into("<i", header, 88, 1)
        struct.pack_into("<i", header, 108, 20140)
        struct.pack_into("<3f", header, 196, *origin)
        header[208:212] = b"MAP "
        header[212:216] = b"DD\x00\x00"
        struct.pack_into("<f", header, 216, float(values.std()))
        struct.pack_into("<i", header, 220, 1)
        label = f"PCS Workbench: {volume.name}; values {volume.units}; coordinates Angstrom".encode("ascii")[:80]
        header[224:224 + len(label)] = label
        with path.open("wb") as fh:
            fh.write(header)
            _write_binary(fh, values, "<f4", progress, cancel)
        return
    if ext == ".nrrd":
        typename = "float" if dtype == "float32" else "double"
        directions = " ".join("(" + ",".join(f"{v:.17g}" for v in row) + ")" for row in np.diag(spacing))
        text = (f"NRRD0005\ntype: {typename}\ndimension: 3\nsizes: {' '.join(map(str, shape))}\n"
                "kinds: domain domain domain\n"
                f"space dimension: 3\nspace directions: {directions}\nspace units: \"angstrom\" \"angstrom\" \"angstrom\"\n"
                f"space origin: ({','.join(map(str, origin))})\nendian: little\n"
                f"encoding: {'gzip' if compressed else 'raw'}\nsample units: \"{volume.units}\"\n\n")
        with path.open("wb") as fh:
            fh.write(text.encode("ascii"))
            if compressed:
                with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as stream:
                    _write_binary(stream, values, "<f4" if dtype == "float32" else "<f8", progress, cancel)
            else:
                _write_binary(fh, values, "<f4" if dtype == "float32" else "<f8", progress, cancel)
        return
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        if ext in (".cube", ".cub"):
            factor = BOHR_ANGSTROM ** 3 if volume.units == "angstrom^-3" else 1.0
            units = "bohr^-3" if factor != 1 else volume.units
            fh.write(f"PCS Workbench {volume.name}; values {units}\nCoordinates Bohr; OUTER X, MIDDLE Y, INNER Z\n")
            fh.write(f"{len(volume.atoms):5d} " + " ".join(f"{v / BOHR_ANGSTROM:.17g}" for v in origin) + "\n")
            for n, vector in zip(shape, np.diag(spacing / BOHR_ANGSTROM)):
                fh.write(f"{n:5d} " + " ".join(f"{v:.17g}" for v in vector) + "\n")
            for symbol, *xyz in volume.atoms:
                number = ELEMENTS.index(symbol) + 1
                fh.write(f"{number} {number}.0 " + " ".join(f"{v / BOHR_ANGSTROM:.17g}" for v in xyz) + "\n")
            _write_ascii(fh, values, factor=factor, progress=progress, cancel=cancel)
        elif ext == ".dx":
            counts = " ".join(map(str, shape))
            fh.write(f"# PCS Workbench {volume.name}; {volume.units}; coordinates Angstrom\nobject 1 class gridpositions counts {counts}\n")
            fh.write("origin " + " ".join(map(str, origin)) + "\n")
            for vector in np.diag(spacing):
                fh.write("delta " + " ".join(map(str, vector)) + "\n")
            fh.write(f"object 2 class gridconnections counts {counts}\nobject 3 class array type double rank 0 items {values.size} data follows\n")
            _write_ascii(fh, values, progress=progress, cancel=cancel)
            fh.write('attribute "dep" string "positions"\nobject "field" class field\ncomponent "positions" value 1\ncomponent "connections" value 2\ncomponent "data" value 3\n')
        elif ext == ".vtk":
            fh.write(f"# vtk DataFile Version 3.0\nPCS Workbench {volume.name} {volume.units}\nASCII\nDATASET STRUCTURED_POINTS\n")
            fh.write("DIMENSIONS " + " ".join(map(str, shape)) + "\nORIGIN " + " ".join(map(str, origin)) + "\nSPACING " + " ".join(map(str, spacing)) + "\n")
            fh.write(f"POINT_DATA {values.size}\nSCALARS {volume.name} double 1\nLOOKUP_TABLE default\n")
            _write_ascii(fh, values, "F", progress=progress, cancel=cancel)
        elif ext == ".vti":
            extent = " ".join(f"0 {n - 1}" for n in shape)
            fh.write('<?xml version="1.0"?>\n<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
            fh.write(f'<ImageData WholeExtent="{extent}" Origin="{" ".join(map(str, origin))}" Spacing="{" ".join(map(str, spacing))}">\n')
            fh.write(f'<Piece Extent="{extent}"><PointData Scalars={quoteattr(volume.name)}><DataArray type="Float64" Name={quoteattr(volume.name)} format="ascii">\n')
            _write_ascii(fh, values, "F", progress=progress, cancel=cancel)
            fh.write('</DataArray></PointData><CellData/></Piece></ImageData></VTKFile>\n')


def _write_binary(fh, values, dtype, progress, cancel):
    completed = 0
    for chunk in _chunks(values, "F"):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Export cancelled.")
        converted = np.asarray(chunk, dtype=dtype)
        if not np.all(np.isfinite(converted)):
            raise ValueError("Values cannot be represented with the selected precision.")
        fh.write(converted.tobytes())
        completed += chunk.size
        if progress:
            progress(completed / values.size)


def export_volume(path, volume, dtype="float64", compressed=True, progress=None, cancel=None):
    """Write a volume and metadata sidecar, preserving an existing output on failure."""
    path = Path(path)
    if path.suffix.lower() not in VOLUME_FORMATS:
        raise ValueError("Unsupported volume format.")
    if dtype not in ("float32", "float64"):
        raise ValueError("Precision must be float32 or float64.")
    validate_volume(volume)
    # Text formats store Float64. MRC/CCP4 supports Float32 scalar maps.
    effective_dtype = "float32" if path.suffix.lower() in (".mrc", ".ccp4", ".map") else (
        dtype if path.suffix.lower() in (".npy", ".npz", ".nrrd") else "float64"
    )
    limit = np.finfo(effective_dtype).max
    if volume.values.min() < -limit or volume.values.max() > limit:
        raise ValueError("Values cannot be represented with the selected precision.")
    metadata = dict(volume.metadata, field=volume.name, value_units=volume.units,
                    origin=volume.origin, spacing=volume.spacing, shape=volume.values.shape,
                    atoms=volume.atoms, precision=effective_dtype, coordinate_units="angstrom")
    if path.suffix.lower() in (".cube", ".cub"):
        metadata.update(coordinate_units="bohr", origin=volume.origin / BOHR_ANGSTROM,
                        spacing=np.array(volume.spacing) / BOHR_ANGSTROM,
                        value_units="bohr^-3" if volume.units == "angstrom^-3" else volume.units)
        metadata["atoms"] = [(atom[0], *(np.array(atom[1:]) / BOHR_ANGSTROM)) for atom in volume.atoms]
    volume = replace(volume, metadata=metadata)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(fd)
    temporary = Path(temporary)
    sidecar = Path(str(path) + ".json")
    metadata_temp = Path(str(temporary) + ".json")
    try:
        old_metadata = sidecar.read_bytes() if sidecar.exists() else None
        _write_volume(temporary, volume, effective_dtype, compressed, progress, cancel)
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Export cancelled.")
        metadata_temp.write_text(json.dumps(metadata, default=_json_value, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(metadata_temp, sidecar)
        try:
            os.replace(temporary, path)
        except OSError:
            if old_metadata is None:
                sidecar.unlink()
            else:
                sidecar.write_bytes(old_metadata)
            raise
    finally:
        temporary.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)


def export_surface(path, volume, level, progress=None, cancel=None):
    """Export the selected isosurface; mesh coordinates remain Angstrom."""
    path = Path(path)
    if path.suffix.lower() not in SURFACE_FORMATS:
        raise ValueError("Unsupported surface format.")
    validate_volume(volume)
    if not np.isfinite(level) or not volume.values.min() < level < volume.values.max():
        raise ValueError("Isovalue must lie strictly inside the selected field range.")
    if min(volume.values.shape) < 2:
        raise ValueError("Surface extraction requires at least two points along each axis.")
    import pyvista as pv
    grid = pv.ImageData(dimensions=volume.values.shape, origin=volume.origin, spacing=volume.spacing)
    grid.point_data[volume.name] = volume.values.ravel(order="F")
    surface = grid.contour([level], scalars=volume.name).triangulate()
    if surface.n_points == 0:
        raise ValueError("No surface intersects this isovalue.")
    if cancel is not None and cancel.is_set():
        raise InterruptedError("Export cancelled.")
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(fd)
    metadata_temp = Path(temporary + ".json")
    sidecar = Path(str(path) + ".json")
    try:
        old_metadata = sidecar.read_bytes() if sidecar.exists() else None
        surface.save(temporary)
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Export cancelled.")
        metadata = dict(volume.metadata, field=volume.name, value_units=volume.units,
                        coordinate_units="angstrom", isovalue=float(level),
                        origin=volume.origin, spacing=volume.spacing,
                        vertices=int(surface.n_points), triangles=int(surface.n_cells))
        metadata_temp.write_text(json.dumps(metadata, default=_json_value, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(metadata_temp, sidecar)
        try:
            os.replace(temporary, path)
        except OSError:
            if old_metadata is None:
                sidecar.unlink()
            else:
                sidecar.write_bytes(old_metadata)
            raise
    finally:
        Path(temporary).unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
    if progress:
        progress(1.0)
