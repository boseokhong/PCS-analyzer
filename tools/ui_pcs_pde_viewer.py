# tools/ui_pcs_pde_viewer.py
"""
3D PyVista viewer for PCS-PDE results.

Separated into:
- compute_pcs_pde_result(): expensive FFT/PDE computation
- open_or_refresh_pcs_pde_view(): render or refresh only
- export_pcs_pde_png(): off-screen PNG export
"""

from __future__ import annotations

import numpy as np

try:
    import pyvista as pv
except Exception as exc:
    pv = None
    _PV_ERROR = exc
else:
    _PV_ERROR = None

from scipy.ndimage import map_coordinates, gaussian_filter

from tools.logic_structure_helpers import get_cpk_color, radius_for_element, calculate_bonds
from tools.logic_pcs_pde import (
    compute_pcs_field_from_density,
    rank2_chi,
    point_pcs_from_tensor,
)


CAMERA_PRESETS = {
    "iso": "iso",
    "xy": "xy",
    "xz": "xz",
    "yz": "yz",
}


def _make_uniform_grid_from_ext(
    ext: np.ndarray,
    values: np.ndarray,
) -> "pv.ImageData":
    ext = np.asarray(ext, dtype=float)
    vals = np.asarray(values, dtype=float)

    nx, ny, nz = vals.shape
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("Grid must have at least 2 points along each axis.")

    x_min, x_max, y_min, y_max, z_min, z_max = map(float, ext)

    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    dz = (z_max - z_min) / (nz - 1)

    grid = pv.ImageData()
    grid.origin = (x_min, y_min, z_min)
    grid.spacing = (dx, dy, dz)
    grid.dimensions = (nx, ny, nz)
    grid.point_data["values"] = vals.ravel(order="F")
    return grid


def _interpolate_scalar_field_cubic_from_ext(
    points: np.ndarray,
    field: np.ndarray,
    ext: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    fld = np.asarray(field, dtype=float)
    ext = np.asarray(ext, dtype=float)

    nx, ny, nz = fld.shape
    x_min, x_max, y_min, y_max, z_min, z_max = map(float, ext)

    dx_i = (x_max - x_min) / (nx - 1) if nx > 1 else 1.0
    dy_i = (y_max - y_min) / (ny - 1) if ny > 1 else 1.0
    dz_i = (z_max - z_min) / (nz - 1) if nz > 1 else 1.0

    interp_order = 3 if (nx >= 4 and ny >= 4 and nz >= 4) else 1

    u = (pts[:, 0] - x_min) / dx_i
    v = (pts[:, 1] - y_min) / dy_i
    w = (pts[:, 2] - z_min) / dz_i

    inside = (
        (u >= 0) & (u <= nx - 1) &
        (v >= 0) & (v <= ny - 1) &
        (w >= 0) & (w <= nz - 1)
    )

    out = np.full(len(pts), np.nan, dtype=float)

    if np.any(inside):
        out[inside] = map_coordinates(
            fld,
            [u[inside], v[inside], w[inside]],
            order=interp_order,
            mode="nearest",
            prefilter=True,
        )

    return out


def _guess_metal_index(elements: list[str]) -> int:
    preferred = {
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
        "Sc", "Y", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    }
    for i, el in enumerate(elements):
        if el in preferred:
            return i
    return 0


def _print_atom_pcs_tsv(rows: list[dict]) -> None:
    print("\n[PCS atom comparison]")
    print("Ref\tAtom\tX\tY\tZ\tPCS_PDE_ppm\tPCS_Point_ppm\tDelta_PDE_minus_Point_ppm")
    for row in rows:
        pde_s = f"{row['pcs_pde']:.6f}" if np.isfinite(row["pcs_pde"]) else "nan"
        pt_s = f"{row['pcs_point']:.6f}" if np.isfinite(row["pcs_point"]) else "nan"
        dl_s = f"{row['delta']:.6f}" if np.isfinite(row["delta"]) else "nan"
        print(
            f"{row['ref']}\t{row['atom']}\t"
            f"{row['x']:.6f}\t{row['y']:.6f}\t{row['z']:.6f}\t"
            f"{pde_s}\t{pt_s}\t{dl_s}"
        )

    pde = np.array([r["pcs_pde"] for r in rows], dtype=float)
    point = np.array([r["pcs_point"] for r in rows], dtype=float)
    valid = np.isfinite(pde) & np.isfinite(point)

    if np.any(valid):
        diff = pde[valid] - point[valid]
        rmsd = float(np.sqrt(np.mean(diff ** 2)))
        max_abs = float(np.max(np.abs(diff)))
        print(f"\n[PCS summary] n={int(valid.sum())}  RMSD={rmsd:.6f} ppm  max|Δ|={max_abs:.6f} ppm")


def _add_bonds_tube(plotter, coords: np.ndarray, elements: list[str]) -> None:
    try:
        bonds = calculate_bonds(coords, elements)
    except Exception:
        bonds = []

    bond_radius = 0.08
    for i, j in bonds:
        line = pv.Line(coords[i], coords[j], resolution=1)
        tube = line.tube(radius=bond_radius)
        plotter.add_mesh(
            tube,
            color="#555A60",
            smooth_shading=True,
        )


def _add_atoms_pretty(
    plotter,
    coords: np.ndarray,
    elements: list[str],
    selected_index: int | None = None,
) -> None:
    for i, (xyz, el) in enumerate(zip(coords, elements)):
        sphere = pv.Sphere(
            radius=radius_for_element(el),
            center=xyz,
            theta_resolution=28,
            phi_resolution=28,
        )
        plotter.add_mesh(
            sphere,
            color=get_cpk_color(el),
            smooth_shading=True,
            specular=0.25,
            ambient=0.18,
        )


def _add_labels(plotter, coords: np.ndarray, elements: list[str]) -> None:
    if len(coords) == 0:
        return

    texts = [f"{i + 1}:{el}" for i, el in enumerate(elements)]
    plotter.add_point_labels(
        coords,
        texts,
        font_size=10,
        point_size=0,
        shape_opacity=0.0,
        always_visible=False,
    )


def _add_single_isosurface_style(
    plotter,
    surf,
    *,
    color: str,
    style: str,
    opacity: float,
    ambient: float,
    name_base: str,
) -> None:
    if surf is None or surf.n_points == 0:
        return

    if style in ("surface", "both"):
        plotter.add_mesh(
            surf,
            color=color,
            opacity=opacity,
            style="surface",
            ambient=ambient,
            smooth_shading=True,
            name=name_base + "_surf",
        )

    if style in ("mesh", "both"):
        plotter.add_mesh(
            surf,
            color=color,
            opacity=min(opacity * 1.5, 1.0),
            style="wireframe",
            line_width=1,
            name=name_base + "_wire",
        )


def _add_isosurface_for_level(
    plotter,
    grid,
    level_ppm: float,
    pos_color: str,
    neg_color: str,
    style: str,
    opacity: float,
    level_index: int,
    ambient: float = 0.2,
) -> None:
    def _add(surf, color: str, suffix: str):
        name_base = f"pcs_lv{level_index}_{suffix}"

        if style in ("surface", "both"):
            plotter.add_mesh(
                surf,
                color=color,
                opacity=opacity,
                style="surface",
                ambient=ambient,
                smooth_shading=True,
                name=name_base + "_surf",
            )
        if style in ("mesh", "both"):
            plotter.add_mesh(
                surf,
                color=color,
                opacity=min(opacity * 1.5, 1.0),
                style="wireframe",
                line_width=1,
                name=name_base + "_wire",
            )

    try:
        surf_pos = grid.contour(isosurfaces=[float(level_ppm)], scalars="values")
        if surf_pos.n_points > 0:
            _add(surf_pos, pos_color, "pos")
    except Exception:
        pass

    try:
        surf_neg = grid.contour(isosurfaces=[-float(level_ppm)], scalars="values")
        if surf_neg.n_points > 0:
            _add(surf_neg, neg_color, "neg")
    except Exception:
        pass

def _safe_unit_vector(v: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        raise ValueError(f"{name} must be a non-zero 3-vector.")
    return arr / n


def _build_oblique_plane_basis(
    z_axis: np.ndarray,
    user_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an orthonormal basis for the plane defined by:
      - z_axis
      - user_vector

    Returns
    -------
    (plane_normal, inplane_perp_to_z, z_axis_unit)
    """
    ez = _safe_unit_vector(z_axis, "z_axis")
    v = _safe_unit_vector(user_vector, "user_vector")

    n = np.cross(ez, v)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-10:
        raise ValueError("user_vector is parallel to z_axis; plane is not uniquely defined.")
    n /= n_norm

    w = np.cross(n, ez)
    w = _safe_unit_vector(w, "inplane_perp_to_z")

    return n, w, ez


def _project_points_to_plane_2d(
    points: np.ndarray,
    origin: np.ndarray,
    inplane_x: np.ndarray,
    inplane_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rel = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    x2d = rel @ np.asarray(inplane_x, dtype=float)
    y2d = rel @ np.asarray(inplane_y, dtype=float)
    return x2d, y2d


def _parse_signed_contour_levels(text: str | None) -> np.ndarray | None:
    """
    Parse a comma-separated contour level string.

    Example
    -------
    "1,2,5,10" -> [-10, -5, -2, -1, 1, 2, 5, 10]
    """
    if text is None:
        return None

    s = str(text).strip()
    if not s:
        return None

    vals = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(abs(float(p)))

    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        return None

    neg = [-v for v in reversed(vals)]
    pos = vals
    return np.asarray(neg + pos, dtype=float)


def show_oblique_pcs_slice_plot(
    *,
    atoms: list[tuple[str, float, float, float]],
    pcs_field: np.ndarray,
    ext: np.ndarray,
    metal_xyz: np.ndarray,
    z_axis: np.ndarray,
    user_vector: np.ndarray,
    plane_tol_atoms: float = 0.8,
    levels: int = 31,
    contour_levels_text: str | None = None,
    cmap: str = "custom_blue_red",
    show_atom_labels: bool = False,
    title: str = "PCS oblique slice",
    save_path: str | None = None,
    dpi: int = 600,
    transparent: bool = False,
):
    if pv is None:
        raise RuntimeError(f"PyVista import failed: {_PV_ERROR}")

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    grid_pcs = _make_uniform_grid_from_ext(
        np.asarray(ext, dtype=float),
        np.asarray(pcs_field, dtype=float),
    )

    plane_normal, inplane_x, inplane_y = _build_oblique_plane_basis(z_axis, user_vector)

    slc = grid_pcs.slice(
        normal=plane_normal,
        origin=np.asarray(metal_xyz, dtype=float),
    )

    pts = np.asarray(slc.points, dtype=float)
    vals = np.asarray(slc.point_data["values"], dtype=float)

    if pts.size == 0 or vals.size == 0:
        raise RuntimeError("Slice plane did not intersect the PCS grid.")

    x2d, y2d = _project_points_to_plane_2d(
        pts,
        np.asarray(metal_xyz, dtype=float),
        inplane_x,
        inplane_y,
    )

    custom_levs = _parse_signed_contour_levels(contour_levels_text)

    if custom_levs is not None:
        levs = custom_levs
    else:
        vmax = float(np.nanmax(np.abs(vals)))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        levs = np.linspace(-vmax, vmax, int(levels))

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    if cmap == "custom_blue_red":
        cmap_obj = LinearSegmentedColormap.from_list(
            "custom_blue_red",
            [
                (0.0, "#0000FF"),
                (0.5, "#FFFFFF"),
                (1.0, "#FF0000"),
            ],
            N=256,
        )
    else:
        cmap_obj = cmap

    cf = ax.tricontourf(
        x2d,
        y2d,
        vals,
        levels=levs,
        cmap=cmap_obj,
        extend="both",
    )

    line_levs = levs[::2] if len(levs) >= 4 else levs
    ax.tricontour(
        x2d,
        y2d,
        vals,
        levels=line_levs,
        colors="k",
        linewidths=0.2,
        alpha=0.12,
    )

    atom_rows = []
    for idx, (el, x, y, z) in enumerate(atoms, start=1):
        p = np.array([x, y, z], dtype=float)
        rel = p - np.asarray(metal_xyz, dtype=float)
        dist_from_plane = abs(float(np.dot(rel, plane_normal)))
        if dist_from_plane > float(plane_tol_atoms):
            continue

        ax_x = float(np.dot(rel, inplane_x))
        ax_y = float(np.dot(rel, inplane_y))
        atom_rows.append((idx, el, ax_x, ax_y, dist_from_plane))

        ax.scatter(
            ax_x,
            ax_y,
            s=40,
            c=get_cpk_color(el),
            edgecolors="gray",
            linewidths=1.0,
            zorder=4,
        )

        if show_atom_labels:
            ax.text(
                ax_x,
                ax_y,
                f"{idx}:{el}",
                fontsize=7,
                ha="left",
                va="bottom",
                color="#333333",
                zorder=5,
            )

    ax.scatter(
        [0.0],
        [0.0],
        s=80,
        c="gold",
        edgecolors="black",
        linewidths=1.0,
        zorder=6,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("orthogonal to z-axis (Å)")
    ax.set_ylabel("z-axis direction (Å)")
    ax.set_title(title)

    cbar = fig.colorbar(
        cf,
        ax=ax,
        label="PCS (ppm)",
        fraction=0.05,
        pad=0.03,
        shrink=0.90,
        aspect=30,
    )
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("PCS (ppm)", fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(
            save_path,
            dpi=int(dpi),
            bbox_inches="tight",
            transparent=bool(transparent),
        )
        print(f"[slice] saved plot: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return {
        "slice_polydata": slc,
        "plane_normal": np.asarray(plane_normal, dtype=float),
        "inplane_x": np.asarray(inplane_x, dtype=float),
        "inplane_y": np.asarray(inplane_y, dtype=float),
        "x2d": np.asarray(x2d, dtype=float),
        "y2d": np.asarray(y2d, dtype=float),
        "values": np.asarray(vals, dtype=float),
        "atom_rows": atom_rows,
    }

def compute_pcs_pde_result(
    atoms: list,
    origin: np.ndarray,
    spacing: tuple[float, float, float],
    rho: np.ndarray,
    chi_tensor: np.ndarray,
    params: dict,
    *,
    chi_raw: np.ndarray | None = None,
    temperature: float | None = None,
) -> dict:
    if pv is None:
        raise RuntimeError(f"PyVista import failed: {_PV_ERROR}")

    rho = np.asarray(rho, dtype=float)
    origin = np.asarray(origin, dtype=float)
    chi_tensor = np.asarray(chi_tensor, dtype=float)

    fft_pad_factor = int(params.get("fft_pad_factor", 2))
    normalize_density = bool(params.get("normalize_density", True))
    normalization_target = float(params.get("normalization_target", 1.0))

    chi_r2 = rank2_chi(chi_tensor)

    coords = np.array([[x, y, z] for _, x, y, z in atoms], dtype=float)
    elements = [el for el, *_ in atoms]

    pcs_field, source_term, origin_used, spacing_used, meta = compute_pcs_field_from_density(
        rho=rho,
        origin=origin,
        spacing=spacing,
        chi_tensor=chi_tensor,
        fft_pad_factor=fft_pad_factor,
        normalize_density=normalize_density,
        normalization_target=normalization_target,
    )

    rho_used = np.asarray(meta.get("rho_used"), dtype=float)
    ext_used = np.asarray(meta["ext"], dtype=float)

    metal_idx = _guess_metal_index(elements)
    metal_xyz = coords[metal_idx]

    pcs_pde = _interpolate_scalar_field_cubic_from_ext(
        points=coords,
        field=pcs_field,
        ext=ext_used,
    )

    pcs_point = point_pcs_from_tensor(
        points=coords,
        metal_xyz=metal_xyz,
        chi_tensor=chi_tensor,
    )

    atom_rows = []
    for ref, (el, xyz, v_pde, v_point) in enumerate(
        zip(elements, coords, pcs_pde, pcs_point), start=1
    ):
        delta = float(v_pde - v_point) if (np.isfinite(v_pde) and np.isfinite(v_point)) else np.nan
        atom_rows.append({
            "ref": ref,
            "atom": el,
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
            "pcs_pde": float(v_pde) if np.isfinite(v_pde) else np.nan,
            "pcs_point": float(v_point) if np.isfinite(v_point) else np.nan,
            "delta": delta,
        })

    _print_atom_pcs_tsv(atom_rows)

    return {
        "pcs_field": pcs_field,
        "source_term": source_term,
        "origin": np.asarray(origin_used, dtype=float),
        "spacing": tuple(float(v) for v in spacing_used),
        "ext": np.asarray(ext_used, dtype=float),
        "rho_used": rho_used,
        "metal_index": int(metal_idx),
        "metal_xyz": np.asarray(metal_xyz, dtype=float),
        "atom_rows": atom_rows,
        "temperature": None if temperature is None else float(temperature),
        "chi_raw": None if chi_raw is None else np.asarray(chi_raw, dtype=float),
        "chi_converted": np.asarray(chi_tensor, dtype=float),
        "chi_rank2": np.asarray(chi_r2, dtype=float),
        "fft_pad_factor": int(fft_pad_factor),
        "normalize_density": bool(normalize_density),
        "normalization_target": float(normalization_target),
        "normalization_info": meta.get("normalization_info"),
        "density_before": meta.get("density_before"),
        "density_after": meta.get("density_after"),
        "atoms": list(atoms),
    }


def _populate_pcs_pde_scene(
    plotter,
    result: dict,
    params: dict,
) -> None:
    if pv is None:
        raise RuntimeError(f"PyVista import failed: {_PV_ERROR}")

    atoms = result["atoms"]
    coords = np.array([[x, y, z] for _, x, y, z in atoms], dtype=float)
    elements = [el for el, *_ in atoms]
    metal_idx = int(result["metal_index"])

    rho_used = np.asarray(result["rho_used"], dtype=float)
    pcs_field = np.asarray(result["pcs_field"], dtype=float)
    ext_used = np.asarray(result["ext"], dtype=float)

    density_iso_frac = float(params.get("density_iso", 0.005))
    show_atoms = bool(params.get("show_atoms", True))
    show_bonds = bool(params.get("show_bonds", True))
    show_density = bool(params.get("show_density", True))
    show_pcs = bool(params.get("show_pcs", True))
    show_labels = bool(params.get("show_labels", False))
    show_grid = bool(params.get("show_grid", False))
    show_outline = bool(params.get("show_outline", False))
    background = str(params.get("background_color", "white"))

    density_style = str(params.get("density_style", "both"))
    density_color = str(params.get("density_color", "#27af91"))
    density_opacity = float(params.get("density_opacity", 0.15))

    ambient_light = float(params.get("ambient_light", 0.50))
    smooth_pcs_display = bool(params.get("smooth_pcs_display", False))
    smooth_pcs_sigma = float(params.get("smooth_pcs_sigma", 1.0))
    camera_preset = str(params.get("camera_preset", "iso"))
    level_styles = list(params.get("level_styles", []))

    plotter.clear()
    plotter.set_background(background)

    if show_atoms:
        _add_atoms_pretty(plotter, coords, elements, selected_index=metal_idx)

    if show_bonds and len(coords) >= 2:
        _add_bonds_tube(plotter, coords, elements)

    if show_labels:
        _add_labels(plotter, coords, elements)

    grid_rho = _make_uniform_grid_from_ext(ext_used, rho_used)

    if smooth_pcs_display:
        pcs_for_display = gaussian_filter(pcs_field, sigma=smooth_pcs_sigma)
    else:
        pcs_for_display = pcs_field
    grid_pcs = _make_uniform_grid_from_ext(ext_used, pcs_for_display)

    rho_max = float(np.max(np.abs(rho_used)))
    density_iso = max(rho_max * density_iso_frac, 1e-12)

    if show_density:
        try:
            rho_iso = grid_rho.contour(isosurfaces=[density_iso], scalars="values")
            _add_single_isosurface_style(
                plotter,
                rho_iso,
                color=density_color,
                style=density_style,
                opacity=density_opacity,
                ambient=ambient_light,
                name_base="density_iso",
            )
        except Exception as exc:
            print(f"[viewer] density contour failed: {exc}")

    if show_pcs:
        for idx, item in enumerate(level_styles):
            try:
                lv = abs(float(item["ppm"]))
                if lv <= 0:
                    continue
                _add_isosurface_for_level(
                    plotter,
                    grid_pcs,
                    level_ppm=lv,
                    pos_color=str(item["pos_color"]),
                    neg_color=str(item["neg_color"]),
                    style=str(item["style"]),
                    opacity=float(item["opacity"]),
                    level_index=idx,
                    ambient=ambient_light,
                )
            except Exception as exc:
                print(f"[viewer] level render failed ({item}): {exc}")

    if show_outline:
        plotter.add_mesh(grid_rho.outline(), color="gray", line_width=1)

    plotter.add_axes()

    if show_grid:
        plotter.show_grid()

    try:
        plotter.camera_position = CAMERA_PRESETS.get(camera_preset, "iso")
    except Exception:
        pass


def open_or_refresh_pcs_pde_view(
    result: dict,
    params: dict,
    *,
    plotter=None,
    window_size=(900, 900),
):
    if pv is None:
        raise RuntimeError(f"PyVista import failed: {_PV_ERROR}")

    try:
        if plotter is not None:
            _ = plotter.renderer.actors
    except Exception:
        plotter = None

    created = False
    if plotter is None:
        plotter = pv.Plotter(title="PCS-PDE Viewer", window_size=window_size)
        created = True

    _populate_pcs_pde_scene(plotter, result, params)

    if created:
        plotter.show(auto_close=False)
    else:
        try:
            plotter.render()
            plotter.update()
        except Exception:
            pass

    return plotter


def close_pcs_pde_view(plotter):
    if plotter is None:
        return
    try:
        plotter.close()
    except Exception:
        pass


def export_pcs_pde_png(
    result: dict,
    params: dict,
    path: str,
    *,
    dpi: int = 600,
    width_inch: float = 6.0,
    transparent: bool = False,
):
    if pv is None:
        raise RuntimeError(f"PyVista import failed: {_PV_ERROR}")

    target_px = int(round(float(width_inch) * int(dpi)))
    off = pv.Plotter(off_screen=True, window_size=(target_px, target_px))
    _populate_pcs_pde_scene(off, result, params)
    off.screenshot(path, transparent_background=bool(transparent))
    off.close()