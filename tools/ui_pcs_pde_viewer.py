"""
3D PyVista viewer for PCS-PDE results.

FFT version:
- zero padding
- optional density normalization
- cubic interpolation for nuclear PCS extraction
- optional automatic contour rescaling for visualization when density
  normalization is enabled

Extended viewer features:
- pretty atom rendering
- tube bonds
- density style: surface / mesh / both
- PCS style: surface / mesh / both
- oblique 2D PCS slice plotting for a plane defined by:
    metal centre + z-axis + user vector
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

    # 전체 좌표를 인덱스로 변환
    u = (pts[:, 0] - x_min) / dx_i
    v = (pts[:, 1] - y_min) / dy_i
    w = (pts[:, 2] - z_min) / dz_i

    # 그리드 범위 밖은 nan 처리
    inside = (
        (u >= 0) & (u <= nx - 1) &
        (v >= 0) & (v <= ny - 1) &
        (w >= 0) & (w <= nz - 1)
    )

    out = np.full(len(pts), np.nan, dtype=float)

    if np.any(inside):
        # map_coordinates에 한 번에 넘김 → prefilter 1회만 실행
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


# ---------------------------------------------------------------------------
# 3D rendering helpers
# ---------------------------------------------------------------------------

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

        if selected_index is not None and i == selected_index:
            ring = pv.Sphere(
                radius=radius_for_element(el) * 1.28,
                center=xyz,
                theta_resolution=28,
                phi_resolution=28,
            )
            plotter.add_mesh(
                ring,
                style="wireframe",
                color="magenta",
                line_width=2,
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


# ---------------------------------------------------------------------------
# Oblique 2D slice helpers
# ---------------------------------------------------------------------------

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

    Examples
    --------
    "1,2,5,10" -> array([-10., -5., -2., -1., 1., 2., 5., 10.])
    "" or None -> None
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
    """
    Open a matplotlib 2D contour plot for a plane defined by:

        metal centre + z-axis + user_vector

    Parameters
    ----------
    atoms :
        [(el, x, y, z), ...]
    pcs_field :
        3D PCS scalar field.
    origin, spacing :
        Grid definition for pcs_field.
    metal_xyz :
        Point through which the plane passes.
    z_axis :
        Axis that must lie in the plane.
    user_vector :
        Second direction defining the plane.
    plane_tol_atoms :
        Only atoms within this distance from the plane are shown.
    """
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
        vmax = float(np.max(np.abs(levs)))
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
                (0.0, "#0000FF"),  # negative
                (0.5, "#FFFFFF"),  # zero
                (1.0, "#FF0000"),  # positive
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

    # Metal origin marker
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
    #legend control
    cbar = fig.colorbar(
        cf,
        ax=ax,
        label="PCS (ppm)",
        fraction=0.05,  # colorbar thickness
        pad=0.03,  # distance from plot
        shrink=0.90,  # overall height
        aspect=30,  # long/thin ratio
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


# ---------------------------------------------------------------------------
# Main 3D viewer
# ---------------------------------------------------------------------------

def show_pcs_pde_view(
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

    density_iso_frac = float(params.get("density_iso", 0.005))
    pcs_level_neg_user = float(params.get("pcs_level_neg", -2.0))
    pcs_level_pos_user = float(params.get("pcs_level_pos", 2.0))

    show_bonds = bool(params.get("show_bonds", True))
    show_density = bool(params.get("show_density", True))
    show_pcs = bool(params.get("show_pcs", True))
    show_labels = bool(params.get("show_labels", False))
    show_grid = bool(params.get("show_grid", False))
    show_outline = bool(params.get("show_outline", False))

    background = str(params.get("background_color", "white"))

    fft_pad_factor = int(params.get("fft_pad_factor", 2))
    normalize_density = bool(params.get("normalize_density", True))
    normalization_target = float(params.get("normalization_target", 1.0))
    auto_scale_pcs_levels = bool(params.get("auto_scale_pcs_levels", True))

    density_style = str(params.get("density_style", "both"))
    density_color = str(params.get("density_color", "#27af91"))
    density_opacity = float(params.get("density_opacity", 0.15))

    pcs_style = str(params.get("pcs_style", "surface"))
    pcs_pos_color = str(params.get("pcs_pos_color", "#ff0000"))
    pcs_neg_color = str(params.get("pcs_neg_color", "#0000ff"))
    pcs_opacity = float(params.get("pcs_opacity", 0.30))

    ambient_light = float(params.get("ambient_light", 0.50))
    smooth_pcs_display = bool(params.get("smooth_pcs_display", False))
    smooth_pcs_sigma = float(params.get("smooth_pcs_sigma", 1.0))

    rho = np.asarray(rho, dtype=float)
    origin = np.asarray(origin, dtype=float)
    chi_tensor = np.asarray(chi_tensor, dtype=float)

    chi_r2 = rank2_chi(chi_tensor)

    print("[viewer] ─── Chi tensor summary ───")
    if chi_raw is not None:
        print(f"[viewer] χ raw from ORCA (cm^3*K/mol):\n{np.asarray(chi_raw, dtype=float)}")
    if temperature is not None:
        print(f"[viewer] temperature used: {float(temperature):.6g} K")
    print(f"[viewer] χ converted  (Å³/molecule):\n{chi_tensor}")
    print(f"[viewer] χ rank-2 traceless:\n{chi_r2}")
    print(f"[viewer] Tr(χ_converted) = {np.trace(chi_tensor):.4e}")
    print(f"[viewer] Tr(χ_rank2)     = {np.trace(chi_r2):.2e}  (≈0)")

    coords = np.array([[x, y, z] for _, x, y, z in atoms], dtype=float)
    elements = [el for el, *_ in atoms]

    print("[viewer] ─── Grid / structure summary ───")
    print(f"[viewer] rho shape: {rho.shape}")
    print(f"[viewer] origin Å: {origin}")
    print(f"[viewer] spacing Å: {spacing}")
    print(f"[viewer] atom min Å: {coords.min(axis=0)}")
    print(f"[viewer] atom max Å: {coords.max(axis=0)}")
    print(f"[viewer] grid min Å: {origin}")
    print(f"[viewer] grid max Å: {origin + np.array(spacing, dtype=float) * (np.array(rho.shape) - 1)}")

    print("[viewer] ─── Starting PCS computation (FFT branch) ───")
    print(f"[viewer] user PCS levels: [{pcs_level_neg_user}, {pcs_level_pos_user}]")
    print(f"[viewer] fft_pad_factor: {fft_pad_factor}")
    print(f"[viewer] normalize_density: {normalize_density}")
    print(f"[viewer] normalization_target: {normalization_target}")
    print(f"[viewer] auto_scale_pcs_levels: {auto_scale_pcs_levels}")

    pcs_field, source_term, origin_used, spacing_used, meta = compute_pcs_field_from_density(
        rho=rho,
        origin=origin,
        spacing=spacing,
        chi_tensor=chi_tensor,
        fft_pad_factor=fft_pad_factor,
        normalize_density=normalize_density,
        normalization_target=normalization_target,
    )

    print(f"[viewer] PCS min/max: {np.nanmin(pcs_field):.4f} / {np.nanmax(pcs_field):.4f} ppm")

    rho_used = np.asarray(meta.get("rho_used"), dtype=float)
    ext_used = np.asarray(meta["ext"], dtype=float)

    rho_max = float(np.max(np.abs(rho_used)))
    density_iso = max(rho_max * density_iso_frac, 1e-12)

    print(f"[viewer] density_iso: {density_iso:.3e}")

    scale_factor = 1.0
    norm_info = meta.get("normalization_info")
    if normalize_density and norm_info is not None:
        scale_factor = float(norm_info["scale_factor"])

    if auto_scale_pcs_levels and normalize_density:
        pcs_level_neg_disp = pcs_level_neg_user * scale_factor
        pcs_level_pos_disp = pcs_level_pos_user * scale_factor
    else:
        pcs_level_neg_disp = pcs_level_neg_user
        pcs_level_pos_disp = pcs_level_pos_user

    pcs_levels_display = [v for v in (pcs_level_neg_disp, pcs_level_pos_disp) if abs(v) > 1e-12]

    print(f"[viewer] display PCS levels: {pcs_levels_display}")
    if auto_scale_pcs_levels and normalize_density:
        print(f"[viewer] contour levels scaled by normalization factor {scale_factor:.6f}")

    metal_idx = _guess_metal_index(elements)
    metal_xyz = coords[metal_idx]

    import time

    t0 = time.perf_counter()
    pcs_pde = _interpolate_scalar_field_cubic_from_ext(
        points=coords,
        field=pcs_field,
        ext=ext_used,
    )
    print(f"[time] interpolate_scalar_field: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    pcs_point = point_pcs_from_tensor(
        points=coords,
        metal_xyz=metal_xyz,
        chi_tensor=chi_tensor,
    )
    print(f"[time] point_pcs_from_tensor: {time.perf_counter() - t0:.2f}s")

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

    t0 = time.perf_counter()
    pl = pv.Plotter(title="PCS-PDE Viewer", window_size=(900, 900))
    pl.set_background(background)
    print(f"[time] pv.Plotter() init: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    _add_atoms_pretty(pl, coords, elements, selected_index=metal_idx)
    print(f"[time] add_atoms_pretty: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    if show_bonds and len(coords) >= 2:
        _add_bonds_tube(pl, coords, elements)
    print(f"[time] add_bonds_tube: {time.perf_counter() - t0:.2f}s")

    if show_labels:
        _add_labels(pl, coords, elements)

    t0 = time.perf_counter()
    grid_rho = _make_uniform_grid_from_ext(ext_used, rho_used)
    if smooth_pcs_display:
        pcs_for_display = gaussian_filter(pcs_field, sigma=smooth_pcs_sigma)
        print(f"[viewer] PCS display smoothing: gaussian sigma={smooth_pcs_sigma}")
    else:
        pcs_for_display = pcs_field
    grid_pcs = _make_uniform_grid_from_ext(ext_used, pcs_for_display)
    print(f"[time] make_uniform_grid (rho+pcs): {time.perf_counter() - t0:.2f}s")

    if show_density:
        t0 = time.perf_counter()
        try:
            rho_iso = grid_rho.contour(isosurfaces=[density_iso], scalars="values")
            _add_single_isosurface_style(
                pl,
                rho_iso,
                color=density_color,
                style=density_style,
                opacity=density_opacity,
                ambient=ambient_light,
                name_base="density_iso",
            )
        except Exception as exc:
            print(f"[viewer] density contour failed: {exc}")
        print(f"[time] density contour + add_mesh: {time.perf_counter() - t0:.2f}s")

    if show_pcs and pcs_levels_display:
        t0 = time.perf_counter()
        abs_levels = sorted({abs(float(v)) for v in pcs_levels_display if abs(v) > 1e-12})
        for idx, lv in enumerate(abs_levels):
            _add_isosurface_for_level(
                pl,
                grid_pcs,
                level_ppm=lv,
                pos_color=pcs_pos_color,
                neg_color=pcs_neg_color,
                style=pcs_style,
                opacity=pcs_opacity,
                level_index=idx,
                ambient=ambient_light,
            )
        print(f"[time] pcs contour + add_mesh: {time.perf_counter() - t0:.2f}s")

    if show_outline:
        pl.add_mesh(grid_rho.outline(), color="gray", line_width=1)

    pl.add_axes()

    if show_grid:
        pl.show_grid()

    try:
        pl.camera_position = "iso"
    except Exception:
        pass

    print("[viewer] opening window…")
    t0 = time.perf_counter()
    pl.show(auto_close=False)
    print(f"[time] pl.show() (until window closed): {time.perf_counter() - t0:.2f}s")
    print("[viewer] window closed")

    return {
        "pcs_field": pcs_field,
        "source_term": source_term,
        "origin": np.asarray(origin_used, dtype=float),
        "spacing": tuple(float(v) for v in spacing_used),
        "ext": np.asarray(ext_used, dtype=float),
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
        "normalization_info": norm_info,
        "density_before": meta.get("density_before"),
        "density_after": meta.get("density_after"),
        "auto_scale_pcs_levels": bool(auto_scale_pcs_levels),
        "pcs_level_neg_user": float(pcs_level_neg_user),
        "pcs_level_pos_user": float(pcs_level_pos_user),
        "pcs_level_neg_display": float(pcs_level_neg_disp),
        "pcs_level_pos_display": float(pcs_level_pos_disp),
    }