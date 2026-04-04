# tools/logic_pcs_pde.py
"""
FFT PCS-PDE forward solver.

This module provides:
- susceptibility tensor rank-2 extraction
- density diagnostics / normalization
- array-size-factor zero padding
- FFT-based PDE PCS calculation
- point-dipole PCS reference
"""

from __future__ import annotations

import numpy as np

try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as _fft
    pyfftw.interfaces.cache.enable()
    pyfftw.config.NUM_THREADS = __import__("os").cpu_count()
    PYFFTW_AVAILABLE = True
    PYFFTW_THREADS = pyfftw.config.NUM_THREADS
    print("[fft] using pyfftw")
except ImportError:
    import numpy.fft as _fft
    PYFFTW_AVAILABLE = False
    PYFFTW_THREADS = 1
    print("[fft] no pyfftw found, using numpy.fft")


def rank2_chi(chi_tensor: np.ndarray) -> np.ndarray:
    chi = np.asarray(chi_tensor, dtype=float)
    if chi.shape != (3, 3):
        raise ValueError("chi_tensor must be (3, 3).")
    return chi - np.eye(3, dtype=float) * (np.trace(chi) / 3.0)


def summarize_density(
    rho: np.ndarray,
    spacing: tuple[float, float, float],
    label: str = "rho",
) -> dict:
    rho = np.asarray(rho, dtype=float)
    dx, dy, dz = map(float, spacing)
    voxel_volume = dx * dy * dz

    s = float(np.sum(rho))
    sa = float(np.sum(np.abs(rho)))
    integ_sum = s * voxel_volume
    integ_abs_sum = sa * voxel_volume
    integ_trapz = float(_triple_trapz(rho, spacing))

    out = {
        "min": float(np.min(rho)),
        "max": float(np.max(rho)),
        "sum": s,
        "abs_sum": sa,
        "voxel_volume": float(voxel_volume),
        "integral_sum": float(integ_sum),
        "integral_abs_sum": float(integ_abs_sum),
        "integral_trapz": float(integ_trapz),
    }

    print(f"[density] {label}")
    print(f"[density] min/max        = {out['min']:.6e} / {out['max']:.6e}")
    print(f"[density] sum            = {out['sum']:.6e}")
    print(f"[density] abs_sum        = {out['abs_sum']:.6e}")
    print(f"[density] voxel_volume   = {out['voxel_volume']:.6e}")
    print(f"[density] sum*dV         = {out['integral_sum']:.6e}")
    print(f"[density] abs_sum*dV     = {out['integral_abs_sum']:.6e}")
    print(f"[density] trapz integral = {out['integral_trapz']:.6e}")

    return out


def _triple_trapz(
    rho: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """
    Replicate nested trapz integration over the array, then multiply by dx*dy*dz.
    """
    dx, dy, dz = map(float, spacing)
    val = np.trapezoid(
        np.trapezoid(
            np.trapezoid(rho, axis=0),
            axis=0,
        ),
        axis=0,
    )
    return float(val * dx * dy * dz)


def normalize_density_to_integral(
    rho: np.ndarray,
    spacing: tuple[float, float, float],
    target_integral: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Normalize rho so that the nested trapezoidal integral equals target_integral.
    """
    rho = np.asarray(rho, dtype=float)
    current_integral = _triple_trapz(rho, spacing)

    if abs(current_integral) < 1e-20:
        raise ValueError("Density integral is too small to normalize.")

    scale = float(target_integral) / current_integral
    rho_norm = rho * scale

    info = {
        "old_integral": float(current_integral),
        "target_integral": float(target_integral),
        "scale_factor": float(scale),
        "new_integral": float(_triple_trapz(rho_norm, spacing)),
    }

    print("[density] normalization")
    print(f"[density] old trapz     = {info['old_integral']:.6e}")
    print(f"[density] target        = {info['target_integral']:.6e}")
    print(f"[density] scale factor  = {info['scale_factor']:.6e}")
    print(f"[density] new trapz     = {info['new_integral']:.6e}")

    return rho_norm, info


def zero_pad_density(
    rho: np.ndarray,
    origin: np.ndarray,
    spacing: tuple[float, float, float],
    pad_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], np.ndarray]:
    """
    Zero-pad symmetrically by a multiple of the original array size.

    Returns
    -------
    rho_pad, origin_pad, spacing_orig, ext_pad

    ext_pad is:
        [xmin, xmax, ymin, ymax, zmin, zmax]
    using the same convention as the reference workflow.
    """
    rho = np.asarray(rho, dtype=float)
    origin = np.asarray(origin, dtype=float)

    p = int(pad_factor)
    dx, dy, dz = map(float, spacing)
    nx, ny, nz = rho.shape

    # Original extents:
    # [x0, x0+(nx-1)dx, ...]
    ext = np.array([
        origin[0], origin[0] + (nx - 1) * dx,
        origin[1], origin[1] + (ny - 1) * dy,
        origin[2], origin[2] + (nz - 1) * dz,
    ], dtype=float)

    if p <= 0:
        return rho, origin, spacing, ext

    # Pad by multiples of original size on each side
    px, py, pz = p * nx, p * ny, p * nz

    rho_pad = np.pad(
        rho,
        ((px, px), (py, py), (pz, pz)),
        mode="constant",
        constant_values=0.0,
    )

    # Expand ext by original box lengths on each side
    xlen = abs(ext[1] - ext[0])
    ylen = abs(ext[3] - ext[2])
    zlen = abs(ext[5] - ext[4])

    ext_pad = np.array([
        ext[0] - xlen * p, ext[1] + xlen * p,
        ext[2] - ylen * p, ext[3] + ylen * p,
        ext[4] - zlen * p, ext[5] + zlen * p,
    ], dtype=float)

    # Keep origin only as metadata for display if needed
    origin_pad = np.array([ext_pad[0], ext_pad[2], ext_pad[4]], dtype=float)

    print(f"[fft-pad] pad_factor={p}")
    print(f"[fft-pad] old shape={rho.shape}  new shape={rho_pad.shape}")
    print(f"[fft-pad] old ext={ext}")
    print(f"[fft-pad] new ext={ext_pad}")

    return rho_pad, origin_pad, spacing, ext_pad


def _fft_k_axis(n: int, d: float) -> np.ndarray:
    if n <= 0:
        raise ValueError("Grid size n must be positive.")
    if d <= 0:
        raise ValueError("Grid spacing d must be positive.")
    freq = np.fft.fftfreq(n, d=d)
    return 1j * 2.0 * np.pi * freq

# fast calc
def _rfft_k_axis(n: int, d: float) -> np.ndarray:
    """rfft용 (nz//2+1 길이)"""
    if n <= 0:
        raise ValueError("Grid size n must be positive.")
    if d <= 0:
        raise ValueError("Grid spacing d must be positive.")
    freq = np.fft.rfftfreq(n, d=d)
    return 1j * 2.0 * np.pi * freq

def compute_pcs_field_from_density(
    rho: np.ndarray,
    origin: np.ndarray,
    spacing: tuple[float, float, float],
    chi_tensor: np.ndarray,
    *,
    fft_pad_factor: int = 2,
    normalize_density: bool = True,
    normalization_target: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float], dict]:
    rho = np.asarray(rho, dtype=float)
    origin = np.asarray(origin, dtype=float)
    chi_tensor = np.asarray(chi_tensor, dtype=float)

    if rho.ndim != 3:
        raise ValueError("rho must be a 3D array.")

    rho_work, origin_work, spacing_work, ext_work = zero_pad_density(
        rho=rho,
        origin=origin,
        spacing=spacing,
        pad_factor=fft_pad_factor,
    )

    nx, ny, nz = rho_work.shape

    # Effective FFT spacings from padded extents
    dx_fft = float((ext_work[1] - ext_work[0]) / nx)
    dy_fft = float((ext_work[3] - ext_work[2]) / ny)
    dz_fft = float((ext_work[5] - ext_work[4]) / nz)

    chi_r2 = rank2_chi(chi_tensor)

    print(f"[fft] full grid shape: {rho_work.shape}  ({rho_work.size} voxels)")
    print(f"[fft] chi rank-2 trace check: {np.trace(chi_r2):.2e}")
    print(f"[fft] original spacing  = {spacing_work}")
    print(f"[fft] effective spacing = ({dx_fft:.6f}, {dy_fft:.6f}, {dz_fft:.6f})")
    print(f"[fft] ext              = {ext_work}")

    density_before = summarize_density(rho_work, spacing_work, label="rho_work_before_norm")

    norm_info = None
    if normalize_density:
        rho_work, norm_info = normalize_density_to_integral(
            rho_work,
            spacing_work,
            target_integral=normalization_target,
        )
        density_after = summarize_density(rho_work, spacing_work, label="rho_work_after_norm")
    else:
        density_after = density_before

    # ── fast calc 교체 후 ────────────────────────────────────────────
    kx = _fft_k_axis(nx, dx_fft)[:, None, None]
    ky = _fft_k_axis(ny, dy_fft)[None, :, None]
    kz = _rfft_k_axis(nz, dz_fft)[None, None, :]  # ← rfftfreq, 길이 nz//2+1

    # meshgrid 없이 브로드캐스팅으로 직접 계산
    chi = chi_r2
    K = -(1.0 / 3.0) * (
            chi[0, 0] * kx * kx
            + chi[0, 1] * kx * ky
            + chi[0, 2] * kx * kz
            + chi[1, 0] * ky * kx
            + chi[1, 1] * ky * ky
            + chi[1, 2] * ky * kz
            + chi[2, 0] * kz * kx
            + chi[2, 1] * kz * ky
            + chi[2, 2] * kz * kz
    )

    L = kx * kx + ky * ky + kz * kz
    L[0, 0, 0] = 1.0 + 0.0j  # in-place DC 처리, L.copy() 불필요

    rho_f = _fft.rfftn(rho_work)
    source_like_f = K * rho_f  # source_like용 따로 보관
    rho_f *= K  # in-place, 임시 배열 없음
    rho_f /= L  # in-place

    pcs_field = 1e6 * _fft.irfftn(rho_f, s=rho_work.shape)
    pcs_field = np.asarray(pcs_field, dtype=float)

    source_like = 1e6 * np.real(_fft.irfftn(source_like_f, s=rho_work.shape))

    print(f"[fft] σ_PCS: min={float(np.nanmin(pcs_field)):.4f}  max={float(np.nanmax(pcs_field)):.4f} ppm")

    meta = {
        "fft_pad_factor": int(fft_pad_factor),
        "normalize_density": bool(normalize_density),
        "normalization_target": float(normalization_target),
        "density_before": density_before,
        "density_after": density_after,
        "normalization_info": norm_info,
        "rho_used": rho_work,
        "ext": np.asarray(ext_work, dtype=float),
        "effective_spacing_fft": (dx_fft, dy_fft, dz_fft),
    }

    return pcs_field, source_like, origin_work, spacing_work, meta


def point_pcs_from_tensor(
    points: np.ndarray,
    metal_xyz: np.ndarray,
    chi_tensor: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    metal = np.asarray(metal_xyz, dtype=float)
    chi = rank2_chi(np.asarray(chi_tensor, dtype=float))

    out = np.full(len(pts), np.nan, dtype=float)
    I = np.eye(3, dtype=float)

    for i, p in enumerate(pts):
        R = p - metal
        r = float(np.linalg.norm(R))
        if r < 1e-12:
            continue
        T = 3.0 * np.outer(R, R) / (r ** 5) - I / (r ** 3)
        out[i] = 1e6 * float(np.trace(T @ chi) / (12.0 * np.pi))

    return out