"""Padding sensitivity checks without widgets or changes to the base result."""

import numpy as np


def padding_check_options(params):
    """Validate an opt-in comparison; tolerance is an absolute PCS value in ppm."""
    if not params.get("padding_check_enabled", False):
        return None
    try:
        factor = float(params.get("padding_check_factor", 3))
        tolerance = float(params.get("padding_check_tolerance", 0.01))
    except (TypeError, ValueError):
        raise ValueError("Padding comparison factor and tolerance must be numeric.") from None
    if not np.isfinite(factor) or factor != int(factor) or factor <= params["fft_pad_factor"]:
        raise ValueError("Comparison padding factor must be an integer greater than the base factor.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Padding tolerance must be positive and finite (ppm).")
    return {"comparison_factor": int(factor), "tolerance_ppm": tolerance}


def compare_atom_pcs(base, comparison, options):
    """Match every non-centre atom by reference, element and exact coordinates."""
    metal_ref = int(base["metal_index"]) + 1
    expected = {i for i in range(1, len(base["atoms"]) + 1) if i != metal_ref}

    def indexed(rows):
        result, duplicates = {}, set()
        for row in rows:
            ref = row["ref"]
            if ref in result:
                duplicates.add(ref)
            result[ref] = row
        return result, duplicates

    left, dup_left = indexed(base.get("atom_rows", []))
    right, dup_right = indexed(comparison.get("atom_rows", []))
    differences, invalid = [], []
    for ref in sorted(expected):
        a, b = left.get(ref), right.get(ref)
        atom = base["atoms"][ref - 1]
        if (a is None or b is None or ref in dup_left | dup_right
                or a["atom"] != atom[0] or b["atom"] != atom[0]
                or not np.array_equal([a[k] for k in ("x", "y", "z")], atom[1:])
                or not np.array_equal([b[k] for k in ("x", "y", "z")], atom[1:])
                or not np.isfinite(a["pcs_pde"]) or not np.isfinite(b["pcs_pde"])):
            invalid.append(ref)
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            difference = float(np.float64(b["pcs_pde"]) - np.float64(a["pcs_pde"]))
        if not np.isfinite(difference):
            invalid.append(ref)
            continue
        differences.append((ref, atom[0], difference))
    values = np.array([row[2] for row in differences])
    maximum = float(np.max(np.abs(values))) if values.size else None
    worst = max(differences, key=lambda row: abs(row[2])) if differences else None
    complete = bool(expected) and not invalid
    return dict(options, base_factor=base["fft_pad_factor"],
                status=("Incomplete" if not complete else "Within tolerance"
                        if maximum <= options["tolerance_ppm"] else "Above tolerance"),
                expected_atoms=len(expected), compared_atoms=len(differences), invalid_refs=invalid,
                max_abs_ppm=maximum,
                rms_ppm=(float(maximum * np.sqrt(np.mean((values / maximum) ** 2)))
                         if maximum else 0.0 if values.size else None),
                worst_ref=worst[0] if worst else None, worst_atom=worst[1] if worst else None,
                atom_differences=[{"ref": ref, "atom": atom, "delta_ppm": value}
                                  for ref, atom, value in differences])


def compute_padding_check(snapshot, base, options, compute):
    """Run only the larger grid with the base density scale held fixed.

    For legacy after-padding normalization, reuse its recorded multiplier rather
    than renormalizing the new padded array. Only small comparison statistics
    survive this function; the comparison volume is released on return.
    """
    kwargs = snapshot.compute_kwargs()
    params = kwargs["params"]
    params["fft_pad_factor"] = options["comparison_factor"]
    params["normalize_density"] = False
    scale = 1.0
    if base["normalize_density"]:
        scale = float((base.get("normalization_info") or {}).get("scale_factor", np.nan))
        if not np.isfinite(scale):
            raise ValueError("Base density normalization scale is unavailable; recalculate first.")
    kwargs["rho"] = snapshot.rho * scale
    comparison = compute(**kwargs)
    return dict(compare_atom_pcs(base, comparison, options), normalization_scale=scale)


def padding_check_failure(base_factor, options, error):
    return dict(options, base_factor=base_factor, status="Failed", error=str(error))


def format_padding_check(check):
    lines = ["\nPadding Convergence\n", f"  Factors: {check['base_factor']} -> {check['comparison_factor']}\n",
             f"  Tolerance: {check['tolerance_ppm']:.6g} ppm\n", f"  Status: {check['status']}\n"]
    if "error" in check:
        lines.append(f"  Error: {check['error']}\n")
    else:
        lines.append(f"  Atoms: {check['compared_atoms']}/{check['expected_atoms']} non-centre pairs\n")
        if check["max_abs_ppm"] is not None:
            lines.extend([f"  Max |delta PCS|: {check['max_abs_ppm']:.6g} ppm "
                          f"({check['worst_atom']}, Ref {check['worst_ref']})\n",
                          f"  RMS delta PCS: {check['rms_ppm']:.6g} ppm\n"])
        if check["invalid_refs"]:
            lines.append("  Missing/invalid references: " + ", ".join(map(str, check["invalid_refs"])) + "\n")
        lines.append("  Same density normalization scale used for both calculations.\n")
    lines.append(f"  Base padding {check['base_factor']} result retained.\n")
    return "".join(lines)
