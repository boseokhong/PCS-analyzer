"""UI-independent preparation of existing molecular-view coordinates.

Inputs are plain structure rows, Ref IDs and numerical/visibility values.
No Tk/Qt variables, windows or render objects belong here. This intentionally
preserves the legacy output contract (including labels in ``elements``);
renderer migration and audit fixes are separate changes. Coordinates remain
metal-centred working display coordinates, not ensemble PCS evaluation sites.
"""

import numpy as np

from logic.rotate_align import rotate_coordinates, rotate_euler


def apply_molecule_fit_override(abs_coords, ref_ids, metal, fit_override, *, donor_rotation=None):
    """
    Apply fit override to raw coordinates using the same logic used by the main 2D/table pipeline.
    """
    fo = fit_override
    if not fo:
        return abs_coords, metal

    mode = (fo.get("mode") or "").lower()

    if mode == "theta_alpha_multi":
        id2idx = {rid: i for i, rid in enumerate(ref_ids)}
        donor_ids = fo.get("donor_ids") or []
        if donor_ids:
            donor_pts = [abs_coords[id2idx[rid]] for rid in donor_ids if rid in id2idx]
            if donor_pts:
                if donor_rotation is None:
                    from logic.fitting import _angles_to_rotation_multi
                    donor_rotation = _angles_to_rotation_multi
                abs_coords = donor_rotation(
                    points=abs_coords,
                    metal=metal,
                    donor_points=donor_pts,
                    theta_deg=fo.get("theta", 0.0),
                    alpha_deg=fo.get("alpha", 0.0),
                    axis_mode=fo.get("axis_mode", "bisector"),
                )

    elif mode == "euler_global":
        ax_deg = float(fo.get("ax", 0.0))
        ay_deg = float(fo.get("ay", 0.0))
        az_deg = float(fo.get("az", 0.0))
        coords0 = abs_coords - metal
        rot0 = rotate_euler(coords0, ax_deg, ay_deg, az_deg)
        abs_coords = rot0 + metal

    elif mode == "full_tensor":
        euler_deg = fo.get("euler_deg", (0.0, 0.0, 0.0))
        ax_deg, ay_deg, az_deg = euler_deg

        metal_pos = fo.get("metal_pos")
        if metal_pos is not None:
            metal = np.array(metal_pos, dtype=float)

        coords0 = abs_coords - metal
        rot0 = rotate_euler(coords0, ax_deg, ay_deg, az_deg)
        abs_coords = rot0 + metal

    return abs_coords, metal

def prepare_molecule_view_data(
    atom_data, raw_ref_ids, *, metal, angles=(0.0, 0.0, 0.0),
    fit_override=None, visible_ids, reference_atom_data=(), reference_ref_ids=(),
    moving_ref_ids=(), donor_rotation=None,
):
    """
    Build the currently visible 3D coordinates using:
    - raw structure
    - current metal center
    - fit override
    - current x/y/z rotation
    - current element checklist
    """
    if not atom_data:
        return None

    coords_abs = np.array([[x, y, z] for _, x, y, z in atom_data], dtype=float)
    labels = [a for a, *_ in atom_data]
    ref_ids = list(raw_ref_ids)

    metal = np.array(metal, dtype=float)

    original_metal = metal.copy()
    coords_abs, metal = apply_molecule_fit_override(
        coords_abs, ref_ids, metal, fit_override, donor_rotation=donor_rotation
    )

    coords0 = coords_abs - metal

    ax_deg, ay_deg, az_deg = (float(value) for value in angles)

    coords_rot = rotate_coordinates(coords0, ax_deg, ay_deg, az_deg, (0.0, 0.0, 0.0))

    rows = []
    for label, coord, rid in zip(labels, coords_rot, ref_ids):
        if int(rid) not in visible_ids:
            continue
        rows.append((label, coord, rid))

    if not rows:
        return None

    out_labels = [r[0] for r in rows]
    out_coords = np.array([r[1] for r in rows], dtype=float)
    out_ref_ids = [r[2] for r in rows]
    out_elements = list(out_labels)

    result = {
        "labels": out_labels,
        "coords": out_coords,
        "ref_ids": out_ref_ids,
        "elements": out_elements,
    }

    # Optional reference geometry for enabled torsional fragments.
    if reference_atom_data:
        raw = reference_atom_data
        raw_ids2 = list(reference_ref_ids or range(1, len(raw) + 1))
        if raw and len(raw_ids2) == len(raw):
            ref_abs = np.asarray([[x, y, z] for _, x, y, z in raw], dtype=float)
            ref_labels0 = [a for a, *_ in raw]
            ref_abs, ref_metal = apply_molecule_fit_override(
                ref_abs, raw_ids2, original_metal.copy(), fit_override,
                donor_rotation=donor_rotation,
            )
            ref0 = ref_abs - ref_metal
            ref_rot = rotate_coordinates(ref0, ax_deg, ay_deg, az_deg, (0.0, 0.0, 0.0))
            moving_refs = set(moving_ref_ids)
            ref_rows = []
            for lbl, coord, rid in zip(ref_labels0, ref_rot, raw_ids2):
                if rid not in moving_refs:
                    continue
                if int(rid) not in visible_ids:
                    continue
                ref_rows.append((lbl, coord, rid))
            if ref_rows:
                result["reference_labels"] = [r[0] for r in ref_rows]
                result["reference_coords"] = np.asarray([r[1] for r in ref_rows], dtype=float)
                result["reference_ref_ids"] = [r[2] for r in ref_rows]

    return result
