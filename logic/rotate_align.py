# logic/rotate_align.py

import numpy as np

def euler_matrix(angle_x, angle_y, angle_z, order="XYZ", degrees=True):
    """
    Active rotation matrix for extrinsic rotations about fixed axes.
    With row-vector coords (N,3) using coords @ R.T, this applies rotations in the
    order given by `order` (e.g., "XYZ" means X then Y then Z).

    order: "XYZ", "ZYX" etc.
    """
    ax, ay, az = angle_x, angle_y, angle_z
    if degrees:
        ax = np.radians(ax); ay = np.radians(ay); az = np.radians(az)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax),  np.cos(ax)]], dtype=float)

    Ry = np.array([[ np.cos(ay), 0, np.sin(ay)],
                   [0,           1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]], dtype=float)

    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az),  np.cos(az), 0],
                   [0,           0,          1]], dtype=float)

    mats = {"X": Rx, "Y": Ry, "Z": Rz}

    order = (order or "XYZ").upper()
    if any(ch not in "XYZ" for ch in order) or len(order) != 3:
        raise ValueError(f"Invalid order: {order} (use e.g. 'XYZ', 'ZYX')")

    # For column-vector convention: v' = R @ v, applying X then Y then Z => R = Rz @ Ry @ Rx
    # We keep that same meaning and later apply with row-vectors as coords @ R.T
    R = np.eye(3)
    for ch in order:
        R = mats[ch] @ R
    return R

def rotate_euler(coords, angle_x, angle_y, angle_z, order="XYZ"):
    """
    Rotate around origin using a fixed extrinsic Euler order (default: XYZ).
    coords: (N,3) row-vectors
    """
    coords = np.asarray(coords, dtype=float)
    R = euler_matrix(angle_x, angle_y, angle_z, order=order, degrees=True)
    return coords @ R.T

def rotate_coordinates(coords, angle_x, angle_y, angle_z, center, order="XYZ"):
    """
    중심(center) 기준으로 회전 → 다시 center 복귀.
    - filter_atoms에서 center=(0,0,0)로 호출하면 원점 기준 회전만 수행
    - 3D창에서는 중심 원자 좌표를 center로 넣어 '그 점'을 축으로 회전
    """
    c = np.asarray(center, float)
    coords0 = np.asarray(coords, float) - c
    rot0 = rotate_euler(coords0, angle_x, angle_y, angle_z, order=order)
    return rot0 + c

def rotate_about_center(coords, angle_x, angle_y, angle_z, center, order="XYZ"):
    """(하위호환) 중심 기준 회전 후 복귀"""
    return rotate_coordinates(coords, angle_x, angle_y, angle_z, center, order=order)

def align_xyz(vec1, vec2, coords):
    """Align vec1->vec2 via Rodrigues formula; keep first atom at origin."""
    coords0 = np.asarray(coords, float) - np.asarray(coords, float)[0]
    v1 = np.asarray(vec1, dtype=float);
    v2 = np.asarray(vec2, dtype=float)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return coords0
    v1 /= np.linalg.norm(v1);
    v2 /= np.linalg.norm(v2)
    k = np.cross(v1, v2);
    c = float(np.dot(v1, v2))
    if np.linalg.norm(k) < 1e-12:
        return -coords0 if c < 0 else coords0
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=float)
    R = np.eye(3) + K + K @ K * ((1 - c) / (np.linalg.norm(k) ** 2))
    rot = coords0 @ R.T
    rot -= rot[0]
    return rot

def translate_to_origin(coords, center):
    return coords - center

def translate_from_origin(coords, center):
    return coords + center