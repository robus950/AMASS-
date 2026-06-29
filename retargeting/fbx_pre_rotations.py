"""FBX PreRotation data extracted from XBR/SK_Mannequin.fbx.

Each bone in the FBX skeleton has a PreRotation property (Euler angles in
degrees, order X/Y/Z) that defines the bone's default orientation in its
parent frame. These must be applied as additional local rotations during FK
to correctly orient the skeleton.

Source: XBR/SK_Mannequin.fbx (binary FBX v7700)
"""

import numpy as np


# PreRotation Euler angles [X, Y, Z] in degrees for each FK bone.
# Extracted from SK_Mannequin.fbx PreRotation properties.
# FBX bones not present in our FK skeleton are omitted (twist bones, etc.).

FBX_PREROTATION_DEG = {
    "hc_Abdomen":    [90.0,    180.0,   0.0],
    "Spine":         [-92.27,    0.0,  180.0],
    "hc_Chest":      [-92.27,    0.0,  180.0],
    "neck":          [-90.0,     0.0,  180.0],
    "hc_Head":       [89.99,     0.0,    0.0],
    "hc_Shoulder_L": [87.92,    65.33, -154.21],
    "hc_Shoulder_R": [-173.41,  -79.45,   63.42],
    "hc_Elbow_L":    [1.98,     -0.63,   -3.71],
    "hc_Elbow_R":    [-6.85,    -3.83,   -0.23],
    "hc_Hand_L":     [-4.02,    -1.25,  -86.72],
    "hc_Hand_R":     [-0.02,    -0.99,   89.83],
    "hc_Hip_L":      [1.73,      0.12, -179.98],
    "hc_Hip_R":      [1.73,     -1.30,  179.96],
    "hc_Knee_L":     [7.22,     -0.47,    0.84],
    "hc_Knee_R":     [7.26,      0.03,   -0.43],
    "hc_Foot_L":     [-98.97,   -0.90,   -0.48],
    "hc_Foot_R":     [-98.97,   -0.14,    0.67],
}


def euler_xyz_to_rotmat(euler_deg):
    """Convert Euler angles [X, Y, Z] in degrees to a 3x3 rotation matrix.

    FBX rotation order is X/Y/Z: R = Rz @ Ry @ Rx
    (Rx applied first to a vector, then Ry, then Rz.)
    """
    rx, ry, rz = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]], dtype=np.float64)

    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], dtype=np.float64)

    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx


# Precomputed rotation matrices keyed by bone name.
PREROTATION_MATRICES = {
    name: euler_xyz_to_rotmat(deg)
    for name, deg in FBX_PREROTATION_DEG.items()
}
