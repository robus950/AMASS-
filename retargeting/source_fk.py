"""Step 3: Forward kinematics on the 14-joint hc_* skeleton.

FK formula: p_child = p_parent + R_parent @ offset_rest
where offset_rest = p_child_rest - p_parent_rest
"""

import numpy as np
from .skeleton_def import (
    HC_REST_POSE, FK_EDGES, FK_JOINTS_ORDERED,
    SRC_TO_HC, NPZ_IDX, RAW_ORDER_24,
)


def quat_to_rotmat(q_xyzw):
    """
    Convert quaternion (xyzw) to 3x3 rotation matrix.
    """
    q = np.asarray(q_xyzw, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    shape = q.shape[:-1]
    R = np.zeros(shape + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (yy + zz)
    R[..., 0, 1] = 2 * (xy - zw)
    R[..., 0, 2] = 2 * (xz + yw)
    R[..., 1, 0] = 2 * (xy + zw)
    R[..., 1, 1] = 1 - 2 * (xx + zz)
    R[..., 1, 2] = 2 * (yz - xw)
    R[..., 2, 0] = 2 * (xz - yw)
    R[..., 2, 1] = 2 * (yz + xw)
    R[..., 2, 2] = 1 - 2 * (xx + yy)
    return R


def compute_fk(quats_rh):
    """
    Compute global positions and rotations for all FK joints.

    Parameters
    ----------
    quats_rh : np.ndarray [F, 24, 4] float64
        NPZ quaternions in right-handed frame (xyzw).

    Returns
    -------
    positions : np.ndarray [F, N_FK, 3]
    rotations : np.ndarray [F, N_FK, 3, 3]
    """
    n_frames = quats_rh.shape[0]
    n_joints = len(FK_JOINTS_ORDERED)

    positions = np.zeros((n_frames, n_joints, 3), dtype=np.float64)
    rotations = np.zeros((n_frames, n_joints, 3, 3), dtype=np.float64)

    fk_idx = {name: i for i, name in enumerate(FK_JOINTS_ORDERED)}

    rest_offset = {}
    rest_pos = {}
    for name in FK_JOINTS_ORDERED:
        rest_pos[name] = HC_REST_POSE[name].copy()

    for child, parent in FK_EDGES.items():
        rest_offset[child] = rest_pos[child] - rest_pos[parent]

    for f in range(n_frames):
        root_idx = fk_idx["hc_Abdomen"]
        root_quat = quats_rh[f, NPZ_IDX["Abdomen"]]
        rotations[f, root_idx] = quat_to_rotmat(root_quat)
        positions[f, root_idx] = rest_pos["hc_Abdomen"].copy()

        for child_name in FK_JOINTS_ORDERED[1:]:
            ci = fk_idx[child_name]
            parent_name = FK_EDGES[child_name]
            pi = fk_idx[parent_name]

            R_parent = rotations[f, pi]
            p_parent = positions[f, pi]

            offset = rest_offset[child_name]
            positions[f, ci] = p_parent + R_parent @ offset

            if child_name in _FK_TO_NPZ:
                npz_name = _FK_TO_NPZ[child_name]
                R_local = quat_to_rotmat(quats_rh[f, NPZ_IDX[npz_name]])
                rotations[f, ci] = R_parent @ R_local
            else:
                rotations[f, ci] = R_parent.copy()

    return positions, rotations


# Reverse mapping: FK joint name -> NPZ joint name
_FK_TO_NPZ = {v: k for k, v in SRC_TO_HC.items()}


def compute_fk_positions_only(quats_rh):
    """Convenience: compute FK, return only positions [F, N_FK, 3]."""
    positions, _ = compute_fk(quats_rh)
    return positions


def compute_fk_rotations_only(quats_rh):
    """Convenience: compute FK, return only rotations [F, N_FK, 3, 3]."""
    _, rotations = compute_fk(quats_rh)
    return rotations
