"""Step 4: Recover SMPL-H local axis-angle rotations from FK data.

Uses bone-direction alignment: for each SMPL bone (parent→child), the parent
joint's local rotation is computed to align the SMPL rest bone direction with
the corresponding XBR bone direction in world space.
"""

import numpy as np
from .skeleton_def import (
    JOINT_MAP_FK_TO_SMPLH, BONE_MAP_XBR_TO_SMPLH,
    SMPL_REST_POSE, NUM_SMPLH_BODY_JOINTS, NUM_SMPLH_JOINTS,
    FK_JOINTS_ORDERED,
)


def rotmat_to_axis_angle(R):
    """Convert 3x3 rotation matrix to axis-angle 3-vector."""
    R = np.asarray(R, dtype=np.float64)
    shape = R.shape[:-2]
    aa = np.zeros(shape + (3,), dtype=np.float64)

    cos_theta = np.clip((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    small = theta < 1e-8
    large = ~small

    if np.any(large):
        sin_theta = np.sin(theta[large])
        rx = (R[..., 2, 1] - R[..., 1, 2])[large] / (2 * sin_theta)
        ry = (R[..., 0, 2] - R[..., 2, 0])[large] / (2 * sin_theta)
        rz = (R[..., 1, 0] - R[..., 0, 1])[large] / (2 * sin_theta)

        k = np.stack([rx, ry, rz], axis=-1)
        k_norm = np.linalg.norm(k, axis=-1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            k = np.where(k_norm > 1e-12, k / k_norm, k)
        aa[large] = k * theta[large, None]

    if np.any(small):
        rx = (R[..., 2, 1] - R[..., 1, 2])[small] / 2.0
        ry = (R[..., 0, 2] - R[..., 2, 0])[small] / 2.0
        rz = (R[..., 1, 0] - R[..., 0, 1])[small] / 2.0
        aa[small] = np.stack([rx, ry, rz], axis=-1)

    return aa


def axis_angle_to_rotmat(aa):
    """Convert axis-angle 3-vector to 3x3 rotation matrix (Rodrigues)."""
    aa = np.asarray(aa, dtype=np.float64)
    scalar_input = aa.ndim == 1
    if scalar_input:
        aa = aa[None, :]
    theta = np.linalg.norm(aa, axis=-1)

    shape = aa.shape[:-1]
    R = np.zeros(shape + (3, 3), dtype=np.float64)

    small = theta < 1e-8
    large = ~small

    R[small, 0, 0] = 1.0
    R[small, 1, 1] = 1.0
    R[small, 2, 2] = 1.0

    idx = np.where(large)
    k = aa[idx] / theta[idx, None]
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    ct = np.cos(theta[idx])
    st = np.sin(theta[idx])
    vt = 1 - ct

    R_large = R[idx]
    R_large[..., 0, 0] = kx * kx * vt + ct
    R_large[..., 0, 1] = kx * ky * vt - kz * st
    R_large[..., 0, 2] = kx * kz * vt + ky * st
    R_large[..., 1, 0] = kx * ky * vt + kz * st
    R_large[..., 1, 1] = ky * ky * vt + ct
    R_large[..., 1, 2] = ky * kz * vt - kx * st
    R_large[..., 2, 0] = kx * kz * vt - ky * st
    R_large[..., 2, 1] = ky * kz * vt + kx * st
    R_large[..., 2, 2] = kz * kz * vt + ct
    R[idx] = R_large

    if scalar_input:
        return R[0]
    return R


def rotation_between_vectors(a, b):
    """
    Minimal rotation matrix that rotates unit vector a to unit vector b.

    Returns (R, angle) where R is the [3,3] rotation matrix and angle is in radians.
    """
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    cos_angle = np.clip(np.dot(a, b), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if angle < 1e-8:
        return np.eye(3)

    axis = np.cross(a, b)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-12:
        # Vectors are parallel or anti-parallel
        if cos_angle > 0:
            return np.eye(3)
        # Anti-parallel: 180-degree rotation about any perpendicular axis
        if abs(a[0]) < 0.9:
            perp = np.array([1.0, 0.0, 0.0])
        else:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis = axis / np.linalg.norm(axis)
    else:
        axis = axis / axis_norm

    return axis_angle_to_rotmat(axis * angle)


# SMPL-H kintree parents for body joints (indices 0-21)
SMPLH_PARENTS = np.array([
    -1,   # 0:  Pelvis
     0,   # 1:  L_Hip
     0,   # 2:  R_Hip
     0,   # 3:  Spine1
     1,   # 4:  L_Knee
     2,   # 5:  R_Knee
     3,   # 6:  Spine2
     4,   # 7:  L_Ankle
     5,   # 8:  R_Ankle
     6,   # 9:  Spine3
     7,   # 10: L_Foot
     8,   # 11: R_Foot
     9,   # 12: Neck
     9,   # 13: L_Collar
     9,   # 14: R_Collar
    12,   # 15: Head
    13,   # 16: L_Shoulder
    14,   # 17: R_Shoulder
    16,   # 18: L_Elbow
    17,   # 19: R_Elbow
    18,   # 20: L_Wrist
    19,   # 21: R_Wrist
], dtype=np.int32)


def _precompute_rest_dirs():
    """Precompute SMPL rest bone directions (world frame at rest)."""
    rest_dirs = {}
    for child_idx in range(1, NUM_SMPLH_BODY_JOINTS):
        parent_idx = SMPLH_PARENTS[child_idx]
        if parent_idx < 0:
            continue
        direction = SMPL_REST_POSE[child_idx] - SMPL_REST_POSE[parent_idx]
        rest_dirs[(parent_idx, child_idx)] = direction / np.linalg.norm(direction)
    return rest_dirs


# Precomputed at import time: (parent_idx, child_idx) → normalized rest direction
_SMPL_REST_DIRS = _precompute_rest_dirs()


# Build a reverse map: for each joint, which child bone provides its direction constraint?
# joint_idx → (xbr_parent_name, xbr_child_name, smpl_child_idx)
def _build_joint_constraint_map():
    """For each SMPL joint, find which XBR bone constrains its local rotation."""
    joint_to_constraint = {}  # joint_idx → (xbr_p, xbr_c, smpl_child_idx)
    for (p_idx, c_idx), (xbr_p, xbr_c) in BONE_MAP_XBR_TO_SMPLH.items():
        # Bone (p_idx → c_idx) constrains R_local(p_idx)
        # Only set if p_idx doesn't already have a constraint
        if p_idx not in joint_to_constraint:
            joint_to_constraint[p_idx] = (xbr_p, xbr_c, c_idx)
    return joint_to_constraint


_JOINT_CONSTRAINT = _build_joint_constraint_map()


def _xbr_to_smpl_direction(dir_xbr):
    """
    Convert a direction vector from XBR FK coordinate axes to SMPL coordinate axes.

    XBR FK (after conjugate LH→RH, identity root):
      X=left,  Y=forward,  Z=up
    SMPL:
      X=left,  Y=up,       Z=forward

    Transform: (x, y, z) → (x, z, y)
    (X unchanged since both use X=left after LH→RH conversion;
     swap Y↔Z because XBR has Z=up while SMPL has Y=up)
    """
    return np.array([dir_xbr[0], dir_xbr[2], dir_xbr[1]], dtype=np.float64)


def _xbr_to_smpl_rotation(R_xbr):
    """
    Convert a 3×3 rotation matrix from XBR FK coordinate axes to SMPL axes.

    R_smpl = S @ R_xbr @ S^T  where S is the axis swap matrix.
    XBR FK (after conjugate):  X=left, Y=forward, Z=up
    SMPL:                       X=left, Y=up,      Z=forward
    v_xbr = S @ v_smpl: (x, y, z) → (x, z, y)
    S = [[1,0,0],[0,0,1],[0,1,0]]
    """
    S = np.array([[1, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0]], dtype=np.float64)
    # R_smpl = S^T @ R_xbr @ S  (since S is orthogonal, S^{-1} = S^T)
    return S.T @ R_xbr @ S


# Reverse mapping: SMPL body joint index → FK joint name (only for direct FK counterparts)
_SMPL_TO_FK = {v: k for k, v in JOINT_MAP_FK_TO_SMPLH.items()}


def recover_smplh_poses(fk_positions, fk_rotations):
    """
    Recover SMPL-H local axis-angle body poses from FK data.

    Uses bone-direction alignment for joints with a child bone constraint,
    and falls back to direct FK rotation for leaf joints (wrists, ankles).

    Parameters
    ----------
    fk_positions : np.ndarray [F, N_FK, 3]  (XBR coords, cm)
    fk_rotations : np.ndarray [F, N_FK, 3, 3]  (XBR coords)

    Returns
    -------
    smplh_poses : np.ndarray [F, 22, 3]
    """
    n_frames = fk_positions.shape[0]
    n_body = NUM_SMPLH_BODY_JOINTS
    fk_idx = {name: i for i, name in enumerate(FK_JOINTS_ORDERED)}

    smplh_local_aa = np.zeros((n_frames, n_body, 3), dtype=np.float64)
    smplh_global_R = np.zeros((n_body, 3, 3), dtype=np.float64)

    # Topological order (parents before children)
    smplh_order = list(range(n_body))

    # Rotation from FK coords (Z=up) to SMPL coords (Y=up):
    # +90° around X axis maps FK Z-up to SMPL Y-up.
    R_fk_to_smpl = axis_angle_to_rotmat(np.array([np.pi / 2, 0.0, 0.0]))

    for f in range(n_frames):
        smplh_global_R[0] = R_fk_to_smpl.copy()
        smplh_local_aa[f, 0] = rotmat_to_axis_angle(R_fk_to_smpl)

        # ── Process joints in topological order ──
        for j in smplh_order[1:]:  # skip root
            parent_j = SMPLH_PARENTS[j]
            R_parent = smplh_global_R[parent_j]

            # Check if joint j has a bone constraint (j is the PARENT of a mapped bone)
            if j in _JOINT_CONSTRAINT:
                xbr_p_name, xbr_c_name, smpl_child_idx = _JOINT_CONSTRAINT[j]

                if xbr_p_name in fk_idx and xbr_c_name in fk_idx:
                    # XBR bone direction (XBR axes, cm)
                    p_xbr = fk_positions[f, fk_idx[xbr_p_name]]
                    c_xbr = fk_positions[f, fk_idx[xbr_c_name]]
                    bone_xbr = c_xbr - p_xbr
                    bone_len = np.linalg.norm(bone_xbr)

                    if bone_len > 1e-6:
                        # Convert to SMPL axes and normalize
                        bone_dir_smpl = _xbr_to_smpl_direction(bone_xbr)
                        bone_dir_smpl = bone_dir_smpl / bone_len

                        # Target direction in joint j's parent frame
                        target_dir = R_parent.T @ bone_dir_smpl

                        # SMPL rest bone direction from j to smpl_child_idx
                        rest_dir = _SMPL_REST_DIRS[(j, smpl_child_idx)]

                        R_local = rotation_between_vectors(rest_dir, target_dir)

                        smplh_global_R[j] = R_parent @ R_local
                        smplh_local_aa[f, j] = rotmat_to_axis_angle(R_local)
                        continue

            # Fallback: leaf joints with direct FK counterpart use FK global rotation
            if j in _SMPL_TO_FK:
                fk_name = _SMPL_TO_FK[j]
                if fk_name in fk_idx:
                    R_global_xbr = fk_rotations[f, fk_idx[fk_name]]
                    R_global = _xbr_to_smpl_rotation(R_global_xbr)
                    smplh_global_R[j] = R_global
                    R_local = R_parent.T @ R_global
                    smplh_local_aa[f, j] = rotmat_to_axis_angle(R_local)
                    continue

            # No data at all: identity rotation
            smplh_global_R[j] = R_parent.copy()
            smplh_local_aa[f, j] = np.zeros(3, dtype=np.float64)

    return smplh_local_aa


def build_full_pose(smplh_body_aa):
    """Build full SMPL-H pose vector from body joint axis-angles."""
    n_frames = smplh_body_aa.shape[0]
    fullpose = np.zeros((n_frames, NUM_SMPLH_JOINTS * 3), dtype=np.float64)

    for j in range(NUM_SMPLH_BODY_JOINTS):
        fullpose[:, j * 3:(j + 1) * 3] = smplh_body_aa[:, j, :]

    return fullpose
