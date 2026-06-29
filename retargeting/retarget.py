"""Main entry point: Nokov NPZ → SMPL-H / AMASS retargeting pipeline."""

import sys
import os.path as osp
import time

import numpy as np

from .parse_npz import parse_npz
from .coord_convert import quat_lh_to_rh, pos_lh_to_rh, quat_normalize
from .source_fk import compute_fk
from .recover import recover_smplh_poses, build_full_pose
from .build_amass import build_amass_npz


def retarget(npz_path, output_path, gender="male", fps=None,
             vertical_offset=0.0):
    """
    Retarget a Nokov/XBR NPZ file to SMPL-H AMASS format.

    Parameters
    ----------
    npz_path : str
        Path to input NPZ (e.g., work/mocap/0416.npz).
    output_path : str
        Path for output AMASS npz.
    gender : str
    fps : float or None
        Override FPS. If None, uses NPZ metadata or defaults to 60.
    vertical_offset : float
        Vertical offset in cm added to SMPL Y (up) translation.

    Returns
    -------
    result : dict
        Keys: poses, trans, fullpose, root_orient, pose_body, pose_hand, betas.
    """
    t0 = time.time()

    # Step 1: Parse
    print("=" * 60)
    print(f"Retargeting: {npz_path} -> {output_path}")
    print("=" * 60)
    print("\n[1/5] Parsing NPZ...")
    quats_lh, root_trans_lh, npz_fps = parse_npz(npz_path)
    n_frames = quats_lh.shape[0]
    print(f"  frames: {n_frames}, joints: {quats_lh.shape[1]}, root_trans: {root_trans_lh.shape}")
    print(f"  npz fps: {npz_fps}")

    if fps is None:
        fps = npz_fps

    # Step 2: LH -> RH
    print("\n[2/5] Converting LH -> RH coordinate system...")
    quats_rh = quat_lh_to_rh(quats_lh)
    quats_rh = quat_normalize(quats_rh)
    root_trans_rh = pos_lh_to_rh(root_trans_lh)

    # Step 3: Source FK
    print("\n[3/5] Computing source FK on 14-joint skeleton...")
    fk_positions, fk_rotations = compute_fk(quats_rh)
    print(f"  FK positions: {fk_positions.shape}")
    print(f"  FK rotations: {fk_rotations.shape}")

    # Override abdomen rotation to identity - the Nokov abdomen quaternion
    # encodes a ~180 degree calibration-frame rotation that flips the skeleton.
    from .skeleton_def import FK_JOINTS_ORDERED, FK_EDGES, HC_REST_POSE, SRC_TO_HC, NPZ_IDX
    from .source_fk import quat_to_rotmat
    fk_idx = {name: i for i, name in enumerate(FK_JOINTS_ORDERED)}
    _FK_TO_NPZ = {v: k for k, v in SRC_TO_HC.items()}
    rest_offset = {}
    for child, parent in FK_EDGES.items():
        rest_offset[child] = HC_REST_POSE[child] - HC_REST_POSE[parent]

    n_frames = quats_rh.shape[0]
    n_joints = len(FK_JOINTS_ORDERED)
    I3 = np.eye(3, dtype=np.float64)
    abdomen_idx = fk_idx["hc_Abdomen"]

    # Recompute FK with identity root
    fk_rotations[:, abdomen_idx] = I3
    for f in range(n_frames):
        fk_positions[f, abdomen_idx] = HC_REST_POSE["hc_Abdomen"].copy()
        for child_name in FK_JOINTS_ORDERED[1:]:
            ci = fk_idx[child_name]
            parent_name = FK_EDGES[child_name]
            pi = fk_idx[parent_name]
            R_parent = fk_rotations[f, pi]
            p_parent = fk_positions[f, pi]
            offset = rest_offset[child_name]
            fk_positions[f, ci] = p_parent + R_parent @ offset
            if child_name in _FK_TO_NPZ:
                npz_name = _FK_TO_NPZ[child_name]
                R_local = quat_to_rotmat(quats_rh[f, NPZ_IDX[npz_name]])
                fk_rotations[f, ci] = R_parent @ R_local
            else:
                fk_rotations[f, ci] = R_parent.copy()

    # Calibrate: compute frame-0 inverse rotations and apply to all frames.
    calib_inv = {ci: fk_rotations[0, ci].T.copy() for ci in range(n_joints)}
    for f in range(n_frames):
        fk_positions[f, abdomen_idx] = HC_REST_POSE["hc_Abdomen"].copy()
        for ci in range(n_joints):
            fk_rotations[f, ci] = calib_inv[ci] @ fk_rotations[f, ci]
        for child_name in FK_JOINTS_ORDERED[1:]:
            ci = fk_idx[child_name]
            parent_name = FK_EDGES[child_name]
            pi = fk_idx[parent_name]
            R_parent = fk_rotations[f, pi]
            p_parent = fk_positions[f, pi]
            offset = rest_offset[child_name]
            fk_positions[f, ci] = p_parent + R_parent @ offset

    # Step 4: Recover SMPL-H rotations
    print("\n[4/5] Recovering SMPL-H local axis-angle rotations...")
    smplh_body_aa = recover_smplh_poses(fk_positions, fk_rotations)
    print(f"  Body pose: {smplh_body_aa.shape}")

    # Frame-0 calibration at SMPL level: invert the local rotation of each
    # joint at frame 0, then apply to all frames so frame 0 is identity.
    # Skip pelvis (j=0): the root orientation is a fixed coordinate-system
    # rotation (FK Z-up -> SMPL Y-up), not a calibration offset.
    from .recover import axis_angle_to_rotmat, rotmat_to_axis_angle
    for j in range(1, smplh_body_aa.shape[1]):  # skip pelvis (j=0)
        R_calib = axis_angle_to_rotmat(smplh_body_aa[0, j])
        R_calib_inv = R_calib.T
        for f in range(n_frames):
            R_frame = axis_angle_to_rotmat(smplh_body_aa[f, j])
            R_calibrated = R_calib_inv @ R_frame
            smplh_body_aa[f, j] = rotmat_to_axis_angle(R_calibrated)

    fullpose = build_full_pose(smplh_body_aa)
    print(f"  Full pose: {fullpose.shape}")

    # Step 5: Build AMASS
    print("\n[5/5] Building AMASS npz...")
    trans = root_trans_rh.astype(np.float64)
    if vertical_offset != 0.0:
        trans[:, 1] += vertical_offset  # SMPL Y = up
        print(f"  Vertical offset: {vertical_offset:.1f} cm")

    build_amass_npz(fullpose, trans, output_path, gender=gender,
                    surface_model_type="smplh", fps=fps)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({n_frames/elapsed:.0f} fps)")

    return {
        "poses": fullpose,
        "trans": trans,
        "fullpose": fullpose,
        "root_orient": fullpose[:, :3],
        "pose_body": fullpose[:, 3:66],
        "pose_hand": fullpose[:, 66:],
        "betas": np.zeros(16, dtype=np.float64),
    }


if __name__ == "__main__":
    base = osp.dirname(osp.dirname(osp.abspath(__file__)))
    input_npz = osp.join(base, "work", "mocap", "0416.npz")
    output_npz = osp.join(base, "work", "mocap", "0416_amass.npz")
    retarget(input_npz, output_npz)
