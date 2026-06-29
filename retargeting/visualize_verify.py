"""Visual verification: compare source XBR FK joints with retargeted SMPL-H FK joints."""

import sys
import os.path as osp
import numpy as np

base = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, base)

from retargeting.parse_npz import parse_npz
from retargeting.coord_convert import quat_lh_to_rh, quat_normalize
from retargeting.source_fk import compute_fk
from retargeting.skeleton_def import (
    FK_JOINTS_ORDERED, HC_REST_POSE, FK_EDGES,
    JOINT_MAP_FK_TO_SMPLH, SMPL_REST_POSE,
    NUM_SMPLH_BODY_JOINTS, SMPLH_BODY_JOINT_NAMES,
)
from retargeting.recover import SMPLH_PARENTS


def compute_smpl_fk(pose_body_aa, root_orient_aa):
    """
    Compute SMPL-H body FK joint positions from axis-angle poses.

    Parameters
    ----------
    pose_body_aa : [F, 63] or [F, 21, 3]
        21 body joints axis-angle (SMPL joints 1-21)
    root_orient_aa : [F, 3]
        Root orientation (Pelvis, joint 0)

    Returns
    -------
    positions : [F, 22, 3]
        Global joint positions in meters
    """
    from retargeting.recover import axis_angle_to_rotmat

    n_frames = pose_body_aa.shape[0]
    if pose_body_aa.ndim == 2:
        pose_body_aa = pose_body_aa.reshape(n_frames, 21, 3)

    # Build full 22-joint axis-angle array
    full_aa = np.zeros((n_frames, 22, 3), dtype=np.float64)
    full_aa[:, 0, :] = root_orient_aa
    full_aa[:, 1:, :] = pose_body_aa

    # Rest pose in meters
    rest = SMPL_REST_POSE.copy()  # [22, 3]

    positions = np.zeros((n_frames, 22, 3), dtype=np.float64)

    for f in range(n_frames):
        # Compute local rotation matrices
        R_local = np.zeros((22, 3, 3), dtype=np.float64)
        for j in range(22):
            R_local[j] = axis_angle_to_rotmat(full_aa[f, j])

        # FK
        R_global = np.zeros((22, 3, 3), dtype=np.float64)
        for j in range(22):
            parent = SMPLH_PARENTS[j]
            if parent < 0:
                R_global[j] = R_local[j]
                positions[f, j] = rest[j]
            else:
                R_global[j] = R_global[parent] @ R_local[j]
                offset = rest[j] - rest[parent]
                positions[f, j] = positions[f, parent] + R_global[parent] @ offset

    return positions


def compute_xbr_fk(quats_rh):
    """Compute XBR FK positions (in cm, converted to meters)."""
    positions, _ = compute_fk(quats_rh)
    return positions / 100.0  # cm → m


def compare_joint_motion(npz_path, amass_path):
    """Compare source and retargeted joint motion."""

    # Load source
    quats_lh, _, _ = parse_npz(npz_path)
    quats_rh = quat_lh_to_rh(quats_lh)
    quats_rh = quat_normalize(quats_rh)
    xbr_pos = compute_xbr_fk(quats_rh)  # [F, 17, 3] in meters

    # Load retargeted
    data = np.load(amass_path, allow_pickle=True)
    pose_body = data["pose_body"]  # [F, 63]
    root_orient = data["root_orient"]  # [F, 3]
    smpl_pos = compute_smpl_fk(pose_body, root_orient)  # [F, 22, 3] in meters

    n_frames = smpl_pos.shape[0]

    # ── Per-joint motion summary ──
    print("=" * 70)
    print("Joint Motion Comparison (std of position across frames, in meters)")
    print("=" * 70)
    print(f"{'SMPL Joint':18s} {'SMPL std':>10s} {'XBR src':>10s} {'Note'}")
    print("-" * 70)

    fk_name_to_idx = {name: i for i, name in enumerate(FK_JOINTS_ORDERED)}

    for smpl_j in range(22):
        smpl_std = np.std(smpl_pos[:, smpl_j, :], axis=0)
        smpl_max_std = np.max(smpl_std)

        # Find corresponding XBR joint
        xbr_std_str = "N/A"
        for fk_name, sj in JOINT_MAP_FK_TO_SMPLH.items():
            if sj == smpl_j and fk_name in fk_name_to_idx:
                xbr_pos_j = xbr_pos[:, fk_name_to_idx[fk_name], :]
                xbr_std = np.std(xbr_pos_j, axis=0)
                xbr_max_std = np.max(xbr_std)
                xbr_std_str = f"{xbr_max_std:.4f}"
                break

        motion_flag = "← MOVING" if smpl_max_std > 0.01 else ""
        print(f"{SMPLH_BODY_JOINT_NAMES[smpl_j]:18s} {smpl_max_std:10.4f} {xbr_std_str:>10s} {motion_flag}")

    # ── Frame-by-frame: check that only left arm moves ──
    print("\n" + "=" * 70)
    print("Key Frame Positions (meters) — verifying T-pose + left arm motion")
    print("=" * 70)

    key_frames = [0, 45, 90, 135, 180]
    for f in key_frames:
        if f >= n_frames:
            break
        print(f"\n--- Frame {f} ---")
        # Left arm: Shoulder → Elbow → Wrist
        l_shoulder = smpl_pos[f, 16]
        l_elbow = smpl_pos[f, 18]
        l_wrist = smpl_pos[f, 20]
        r_shoulder = smpl_pos[f, 17]
        r_elbow = smpl_pos[f, 19]
        r_wrist = smpl_pos[f, 21]

        # Bone lengths
        l_upper_len = np.linalg.norm(l_elbow - l_shoulder)
        l_forearm_len = np.linalg.norm(l_wrist - l_elbow)
        r_upper_len = np.linalg.norm(r_elbow - r_shoulder)
        r_forearm_len = np.linalg.norm(r_wrist - r_elbow)

        print(f"  L_Shoulder: {l_shoulder}")
        print(f"  L_Elbow:    {l_elbow}  (upper arm: {l_upper_len:.3f}m)")
        print(f"  L_Wrist:    {l_wrist}  (forearm: {l_forearm_len:.3f}m)")
        print(f"  R_Shoulder: {r_shoulder}")
        print(f"  R_Elbow:    {r_elbow}  (upper arm: {r_upper_len:.3f}m)")
        print(f"  R_Wrist:    {r_wrist}  (forearm: {r_forearm_len:.3f}m)")

        # Angle between left upper arm and T-pose upper arm (should be > 0 for moving arm)
        l_upper_dir = (l_elbow - l_shoulder) / (l_upper_len + 1e-12)
        # T-pose left upper arm direction in SMPL: roughly along +X
        tpose_upper = np.array([1.0, 0.0, 0.0])
        angle = np.arccos(np.clip(np.dot(l_upper_dir, tpose_upper), -1, 1))
        print(f"  Left upper arm angle from T-pose: {np.degrees(angle):.1f}°")

    # ── Save OBJ stick figure for key frames ──
    print("\n" + "=" * 70)
    print("Saving OBJ stick figures...")
    print("=" * 70)

    out_dir = osp.dirname(amass_path)

    # Joint pairs to draw as bones
    bone_pairs = []
    for child in range(1, 22):
        parent = SMPLH_PARENTS[child]
        if parent >= 0:
            bone_pairs.append((parent, child))

    for f in key_frames:
        if f >= n_frames:
            break
        obj_path = osp.join(out_dir, f"frame_{f:04d}.obj")
        with open(obj_path, 'w') as fp:
            fp.write("# SMPL-H FK skeleton\n")
            fp.write(f"# Frame {f}\n")

            # Vertices (spheres at joint positions)
            for j in range(22):
                p = smpl_pos[f, j]
                fp.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

            # Bones as lines (using v for vertices, l for lines)
            for p_idx, c_idx in bone_pairs:
                fp.write(f"l {p_idx+1} {c_idx+1}\n")

        print(f"  Saved: {obj_path}")

    print(f"\nDone. OBJ files saved to {out_dir}/")
    print("Open in Blender/MeshLab to inspect the skeleton pose.")


if __name__ == "__main__":
    npz_path = osp.join(base, "work", "mocap", "0416.npz")
    amass_path = osp.join(base, "work", "mocap", "0416_amass_test.npz")
    compare_joint_motion(npz_path, amass_path)
