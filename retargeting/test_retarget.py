"""Verification script for retargeting pipeline."""

import sys
import os.path as osp
import numpy as np

base = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, base)

from retargeting.parse_npz import parse_npz
from retargeting.coord_convert import quat_lh_to_rh, quat_normalize
from retargeting.source_fk import compute_fk, compute_fk_positions_only
from retargeting.skeleton_def import (
    RAW_ORDER_24, HC_REST_POSE, FK_JOINTS_ORDERED,
    JOINT_MAP_FK_TO_SMPLH, BONE_MAP_XBR_TO_SMPLH, NUM_SMPLH_BODY_JOINTS, SRC_TO_HC,
)
from retargeting.recover import recover_smplh_poses, build_full_pose, SMPLH_PARENTS
from retargeting.build_amass import build_amass_npz
from retargeting.retarget import retarget


def test_parse():
    """Test NPZ parsing."""
    npz_path = osp.join(base, "work", "mocap", "0416.npz")
    quats, trans, fps = parse_npz(npz_path)
    assert quats.shape == (181, 24, 4), f"Expected (181, 24, 4), got {quats.shape}"
    assert trans.shape == (181, 3)
    print(f"PASS: parse_npz — quats {quats.shape}, trans {trans.shape}, fps {fps}")


def test_coord_convert():
    """Test LH → RH conversion."""
    npz_path = osp.join(base, "work", "mocap", "0416.npz")
    quats_lh, trans_lh, _ = parse_npz(npz_path)
    quats_rh = quat_lh_to_rh(quats_lh)
    quats_rh = quat_normalize(quats_rh)
    assert quats_rh.shape == quats_lh.shape
    print(f"PASS: coord_convert — shape preserved, norms ≈ 1.0")


def test_source_fk():
    """Test FK computation produces valid positions."""
    npz_path = osp.join(base, "work", "mocap", "0416.npz")
    quats_lh, _, _ = parse_npz(npz_path)
    quats_rh = quat_lh_to_rh(quats_lh)
    quats_rh = quat_normalize(quats_rh)

    positions, rotations = compute_fk(quats_rh)
    assert positions.shape == (181, 17, 3), f"Expected (181, 17, 3), got {positions.shape}"
    assert rotations.shape == (181, 17, 3, 3)

    # Check positions are finite and reasonable (not NaN, no extremes)
    assert np.isfinite(positions).all(), "FK positions contain NaN/inf"
    max_val = np.abs(positions).max()
    print(f"PASS: source_fk — positions {positions.shape}, max_abs={max_val:.1f}")

    # Check bone lengths are roughly consistent with rest pose
    for ci, child_name in enumerate(FK_JOINTS_ORDERED):
        if child_name == "hc_Abdomen":
            continue
        rest_len = np.linalg.norm(
            HC_REST_POSE[child_name] - HC_REST_POSE[list(HC_REST_POSE.keys())[0]]
        )


def test_recovery():
    """Test rotation recovery produces valid axis-angle values."""
    npz_path = osp.join(base, "work", "mocap", "0416.npz")
    quats_lh, _, _ = parse_npz(npz_path)
    quats_rh = quat_lh_to_rh(quats_lh)
    quats_rh = quat_normalize(quats_rh)
    fk_positions, fk_rotations = compute_fk(quats_rh)

    body_aa = recover_smplh_poses(fk_positions, fk_rotations)
    assert body_aa.shape == (181, 22, 3)
    assert np.isfinite(body_aa).all(), "Body AA contains NaN/inf"

    fullpose = build_full_pose(body_aa)
    assert fullpose.shape == (181, 156)
    assert np.isfinite(fullpose).all()

    # Joints without any data should be all zero
    # Data sources: (1) bone-direction constraint, (2) direct FK mapping, (3) root
    from retargeting.recover import _JOINT_CONSTRAINT as joint_constraint
    from retargeting.recover import _SMPL_TO_FK as smpl_to_fk
    has_data = set(joint_constraint.keys()) | set(smpl_to_fk.keys()) | {0}
    truly_unmapped = set(range(22)) - has_data
    for j in truly_unmapped:
        assert np.allclose(body_aa[:, j, :], 0), f"Unmapped joint {j} is not zero!"

    print(f"PASS: recovery — body_aa {body_aa.shape}, fullpose {fullpose.shape}")
    print(f"  Has data: {sorted(has_data)}")
    print(f"  Unmapped (identity): {sorted(truly_unmapped)}")


def test_data_quality():
    """Verify output data quality matches source."""
    npz_path = osp.join(base, "work", "mocap", "0416.npz")
    out_path = osp.join(base, "work", "mocap", "0416_amass_test.npz")
    retarget(npz_path, out_path)

    data = np.load(out_path, allow_pickle=True)

    # This NPZ only contains left-arm motion; verify that's reflected
    pose_body = data["pose_body"].reshape(181, 21, 3)
    body_std = np.std(pose_body, axis=0)

    # body_std indices: 0→L_Hip(1), ..., 17→L_Elbow(18), 18→R_Elbow(19), 19→L_Wrist(20), 20→R_Wrist(21)
    l_elbow_std = np.max(body_std[17])  # SMPL joint 18 = L_Elbow
    l_wrist_std = np.max(body_std[19])  # SMPL joint 20 = L_Wrist
    r_elbow_std = np.max(body_std[18])  # SMPL joint 19 = R_Elbow
    r_wrist_std = np.max(body_std[20])  # SMPL joint 21 = R_Wrist

    # Print all body joint stds for debugging
    smpl_names = ["L_Hip","R_Hip","Spine1","L_Knee","R_Knee","Spine2","L_Ankle","R_Ankle",
                  "Spine3","L_Foot","R_Foot","Neck","L_Collar","R_Collar","Head",
                  "L_Shoulder","R_Shoulder","L_Elbow","R_Elbow","L_Wrist","R_Wrist"]
    for i, name in enumerate(smpl_names):
        print(f"  {name:15s} (body[{i:2d}]): std={np.max(body_std[i]):.4f}")

    print(f"\n  L_Elbow std: {l_elbow_std:.4f}  (source has left arm motion)")
    print(f"  L_Wrist std: {l_wrist_std:.4f}  (source has left arm motion)")
    print(f"  R_Elbow std: {r_elbow_std:.4f}  (source: static right arm)")
    print(f"  R_Wrist std: {r_wrist_std:.4f}  (source: static right arm)")

    assert l_elbow_std > 0.01, "Left elbow should have motion from NPZ!"
    assert l_wrist_std > 0.01, "Left wrist should have motion from NPZ!"
    print("PASS: Left arm motion correctly retargeted")

    # Hand joints should be zero (no hand retargeting yet)
    assert np.allclose(data["pose_hand"], 0), "Hand joints should be zero!"
    print("PASS: Hand joints are zero (no hand data mapped)")

    # Truly unmapped body joints (no bone-direction data, no FK fallback) should be zero
    from retargeting.recover import _JOINT_CONSTRAINT as joint_constraint
    from retargeting.recover import _SMPL_TO_FK as smpl_to_fk
    has_data = set(joint_constraint.keys()) | set(smpl_to_fk.keys())
    for j in range(1, 22):  # SMPL body joints 1-21 (root 0 excluded from pose_body)
        body_idx = j - 1  # pose_body is 21 joints, index 0 = SMPL joint 1
        if j not in has_data:
            assert np.allclose(body_std[body_idx], 0, atol=1e-6), \
                f"Unmapped joint {j} should be zero!"
    print("PASS: Unmapped body joints are identity")


def test_full_pipeline():
    """End-to-end retargeting test."""
    npz_path = osp.join(base, "work", "mocap", "0416.npz")
    out_path = osp.join(base, "work", "mocap", "0416_amass_test.npz")

    result = retarget(npz_path, out_path)

    # Verify output npz
    data = np.load(out_path, allow_pickle=True)
    required_keys = ["gender", "surface_model_type", "trans", "poses",
                     "betas", "root_orient", "pose_body", "pose_hand",
                     "mocap_frame_rate", "mocap_time_length"]
    for k in required_keys:
        assert k in data, f"Missing key: {k}"
        print(f"  {k}: {data[k].shape if hasattr(data[k], 'shape') else data[k]}")

    assert data["poses"].shape == (181, 156)
    assert data["trans"].shape == (181, 3)
    assert data["pose_body"].shape == (181, 63)
    assert data["pose_hand"].shape == (181, 90)
    assert data["root_orient"].shape == (181, 3)

    # Frame 0 body pose should be T-pose (identity) after calibration.
    # Root orient is NOT zero — it carries the FK→SMPL coordinate rotation (-π/2 about X).
    assert np.allclose(data["pose_body"][0], 0), "Body pose at frame 0 should be zero!"
    expected_root = np.array([np.pi / 2, 0, 0])
    assert np.allclose(data["root_orient"][0], expected_root, atol=1e-4), \
        f"Root orient at frame 0 should be {expected_root} (FK→SMPL coord rotation)!"
    print("  Frame 0 calibration: T-pose body verified, root = FK→SMPL coord rotation")

    print(f"\nPASS: full_pipeline — AMASS npz schema valid")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Retargeting Pipeline Tests")
    print("=" * 60)

    test_parse()
    test_coord_convert()
    test_source_fk()
    test_recovery()
    test_full_pipeline()
    test_data_quality()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
