"""Step 5: Build AMASS-format npz from SMPL-H pose data."""

import os.path as osp
import numpy as np


def build_amass_npz(fullpose, trans, output_path, gender="male",
                    surface_model_type="smplh", fps=60.0, betas=None):
    """
    Save retargeted data as AMASS-compatible npz.

    Parameters
    ----------
    fullpose : np.ndarray [F, 156]
        Full SMPL-H pose vector (axis-angle).
    trans : np.ndarray [F, 3]
        Global translation per frame.
    output_path : str or Path
    gender : str
    surface_model_type : str
    fps : float
    betas : np.ndarray or None
        Shape parameters. If None, defaults to zeros(16).
    """
    n_frames = len(fullpose)

    if betas is None:
        betas = np.zeros(16, dtype=np.float64)

    # Split full pose into parts (following AMASS/MoSh++ convention)
    root_orient = fullpose[:, :3]      # [F, 3]
    pose_body = fullpose[:, 3:66]      # [F, 63]
    pose_hand = fullpose[:, 66:]       # [F, 90]

    data = {
        "gender": gender,
        "surface_model_type": surface_model_type,
        "mocap_frame_rate": float(fps),
        "mocap_time_length": float(n_frames / fps),

        "trans": trans.astype(np.float32),
        "poses": fullpose.astype(np.float32),
        "betas": betas.astype(np.float32),
        "num_betas": len(betas),

        "root_orient": root_orient.astype(np.float32),
        "pose_body": pose_body.astype(np.float32),
        "pose_hand": pose_hand.astype(np.float32),
    }

    np.savez(output_path, **data)
    print(f"Saved AMASS npz: {output_path}")
    print(f"  frames: {n_frames}, fps: {fps}, duration: {n_frames/fps:.2f}s")
    print(f"  poses: {fullpose.shape}, trans: {trans.shape}")
    return output_path
