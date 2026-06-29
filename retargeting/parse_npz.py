"""Step 1: Parse 0416.npz — extract 24 joint quaternions and root translation."""

import numpy as np
from .skeleton_def import RAW_ORDER_24, NPZ_IDX


def parse_npz(npz_path):
    """
    Parse a Nokov/XBR NPZ file.

    Parameters
    ----------
    npz_path : str or Path

    Returns
    -------
    quats : np.ndarray [F, 24, 4] float64
        Quaternion rotations (xyzw) for each of the 24 joints in RAW_ORDER_24.
    root_trans : np.ndarray [F, 3] float64
        Root translation (always zero for in-place capture).
    fps : float
        Estimated frame rate (default 60 if not found).
    """
    data = np.load(npz_path, allow_pickle=True)
    raw = data["raw_joint_data"].astype(np.float64)  # [F, 99]

    n_frames = raw.shape[0]
    quats = raw[:, :96].reshape(n_frames, 24, 4).copy()  # [F, 24, 4] xyzw
    root_trans = raw[:, 96:].copy()                      # [F, 3]

    fps = float(data.get("fps", data.get("mocap_frame_rate", 60.0)))

    return quats, root_trans, fps


def get_joint_quat(quats, joint_name):
    """Extract quaternion for a single named joint from the [F, 24, 4] array."""
    idx = NPZ_IDX[joint_name]
    return quats[:, idx, :]  # [F, 4]
