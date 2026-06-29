"""Step 2: Left-handed → Right-handed coordinate system conversion.

Nokov exports data in Unity's left-handed coordinate system:
  Unity LH: X=right, Y=up, Z=forward
  SMPL RH:  X=right, Y=up, Z=forward

Conversion (same axis directions, different handedness):
  - Quaternion: q_rh = conjugate(q_lh) = [-x, -y, -z, w]
  - Position:   unchanged (axes point the same way)
"""

import numpy as np


def quat_lh_to_rh(quats_xyzw):
    """
    Convert quaternions from Unity left-handed to standard right-handed.

    A rotation in LH is the inverse of the same rotation in RH, so we
    conjugate the quaternion (negate all vector components, keep w).

    Parameters
    ----------
    quats_xyzw : np.ndarray [..., 4]
        Quaternions in xyzw order, Unity left-handed frame.

    Returns
    -------
    quats_rh : np.ndarray [..., 4]
        Quaternions in xyzw order, standard right-handed frame.
    """
    q = np.asarray(quats_xyzw, dtype=np.float64)
    result = q.copy()
    result[..., 0] = -q[..., 0]  # x → -x
    result[..., 1] = -q[..., 1]  # y → -y
    result[..., 2] = -q[..., 2]  # z → -z
    # w unchanged
    return result


def pos_lh_to_rh(positions):
    """
    Convert positions from Unity left-handed to standard right-handed.

    Since both systems have X=right, Y=up, Z=forward, positions are unchanged.

    Parameters
    ----------
    positions : np.ndarray [..., 3]

    Returns
    -------
    pos_rh : np.ndarray [..., 3]
    """
    return np.asarray(positions, dtype=np.float64).copy()


def quat_normalize(q_xyzw):
    """Normalize quaternions to unit length, in-place."""
    norms = np.linalg.norm(q_xyzw, axis=-1, keepdims=True)
    mask = norms > 1e-12
    with np.errstate(divide='ignore', invalid='ignore'):
        q_xyzw = np.where(mask, q_xyzw / norms, q_xyzw)
    return q_xyzw
