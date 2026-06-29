"""Skeleton definitions for 0416.npz (Nokov/XBR) mocap data."""

import numpy as np

# ── NPZ raw column layout ──────────────────────────────────────────────────
# raw_joint_data shape: [F, 99]
#   first 96 = 24 joints × 4-component quaternion (xyzw)
#   last 3   = root translation (always zero for in-place capture)

RAW_ORDER_24 = [
    "Shoulder_L",   # 0
    "Shoulder_R",   # 1
    "Elbow_L",      # 2
    "Elbow_R",      # 3
    "Hand_L",       # 4
    "Hand_R",       # 5
    "Hip_L",        # 6
    "Hip_R",        # 7
    "Knee_R",       # 8
    "Knee_L",       # 9
    "Foot_R",       # 10
    "Foot_L",       # 11
    "Head",         # 12
    "Abdomen",      # 13
    "Thumb2_L",     # 14
    "Thumb2_R",     # 15
    "Index1_L",     # 16
    "Index1_R",     # 17
    "Middle1_L",    # 18
    "Middle1_R",    # 19
    "Ring1_L",      # 20
    "Ring1_R",      # 21
    "Pinky1_L",     # 22
    "Pinky1_R",     # 23
]

NPZ_IDX = {name: i for i, name in enumerate(RAW_ORDER_24)}

# ── XBR T-Pose skeleton positions from XBR_links.yml (cm, relative to hc_Abdomen) ─
# Coordinate system: X=right, Y=up, Z=forward

HC_REST_POSE = {
    "hc_Abdomen":    np.array([0.0,      0.0,      0.0],     dtype=np.float64),
    "Spine":         np.array([0.0,     -0.567,    9.489],   dtype=np.float64),
    "hc_Chest":      np.array([0.0,      0.783,   28.696],   dtype=np.float64),
    "neck":          np.array([0.0,      5.142,   58.355],   dtype=np.float64),
    "hc_Head":       np.array([0.0,      3.245,   67.451],   dtype=np.float64),
    "hc_Shoulder_L": np.array([18.023,   9.015,   51.084],   dtype=np.float64),
    "hc_Shoulder_R": np.array([-18.023,  9.015,   51.084],   dtype=np.float64),
    "hc_Elbow_L":    np.array([47.860,   9.064,   50.570],   dtype=np.float64),
    "hc_Elbow_R":    np.array([-47.708, 11.957,   50.273],   dtype=np.float64),
    "hc_Hand_L":     np.array([74.807,  10.025,   49.768],   dtype=np.float64),
    "hc_Hand_R":     np.array([-74.657, 11.386,   51.324],   dtype=np.float64),
    "hc_Hip_L":      np.array([9.006,   -0.202,   -2.768],   dtype=np.float64),
    "hc_Hip_R":      np.array([-9.006,  -0.202,   -2.768],   dtype=np.float64),
    "hc_Knee_L":     np.array([9.098,    1.080,  -45.322],   dtype=np.float64),
    "hc_Knee_R":     np.array([-9.973,   1.079,  -45.312],   dtype=np.float64),
    "hc_Foot_L":     np.array([8.933,    7.332,  -85.031],   dtype=np.float64),
    "hc_Foot_R":     np.array([-10.895,  7.355,  -85.006],   dtype=np.float64),
}

# Parent-child edges: child → parent
FK_EDGES = {
    "Spine":         "hc_Abdomen",
    "hc_Chest":      "Spine",
    "neck":          "hc_Chest",
    "hc_Head":       "neck",
    "hc_Shoulder_L": "hc_Chest",
    "hc_Shoulder_R": "hc_Chest",
    "hc_Elbow_L":    "hc_Shoulder_L",
    "hc_Elbow_R":    "hc_Shoulder_R",
    "hc_Hand_L":     "hc_Elbow_L",
    "hc_Hand_R":     "hc_Elbow_R",
    "hc_Hip_L":      "hc_Abdomen",
    "hc_Hip_R":      "hc_Abdomen",
    "hc_Knee_L":     "hc_Hip_L",
    "hc_Knee_R":     "hc_Hip_R",
    "hc_Foot_L":     "hc_Knee_L",
    "hc_Foot_R":     "hc_Knee_R",
}

# FK joint list in traversal order (root first)
FK_JOINTS_ORDERED = [
    "hc_Abdomen",
    "Spine",
    "hc_Chest",
    "neck",
    "hc_Head",
    "hc_Shoulder_L", "hc_Shoulder_R",
    "hc_Elbow_L", "hc_Elbow_R",
    "hc_Hand_L", "hc_Hand_R",
    "hc_Hip_L", "hc_Hip_R",
    "hc_Knee_L", "hc_Knee_R",
    "hc_Foot_L", "hc_Foot_R",
]

# NPZ joint name → FK joint name (only joints that have NPZ quaternion data)
SRC_TO_HC = {
    "Abdomen":     "hc_Abdomen",
    "Shoulder_L":  "hc_Shoulder_L",
    "Shoulder_R":  "hc_Shoulder_R",
    "Elbow_L":     "hc_Elbow_L",
    "Elbow_R":     "hc_Elbow_R",
    "Hand_L":      "hc_Hand_L",
    "Hand_R":      "hc_Hand_R",
    "Hip_L":       "hc_Hip_L",
    "Hip_R":       "hc_Hip_R",
    "Knee_L":      "hc_Knee_L",
    "Knee_R":      "hc_Knee_R",
    "Foot_L":      "hc_Foot_L",
    "Foot_R":      "hc_Foot_R",
}

# ── SMPL-H rest pose joint positions (meters, betas=0, from SMPL-X neutral model) ──

SMPL_REST_POSE = np.array([
    [ 0.003123, -0.351407,  0.012037],  # 0:  Pelvis
    [ 0.061313, -0.444171, -0.013965],  # 1:  L_Hip
    [-0.060144, -0.455316, -0.009214],  # 2:  R_Hip
    [ 0.000361, -0.241517, -0.015581],  # 3:  Spine1
    [ 0.116008, -0.822924, -0.023361],  # 4:  L_Knee
    [-0.104354, -0.817696, -0.026038],  # 5:  R_Knee
    [ 0.009808, -0.109664, -0.021521],  # 6:  Spine2
    [ 0.072555, -1.225984, -0.055237],  # 7:  L_Ankle
    [-0.088937, -1.228423, -0.046230],  # 8:  R_Ankle
    [-0.001522, -0.057428,  0.006926],  # 9:  Spine3
    [ 0.119812, -1.283981,  0.062980],  # 10: L_Foot
    [-0.127750, -1.286752,  0.072819],  # 11: R_Foot
    [-0.013687,  0.107739, -0.024690],  # 12: Neck
    [ 0.044842,  0.027515, -0.000295],  # 13: L_Collar
    [-0.049217,  0.026910, -0.006474],  # 14: R_Collar
    [ 0.011097,  0.268190, -0.003952],  # 15: Head
    [ 0.164081,  0.085243, -0.015756],  # 16: L_Shoulder
    [-0.151795,  0.080435, -0.019143],  # 17: R_Shoulder
    [ 0.418204,  0.013093, -0.058214],  # 18: L_Elbow
    [-0.422944,  0.043942, -0.045610],  # 19: R_Elbow
    [ 0.670191,  0.036314, -0.060687],  # 20: L_Wrist
    [-0.672212,  0.039410, -0.060935],  # 21: R_Wrist
], dtype=np.float64)

SMPLH_BODY_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]

# Map from FK joint name to SMPL-H body joint index
JOINT_MAP_FK_TO_SMPLH = {
    "hc_Abdomen":    0,   # Pelvis
    "hc_Chest":      9,   # Spine3
    "hc_Hip_L":      1,   # L_Hip
    "hc_Hip_R":      2,   # R_Hip
    "hc_Knee_L":     4,   # L_Knee
    "hc_Knee_R":     5,   # R_Knee
    "hc_Foot_L":     7,   # L_Ankle
    "hc_Foot_R":     8,   # R_Ankle
    "hc_Shoulder_L": 16,  # L_Shoulder
    "hc_Shoulder_R": 17,  # R_Shoulder
    "hc_Elbow_L":    18,  # L_Elbow
    "hc_Elbow_R":    19,  # R_Elbow
    "hc_Hand_L":     20,  # L_Wrist
    "hc_Hand_R":     21,  # R_Wrist
}

# XBR bone (parent FK joint, child FK joint) → SMPL bone (parent body idx, child body idx)
# Used for bone-direction alignment: each SMPL bone gets its direction from an XBR bone.
BONE_MAP_XBR_TO_SMPLH = {
    # SMPL bone (p_idx, c_idx) → XBR bone (parent_name, child_name)
    (0, 1):   ("hc_Abdomen",    "hc_Hip_L"),       # Pelvis → L_Hip
    (0, 2):   ("hc_Abdomen",    "hc_Hip_R"),       # Pelvis → R_Hip
    (0, 3):   ("hc_Abdomen",    "Spine"),           # Pelvis → Spine1
    (1, 4):   ("hc_Hip_L",      "hc_Knee_L"),       # L_Hip → L_Knee
    (2, 5):   ("hc_Hip_R",      "hc_Knee_R"),       # R_Hip → R_Knee
    (3, 6):   ("Spine",         "hc_Chest"),         # Spine1 → Spine2
    (4, 7):   ("hc_Knee_L",     "hc_Foot_L"),        # L_Knee → L_Ankle
    (5, 8):   ("hc_Knee_R",     "hc_Foot_R"),        # R_Knee → R_Ankle
    (9, 12):  ("hc_Chest",      "neck"),             # Spine3 → Neck
    (9, 13):  ("hc_Chest",      "hc_Shoulder_L"),    # Spine3 → L_Collar (approx)
    (9, 14):  ("hc_Chest",      "hc_Shoulder_R"),    # Spine3 → R_Collar (approx)
    (12, 15): ("neck",          "hc_Head"),           # Neck → Head
    (13, 16): ("hc_Chest",      "hc_Shoulder_L"),    # L_Collar → L_Shoulder
    (14, 17): ("hc_Chest",      "hc_Shoulder_R"),    # R_Collar → R_Shoulder
    (16, 18): ("hc_Shoulder_L", "hc_Elbow_L"),       # L_Shoulder → L_Elbow
    (17, 19): ("hc_Shoulder_R", "hc_Elbow_R"),       # R_Shoulder → R_Elbow
    (18, 20): ("hc_Elbow_L",    "hc_Hand_L"),        # L_Elbow → L_Wrist
    (19, 21): ("hc_Elbow_R",    "hc_Hand_R"),        # R_Elbow → R_Wrist
}

# SMPL-H joints that have no FK counterpart → set to zero rotation
SMPLH_UNMAPPED = sorted(set(range(22)) - set(JOINT_MAP_FK_TO_SMPLH.values()))

NUM_SMPLH_JOINTS = 52
NUM_SMPLH_BODY_JOINTS = 22
NUM_FK_JOINTS = len(FK_JOINTS_ORDERED)  # 17
