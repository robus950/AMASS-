#!/usr/bin/env python
"""
Fast CMU to AMASS converter: bypasses chumpy's full-vertex LBS by computing
only the ~120 vertices actually needed for marker projection (~87x speedup).

Uses scipy.optimize.minimize directly with a pure-numpy forward pass.
"""
import sys, os, os.path as osp
import pickle
import time
from glob import glob
import numpy as np
from scipy.optimize import minimize
from collections import OrderedDict

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

sys.path.insert(0, '/home/user/retargeting/moshpp/src')
from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers
setup_mosh_omegaconf_resolvers()

# ── Load model and stage I data ──────────────────────────────────────────────

support_base_dir = '/home/user/retargeting/amass/support_data'
work_base_dir = '/home/user/retargeting/amass/work'
stagei_fname = osp.join(work_base_dir, 'mosh_results/subjects/87/male_stagei.pkl')
stagei_data = pickle.load(open(stagei_fname, 'rb'))

# Load SMPL-X model
with open(osp.join(support_base_dir, 'smplx/male/model.pkl'), 'rb') as f:
    model_data = pickle.load(f, encoding='latin-1')

v_template = model_data['v_template']           # [10475, 3]
shapedirs = model_data['shapedirs']              # [10475, 3, num_betas]
posedirs = model_data['posedirs']                # [10475, 3, num_pose_params]
weights = model_data['weights']                  # [10475, 55 or similar]
J_regressor = model_data['J_regressor']           # [55, 10475]
kintree_table = model_data['kintree_table']       # [2, 55]

# Get betas from stage I
betas = stagei_data['betas'][:16]
all_marker_labels = stagei_data['latent_labels']
marker_vids = stagei_data['markers_latent_vids']  # {label: vertex_id}
marker_positions = stagei_data['markers_latent']  # [n_markers, 3] — positions on template

n_joints = J_regressor.shape[0]
print(f'SMPL-X: {n_joints} joints, {v_template.shape[0]} vertices, {len(betas)} betas', flush=True)
print(f'Markers: {len(all_marker_labels)} latent markers', flush=True)

# ── Precompute: which vertices do we need? ───────────────────────────────────

# The marker locations on the canonical body are computed from the nearest
# neighbors of each marker. We need the vertices for those neighbors.
# From stage I, each marker has a primary vertex id (marker_vids).
# For accurate projection, we need a small neighborhood around each marker.

from sklearn.neighbors import NearestNeighbors

# Compute v_shaped (canonical body with betas)
v_shaped = v_template + np.tensordot(betas, shapedirs[:, :, :len(betas)], axes=([0], [2]))

# For each marker, find 8 nearest vertices on the canonical body
sknbrs = NearestNeighbors(algorithm='kd_tree', n_neighbors=8).fit(v_shaped)
_, closest = sknbrs.kneighbors(marker_positions)
closest = np.vstack(closest)  # [n_markers, 8]

# Unique vertex indices needed
needed_vids = np.unique(closest[:, :3])  # Only first 3 neighbors used for projection
print(f'Need {len(needed_vids)} unique vertices out of {v_template.shape[0]} ({100*len(needed_vids)/v_template.shape[0]:.1f}%)', flush=True)

# ── Fast LBS for specific vertices ───────────────────────────────────────────

def rodrigues_np(r):
    """Axis-angle [3] to rotation matrix [3,3]."""
    theta = np.linalg.norm(r)
    if theta < 1e-8:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K.dot(K)

def fast_lbs(pose_full, trans, v_template, shapedirs, betas, weights,
             J_regressor, kintree_table, vids):
    """
    Compute LBS vertex positions for specific vertex indices only.
    Returns [len(vids), 3] vertex positions.
    All inputs are numpy arrays.
    """
    n_v = len(vids)
    n_j = J_regressor.shape[0]

    # Shape blend for needed vertices only
    v_shaped_sel = v_template[vids] + np.tensordot(betas, shapedirs[vids, :, :len(betas)], axes=([0], [2]))  # [n_v, 3]

    # Joints from full v_template (joint regression needs full shape blend)
    v_shaped_all = v_template + np.tensordot(betas, shapedirs[:, :, :len(betas)], axes=([0], [2]))
    if J_regressor.shape[1] == v_shaped_all.shape[0]:
        J_rest = J_regressor.dot(v_shaped_all)  # [n_j, 3]
    else:
        J_rest = J_regressor

    # Rodrigues: axis-angle to rotation matrices for all joints
    rot_mats = np.zeros((n_j, 3, 3), dtype=np.float64)
    for i in range(n_j):
        rot_mats[i] = rodrigues_np(pose_full[3*i:3*i+3])

    # Forward kinematics
    parents = kintree_table[0].astype(np.int64)
    parents[parents > 1e9] = -1

    transforms = np.zeros((n_j, 4, 4), dtype=np.float64)
    for i in range(n_j):
        R = rot_mats[i]
        T_local = np.eye(4)
        T_local[:3, :3] = R
        T_local[:3, 3] = J_rest[i] - R.dot(J_rest[i])
        if parents[i] >= 0:
            transforms[i] = transforms[parents[i]].dot(T_local)
        else:
            transforms[i] = T_local

    # LBS for selected vertices only
    W_sel = weights[vids]  # [n_v, n_j]
    T_blend = np.einsum('vj,jab->vab', W_sel, transforms)  # [n_v, 4, 4]
    v_shaped_homo = np.hstack([v_shaped_sel, np.ones((n_v, 1))])
    v = np.einsum('vab,vb->va', T_blend[:, :3, :], v_shaped_homo) + trans

    return v


def compute_marker_positions(pose_full, trans, verts_sel, verts_sel_vids, closest, marker_coeffs):
    """
    Compute marker positions from posed vertices.
    verts_sel: [n_v, 3] vertex positions for the selected vertex indices
    verts_sel_vids: list of vertex indices corresponding to verts_sel
    closest: [n_markers, 8] nearest neighbor vertex indices
    marker_coeffs: [n_markers, 3] projection coefficients

    Returns: [n_markers, 3] marker positions
    """
    # Build map from global vid to position in verts_sel
    vid_to_idx = {vid: i for i, vid in enumerate(verts_sel_vids)}

    n_markers = closest.shape[0]
    result = np.zeros((n_markers, 3), dtype=np.float64)

    for m in range(n_markers):
        c0 = vid_to_idx[closest[m, 0]]
        c1 = vid_to_idx[closest[m, 1]]
        c2 = vid_to_idx[closest[m, 2]]

        e1 = verts_sel[c1] - verts_sel[c0]
        e2 = verts_sel[c2] - verts_sel[c0]

        nrm_e1 = np.linalg.norm(e1)
        f1 = e1 / max(nrm_e1, 1e-16)
        cross_e1_e2 = np.cross(e1, e2)
        nrm_cross = np.linalg.norm(cross_e1_e2)
        f2 = cross_e1_e2 / max(nrm_cross, 1e-16)
        f3 = np.cross(f1, f2)

        result[m] = (verts_sel[c0] +
                     marker_coeffs[m, 0] * f1 +
                     marker_coeffs[m, 1] * f2 +
                     marker_coeffs[m, 2] * f3)

    return result


# ── Precompute marker coefficients on canonical body ─────────────────────────

# Compute projection coefficients on canonical body (same as TransformedCoeffs)
# These are fixed during Stage II since betas don't change
canonical_verts = v_shaped[needed_vids]
canonical_verts_all = v_shaped

# Build global-to-sel mapping
vid_to_idx_canon = {vid: i for i, vid in enumerate(needed_vids)}

# Precompute marker projection coefficients
# On canonical body: diff = marker - v0; e1 = v1 - v0; e2 = v2 - v0
n_markers = len(all_marker_labels)
marker_coeffs = np.zeros((n_markers, 3), dtype=np.float64)
marker_closest = closest  # [n_markers, 8] — idx into global vertex numbering

for m in range(n_markers):
    marker_pos = marker_positions[m]
    c0 = vid_to_idx_canon[closest[m, 0]]
    c1 = vid_to_idx_canon[closest[m, 1]]
    c2 = vid_to_idx_canon[closest[m, 2]]

    diff = marker_pos - canonical_verts[c0]
    e1 = canonical_verts[c1] - canonical_verts[c0]
    e2 = canonical_verts[c2] - canonical_verts[c0]

    nrm_e1 = np.linalg.norm(e1)
    f1 = e1 / max(nrm_e1, 1e-16)
    cross_e1_e2 = np.cross(e1, e2)
    nrm_cross = np.linalg.norm(cross_e1_e2)
    f2 = cross_e1_e2 / max(nrm_cross, 1e-16)
    f3 = np.cross(f1, f2)

    marker_coeffs[m, 0] = np.dot(diff, f1)
    marker_coeffs[m, 1] = np.dot(diff, f2)
    marker_coeffs[m, 2] = np.dot(diff, f3)

print(f'Precomputed marker coefficients for {n_markers} markers', flush=True)


# ── MoCap files to convert ─────────────────────────────────────────────────

mocap_base = osp.join(work_base_dir, 'CMU/c3d/subjects/87')
if len(sys.argv) > 1:
    mocap_fnames = sys.argv[1:]
else:
    mocap_fnames = sorted(glob(osp.join(mocap_base, '*.c3d')))

print(f'\nConverting {len(mocap_fnames)} files:', flush=True)
for f in mocap_fnames:
    print(f'  {f}', flush=True)

for mocap_fname in mocap_fnames:
    basename = osp.splitext(osp.basename(mocap_fname))[0]  # e.g. "87_03"
    print(f'\n{"="*60}')
    print(f'Processing: {basename}')
    print(f'{"="*60}', flush=True)

    # ── MoCap data loading ──────────────────────────────────────────────────

    from moshpp.tools.mocap_interface import MocapSession

    mocap = MocapSession(
        mocap_fname, mocap_unit='mm',
        ignore_stared_labels=False,
        remove_label_before_colon=True,
        only_markers=all_marker_labels
    )
    print(f'Mocap: {len(mocap)} frames, {len(mocap.labels)} markers, {mocap.frame_rate} fps', flush=True)

    # ── Stage II per-frame optimization ──────────────────────────────────────

    MAXITER = 3
    WT_DATA_BASE = 400.0
    WT_POSE = 1.6
    WT_VELO = 2.5
    NUM_POSE = n_joints * 3  # 165
    NUM_TRAIN_MARKERS = len(all_marker_labels)

    pose_root_ids = [0, 1, 2]
    pose_body_ids = list(range(3, 66))
    pose_body_ids = [i for i in pose_body_ids if i not in range(30, 36)]
    opt_pose_ids = pose_root_ids + pose_body_ids

    print(f'Optimizing {len(opt_pose_ids)} pose params + 3 trans = {len(opt_pose_ids)+3} total', flush=True)
    print(f'Processing {len(mocap)} frames with maxiter={MAXITER}', flush=True)

    stageii_perframe = {
        'fullpose': [], 'trans': [], 'markers_sim': [],
        'markers_obs': [], 'labels_obs': [],
        'stageii_errs': {'data': [], 'velo': []},
    }

    pose_prev = None
    trans_prev = None
    state = {'pose_prev': None, 'x0_full_pose': None}
    total_t0 = time.time()

    observed_markers_dict = mocap.markers_asdict()
    selected_frames = range(0, len(mocap))

    for fIdx, t in enumerate(selected_frames):
        avail_labels = [l for l in all_marker_labels if l in observed_markers_dict[t]]
        avail_indices = [all_marker_labels.index(l) for l in avail_labels]
        markers_obs = np.array([observed_markers_dict[t][l] for l in avail_labels], dtype=np.float64)
        n_avail = len(markers_obs)

        if n_avail == 0:
            continue

        wt_data = WT_DATA_BASE * (NUM_TRAIN_MARKERS / n_avail)

        if pose_prev is not None:
            x0_full_pose = pose_prev.copy()
            x0_trans = trans_prev.copy()
        else:
            x0_full_pose = np.zeros(NUM_POSE, dtype=np.float64)
            x0_trans = np.mean(markers_obs, axis=0)

        x0_pose_subset = x0_full_pose[opt_pose_ids].copy()
        state['x0_full_pose'] = x0_full_pose
        state['pose_prev'] = pose_prev

        def make_objective(step_wt_pose):
            def objective(x):
                trans = x[:3]
                pose_subset = x[3:]
                full_pose = np.zeros(NUM_POSE, dtype=np.float64)
                full_pose[opt_pose_ids] = pose_subset

                verts = fast_lbs(full_pose, trans, v_template, shapedirs, betas,
                                weights, J_regressor, kintree_table, needed_vids)
                all_markers = compute_marker_positions(
                    full_pose, trans, verts, needed_vids, marker_closest, marker_coeffs)

                markers_sim = all_markers[avail_indices]
                data_err = np.sum((markers_sim - markers_obs)**2) * wt_data**2

                body_pose = pose_subset[3:]
                pose_err = np.sum(body_pose**2) * step_wt_pose**2

                velo_err = 0.0
                if state['pose_prev'] is not None:
                    x0_full = state['x0_full_pose']
                    velo = full_pose - (x0_full + (x0_full - state['pose_prev']))
                    velo_err = np.sum(velo**2) * WT_VELO**2

                return data_err + pose_err + velo_err
            return objective

        x0 = np.concatenate([x0_trans, x0_pose_subset])

        t0 = time.time()
        res1 = minimize(make_objective(WT_POSE * 10.0), x0, method='L-BFGS-B',
                        options={'maxiter': MAXITER, 'disp': False})
        res2 = minimize(make_objective(WT_POSE), res1.x, method='L-BFGS-B',
                        options={'maxiter': MAXITER, 'disp': False})
        elapsed = time.time() - t0

        x_opt = res2.x
        trans_opt = x_opt[:3]
        full_pose_opt = np.zeros(NUM_POSE, dtype=np.float64)
        full_pose_opt[opt_pose_ids] = x_opt[3:]

        stageii_perframe['fullpose'].append(full_pose_opt.copy())
        stageii_perframe['trans'].append(trans_opt.copy())

        verts_final = fast_lbs(full_pose_opt, trans_opt, v_template, shapedirs, betas,
                               weights, J_regressor, kintree_table, needed_vids)
        all_markers_final = compute_marker_positions(
            full_pose_opt, trans_opt, verts_final, needed_vids, marker_closest, marker_coeffs)
        markers_sim_final = all_markers_final[avail_indices]
        data_err = np.sum((markers_sim_final - markers_obs)**2)

        stageii_perframe['markers_sim'].append(markers_sim_final)
        stageii_perframe['markers_obs'].append(markers_obs.copy())
        stageii_perframe['labels_obs'].append(avail_labels)
        stageii_perframe['stageii_errs']['data'].append(data_err)

        pose_prev = full_pose_opt
        trans_prev = trans_opt

        if fIdx % 10 == 0 or fIdx == len(selected_frames) - 1:
            elapsed_total = time.time() - total_t0
            fps = (fIdx + 1) / elapsed_total if elapsed_total > 0 else 0
            eta = (len(selected_frames) - fIdx - 1) / fps if fps > 0 else 0
            print(f'  [{fIdx:04d}/{len(selected_frames)}] data_err={data_err:.2e} | '
                  f'{elapsed:.1f}s/frame | ETA: {eta/60:.1f}min', flush=True)

    total_elapsed = time.time() - total_t0
    print(f'Stage II done in {total_elapsed/60:.1f}min ({total_elapsed/len(selected_frames):.1f}s/frame)', flush=True)

    # ── Save as AMASS npz ───────────────────────────────────────────────────

    poses = np.array(stageii_perframe['fullpose'])
    trans = np.array(stageii_perframe['trans'])

    amass_data = {
        'gender': 'male',
        'surface_model_type': 'smplx',
        'mocap_frame_rate': mocap.frame_rate,
        'mocap_time_length': len(mocap) / mocap.frame_rate,
        'markers_latent': marker_positions,
        'latent_labels': all_marker_labels,
        'markers_latent_vids': marker_vids,
        'trans': trans,
        'poses': poses,
        'betas': betas,
        'num_betas': 16,
        'root_orient': poses[:, :3],
        'pose_body': poses[:, 3:66],
        'pose_hand': poses[:, 66:],
        'pose_jaw': poses[:, 66:69],
        'pose_eye': poses[:, 69:75],
    }

    out_npz = osp.join(work_base_dir, f'mosh_results/subjects/87/{basename}_poses.npz')
    np.savez(out_npz, **amass_data)
    print(f'Saved: {out_npz}', flush=True)

print('\nAll done!', flush=True)
