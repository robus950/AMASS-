#!/usr/bin/env python
"""
GPU-accelerated CMU to AMASS converter using PyTorch + smplx.
All frames batched and optimized in parallel on GPU (~100x faster than CPU numpy).
"""
import sys, os, os.path as osp
import pickle, time
from glob import glob
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

device = torch.device('cuda')
print(f'Device: {device} ({torch.cuda.get_device_name(0)})', flush=True)

# ── Paths ───────────────────────────────────────────────────────────────────

support_base_dir = '/home/user/retargeting/amass/support_data'
work_base_dir = '/home/user/retargeting/amass/work'
smplx_dir = osp.join(support_base_dir, 'smplx')
mocap_base = osp.join(work_base_dir, 'CMU/c3d/subjects/87')
stagei_fname = osp.join(work_base_dir, 'mosh_results/subjects/87/male_stagei.pkl')

# ── Load stage I & model ─────────────────────────────────────────────────────

stagei_data = pickle.load(open(stagei_fname, 'rb'))
with open(osp.join(smplx_dir, 'male/model.pkl'), 'rb') as f:
    model_data = pickle.load(f, encoding='latin-1')

betas_np = stagei_data['betas'][:16]
all_marker_labels = stagei_data['latent_labels']
marker_positions = stagei_data['markers_latent']  # [n_markers, 3] — template positions
marker_vids = stagei_data['markers_latent_vids']
n_markers = len(all_marker_labels)

# Compute canonical v_shaped for nearest-neighbor lookup
v_template = model_data['v_template']
shapedirs = model_data['shapedirs']
v_shaped = v_template + np.tensordot(betas_np, shapedirs[:, :, :len(betas_np)], axes=([0], [2]))

# Find nearest neighbor vertices for each marker (8 neighbors, use first 3)
sknbrs = NearestNeighbors(algorithm='kd_tree', n_neighbors=8).fit(v_shaped)
_, closest = sknbrs.kneighbors(marker_positions)
closest = np.vstack(closest)  # [n_markers, 8]
needed_vids = np.unique(closest[:, :3])
print(f'Need {len(needed_vids)} unique vertices out of {v_template.shape[0]}', flush=True)

# Precompute marker projection coefficients on canonical body
marker_coeffs = np.zeros((n_markers, 3), dtype=np.float64)
for m in range(n_markers):
    c0, c1, c2 = closest[m, 0], closest[m, 1], closest[m, 2]
    diff = marker_positions[m] - v_shaped[c0]
    e1 = v_shaped[c1] - v_shaped[c0]
    e2 = v_shaped[c2] - v_shaped[c0]
    nrm_e1 = np.linalg.norm(e1)
    f1 = e1 / max(nrm_e1, 1e-16)
    c = np.cross(e1, e2)
    nrm_c = np.linalg.norm(c)
    f2 = c / max(nrm_c, 1e-16)
    f3 = np.cross(f1, f2)
    marker_coeffs[m] = [np.dot(diff, f1), np.dot(diff, f2), np.dot(diff, f3)]

# Convert to torch tensors (on GPU)
vid_map = {int(v): i for i, v in enumerate(needed_vids)}
c0_idx = torch.tensor([vid_map[int(closest[m, 0])] for m in range(n_markers)], device=device, dtype=torch.long)
c1_idx = torch.tensor([vid_map[int(closest[m, 1])] for m in range(n_markers)], device=device, dtype=torch.long)
c2_idx = torch.tensor([vid_map[int(closest[m, 2])] for m in range(n_markers)], device=device, dtype=torch.long)
mcoeffs = torch.tensor(marker_coeffs, device=device, dtype=torch.float32)  # [n_markers, 3]
needed_vids_t = torch.tensor(needed_vids, device=device, dtype=torch.long)
betas_t = torch.tensor(betas_np, device=device, dtype=torch.float32).unsqueeze(0)  # [1, 16]

print(f'Precomputed marker coefficients, GPU tensors ready', flush=True)

# ── SMPL-X model ────────────────────────────────────────────────────────────

import smplx
smplx_model = smplx.SMPLX(
    smplx_dir, gender='male', num_betas=16,
    use_face=False, use_pca=False, flat_hand_mean=False,
    create_body_pose=True,
    create_global_orient=True,
    create_transl=True,
    create_jaw_pose=False,
    create_leye_pose=False,
    create_reye_pose=False,
    create_left_hand_pose=False,
    create_right_hand_pose=False,
    create_expression=False,
    create_betas=True,
).to(device)
# smplx model has no batchnorm/dropout — eval or train mode both fine

# ── Marker projection function (PyTorch) ────────────────────────────────────

def compute_markers_torch(verts_sel, c0, c1, c2, coeffs):
    """verts_sel: [B, n_v, 3], returns: [B, n_markers, 3]"""
    v0 = verts_sel[:, c0, :]  # [B, n_markers, 3]
    v1 = verts_sel[:, c1, :]
    v2 = verts_sel[:, c2, :]

    e1 = v1 - v0
    e2 = v2 - v0
    nrm_e1 = torch.linalg.norm(e1, dim=-1, keepdim=True).clamp(min=1e-16)
    f1 = e1 / nrm_e1
    cross_e1_e2 = torch.cross(e1, e2, dim=-1)
    nrm_cross = torch.linalg.norm(cross_e1_e2, dim=-1, keepdim=True).clamp(min=1e-16)
    f2 = cross_e1_e2 / nrm_cross
    f3 = torch.cross(f1, f2, dim=-1)

    return v0 + coeffs[None, :, 0:1] * f1 + coeffs[None, :, 1:2] * f2 + coeffs[None, :, 2:3] * f3

# ── Load mocap data ────────────────────────────────────────────────────────

from moshpp.tools.mocap_interface import MocapSession

def load_mocap(mocap_fname):
    mocap = MocapSession(mocap_fname, mocap_unit='mm',
                         ignore_stared_labels=False,
                         remove_label_before_colon=True,
                         only_markers=None)
    observed_dict = mocap.markers_asdict()

    n_frames = len(mocap)
    # Build per-frame marker observations and masks
    markers_obs_list = []
    mask_list = []
    for t in range(n_frames):
        frame_markers = np.zeros((n_markers, 3), dtype=np.float32)
        frame_mask = np.zeros(n_markers, dtype=np.float32)
        for m_idx, label in enumerate(all_marker_labels):
            if label in observed_dict[t]:
                frame_markers[m_idx] = observed_dict[t][label]
                frame_mask[m_idx] = 1.0
        markers_obs_list.append(frame_markers)
        mask_list.append(frame_mask)

    markers_obs_t = torch.tensor(np.array(markers_obs_list), device=device, dtype=torch.float32)
    mask_t = torch.tensor(np.array(mask_list), device=device, dtype=torch.float32)
    n_avail = mask_t.sum(dim=1)  # [n_frames]

    return markers_obs_t, mask_t, n_avail, mocap.frame_rate, n_frames

# ── Optimization ────────────────────────────────────────────────────────────

# Body joints without toes: joints 10,11 are toe joints
# SMPL-X body: 21 joints, indexed 0-20. Toes at 10,11.
# body_pose shape: [21, 3] → 63 params
# Optimized: [19, 3] → 57 params (exclude joints 10,11)
BODY_JOINTS = 21
TOE_JOINT_IDS = [10, 11]
OPT_BODY_JOINTS = [j for j in range(BODY_JOINTS) if j not in TOE_JOINT_IDS]  # 19 joints

WT_DATA_BASE = 400.0
WT_POSE = 1.6
WT_SMOOTH = 2.5  # acceleration smoothness (replaces velocity term)
LR = 0.05
MAX_ITER = 300

def optimize_mocap(markers_obs_t, mask_t, n_avail_t, n_frames, basename):
    print(f'Optimizing {n_frames} frames on GPU...', flush=True)
    n_train = mask_t.shape[1]

    wt_data = WT_DATA_BASE * (n_train / n_avail_t.clamp(min=1))  # [n_frames]

    # Init params
    global_orient = torch.zeros(n_frames, 3, device=device, dtype=torch.float32, requires_grad=True)
    body_pose_opt = torch.zeros(n_frames, len(OPT_BODY_JOINTS) * 3, device=device, dtype=torch.float32, requires_grad=True)
    transl = markers_obs_t.mean(dim=1).clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([global_orient, body_pose_opt, transl], lr=LR)

    def build_full_body_pose():
        full = torch.zeros(n_frames, BODY_JOINTS, 3, device=device, dtype=torch.float32)
        opt = body_pose_opt.view(n_frames, len(OPT_BODY_JOINTS), 3)
        for i, j_id in enumerate(OPT_BODY_JOINTS):
            full[:, j_id, :] = opt[:, i, :]
        return full.view(n_frames, BODY_JOINTS * 3)

    # Zero tensors for non-optimized pose components
    zeros_B3 = torch.zeros(n_frames, 3, device=device)
    zeros_B45 = torch.zeros(n_frames, 45, device=device)
    zeros_B10 = torch.zeros(n_frames, 10, device=device)

    def forward():
        body_pose_full = build_full_body_pose()
        smpl_out = smplx_model(
            betas=betas_t.expand(n_frames, -1),
            body_pose=body_pose_full,
            global_orient=global_orient,
            transl=transl,
            jaw_pose=zeros_B3,
            leye_pose=zeros_B3,
            reye_pose=zeros_B3,
            left_hand_pose=zeros_B45,
            right_hand_pose=zeros_B45,
            expression=zeros_B10,
            return_verts=True,
        )
        all_verts = smpl_out.vertices
        verts_sel = all_verts[:, needed_vids_t, :]
        markers_sim = compute_markers_torch(verts_sel, c0_idx, c1_idx, c2_idx, mcoeffs)

        diff = (markers_sim - markers_obs_t) * mask_t.unsqueeze(-1)
        data_loss = ((diff ** 2).sum(dim=(-1, -2)) * wt_data**2).mean()

        pose_loss = (body_pose_opt ** 2).mean() * WT_POSE**2

        smooth_loss = torch.tensor(0.0, device=device)
        if n_frames > 2:
            accel = body_pose_opt[2:] - 2 * body_pose_opt[1:-1] + body_pose_opt[:-2]
            smooth_loss = (accel ** 2).mean() * WT_SMOOTH**2
            accel_t = transl[2:] - 2 * transl[1:-1] + transl[:-2]
            smooth_loss = smooth_loss + (accel_t ** 2).mean() * WT_SMOOTH**2

        return data_loss + pose_loss + smooth_loss, data_loss, pose_loss, smooth_loss

    t0 = time.time()
    for it in range(MAX_ITER):
        optimizer.zero_grad()
        total, data_l, pose_l, smooth_l = forward()
        total.backward()
        optimizer.step()

        if it % 20 == 0 or it == MAX_ITER - 1:
            elapsed = time.time() - t0
            print(f'  iter {it:4d}/{MAX_ITER} | total={total.item():.2e} | '
                  f'data={data_l.item():.2e} | pose={pose_l.item():.2e} | '
                  f'smooth={smooth_l.item():.2e} | {elapsed:.1f}s', flush=True)

    total_time = time.time() - t0
    print(f'GPU optimization done in {total_time:.1f}s ({total_time/n_frames*1000:.0f}ms/frame)', flush=True)

    # ── Extract results ──────────────────────────────────────────────────

    with torch.no_grad():
        final_body = build_full_body_pose()
        smpl_out = smplx_model(
            betas=betas_t.expand(n_frames, -1),
            body_pose=final_body,
            global_orient=global_orient,
            transl=transl,
            jaw_pose=zeros_B3,
            leye_pose=zeros_B3,
            reye_pose=zeros_B3,
            left_hand_pose=zeros_B45,
            right_hand_pose=zeros_B45,
            expression=zeros_B10,
            return_verts=True,
        )
        final_markers = compute_markers_torch(
            smpl_out.vertices[:, needed_vids_t, :],
            c0_idx, c1_idx, c2_idx, mcoeffs)

        # RMS error
        diff = (final_markers - markers_obs_t) * mask_t.unsqueeze(-1)
        raw_sse = (diff ** 2).sum(dim=(-1, -2))
        n_per_frame = mask_t.sum(dim=1) * 3
        rms = torch.sqrt(raw_sse.sum() / n_per_frame.sum())
        print(f'Final RMS marker error: {rms.item()*100:.2f} cm', flush=True)

    # Build full 165-DOF pose
    opt_pose_np = torch.cat([global_orient.detach(), final_body.detach()], dim=1).cpu().numpy()
    full_pose = np.zeros((n_frames, 165), dtype=np.float32)
    full_pose[:, :3] = opt_pose_np[:, :3]  # root
    full_pose[:, 3:66] = opt_pose_np[:, 3:]  # body
    transl_np = transl.detach().cpu().numpy()

    return full_pose, transl_np

# ── Main ─────────────────────────────────────────────────────────────────────

mocap_fnames = sorted(glob(osp.join(mocap_base, '*.c3d')))
if len(sys.argv) > 1:
    mocap_fnames = sys.argv[1:]

print(f'\nConverting {len(mocap_fnames)} files:', flush=True)
for f in mocap_fnames:
    print(f'  {f}', flush=True)

for mocap_fname in mocap_fnames:
    basename = osp.splitext(osp.basename(mocap_fname))[0]
    print(f'\n{"="*60}\nProcessing: {basename}\n{"="*60}', flush=True)

    markers_obs_t, mask_t, n_avail, frame_rate, n_frames = load_mocap(mocap_fname)
    print(f'Loaded: {n_frames} frames, {int(mask_t.sum().item()/n_frames)} avg markers/frame', flush=True)

    poses, trans = optimize_mocap(markers_obs_t, mask_t, n_avail, n_frames, basename)

    # Save
    amass_data = {
        'gender': 'male',
        'surface_model_type': 'smplx',
        'mocap_frame_rate': frame_rate,
        'mocap_time_length': n_frames / frame_rate,
        'markers_latent': marker_positions,
        'latent_labels': all_marker_labels,
        'markers_latent_vids': marker_vids,
        'trans': trans,
        'poses': poses,
        'betas': betas_np,
        'num_betas': 16,
        'root_orient': poses[:, :3],
        'pose_body': poses[:, 3:66],
        'pose_hand': poses[:, 66:],
        'pose_jaw': poses[:, 66:69],
        'pose_eye': poses[:, 69:75],
    }
    out_npz = osp.join(work_base_dir, f'mosh_results/subjects/87/{basename}_poses_gpu.npz')
    np.savez(out_npz, **amass_data)
    print(f'Saved: {out_npz}', flush=True)

print('\nAll done!', flush=True)
