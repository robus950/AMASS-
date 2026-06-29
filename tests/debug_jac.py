#!/usr/bin/env python
"""Debug script to find where Jacobian computation hangs."""
import sys
import time
import os.path as osp

sys.path.insert(0, '/home/user/retargeting/moshpp/src')
from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers
setup_mosh_omegaconf_resolvers()

from moshpp.mosh_head import MoSh
import chumpy as ch
import numpy as np
import scipy.sparse as sp

# Find the c3d file
from glob import glob
support_base_dir = '/home/user/retargeting/amass/support_data'
work_base_dir = '/home/user/retargeting/amass/work'
mocap_base_dir = osp.join(work_base_dir, 'CMU/c3d/subjects')
mocap_fnames = glob(osp.join(mocap_base_dir, '87', '87_03.c3d'))
mocap_fname = mocap_fnames[0]
print(f'Testing with: {mocap_fname}')

job = {
    'mocap.fname': mocap_fname,
    'dirs.support_base_dir': support_base_dir,
    'dirs.work_base_dir': osp.join(work_base_dir, 'mosh_results'),
    'surface_model.type': 'smplx',
    'surface_model.gender': 'male',
    'moshpp.pose_body_prior_fname': None,
    'moshpp.head_marker_corr_fname': None,
    'moshpp.optimize_fingers': False,
    'moshpp.optimize_face': False,
    'moshpp.optimize_betas': True,
    'moshpp.optimize_dynamics': False,
    'runtime.stagei_only': False,
    'moshpp.verbosity': 1,
    'opt_settings.maxiter': 2,
}

cfg = MoSh.prepare_cfg(**job)
print(f'Stage II file: {cfg.dirs.stageii_fname}')

# Load mocap data
from moshpp.tools.cfg_helper import get_mocap_markers
marker_meta, labels_obs, markers_obs = get_mocap_markers(cfg)
print(f'Loaded {len(labels_obs)} frames, {len(markers_obs[0])} markers')

# Set up surface model
from moshpp.tools.cfg_helper import load_surface_model
model_type = cfg.surface_model.type
gender = cfg.surface_model.gender
num_betas = cfg.surface_model.num_betas
v_template_fname = getattr(cfg.surface_model, 'v_template_fname', None)
pose_hand_prior_fname = cfg.moshpp.pose_hand_prior_fname
surface_model_fname = osp.join(cfg.dirs.support_base_dir, cfg.surface_model.surface_model_fname)

can_model = load_surface_model(
    surface_model_fname, model_type=model_type, gender=gender,
    num_betas=num_betas, v_template_fname=v_template_fname,
    pose_hand_prior_fname=pose_hand_prior_fname)
print(f'Loaded surface model: {can_model.model_type}')

# Set up marker correlation from the model
from moshpp.tools.cfg_helper import get_markers_from_model
can_mrk_ids, can_meta, head_mrk_ids, can_mrk_correspondences = get_markers_from_model(cfg, can_model)
print(f'Marker IDs: {can_mrk_ids}')

# Build latent markers
from moshpp.tools.cfg_helper import get_marker_corr
marker_corr, head_mrk_corr = get_marker_corr(cfg, labels_obs)
print(f'Marker correlation type: {type(marker_corr)}')

# Get the non-NaN marker indices for the first frame
first_frame_markers = markers_obs[0]
valid_mask = ~np.any(np.isnan(first_frame_markers), axis=1)
valid_labels = labels_obs[valid_mask]
valid_markers = first_frame_markers[valid_mask]
print(f'Valid markers in frame 0: {len(valid_markers)}')

# Build latent markers as chumpy variables
markers_latent = ch.array(valid_markers.copy())
latent_labels = valid_labels

# Build the data objective
from moshpp.transformed_lm import TransformedCoeffs
marker_layout_fname = osp.join(cfg.dirs.support_base_dir, cfg.moshpp.marker_layout_fname)
print(f'Marker layout: {marker_layout_fname}')

tc = TransformedCoeffs(
    can_model=can_model,
    can_mrk_ids=can_mrk_ids,
    can_meta=can_meta,
    marker_layout_fname=marker_layout_fname,
    latent_labels=latent_labels,
    head_mrk_ids=head_mrk_ids,
    head_mrk_corr=head_mrk_corr,
    marker_corr=marker_corr
)

# Check that tc has closest attribute
print(f'tc.closest: {type(tc.closest)}')
print(f'tc.closest shape: {np.array(tc.closest).shape if hasattr(tc.closest, "__len__") else tc.closest}')

from moshpp.transformed_lm import TransformedLms
data_model = TransformedLms(transformed_coeffs=tc, can_body=can_model)
print(f'data_model built, dterms: {data_model.dterms}')

# Test forward pass
t0 = time.time()
r_val = data_model.r
t1 = time.time()
print(f'Forward pass (r): {t1-t0:.3f}s, shape={r_val.shape}')

# Test Jacobian columns one by one
free_vars = [can_model.trans, markers_latent]
pose_ids = list(range(66))  # body pose
v_poses = can_model.pose[pose_ids]
free_vars.append(v_poses)
free_vars.append(can_model.betas)

print(f'\nFree variables:')
for fv in free_vars:
    print(f'  {fv.short_name}: shape={fv.r.shape}, size={fv.r.size}')

# Test dr_wrt for each
for i, fv in enumerate(free_vars):
    print(f'\n--- Testing dr_wrt(data_model, freevar[{i}]: {fv.short_name}) ---')
    t0 = time.time()
    jac = data_model.dr_wrt(fv)
    t1 = time.time()
    if jac is not None:
        print(f'  Shape: {jac.shape}, nnz: {jac.nnz if sp.issparse(jac) else np.count_nonzero(jac)}, time: {t1-t0:.3f}s')
    else:
        print(f'  Result: None, time: {t1-t0:.3f}s')

print('\nDone!')
