#!/usr/bin/env python
"""Minimal test: measure forward pass time for a single frame."""
import sys
import time
import os.path as osp
import traceback

sys.path.insert(0, '/home/user/retargeting/moshpp/src')
from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers
setup_mosh_omegaconf_resolvers()

from moshpp.mosh_head import MoSh
import chumpy as ch
import numpy as np
import scipy.sparse as sp
from glob import glob

support_base_dir = '/home/user/retargeting/amass/support_data'
work_base_dir = '/home/user/retargeting/amass/work'
mocap_base_dir = osp.join(work_base_dir, 'CMU/c3d/subjects')
mocap_fnames = glob(osp.join(mocap_base_dir, '87', '87_03.c3d'))
mocap_fname = mocap_fnames[0]
print(f'Testing with: {mocap_fname}', flush=True)

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

print('Preparing cfg...', flush=True)
cfg = MoSh.prepare_cfg(**job)
print(f'Stage II: {cfg.dirs.stageii_fname}', flush=True)

# Use MoSh to handle marker loading
mp = MoSh(**cfg)

from moshpp.chmosh import mosh_stagei

print('Loading mocap...', flush=True)
t0 = time.time()
mp.mosh_stagei_load_mocap()
t1 = time.time()
print(f'Mocap loaded in {t1-t0:.1f}s. Frames: {len(mp.labels_obs)}, Markers: {len(mp.markers_obs[0])}', flush=True)

print('Loading surface model...', flush=True)
t0 = time.time()
mp.mosh_stagei_load_surface_model()
t1 = time.time()
print(f'Surface model loaded in {t1-t0:.1f}s', flush=True)

print('Getting marker correspondences...', flush=True)
t0 = time.time()
mp.mosh_stagei_get_mrk_corr()
t1 = time.time()
print(f'Marker correspondences in {t1-t0:.1f}s', flush=True)

# Access the internal objects to check what we have
can_model = mp.can_model
opt_models = mp.opt_models

print(f'\ncan_model type: {type(can_model).__name__}', flush=True)
print(f'can_model dterms: {can_model.dterms}', flush=True)
print(f'opt_models count: {len(opt_models)}', flush=True)

# Test forward pass on the model
print('\nTesting forward pass (model.r)...', flush=True)
t0 = time.time()
r = can_model.r
t1 = time.time()
print(f'model.r: shape={r.shape}, time={t1-t0:.3f}s', flush=True)

# Test TransformedLms forward pass
from moshpp.transformed_lm import TransformedCoeffs, TransformedLms

print('\nBuilding TransformedCoeffs...', flush=True)
marker_layout_fname = osp.join(cfg.dirs.support_base_dir, cfg.moshpp.marker_layout_fname)
print(f'marker_layout_fname: {marker_layout_fname}', flush=True)
print(f'  exists: {osp.exists(marker_layout_fname)}', flush=True)

# Get can_mrk_ids, can_meta, head_mrk_ids
from moshpp.chmosh import mosh_stagei
# These are computed in mosh_stagei_load_mrk_corr
can_mrk_ids = mp.can_mrk_ids
can_meta = mp.can_meta
head_mrk_ids = mp.head_mrk_ids

# Get latent markers
latent_labels = np.array(mp.labels_obs)
markers_obs = mp.markers_obs[0]
valid_mask = ~np.any(np.isnan(markers_obs), axis=1)
markers_latent = ch.array(markers_obs[valid_mask].copy())
print(f'markers_latent: shape={markers_latent.r.shape}', flush=True)

# Build TransformedCoeffs
from moshpp.tools.cfg_helper import get_marker_corr
marker_corr, head_mrk_corr = get_marker_corr(cfg, latent_labels)

t0 = time.time()
tc = TransformedCoeffs(
    can_model=can_model,
    can_mrk_ids=can_mrk_ids,
    can_meta=can_meta,
    marker_layout_fname=marker_layout_fname,
    latent_labels=latent_labels[valid_mask],
    head_mrk_ids=head_mrk_ids,
    head_mrk_corr=head_mrk_corr,
    marker_corr=marker_corr
)
t1 = time.time()
print(f'TransformedCoeffs built in {t1-t0:.1f}s', flush=True)
print(f'tc.closest: {tc.closest}', flush=True)

# Build TransformedLms
print('\nBuilding TransformedLms...', flush=True)
t0 = time.time()
data_model = TransformedLms(transformed_coeffs=tc, can_body=can_model)
t1 = time.time()
print(f'TransformedLms built in {t1-t0:.1f}s', flush=True)

# Test forward pass
print('\nTesting forward pass (data_model.r)...', flush=True)
t0 = time.time()
r_val = data_model.r
t1 = time.time()
print(f'data_model.r: shape={r_val.shape}, time={t1-t0:.3f}s', flush=True)

# Test 2nd forward pass (should use cache)
print('Testing second forward pass...', flush=True)
t0 = time.time()
r_val2 = data_model.r
t1 = time.time()
print(f'data_model.r (cached): time={t1-t0:.3f}s', flush=True)

# Test change + recompute
print('\nTesting recompute after changing pose...', flush=True)
t0 = time.time()
can_model.pose[:] = can_model.pose.r + 0.001 * np.random.randn(*can_model.pose.r.shape)
r_val3 = data_model.r
t1 = time.time()
print(f'Recompute after pose change: time={t1-t0:.3f}s', flush=True)

print('\nDone!', flush=True)
