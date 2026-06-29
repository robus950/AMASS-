#!/usr/bin/env python
"""Bare-minimum benchmark: forward pass + single dr_wrt call."""
import sys, time
sys.path.insert(0, '/home/user/retargeting/moshpp/src')

from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers
setup_mosh_omegaconf_resolvers()

from moshpp.mosh_head import MoSh
import os.path as osp
from glob import glob

support_base_dir = '/home/user/retargeting/amass/support_data'
work_base_dir = '/home/user/retargeting/amass/work'

job = {
    'mocap.fname': osp.join(work_base_dir, 'CMU/c3d/subjects/87/87_03.c3d'),
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

# Load surface model directly (bypass MoSh pipeline)
from moshpp.tools.cfg_helper import load_surface_model
model_type = cfg.surface_model.type
gender = cfg.surface_model.gender
num_betas = cfg.surface_model.num_betas
surface_model_fname = osp.join(cfg.dirs.support_base_dir, cfg.surface_model.surface_model_fname)

print(f'Loading model from {surface_model_fname}...', flush=True)
t0 = time.time()
can_model = load_surface_model(
    surface_model_fname, model_type=model_type, gender=gender,
    num_betas=num_betas, v_template_fname=None,
    pose_hand_prior_fname=None)
t1 = time.time()
print(f'Model loaded in {t1-t0:.1f}s', flush=True)
print(f'Model type: {can_model.model_type}', flush=True)
print(f'Model dterms: {can_model.dterms}', flush=True)
print(f'pose shape: {can_model.pose.r.shape}', flush=True)
print(f'betas shape: {can_model.betas.r.shape}', flush=True)
print(f'trans shape: {can_model.trans.r.shape}', flush=True)

# Test single forward pass
print('\nBenchmarking model.r...', flush=True)
for i in range(5):
    t0 = time.time()
    r = can_model.r
    t1 = time.time()
    print(f'  Pass {i}: shape={r.shape}, time={t1-t0:.4f}s', flush=True)

# Test changing pose and recomputing
print('\nBenchmarking after parameter change...', flush=True)
import numpy as np
for i in range(3):
    can_model.pose[:] = can_model.pose.r + 0.01 * np.random.randn(*can_model.pose.r.shape)
    t0 = time.time()
    r = can_model.r
    t1 = time.time()
    print(f'  After change {i}: time={t1-t0:.4f}s', flush=True)

# Test dr_wrt
import chumpy as ch
print('\nBenchmarking dr_wrt(pose)...', flush=True)
t0 = time.time()
jac = can_model.dr_wrt(can_model.pose)
t1 = time.time()
if jac is not None:
    import scipy.sparse as sp
    print(f'  dr_wrt(pose): shape={jac.shape}, time={t1-t0:.2f}s', flush=True)

print('\nBenchmarking dr_wrt(pose[0:3])...', flush=True)
pose_subset = can_model.pose[list(range(3))]
t0 = time.time()
jac = can_model.dr_wrt(pose_subset)
t1 = time.time()
if jac is not None:
    import scipy.sparse as sp
    print(f'  dr_wrt(pose[0:3]): shape={jac.shape}, time={t1-t0:.2f}s', flush=True)

print('\nDone!', flush=True)
