#!/usr/bin/env python
"""Test LBS forward pass performance."""
import sys, time, numpy as np
sys.path.insert(0, '/home/user/retargeting/moshpp/src')
from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers
setup_mosh_omegaconf_resolvers()
from moshpp.models.smpl_fast_derivatives import load_surface_model

t0 = time.time()
sm = load_surface_model(
    surface_model_fname='/home/user/retargeting/amass/support_data/smplx/male/model.pkl',
    pose_hand_prior_fname='/home/user/retargeting/amass/support_data/smplx/pose_hand_prior.npz',
    use_hands_mean=True, surface_model_type='smplx',
)
print(f'load_surface_model: {time.time()-t0:.1f}s', flush=True)

# Test forward pass
np.random.seed(42)
sm.pose[:] = np.random.randn(len(sm.pose.r)) * 0.1
sm.betas[:] = np.random.randn(len(sm.betas.r)) * 0.5

t0 = time.time()
v = sm.r
print(f'Forward pass (compute r): {time.time()-t0:.1f}s, shape={v.shape}', flush=True)

# Test multiple forward passes
t0 = time.time()
for _ in range(13):
    _ = sm.r
print(f'13 forward passes: {time.time()-t0:.1f}s', flush=True)

# Now test creating SmplModelLBS 13 times (simulating can_model + 12 opt_models)
from moshpp.models.smpl_fast_derivatives import SmplModelLBS
import chumpy as ch

betas = ch.array(np.zeros(len(sm.betas)))
temp_model = sm  # the Struct

t0 = time.time()
m1 = SmplModelLBS(trans=ch.array(np.zeros(3)), pose=ch.array(np.zeros(temp_model.pose.size)),
                  betas=betas, temp_model=temp_model)
print(f'1st SmplModelLBS: {time.time()-t0:.1f}s', flush=True)

t0 = time.time()
for i in range(13):
    m = SmplModelLBS(trans=ch.array(np.zeros(3)), pose=ch.array(np.zeros(temp_model.pose.size)),
                     betas=betas, temp_model=temp_model)
print(f'13 SmplModelLBS: {time.time()-t0:.1f}s', flush=True)

print('DONE', flush=True)
