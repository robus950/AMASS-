#!/usr/bin/env python
"""Minimal test to trace MoSh++ execution step by step."""
import sys, os, time
sys.path.insert(0, '/home/user/retargeting/moshpp/src')
from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers
setup_mosh_omegaconf_resolvers()
from omegaconf import OmegaConf

support_base_dir = '/home/user/retargeting/amass/support_data'
work_base_dir = '/home/user/retargeting/amass/work'

cfg = OmegaConf.load('/home/user/retargeting/moshpp/support_data/conf/moshpp_conf.yaml')
override = OmegaConf.create({
    'dirs': {
        'support_base_dir': support_base_dir,
        'work_base_dir': work_base_dir,
    },
    'surface_model': {'type': 'smplx', 'gender': 'male'},
    'moshpp': {
        'pose_body_prior_fname': None,
        'head_marker_corr_fname': None,
        'optimize_fingers': False,
        'optimize_face': False,
        'optimize_dynamics': False,
    },
    'runtime': {'stagei_only': False},
    'mocap': {'fname': '/home/user/retargeting/amass/work/CMU/c3d/subjects/87/87_03.c3d'},
})
cfg = OmegaConf.merge(cfg, override)

from moshpp.mosh_head import MoSh
mp = MoSh(dict_cfg=cfg)

from moshpp.chmosh import mosh_stagei, mosh_stageii

# Get stage I frames
print('Preparing stage I frames...', flush=True)
stagei_frames = mp.prepare_stagei_frames()
print(f'Got {len(stagei_frames)} frames', flush=True)

# Just test loading models (which triggers verts_decorated)
print('Loading models...', flush=True)
t0 = time.time()
from moshpp.models.bodymodel_loader import load_moshpp_models
can_model, opt_models = load_moshpp_models(
    surface_model_fname=cfg.surface_model.fname,
    surface_model_type=cfg.surface_model.type,
    optimize_face=False,
    num_beta_shared_models=len(stagei_frames),
    pose_hand_prior_fname=cfg.moshpp.pose_hand_prior_fname,
    pose_body_prior_fname=cfg.moshpp.pose_body_prior_fname,
    use_hands_mean=cfg.surface_model.use_hands_mean,
    dof_per_hand=cfg.surface_model.dof_per_hand,
)
print(f'Models loaded in {time.time()-t0:.1f}s', flush=True)

# Test a single gradient evaluation
print('Testing gradient...', flush=True)
t0 = time.time()
dr = can_model.compute_dr_wrt(can_model.pose)
print(f'Pose grad: {dr.shape}, time={time.time()-t0:.1f}s', flush=True)

t0 = time.time()
dr_b = can_model.compute_dr_wrt(can_model.betas)
print(f'Betas grad: {dr_b.shape}, time={time.time()-t0:.1f}s', flush=True)

# Test surface distance setup
print('Testing marker latent setup...', flush=True)
t0 = time.time()
from moshpp.marker_layout.edit_tools import marker_layout_load
from moshpp.chmosh import prepare_mosh_markers_latent
marker_meta = marker_layout_load(cfg.dirs.marker_layout.fname, include_nan=True)
print(f'marker_meta loaded: marker_types={list(marker_meta["marker_type_mask"].keys())}', flush=True)
markers_latent, dist_obj = prepare_mosh_markers_latent(can_model=can_model, marker_meta=marker_meta)
print(f'Surface distance setup in {time.time()-t0:.1f}s', flush=True)

print('All steps OK!', flush=True)
