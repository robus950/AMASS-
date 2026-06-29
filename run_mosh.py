# -*- coding: utf-8 -*-
"""Run MoSh++ on CMU c3d files to produce AMASS npz."""
import sys
import os

sys.path.insert(0, '/home/user/retargeting/moshpp/src')

from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers, turn_fullpose_into_parts
from omegaconf import OmegaConf
import numpy as np
import pickle

setup_mosh_omegaconf_resolvers()

# Load base config
base_cfg = OmegaConf.load('/home/user/retargeting/moshpp/support_data/conf/moshpp_conf.yaml')

override = OmegaConf.create({
    'dirs': {
        'support_base_dir': '/home/user/retargeting/amass/support_data',
        'work_base_dir': '/home/user/retargeting/amass/work',
    },
    'surface_model': {
        'type': 'smplx',
        'gender': 'male',
    },
    'moshpp': {
        'pose_body_prior_fname': None,
        'pose_hand_prior_fname': None,
        'head_marker_corr_fname': None,
    },
    'runtime': {
        'stagei_only': False,
    },
})

cfg = OmegaConf.merge(base_cfg, override)

# Process each c3d file
c3d_files = [
    '/home/user/retargeting/amass/work/CMU/c3d/subjects/87/87_03.c3d',
    '/home/user/retargeting/amass/work/CMU/c3d/subjects/87/87_04.c3d',
]

for c3d_fname in c3d_files:
    if not os.path.exists(c3d_fname):
        print(f'File not found: {c3d_fname}')
        continue

    print(f'\n{"="*60}')
    print(f'Processing: {c3d_fname}')

    job_cfg = OmegaConf.merge(cfg, OmegaConf.create({'mocap': {'fname': c3d_fname}}))

    from moshpp.mosh_head import run_moshpp_once
    from moshpp.mosh_head import MoSh
    from moshpp.chmosh import mosh_stagei, mosh_stageii

    mp = MoSh(dict_cfg=job_cfg)

    print('Running Stage I...')
    mp.mosh_stagei(mosh_stagei)

    print('Running Stage II...')
    mp.mosh_stageii(mosh_stageii)

    # Convert to AMASS npz
    stageii_fname = mp.cfg.dirs.stageii_fname
    print(f'Converting to npz: {stageii_fname}')

    amass_data = MoSh.load_as_amass_npz(stageii_fname)
    for k, v in amass_data.items():
        print(f'  {k}: {v.shape if hasattr(v, "shape") else v}')

    out_npz = stageii_fname.replace('_stageii.pkl', '_poses.npz')
    np.savez(out_npz, **amass_data)
    print(f'Saved: {out_npz}')
