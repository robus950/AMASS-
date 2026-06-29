#!/usr/bin/env python
"""Convert CMU c3d files to AMASS npz format using MoSh++."""
import sys, os, os.path as osp
from glob import glob

sys.path.insert(0, '/home/user/retargeting/moshpp/src')

from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers
setup_mosh_omegaconf_resolvers()

from moshpp.mosh_head import run_moshpp_once, MoSh
import numpy as np

support_base_dir = '/home/user/retargeting/amass/support_data'
work_base_dir = '/home/user/retargeting/amass/work'
mocap_base_dir = osp.join(work_base_dir, 'CMU/c3d/subjects')

mocap_fnames = glob(osp.join(mocap_base_dir, '87', '*.c3d'))
print(f'Found {len(mocap_fnames)} c3d files:')
for f in mocap_fnames:
    print(f'  {f}')

for mocap_fname in mocap_fnames:
    print(f'\n{"="*60}')
    print(f'Processing: {mocap_fname}')

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
        'opt_settings.maxiter': 30,
        'moshpp.stagei_frame_picker.least_avail_markers': 0.3,
        'moshpp.stagei_frame_picker.num_frames': 4,
        'dirs.write_optimized_marker_layout': False,
    }

    run_moshpp_once(job)

    # Convert to AMASS npz
    cfg = MoSh.prepare_cfg(**job)
    stageii_fname = cfg.dirs.stageii_fname
    print(f'Converting to npz: {stageii_fname}')
    amass_data = MoSh.load_as_amass_npz(stageii_fname)
    for k, v in amass_data.items():
        print(f'  {k}: {v.shape if hasattr(v, "shape") else v}')

    out_npz = stageii_fname.replace('_stageii.pkl', '_poses.npz')
    np.savez(out_npz, **amass_data)
    print(f'Saved: {out_npz}')

print('\nDone!')
