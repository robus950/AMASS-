#!/usr/bin/env python
"""
Quick Stage II test on 3 frames using L-BFGS-B (chumpy's native _use_numeric path).
Stage I output already exists - reuses it.
"""
import sys, os, os.path as osp
import time

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

sys.path.insert(0, '/home/user/retargeting/moshpp/src')

from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers
setup_mosh_omegaconf_resolvers()

import numpy as np
from moshpp.mosh_head import MoSh

support_base_dir = '/home/user/retargeting/amass/support_data'
work_base_dir = '/home/user/retargeting/amass/work'
mocap_fname = osp.join(work_base_dir, 'CMU/c3d/subjects/87/87_03.c3d')

print(f'Processing: {mocap_fname}', flush=True)

job = {
    'mocap.fname': mocap_fname,
    'mocap.end_fidx': 3,  # Only 3 frames for testing
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
    'opt_settings.maxiter': 3,
    'moshpp.stagei_frame_picker.least_avail_markers': 0.3,
    'moshpp.stagei_frame_picker.num_frames': 4,
    'dirs.write_optimized_marker_layout': False,
}

print('Preparing config...', flush=True)
cfg = MoSh.prepare_cfg(**job)
print(f'Stage I fname: {cfg.dirs.stagei_fname}', flush=True)
print(f'Stage I exists: {osp.exists(cfg.dirs.stagei_fname)}', flush=True)
print(f'Stage II fname: {cfg.dirs.stageii_fname}', flush=True)

print('Creating MoSh object...', flush=True)
mp = MoSh(**job)

# Stage I: will load from existing file
print('Loading Stage I results...', flush=True)
t0 = time.time()
mp.mosh_stagei(None)  # Will load from existing file
print(f'Stage I loaded in {time.time()-t0:.1f}s', flush=True)

# We need to import the actual mosh_stageii function
from moshpp.chmosh import mosh_stagei, mosh_stageii

# Run Stage II
print('Running Stage II on 3 frames...', flush=True)
t0 = time.time()
mp.mosh_stageii(mosh_stageii)
elapsed = time.time() - t0
print(f'Stage II complete in {elapsed:.1f}s ({elapsed/3:.1f}s per frame)', flush=True)

# Check results
if mp.stageii_data:
    errs = mp.stageii_data.get('stageii_debug_details', {}).get('stageii_errs', {})
    for k, v in errs.items():
        print(f'  {k} loss: {v[-1]:2.2e} (final frame)', flush=True)
    print(f'Stage II data keys: {list(mp.stageii_data.keys())}', flush=True)

# Convert to AMASS npz
stageii_fname = mp.stageii_fname
if osp.exists(stageii_fname):
    print(f'Converting to npz: {stageii_fname}', flush=True)
    amass_data = MoSh.load_as_amass_npz(stageii_fname)
    for k, v in amass_data.items():
        print(f'  {k}: {v.shape if hasattr(v, "shape") else v}', flush=True)

    out_npz = stageii_fname.replace('_stageii.pkl', '_poses.npz')
    np.savez(out_npz, **amass_data)
    print(f'Saved: {out_npz}', flush=True)
else:
    print(f'ERROR: Stage II file not found: {stageii_fname}', flush=True)

print('Done!', flush=True)
