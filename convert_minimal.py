#!/usr/bin/env python
"""
Minimal CMU conversion - runs only stagei with tiny budget to test.
"""
import sys, os, os.path as osp
from glob import glob

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

sys.path.insert(0, '/home/user/retargeting/moshpp/src')

from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers
setup_mosh_omegaconf_resolvers()

import numpy as np
from moshpp.mosh_head import MoSh
from moshpp.chmosh import mosh_stagei

support_base_dir = '/home/user/retargeting/amass/support_data'
work_base_dir = '/home/user/retargeting/amass/work'
mocap_fname = osp.join(work_base_dir, 'CMU/c3d/subjects/87/87_03.c3d')

print(f'Processing: {mocap_fname}', flush=True)

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
    'moshpp.stagei_frame_picker.least_avail_markers': 0.3,
    'dirs.write_optimized_marker_layout': False,
}

print('Preparing config...', flush=True)
cfg = MoSh.prepare_cfg(**job)
print(f'Stage II target: {cfg.dirs.stageii_fname}', flush=True)

# Monkey-patch ch.minimize to use L-BFGS-B
import chumpy as ch
_orig_minimize = ch.minimize

def _patched_minimize(fun, x0, method='dogleg', bounds=None, constraints=(),
                      tol=None, callback=None, options=None):
    if options is None:
        options = {}
    options.setdefault('maxiter', 5)
    options.setdefault('maxfev', 50)
    options.setdefault('disp', True)
    print(f'[patched] minimize called with method=Nelder-Mead, maxiter={options.get("maxiter")}, maxfev={options.get("maxfev")}', flush=True)
    return _orig_minimize(fun, x0, method='Nelder-Mead', bounds=bounds,
                          constraints=constraints, tol=tol,
                          callback=callback, options=options)

ch.minimize = _patched_minimize

print('Creating MoSh object...', flush=True)
mp = MoSh(**job)
print(f'Stage I fname: {mp.stagei_fname}', flush=True)
print(f'Stage I exists: {osp.exists(mp.stagei_fname)}', flush=True)

print('Running mosh_stagei...', flush=True)
mp.mosh_stagei(mosh_stagei)
print('Stage I complete!', flush=True)
print(f'Stage I data keys: {list(mp.stagei_data.keys()) if mp.stagei_data else None}', flush=True)

# Stage II: per-frame optimization
if not cfg.runtime.stagei_only:
    from moshpp.chmosh import mosh_stageii
    print('Running mosh_stageii...', flush=True)
    mp.mosh_stageii(mosh_stageii)
    print('Stage II complete!', flush=True)

    # Convert to AMASS npz
    stageii_fname = mp.stageii_fname
    print(f'Converting to npz: {stageii_fname}', flush=True)
    amass_data = MoSh.load_as_amass_npz(stageii_fname)
    for k, v in amass_data.items():
        print(f'  {k}: {v.shape if hasattr(v, "shape") else v}', flush=True)

    import numpy as np
    out_npz = stageii_fname.replace('_stageii.pkl', '_poses.npz')
    np.savez(out_npz, **amass_data)
    print(f'Saved: {out_npz}', flush=True)
else:
    print('Stage II skipped (stagei_only=True)', flush=True)

print('Done!', flush=True)
