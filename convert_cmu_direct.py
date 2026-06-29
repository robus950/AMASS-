#!/usr/bin/env python
"""
Direct CMU to AMASS converter using MoSh++ but with scipy-based optimization.

Key difference from convert_cmu.py: uses scipy.optimize.minimize directly with
finite-difference gradients (much faster than chumpy graph traversal) and a small
iteration budget for testing.
"""
import sys, os, os.path as osp
from glob import glob

sys.path.insert(0, '/home/user/retargeting/moshpp/src')

from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers
setup_mosh_omegaconf_resolvers()

from moshpp.mosh_head import run_moshpp_once, MoSh
import numpy as np

# Override chumpy minimize to use scipy L-BFGS-B with numerical gradients
import chumpy.optimization as chopt
_orig_minimize = chopt.minimize

def patched_minimize(fun, x0, method='dogleg', bounds=None, constraints=(),
                     tol=None, callback=None, options=None):
    """Force use of scipy L-BFGS-B instead of dogleg."""
    if options is None:
        options = {}
    # Force small iteration budget
    options.setdefault('maxiter', 10)
    options.setdefault('maxfun', 500)
    options.setdefault('disp', 0)
    return _orig_minimize(fun, x0, method='L-BFGS-B', bounds=bounds,
                          constraints=constraints, tol=tol,
                          callback=callback, options=options)

chopt.minimize = patched_minimize

support_base_dir = '/home/user/retargeting/amass/support_data'
work_base_dir = '/home/user/retargeting/amass/work'
mocap_base_dir = osp.join(work_base_dir, 'CMU/c3d/subjects')

mocap_fnames = glob(osp.join(mocap_base_dir, '87', '87_03.c3d'))
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
        'opt_settings.maxiter': 5,  # Small for testing
    }

    try:
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
    except Exception as e:
        import traceback
        print(f'ERROR: {e}')
        traceback.print_exc()

print('\nDone!')
