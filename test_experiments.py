#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_experiments.py -- smoke test for experiments.py, analyze.py, intel_data.py
and evaluate_relations.py that needs neither the THOR nor the Intel data. It
simulates a robot driving a loop through a small building (lidar scans
ray-cast against walls, odometry with heading drift) and runs a reduced
version of every sweep through the full pipeline:
  thor   data in the format of load_data.get_lidar / get_joint
  intel  a CARMEN .clf log (180 beams, 1 deg) plus a .relations file,
         written to results_test/data/ and read back by intel_data.py

  python test_experiments.py            # both, about 5-10 minutes
  python test_experiments.py intel      # only the Intel pipeline
"""
from __future__ import division, print_function

import os
import shutil
import sys

import numpy as np

import experiments
import analyze

WALLS = np.array([  # x1, y1, x2, y2 (m)
    [-1, -1, 9, -1], [9, -1, 9, 5], [9, 5, -1, 5], [-1, 5, -1, -1],   # outer
    [3, -1, 3, 1.5], [3, 2.8, 3, 5],                                  # wall with door
    [6, 1.8, 7, 1.8], [7, 1.8, 7, 2.6], [7, 2.6, 6, 2.6], [6, 2.6, 6, 1.8],  # box
])


def raycast(x, y, th, angles, max_r=30.0):
    a = th + angles
    dx, dy = np.cos(a)[:, None], np.sin(a)[:, None]
    x1, y1, x2, y2 = [WALLS[:, i][None, :] for i in range(4)]
    ex, ey = x2 - x1, y2 - y1
    den = dx * ey - dy * ex
    with np.errstate(divide='ignore', invalid='ignore'):
        t = ((x1 - x) * ey - (y1 - y) * ex) / den
        u = ((x1 - x) * dy - (y1 - y) * dx) / den
    t[(np.abs(den) < 1e-12) | (t <= 0) | (u < 0) | (u > 1)] = np.inf
    r = t.min(axis=1)
    r[~np.isfinite(r)] = 0.0
    return np.minimum(r, max_r)


def simulate(n_scans, angles, seed=1):
    """Ground-truth poses, odometry poses and scans for one loop."""
    rng = np.random.RandomState(seed)
    waypoints = np.array([[0.5, 0.2], [8, 0.2], [8, 4], [0.5, 4], [0.5, 0.2]])
    seg = np.diff(waypoints, axis=0)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
    s = np.linspace(0, cum[-1] - 1e-6, n_scans)
    xs = np.interp(s, cum, waypoints[:, 0])
    ys = np.interp(s, cum, waypoints[:, 1])
    th = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))

    # odometry: integrate true increments with a small heading drift + noise
    ox, oy, oth = [xs[0]], [ys[0]], [th[0]]
    for k in range(1, n_scans):
        d = np.hypot(xs[k] - xs[k - 1], ys[k] - ys[k - 1])
        dth = th[k] - th[k - 1] + 2e-4 * 700.0 / n_scans + rng.normal(0, 1e-3)
        oth.append(oth[-1] + dth)
        ox.append(ox[-1] + d * np.cos(oth[-1]))
        oy.append(oy[-1] + d * np.sin(oth[-1]))
    scans = [np.maximum(raycast(xs[k], ys[k], th[k], angles)
                        + rng.normal(0, 0.01, angles.size), 0.0) for k in range(n_scans)]
    return (xs, ys, th), (np.array(ox), np.array(oy), np.array(oth)), scans


def synthetic_data(n_scans=700, seed=1):
    """THOR-format data: (lid, joint, angles)."""
    angles = experiments.THOR_ANGLES
    _, (ox, oy, oth), scans = simulate(n_scans, angles[0], seed)
    lid = []
    for k in range(n_scans):
        lid.append({'t': np.array([[0.025 * k]]),
                    'pose': np.array([[ox[k], oy[k], oth[k]]]),
                    'rpy': np.array([[0.0, 0.0, oth[k]]]),
                    'scan': np.array([scans[k]])})
    joint = {'ts': np.array([0.025 * np.arange(n_scans)]),
             'rpy': np.zeros((3, n_scans)),
             'head_angles': np.zeros((2, n_scans))}
    return lid, joint, angles


def write_intel_files(data_dir, n_scans=500, seed=2):
    """A CARMEN log and a relations file in the format of the Freiburg benchmark."""
    import intel_data
    angles = intel_data.intel_angles(180)[0]
    (xs, ys, th), (ox, oy, oth), scans = simulate(n_scans, angles, seed)
    t = 1000.0 + 0.2 * np.arange(n_scans)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, 'intel.clf'), 'w') as f:
        f.write('# synthetic CARMEN log\nPARAM robot_length 0.5\n')
        for k in range(n_scans):
            f.write('ODOM %f %f %f 0 0 0 %f host %f\n' % (ox[k], oy[k], oth[k], t[k], t[k]))
            f.write('FLASER 180 ' + ' '.join('%.3f' % r for r in scans[k]) +
                    ' %f %f %f %f %f %f %f host %f\n'
                    % (ox[k], oy[k], oth[k], ox[k], oy[k], oth[k], t[k], t[k]))
    rng = np.random.RandomState(seed)
    with open(os.path.join(data_dir, 'intel.relations'), 'w') as f:
        for _ in range(300):
            i = rng.randint(0, n_scans - 60)
            j = i + rng.randint(1, 60)
            c, s = np.cos(th[i]), np.sin(th[i])
            dx, dy = xs[j] - xs[i], ys[j] - ys[i]
            dyaw = (th[j] - th[i] + np.pi) % (2 * np.pi) - np.pi
            f.write('%f %f %f %f 0 0 0 %f\n' % (t[i], t[j], c * dx + s * dy,
                                                -s * dx + c * dy, dyaw))


def reduce_sweeps():
    experiments.SWEEPS[:] = [('particles', 'N', [1, 10, 30]),
                      ('noise', 'sigma', [0.0, 1e-3, 5e-2]),
                      ('range', 'max_range', [30.0, 4.0, 2.0])]
    experiments.DEFAULT['N'] = 10


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    out = 'results_test'
    shutil.rmtree(out, ignore_errors=True)
    reduce_sweeps()
    if which in ('all', 'thor'):
        data = synthetic_data()
        real_loader = experiments.load_dataset
        experiments.load_dataset = lambda dataset, data_dir='data': \
            data if dataset == '0' else real_loader(dataset, data_dir)
        sys.argv = ['experiments.py', '--seeds', '2', '--out', out]
        experiments.main()
    if which in ('all', 'intel'):
        data_dir = os.path.join(out, 'data')
        write_intel_files(data_dir)
        sys.argv = ['experiments.py', '--datasets', 'intel', '--seeds', '2',
                    '--out', out, '--data-dir', data_dir, '--jobs', '2']
        experiments.main()
    sys.argv = ['analyze.py', '--out', out,
                '--relations', os.path.join(out, 'data', 'intel.relations')]
    analyze.main()
    print('\nSmoke test passed. Inspect %s/analysis/' % out)
