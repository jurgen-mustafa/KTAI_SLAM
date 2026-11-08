"""
test_experiments.py -- smoke test for experiments.py and analyze.py that does
not need the THOR data. It simulates a robot walking a loop through a small
building (lidar scans ray-cast against walls, odometry with heading drift),
in the same format as load_data.get_lidar / get_joint, and runs a reduced
version of every sweep through the full pipeline.

  python test_experiments.py          # takes a few minutes; output in results_test/
"""
from __future__ import division, print_function

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


def synthetic_data(n_scans=700, seed=1):
    rng = np.random.RandomState(seed)
    waypoints = np.array([[0.5, 0.2], [8, 0.2], [8, 4], [0.5, 4], [0.5, 0.2]])
    seg = np.diff(waypoints, axis=0)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
    s = np.linspace(0, cum[-1] - 1e-6, n_scans)
    xs = np.interp(s, cum, waypoints[:, 0])
    ys = np.interp(s, cum, waypoints[:, 1])
    th = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
    angles = np.arange(-135, 135.25, 0.25) * np.pi / 180.0

    # odometry: integrate true increments with a small heading drift + noise
    ox, oy, oth = [xs[0]], [ys[0]], [th[0]]
    for k in range(1, n_scans):
        d = np.hypot(xs[k] - xs[k - 1], ys[k] - ys[k - 1])
        dth = th[k] - th[k - 1] + 2e-4 + rng.normal(0, 1e-3)
        oth.append(oth[-1] + dth)
        ox.append(ox[-1] + d * np.cos(oth[-1]))
        oy.append(oy[-1] + d * np.sin(oth[-1]))

    lid = []
    for k in range(n_scans):
        scan = raycast(xs[k], ys[k], th[k], angles) + rng.normal(0, 0.01, angles.size)
        lid.append({'t': np.array([[0.025 * k]]),
                    'pose': np.array([[ox[k], oy[k], oth[k]]]),
                    'rpy': np.array([[0.0, 0.0, oth[k]]]),
                    'scan': np.array([np.maximum(scan, 0.0)])})
    joint = {'ts': np.array([0.025 * np.arange(n_scans)]),
             'rpy': np.zeros((3, n_scans)),
             'head_angles': np.zeros((2, n_scans))}
    return lid, joint


if __name__ == '__main__':
    data = synthetic_data()
    experiments.load_dataset = lambda dataset, data_dir='data': data
    # reduced sweeps so the test stays quick
    experiments.SWEEPS[:] = [('particles', 'N', [1, 10, 30]),
                             ('noise', 'sigma', [0.0, 1e-3, 5e-2]),
                             ('range', 'max_range', [30.0, 4.0, 2.0])]
    experiments.DEFAULT['N'] = 10
    out = 'results_test'
    shutil.rmtree(out, ignore_errors=True)
    sys.argv = ['experiments.py', '--seeds', '2', '--out', out]
    experiments.main()
    sys.argv = ['analyze.py', '--out', out]
    analyze.main()
    print('\nSmoke test passed. Inspect %s/analysis/' % out)
