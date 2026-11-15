#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_relations.py -- ground-truth error of the SLAM trajectories on the
Intel dataset, with the relative-relations metric of the Freiburg SLAM
benchmark (Kuemmerle et al., 2009, "On measuring the accuracy of SLAM
algorithms", Autonomous Robots 27(4)).

For every relation (t1, t2, delta*) in data/intel.relations, the estimated
relative pose delta = x(t1)^-1 (+) x(t2) is computed from the best-particle
trajectory and compared with delta*:
  translational error  ||delta_xy - delta*_xy||       (m)
  rotational error     |wrap(delta_yaw - delta*_yaw)|  (deg)
Because only relative poses are compared, no alignment of the map frame with
a ground-truth frame is needed. The same is computed for the raw odometry of
each run, as a reference.

analyze.py calls this automatically for the Intel dataset. Standalone:
  python evaluate_relations.py                      # default folder, intel
  python evaluate_relations.py --out my_results --relations data/intel.relations
Writes <out>/analysis/relations_<dataset>.csv with one row per run.
"""
from __future__ import division, print_function

import argparse
import glob
import os

import numpy as np

from intel_data import load_relations

TIME_TOL = 1e-3   # s; a relation timestamp must match a scan timestamp this closely

COLUMNS = ['rel_trans', 'rel_trans_std', 'rel_trans_sq', 'rel_rot', 'rel_rot_std',
           'rel_rot_sq', 'odom_rel_trans', 'odom_rel_rot', 'n_rel', 'n_rel_total']


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def match_times(times, query):
    """Index of the scan whose timestamp equals each query time (or -1)."""
    order = np.argsort(times)
    ts = times[order]
    pos = np.clip(np.searchsorted(ts, query), 1, len(ts) - 1)
    left, right = ts[pos - 1], ts[pos]
    pick = np.where(np.abs(query - left) <= np.abs(query - right), pos - 1, pos)
    ok = np.abs(ts[pick] - query) <= TIME_TOL
    return np.where(ok, order[pick], -1)


def relation_errors(times, x, y, th, rel):
    i = match_times(times, rel[:, 0])
    j = match_times(times, rel[:, 1])
    ok = (i >= 0) & (j >= 0)
    i, j, r = i[ok], j[ok], rel[ok]
    dx, dy = x[j] - x[i], y[j] - y[i]
    c, s = np.cos(th[i]), np.sin(th[i])
    est_x, est_y = c * dx + s * dy, -s * dx + c * dy        # in frame of pose i
    est_th = wrap(th[j] - th[i])
    e_t = np.hypot(est_x - r[:, 2], est_y - r[:, 3])
    e_r = np.abs(wrap(est_th - r[:, 4]))
    return e_t, np.rad2deg(e_r), int(ok.sum())


def evaluate_run(npz_path, rel):
    d = np.load(npz_path)
    if 'times' not in d or 'traj_theta' not in d:
        raise ValueError('%s was made by an older experiments.py (no timestamps)' % npz_path)
    t = d['times']
    e_t, e_r, n = relation_errors(t, d['traj'][:, 0], d['traj'][:, 1], d['traj_theta'], rel)
    o_t, o_r, _ = relation_errors(t, d['odom'][:, 0], d['odom'][:, 1], d['odom_theta'], rel)
    if n == 0:
        nan = float('nan')
        return dict((c, nan) for c in COLUMNS[:-2]), 0, len(rel)
    res = {'rel_trans': float(e_t.mean()), 'rel_trans_std': float(e_t.std()),
           'rel_trans_sq': float(np.mean(e_t ** 2)),
           'rel_rot': float(e_r.mean()), 'rel_rot_std': float(e_r.std()),
           'rel_rot_sq': float(np.mean(e_r ** 2)),
           'odom_rel_trans': float(o_t.mean()), 'odom_rel_rot': float(o_r.mean()),
           'n_rel': n, 'n_rel_total': len(rel)}
    return res, n, len(rel)


def evaluate_folder(out_dir, dataset, relations_path):
    """Dict {(config, seed): metrics} for every saved run of this dataset."""
    rel = load_relations(relations_path)
    results = {}
    for f in sorted(glob.glob(os.path.join(out_dir, 'runs', 'd%s' % dataset, '*.npz'))):
        stem = os.path.basename(f)[:-4]
        config, seed = stem.rsplit('_seed', 1)
        try:
            results[(config, int(seed))] = evaluate_run(f, rel)[0]
        except ValueError as err:
            print('skipped: %s' % err)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='result_experiment_sweeps')
    p.add_argument('--dataset', default='intel')
    p.add_argument('--relations', default=os.path.join('data', 'intel.relations'))
    args = p.parse_args()

    results = evaluate_folder(args.out, args.dataset, args.relations)
    adir = os.path.join(args.out, 'analysis')
    if not os.path.isdir(adir):
        os.makedirs(adir)
    path = os.path.join(adir, 'relations_%s.csv' % args.dataset)
    with open(path, 'w') as f:
        f.write(','.join(['config', 'seed'] + COLUMNS) + '\n')
        for (config, seed), r in sorted(results.items()):
            f.write(','.join([config, str(seed)] + [str(r[c]) for c in COLUMNS]) + '\n')
    print('%-26s %4s %12s %12s %12s %12s %8s' % (
        'config', 'seed', 'trans (m)', 'rot (deg)', 'odom trans', 'odom rot', 'used'))
    for (config, seed), r in sorted(results.items()):
        print('%-26s %4d %12.3f %12.2f %12.3f %12.2f %4d/%d' % (
            config, seed, r['rel_trans'], r['rel_rot'], r['odom_rel_trans'],
            r['odom_rel_rot'], r['n_rel'], r['n_rel_total']))
    print('\nWritten to %s' % path)


if __name__ == '__main__':
    main()
