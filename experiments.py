"""
experiments.py -- parameter sweeps for the three research questions, built on
the yashv28 Particle-Filter-SLAM code (THOR humanoid data).

  RQ1  number of particles  N          in [1, 10, 30, 100, 300]
  RQ2  motion-noise sigma               in [0, 1e-4, 1e-3, 1e-2, 5e-2]
  RQ3  maximum lidar range (m)          in [30, 15, 8, 4, 2]
  Baseline: odometry only (N = 1, sigma = 0, no scan-matching correction)

Every sweep changes ONE parameter and keeps the others at the repo defaults
(N = 100, sigma = 1e-3, range = 30 m, resolution = 0.05 m, 3x3 window of +-5 cm).

The SLAM loop below follows SLAM.py step by step and reuses the repo's own
mapConvert / drawMap / mapCorrelation unchanged. Differences from SLAM.py:
  1. FIX  Resampling: in SLAM.py the copy `particle_New[m, :] = particles[i, :]`
          sits outside the for-loop, so all but one particle are reset to (0,0,0).
          Here: the same low-variance resampler, correctly implemented.
  2. FIX  Motion noise: SLAM.py draws ONE N(0, sigma) sample per particle and
          broadcasts it to x, y and yaw (noise = [n, n, 10n]). Here: an
          independent sample per dimension, still scaled by factor [1, 1, 10].
  3. Resampling threshold scales with N (N_eff < 0.35 * N), so that it means
     the same for every particle count (the repo uses 35 for N = 100).
  4. Lidar range is passed to mapConvert as its arg_max argument (our modified
     SLAM_helper.py: valid returns are 0.1 < r < arg_max; the repo used 30).
  5. Cell indices are cast from int16 to int32 before drawMap, because
     OpenCV 4+ only accepts int32 contour points (no change in behaviour).
  6. Instrumentation: N_eff per step, best-particle trajectory, odometry
     trajectory, final map, run time. No live plotting.

Usage (Python 2.7, run from the repo root, data pulled with `git lfs pull`):
  python experiments.py --dry-run                       # list the runs
  python experiments.py                                 # everything, dataset 0
  python experiments.py --experiments particles --jobs 2
  python experiments.py --max-scans 300 --seeds 1       # quick check
Results are appended to <out>/results.csv; finished runs are skipped when you
restart, so an interrupted sweep can simply be started again.
"""
from __future__ import division, print_function

import argparse
import os
import random
import sys
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import load_data as ld
import MapUtils as maput
from SLAM_helper import mapConvert, drawMap


# --------------------------------------------------------------------------
# Experiment definition
# --------------------------------------------------------------------------
DEFAULT = {'N': 100, 'sigma': 1e-3, 'max_range': 30.0, 'scan_matching': 1}
BASELINE = {'N': 1, 'sigma': 0.0, 'max_range': 30.0, 'scan_matching': 0}

SWEEPS = [  # (experiment name, parameter, values)
    ('particles', 'N', [1, 10, 30, 100, 300]),
    ('noise', 'sigma', [0.0, 1e-4, 1e-3, 1e-2, 5e-2]),
    ('range', 'max_range', [30.0, 15.0, 8.0, 4.0, 2.0]),
]

# Fixed settings (repo defaults)
NEFF_RATIO = 0.35                                  # 35 / 100 in SLAM.py
FACTOR = np.array([1.0, 1.0, 10.0])                # noise scaling x, y, yaw
WINDOW = np.arange(-0.05, 0.06, 0.05)              # scan-matching offsets (m)
MAP_RES, MAP_LIM = 0.05, 40.0                      # cell size, +- extent (m)
JUMP_THRESHOLD = 0.10                              # m per scan = a "gap"

RESULT_COLUMNS = [
    'dataset', 'config', 'seed', 'N', 'sigma', 'max_range', 'scan_matching',
    'n_scans', 'runtime_s', 'occ_cells', 'free_cells', 'occ_ratio', 'scan_agreement',
    'path_len', 'odom_path_len', 'path_ratio', 'mean_dev', 'final_dev',
    'max_step', 'n_jumps', 'mean_neff', 'resample_frac',
]


def config_id(cfg):
    return 'N{0}_sig{1:g}_r{2:g}_sm{3}'.format(
        int(cfg['N']), float(cfg['sigma']), float(cfg['max_range']),
        int(cfg['scan_matching']))


def build_configs(experiments):
    """Unique configurations needed for the selected experiments."""
    configs = []
    if 'baseline' in experiments:
        configs.append(dict(BASELINE))
    for name, param, values in SWEEPS:
        if name in experiments:
            for v in values:
                cfg = dict(DEFAULT)
                cfg[param] = v
                configs.append(cfg)
    unique = {}
    for cfg in configs:
        unique[config_id(cfg)] = cfg
    return [unique[k] for k in sorted(unique)]


def seeds_for(cfg, n_seeds):
    # sigma = 0 makes the run deterministic, so extra seeds are identical
    return [0] if float(cfg['sigma']) == 0.0 else list(range(n_seeds))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
_DATA_CACHE = {}


def load_dataset(dataset, data_dir='data'):
    if dataset not in _DATA_CACHE:
        joint = ld.get_joint(os.path.join(data_dir, 'train_joint%d' % dataset))
        lid = ld.get_lidar(os.path.join(data_dir, 'train_lidar%d' % dataset))
        _DATA_CACHE[dataset] = (lid, joint)
    return _DATA_CACHE[dataset]


def new_map():
    m = {'res': MAP_RES, 'xmin': -MAP_LIM, 'ymin': -MAP_LIM,
         'xmax': MAP_LIM, 'ymax': MAP_LIM}
    m['sizex'] = int(np.ceil((m['xmax'] - m['xmin']) / m['res'] + 1))
    m['sizey'] = int(np.ceil((m['ymax'] - m['ymin']) / m['res'] + 1))
    m['log_map'] = np.zeros((m['sizex'], m['sizey']))
    m['map'] = np.zeros((m['sizex'], m['sizey']), dtype=np.int8)
    m['show_map'] = 0.5 * np.ones((m['sizex'], m['sizey'], 3), dtype=np.int8)
    return m


# --------------------------------------------------------------------------
# One SLAM run
# --------------------------------------------------------------------------
def low_variance_resample(particles, weight):
    """Low-variance (systematic) resampling, as intended in SLAM.py."""
    n = particles.shape[0]
    positions = np.random.uniform(0, 1.0 / n) + np.arange(n) / float(n)
    cumsum = np.cumsum(weight.ravel())
    cumsum[-1] = 1.0                               # guard against round-off
    idx = np.searchsorted(cumsum, positions)
    return particles[idx].copy()


def run_slam(cfg, seed, lid, joint, max_scans=None, verbose=True, label=''):
    np.random.seed(seed)
    random.seed(seed)
    t_start = time.time()

    N = int(cfg['N'])
    sigma = float(cfg['sigma'])
    max_range = float(cfg['max_range'])
    window = WINDOW if int(cfg['scan_matching']) else np.array([0.0])
    n_threshold = NEFF_RATIO * N

    angles = np.array([np.arange(-135, 135.25, 0.25) * np.pi / 180.0])
    mapfig = new_map()
    x_im = np.arange(mapfig['xmin'], mapfig['xmax'] + mapfig['res'], mapfig['res'])
    y_im = np.arange(mapfig['ymin'], mapfig['ymax'] + mapfig['res'], mapfig['res'])

    particles = np.zeros((N, 3))
    weight = np.ones((N, 1)) / N
    pos_phy, posX_map, posY_map = {}, {}, {}

    ts = joint['ts']
    h_angle = joint['head_angles']
    rpy_robot = joint['rpy']

    timeline = len(lid) if max_scans is None else min(len(lid), int(max_scans))

    # first scan initialises the map
    lid_p = lid[0]
    ind_0 = np.argmin(np.absolute(ts - lid_p['t'][0][0]))
    pos_phy, posX_map, posY_map = mapConvert(
        lid_p['scan'], rpy_robot[:, ind_0], h_angle[:, ind_0], angles, particles,
        N, pos_phy, posX_map, posY_map, mapfig, max_range)
    mapfig = drawMap(particles[0, :], posX_map[0].astype(np.int32),
                     posY_map[0].astype(np.int32), mapfig)

    pose_p, yaw_p = lid_p['pose'], lid_p['rpy'][0, 2]
    pose_0, yaw_0 = np.array(lid_p['pose'][0][:2], dtype=float), yaw_p
    rot_0 = np.array([[np.cos(yaw_0), np.sin(yaw_0)],
                      [-np.sin(yaw_0), np.cos(yaw_0)]])

    traj = [np.zeros(2)]
    odom = [np.zeros(2)]
    neff_log = []
    resampled = []
    agreement = []

    for t in range(1, timeline):
        if verbose and t % 1000 == 0:
            print('  [{0}] scan {1}/{2}'.format(label, t, timeline))
            sys.stdout.flush()
        lid_c = lid[t]
        pose_c, yaw_c = lid_c['pose'], lid_c['rpy'][0, 2]

        # ---- prediction: odometry delta, applied in each particle's frame --
        dx_gb = pose_c[0][0] - pose_p[0][0]
        dy_gb = pose_c[0][1] - pose_p[0][1]
        dth = yaw_c - yaw_p
        dx_lc = np.cos(yaw_p) * dx_gb + np.sin(yaw_p) * dy_gb
        dy_lc = -np.sin(yaw_p) * dx_gb + np.cos(yaw_p) * dy_gb
        yaw_est = particles[:, 2]
        ut = np.column_stack([np.cos(yaw_est) * dx_lc - np.sin(yaw_est) * dy_lc,
                              np.sin(yaw_est) * dx_lc + np.cos(yaw_est) * dy_lc,
                              np.full(N, dth)])
        noise = np.random.normal(0.0, sigma, (N, 3)) * FACTOR   # FIX 2
        particles = particles + ut + noise

        # ---- update: map correlation with a local x/y search ---------------
        ind_i = np.argmin(np.absolute(ts - lid_c['t'][0][0]))
        pos_phy, posX_map, posY_map = mapConvert(
            lid_c['scan'], rpy_robot[:, ind_i], h_angle[:, ind_i], angles, particles,
            N, pos_phy, posX_map, posY_map, mapfig, max_range)

        corr = np.zeros((N, 1))
        for k in range(N):
            c = maput.mapCorrelation(mapfig['map'], x_im, y_im, pos_phy[k], window, window)
            ix, iy = np.unravel_index(np.argmax(c), c.shape)
            corr[k] = c[ix, iy]
            particles[k, 0] += window[ix]
            particles[k, 1] += window[iy]

        wtmp = np.log(weight) + corr          # raw correlation as log-likelihood
        wtmp = wtmp - wtmp.max()
        wtmp = wtmp - np.log(np.sum(np.exp(wtmp)))
        weight = np.exp(wtmp)
        ind_best = int(weight.argmax())
        n_pts = pos_phy[ind_best].shape[1]
        if n_pts:   # fraction of the scan that lands on already-occupied cells
            agreement.append(float(corr[ind_best, 0]) / n_pts)

        # ---- mapping with the best particle (as in SLAM.py) ----------------
        mapfig = drawMap(particles[ind_best, :], posX_map[ind_best].astype(np.int32),
                         posY_map[ind_best].astype(np.int32), mapfig)

        traj.append(particles[ind_best, :2].copy())
        odom.append(rot_0.dot(np.array(pose_c[0][:2], dtype=float) - pose_0))
        pose_p, yaw_p = pose_c, yaw_c

        # ---- resampling -----------------------------------------------------
        n_eff = 1.0 / np.sum(np.square(weight))
        neff_log.append(n_eff)
        if n_eff < n_threshold:                                 # FIX 1
            particles = low_variance_resample(particles, weight)
            weight = np.ones((N, 1)) / N
            resampled.append(1)
        else:
            resampled.append(0)

    runtime = time.time() - t_start
    traj = np.array(traj)
    odom = np.array(odom)
    metrics = compute_metrics(mapfig['log_map'], traj, odom, neff_log, resampled,
                              agreement)
    metrics['n_scans'] = timeline
    metrics['runtime_s'] = runtime
    ternary = np.sign(mapfig['log_map']).astype(np.int8)   # 1 occ, -1 free, 0 unknown
    extras = {'traj': traj, 'odom': odom, 'neff': np.array(neff_log),
              'resampled': np.array(resampled, dtype=np.int8), 'map': ternary}
    return metrics, extras


def compute_metrics(log_map, traj, odom, neff_log, resampled, agreement):
    occ = int(np.sum(log_map > 0))
    free = int(np.sum(log_map < 0))
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    odom_steps = np.linalg.norm(np.diff(odom, axis=0), axis=1)
    dev = np.linalg.norm(traj - odom, axis=1)
    path_len = float(np.sum(steps))
    odom_len = float(np.sum(odom_steps))
    return {
        'occ_cells': occ,
        'free_cells': free,
        'occ_ratio': occ / float(max(occ + free, 1)),
        'scan_agreement': float(np.mean(agreement)) if agreement else float('nan'),
        'path_len': path_len,
        'odom_path_len': odom_len,
        'path_ratio': path_len / odom_len if odom_len > 0 else float('nan'),
        'mean_dev': float(np.mean(dev)),
        'final_dev': float(dev[-1]),
        'max_step': float(np.max(steps)) if len(steps) else 0.0,
        'n_jumps': int(np.sum(steps > JUMP_THRESHOLD)),
        'mean_neff': float(np.mean(neff_log)) if neff_log else float('nan'),
        'resample_frac': float(np.mean(resampled)) if resampled else float('nan'),
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def mkdir_p(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def render_map(ax, ternary, traj=None, odom=None, title='', margin=40):
    """Grey = unknown, white = free, black = occupied; red = SLAM path,
    blue dashed = raw odometry. Cropped to the explored area."""
    img = np.full(ternary.shape, 0.5)
    img[ternary < 0] = 1.0
    img[ternary > 0] = 0.0
    known = np.argwhere(ternary != 0)
    if len(known):
        (x0, y0), (x1, y1) = known.min(0) - margin, known.max(0) + margin
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, ternary.shape[0] - 1), min(y1, ternary.shape[1] - 1)
    else:
        x0, y0, x1, y1 = 0, 0, ternary.shape[0] - 1, ternary.shape[1] - 1
    to_m = lambda i: -MAP_LIM + i * MAP_RES
    extent = [to_m(x0), to_m(x1), to_m(y0), to_m(y1)]
    ax.imshow(img[x0:x1 + 1, y0:y1 + 1].T, cmap='gray', vmin=0, vmax=1,
              origin='lower', extent=extent, interpolation='nearest')
    if odom is not None:
        ax.plot(odom[:, 0], odom[:, 1], 'b--', lw=0.7, label='odometry')
    if traj is not None:
        ax.plot(traj[:, 0], traj[:, 1], 'r-', lw=0.8, label='SLAM')
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def save_run(out_dir, dataset, cfg, seed, metrics, extras):
    run_dir = os.path.join(out_dir, 'runs', 'd%d' % dataset)
    mkdir_p(run_dir)
    stem = os.path.join(run_dir, '%s_seed%d' % (config_id(cfg), seed))
    np.savez_compressed(stem + '.npz', **extras)
    fig, ax = plt.subplots(figsize=(5, 5))
    render_map(ax, extras['map'], extras['traj'], extras['odom'],
               '%s  seed %d' % (config_id(cfg), seed))
    ax.legend(loc='lower right', fontsize=7)
    fig.savefig(stem + '.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def read_done(csv_path):
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            header = f.readline().strip().split(',')
            for line in f:
                row = dict(zip(header, line.strip().split(',')))
                if 'dataset' in row:
                    done.add((int(row['dataset']), row['config'], int(row['seed'])))
    return done


def append_row(csv_path, row):
    new = not os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if new:
            f.write(','.join(RESULT_COLUMNS) + '\n')
        f.write(','.join(str(row[c]) for c in RESULT_COLUMNS) + '\n')


# --------------------------------------------------------------------------
# Task execution
# --------------------------------------------------------------------------
def run_task(task):
    dataset, cfg, seed, out_dir, max_scans, data_dir = task
    label = 'd%d %s s%d' % (dataset, config_id(cfg), seed)
    try:
        lid, joint = load_dataset(dataset, data_dir)
        metrics, extras = run_slam(cfg, seed, lid, joint, max_scans, label=label)
        save_run(out_dir, dataset, cfg, seed, metrics, extras)
        row = dict(metrics)
        row.update(cfg)
        row.update({'dataset': dataset, 'config': config_id(cfg), 'seed': seed})
        return row
    except Exception:
        print('!! run failed: %s' % label)
        traceback.print_exc()
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiments', nargs='+', default=['all'],
                   choices=['all', 'baseline', 'particles', 'noise', 'range'])
    p.add_argument('--datasets', nargs='+', type=int, default=[0],
                   help='THOR training sequence(s), 0-3 (default: 0)')
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--out', default='result_experiment_sweeps')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--jobs', type=int, default=1, help='parallel runs')
    p.add_argument('--max-scans', type=int, default=None,
                   help='only process the first K scans (quick tests)')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    exps = ['baseline', 'particles', 'noise', 'range'] if 'all' in args.experiments \
        else args.experiments
    if args.max_scans is not None and args.out == 'result_experiment_sweeps':
        args.out = 'Result_Experiment_sweeps_quick'   # keep test runs apart
    mkdir_p(args.out)
    csv_path = os.path.join(args.out, 'results.csv')
    done = read_done(csv_path)

    tasks = []
    for d in args.datasets:
        for cfg in build_configs(exps):
            for s in seeds_for(cfg, args.seeds):
                if (d, config_id(cfg), s) not in done:
                    tasks.append((d, cfg, s, args.out, args.max_scans, args.data_dir))
    tasks.sort(key=lambda t: t[1]['N'])     # cheap runs first

    print('%d runs to do (%d already in %s)' % (len(tasks), len(done), csv_path))
    if args.dry_run:
        for t in tasks:
            print('  dataset %d  %-28s seed %d' % (t[0], config_id(t[1]), t[2]))
        return

    if args.jobs > 1:
        import multiprocessing
        pool = multiprocessing.Pool(args.jobs)
        results = pool.imap_unordered(run_task, tasks)
    else:
        results = (run_task(t) for t in tasks)

    for i, row in enumerate(results):
        if row is not None:
            append_row(csv_path, row)
            print('[%d/%d] done %s seed %s: %.0f s, agreement %.3f, jumps %d, mean N_eff %.2f'
                  % (i + 1, len(tasks), row['config'], row['seed'], row['runtime_s'],
                     row['scan_agreement'], row['n_jumps'], row['mean_neff']))
            sys.stdout.flush()
    print('Finished. Now run: python analyze.py --out %s' % args.out)


if __name__ == '__main__':
    main()
