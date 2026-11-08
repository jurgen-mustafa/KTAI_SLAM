#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze.py -- turn <out>/results.csv from experiments.py into tables and figures.

  python analyze.py                  # uses results/
  python analyze.py --out results_quick

Writes to <out>/analysis/:
  summary_d<k>.csv          mean and std per configuration, per sweep
  table_<sweep>_d<k>.tex    booktabs table for the ICML template
  metrics_<sweep>_d<k>.png  metric vs parameter, mean +- std over seeds,
                            dashed line = odometry-only baseline
  maps_<sweep>_d<k>.png     final map of seed 0 for every value (+ baseline)

Map IoU = mean pairwise intersection-over-union of the occupied cells of the
final maps of different seeds of one configuration (1 = identical maps).
"""
from __future__ import division, print_function

import argparse
import csv
import itertools
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments import (BASELINE, DEFAULT, SWEEPS, config_id, mkdir_p,
                         render_map)

METRICS = [  # column, label, compare with baseline?
    ('scan_agreement', 'Scan-to-map agreement', True),
    ('map_iou', 'Map IoU across seeds', False),
    ('path_ratio', 'Path length / odometry', True),
    ('n_jumps', 'Jumps > 10 cm per run', True),
    ('mean_neff', 'Mean $N_{eff}$', False),
    ('runtime_s', 'Runtime (s)', True),
    ('occ_ratio', 'Occupied / known cells', True),
]
PARAM_LABEL = {'N': 'Number of particles $N$',
               'sigma': r'Motion noise $\sigma$',
               'max_range': 'Max. lidar range (m)'}
NUMERIC = ['dataset', 'seed', 'N', 'sigma', 'max_range', 'scan_matching',
           'n_scans', 'runtime_s', 'occ_cells', 'free_cells', 'occ_ratio', 'scan_agreement',
           'path_len', 'odom_path_len', 'path_ratio', 'mean_dev', 'final_dev',
           'max_step', 'n_jumps', 'mean_neff', 'resample_frac']


def load_rows(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in NUMERIC:
            r[k] = float(r[k])
    return rows


def matches(row, cfg):
    return all(np.isclose(row[k], float(v)) for k, v in cfg.items())


def map_iou(out_dir, dataset, cid, seeds):
    masks = []
    for s in seeds:
        path = os.path.join(out_dir, 'runs', 'd%d' % dataset, '%s_seed%d.npz' % (cid, s))
        if os.path.exists(path):
            masks.append(np.load(path)['map'] > 0)
    if len(masks) < 2:
        return float('nan')
    ious = []
    for a, b in itertools.combinations(masks, 2):
        union = np.logical_or(a, b).sum()
        ious.append(np.logical_and(a, b).sum() / float(union) if union else 1.0)
    return float(np.mean(ious))


def summarise(rows, out_dir, dataset, cfg):
    sel = [r for r in rows if r['dataset'] == dataset and matches(r, cfg)]
    if not sel:
        return None
    cid = config_id(cfg)
    s = {'config': cid, 'n_seeds': len(sel)}
    for col, _, _ in METRICS:
        if col == 'map_iou':
            s[col + '_mean'] = map_iou(out_dir, dataset, cid, [int(r['seed']) for r in sel])
            s[col + '_std'] = float('nan')
        else:
            vals = np.array([r[col] for r in sel])
            s[col + '_mean'] = float(vals.mean())
            s[col + '_std'] = float(vals.std(ddof=1)) if len(vals) > 1 else float('nan')
    return s


def fmt(mean, std, digits):
    if np.isnan(mean):
        return '--'
    if np.isnan(std):
        return '%.*f' % (digits, mean)
    return r'%.*f $\pm$ %.*f' % (digits, mean, digits, std)


def write_tex(path, param, values, stats, base):
    digits = {'scan_agreement': 3, 'occ_ratio': 3, 'map_iou': 2, 'path_ratio': 2, 'n_jumps': 1,
              'mean_neff': 2, 'runtime_s': 0}
    with open(path, 'w') as f:
        f.write('\\begin{tabular}{l' + 'c' * len(METRICS) + '}\n\\toprule\n')
        short = {'N': '$N$', 'sigma': r'$\sigma$', 'max_range': 'Range (m)'}[param]
        f.write(' & '.join([short] + [m[1] for m in METRICS]) + ' \\\\\n\\midrule\n')
        lines = [('odometry only', base)] if base else []
        lines += [('%g' % v, st) for v, st in zip(values, stats) if st]
        for name, st in lines:
            cells = [fmt(st[c + '_mean'], st[c + '_std'], digits[c]) for c, _, _ in METRICS]
            f.write(' & '.join([name] + cells) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')


def plot_metrics(path, param, values, stats, base, title):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes.ravel()[-1].axis('off')
    for ax, (col, label, cmp_base) in zip(axes.ravel(), METRICS):
        xs = [v for v, st in zip(values, stats) if st]
        m = [st[col + '_mean'] for st in stats if st]
        e = [0 if np.isnan(st[col + '_std']) else st[col + '_std'] for st in stats if st]
        ax.errorbar(xs, m, yerr=e, marker='o', capsize=3)
        if cmp_base and base:
            ax.axhline(base[col + '_mean'], ls='--', c='grey', label='odometry only')
            ax.legend(fontsize=7)
        if param == 'N':
            ax.set_xscale('log')
        elif param == 'sigma':
            ax.set_xscale('symlog', linthresh=1e-4) if _new_mpl() \
                else ax.set_xscale('symlog', linthreshx=1e-4)
            ax.set_xlim(-3e-5, max(values) * 2)
        ax.set_xlabel(PARAM_LABEL[param])
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _new_mpl():
    major, minor = [int(x) for x in matplotlib.__version__.split('.')[:2]]
    return (major, minor) >= (3, 3)


def plot_maps(path, out_dir, dataset, param, values, title):
    panels = [('odometry only', BASELINE)]
    for v in values:
        cfg = dict(DEFAULT)
        cfg[param] = v
        panels.append(('%s = %g' % (param, v), cfg))
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.4))
    for ax, (name, cfg) in zip(axes, panels):
        f = os.path.join(out_dir, 'runs', 'd%d' % dataset, '%s_seed0.npz' % config_id(cfg))
        if os.path.exists(f):
            d = np.load(f)
            render_map(ax, d['map'], d['traj'], d['odom'], name)
        else:
            ax.set_title(name + '\n(missing)', fontsize=8)
            ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='results')
    args = p.parse_args()

    rows = load_rows(os.path.join(args.out, 'results.csv'))
    adir = os.path.join(args.out, 'analysis')
    mkdir_p(adir)

    for dataset in sorted(set(int(r['dataset']) for r in rows)):
        base = summarise(rows, args.out, dataset, BASELINE)
        summary_rows = [dict(base, sweep='baseline', value='')] if base else []
        for name, param, values in SWEEPS:
            stats = []
            for v in values:
                cfg = dict(DEFAULT)
                cfg[param] = v
                stats.append(summarise(rows, args.out, dataset, cfg))
            if not any(stats):
                continue
            for v, st in zip(values, stats):
                if st:
                    summary_rows.append(dict(st, sweep=name, value=v))
            tag = '%s_d%d' % (name, dataset)
            title = '%s sweep, THOR sequence %d' % (name, dataset)
            write_tex(os.path.join(adir, 'table_%s.tex' % tag), param, values, stats, base)
            plot_metrics(os.path.join(adir, 'metrics_%s.png' % tag), param, values,
                         stats, base, title)
            plot_maps(os.path.join(adir, 'maps_%s.png' % tag), args.out, dataset,
                      param, values, title)

        cols = ['sweep', 'value', 'config', 'n_seeds'] + \
            [c + s for c, _, _ in METRICS for s in ('_mean', '_std')]
        with open(os.path.join(adir, 'summary_d%d.csv' % dataset), 'w') as f:
            f.write(','.join(cols) + '\n')
            for r in summary_rows:
                f.write(','.join(str(r[c]) for c in cols) + '\n')

        print('\nTHOR sequence %d' % dataset)
        print('%-10s %-26s %5s %10s %8s %8s %8s %8s' % (
            'sweep', 'config', 'seeds', 'agreement', 'IoU', 'path_r', 'jumps', 'N_eff'))
        for r in summary_rows:
            print('%-10s %-26s %5d %10.3f %8.2f %8.2f %8.1f %8.2f' % (
                r['sweep'], r['config'], r['n_seeds'], r['scan_agreement_mean'],
                r['map_iou_mean'], r['path_ratio_mean'], r['n_jumps_mean'],
                r['mean_neff_mean']))
    print('\nFigures and tables written to %s' % adir)
