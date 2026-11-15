#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze.py -- turn <out>/results.csv from experiments.py into tables and figures.

  python analyze.py                  # uses result_experiment_sweeps/
  python analyze.py --out results_quick

Writes to <out>/analysis/, per dataset <d> (0-3 for THOR, intel):
  summary_d<d>.csv          mean and std per configuration, per sweep
  table_<sweep>_d<d>.tex    booktabs table for the ICML template
  metrics_<sweep>_d<d>.png  metric vs parameter, mean +- std over seeds,
                            dashed line = odometry-only baseline
  maps_<sweep>_d<d>.png     final map of seed 0 for every value (+ baseline)

Map IoU = mean pairwise intersection-over-union of the occupied cells of the
final maps of different seeds of one configuration (1 = identical maps).

For the Intel dataset, if data/intel.relations exists (or --relations points to
it), the ground-truth relation errors of evaluate_relations.py are added as two
extra metrics (translational error in m, rotational error in deg), and
relations_intel.csv is written with one row per run.
"""
from __future__ import division, print_function

import argparse
import csv
import itertools
import math
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
RELATION_METRICS = [
    ('rel_trans', 'Relation error transl. (m)', True),
    ('rel_rot', 'Relation error rot. (deg)', True),
]
DIGITS = {'scan_agreement': 3, 'occ_ratio': 3, 'map_iou': 2, 'path_ratio': 2,
          'n_jumps': 1, 'mean_neff': 2, 'runtime_s': 0, 'rel_trans': 3, 'rel_rot': 2}
PARAM_LABEL = {'N': 'Number of particles $N$',
               'sigma': r'Motion noise $\sigma$',
               'max_range': 'Max. lidar range (m)'}
NUMERIC = ['seed', 'N', 'sigma', 'max_range', 'scan_matching',
           'n_scans', 'runtime_s', 'occ_cells', 'free_cells', 'occ_ratio', 'scan_agreement',
           'path_len', 'odom_path_len', 'path_ratio', 'mean_dev', 'final_dev',
           'max_step', 'n_jumps', 'mean_neff', 'resample_frac']


def dataset_title(dataset):
    return 'Intel Research Lab' if dataset == 'intel' else 'THOR sequence %s' % dataset


def load_rows(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['dataset'] = r['dataset'].strip()
        for k in NUMERIC:
            r[k] = float(r[k])
    return rows


def add_relation_errors(rows, out_dir, dataset, relations_path):
    import evaluate_relations as er
    per_run = er.evaluate_folder(out_dir, dataset, relations_path)
    path = os.path.join(out_dir, 'analysis', 'relations_%s.csv' % dataset)
    with open(path, 'w') as f:
        f.write(','.join(['config', 'seed'] + er.COLUMNS) + '\n')
        for (config, seed), r in sorted(per_run.items()):
            f.write(','.join([config, str(seed)] + [str(r[c]) for c in er.COLUMNS]) + '\n')
    for r in rows:
        if r['dataset'] == dataset:
            res = per_run.get((r['config'], int(r['seed'])), {})
            r['rel_trans'] = res.get('rel_trans', float('nan'))
            r['rel_rot'] = res.get('rel_rot', float('nan'))
    return len(per_run)


def matches(row, cfg):
    return all(np.isclose(row[k], float(v)) for k, v in cfg.items())


def map_iou(out_dir, dataset, cid, seeds):
    masks = []
    for s in seeds:
        path = os.path.join(out_dir, 'runs', 'd%s' % dataset, '%s_seed%d.npz' % (cid, s))
        if os.path.exists(path):
            masks.append(np.load(path)['map'] > 0)
    if len(masks) < 2:
        return float('nan')
    ious = []
    for a, b in itertools.combinations(masks, 2):
        union = np.logical_or(a, b).sum()
        ious.append(np.logical_and(a, b).sum() / float(union) if union else 1.0)
    return float(np.mean(ious))


def summarise(rows, out_dir, dataset, cfg, metrics):
    sel = [r for r in rows if r['dataset'] == dataset and matches(r, cfg)]
    if not sel:
        return None
    cid = config_id(cfg)
    s = {'config': cid, 'n_seeds': len(sel)}
    for col, _, _ in metrics:
        if col == 'map_iou':
            s[col + '_mean'] = map_iou(out_dir, dataset, cid, [int(r['seed']) for r in sel])
            s[col + '_std'] = float('nan')
        else:
            vals = np.array([r.get(col, float('nan')) for r in sel], dtype=float)
            s[col + '_mean'] = float(vals.mean())
            s[col + '_std'] = float(vals.std(ddof=1)) if len(vals) > 1 else float('nan')
    return s


def fmt(mean, std, digits):
    if np.isnan(mean):
        return '--'
    if np.isnan(std):
        return '%.*f' % (digits, mean)
    return r'%.*f $\pm$ %.*f' % (digits, mean, digits, std)


def write_tex(path, param, values, stats, base, metrics):
    with open(path, 'w') as f:
        f.write('\\begin{tabular}{l' + 'c' * len(metrics) + '}\n\\toprule\n')
        short = {'N': '$N$', 'sigma': r'$\sigma$', 'max_range': 'Range (m)'}[param]
        f.write(' & '.join([short] + [m[1] for m in metrics]) + ' \\\\\n\\midrule\n')
        lines = [('odometry only', base)] if base else []
        lines += [('%g' % v, st) for v, st in zip(values, stats) if st]
        for name, st in lines:
            cells = [fmt(st[c + '_mean'], st[c + '_std'], DIGITS[c]) for c, _, _ in metrics]
            f.write(' & '.join([name] + cells) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')


def _new_mpl():
    major, minor = [int(x) for x in matplotlib.__version__.split('.')[:2]]
    return (major, minor) >= (3, 3)


def plot_metrics(path, param, values, stats, base, title, metrics):
    ncols = 4
    nrows = int(math.ceil(len(metrics) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows))
    axes = axes.ravel()
    for ax in axes[len(metrics):]:
        ax.axis('off')
    for ax, (col, label, cmp_base) in zip(axes, metrics):
        xs = [v for v, st in zip(values, stats) if st]
        m = [st[col + '_mean'] for st in stats if st]
        e = [0 if np.isnan(st[col + '_std']) else st[col + '_std'] for st in stats if st]
        ax.errorbar(xs, m, yerr=e, marker='o', capsize=3)
        if cmp_base and base and not np.isnan(base[col + '_mean']):
            ax.axhline(base[col + '_mean'], ls='--', c='grey', label='odometry only')
            ax.legend(fontsize=7)
        if param == 'N':
            ax.set_xscale('log')
        elif param == 'sigma':
            if _new_mpl():
                ax.set_xscale('symlog', linthresh=1e-4)
            else:
                ax.set_xscale('symlog', linthreshx=1e-4)
            ax.set_xlim(-3e-5, max(values) * 2)
        ax.set_xlabel(PARAM_LABEL[param])
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_maps(path, out_dir, dataset, param, values, title):
    panels = [('odometry only', BASELINE)]
    for v in values:
        cfg = dict(DEFAULT)
        cfg[param] = v
        panels.append(('%s = %g' % (param, v), cfg))
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.4))
    for ax, (name, cfg) in zip(axes, panels):
        f = os.path.join(out_dir, 'runs', 'd%s' % dataset, '%s_seed0.npz' % config_id(cfg))
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
    p.add_argument('--out', default='result_experiment_sweeps')
    p.add_argument('--relations', default=os.path.join('data', 'intel.relations'),
                   help='relations file for the Intel dataset')
    args = p.parse_args()

    rows = load_rows(os.path.join(args.out, 'results.csv'))
    adir = os.path.join(args.out, 'analysis')
    mkdir_p(adir)

    datasets = sorted(set(r['dataset'] for r in rows))
    for dataset in datasets:
        metrics = list(METRICS)
        if dataset == 'intel':
            if os.path.exists(args.relations):
                n = add_relation_errors(rows, args.out, dataset, args.relations)
                print('Ground-truth relation errors computed for %d Intel runs' % n)
                metrics = RELATION_METRICS + metrics
            else:
                print('Note: %s not found, Intel results without ground-truth errors'
                      % args.relations)

        base = summarise(rows, args.out, dataset, BASELINE, metrics)
        summary_rows = [dict(base, sweep='baseline', value='')] if base else []
        for name, param, values in SWEEPS:
            stats = []
            for v in values:
                cfg = dict(DEFAULT)
                cfg[param] = v
                stats.append(summarise(rows, args.out, dataset, cfg, metrics))
            if not any(stats):
                continue
            for v, st in zip(values, stats):
                if st:
                    summary_rows.append(dict(st, sweep=name, value=v))
            tag = '%s_d%s' % (name, dataset)
            title = '%s sweep, %s' % (name, dataset_title(dataset))
            write_tex(os.path.join(adir, 'table_%s.tex' % tag), param, values, stats,
                      base, metrics)
            plot_metrics(os.path.join(adir, 'metrics_%s.png' % tag), param, values,
                         stats, base, title, metrics)
            plot_maps(os.path.join(adir, 'maps_%s.png' % tag), args.out, dataset,
                      param, values, title)

        cols = ['sweep', 'value', 'config', 'n_seeds'] + \
            [c + s for c, _, _ in metrics for s in ('_mean', '_std')]
        with open(os.path.join(adir, 'summary_d%s.csv' % dataset), 'w') as f:
            f.write(','.join(cols) + '\n')
            for r in summary_rows:
                f.write(','.join(str(r[c]) for c in cols) + '\n')

        has_rel = metrics[0][0] == 'rel_trans'
        print('\n' + dataset_title(dataset))
        head = '%-10s %-26s %5s %10s %8s %8s %8s %8s' % (
            'sweep', 'config', 'seeds', 'agreement', 'IoU', 'path_r', 'jumps', 'N_eff')
        print(head + ('  %9s %9s' % ('err (m)', 'err (deg)') if has_rel else ''))
        for r in summary_rows:
            line = '%-10s %-26s %5d %10.3f %8.2f %8.2f %8.1f %8.2f' % (
                r['sweep'], r['config'], r['n_seeds'], r['scan_agreement_mean'],
                r['map_iou_mean'], r['path_ratio_mean'], r['n_jumps_mean'],
                r['mean_neff_mean'])
            if has_rel:
                line += '  %9.3f %9.2f' % (r['rel_trans_mean'], r['rel_rot_mean'])
            print(line)
    print('\nFigures and tables written to %s' % adir)


if __name__ == '__main__':
    main()
