#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intel_data.py -- load the Intel Research Lab dataset (CARMEN .clf log) into the
same structures that load_data.get_lidar / get_joint produce for THOR, so the
SLAM code can run on it unchanged.

Download both files from
  http://ais.informatik.uni-freiburg.de/slamevaluation/datasets.php
and put them in the data/ folder:
  data/intel.clf          raw log (laser + odometry)
  data/intel.relations    manually verified relative poses (ground truth)

Only FLASER lines are used:
  FLASER n r_1 ... r_n  x y theta  odom_x odom_y odom_theta  ipc_timestamp host logger_timestamp
with n = 180 beams covering -90 deg .. +89 deg in 1 deg steps.

Mapping onto the THOR structures:
  lid[k]['t']     [[ipc_timestamp]]
  lid[k]['pose']  [[odom_x, odom_y, odom_theta]]     raw odometry
  lid[k]['rpy']   [[0, 0, odom_theta]]               yaw only (planar robot)
  lid[k]['scan']  [[r_1 ... r_n]]
  joint['ts']           all scan timestamps (1 x K)
  joint['rpy']          zeros (3 x K): no body roll or pitch
  joint['head_angles']  zeros (2 x K): laser fixed and level
With zero head angles and zero roll/pitch, convertFrame in SLAM_helper.py only
lifts every point by its fixed THOR mounting height (about 1.4 m), so the
ground filter (z > 0.1 m) keeps all beams and the 2D geometry is unchanged.
"""
from __future__ import division, print_function

import numpy as np


def intel_angles(n_beams=180):
    """Beam angles in rad, shape (1, n): -90 deg .. +89 deg in 1 deg steps."""
    return np.array([np.deg2rad(-90.0 + np.arange(n_beams) * (180.0 / n_beams))])


def load_clf(path):
    """Parse a CARMEN log. Returns (lid, joint, angles)."""
    lid, times = [], []
    n_beams = None
    with open(path) as f:
        for line in f:
            tok = line.split()
            if len(tok) < 3 or tok[0] != 'FLASER':
                continue
            n = int(tok[1])
            if len(tok) < n + 9:
                continue                                  # truncated line
            if n_beams is None:
                n_beams = n
            elif n != n_beams:
                continue                                  # inconsistent scan size
            ranges = np.array([float(v) for v in tok[2:2 + n]])
            ox, oy, oth = [float(v) for v in tok[n + 5:n + 8]]
            t = float(tok[n + 8])
            lid.append({'t': np.array([[t]]),
                        'pose': np.array([[ox, oy, oth]]),
                        'rpy': np.array([[0.0, 0.0, oth]]),
                        'scan': np.array([ranges])})
            times.append(t)
    if not lid:
        raise ValueError('No FLASER lines found in %s' % path)
    k = len(lid)
    joint = {'ts': np.array([times]),
             'rpy': np.zeros((3, k)),
             'head_angles': np.zeros((2, k))}
    return lid, joint, intel_angles(n_beams)


def load_relations(path):
    """Parse a relations file of the Freiburg SLAM benchmark (Kuemmerle et al.
    2009). Each line: t1 t2 x y z roll pitch yaw, the true pose of the scan at
    t2 expressed in the frame of the scan at t1. Returns an (M, 5) array with
    columns t1, t2, dx, dy, dyaw."""
    rows = []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if len(tok) < 8:
                continue
            try:
                v = [float(x) for x in tok[:8]]
            except ValueError:
                continue                                  # header or comment
            rows.append([v[0], v[1], v[2], v[3], v[7]])
    if not rows:
        raise ValueError('No relations found in %s' % path)
    return np.array(rows)
