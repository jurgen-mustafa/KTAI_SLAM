# KTAI SLAM — Stress-testing particle-filter SLAM

Group project for **Key Topics in AI** (Assignment 3, Open Universiteit). We take the particle-filter SLAM implementation from [yashv28/Particle-Filter-SLAM](https://github.com/yashv28/Particle-Filter-SLAM), fix two bugs, and study how it behaves under controlled parameter changes on two datasets:

- **THOR** humanoid lidar data (the four sequences shipped with the original repo, no ground truth)
- **Intel Research Lab** from the [Freiburg SLAM benchmark](http://ais.informatik.uni-freiburg.de/slamevaluation/datasets.php), evaluated against manually verified ground-truth relations (Kümmerle et al., 2009)

## Research questions

| | Question | Values tested |
|---|---|---|
| RQ1 | How does the **number of particles** affect the quality and consistency of the map? | N = 1, 10, 30, 100, 300 |
| RQ2 | How does the **motion noise** affect the estimated trajectory and the map? | σ = 0, 1e-4, 1e-3, 1e-2, 5e-2 |
| RQ3 | How robust is the model to a **reduced lidar range**? (stress test) | 30, 15, 8, 4, 2 m |

Each sweep changes one parameter and keeps the others at their defaults (N = 100, σ = 1e-3, range 30 m, map resolution 0.05 m, ±5 cm scan-matching window). All sweeps are compared with an **odometry-only baseline** (N = 1, σ = 0, no scan matching). Every configuration runs with 5 random seeds, which gives 62 runs per dataset.

## Repository structure

| File | Purpose |
|---|---|
| `SLAM.py` | Original single-run SLAM script, extended with `--particles`, `--noise` and `--max-range` options |
| `SLAM_helper.py` | Frame transforms, scan-to-map conversion and map update; the lidar range is now an argument (`arg_max`) |
| `MapUtils.py`, `load_data.py` | Unchanged from the original repo (map correlation, THOR data loading) |
| `experiments.py` | Runs the baseline and all three sweeps; writes metrics, maps and raw trajectories per run |
| `analyze.py` | Turns the results into LaTeX tables, metric plots and map comparison figures |
| `intel_data.py` | Loads the Intel CARMEN log (`intel.clf`) and ground-truth relations (`intel.relations`) |
| `evaluate_relations.py` | Computes the ground-truth translational and rotational error for Intel runs |
| `test_experiments.py` | Smoke test on simulated data; checks the whole pipeline without the real datasets |
| `data/` | Datasets (see below) |

## Requirements

The code targets **Python 2.7**, as in the original repo. Dependencies:

- numpy
- scipy
- matplotlib
- opencv-python (cv2)

The easiest setup is a conda environment:

```bash
conda create -n pfslam python=2.7 numpy scipy matplotlib
conda activate pfslam
pip install opencv-python==4.2.0.32    # last OpenCV release with Python 2.7 wheels
```

## Getting the data

All files go in the `data/` folder.

**THOR** (`train_joint0–3.mat`, `train_lidar0–3.mat`): these are stored with Git LFS in the original repo. If you cloned that repo, fetch them with:

```bash
git lfs install
git lfs pull
```

Check that the files are real: `train_lidar0.mat` should be about 30 MB, not ~130 bytes. A tiny file is an LFS placeholder and causes `ValueError: Unknown mat file type`. `git lfs ls-files` marks downloaded files with `*` and placeholders with `-`.

**Intel Research Lab**:

```bash
cd data
wget http://ais.informatik.uni-freiburg.de/slamevaluation/datasets/intel.clf
wget http://ais.informatik.uni-freiburg.de/slamevaluation/datasets/intel.relations
cd ..
```

## Quick check

Run the smoke test first. It simulates a small building in both data formats and runs a reduced version of every sweep, so it needs no real data (about 5–10 minutes):

```bash
python test_experiments.py          # THOR format and Intel format
python test_experiments.py intel    # Intel format only
```

It ends with `Smoke test passed` and writes its output to `results_test/`.

## Running the experiments

Run all commands from the repository root.

**THOR** (we used sequence 2):

```bash
python experiments.py --datasets 2 --out result_experiment_sweeps
python analyze.py --out result_experiment_sweeps
```

**Intel**, in parallel in the background:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nohup python experiments.py --datasets intel --jobs 12 --out results_experiment_intel > intel_log.txt 2>&1 &
tail -f intel_log.txt                                     # follow progress
python analyze.py --out results_experiment_intel          # when all runs are done
```

Set `--jobs` to the number of cores you want to use, keeping in mind that each run needs about 200–400 MB of memory. The `export` line stops numpy from starting extra threads in each job. To stop a parallel run, use `pkill -f experiments.py`.

**Useful options for `experiments.py`:**

| Option | Effect |
|---|---|
| `--dry-run` | List the runs without executing them |
| `--experiments baseline particles noise range` | Run only some of the sweeps (useful to split work across machines) |
| `--seeds 5` | Number of random seeds per configuration |
| `--max-scans 500` | Only use the first scans; quick timing test, written to a separate `_quick` folder |
| `--jobs 4` | Number of runs in parallel |
| `--out <folder>` | Output folder |

Runs are resumable: finished runs are stored in `results.csv`, and restarting the same command skips them.

**Single run with the original script** (opens a map window at the end):

```bash
python SLAM.py --particles 100 --noise 0.001 --max-range 30
```

Note that `SLAM.py` still contains the original resampling and noise behaviour (see below), so its maps are not directly comparable with the sweep results. All results in the report come from `experiments.py`.

**Approximate runtimes:** a full THOR sweep takes a few hours sequentially; the Intel sweep takes about 8–15 hours sequentially or 1–4 hours in parallel, depending on the number of jobs. Runs with N = 300 dominate the total.

## Output

```
<out folder>/
├── results.csv                 one row per run with all metrics
├── runs/d<dataset>/            per run: map image (.png) and raw data (.npz: map, trajectories, N_eff, timestamps)
└── analysis/
    ├── summary_d<dataset>.csv      mean and std per configuration
    ├── table_<sweep>_d<dataset>.tex  LaTeX table (booktabs) for the report
    ├── metrics_<sweep>_d<dataset>.png  metric vs. parameter, baseline as dashed line
    ├── maps_<sweep>_d<dataset>.png     final maps side by side
    └── relations_intel.csv         ground-truth errors per run (Intel only)
```

**Metrics:**

| Metric | Meaning |
|---|---|
| Scan-to-map agreement | Fraction of each new scan that lands on already-mapped walls (higher = more consistent) |
| Map IoU across seeds | Overlap of the occupied cells between seeds (1 = identical maps) |
| Path length / odometry | Ratio of estimated path length to odometry path length (jitter) |
| Jumps > 10 cm | Number of steps where the estimate jumps more than 10 cm |
| Mean N_eff | Effective number of particles (weight degeneracy) |
| Relation error (Intel) | Mean translational (m) and rotational (deg) error against ground truth |

## Changes to the original code

`experiments.py` uses the original `mapConvert`, `drawMap` and `mapCorrelation` unchanged, with these modifications to the SLAM loop:

1. **Resampling fix.** In the original, the line copying a particle sits outside the resampling loop, so all particles but one are reset to (0, 0, 0). We use a correct low-variance resampler.
2. **Independent motion noise.** The original draws one noise sample per particle and copies it to x, y and yaw, so particles only spread along a diagonal. We draw an independent sample per dimension (still scaled by [1, 1, 10]).
3. **Resampling threshold scales with N** (N_eff < 0.35·N), so it means the same for every particle count.
4. **Lidar range** is passed to `mapConvert` as `arg_max`.
5. **int32 contour points** before `drawMap`, because newer OpenCV versions reject the original int16 indices.
6. **Configurable beam angles**, so the same loop handles THOR (270°, 1081 beams) and Intel (180°, 180 beams).

## Known limitations

- THOR has no ground truth, so THOR results rely on proxy metrics.
- The particle weights use the raw map correlation as log-likelihood, which makes them nearly one-hot; more particles therefore help less than expected.
- Scan matching only searches x/y offsets, never heading, and a single shared map is updated by the best particle (no per-particle maps as in FastSLAM).
- Motion noise σ is applied per scan, so σ values are not directly comparable between datasets with different scan rates.

## Credits

- SLAM implementation: [yashv28/Particle-Filter-SLAM](https://github.com/yashv28/Particle-Filter-SLAM)
- Intel Research Lab log: Dirk Hähnel; ground-truth relations: [Freiburg SLAM benchmark](http://ais.informatik.uni-freiburg.de/slamevaluation/datasets.php), Kümmerle et al., "On measuring the accuracy of SLAM algorithms", *Autonomous Robots* 27(4), 2009
