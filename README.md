# Visibility-preserving pursuit-evasion experiments

This source release reproduces the pursuit-evasion computations on the four final
obstacle scenes A, B, C and D. It contains the proposed finite matrix-game policy
and five comparison pursuit laws: proportional navigation (PN), pure pursuit
(PP), constant bearing (CB), artificial potential field (APF) and Apollonius pursuit.
The evader computes its mixed strategy at the actual state reached by each method.

Only source code and the four static obstacle geometries are included. Initial
positions, trajectories, outcomes, timing measurements and summary tables are
generated when the commands below are run.

## Installation

Use Python 3.12. From this directory, create a virtual environment and install the
pinned numerical dependencies. In Windows PowerShell:

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\Activate.ps1
```

On Linux or macOS, use `.venv/bin/python -m pip install -r requirements.txt`
and `source .venv/bin/activate` after creating the environment. If activation is
unavailable, replace `python` in subsequent commands with the environment's Python
executable.
The module runs directly from this directory. Python, NumPy and SciPy handle the
experiment, while a compiled C kernel evaluates geometry and forecast scores.
Nashpy solves the two linear programs using SciPy/HiGHS.

Install Clang with LLD, GCC (MinGW on Windows), or Zig. Build the shared kernel:

```text
python -m pursuit_evasion build --output ../kernel
```

Use `--compiler /path/to/compiler` if the compiler is not on PATH. The build uses
`-O3 -ffp-contract=off`. It produces `forecast.dll` on Windows, `forecast.so` on
Linux and `forecast.dylib` on macOS. Replace the `.dll` extension in the commands
below on those platforms. The kernel source is included in this distribution.

## Reproduce the experiments

Run all six methods sequentially on 100 generated pairs in each of the four scenes.
Initial pairs are generated automatically when execution starts:

```text
python -m pursuit_evasion run --kernel ../kernel/forecast.dll --output ../experiment_results
```

Run one initial pair in one scene first:

```text
python -m pursuit_evasion run --kernel ../kernel/forecast.dll --scenes A --trials 1 --output ../single_pair
```

For proposed-method resource measurements, run each scene in a separate process
and output directory, for example:

```text
python -m pursuit_evasion run --kernel ../kernel/forecast.dll --scenes A --methods full --output ../proposed_resources_A
```

All output paths must lie outside this source release. A run records the selected
scenes, obstacle geometry, methods, trial count, seed settings and numerical
parameters in `protocol.json`. Repeating the same command resumes completed
records when these inputs match. A changed configuration requires a new output directory.
Computation failures are recorded and produce a nonzero exit code. Use a fresh
output directory to retry a failed run.

Recompute the summaries without running any games:

```text
python -m pursuit_evasion summarize --input ../experiment_results
```

The generated `summary.csv` and `summary.json` report captures, escapes, cutoffs,
incomplete computations, capture rates, conditional capture and escape times,
matched capture times, computation times and
process peak memory. Matched times use only initial pairs captured by all six
methods and are available after the six-method batch is complete. For a subset
of methods, `matched_comparison_available` is false and matched results are null.
Partial batches report missing games and leave capture rates unset.
The `runs` directory contains the complete trajectories and per-step controls
needed to draw result figures, together with equilibrium gaps and timing samples.

## Reproduce the figures

Install the plotting dependencies with `python -m pip install -r requirements-plot.txt`.
For the manuscript typography, install TeX with the `amsmath`, `newtxtext` and
`newtxmath` packages and put `latex`, `dvipng` and `kpsewhich` on PATH.

```text
python -m pursuit_evasion plot --input ../experiment_results --output ../figures --timing-inputs ../proposed_resources_A ../proposed_resources_B ../proposed_resources_C ../proposed_resources_D
```

Run the resource command separately for A, B, C and D to create those four timing
inputs. Without `--timing-inputs`, the timing plot uses the proposed-method
decisions in the six-method batch. Every time value is read from generated
records. `--text-engine native` renders without TeX, using Matplotlib fonts.

The output comprises four six-method trajectory figures (`Fig.3`--`Fig.6`),
capture counts (`Fig.7`), capture rates (`Fig.8`), three visibility comparisons
(`Fig.9`--`Fig.11`) and computation times (`Fig.12`), all as vector PDFs.
These are asset filenames; manuscript figure numbers are assigned by LaTeX.
The fixed illustrative trials are recorded in `figure_selection.json`:
A24, B31, C8 and D80 for trajectories; A72/CB, B61/PP and C1/APF for visibility.
They are illustrative cases, while the aggregate figures use all 100 trials in
each scene. The plot command reads these indices directly and performs no search.

## Initial positions and randomization

The default trial seeds are 261494 through 261593.
For seed `s`, the initial distance is drawn uniformly from 3 to 6 metres
using `SeedSequence([1, s])`. Position generation uses
`SeedSequence([2, s, scene_code])`, with codes A=1, B=10, C=18 and D=22.
The pursuer is drawn uniformly from `[-10,10]^2`, followed by a uniform bearing
for the evader at that fixed distance. A pair is rejected if the evader leaves
the initial-position window, either position is inside an obstacle, the line
of sight is obstructed, or its obstacle distance is at most `1e-8` metres.
Rejection redraws positions and bearing, keeping the distance and seed fixed.
This window does not constrain subsequent motion.

Both players use `SeedSequence(s).spawn(2)`, with the first stream for the pursuer
and the second for the evader. Each method starts from the same pair and random
streams. The baseline pursuer consumes its draw even though its control is
deterministic. Selected controls are saved in new run records. Using the same
dependency versions and numerical backend helps reproduce control selection
when linear programs have multiple optimal solutions.

## Mathematical computation

The fixed settings are `v_P=0.30 m/s`, `v_E=0.20 m/s`, `R=0.5 m`, `h=0.1 s`,
`H=20`, `Q=16`, control magnitudes `0, 0.5, 1`, `S=49`, `D=50 m`, distance-grid
spacing `0.025 m`, and equilibrium-gap tolerance `1e-7`.
In `config.py`, `PARAMETERS` contains physical parameters and the observation
cutoff; `Settings` contains the prediction and numerical settings. The distance
grid and strategy computation use the same `Settings` instance during a run.
PN uses navigation gain 4. APF uses influence distance `0.75 m`, gain `0.00625`
and distance floor `R/1e6`. The CB velocity estimate uses the preceding observation;
its initial estimate is zero. The Apollonius reference circle uses the initial pair.

At each observation:

1. Check straight predicted motions against polygon/ellipse geometry. Proposed
   pursuit and evasion controls must remain admissible for the prediction horizon
   `Hh`. Comparison pursuit controls are filtered over the executed interval `h`.
2. For each admissible prediction pair, evaluate separation and visibility at
   `0,h,...,Hh`. The first predicted capture/escape has cost `-1+t/(4Hh)` or
   `1-t/(4Hh)`. Otherwise, let `beta` be final separation minus the minimum
   sight score; the cost is `beta / (2 * (D + abs(beta)))`, which lies between
   `-0.5` and `0.5`. The score uses a padded obstacle distance grid and
   49 equally spaced points on the predicted line of sight.
3. Solve the zero-sum game, with the pursuer minimizing and evader maximizing.
   The proposed policy independently samples both distributions. Each baseline
   chooses the admissible discrete pursuit control with maximum projection onto
   its nominal direction; the evader samples the same full-game evasion rule.
   Thus the baseline directions are projected onto the 16-direction control
   alphabet, with legality checked over the executed interval.
4. Hold the selected pair for `h` or until termination. Polynomial roots partition
   the interval for capture-distance and visibility changes. Capture requires
   visibility and separation at most `R`; escape requires obstruction and
   separation greater than `R`. Capture takes precedence at a shared boundary.

The physical workspace is unbounded. Each of the three obstacles in a scene is
a polygon/ellipse union. Geometry predicates determine legality and visibility;
the raster field only evaluates the payoff score. An open escape interval is
recorded by its left endpoint together with an interior witness time and positions.
The 120-second cutoff applies only to runs that have not captured or escaped.
The numerical experiments implement the sampled policies; the separate sufficient
conditions in the paper determine when a capture guarantee applies.

## Measurements

`matrix_seconds` measures admissible action generation, payoff construction and equilibrium
solution for one observation, with `lp_seconds` separately measuring the linear
programs. The timed build also computes the forecast escape probabilities recorded
for the proposed policy. All timing samples are retained. Process peak memory includes imports,
distance-field construction and trajectory storage. All requested cases run
sequentially in one process, so the memory high-water mark is cumulative across
that invocation. Run one scene with `--methods full` per invocation to measure
the proposed method on that scene. Computation time and peak memory depend on
hardware, compiler and numerical libraries. The reported manuscript timings
correspond to the compiled kernel; the pure NumPy geometry routines provide the
underlying predicates and distance-field construction.

## License

This software is licensed under the [MIT License](LICENSE).
Copyright (c) 2026 The project contributors.
