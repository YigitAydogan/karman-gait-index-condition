# Entrainment versus station keeping: an index condition for Kármán gaiting

Code and saved outputs for the manuscript

> Y. Aydoğan and O. Akanyeti, "Entrainment versus station keeping: an index condition for Kármán gaiting", submitted to *Chaos: An Interdisciplinary Journal of Nonlinear Science*.

The study is entirely computational. A Stuart-Landau central pattern generator (CPG) drives a pitching body that moves along a prescribed von Kármán wake. The scripts compute passive equilibria, entrainment at fixed position, station-holding orbits with free position, their Floquet multipliers and fixed-position branches, numerical continuation and bifurcations, noise and the experimental predictions, the mechanism and flow-speed analyses, and the figures and tables of the paper.

## Repository layout

```
scripts/                                 Python scripts, run in numerical order
outputs/karman_gait_project/             every saved output (results, checkpoints, summaries, tables, figures)
outputs/karman_gait_project/manuscript_figures/   the twelve figures exactly as they appear in the manuscript
MANIFEST.sha256                          SHA-256 checksum of every file in outputs/
requirements.txt                         Python packages
CITATION.cff, .zenodo.json               citation and archive metadata
```

Each script writes to its own folder, `outputs/karman_gait_project/<script name>/`, which holds its results (`.npz`, `.json`), a `summary_*.json` with the key numbers and, for the long runs, a `checkpoints_full/` folder.

## Scripts and where their results appear

Section, figure and table numbers refer to the submitted manuscript.

| Script | Content | Manuscript |
|---|---|---|
| 01 `passive_baseline` | Passive body: equilibria, stability, time to loss | Sec. III A, Fig. 2 |
| 02 `passive_robustness` | Passive equilibria over a Latin hypercube sample of four wake families | Sec. III A, Fig. 2 |
| 02b `passive_cases` | Floquet recheck of the passive equilibria | Sec. III A |
| 03 `cpg_tongues` | Entrainment at fixed position, Arnold tongues, node and saddle locks | Sec. III B, Fig. 3 |
| 04, 04b `free_position`, `locked_exceptions` | First tests with free position | Development, superseded by 05 to 10 |
| 05, 05b, 05c `holding_tongue`, `holding_dense`, `exceptions_chaos` | Station-holding orbits on dense grids, tests for chaos | Secs. IV and V A |
| 06, 06c `robustness`, `body_aL` | Body, CPG and wake parameter variants | Sec. V B, Table VII |
| 07 `index_condition` | First verification of the index condition | Development, superseded by 08 |
| 08 `homotopy` | Fixed-position branch by homotopy, index test on parameter sets | Sec. IV E, Table III, Fig. 6 |
| 09 `model_structure` | 16 model variants and 20 joint parameter samples | Secs. IV E and V B, Tables III and VI, Fig. 6; orbits of Fig. 8 |
| 10 `continuation` | Pseudo-arclength continuation, folds, torus bifurcations, areas | Secs. IV E and V A, Tables III and IV, Figs. 5 to 7; orbits of Fig. 12 |
| 11 `noise_predictions` | Reaching the station, noise, phase signature, lateral line, flow speed | Sec. VI, Tables VIII to X, Fig. 9 |
| 12 `verification` | Convergence, exact versus finite-difference multipliers, recomputation | Sec. IV E, Appendix C, Table XII |
| 13 `figures` | Figures and tables from the saved outputs | Figs. 1 to 3, 5, 7, 9 and 10, Tables I to XII |
| 14 `referee_analyses` | Exclusive areas, orbit accounting, passive bounds, residence bouts, phase discrimination, hypothesis margins, phase-lag sweep, confinement, stochastic time step | Secs. III A, IV E, V and VI, Tables III to V, VIII and IX |
| 15 `collect_bundle` | Collects all outputs into one archive | Repository |
| 16 `branch_change_check` | Identifies the single branch change of the convergence check | Appendix C, Table XII |
| 17 `biology_questions` | Fate of passive bodies, position of stable holding relative to the vortex peak, control in uniform flow | Secs. VI F and VI G, Fig. 10, Table XI |
| 18 `examples_and_locked_branches` | Trajectories of passive bodies and the two locked rhythms at fixed position | Secs. III A and III B, Figs. 2 and 3 |
| 19 `cfd_background` | Illustrative lattice Boltzmann simulation of the cylinder wake (not part of the model) | Fig. 1a |
| 20 `panel_a_cylinder_cfd` | Cylinder-wake illustration aligned to the drawing of Fig. 1a | Fig. 1a |
| 21 `mechanism_power_heave` | Wake cues held fixed one at a time, mechanical actuation proxy, heave sweep | Secs. V B and VI G, Fig. 11 |
| 22 `sensing_phase_speed_passive` | Sensing under noise, phase signature across phase lags, flow speed with retuning, passive bodies under noise | Secs. III A, VI H, VI I and VI J, Fig. 11 |
| 23 `adaptive_rhythm` | Phase-locked adaptive rhythm and retuning across flow speeds | Sec. VI J |
| 24 `final_figures` | Figures added or updated at a late stage of the work | Figs. 4, 6, 8, 11 and 12 |

Scripts 04, 04b and 07 are kept for completeness; their results were superseded by later scripts and are not used in the manuscript.

### Figures

`outputs/karman_gait_project/manuscript_figures/` holds the twelve figures exactly as they appear in the manuscript. Figures 4, 6, 8, 11 and 12 are produced by Script 24, and Figures 5 and 7 by Script 13. Figures 1, 2, 3, 9 and 10 include panels that were assembled at a late stage of the work from the saved outputs of Scripts 14, 17, 18, 19 and 20; Script 13 produces their earlier versions, and the final versions are provided as files.

## Running the scripts

The scripts were written for Google Colab. Each one is self-contained: paste it into a single cell and run it. They read and write `MyDrive/karman_gait_project/` on Google Drive; outside Colab they use `./karman_gait_project/` relative to the working directory.

* Scripts 01 to 09, 13, 20 and 24 need only NumPy and Matplotlib (Script 20 also SciPy, Pillow and a C compiler) and run on a CPU.
* Scripts 10, 11, 12, 14, 16 to 19 and 21 to 23 use JAX and were run on an NVIDIA L4 GPU. They also run on a CPU, more slowly.
* Every long script saves checkpoints. If a run stops, running the script again resumes it.
* Most scripts have a `MODE` setting: `"FULL"` reproduces the manuscript, `"QUICK"` is a short test.

Later scripts read the outputs of earlier ones, so a full reproduction runs them in numerical order. Runtimes are given in the header of each script.

### Reproducing the figures without rerunning the simulations

All saved outputs are included, so the figures and tables can be rebuilt directly:

```bash
pip install -r requirements.txt
cd outputs
python ../scripts/script13_figures.py
python ../scripts/script24_final_figures.py
```

The outputs are written to `outputs/karman_gait_project/script13_figures/` and `outputs/karman_gait_project/script24_final_figures/`.

To rerun any single analysis in Colab, copy `outputs/karman_gait_project/` to `MyDrive/karman_gait_project/` first, so that the script finds the outputs of the scripts it depends on.

## Checking file integrity

```bash
sha256sum -c MANIFEST.sha256
```

## Licence

The code in `scripts/` is released under the MIT licence (`LICENSE`). The outputs, figures and tables in `outputs/` are released under the Creative Commons Attribution 4.0 International licence (`LICENSE-outputs`).

## Contact

Yiğit Aydoğan, Department of Computer Science, Aberystwyth University, yia@aber.ac.uk
