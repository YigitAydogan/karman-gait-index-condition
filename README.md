# Entrainment versus station keeping: an index condition for Kármán gaiting

Code and saved outputs for the manuscript

> Y. Aydoğan and O. Akanyeti, "Entrainment versus station keeping: an index condition for Kármán gaiting", submitted to *Chaos: An Interdisciplinary Journal of Nonlinear Science*.

The study is entirely computational. A Stuart-Landau central pattern generator (CPG) drives a pitching body that moves along a prescribed von Kármán wake. The scripts compute passive equilibria, entrainment at fixed position, station-holding orbits with free position, their Floquet multipliers and fixed-position branches, numerical continuation and bifurcations, noise and the experimental predictions, and every figure and table of the paper.

## Repository layout

```
scripts/                      22 Python scripts, run in numerical order
outputs/karman_gait_project/  every saved output (results, checkpoints, summaries, tables, figures)
MANIFEST.sha256               SHA-256 checksum of every file in outputs/
requirements.txt              Python packages
CITATION.cff, .zenodo.json    citation and archive metadata
```

Each script writes to its own folder, `outputs/karman_gait_project/<script name>/`, which holds its results (`.npz`, `.json`), a `summary_*.json` with the key numbers and, for the long runs, a `checkpoints_full/` folder. The final figures and the tables of the manuscript are in `outputs/karman_gait_project/script13_figures/`.

## Scripts and where their results appear

| Script | Content | Manuscript |
|---|---|---|
| 01 `passive_baseline` | Passive body: equilibria, stability, time to loss | Sec. III, Fig. 2a and 2c |
| 02 `passive_robustness` | Passive equilibria over a Latin hypercube sample of four wake families | Sec. III, Fig. 2b |
| 02b `passive_cases` | Floquet recheck of the passive equilibria | Sec. III, passive counts |
| 03 `cpg_tongues` | Entrainment at fixed position, Arnold tongues, node and saddle locks | Sec. IV, Fig. 3 |
| 04, 04b `free_position`, `locked_exceptions` | First tests with free position | Development, superseded by 05 to 10 |
| 05, 05b, 05c `holding_tongue`, `holding_dense`, `exceptions_chaos` | Station-holding orbits on dense grids, tests for chaos | Secs. V and VI |
| 06, 06c `robustness`, `body_aL` | Body, CPG and wake parameter variants | Sec. VII, Table VI |
| 07 `index_condition` | First verification of the index condition | Development, superseded by 08 |
| 08 `homotopy` | Fixed-position branch by homotopy, index test on parameter sets | Sec. V, Table II, Fig. 5a |
| 09 `model_structure` | 16 model variants and 20 joint parameter samples | Secs. V and VII, Tables II and V, Fig. 5b |
| 10 `continuation` | Pseudo-arclength continuation, folds, torus bifurcations, areas | Secs. V and VI, Tables II and III, Figs. 4, 5c and 6 |
| 11 `noise_predictions` | Reaching the station, noise, phase signature, lateral line, flow speed | Sec. VIII, Tables VII and IX, Fig. 7 |
| 12 `verification` | Convergence, exact versus finite-difference multipliers, recomputation | Sec. V, Table II, Appendix C, Table XI |
| 13 `figures` | All figures and tables from the saved outputs | Figs. 1 to 8, Tables I to XI |
| 14 `referee_analyses` | Exclusive areas, orbit accounting, passive bounds, residence bouts, phase discrimination, hypothesis margins, phase-lag sweep, confinement, stochastic time step | Secs. III, V, VI and VIII, Tables II to IV, VII and VIII |
| 15 `collect_bundle` | Collects all outputs into one archive | Repository |
| 16 `branch_change_check` | Identifies the single branch change of the convergence check | Appendix C, Table XI |
| 17 `biology_questions` | Fate of passive bodies, position of stable holding relative to the vortex peak, control experiment in uniform flow | Sec. IX, Fig. 8, Table X |

Scripts 04, 04b and 07 are kept for completeness; their results were superseded by later scripts and are not used in the manuscript.

## Running the scripts

The scripts were written for Google Colab. Each one is self-contained: paste it into a single cell and run it. It mounts Google Drive and reads and writes `MyDrive/karman_gait_project/`. Outside Colab, the scripts use `./karman_gait_project/` relative to the working directory.

* Scripts 01 to 09 need only NumPy and Matplotlib and run on a CPU.
* Scripts 10, 11, 12, 14, 16 and 17 use JAX and were run on an NVIDIA L4 GPU. They also run on a CPU, more slowly.
* Every long script saves checkpoints. If a run stops, running the script again resumes it.
* Most scripts have a `MODE` setting: `"FULL"` reproduces the manuscript, `"QUICK"` is a short test.

Later scripts read the outputs of earlier ones, so a full reproduction runs them in numerical order. Runtimes are given in the header of each script.

### Reproducing the figures and tables without rerunning the simulations

All saved outputs are included, so the figures and tables can be rebuilt directly:

```bash
pip install -r requirements.txt
cd outputs
python ../scripts/script13_figures.py
```

The figures and tables are written to `outputs/karman_gait_project/script13_figures/`. Figure 1 combines a drawn schematic, stored as `Fig1_source.png`, with panel c, which Script 13 generates.

To rerun any single analysis in Colab, copy `outputs/karman_gait_project/` to `MyDrive/karman_gait_project/` first, so that the script finds the outputs of the scripts it depends on.

## Checking file integrity

```bash
sha256sum -c MANIFEST.sha256
```

## Licence

The code in `scripts/` is released under the MIT licence (`LICENSE`). The outputs, figures and tables in `outputs/` are released under the Creative Commons Attribution 4.0 International licence (`LICENSE-outputs`).

## Contact

Yiğit Aydoğan, Department of Computer Science, Aberystwyth University, yia@aber.ac.uk
