# Cut Polytope based Hamiltonian Engineering

Code to reproduce the numerical results and figures of the paper
[arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX). <!-- TODO: insert the arXiv identifier (here and in the Citation section) once the paper is online -->

## Setup

### Requirements

* [uv](https://docs.astral.sh/uv/getting-started/installation/), which manages the Python
  version and all dependencies. It can be installed with
  `curl -LsSf https://astral.sh/uv/install.sh | sh`.
* A LaTeX installation with the packages `lmodern`, `amsmath`, `mathtools` and `dsfont`. It is
  only needed for plotting, since the figures are typeset with LaTeX (`text.usetex`).

### Installation

Clone the repository and install the environment from within the repository root:

```bash
git clone https://github.com/ThomasJFriese/cut-polytope-hamiltonian-engineering.git
cd cut-polytope-hamiltonian-engineering
uv sync
```

`uv sync` downloads Python 3.13 if necessary (see `.python-version`), creates a virtual
environment in `.venv` and installs the exact package versions pinned in `uv.lock`. All
commands below are run through `uv run`, which uses this environment, so there is no need to
activate it manually.

## Reproducing the figures

Every experiment runs in three stages:

```bash
uv run main.py instances <experiment>                              # generate the instances
OMP_NUM_THREADS=1 uv run main.py compute <experiment> --cores N    # run the numerics
uv run main.py plot <experiment>                                   # draw the figure(s); `plot all` draws all of them
```

| Experiment           | Figure(s)                                     |
|----------------------|-----------------------------------------------|
| `complete_bipartite` | `plots/complete_bipartite.pdf`                |
| `edge_surpression`   | `plots/edge_surpression.pdf`                  |
| `hofstadter`         | `plots/hofstadter_combined.pdf`               |
| `chiral_clock`       | `plots/chiral_clock_phi.pdf`, `plots/chiral_clock_g.pdf` |
| `feasibility`        | `plots/feasibility.pdf`                       |

The instances used in the paper are included in `instances/paper_instances`, so the first stage
can be skipped. Results are written to `results/paper_results`, which is not part of the
repository.

Notes:

* Run the computations with one BLAS thread per worker, e.g.
  `OMP_NUM_THREADS=1 uv run main.py compute feasibility --cores 100`.
* The feasibility numerics take several hundred core hours. They stream their results to CSV
  backups and resume from them when restarted; `functions.feasibility.assemble_feasibility`
  builds the results file from the backups of an unfinished run.
* The optimal solutions of the 20 qubit instances (`complete_bipartite`, `edge_surpression`)
  solve an LP over all 2^20 pulses and need a lot of memory.
* The feasibility ensemble is generated from fixed seeds. All other experiments sample pulses
  (and the edge suppression graphs) without a fixed seed, so a rerun reproduces the results
  statistically rather than exactly.

## Structure

```
main.py                       command line entry point
functions/
    algo.py                   ray binary search, Gaussian rounding, informed LP algorithm and baselines
    expectation.py            expected correlation of k-rounded Gaussians and its inverse
    compute_lut.py            lookup tables for the inverse (luts/)
    instances.py              instance storage and generators, feasibility ensemble
    feasibility.py            feasibility numerics of appendix D
    plot_computations.py      instance generation and computations for each experiment
    plots.py                  figures
    config.py                 paths and numerical tolerance
instances/paper_instances/    instances used in the paper
luts/                         precomputed lookup tables for k = 3, 5, inf
plots/                        figures of the paper
```

## Citation

If you use this code, please cite the paper
[arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX).

## License

The code is released under the [MIT License](LICENSE).
