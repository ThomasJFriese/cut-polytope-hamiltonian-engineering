"""Generate the instances, run the numerics and draw the figures of the paper.

    uv run main.py instances <experiment>
    OMP_NUM_THREADS=1 uv run main.py compute <experiment> [--cores N]
    uv run main.py plot <experiment | all>
"""
from functions import plot_computations as pc
from functions import plots
import argparse


EXPERIMENTS = {
    "complete_bipartite": {
        "instances": pc.complete_bipartite_plot_instances,
        "compute": lambda cores: pc.complete_bipartite_plot_computations(),
        "plot": plots.complete_bipartite_plot,
    },
    "edge_surpression": {
        "instances": pc.edge_surpression_plot_instances,
        "compute": pc.edge_surpression_plot_computations,
        "plot": plots.edge_surpression_plot,
    },
    "hofstadter": {
        "instances": pc.hofstadter_plot_instances,
        "compute": pc.hofstadter_plot_computations,
        "plot": plots.hofstadter_plot_combined,
    },
    "chiral_clock": {
        "instances": pc.chiral_clock_plot_instances,
        "compute": pc.chiral_clock_plot_computations,
        "plot": lambda: (plots.chiral_clock_phi_plot(), plots.chiral_clock_g_plot()),
    },
    "feasibility": {
        "instances": pc.feasibility_plot_instances,
        "compute": pc.feasibility_plot_computations,
        "plot": plots.feasibility_plot,
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["instances", "compute", "plot"])
    parser.add_argument("experiment", choices=list(EXPERIMENTS) + ["all"])
    parser.add_argument("--cores", type=int, default=16, help="worker processes for compute")
    args = parser.parse_args()

    if args.experiment == "all" and args.stage != "plot":
        parser.error("'all' is only available for plot")

    for name in (EXPERIMENTS if args.experiment == "all" else [args.experiment]):
        if args.stage == "compute":
            EXPERIMENTS[name]["compute"](args.cores)
        else:
            EXPERIMENTS[name][args.stage]()
