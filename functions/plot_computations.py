from functions.instances import nx_to_target_dict, save_instance, generate_Hofstadter, generate_chiral_clock
from functions.config import INSTANCES_DIR, RESULTS_DIR
from functions.algo import exact_algo, explicit_construction, cut_poly_algo, eff_relax_algo
from functions.compute_lut import load_k_LUT
from functions.feasibility import base_ensemble, stress_ensemble, save_ensemble, load_ensemble, run_feasibility
from itertools import product, repeat
import networkx as nx
import concurrent.futures
import numpy as np


### Complete bipartite graph

def complete_bipartite_plot_instances():
    # keep the edges of K_{10,10} and suppress all other pairs of the 20 qubits
    n = 20
    K = 10

    target_dict = {}
    for i in range(K):
        for j in range(i+1, K):
            target_dict[i, j] = 0

        for j in range(K, n):
            target_dict[i, j] = 1

    for i in range(K, n):
        for j in range(i+1, n):
            target_dict[i, j] = 0

    save_instance(n=n, target_dict=target_dict, h_type="qubit", instance_file="paper_instances/complete_bipartite.hdf5")


def complete_bipartite_plot_computations():
    n = 20
    edges = n * (n - 1) // 2
    ratios = np.linspace(2, 4, 20)
    runs_per_ratio = 50

    instance_file = "paper_instances/complete_bipartite.hdf5"
    results = "paper_results/complete_bipartite/complete_bipartite"
    (RESULTS_DIR / "paper_results/complete_bipartite").mkdir(parents=True, exist_ok=True)

    exact_algo(2, instance_file, f"{results}_exact.hdf5")
    explicit_construction(instance_file, f"{results}_explicit.hdf5")

    for ratio, r in product(ratios, range(runs_per_ratio)):
        pulse_n = int(ratio * edges)

        cut_poly_algo(2, pulse_n, instance_file, f"{results}_{ratio}_{r}_cut_poly.hdf5")
        eff_relax_algo(2, pulse_n, instance_file, f"{results}_{ratio}_{r}_eff_relax.hdf5")

        print(f"Finished ratio {ratio} | run {r}")


### Edge suppression

def edge_surpression_plot_instances():
    # keep the m edges of a random G(n, m) and suppress all other pairs
    n = 20
    graphs_per_density = 20
    max_edges = n * (n - 1) // 2

    (INSTANCES_DIR / "paper_instances/edge_surpression").mkdir(parents=True, exist_ok=True)

    for m in range(10, max_edges + 1, 10):
        for r in range(graphs_per_density):
            instance = nx_to_target_dict(nx.gnm_random_graph(n=n, m=m))
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in instance.keys():
                        instance[i, j] = 0

            save_instance(n=n, target_dict=instance, h_type="qubit", instance_file=f"paper_instances/edge_surpression/n={n}_m={m}_instance={r}.hdf5")


def edge_surpression_plot_computations_single_instance(instance, pulse_n, best_of_runs):
    instance_file = f"paper_instances/edge_surpression/{instance}.hdf5"
    results = f"paper_results/edge_surpression/{instance}"

    exact_algo(2, instance_file, f"{results}_exact.hdf5")
    explicit_construction(instance_file, f"{results}_explicit.hdf5")
    cut_poly_algo(2, pulse_n, instance_file, f"{results}_cut_poly.hdf5", best_of_runs=best_of_runs)
    eff_relax_algo(2, pulse_n, instance_file, f"{results}_eff_relax.hdf5", best_of_runs=best_of_runs)

    return instance


def edge_surpression_plot_computations(cores):
    n = 20
    graphs_per_density = 20
    best_of_runs = 20
    fixed_s_m_ratio = 3
    max_edges = n * (n - 1) // 2

    (RESULTS_DIR / "paper_results/edge_surpression").mkdir(parents=True, exist_ok=True)

    instances = [f"n={n}_m={m}_instance={r}" for m in range(10, max_edges + 1, 10) for r in range(graphs_per_density)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        for instance in executor.map(edge_surpression_plot_computations_single_instance, instances, repeat(fixed_s_m_ratio * max_edges), repeat(best_of_runs)):
            print(f"Finished {instance}.")


### Hofstadter model

def hofstadter_plot_instances():
    golden_ratio = (1 + 5 ** 0.5) / 2
    phi = golden_ratio * 2 * np.pi

    for n in range(3, 21):
        generate_Hofstadter(n, phi, f"paper_instances/hofstadter_{n}.hdf5")


def hofstadter_plot_computations_single_instance(args, runs_per_model, fixed_s_m_ratio):
    n, k = args
    edges = 2 * n * (n - 1)

    instance = f"hofstadter_{n}"
    instance_file = f"paper_instances/hofstadter_{n}.hdf5"

    for r in range(runs_per_model):
        cut_poly_file = f"paper_results/hofstadter/{instance}_k={k}_cut_poly_{r}.hdf5"
        eff_relax_file = f"paper_results/hofstadter/{instance}_k={k}_eff_relax_{r}.hdf5"

        cut_poly_algo(k, fixed_s_m_ratio * edges, instance_file, cut_poly_file)
        eff_relax_algo(k, fixed_s_m_ratio * edges, instance_file, eff_relax_file)

        print(f"Finished n={n}-k={k}-r={r}")

    return instance


def hofstadter_plot_computations(cores):
    ns = range(3, 21)
    ks = [3, 4, np.inf]
    runs_per_model = 50
    fixed_s_m_ratio = 6

    (RESULTS_DIR / "paper_results/hofstadter").mkdir(parents=True, exist_ok=True)

    # compute missing LUTs before forking
    for k in [3, np.inf]:
        load_k_LUT(k)

    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        for instance in executor.map(hofstadter_plot_computations_single_instance, product(ns, ks), repeat(runs_per_model), repeat(fixed_s_m_ratio)):
            print(f"Finished {instance}.")


### Chiral clock model

def chiral_clock_plot_instances():
    q = 4
    ks = [3, 4, 5]
    phi_steps = 21
    gs = np.linspace(0, 2, 21)

    (INSTANCES_DIR / "paper_instances/chiral_clock").mkdir(parents=True, exist_ok=True)

    for k in ks:
        for phi in np.linspace(0, np.pi / k, phi_steps):
            generate_chiral_clock(q, phi, 0, f"paper_instances/chiral_clock/phi={phi}_g=0.hdf5")

    for g in gs:
        generate_chiral_clock(q, 0, g, f"paper_instances/chiral_clock/phi=0_g={g}.hdf5")


def chiral_clock_plot_computations_single_instance(args, runs_per_model, fixed_s_D_ratio):
    instance, k = args
    q = 4
    edges = 2 * q ** 2
    D = edges if k == 2 else 2 * edges

    instance_file = f"paper_instances/chiral_clock/{instance}.hdf5"

    exact_file = f"paper_results/chiral_clock/{instance}_k={k}_exact.hdf5"
    exact_algo(k, instance_file, exact_file, fix_last=True)

    for r in range(runs_per_model):
        cut_poly_file = f"paper_results/chiral_clock/{instance}_k={k}_cut_poly_{r}.hdf5"
        eff_relax_file = f"paper_results/chiral_clock/{instance}_k={k}_eff_relax_{r}.hdf5"

        cut_poly_algo(k, fixed_s_D_ratio * D, instance_file, cut_poly_file)
        eff_relax_algo(k, fixed_s_D_ratio * D, instance_file, eff_relax_file)

    print(f"Finished {instance}-k={k}")

    return instance


def chiral_clock_plot_computations(cores):
    phi_ks = [3, 4, 5]
    phi_steps = 21
    g_ks = [2, 3, 4, 5]
    gs = np.linspace(0, 2, 21)
    runs_per_model = 50
    fixed_s_D_ratio = 3

    (RESULTS_DIR / "paper_results/chiral_clock").mkdir(parents=True, exist_ok=True)

    # compute missing LUTs before forking
    for k in [3, 5]:
        load_k_LUT(k)

    tasks = [(f"phi={phi}_g=0", k) for k in phi_ks for phi in np.linspace(0, np.pi / k, phi_steps)]
    tasks += [(f"phi=0_g={g}", k) for g in gs for k in g_ks]

    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        for instance in executor.map(chiral_clock_plot_computations_single_instance, tasks, repeat(runs_per_model), repeat(fixed_s_D_ratio)):
            print(f"Finished {instance}.")


### Feasibility (appendix D)

def feasibility_plot_instances():
    # the ensemble is stored as seeds, from which run_instance rebuilds every instance
    specs = base_ensemble(instances_per_cell=48, diagnostics_per_cell=8)
    specs += stress_ensemble(base_instances=20, start_id=len(specs))

    save_ensemble(specs)

    return specs


def feasibility_plot_computations(cores):
    specs = load_ensemble()

    base = [spec for spec in specs if spec["arm"] == "base"]
    stress = [spec for spec in specs if spec["arm"] == "stress"]

    # the transition of the stress arm moves well past s = 3 D, so it needs a wider window
    run_feasibility(stress, results_file="paper_results/feasibility_stress.hdf5", cores=cores,
                    r_max=12.0, repetitions=100, repetitions_floor=25)

    run_feasibility(base, results_file="paper_results/feasibility.hdf5", cores=cores,
                    r_max=3.0, repetitions=200, repetitions_floor=50)
