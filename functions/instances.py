from functions.config import INSTANCES_DIR
import h5py
import numpy as np
import ast
import networkx as nx


# An instance is a target {(i, j): coefficient} on the interaction edges i < j of n sites;
# edges with coefficient 0 are to be suppressed.

def save_instance(n, target_dict, h_type, instance_file, descr=""):
    dt = h5py.special_dtype(vlen=str)
    connec = np.array([str(k) for k in target_dict.keys()], dtype=dt)
    coeffs = np.array(list(target_dict.values()), dtype=np.complex128)

    with h5py.File(INSTANCES_DIR / instance_file, "w") as f:
        f.attrs["n"] = n
        f.attrs["hamiltonian type"] = h_type
        f.attrs["description"] = descr

        f.create_dataset("connectivity", data=connec, compression="gzip")
        f.create_dataset("coefficients", data=coeffs, compression="gzip")


def load_instance(instance_file):
    with h5py.File(INSTANCES_DIR / instance_file, "r") as f:
        n = f.attrs["n"]
        h_type = f.attrs["hamiltonian type"]
        descr = f.attrs["description"]

        connec_strs = f["connectivity"][:]
        coeffs = f["coefficients"][:]

    target_dict = {}
    for k_bytes, v in zip(connec_strs, coeffs):
        k_str = k_bytes.decode('utf-8') if isinstance(k_bytes, bytes) else k_bytes

        target_dict[ast.literal_eval(k_str)] = v

    return n, target_dict, h_type, descr


def nx_to_target_dict(G):
    return {(u, v): 1 for (u, v) in G.edges if u < v}


def hollow_matrix(n, target_dict):
    """Hermitian target matrix H with zero diagonal, and the mask of the interaction edges."""
    H = np.zeros(shape=(n, n), dtype=np.complex128)
    mask = np.identity(n=n, dtype=int)

    for k, v in target_dict.items():
        H[k[0], k[1]] = v
        H[k[1], k[0]] = np.conj(v)
        mask[k[0], k[1]] = 1
        mask[k[1], k[0]] = 1

    return H, mask


def load_hollow_matrix(instance_file):
    n, target_dict, _, _ = load_instance(instance_file)

    H, mask = hollow_matrix(n, target_dict)

    return n, H, mask


def generate_Hofstadter(n, phase, instance_file):
    """n x n Hofstadter model: the bonds along i carry the phase exp(i phase j)."""
    def grid_to_index(i, j):
        return i * n + j

    target_dict = {}

    for i in range(n):
        for j in range(n):
            if i > 0:
                target_dict[(grid_to_index(i-1, j), grid_to_index(i, j))] = np.exp(1j * phase * j)

            if i < n-1:
                target_dict[(grid_to_index(i, j), grid_to_index(i+1, j))] = np.exp(1j * phase * j)

            if j > 0:
                target_dict[(grid_to_index(i, j-1), grid_to_index(i, j))] = 1

            if j < n-1:
                target_dict[(grid_to_index(i, j), grid_to_index(i, j+1))] = 1

    save_instance(n * n, target_dict, "fermionic", instance_file, descr=f"{n}x{n} Hofstadter model with base phase {phase}.")


def generate_chiral_clock(q, phi, g, instance_file, theta=0):
    # sites 0..q-1 are X_a, q..2q-1 are Z_a, 2q is the identity
    n = 2 * q + 1

    target_dict = {
        (i, j) : 0
        for i in range(n) for j in range(i+1, n) if not (i < q and j == i + q)
    }

    for a in range(q - 1):
        target_dict[q + a, q + a + 1] = np.exp(1j * phi)

    for a in range(q):
        target_dict[a, 2 * q] = g * np.exp(1j * theta)

    save_instance(n, target_dict, "qudit", instance_file, descr=f"Chiral clock model on an open chain of q={q} qudits with phi={phi}, theta={theta}, g={g}, from all two-site terms of distinct qudits.")


### Instance ensemble of the feasibility numerics (appendix D)
#
# M is normalised to max_ij |M_ij| = 1, so that the saturation ghat * ||M||_inf equals ghat.

ER_DENSITIES = [0.1, 0.2, 0.35, 0.5, 0.75]
REGULAR_DEGREES = [3, 4, 6]
GRAPH_FAMILIES = ["erdos_renyi", "regular", "complete", "complete_bipartite", "lattice", "tree"]

REAL_WEIGHT_LAWS = ["rademacher", "gaussian", "uniform", "student_t3", "ones"]
COMPLEX_WEIGHT_LAWS = ["unit_phase", "rooted_phase", "complex_gaussian", "student_t3_modulus"]

HOFSTADTER_FLUX = 8 / 13


def grid_shape(n):
    # most square n = L1 * L2 with both sides at least 3, so the periodic lattice is simple
    for L1 in range(int(np.sqrt(n)), 2, -1):
        if n % L1 == 0 and n // L1 >= 3:
            return L1, n // L1

    return None


def weight_laws(k, family):
    if k == 2:
        return list(REAL_WEIGHT_LAWS)

    laws = [law for law in COMPLEX_WEIGHT_LAWS if not (law == "rooted_phase" and np.isinf(k))]

    if family == "lattice":
        laws = laws + ["hofstadter"]

    return laws


def er_densities(n):
    # only densities above the connectivity threshold ln(n) / n
    return [p for p in ER_DENSITIES if p * (n - 1) >= np.log(n)]


def sample_graph(family, n, rng, attempts=100):
    """Connected graph from the family; its parameter is drawn once, outside the rejection loop."""
    if family == "erdos_renyi":
        param = float(rng.choice(er_densities(n)))
    elif family == "regular":
        param = float(rng.choice(REGULAR_DEGREES))
    elif family == "complete_bipartite":
        param = float(n // 2)
    elif family == "lattice":
        param = float(grid_shape(n)[0])
    elif family in ["complete", "tree"]:
        param = np.nan
    else:
        raise ValueError(f"Unknown graph family {family}.")

    for _ in range(attempts):
        if family == "erdos_renyi":
            G = nx.erdos_renyi_graph(n=n, p=param, seed=int(rng.integers(2 ** 31)))

        elif family == "regular":
            G = nx.random_regular_graph(d=int(param), n=n, seed=int(rng.integers(2 ** 31)))

        elif family == "complete":
            G = nx.complete_graph(n)

        elif family == "complete_bipartite":
            G = nx.complete_bipartite_graph(n // 2, n - n // 2)

        elif family == "lattice":
            L1, L2 = grid_shape(n)
            G = nx.convert_node_labels_to_integers(
                nx.grid_2d_graph(L1, L2, periodic=True), label_attribute="grid"
            )

        elif family == "tree":
            G = nx.random_labeled_tree(n, seed=int(rng.integers(2 ** 31)))

        if nx.is_connected(G):
            return G, param

    raise RuntimeError(f"Could not draw a connected graph from family {family} on n={n}.")


def sample_weights(law, G, k, rng):
    edges = sorted((min(u, v), max(u, v)) for u, v in G.edges())
    m = len(edges)

    if law == "rademacher":
        w = rng.choice([-1.0, 1.0], size=m)

    elif law == "gaussian":
        w = rng.standard_normal(m)

    elif law == "uniform":
        w = rng.uniform(-1.0, 1.0, size=m)

    elif law == "student_t3":
        w = rng.standard_t(3, size=m)

    elif law == "ones":
        w = np.ones(m)

    elif law == "unit_phase":
        w = np.exp(2j * np.pi * rng.random(m))

    elif law == "rooted_phase":
        # phases that are already k-th roots of unity
        w = np.exp(2j * np.pi * rng.integers(k, size=m) / k)

    elif law == "complex_gaussian":
        w = (rng.standard_normal(m) + 1j * rng.standard_normal(m)) / np.sqrt(2)

    elif law == "student_t3_modulus":
        w = np.abs(rng.standard_t(3, size=m)) * np.exp(2j * np.pi * rng.random(m))

    elif law == "hofstadter":
        # Landau gauge on the torus: the bonds along x carry the phase 2 pi Phi y
        grid = nx.get_node_attributes(G, "grid")
        w = np.empty(m, dtype=np.complex128)
        for e, (u, v) in enumerate(edges):
            (x_u, y_u), (x_v, y_v) = grid[u], grid[v]
            w[e] = np.exp(2j * np.pi * HOFSTADTER_FLUX * y_u) if y_u == y_v else 1.0

    else:
        raise ValueError(f"Unknown weight law {law}.")

    return {e: v for e, v in zip(edges, np.asarray(w, dtype=np.complex128))}


def gauge_matrix(n, k, mask, rng):
    # P_ij = z_i conj(z_j) with z_i k-th roots of unity: an extreme point with ghat = 1
    if np.isinf(k):
        z = np.exp(2j * np.pi * rng.random(n))
    else:
        z = np.exp(2j * np.pi * rng.integers(k, size=n) / k)

    return np.outer(z, np.conj(z)) * (mask - np.identity(n, dtype=int))


def sample_ensemble_instance(k, n, family, law, theta, rng):
    G, family_param = sample_graph(family, n, rng)
    target_dict = sample_weights(law, G, k, rng)

    M, mask = hollow_matrix(n, target_dict)
    M = M / np.max(np.abs(M))

    # stress arm: interpolate towards P (theta = 0 is the base ensemble); P is drawn in
    # either case to keep the rng stream aligned
    P = gauge_matrix(n, k, mask, rng)
    if theta > 0:
        M = (1 - theta) * M + theta * P
        M = M / np.max(np.abs(M))

    return M, mask, family_param
