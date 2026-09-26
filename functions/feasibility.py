"""Feasibility numerics of appendix D.

For every instance the ray binary search gives ghat and X, pulses are rounded from X, and
the smallest number of pulses s is bisected at which the LP reaches the quantum run time
1 / ghat, i.e. alpha_max >= ghat. All LPs are solved with the HiGHS dual simplex.
"""

from functions.algo import cut_poly_ray, gaussian_rounding
from functions.compute_lut import k_exp_inv_for
from functions.config import RESULTS_DIR, INSTANCES_DIR
from functions.instances import GRAPH_FAMILIES, grid_shape, weight_laws, sample_ensemble_instance
from itertools import product
from pathlib import Path
import concurrent.futures
import contextlib
import csv
import hashlib
import h5py
import numpy as np
import pandas as pd
import scipy


RAY_TOLERANCE = 1e-12      # bisection width of the ray binary search
LP_TOLERANCE = 1e-9        # residual threshold of the LPs, scaled by sqrt(D)
LP_METHOD = "highs-ds"     # simplex, for vertex solutions
TRIVIAL_SATURATION = 1e-3  # base instances with saturation > 1 - TRIVIAL_SATURATION are discarded
PULSE_SPACE_SLACK = 1e3    # require k^(n-1) >= PULSE_SPACE_SLACK * s_max


### Building blocks

def psd_factor(X):
    # X is singular, so Cholesky is not applicable
    vals, vecs = np.linalg.eigh(X)

    return vecs * np.sqrt(np.clip(vals, 0.0, None))


def edge_indices(mask):
    return np.nonzero(np.triu(mask, k=1))


def target_dimension(k, m):
    # D = m for real (k = 2) and D = 2m for complex targets
    return m if k == 2 else 2 * m


def real_embedding(k, C):
    return np.real(C) if k == 2 else np.vstack([np.real(C), np.imag(C)])


def chi(k, pulses, i, j):
    # (x_a conj(x_b))_{(a, b) in E} for every pulse x, as a D x s array
    angles = pulses if np.isinf(k) else 2 * np.pi / k * pulses
    ang = np.exp(1j * angles)

    return real_embedding(k, (ang[:, i] * np.conj(ang[:, j])).T)


def sample_chi(k, L, n, s, i, j, rng, block=4096):
    return np.hstack([
        chi(k, gaussian_rounding(L, n, k, min(block, s - start), rng=rng), i, j)
        for start in range(0, s, block)
    ])


def pulse_space_ok(k, n, s_max):
    if np.isinf(k):
        return True

    return (n - 1) * np.log(k) >= np.log(PULSE_SPACE_SLACK * s_max)


def informed_point(k, M, tolerance=RAY_TOLERANCE):
    X, gamma = cut_poly_ray(M, k_exp_inv_for(k), tolerance=tolerance)

    return X, gamma


### Diagnostics: effective dimension and bracket on beta

def pilot_samples(k, L, n, N, i, j, mean, rng, keep, block=4096):
    """Covariance of chi(x) over N pulses, centred at the exact mean, and the first `keep` samples."""
    Sigma = np.zeros((len(mean), len(mean)))
    Y = []
    drawn = 0

    while drawn < N:
        size = min(block, N - drawn)
        Yb = (sample_chi(k, L, n, size, i, j, rng).T - mean)

        Sigma += Yb.T @ Yb
        if drawn < keep:
            Y.append(Yb[:keep - drawn])

        drawn += size

    return Sigma / N, np.vstack(Y)


def effective_dimension(Sigma, threshold=1e-6):
    """Number of eigenvalues of Sigma above threshold * Tr(Sigma) / D, and the spectral gap there."""
    D = len(Sigma)

    vals, vecs = np.linalg.eigh(Sigma)
    vals, vecs = vals[::-1], vecs[:, ::-1]

    D_eff = int(np.sum(vals > threshold * np.trace(Sigma) / D))

    gap = vals[D_eff - 1] / vals[D_eff] if 0 < D_eff < D else np.inf

    return D_eff, vals, vecs, gap


def saturated_edges(X, i, j):
    # edges with |X_ij| = 1 are deterministic and reduce the effective dimension
    return int(np.sum(np.abs(X[i, j]) > 1 - 1e-8))


def positive_fraction(Y, V, best=np.inf, best_v=None):
    frac = np.mean((Y @ V) > 0, axis=0)

    if frac.min() < best:
        best = float(frac.min())
        best_v = V[:, int(np.argmin(frac))].copy()

    return best, best_v


def beta_upper(Y, vecs, rng, n_random=10000, block=256, n_refine=200, n_starts=4):
    """Upper estimate of beta = inf_v P[<v, y> > 0] over random, principal and refined directions."""
    D = Y.shape[1]

    best, best_v = positive_fraction(Y, np.hstack([vecs, -vecs]))

    for start in range(0, n_random, block):
        R = rng.standard_normal((D, min(block, n_random - start)))
        R /= np.linalg.norm(R, axis=0)

        best, best_v = positive_fraction(Y, R, best, best_v)

    # local refinement by random perturbations with a shrinking radius
    for _ in range(n_starts):
        v, value, radius = best_v.copy(), best, 0.5

        for _ in range(n_refine):
            w = v + radius * rng.standard_normal(D)
            w /= np.linalg.norm(w)

            frac = float(np.mean((Y @ w) > 0))
            if frac < value:
                v, value = w, frac
            else:
                radius *= 0.97

        if value < best:
            best, best_v = value, v

    return best


def kurtosis_objective(Y, v):
    p = Y @ v
    A, B = np.mean(p ** 4), np.mean(p ** 2)

    grad = 4 * (Y.T @ p ** 3) / (len(p) * B ** 2) - 8 * A * (Y.T @ p) / (len(p) * B ** 3)

    return A / B ** 2, grad


def directional_kurtosis(Y, vecs, rng, n_starts=20, n_steps=300, step=0.5):
    """kappa_0 = sup_v E<v, y>^4 / (E<v, y>^2)^2 by projected gradient ascent; beta >= 1 / (4 kappa_0)."""
    D = Y.shape[1]

    starts = [vecs[:, -1], vecs[:, 0]] + list(rng.standard_normal((n_starts, D)))
    best = -np.inf

    for v in starts:
        v = v / np.linalg.norm(v)
        value, grad = kurtosis_objective(Y, v)
        eta = step

        for _ in range(n_steps):
            w = v + eta * grad
            w /= np.linalg.norm(w)

            new_value, new_grad = kurtosis_objective(Y, w)

            if new_value > value:
                v, value, grad = w, new_value, new_grad
                eta *= 1.1
            else:
                eta *= 0.5

            if eta < 1e-12:
                break

        best = max(best, value)

    return best


### LPs

def phase1_feasible(A_eq, b_eq, tolerance):
    """Does A_eq x = b_eq have a solution x >= 0? Returns (feasible, residual).

    Solved as the phase-1 LP with one artificial column per row, which is always feasible,
    so that infeasibility is decided by the residual rather than by the solver status.
    """
    columns = A_eq.shape[1]

    signs = np.where(b_eq >= 0, 1.0, -1.0)
    A_phase1 = np.hstack([A_eq, np.diag(signs)])
    c = np.concatenate([np.zeros(columns), np.ones(len(b_eq))])

    res = scipy.optimize.linprog(
        c, A_eq=A_phase1, b_eq=b_eq, bounds=(0, None), method=LP_METHOD,
    )

    if not res.success:
        print(f"Phase-1 LP of size {A_phase1.shape} returned status {res.status}: {res.message}")
        return False, np.nan

    residual = np.linalg.norm(A_eq @ res.x[:columns] - b_eq)

    return residual <= tolerance, residual


def solve_feasibility_lp(A, b):
    """Is b in the convex hull of the columns of A?"""
    s = A.shape[1]

    A_eq = np.vstack([A, np.ones((1, s))])
    b_eq = np.concatenate([b, [1.0]])

    return phase1_feasible(A_eq, b_eq, LP_TOLERANCE * np.sqrt(len(b)))


def solve_restricted_lp(A, m_vec, maximise=True):
    """Maximise (or minimise) alpha s.t. alpha * m = A lambda, sum lambda = 1, lambda >= 0."""
    s = A.shape[1]

    A_eq = np.zeros((len(m_vec) + 1, s + 1))
    A_eq[:-1, 0] = -m_vec
    A_eq[:-1, 1:] = A
    A_eq[-1, 1:] = 1.0

    b_eq = np.zeros(len(m_vec) + 1)
    b_eq[-1] = 1.0

    c = np.zeros(s + 1)
    c[0] = -1.0 if maximise else 1.0

    res = scipy.optimize.linprog(
        c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method=LP_METHOD,
    )

    if not res.success:
        return np.nan, None

    return res.x[0], res.x[1:]


def run_time_success(A, m_vec, gamma):
    """Is alpha_max >= ghat? Decided as feasibility of A lambda - t m = ghat m, t >= 0."""
    D = len(m_vec)
    s = A.shape[1]

    A_eq = np.zeros((D + 1, s + 1))
    A_eq[:D, :s] = A
    A_eq[:D, s] = -m_vec
    A_eq[D, :s] = 1.0

    b_eq = np.concatenate([gamma * m_vec, [1.0]])

    return phase1_feasible(A_eq, b_eq, LP_TOLERANCE * np.sqrt(D))


def min_successful_pulses(A, m_vec, gamma, lower, upper):
    """Smallest prefix s of the pulses reaching run time 1 / ghat, or None if not within `upper`.

    alpha_max is non-decreasing in s, so the threshold can be bisected.
    """
    if not run_time_success(A[:, :upper], m_vec, gamma)[0]:
        return None

    while upper - lower > 1:
        middle = (lower + upper) // 2

        if run_time_success(A[:, :middle], m_vec, gamma)[0]:
            upper = middle
        else:
            lower = middle

    return upper


def robustly_successful(A, m_vec, gamma, basis, rng, epsilon=1e-6, Q=8):
    """Does the success survive Q random perturbations of m within the span `basis`?"""
    for _ in range(Q):
        u = basis @ rng.standard_normal(basis.shape[1])
        u /= np.linalg.norm(u)

        if not run_time_success(A, m_vec + epsilon * u, gamma)[0]:
            return False

    return True


### Ensemble design

INSTANCE_COLUMNS = [
    "instance_id", "arm", "k", "n", "family", "family_param", "law", "theta", "m", "D",
    "gamma_hat", "saturation", "D_eff", "sat_edges", "D_eff_predicted", "spectral_gap",
    "beta_upper", "kappa_0", "beta_lower", "pulse_space_ok", "discarded", "seed", "hash",
]

THRESHOLD_COLUMNS = [
    "instance_id", "repetition", "D", "s_max", "s_star", "r_star",
    "alpha_star", "alpha_min_star", "conjecture_lp", "robust_s_star",
]

RUNTIME_COLUMNS = [
    "instance_id", "repetition", "r", "s", "alpha", "alpha_min", "run_time", "support",
]


def ensemble_specs(arm, k, n, families, laws, theta, count, rng, diagnostics_count, start_id):
    specs = []

    for c in range(count):
        family = families[c % len(families)]
        law = laws[family][int(rng.integers(len(laws[family])))]

        specs.append({
            "instance_id": start_id + c,
            "arm": arm,
            "k": k,
            "n": n,
            "family": family,
            "law": law,
            "theta": theta,
            "target_saturation": np.nan,
            "diagnostics": c < diagnostics_count,
            "seed": int(rng.integers(2 ** 62)),
        })

    return specs


def base_ensemble(instances_per_cell=48, ks=(2, 3, 4, np.inf), ns=(24, 28, 32, 36),
                  diagnostics_per_cell=8, seed=20260920):
    """Per (k, n) cell, instances spread evenly over the graph families with random weight laws."""
    rng = np.random.default_rng(seed)
    specs = []

    for k, n in product(ks, ns):
        families = [f for f in GRAPH_FAMILIES if f != "lattice" or grid_shape(n) is not None]
        laws = {family: weight_laws(k, family) for family in families}

        specs += ensemble_specs("base", k, n, families, laws, 0.0, instances_per_cell,
                                rng, diagnostics_per_cell, len(specs))

    return specs


def saturation_for_theta(k, n, family, law, seed, theta, tolerance=1e-8):
    rng = np.random.default_rng(seed)
    M, _, _ = sample_ensemble_instance(k, n, family, law, theta, rng)
    _, gamma = informed_point(k, M, tolerance=tolerance)

    return gamma * np.max(np.abs(M))


def theta_for_saturation(k, n, family, law, seed, target, steps=30, atol=1e-3, u_max=12.0):
    """Mixing weight theta at which a stress instance reaches the saturation `target`.

    Bisection on u = log10(1 / (1 - theta)), since the saturation changes almost entirely
    within the last decade of 1 - theta.
    """
    if saturation_for_theta(k, n, family, law, seed, 0.0) >= target:
        return 0.0

    lower, upper = 0.0, u_max

    for _ in range(steps):
        middle = (lower + upper) / 2
        theta = 1 - 10.0 ** -middle
        sigma = saturation_for_theta(k, n, family, law, seed, theta)

        if abs(sigma - target) <= atol:
            return theta

        if sigma < target:
            lower = middle
        else:
            upper = middle

    return 1 - 10.0 ** -upper


def stress_ensemble(base_instances=20, k=4, n=24, families=("complete",),
                    saturations=(0.5, 0.8, 0.9, 0.95, 0.99, 0.997, 0.999),
                    seed=20260921, start_id=0):
    """Base instances interpolated towards the gauge matrix P up to the target saturations.

    Only the complete graph reaches saturation 1 in the limit theta -> 1. The seed only
    depends on the base index, so every sweep interpolates the same instance.
    """
    rng = np.random.default_rng(seed)

    families = list(families)
    laws = {family: weight_laws(k, family) for family in families}

    base = ensemble_specs("stress", k, n, families, laws, 0.0, base_instances, rng,
                          base_instances, start_id)

    return [
        dict(spec,
             target_saturation=sigma,
             theta=theta_for_saturation(k, n, spec["family"], spec["law"],
                                        spec["seed"], sigma),
             instance_id=start_id + t * base_instances + c)
        for t, sigma in enumerate(saturations) for c, spec in enumerate(base)
    ]


def write_table(group, rows, columns):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    df = df.reindex(columns=columns)
    str_dtype = h5py.special_dtype(vlen=str)

    group.attrs["columns"] = list(columns)

    for column in columns:
        data = df[column].to_numpy() if len(df) else np.zeros(0)

        if data.dtype == object:
            data = np.array([str(x) for x in data], dtype=str_dtype)

        group.create_dataset(column, data=data, compression="gzip")


def read_table(group):
    return pd.DataFrame({
        column: (group[column].asstr()[:] if h5py.check_string_dtype(group[column].dtype)
                 else group[column][:])
        for column in group.attrs["columns"]
    })


SPEC_COLUMNS = ["instance_id", "arm", "k", "n", "family", "law", "theta",
                "target_saturation", "diagnostics", "seed"]


def save_ensemble(specs, ensemble_file="paper_instances/feasibility_ensemble.hdf5"):
    (INSTANCES_DIR / ensemble_file).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(INSTANCES_DIR / ensemble_file, "w") as f:
        f.attrs["description"] = "Instance ensemble for the feasibility numerics of appendix D"
        write_table(f, specs, SPEC_COLUMNS)


def load_ensemble(ensemble_file="paper_instances/feasibility_ensemble.hdf5"):
    with h5py.File(INSTANCES_DIR / ensemble_file, "r") as f:
        columns = read_table(f)

    return [
        {"instance_id": int(columns["instance_id"][r]), "arm": str(columns["arm"][r]),
         "k": float(columns["k"][r]), "n": int(columns["n"][r]),
         "family": str(columns["family"][r]), "law": str(columns["law"][r]),
         "theta": float(columns["theta"][r]),
         "target_saturation": float(columns["target_saturation"][r]),
         "diagnostics": bool(columns["diagnostics"][r]),
         "seed": int(columns["seed"][r])}
        for r in range(len(columns["instance_id"]))
    ]


### Per-instance protocol

def repetitions_for(D, repetitions, repetitions_floor=None, reference_D=300):
    # fewer repetitions at large D, where the transition is sharper and the LPs are costlier
    if repetitions_floor is None:
        return repetitions

    return int(max(repetitions_floor, min(repetitions, repetitions * (reference_D / max(D, 1)) ** 2)))


def run_instance(spec, r_max=3.0, repetitions=200, repetitions_floor=None,
                 robust_repetitions=20, runtime_repetitions=20, runtime_points=10,
                 pilot_max=50000, direction_samples=10000):
    k, n, theta = spec["k"], spec["n"], spec["theta"]
    k = k if np.isinf(k) else int(k)

    rng = np.random.default_rng(spec["seed"])

    M, mask, family_param = sample_ensemble_instance(k, n, spec["family"], spec["law"], theta, rng)

    i, j = edge_indices(mask)
    m = len(i)
    D = target_dimension(k, m)
    s_max = int(np.ceil(r_max * D))
    runtime_ratios = np.round(np.linspace(1.2, r_max, runtime_points), 3)

    X, gamma = informed_point(k, M)
    saturation = gamma * np.max(np.abs(M))

    instance = {
        "instance_id": spec["instance_id"], "arm": spec["arm"], "k": float(k), "n": n,
        "family": spec["family"], "family_param": family_param, "law": spec["law"],
        "theta": theta, "m": m, "D": D, "gamma_hat": gamma, "saturation": saturation,
        "D_eff": np.nan, "sat_edges": saturated_edges(X, i, j), "D_eff_predicted": np.nan,
        "spectral_gap": np.nan, "beta_upper": np.nan, "kappa_0": np.nan,
        "beta_lower": np.nan, "pulse_space_ok": float(pulse_space_ok(k, n, s_max)),
        "discarded": 0.0, "seed": spec["seed"],
        "hash": hashlib.sha1(np.ascontiguousarray(M).tobytes()).hexdigest()[:16],
    }
    instance["D_eff_predicted"] = D - (1 if k == 2 else 2) * instance["sat_edges"]

    # saturated instances are discarded from the base ensemble, but recorded
    if spec["arm"] == "base" and saturation > 1 - TRIVIAL_SATURATION:
        instance["discarded"] = 1.0
        return instance, [], []

    L = psd_factor(X)
    m_vec = real_embedding(k, M[i, j].reshape(m, 1)).ravel()
    target = gamma * m_vec

    # diagnostics, on a subsample of the instances only
    basis = None
    if spec["diagnostics"]:
        N = int(min(max(1e4, 100 * D), pilot_max))
        Sigma, Y = pilot_samples(k, L, n, N, i, j, target, rng, keep=min(direction_samples, N))

        D_eff, vals, vecs, gap = effective_dimension(Sigma)
        basis = vecs[:, :D_eff] if D_eff > 0 else None

        kappa_0 = directional_kurtosis(Y, vecs, rng)

        instance.update({
            "D_eff": D_eff, "spectral_gap": gap,
            "beta_upper": beta_upper(Y, vecs, rng), "kappa_0": kappa_0,
            "beta_lower": 1.0 / (4 * kappa_0),
        })

    thresholds, runtimes = [], []

    for rep in range(repetitions_for(D, repetitions, repetitions_floor)):
        A = sample_chi(k, L, n, s_max, i, j, rng)

        s_star = min_successful_pulses(A, m_vec, gamma, 0, s_max)

        # at the threshold: overshoot of alpha over ghat, and whether ghat * m itself
        # lies in the convex hull
        alpha_star, alpha_min_star, conjecture, robust = np.nan, np.nan, np.nan, np.nan

        if s_star is not None:
            alpha_star = solve_restricted_lp(A[:, :s_star], m_vec)[0]
            alpha_min_star = solve_restricted_lp(A[:, :s_star], m_vec, maximise=False)[0]
            conjecture = float(solve_feasibility_lp(A[:, :s_star], target)[0])

            if rep < robust_repetitions and basis is not None:
                robust = float(robustly_successful(A[:, :s_star], m_vec, gamma, basis, rng))

        thresholds.append({
            "instance_id": spec["instance_id"], "repetition": rep, "D": D, "s_max": s_max,
            "s_star": np.nan if s_star is None else s_star,
            "r_star": np.nan if s_star is None else s_star / D,
            "alpha_star": alpha_star, "alpha_min_star": alpha_min_star,
            "conjecture_lp": conjecture, "robust_s_star": robust,
        })

        if rep < runtime_repetitions:
            for r in runtime_ratios:
                s = int(np.ceil(r * D))
                alpha, lam = solve_restricted_lp(A[:, :s], m_vec)
                alpha_min, _ = solve_restricted_lp(A[:, :s], m_vec, maximise=False)

                runtimes.append({
                    "instance_id": spec["instance_id"], "repetition": rep, "r": r, "s": s,
                    "alpha": alpha, "alpha_min": alpha_min,
                    "run_time": gamma / alpha if alpha else np.nan,
                    "support": np.nan if lam is None else int(np.sum(lam > LP_TOLERANCE)),
                })

    return instance, thresholds, runtimes


### Streaming CSV backups

TABLES = {
    "instances": INSTANCE_COLUMNS,
    "thresholds": THRESHOLD_COLUMNS,
    "runtimes": RUNTIME_COLUMNS,
}


def backup_files(results_file):
    directory = (RESULTS_DIR / results_file).parent
    prefix = Path(results_file).stem

    return {name: directory / f"{prefix}_{name}_backup.csv" for name in TABLES}


def read_backups(results_file):
    files = backup_files(results_file)
    tables = {}

    for name, columns in TABLES.items():
        path = files[name]

        if not (path.exists() and path.stat().st_size):
            tables[name] = pd.DataFrame(columns=columns)
            continue

        try:
            df = pd.read_csv(path)
        except pd.errors.ParserError:
            # a killed process can leave a truncated last line
            df = pd.read_csv(path, on_bad_lines="skip")
            print(f"{path.name}: dropped a malformed line (interrupted write?)")

        tables[name] = df[df["instance_id"].notna()]

    return tables


def completed_instances(tables):
    """Ids of the instances whose rows are all on disk."""
    instances, thresholds = tables["instances"], tables["thresholds"]

    if not len(instances):
        return set()

    written = thresholds.groupby("instance_id").size() if len(thresholds) else pd.Series(dtype=int)

    return {
        int(row.instance_id) for row in instances.itertuples()
        if row.discarded == 1.0 or written.get(row.instance_id, 0) > 0
    }


def assemble_feasibility(results_file="paper_results/feasibility.hdf5", **attrs):
    """Build the results file from the CSV backups, e.g. of an unfinished run."""
    tables = read_backups(results_file)
    done = completed_instances(tables)

    with h5py.File(RESULTS_DIR / results_file, "w") as f:
        f.attrs["ray_tolerance"] = RAY_TOLERANCE
        f.attrs["lp_tolerance"] = LP_TOLERANCE
        f.attrs["assembled_from_backup"] = True

        for key, value in attrs.items():
            f.attrs[key] = value

        for name, columns in TABLES.items():
            rows = tables[name]
            rows = rows[rows["instance_id"].astype(int).isin(done)]

            write_table(f.create_group(name), rows, columns)

    print(f"Assembled {results_file} from {len(done)} completed instances: "
          + ", ".join(f"{name} {len(tables[name])}" for name in TABLES))

    return done


### Parallel driver

K_INFINITY_COST = 1.3  # k = inf is the slowest of the complex k at equal D


def _estimated_cost(spec):
    """Cost proxy for longest-first scheduling, from D alone."""
    n, family = spec["n"], spec["family"]
    k = spec["k"] if np.isinf(spec["k"]) else int(spec["k"])

    edges = {"complete": n * (n - 1) // 2, "complete_bipartite": (n // 2) * (n - n // 2),
             "tree": n - 1, "lattice": 2 * n, "regular": 3 * n}.get(family, n * (n - 1) // 4)

    return (K_INFINITY_COST if np.isinf(k) else 1.0) * target_dimension(k, edges) ** 3.5


def _run_instance_task(args):
    spec, kwargs = args

    try:
        return run_instance(spec, **kwargs)
    except Exception as e:
        print(f"Instance {spec['instance_id']} crashed with error: {e}")
        return None, [], []


def run_feasibility(specs, results_file="paper_results/feasibility.hdf5", cores=16,
                    resume=True, order_by_cost=True, **kwargs):
    """Run the protocol over an ensemble in parallel.

    Rows are streamed to CSV backups as the instances finish, and with resume=True a
    restart skips the instances already there. The results file is written at the end.
    Export OMP_NUM_THREADS=1 before starting python to keep the workers single threaded.
    """
    files = backup_files(results_file)
    files["instances"].parent.mkdir(parents=True, exist_ok=True)

    collected = {name: [] for name in TABLES}
    done_ids = set()

    if resume:
        previous = read_backups(results_file)
        done_ids = completed_instances(previous)

        for name in TABLES:
            rows = previous[name]
            collected[name] = rows[rows["instance_id"].astype(int).isin(done_ids)].to_dict("records")

        if done_ids:
            specs = [spec for spec in specs if spec["instance_id"] not in done_ids]
            print(f"Resuming: {len(done_ids)} instances recovered from the backups, "
                  f"{len(specs)} still to run.")

    if order_by_cost:
        specs = sorted(specs, key=_estimated_cost, reverse=True)

    append = bool(done_ids)

    with contextlib.ExitStack() as stack:
        writers = {}
        for name, columns in TABLES.items():
            f = stack.enter_context(open(files[name], "a" if append else "w", newline=""))
            writers[name] = (csv.DictWriter(f, fieldnames=columns), f)

            if not append:
                writers[name][0].writeheader()

        executor = stack.enter_context(concurrent.futures.ProcessPoolExecutor(max_workers=cores))
        futures = [executor.submit(_run_instance_task, (spec, kwargs)) for spec in specs]

        for done, future in enumerate(concurrent.futures.as_completed(futures)):
            instance, thresholds, runtimes = future.result()

            if instance is None:
                continue

            for name, rows in [("instances", [instance]), ("thresholds", thresholds),
                               ("runtimes", runtimes)]:
                collected[name].extend(rows)

                writer, f = writers[name]
                writer.writerows(rows)
                f.flush()

            print(f"Finished instance {instance['instance_id']} "
                  f"({done + 1}/{len(specs)}): k={instance['k']} n={instance['n']} "
                  f"{instance['family']}/{instance['law']} D={instance['D']} "
                  f"sigma={instance['saturation']:.3f}", flush=True)

    with h5py.File(RESULTS_DIR / results_file, "w") as f:
        f.attrs["ray_tolerance"] = RAY_TOLERANCE
        f.attrs["lp_tolerance"] = LP_TOLERANCE

        for key, value in kwargs.items():
            f.attrs[key] = value

        for name, columns in TABLES.items():
            write_table(f.create_group(name), collected[name], columns)


def read_feasibility(results_file="paper_results/feasibility.hdf5"):
    with h5py.File(RESULTS_DIR / results_file, "r") as f:
        return {name: read_table(f[name]) for name in f}
