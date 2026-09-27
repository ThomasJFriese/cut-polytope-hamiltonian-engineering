from functions.compute_lut import k_exp_inv_for
from functions.config import tol, RESULTS_DIR
from functions.instances import load_hollow_matrix
import numpy as np
import h5py
import cvxpy as cp
import itertools


def solve_lp(H_flat, X_corr):
    """Maximise alpha s.t. alpha * H = X_corr @ lam with lam in the simplex.

    Returns the quantum run time 1 / alpha and lam, or (nan, None) if infeasible.
    """
    lam = cp.Variable(X_corr.shape[1])
    alpha = cp.Variable(1)

    W = X_corr @ lam

    constraints = [
        alpha[0] * H_flat == W,
        0 <= lam,
        lam <= 1,
        cp.sum(lam) == 1,
        tol <= alpha
    ]

    problem = cp.Problem(objective=cp.Maximize(alpha[0]), constraints=constraints)

    try:
        problem.solve(solver='HiGHS', highs_options={"solver": "simplex"}, verbose=False)
    except (cp.SolverError, ValueError):
        return np.nan, None

    if problem.status != 'optimal':
        return np.nan, None

    q_time = np.nan if alpha.value[0] < tol else 1 / alpha.value[0]

    return q_time, lam.value


def solve_pulses(H, mask, X_ang):
    """LP over the pulses X_ang (one row of phases per pulse) on the interaction edges."""
    i, j = np.nonzero(np.triu(mask, k=1))
    X_corr = (X_ang[:, i] * np.conj(X_ang[:, j])).T
    H_flat = H[np.triu(mask, k=1) == 1]

    return solve_lp(H_flat, X_corr)


def cut_poly_ray(H, k_exp_inv, tolerance=tol):
    """Ray binary search: the largest gamma for which Z = 1 + k_exp_inv(gamma * H) is psd.

    Returns Z and gamma, where 1 / gamma is the expected quantum run time.
    """
    n = len(H)
    ident = np.identity(n, dtype=np.complex128)

    H_eigvals = np.linalg.eigvalsh(H)

    if H_eigvals[0] > -tol: # H is 0
        return np.array([]), np.inf

    lower = 0
    upper = - 1.0 / H_eigvals[0]

    while upper - lower > tolerance:
        step = (upper + lower) / 2

        Z = ident + k_exp_inv(step * H)

        try:
            Z_eigvals = np.linalg.eigvalsh(Z)
        except np.linalg.LinAlgError:
            upper = step
            continue

        if Z_eigvals[0] < tol:
            upper = step
        else:
            lower = step

    Z = ident + k_exp_inv(lower * H)

    return Z, lower


def k_rounding_func(k, z):
    """Index of the sector of width 2 pi / k containing the angle of z (the angle itself for k = inf)."""
    phi = np.angle(z)
    phi = np.where(phi < 0, 2 * np.pi + phi, phi)

    if np.isinf(k):
        return phi

    interval_edges = [i * 2 * np.pi / k for i in range(k)]

    return np.searchsorted(interval_edges, phi, side='right') - 1


def gaussian_rounding(L, n, k, pulse_n, rng=np.random):
    """pulse_n pulses, one per row, by k-rounding L @ g for complex Gaussian vectors g."""
    REAL = rng.normal(0, 1, size=(n, pulse_n))
    IMAG = rng.normal(0, 1, size=(n, pulse_n))
    NORMAL = REAL + 1j * IMAG

    return k_rounding_func(k, L @ NORMAL).transpose()


def pulse_phases(k, X):
    return np.exp(1j * X) if np.isinf(k) else np.exp(1j * 2 * np.pi / k * X)


def explicit_construction(instance_file, results_file, n_bases=2000, seed=0):
    """Quantum run time of the explicit construction of arXiv:2511.11404 (k = 2).

    The free x and y diagonal entries are set to 0, which reduces the construction to H - lambda_min.
    The run time depends on the basis of degenerate eigenspaces, so n_bases random bases are sampled.
    """
    n, H, mask = load_hollow_matrix(instance_file)

    assert np.allclose(H.imag, 0), "Explicit construction is only implemented for real H."
    H = H.real

    eigvals, eigvecs = np.linalg.eigh(H - np.linalg.eigvalsh(H)[0] * np.identity(n))

    groups = np.split(np.arange(n), np.nonzero(np.diff(eigvals) > np.sqrt(tol))[0] + 1)
    deg_groups = [g for g in groups if len(g) > 1 and np.mean(eigvals[g]) > tol]

    t_fixed = sum(eigvals[g[0]] * np.max(eigvecs[:, g[0]] ** 2) for g in groups if len(g) == 1)
    ts = np.full(n_bases if deg_groups else 1, t_fixed)

    rng = np.random.default_rng(seed)

    for s in range(len(ts)):
        for g in deg_groups:
            Q, R = np.linalg.qr(rng.normal(size=(len(g), len(g))))
            W = eigvecs[:, g] @ (Q * np.sign(np.diag(R))) # Haar random basis
            ts[s] += np.mean(eigvals[g]) * np.sum(np.max(W ** 2, axis=0))

    with h5py.File(RESULTS_DIR / results_file, "w") as f:
        f.attrs["instance"] = instance_file
        f.attrs["algorithm"] = "explicit construction"
        f.attrs["k"] = 2
        f.attrs["q_time"] = np.median(ts)
        f.attrs["q_time_min"] = np.min(ts)
        f.attrs["q_time_max"] = np.max(ts)
        f.attrs["n_bases"] = len(ts)
        f.attrs["degenerate_dims"] = np.array([len(g) for g in deg_groups], dtype=int)


def exact_algo(k, instance_file, results_file, fix_last=False):
    """Optimal solution: the LP over all k^n pulses, or k^(n-1) with the last phase fixed."""
    n, H, mask = load_hollow_matrix(instance_file)

    omega = np.exp(1j * 2 * np.pi / k)
    X = omega ** np.array(list(itertools.product(range(k), repeat=n - fix_last)))

    if fix_last: # global phase invariance
        X = np.append(X, np.ones((len(X), 1)), axis=1)

    q_time, lam = solve_pulses(H, mask, X)

    with h5py.File(RESULTS_DIR / results_file, "w") as f:
        f.attrs["instance"] = instance_file
        f.attrs["algorithm"] = "exact"
        f.attrs["k"] = str(k)
        f.attrs["tolerance"] = tol
        f.attrs["q_time"] = q_time

        if lam is not None:
            f.create_dataset("ev_times", data=lam, compression="gzip")


def eff_relax_algo(k, pulse_n, instance_file, results_file, best_of_runs=1):
    """Uninformed LP relaxation: the LP over uniformly random pulses."""
    n, H, mask = load_hollow_matrix(instance_file)

    best_q_time, best_lam, best_X = np.inf, None, None

    for _ in range(best_of_runs):
        X = gaussian_rounding(np.identity(n=n, dtype=np.complex128), n, k, pulse_n)

        q_time, lam = solve_pulses(H, mask, pulse_phases(k, X))

        if q_time < best_q_time:
            best_q_time, best_lam, best_X = q_time, lam, X

    with h5py.File(RESULTS_DIR / results_file, "w") as f:
        f.attrs["instance"] = instance_file
        f.attrs["algorithm"] = "eff_relax"
        f.attrs["k"] = str(k)
        f.attrs["tolerance"] = tol
        f.attrs["pulse_n"] = pulse_n
        f.attrs["q_time"] = best_q_time

        if best_X is not None:
            f.create_dataset("pulses", data=best_X, compression="gzip")

        if best_lam is not None:
            f.create_dataset("ev_times", data=best_lam, compression="gzip")


def cut_poly_algo(k, pulse_n, instance_file, results_file, best_of_runs=1):
    """Informed LP algorithm: the LP over pulses rounded from the ray binary search solution."""
    n, H, mask = load_hollow_matrix(instance_file)

    best_q_time, best_lam, best_alpha, best_X = np.inf, None, None, None

    if np.linalg.norm(H) < tol:
        best_q_time = 0

    else:
        Z, alpha_exp = cut_poly_ray(H, k_exp_inv_for(k))

        L = np.linalg.cholesky(Z)

        for _ in range(best_of_runs):
            X = gaussian_rounding(L, n, k, pulse_n)

            q_time, lam = solve_pulses(H, mask, pulse_phases(k, X))

            if q_time < best_q_time:
                best_q_time, best_lam, best_alpha, best_X = q_time, lam, alpha_exp, X

    with h5py.File(RESULTS_DIR / results_file, "w") as f:
        f.attrs["instance"] = instance_file
        f.attrs["algorithm"] = "cut_poly"
        f.attrs["k"] = str(k)
        f.attrs["tolerance"] = tol
        f.attrs["pulse_n"] = pulse_n

        if best_alpha is not None:
            f.attrs["q_time_exp"] = 1.0 / best_alpha

        f.attrs["q_time"] = best_q_time
        f.attrs["best_of_runs"] = best_of_runs

        if best_X is not None:
            f.create_dataset("pulses", data=best_X, compression="gzip")

        if best_lam is not None:
            f.create_dataset("ev_times", data=best_lam, compression="gzip")
