from functions.expectation import get_k_expectation_inv
from functions.config import LUTS_DIR
from scipy.interpolate import RegularGridInterpolator
import concurrent.futures
import numpy as np


def inversion_wrapper(args):
    k, i, j, w = args
    k_exp_inv = get_k_expectation_inv(k)

    return i, j, k_exp_inv(w)


def compute_k_LUT(k, gridsize=1000, file=None):
    """Tabulate the inverse of k_expectation on a gridsize x gridsize grid over [-1, 1]^2."""
    if file is None:
        file = f'{k}_LUT.npz'

    x, y = np.linspace(-1, 1, gridsize), np.linspace(-1, 1, gridsize)

    Z = np.full((gridsize, gridsize), np.nan, dtype=np.complex128)

    tasks = [
        (k, i, j, x[i] + 1j * y[j])
        for i in range(gridsize) for j in range(gridsize)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for i, j, z in executor.map(inversion_wrapper, tasks, chunksize=1000):
            Z[i, j] = z

    np.savez_compressed(LUTS_DIR / file, x=x, y=y, Z=Z)


def load_k_LUT(k, gridsize=1000, file=None):
    """Load the LUT of k, computing it first if it does not exist yet."""
    if file is None:
        file = f'{k}_LUT.npz'

    try:
        with np.load(LUTS_DIR / file) as arrays:
            x, y, Z = arrays['x'], arrays['y'], arrays['Z']

    except OSError:
        print("No LUT found, generating new one")

        compute_k_LUT(k, gridsize=gridsize, file=file)

        with np.load(LUTS_DIR / file) as arrays:
            x, y, Z = arrays['x'], arrays['y'], arrays['Z']

    return x, y, Z


def load_k_interpol(k, gridsize=1000, file=None):
    """Inverse of k_expectation by linear interpolation of the LUT."""
    x, y, Z = load_k_LUT(k, gridsize, file)

    interpolator = RegularGridInterpolator((x, y), Z, method='linear', bounds_error=False, fill_value=np.nan)

    def k_exp_inv(w):
        return interpolator((np.real(w), np.imag(w)))

    return k_exp_inv


def k_exp_inv_for(k):
    return get_k_expectation_inv(k) if k in [2, 4] else load_k_interpol(k)
