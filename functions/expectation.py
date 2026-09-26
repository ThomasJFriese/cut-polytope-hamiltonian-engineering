from functions.config import tol
import numpy as np
from scipy.optimize import root
from scipy.special import ellipk, ellipe


def k_expectation(k, z):
    """Expected correlation of two k-rounded complex Gaussians with correlation z."""
    if k == 2:
        return (2 / np.pi) * np.arcsin(z)

    elif k == 4:
        return (2 / np.pi) * np.arcsin(np.real(z)) + 1j * (2 / np.pi) * np.arcsin(np.imag(z))

    elif np.isinf(k):
        if abs(z) <= tol or abs(z) >= 1 - tol:
            return z

        return (z / (np.abs(z) ** 2)) * (ellipe(np.abs(z) ** 2) - (1 - np.abs(z) ** 2) * ellipk(np.abs(z) ** 2))

    else:
        omega = np.exp(1j * 2 * np.pi / k)

        return (k * (2 - omega ** (-1) - omega) / (8 * np.pi ** 2)) * sum(omega ** j * np.arccos(- np.real(omega ** (-j) * z)) ** 2 for j in range(k))


def get_k_expectation_inv(k):
    """Inverse of k_expectation, in closed form for k = 2, 4 and by root finding otherwise."""
    if k == 2:
        def k_exp_inv(w):
            return np.sin((np.pi / 2) * np.real(w))

    elif k == 4:
        def k_exp_inv(w):
            return np.sin((np.pi / 2) * np.real(w)) + 1j * np.sin((np.pi / 2) * np.imag(w))

    else:
        def k_exp_inv(w):
            def residual(v):
                diff = k_expectation(k, v[0] + 1j * v[1]) - w
                return np.array([np.real(diff), np.imag(diff)])

            solution = root(residual, x0=np.array([np.real(w), np.imag(w)]), method='hybr')

            if not solution.success:
                return np.nan

            return solution.x[0] + 1j * solution.x[1]

    return k_exp_inv
