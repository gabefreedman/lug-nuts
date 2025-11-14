import numpy as np


def external_loglik_alt1(M, alpha, beta, gamma):
    # Multivariate Gaussian
    params = np.array([M, alpha, beta, gamma])
    true = np.array([1.0e6, 1.0, 100.0, 1e-3])
    sigma = np.array([1e5, 0.1, 10.0, 1e-4])
    res = (params - true) / sigma
    return float(-0.5 * np.sum(res**2))


def external_grad_alt1(M, alpha, beta, gamma):
    params = np.array([M, alpha, beta, gamma])
    true = np.array([1.0e6, 1.0, 100.0, 1e-3])
    sigma = np.array([1e5, 0.1, 10.0, 1e-4])
    grad = -(params - true) / (sigma**2)
    return grad


def external_loglik_alt2(x, y, z, w, u):
    # Rosenbrock-like distribution
    A = -100.0 * (y - x**2) ** 2 - (1 - x) ** 2
    B = -50.0 * (z - 1e-2) ** 2
    C = -0.5 * ((w - 1000.0) / 200.0) ** 2
    D = -0.5 * ((u - 10.0) / 2.0) ** 2
    return float(A + B + C + D)


def external_grad_alt2(x, y, z, w, u):
    dx = -400.0 * (y - x**2) * (-2 * x) - 2 * (1 - x)
    dy = -200.0 * (y - x**2)
    dz = -100.0 * (z - 1e-2)
    dw = -(w - 1000.0) / (200.0**2)
    du = -(u - 10.0) / (2.0**2)
    return np.array([dx, dy, dz, dw, du])


def external_loglik_alt3(a, b, c, d, e, f):
    # Multivariate correlated Gaussian
    p = np.array([a, b, c, d, e, f])
    true = np.array([1e4, 1.0, 0.01, 100.0, 5.0, 0.01])
    diff = p - true
    cov = np.array(
        [
            [1e-8, 0, 0, 0, 0, 0],
            [0, 10.0, 1.0, 0, 0, 0],
            [0, 1.0, 5e3, 0, 0, 0],
            [0, 0, 0, 0.05, 0.01, 0],
            [0, 0, 0, 0.01, 1.0, 0],
            [0, 0, 0, 0, 0, 1e6],
        ]
    )
    return float(-0.5 * diff @ cov @ diff)


def external_grad_alt3(a, b, c, d, e, f):
    p = np.array([a, b, c, d, e, f])
    true = np.array([1e4, 1.0, 0.01, 100.0, 5.0, 0.01])
    diff = p - true
    cov = np.array(
        [
            [1e-8, 0, 0, 0, 0, 0],
            [0, 10.0, 1.0, 0, 0, 0],
            [0, 1.0, 5e3, 0, 0, 0],
            [0, 0, 0, 0.05, 0.01, 0],
            [0, 0, 0, 0.01, 1.0, 0],
            [0, 0, 0, 0, 0, 1e6],
        ]
    )
    grad = -(cov @ diff)
    return grad
