import numpy as np


def external_loglik_alt1(M, alpha, beta, gamma):
    true = np.array([1.0e6, 1.0, 100.0, 1e-3])
    sigma = np.array([1e5, 0.1, 10.0, 1e-4])
    p = np.array([M, alpha, beta, gamma])
    resid = (p - true) / sigma
    return float(-0.5 * np.sum(resid**2))


def external_grad_alt1(M, alpha, beta, gamma):
    true = np.array([1.0e6, 1.0, 100.0, 1e-3])
    sigma = np.array([1e5, 0.1, 10.0, 1e-4])
    p = np.array([M, alpha, beta, gamma])
    grad = -(p - true) / (sigma**2)
    return grad


def external_loglik_alt2(x, y, z, w, u):
    # Rosenbrock-like structure
    term1 = -100.0 * (y - x**2) ** 2 - (1 - x) ** 2
    term2 = -50.0 * (z - 1e-2) ** 2
    term3 = -0.5 * ((w - 1000.0) / 200.0) ** 2
    term4 = -0.5 * ((u - 10.0) / 2.0) ** 2
    return float(term1 + term2 + term3 + term4)


def external_grad_alt2(x, y, z, w, u):
    # Gradients for each parameter
    dx = -400.0 * (y - x**2) * (-2 * x) - 2 * (1 - x)
    dy = -200.0 * (y - x**2)
    dz = -100.0 * (z - 1e-2)
    dw = -(w - 1000.0) / (200.0**2)
    du = -(u - 10.0) / (2.0**2)
    return np.array([dx, dy, dz, dw, du])


def external_loglik_alt3(a, b, c, d, e, f):
    p = np.array([a, b, c, d, e, f])
    true = np.array([1e4, 1.0, 0.01, 100.0, 5.0, 0.01])
    diff = p - true

    # Precision matrix with correlations (symmetric positive definite)
    prec = np.array(
        [
            [1e-8, 0, 0, 0, 0, 0],
            [0, 10.0, 1.0, 0, 0, 0],
            [0, 1.0, 5000, 0, 0, 0],
            [0, 0, 0, 0.05, 0.01, 0],
            [0, 0, 0, 0.01, 1.0, 0],
            [0, 0, 0, 0, 0, 1e6],
        ]
    )
    return float(-0.5 * diff @ prec @ diff)


def external_grad_alt3(a, b, c, d, e, f):
    p = np.array([a, b, c, d, e, f])
    true = np.array([1e4, 1.0, 0.01, 100.0, 5.0, 0.01])
    diff = p - true
    prec = np.array(
        [
            [1e-8, 0, 0, 0, 0, 0],
            [0, 10.0, 1.0, 0, 0, 0],
            [0, 1.0, 5000, 0, 0, 0],
            [0, 0, 0, 0.05, 0.01, 0],
            [0, 0, 0, 0.01, 1.0, 0],
            [0, 0, 0, 0, 0, 1e6],
        ]
    )
    grad = -(prec @ diff)
    return grad
