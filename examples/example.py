#!/usr/bin/env python3
"""
NUTS inference for an n-parameter model with:
 - fixed uniform priors on [lower_i, upper_i]
 - optional per-parameter transform ('identity' or 'log')
 - optional prior type 'uniform' (uniform in physical variable) or 'uniform_log' (uniform in log-space)
 - ability to hold a subset of parameters fixed
 - wrapper for external likelihood and gradient (external_loglik, external_grad)
 - uses JAX + NumPyro NUTS (with potential_fn) and saves samples to HDF5
"""

import logging
import argparse
import json
import sys
import os
import time

import numpy as np
import jax
import h5py
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import configs.test_fns as tfs  # import test likelihoods and gradients
import lugnuts.lugnuts as lgn


# --------------- Configure logging ---------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("n-params-nuts")

DEFAULT_PARAM_INFO = [
    # Example: first parameter is MBH mass; we want to scale via log
    {"name": "M", "lower": 1e5, "upper": 1e8, "transform": "log", "prior": "uniform"},
    {
        "name": "q",
        "lower": 0.1,
        "upper": 0.99,
        "transform": "identity",
        "prior": "uniform",
    },
    {
        "name": "a",
        "lower": 0.0,
        "upper": 0.999,
        "transform": "identity",
        "prior": "uniform",
    },
    # add more parameter entries as needed...
]
DEFAULT_FREE_MASK = [True] * len(DEFAULT_PARAM_INFO)
DEFAULT_FIXED_VALUES = [None] * len(DEFAULT_PARAM_INFO)
DEFAULT_PARAM_TRUTHS = [1e6, 0.5, 0.1]

# ----------------- External likelihood functions -----------------
ll_funcs = {
    "mixed_gaussian": tfs.external_loglik_alt1,
    "rosenbrock": tfs.external_loglik_alt2,
    "corr_gaussian": tfs.external_loglik_alt3,
}
grad_funcs = {
    "mixed_gaussian": tfs.external_grad_alt1,
    "rosenbrock": tfs.external_grad_alt2,
    "corr_gaussian": tfs.external_grad_alt3,
}


def external_loglik_placeholder(*params: float) -> float:
    """
    Dummy log-likelihood: a multivariate Gaussian centered at arbitrary "true" params.
    Replace this with your package call that returns a scalar log-likelihood.
    Signature must accept parameters as separate positional arguments (one per parameter).
    """
    # Example true parameter vector (for testing only)
    true = np.array([1e6, 0.5, 0.1][: len(params)])
    p = np.array(params)
    # gaussian log-likelihood (unnormalized)
    sigma = np.array([1e5, 0.05, 0.01][: len(params)])
    resid = (p - true) / sigma
    return float(-0.5 * np.sum(resid**2))


def external_grad_placeholder(*params: float) -> np.ndarray:
    """
    Gradient of the dummy log-likelihood above w.r.t. params.
    Replace with your package's gradient (numpy array).
    """
    true = np.array([1e6, 0.5, 0.1][: len(params)])
    p = np.array(params)
    sigma = np.array([1e5, 0.05, 0.01][: len(params)])
    grad = -(p - true) / (sigma**2)  # derivative of -0.5 * sum((p-true)^2 / sigma^2)
    return grad


def main(args):

    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    # Set likelihood and gradient functions
    global external_loglik, external_grad
    external_loglik = external_loglik_placeholder
    external_grad = external_grad_placeholder

    param_info = DEFAULT_PARAM_INFO.copy()
    free_mask = DEFAULT_FREE_MASK.copy()
    fixed_values = DEFAULT_FIXED_VALUES.copy()
    truths = DEFAULT_PARAM_TRUTHS.copy()

    if args.config is not None:
        logger.info(f"Loading param config from {args.config}")
        with open(args.config, "r") as f:
            cfg = json.load(f)
        external_loglik = ll_funcs[cfg.get("ll_func")]
        external_grad = grad_funcs[cfg.get("ll_func")]
        param_info = cfg.get("param_info")
        free_mask = cfg.get("free_mask")
        fixed_values = [None] * len(param_info)
        fixed_values = cfg.get("fixed_vals", fixed_values)
        truths = cfg.get("truths")

    logger.info("Parameter info:")
    for i, p in enumerate(param_info):
        logger.info(
            f"  {i}: name={p['name']}, bounds=[{p['lower']}, {p['upper']}], \
                transform={p.get('transform', 'identity')}, prior={p.get('prior', 'uniform')}, \
                free={free_mask[i]}, fixed_value={fixed_values[i]}"
        )

    start = time.time()
    out = lgn.run_numpyro_nuts(
        param_info=param_info,
        free_mask=free_mask,
        fixed_values=fixed_values,
        ll_fn=external_loglik,
        grad_fn=external_grad,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        rng_seed=args.seed,
        target_accept=0.8,
        hdf5_out=args.out,
        logger=logger,
    )
    duration = time.time() - start
    logger.info(f"Finished sampling. Output file: {out}. Took {duration:.1f} seconds.")

    # Load samples and make a quick plot
    with h5py.File(out, "r") as hf:
        samples = hf["x_samples"][:]  # shape (chains, samples, n)
    chains, samples_count, n = samples.shape
    flat = samples.reshape(-1, n)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    for i in range(n):
        ax = axes[i] if n > 1 else axes
        ax.hist(flat[:, i], bins=50, density=True, alpha=0.7)
        if truths[i] is not None:
            ax.axvline(truths[i], color="r", linestyle="--", label="true")
        ax.set_title(f"Posterior for {param_info[i]['name']}")
        ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NUTS sampling for n-parameter model with fixed uniform priors and optional transforms."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON file with parameter config (overrides DEFAULT_PARAM_INFO).",
    )
    parser.add_argument("--num-warmup", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument(
        "--out",
        type=str,
        default="chains.h5",
        help="HDF5 file to write posterior samples (full x space).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    main(args)
