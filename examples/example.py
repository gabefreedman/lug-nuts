#!/usr/bin/env python3
"""
Example script for running NUTS inference with numpyro using custom external
likelihood and gradient functions.
"""

import logging
import argparse
import json
import sys
import time

import numpy as np
import jax
import h5py
import configs.test_fns as tfs  # example likelihoods and gradients
import lugnuts.lugnuts as lgn
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("n-params-nuts")

DEFAULT_PARAM_INFO = [
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


def external_loglike_placeholder(*params: float) -> float:
    """
    Placeholder log-likelihood: a multivariate Gaussian centered at true parameter values.
    """
    true = np.array([1e6, 0.5, 0.1][: len(params)])
    p = np.array(params)
    sigma = np.array([1e5, 0.05, 0.01][: len(params)])
    resid = (p - true) / sigma
    return float(-0.5 * np.sum(resid**2))


def external_grad_placeholder(*params: float) -> np.ndarray:
    """
    Placeholder gradient for above log-likelihood function.
    """
    true = np.array([1e6, 0.5, 0.1][: len(params)])
    p = np.array(params)
    sigma = np.array([1e5, 0.05, 0.01][: len(params)])
    grad = -(p - true) / (sigma**2)
    return grad


def main(args):

    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    # Set likelihood and gradient functions
    global external_loglike, external_grad
    external_loglike = external_loglike_placeholder
    external_grad = external_grad_placeholder

    param_info = DEFAULT_PARAM_INFO.copy()
    free_mask = DEFAULT_FREE_MASK.copy()
    fixed_values = DEFAULT_FIXED_VALUES.copy()
    truths = DEFAULT_PARAM_TRUTHS.copy()

    if args.config is not None:
        logger.info(f"Loading param config from {args.config}")
        with open(args.config, "r") as f:
            config = json.load(f)
        external_loglike = ll_funcs[config.get("ll_func")]
        external_grad = grad_funcs[config.get("ll_func")]
        param_info = config.get("param_info")
        free_mask = config.get("free_mask")
        fixed_values = [None] * len(param_info)
        fixed_values = config.get("fixed_vals", fixed_values)
        truths = config.get("truths")

    logger.info("Parameter info:")
    for i, p in enumerate(param_info):
        logger.info(
            f"  {i}: name={p['name']}, bounds=[{p['lower']}, {p['upper']}], \
                transform={p.get('transform', 'identity')}, prior={p.get('prior', 'uniform')}, \
                free={free_mask[i]}, fixed_value={fixed_values[i]}"
        )

    start = time.time()
    lgn.run_numpyro_nuts(
        param_info=param_info,
        free_mask=free_mask,
        fixed_values=fixed_values,
        ll_fn=external_loglike,
        grad_fn=external_grad,
        n_adapt=args.n_adapt,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        seed=args.seed,
        target_accept=0.8,
        hdf5_out=args.outfile,
    )
    duration = time.time() - start
    logger.info(f"Finished sampling in {duration:.1f} seconds.")

    # Load samples and plot 1d posteriors
    with h5py.File(args.outfile, "r") as f:
        samples = f["chains"][:]

    if args.plots:
        n_params = len(param_info)
        flat = samples.reshape(-1, n_params)

        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
        for i in range(n_params):
            ax = axes[i] if n_params > 1 else axes
            ax.hist(flat[:, i], bins=25, density=True, histtype="step", lw=3)
            if truths[i] is not None:
                ax.axvline(truths[i], color="black", linestyle="--", lw=3)
            ax.set_xlabel(param_info[i]["name"], fontsize=16)
            ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NUTS sampling for model with user-defined likelihood."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON file with parameter config (configs/[alt1.json, alt2.json, alt3.json] as options).",
    )
    parser.add_argument("--n_adapt", type=int, default=1000)
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--n_chains", type=int, default=1)
    parser.add_argument(
        "--outfile",
        type=str,
        default="chains.h5",
        help="HDF5 file to write posterior samples.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--plots", action="store_true", help="Generate posterior plots")
    args = parser.parse_args()

    main(args)
