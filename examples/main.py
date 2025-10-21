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

from typing import List, Dict, Any, Tuple
import logging
import argparse
import json
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import h5py
import configs.test_fns as tfs  # import test likelihoods and gradients

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# Try importing numpyro (preferred for automatic NUTS adapt). If not present, exit with an explanation.
try:
    import numpyro
    from numpyro.distributions import Normal
    from numpyro.infer import MCMC, NUTS, init_to_value

    HAS_NUMPYRO = True
except Exception:
    HAS_NUMPYRO = False

# --------------- Configure logging ---------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("n-params-nuts")

# ----------------- User: model configuration -----------------
# Example parameter configuration for n parameters. Replace with your actual model info.
# Each entry:
#  - name: string
#  - lower: physical lower bound (float)
#  - upper: physical upper bound (float)
#  - transform: 'identity' or 'log'  (how we parametrize internally; 'log' means param will be represented via log(x))
#  - prior: 'uniform' (uniform in x) or 'uniform_log' (uniform in log x)
# Notes on transforms and jacobians are given in the big comment block below.

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

# Which parameters are free (True) or fixed (False). If fixed, specify fixed_values array.
# By default sample all:
DEFAULT_FREE_MASK = [True] * len(DEFAULT_PARAM_INFO)
DEFAULT_FIXED_VALUES = [None] * len(
    DEFAULT_PARAM_INFO
)  # fill with numbers for fixed params
DEFAULT_PARAM_TRUTHS = [1e6, 0.5, 0.1]  # example true values for plotting

# ----------------- User: external likelihood functions -----------------
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

# ----------------- Explanatory math (brief) -----------------
# Coordinate transforms used here (per-parameter, scalar z --> x):
#
# Let z be an unconstrained real (R). We map z -> s := sigmoid(z) in (0,1).
# 1) identity-bounded transform (maps z -> x in [L, U]):
#    x = L + (U - L) * s(z), where s(z) = 1/(1+exp(-z))
#    dx/dz = (U - L) * s * (1 - s)
#    => log |dx/dz| = log(U-L) + log(s) + log(1 - s)
#
# 2) log transform (map z -> x in [L, U] but parametrize log-scale):
#    u = log(L) + (log(U) - log(L)) * s(z)  # u in [log L, log U]
#    x = exp(u)
#    dx/dz = dx/du * du/ds * ds/dz
#          = exp(u) * (log U - log L) * s * (1 - s)
#          = x * (log U - log L) * s * (1 - s)
#    => log |dx/dz| = log x + log(log U - log L) + log(s) + log(1 - s)
#
# Prior in 'x' space:
#   if prior == 'uniform' over [L, U]:
#       log p_x(x) = -log(U - L)
#   if prior == 'uniform_log' (i.e., uniform in u = log x between log L and log U):
#       p_x(x) = (1 / (log U - log L)) * (1 / x)
#       log p_x(x) = -log(log U - log L) - log(x)
#
# Posterior in z space (after change of variables):
#   log p(z | data) = log L(data | x(z)) + log p_x(x(z)) + sum_i log |dx_i/dz_i|
# So the potential (negative log posterior) U(z) = -[loglik + log_prior_x + logabsdet]
# The sampler samples z (unconstrained), and we transform to x for the likelihood.
#
# If some parameters are fixed, they are substituted after transforming the sampled subset back to full x.
#
# ----------------- External likelihood wrapper (USER must replace) -----------------
# The user must supply these two functions that call their external package:
#   external_loglik(p0, p1, ..., p_{n-1}) -> float  (log-likelihood)
#   external_grad(p0, p1, ..., p_{n-1}) -> numpy array (shape (n,)) equal to d(loglik)/dparams
#
# For demonstration we provide a dummy gaussian log-likelihood and gradient. Replace these functions.


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


# Bind the placeholders to names expected by this script.
# Replace these assignments with imports/calls to your package that provide the two functions:
# external_loglik = external_loglik_placeholder
# external_grad = external_grad_placeholder

# ----------------- Utility: mapping between z_free and full x -----------------


def make_index_maps(free_mask: List[bool]) -> Tuple[List[int], Dict[int, int]]:
    """
    Given a boolean free_mask of length n, return:
      - free_indices: list of indices (0..n-1) that are free, in order
      - index_in_free: dict mapping original index -> position in free vector (only for free indices)
    """
    free_indices = [i for i, v in enumerate(free_mask) if v]
    index_in_free = {i: idx for idx, i in enumerate(free_indices)}
    return free_indices, index_in_free


def sigmoid(z: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(z)


def zfree_to_x_and_dx_dz(
    z_free: jnp.ndarray,
    param_info: List[Dict[str, Any]],
    free_mask: List[bool],
    fixed_values: List[Any],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Map z_free (shape (m,)) to full physical parameter vector x_full (shape (n,))
    Also compute vector of dx_i/dz_j but since each x_i depends on at most one z_j,
    the Jacobian reduces to a vector of diagonal entries for free params (shape (m,))
    Returns:
      x_full: jnp.array shape (n,)
      dx_dz_free: jnp.array shape (m,)
    This function is jax-traceable / jittable (uses jnp ops).
    """
    free_indices, index_in_free = make_index_maps(free_mask)
    x_list = []
    dx_dz_free_list = []

    # But better to use index mapping:
    for i, pinfo in enumerate(param_info):
        L = float(pinfo["lower"])
        U = float(pinfo["upper"])
        transform = pinfo.get("transform", "identity")
        if not free_mask[i]:
            # fixed param: use provided fixed value (must be not None)
            val = fixed_values[i]
            if val is None:
                raise ValueError(
                    f"Parameter {i} ('{pinfo.get('name')}') is fixed but fixed_values not provided."
                )
            x_list.append(jnp.array(val, dtype=jnp.float64))
            # no dx/dz contribution for fixed param
        else:
            # free param: consume next entry from z_free
            zpos = index_in_free[i]
            zval = z_free[zpos]
            s = sigmoid(zval)
            if transform == "identity":
                # x = L + (U-L) * s
                x = L + (U - L) * s
                dx_dz = (U - L) * s * (1.0 - s)
            elif transform == "log":
                if L <= 0:
                    raise ValueError(
                        f"Parameter {i} has 'log' transform but lower bound <= 0."
                    )
                Lu = jnp.log(L)
                Uu = jnp.log(U)
                Du = Uu - Lu
                u = Lu + Du * s  # u in [log L, log U]
                x = jnp.exp(u)
                dx_dz = x * Du * s * (1.0 - s)
            else:
                raise ValueError(
                    f"Unknown transform '{transform}' for parameter index {i}."
                )
            x_list.append(x)
            dx_dz_free_list.append(dx_dz)

    x_full = jnp.stack(x_list)
    if len(dx_dz_free_list) > 0:
        dx_dz_free = jnp.stack(dx_dz_free_list)
    else:
        dx_dz_free = jnp.zeros((0,), dtype=jnp.float64)
    return x_full, dx_dz_free


# ----------------- JAX wrapper for external likelihood + gradient -----------------
# We use jax.custom_vjp so that JAX can call our python external gradient when computing derivatives.
# Under the hood we use host_callback.call to invoke the Python functions at runtime and return results to JAX.
# This makes the wrapper correct and usable by jax.grad, but note that host-calls take the computation outside of
# pure-jitted JAX code and can be slower.


def _make_host_call_loglik(x_full_jax: jnp.ndarray) -> jnp.ndarray:
    """
    Use host_callback.call to invoke the python external_loglik function (which expects python scalars/arrays).
    Returns a jax scalar array (0-d).
    """

    def _py_loglik(x_np):
        # x_np is a numpy array (host side)
        # external_loglik expects separate args, unpack them
        try:
            val = external_loglik(*x_np)
            return np.array(val, dtype=np.float64)
        except Exception:
            # Throw a helpful message if external call fails
            logger.exception("external_loglik call failed on host.")
            raise

    # result_shape must be a jax.ShapeDtypeStruct describing a scalar float64
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    ret = jax.pure_callback(_py_loglik, result_shape, x_full_jax)
    return jnp.asarray(ret).reshape(())
    # return ret


def _make_host_call_grad(x_full_jax: jnp.ndarray) -> jnp.ndarray:
    """
    Use host_callback.call to invoke the python external_grad function.
    Returns a jax array of shape (n,) with dtype float64.
    """
    n = x_full_jax.shape[0]

    def _py_grad(x_np):
        try:
            g = external_grad(*x_np)
            g = np.asarray(g, dtype=np.float64)
            if g.shape != (n,):
                raise ValueError(
                    f"external_grad returned shape {g.shape} but expected {(n,)}"
                )
            return g
        except Exception:
            logger.exception("external_grad call failed on host.")
            raise

    result_shape = jax.ShapeDtypeStruct((n,), jnp.float64)
    ret = jax.pure_callback(_py_grad, result_shape, x_full_jax)
    return jnp.asarray(ret)


def make_loglik_of_z(param_info, free_mask, fixed_values):
    @jax.custom_vjp
    def loglik_of_z(z_free: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log-likelihood evaluated at x(z_free). This is the primal function used by the potential_fn.
        The custom VJP uses external_grad to compute d loglik / d z_free via chain rule:
        d loglik/dz_j = sum_i (d loglik/d x_i) * (d x_i / d z_j)
        where dx_i/dz_j is diagonal (only nonzero when x_i depends on z_j), so it's elementwise multiply.
        """
        x_full, dx_dz_free = zfree_to_x_and_dx_dz(
            z_free, param_info, free_mask, fixed_values
        )
        # call host-side loglik
        ll = _make_host_call_loglik(x_full)
        return ll

    def loglik_of_z_fwd(z_free):
        x_full, dx_dz_free = zfree_to_x_and_dx_dz(
            z_free, param_info, free_mask, fixed_values
        )
        ll = _make_host_call_loglik(x_full)
        # Save x_full and dx_dz_free for the backward pass (needed to compute grad)
        return ll, (x_full, dx_dz_free)

    def loglik_of_z_bwd(res, g):
        """
        res contains saved (x_full, dx_dz_free)
        g is cotangent (scalar)
        returns tuple with gradient w.r.t the input arguments of the forward function:
        (grad_z_free, None, None, None)  # the latter Nones correspond to (param_info, free_mask, fixed_vals)
        """
        x_full, dx_dz_free = res
        grad_x = _make_host_call_grad(x_full)  # shape (n,)
        # Extract grad_x entries corresponding to free params only:
        # Make a mask vector of booleans for free params
        # NOTE: free_mask is a python list; convert to jax bool array
        # We rely on the fact that dx_dz_free was computed in the same order as free indices
        free_inds, _ = make_index_maps(list(free_mask))
        # grad_x_free is grad_x[free_inds]
        grad_x_free = grad_x[jnp.array(free_inds, dtype=jnp.int32)]
        # chain rule: d loglik / dz_free = sum_i (d loglik / dx_i) * (dx_i / dz_free_i) (elementwise)
        grad_z = g * (dx_dz_free * grad_x_free)
        # return gradient tuple matching forward function args: (z_free, param_info, free_mask, fixed_values)
        return (grad_z,)

    # attach custom_vjp rules
    loglik_of_z.defvjp(loglik_of_z_fwd, loglik_of_z_bwd)

    return loglik_of_z


# ----------------- Log prior in x-space -----------------
def log_prior_x(x_full: jnp.ndarray, param_info: List[Dict[str, Any]]) -> jnp.ndarray:
    """
    Compute log prior sum(log p_x(x_i)) given prior types in param_info.
    Supported prior types: 'uniform' and 'uniform_log'.
    Returns scalar log prior.
    """
    logs = []
    for i, pinfo in enumerate(param_info):
        L = float(pinfo["lower"])
        U = float(pinfo["upper"])
        prior_type = pinfo.get("prior", "uniform")
        xi = x_full[i]
        if prior_type == "uniform":
            # Check within bounds (transforms should guarantee this)
            # log p = -log(U-L)
            logs.append(-jnp.log(U - L))
        elif prior_type == "uniform_log":
            # uniform in log x between log L and log U => p_x(x) = 1/(logU - logL) * 1/x
            logs.append(-jnp.log(jnp.log(U) - jnp.log(L)) - jnp.log(xi))
        else:
            raise ValueError(
                f"Unsupported prior type '{prior_type}' for parameter {i}."
            )
    return jnp.sum(jnp.stack(logs))


# ----------------- Potential function U(z_free) -----------------
def make_potential_fn(
    param_info: List[Dict[str, Any]], free_mask: List[bool], fixed_values: List[Any]
):
    """
    Return a JAX-callable potential_fn(z_free) that returns the scalar potential U(z_free) = -log posterior(z_free).
    Uses the loglik_of_z custom-vjp wrapper which will call external_loglik and external_grad as needed.
    """
    loglik_of_z = make_loglik_of_z(param_info, free_mask, fixed_values)

    def potential(z_free: jnp.ndarray) -> jnp.ndarray:
        # z_free is shape (m,)
        # compute x and dx/dz
        x_full, dx_dz_free = zfree_to_x_and_dx_dz(
            z_free, param_info, free_mask, fixed_values
        )
        # log-likelihood (external)
        ll = loglik_of_z(z_free)
        # log-prior in x-space
        lp = log_prior_x(x_full, param_info)
        # log abs det jacobian sum_j log |dx_j/dz_j| for free params
        # dx_dz_free may contain small values; we take log of abs value safely
        logabs = (
            jnp.sum(jnp.log(jnp.abs(dx_dz_free) + 1e-300))
            if dx_dz_free.shape[0] > 0
            else 0.0
        )
        log_posterior = ll + lp + logabs
        U = -log_posterior
        return U

    return jax.jit(potential)


def make_numpyro_model(param_info, free_mask, fixed_values):
    """
    Returns a NumPyro model function that samples unconstrained free parameters
    and adds the custom log-density (likelihood + Jacobian from transformations).

    Args:
        param_info: list of dicts describing parameters (lower, upper, transform)
        free_mask: list of bools
        fixed_values: list of values for fixed parameters
    """
    n_free = sum(free_mask)
    logdensity_fn = make_potential_fn(param_info, free_mask, fixed_values)

    def model():
        # Sample unconstrained free parameters with broad Normal prior
        z = numpyro.sample("z", Normal(jnp.zeros(n_free), 1.0))

        # Register total log-density as a factor
        numpyro.factor("likelihood_and_jac", -logdensity_fn(z))

    return model


# ----------------- Entrypoint: run NUTS (NumPyro) -----------------
def run_numpyro_nuts(
    param_info,
    free_mask,
    fixed_values,
    num_warmup=1000,
    num_samples=2000,
    num_chains=1,
    rng_seed=0,
    target_accept=0.8,
    hdf5_out="chains.h5",
):
    if not HAS_NUMPYRO:
        raise ImportError(
            "NumPyro is not available. Install numpyro to run this script (preferred)."
        )

    free_indices, index_in_free = make_index_maps(free_mask)
    m = len(free_indices)
    n = len(param_info)

    logger.info(
        f"Model has n={n} parameters, sampling m={m} free parameters, using NumPyro NUTS."
    )
    logger.info("Building potential function (JAX-jitted).")

    # potential_fn = make_potential_fn(param_info, free_mask, fixed_values)

    # initial positions: use midpoint of bounds transformed back to z-space via inverse of our transform:
    def initial_z_free_guess():
        # For each free parameter, pick s0 = 0.5 -> z0 = logit(0.5) = 0.0
        # This maps to midpoint; we can optionally pick random init around 0
        z0 = jnp.zeros((m,), dtype=jnp.float64)
        return np.asarray(z0)

    init_z = initial_z_free_guess()
    init_strategy = init_to_value(values={"z": init_z})
    rng_key = jax.random.PRNGKey(rng_seed)

    model_fn = make_numpyro_model(param_info, free_mask, fixed_values)

    # Build NUTS kernel using potential_fn
    kernel = NUTS(
        model_fn,
        init_strategy=init_strategy,
        target_accept_prob=target_accept,
        adapt_mass_matrix=True,
    )  # argument name depends on numpyro version
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

    # NumPyro's MCMC.run expects model args normally; when using potential_fn directly, there are no args.
    # We pass initial values using init_params kwarg if supported. To keep compatibility across versions,
    # we pass run(...) without args and rely on the kernel to initialize from the provided init strategy.
    # Many versions of NumPyro accept "init_params" or automatically init from a jittered position.
    # For robustness, we set the 'initial_params' attribute via mcmc.run(..., init_params=...) if available.
    try:
        logger.info("Starting MCMC (this may take a while).")
        # Some NumPyro versions accept init_params as keyword argument. Try to pass that.
        mcmc.run(rng_key)
    except TypeError:
        # fallback: run without init_params
        mcmc.run(rng_key)

    # Extract samples: depending on how the kernel stores positions,
    # we either get samples by name or via mcmc.get_samples()
    samples = mcmc.get_samples(group_by_chain=True)  # returns dict
    # NumPyro when using potential_fn usually returns samples keyed by 'positions' or named var; handle heuristics:
    # We'll attempt to find the array with shape (chains, samples, m)
    z_samples = None
    for key, val in samples.items():
        arr = np.asarray(val)
        if arr.ndim == 3 and arr.shape[-1] == m:
            z_samples = arr
            logger.info(f"Using samples from key '{key}' as z_free samples.")
            break
        # fallback: if m==1 maybe shape (chains, samples) -> expand dim
        if m == 1 and arr.ndim == 2:
            z_samples = np.expand_dims(arr, -1)
            logger.info(
                f"Using samples from key '{key}' (2D) as single-dim z_free samples."
            )
            break

    if z_samples is None:
        # If we can't find z samples automatically, raise helpful error
        logger.error(
            "Could not locate z_free samples in MCMC output automatically. Keys: "
            + ", ".join(samples.keys())
        )
        raise RuntimeError(
            "Unable to extract z_free samples from NumPyro MCMC output. Check numpyro version or kernel usage."
        )

    chains, samples_count, _ = z_samples.shape
    logger.info(
        f"Collected z_free samples with shape (chains={chains}, samples={samples_count}, m={m})."
    )

    # Transform z_free samples to full x samples and save to HDF5
    logger.info(
        "Transforming z_free samples to physical parameter space and saving to HDF5."
    )

    # We'll iterate and reconstruct x for each chain/sample (could be vectorized, but keep simple and memory-friendly)
    with h5py.File(hdf5_out, "w") as hf:
        hf.attrs["n_params"] = n
        hf.attrs["param_info_json"] = json.dumps(param_info)
        hf.attrs["free_mask_json"] = json.dumps(free_mask)
        hf.attrs["fixed_values_json"] = json.dumps(
            [None if v is None else float(v) for v in fixed_values]
        )
        # dataset for x samples: shape (chains, samples, n)
        ds = hf.create_dataset(
            "x_samples", shape=(chains, samples_count, n), dtype=np.float64
        )
        for c in range(chains):
            for s in range(samples_count):
                zvec = jnp.asarray(z_samples[c, s, :])
                xvec, _ = zfree_to_x_and_dx_dz(
                    zvec, param_info, free_mask, fixed_values
                )
                ds[c, s, :] = np.asarray(xvec)
        logger.info(f"Saved samples to {hdf5_out} (dataset 'x_samples').")
    logger.info("Done.")
    return hdf5_out


# ----------------- Main: parse CLI args and run -----------------
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
        # param_info = cfg.get("param_info", param_info)
        # free_mask = cfg.get("free_mask", free_mask)
        # fixed_values = cfg.get("fixed_values", fixed_values)
        external_loglik = ll_funcs[cfg.get("ll_func")]
        external_grad = grad_funcs[cfg.get("ll_func")]
        param_info = cfg.get("param_info")
        free_mask = cfg.get("free_mask")
        fixed_values = [None] * len(param_info)
        fixed_values = cfg.get("fixed_vals", fixed_values)
        truths = cfg.get("truths")
        print(free_mask)
        print(fixed_values)
        print(truths)

    logger.info("Parameter info:")
    for i, p in enumerate(param_info):
        logger.info(
            f"  {i}: name={p['name']}, bounds=[{p['lower']}, {p['upper']}], \
                transform={p.get('transform', 'identity')}, prior={p.get('prior', 'uniform')}, \
                free={free_mask[i]}, fixed_value={fixed_values[i]}"
        )

    if not HAS_NUMPYRO:
        logger.error(
            "NumPyro not installed. This script is written to use NumPyro's NUTS (preferred). \
            Install numpyro and try again."
        )
        sys.exit(1)

    start = time.time()
    out = run_numpyro_nuts(
        param_info=param_info,
        free_mask=free_mask,
        fixed_values=fixed_values,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        rng_seed=args.seed,
        target_accept=0.8,
        hdf5_out=args.out,
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
