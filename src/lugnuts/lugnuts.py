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
import json
import logging

import numpy as np
import jax
import jax.numpy as jnp
import h5py
import numpyro
from numpyro.distributions import Normal
from numpyro.infer import MCMC, NUTS, init_to_value


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


# Set likelihood and gradient functions
global external_loglik, external_grad
external_loglik = external_loglik_placeholder
external_grad = external_grad_placeholder


# ----------------- Mapping between z_free and full x -----------------


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


def _make_host_call_loglik(x_full_jax: jnp.ndarray, ll_fn) -> jnp.ndarray:
    """
    Use host_callback.call to invoke the python external_loglik function (which expects python scalars/arrays).
    Returns a jax scalar array (0-d).
    """

    def _py_loglik(x_np):
        # x_np is a numpy array (host side)
        # external_loglik expects separate args, unpack them
        val = ll_fn(*x_np)
        return np.array(val, dtype=np.float64)

    # result_shape must be a jax.ShapeDtypeStruct describing a scalar float64
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    ret = jax.pure_callback(_py_loglik, result_shape, x_full_jax)
    return jnp.asarray(ret).reshape(())
    # return ret


def _make_host_call_grad(x_full_jax: jnp.ndarray, grad_fn) -> jnp.ndarray:
    """
    Use host_callback.call to invoke the python external_grad function.
    Returns a jax array of shape (n,) with dtype float64.
    """
    n = x_full_jax.shape[0]

    def _py_grad(x_np):
        g = grad_fn(*x_np)
        g = np.asarray(g, dtype=np.float64)
        if g.shape != (n,):
            raise ValueError(
                f"external_grad returned shape {g.shape} but expected {(n,)}"
            )
        return g

    result_shape = jax.ShapeDtypeStruct((n,), jnp.float64)
    ret = jax.pure_callback(_py_grad, result_shape, x_full_jax)
    return jnp.asarray(ret)


def make_loglik_of_z(param_info, free_mask, fixed_values, ll_fn, grad_fn):
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
        ll = _make_host_call_loglik(x_full, ll_fn)
        return ll

    def loglik_of_z_fwd(z_free):
        x_full, dx_dz_free = zfree_to_x_and_dx_dz(
            z_free, param_info, free_mask, fixed_values
        )
        ll = _make_host_call_loglik(x_full, ll_fn)
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
        grad_x = _make_host_call_grad(x_full, grad_fn)  # shape (n,)
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
    param_info: List[Dict[str, Any]],
    free_mask: List[bool],
    fixed_values: List[Any],
    ll_fn,
    grad_fn,
):
    """
    Return a JAX-callable potential_fn(z_free) that returns the scalar potential U(z_free) = -log posterior(z_free).
    Uses the loglik_of_z custom-vjp wrapper which will call external_loglik and external_grad as needed.
    """
    loglik_of_z = make_loglik_of_z(param_info, free_mask, fixed_values, ll_fn, grad_fn)

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
        return log_posterior

    return jax.jit(potential)


def make_numpyro_model(param_info, free_mask, fixed_values, ll_fn, grad_fn):
    """
    Returns a NumPyro model function that samples unconstrained free parameters
    and adds the custom log-density (likelihood + Jacobian from transformations).

    Args:
        param_info: list of dicts describing parameters (lower, upper, transform)
        free_mask: list of bools
        fixed_values: list of values for fixed parameters
    """
    n_free = sum(free_mask)
    logdensity_fn = make_potential_fn(
        param_info, free_mask, fixed_values, ll_fn, grad_fn
    )

    def model():
        # Sample unconstrained free parameters with broad Normal prior
        z = numpyro.sample("z", Normal(jnp.zeros(n_free), 10.0))

        # Register total log-density as a factor
        numpyro.factor("likelihood_and_jac", logdensity_fn(z))

    return model


# ----------------- Entrypoint: run NUTS (NumPyro) -----------------
def run_numpyro_nuts(
    param_info,
    free_mask,
    fixed_values,
    ll_fn,
    grad_fn,
    num_warmup=1000,
    num_samples=2000,
    num_chains=1,
    rng_seed=0,
    target_accept=0.8,
    hdf5_out="chains.h5",
    logger=logging.getLogger(__name__),
):

    free_indices, _ = make_index_maps(free_mask)
    m = len(free_indices)
    n = len(param_info)

    logger.info(f"Model has n={n} parameters, sampling m={m} free parameters")

    # initial positions
    def initial_z_free_guess():
        z0 = jnp.zeros((m,), dtype=jnp.float64)
        return np.asarray(z0)

    init_z = initial_z_free_guess()
    init_strategy = init_to_value(values={"z": init_z})
    rng_key = jax.random.PRNGKey(rng_seed)

    model_fn = make_numpyro_model(param_info, free_mask, fixed_values, ll_fn, grad_fn)

    # Build NUTS kernel using potential_fn
    kernel = NUTS(
        model_fn, init_strategy=init_strategy, target_accept_prob=target_accept
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

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
            break
        # fallback: if m==1 maybe shape (chains, samples) -> expand dim
        if m == 1 and arr.ndim == 2:
            z_samples = np.expand_dims(arr, -1)
            break

    if z_samples is None:
        # If we can't find z samples automatically, raise helpful error
        logger.error(
            "Could not locate z_free samples in MCMC output automatically. Keys: "
            + ", ".join(samples.keys())
        )
        raise RuntimeError(
            "Unable to extract z_free samples from MCMC output. Check numpyro version or kernel usage."
        )

    chains, samples_count, _ = z_samples.shape
    logger.info(
        f"Collected z_free samples with shape (chains={chains}, samples={samples_count}, m={m})."
    )

    # Transform z_free samples to full x samples and save to HDF5
    logger.info(
        "Transforming z_free samples to physical parameter space and saving to HDF5."
    )

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
